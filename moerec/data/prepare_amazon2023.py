"""Data preparation for the two new datasets (Cell Phones & Accessories, Movies & TV), Amazon Reviews 2023.

1. Subset, following the reverse-engineered AgentRecBench recipe (audit_arb_filter.py on Industrial: ARB kept a random
   57% of the official 5-core items, kept items had the same popularity distribution as dropped ones, and every 5-core
   interaction of kept users on kept items is in ARB): random item subsample of the official 5-core (ITEM_FRAC), plus a
   random user subsample (USER_FRAC) so the scale stays close to the ARB datasets. Seed 303.
2. Benchmark tasks generated with the ARB protocol measured by audit_negatives.py on all 5 ARB datasets:
   cold-start task users have 4-6 interactions, classic task users >= 7 (ARB: visible 3-5 vs >= 6 after masking);
   ~2.85% / ~2.15% of eligible users; ground truth = the user's chronologically last interaction;
   19 negatives uniform over the catalog excluding the user's own items; ground truth at a uniform position.
3. Everything else is prepare_multi.py unchanged: remove users with <= 3 interactions; seed-303 ~5% inference users per
   frequency bin (last interaction masked); SASRec prefix samples with upstream stride; graph without held-out pairs;
   paper scenario = visible history <= 6 cold, >= 7 classic; item texts from the author's build_item_text.
Also writes data/history_stars.npy (the user's rating per visible history position, as export_stars.py).
Usage: python -m moerec.data.prepare_amazon2023 <dataset> <work dir>   (env RAW_DIR = Amazon Reviews 2023 files)
"""
import os, sys, csv, json, time, collections, hashlib
import numpy as np, scipy.sparse as sp
from pathlib import Path

DS, ROOT = sys.argv[1], Path(sys.argv[2])
RAW = Path(os.environ['RAW_DIR'])
CFG = {'cell_phones': dict(cat='Cell_Phones_and_Accessories', item_frac=.7, user_frac=.25),
       # Movies & TV: 41% of the items have an empty title (mostly Prime Video entries with empty metadata), so only
       # titled items are kept (58% of the 5-core items, close to the 57% random share ARB kept for Industrial).
       'movies_tv': dict(cat='Movies_and_TV', item_frac=1.0, user_frac=.25, titled_only=True)}[DS]
EXP = ROOT / DS; OUT = EXP / 'data'; OUT.mkdir(parents=True, exist_ok=True)
from moerec.data.item_text import amazon_text as build_item_text
COLD_SHARE, CLASSIC_SHARE = .0285, .0215


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


start = time.monotonic()
rating_csv = RAW / f'benchmark__5core__rating_only__{CFG["cat"]}.csv'
rows = []
with open(rating_csv) as f:
    r = csv.reader(f); head = next(r); iu, ii, ir, it = (head.index(k) for k in ('user_id', 'parent_asin', 'rating', 'timestamp'))
    for x in r: rows.append((x[iu], x[ii].strip(), float(x[ir]), int(x[it])))
core = (len({x[0] for x in rows}), len({x[1] for x in rows}), len(rows))
rng = np.random.default_rng(303)
I0 = sorted({x[1] for x in rows}); U0 = sorted({x[0] for x in rows})
if CFG.get('titled_only'):
    titled = set()
    with open(RAW / f'raw__meta_categories__meta_{CFG["cat"]}.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if (d.get('title') or '').strip(): titled.add(str(d.get('parent_asin')).strip())
    rows = [x for x in rows if x[1] in titled]
ikeep = dict(zip(I0, rng.random(len(I0)) < CFG['item_frac'])); ukeep = dict(zip(U0, rng.random(len(U0)) < CFG['user_frac']))
raw_hist = collections.defaultdict(list); raw_items = set(); stars_of = {}
for rowno, (u, i, s, t) in enumerate(x for x in rows if ikeep[x[1]] and ukeep[x[0]]):
    raw_hist[u].append((t, rowno, i)); raw_items.add(i); stars_of[(u, rowno)] = s
n_raw = sum(map(len, raw_hist.values())); raw_stats = (len(raw_hist), len(raw_items), n_raw)
freq = {u: len(h) for u, h in raw_hist.items()}
uids = sorted(u for u, n in freq.items() if n > 3); u2i = {u: k for k, u in enumerate(uids)}

# Generated benchmark tasks (ARB protocol)
items_sorted = sorted(raw_items); raw2inner = {r: k + 1 for k, r in enumerate(items_sorted)}; N = len(items_sorted) + 1
cold_pool = [u for u in uids if 4 <= freq[u] <= 6]; classic_pool = [u for u in uids if freq[u] >= 7]
n_cold, n_classic = round(len(uids) * COLD_SHARE), round(len(uids) * CLASSIC_SHARE)
trng = np.random.default_rng(303 + 7)
picked = [('user_cold_start', u) for u in np.asarray(cold_pool)[trng.permutation(len(cold_pool))[:n_cold]]] + \
         [('classic', u) for u in np.asarray(classic_pool)[trng.permutation(len(classic_pool))[:n_classic]]]
task_rows = []
for scen, u in picked:
    h = sorted(raw_hist[u]); gt = h[-1][2]; own = {x[2] for x in h}
    negs = []
    while len(negs) < 19:
        c = items_sorted[int(trng.integers(len(items_sorted)))]
        if c not in own and c not in negs: negs.append(c)
    cands = negs[:]; cands.insert(int(trng.integers(20)), gt)
    assert len(set(cands)) == 20 and gt in cands
    task_rows.append((scen, u, gt, cands))
with open(OUT / 'generated_tasks.jsonl', 'w') as f:
    for k, (scen, u, gt, cands) in enumerate(task_rows):
        f.write(json.dumps({'task': k, 'scenario': scen, 'user_id': u, 'ground_truth': gt, 'candidate_list': cands}) + '\n')

# Item metadata and author-style rich texts
meta = {}
with open(RAW / f'raw__meta_categories__meta_{CFG["cat"]}.jsonl') as f:
    for line in f:
        d = json.loads(line); iid = str(d.get('parent_asin')).strip()
        if iid in raw2inner and iid not in meta: meta[iid] = d
names = [''] * N; texts = [''] * N; no_meta = 0
for r, k in raw2inner.items():
    d = meta.get(r)
    if d is None: no_meta += 1; names[k] = r; texts[k] = r; continue
    names[k] = (d.get('title') or r).strip()
    texts[k] = (build_item_text(d) or names[k]).replace('\n', ' ')

# Test masks from tasks, inference masks from frequency bins (prepare_multi.py)
test = collections.defaultdict(set)
for scen, u, gt, cands in task_rows: test[u].add(raw2inner[gt])
bins = collections.defaultdict(list)
for u in uids:
    if u in test: continue
    n = freq[u]; bins[0 if n <= 6 else 1 if n <= 10 else 2 if n <= 20 else 3 if n <= 50 else 4].append(u)
rng = np.random.default_rng(303); target_n = int(len(uids) * .05)
quotas = {k: len(v) * target_n / sum(map(len, bins.values())) for k, v in bins.items()}; counts = {k: int(v) for k, v in quotas.items()}
for k in sorted(quotas, key=lambda x: quotas[x] - counts[x], reverse=True)[:target_n - sum(counts.values())]: counts[k] += 1
infer = set()
for k, us in sorted(bins.items()): infer.update(np.asarray(us)[rng.permutation(len(us))[:counts[k]]].tolist())
assert not infer & set(test) and len(infer) == target_n

histories = []
for u in uids:
    h = sorted((t, r, raw2inner[i]) for t, r, i in raw_hist[u]); histories.append(h)
masks = {u: set(s) for u, s in test.items()}
for u in infer: masks[u] = {histories[u2i[u]][-1][2]}
removed_occ = 0; held_pairs = []; gt_absent = 0
for u, s in masks.items():
    old = histories[u2i[u]]; clean = [x for x in old if x[2] not in s]; removed_occ += len(old) - len(clean)
    if u in test and len(old) == len(clean): gt_absent += 1
    histories[u2i[u]] = clean
    for iid in s: held_pairs.append([u2i[u], iid])
offsets = [0]; flat = []; flat_stars = []
for u, h in zip(uids, histories):
    flat.extend(x[2] for x in h); flat_stars.extend(stars_of[(u, x[1])] for x in h); offsets.append(len(flat))
np.savez_compressed(OUT / 'histories.npz', items=np.asarray(flat, np.int32), offsets=np.asarray(offsets, np.int64), heldout_pairs=np.asarray(held_pairs, np.int32))
np.save(OUT / 'history_stars.npy', np.asarray(flat_stars, np.float32))

train_users = [u for u in uids if u not in masks]; order = rng.permutation(len(train_users))
val_users = set(np.asarray(train_users)[order[:int(len(train_users) * .1)]].tolist())
train_uid, train_pos, val_uid, val_pos, gate_rows = [], [], [], [], []
for ui, u in enumerate(uids):
    ids = [x[2] for x in histories[ui]]; n = len(ids)
    val_item = ids[-1] if u in val_users and n >= 2 else None
    if val_item is not None: val_uid.append(ui); val_pos.append(n - 1)
    stride = 1 if n < 15 else 2 if n < 50 else 3
    for t in range(1, n, stride):
        if val_item is not None and ids[t] == val_item: continue
        idx = len(train_uid); train_uid.append(ui); train_pos.append(t)
        if u not in masks: gate_rows.append(idx)
np.savez_compressed(OUT / 'samples.npz', train_uid=np.asarray(train_uid, np.int32), train_pos=np.asarray(train_pos, np.int32),
                    val_uid=np.asarray(val_uid, np.int32), val_pos=np.asarray(val_pos, np.int32), gate_rows=np.asarray(gate_rows, np.int64))

n_users = len(uids); n_items = N - 1
u_index = np.repeat(np.arange(n_users), np.diff(offsets)); i_index = np.asarray(flat, np.int64) - 1
csr = sp.coo_matrix((np.ones(len(flat), np.float32), (u_index, i_index)), shape=(n_users, n_items)).tocsr(); csr.data[:] = 1; csr.eliminate_zeros()
for ui, iid in held_pairs: assert csr[ui, iid - 1] == 0
sp.save_npz(OUT / 'train_user_item.npz', csr)
row, col = csr.nonzero(); rows_ = np.r_[row, n_users + col]; cols_ = np.r_[n_users + col, row]; nodes = n_users + n_items
A = sp.coo_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)), shape=(nodes, nodes)).tocsr() + sp.eye(nodes, dtype=np.float32, format='csr')
deg = np.asarray(A.sum(1)).ravel(); D = sp.diags(1 / np.sqrt(deg)); norm = (D @ A @ D).tocoo()
np.savez_compressed(OUT / 'graph.npz', indices=np.vstack([norm.row, norm.col]).astype(np.int64), values=norm.data.astype(np.float32),
                    shape=np.asarray(norm.shape), n_users=n_users, item_num=N)

tasks = []
for k, (scen, u, gt, cands) in enumerate(task_rows):
    ui = u2i[u]; n = len(histories[ui])
    tasks.append({'user_index': ui, 'user_id': u, 'target': raw2inner[gt], 'target_raw_id': gt, 'candidates': [raw2inner[c] for c in cands],
                  'original_scenario': scen, 'scenario': 'user_cold_start' if n <= 6 else 'classic', 'visible_history_length': n,
                  'task_file': f'generated_{k}', 'task_sha256': None, 'label_sha256': None})
(OUT / 'tasks.json').write_text(json.dumps(tasks)); (OUT / 'removed_tasks.json').write_text('[]')
(OUT / 'users.json').write_text(json.dumps(uids)); (OUT / 'item_names.json').write_text(json.dumps(names, ensure_ascii=False))
(OUT / 'item_texts.json').write_text(json.dumps(texts, ensure_ascii=False)); (OUT / 'id2rawid.json').write_text(json.dumps([''] + items_sorted))
(OUT / 'split_users.json').write_text(json.dumps({'test': sorted(test), 'inference': sorted(infer), 'sasrec_validation': sorted(val_users)}))
info = {'status': 'PASS', 'dataset': DS, 'source': str(rating_csv), 'official_5core_users_items_rows': core,
        'item_frac': CFG['item_frac'], 'user_frac': CFG['user_frac'], 'raw_users_items_interactions': raw_stats,
        'eligible_users': n_users, 'item_num_including_padding': N, 'items_without_metadata': no_meta,
        'visible_interactions_after_mask': len(flat), 'masked_occurrences': removed_occ, 'heldout_pairs': len(held_pairs),
        'test_users': len(test), 'test_fraction': len(test) / n_users, 'test_users_gt_absent_from_raw_history': gt_absent,
        'inference_users': len(infer), 'inference_fraction': len(infer) / n_users, 'inference_bin_quotas': counts,
        'sasrec_training_rows': len(train_uid), 'sasrec_validation_rows': len(val_uid), 'gate_training_rows': len(gate_rows),
        'tasks_evaluated': len(tasks), 'paper_scenario_counts': dict(collections.Counter(t['scenario'] for t in tasks)),
        'original_scenario_counts': dict(collections.Counter(t['original_scenario'] for t in tasks)), 'graph_edges': int(csr.nnz), 'graph_nodes': nodes,
        'stars_distribution': dict(collections.Counter(map(float, flat_stars))), 'wall_seconds': time.monotonic() - start}
info['data_hashes'] = {f.name: sha(f) for f in OUT.iterdir() if f.is_file()}
(EXP / 'data-preparation.json').write_text(json.dumps(info, indent=2)); print(json.dumps({k: v for k, v in info.items() if k != 'data_hashes'}), flush=True)
