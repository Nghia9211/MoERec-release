"""Paper Sec. 4.1.2 data preparation for the AgentRecBench datasets Yelp / Video Games / Musical / Industrial.

Same rules as the frozen Goodreads run (prepare_data.py):
remove users with <=3 interactions; supplied AgentRecBench task users form the test set
(their ground-truth item masked); disjoint seed-303 ~5% inference users per frequency bin
(last interaction masked); SASRec prefix samples with upstream stride; graph without any
held-out pair; paper scenario = visible history <=6 cold, >=7 classic.
Item texts use the upstream build_item_text (moerec/data/item_text.py).
Usage: python -m moerec.data.prepare_arb <dataset> --raw <AgentRecBench Dataset dir> --out <work dir>
(writes <work dir>/<dataset>/data/; then run moerec.data.stars for the per-position ratings)
"""
import os, json, time, collections, datetime, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, scipy.sparse as sp
from pathlib import Path

import argparse
ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True); A = ap.parse_args()
DS = A.dataset
R = Path(A.raw)
CFG = {
    'yelp': dict(review=R / 'merged_all/review.json', item=R / 'merged_all/item.json', src='yelp', tasks=R / 'merged_all/tasks', taskset='yelp', kind='yelp',
                 paper=(133977, 32621, 1315233)),
    'video_games': dict(review=R / 'amazon_video_games/review.json', item=R / 'amazon_video_games/item.json', src='amazon', tasks=R / 'amazon_video_games/tasks',
                        taskset='amazon_video_games', kind='amazon', paper=(84797, 17931, 647218)),
    'musical': dict(review=R / 'amazon_musical_raw/review.json', item=R / 'amazon_musical_raw/item.json', src=None, tasks=R / 'amazon_musical_raw/tasks',
                    taskset='amazon_musical_raw', kind='amazon', paper=(60419, 21771, 482046)),
    'industrial': dict(review=R / 'amazon_industrial/review.json', item=R / 'amazon_industrial/item.json', src=None, tasks=R / 'amazon_industrial/tasks',
                       taskset='amazon_industrial', kind='amazon', paper=(39382, 15251, 225237)),
}[DS]
EXP = Path(A.out) / DS
OUT = EXP / 'data'; OUT.mkdir(parents=True, exist_ok=True)
from moerec.data.item_text import amazon_text, yelp_text
build_item_text = amazon_text if CFG['kind'] == 'amazon' else yelp_text


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def ts_of(d):
    if d.get('timestamp') not in (None, ''):
        return int(d['timestamp'])
    for k in ('date', 'date_added', 'date_updated'):
        v = d.get(k)
        if v:
            for fmt in ('%Y-%m-%d %H:%M:%S', '%a %b %d %H:%M:%S %z %Y', '%Y-%m-%d'):
                try: return int(datetime.datetime.strptime(v, fmt).timestamp() * 1000)
                except ValueError: pass
    return None


start = time.monotonic()
raw_hist = collections.defaultdict(list); raw_items = set(); n_raw = 0; bad_ts = 0
for rowno, line in enumerate(open(CFG['review'])):
    d = json.loads(line)
    if CFG['src'] and d.get('source') != CFG['src']: continue
    t = ts_of(d)
    if t is None: bad_ts += 1; t = 0
    u = str(d['user_id']); i = str(d['item_id']).strip()
    raw_hist[u].append((t, rowno, i)); raw_items.add(i); n_raw += 1
raw_stats = (len(raw_hist), len(raw_items), n_raw)
freq = {u: len(h) for u, h in raw_hist.items()}
uids = sorted(u for u, n in freq.items() if n > 3); u2i = {u: k for k, u in enumerate(uids)}

# Supplied benchmark tasks
task_rows = []; removed = []
for scen in ['classic', 'user_cold_start']:
    base = CFG['tasks'] / scen / CFG['taskset']
    for f in sorted((base / 'tasks').glob('task_*.json'), key=lambda p: int(p.stem[5:])):
        d = json.loads(f.read_text()); d = d[0] if isinstance(d, list) else d; uid = str(d['user_id'])
        label = base / 'groundtruth' / f.name.replace('task_', 'groundtruth_'); g = json.loads(label.read_text())
        gt = str(g['ground truth'] if 'ground truth' in g else g['ground_truth']).strip()
        cands = [str(x).strip() for x in d['candidate_list']]
        assert len(cands) == len(set(cands)) == 20 and gt in cands, f.name
        if uid not in u2i: removed.append({'scenario': scen, 'file': f.name, 'uid': uid, 'raw_interactions': freq.get(uid, 0)}); continue
        task_rows.append((scen, f, label, uid, gt, cands))

items_sorted = sorted(raw_items | {c for t in task_rows for c in t[5]})
raw2inner = {r: k + 1 for k, r in enumerate(items_sorted)}; N = len(items_sorted) + 1

# Item metadata and author-style rich texts
meta = {}
for line in open(CFG['item']):
    d = json.loads(line)
    if CFG['src'] and d.get('source') != CFG['src']: continue
    iid = str(d.get('item_id')).strip()
    if iid in raw2inner and iid not in meta: meta[iid] = d
name_key = 'title' if CFG['kind'] == 'amazon' else 'name'
names = [''] * N; texts = [''] * N; no_meta = 0
for r, k in raw2inner.items():
    d = meta.get(r)
    if d is None: no_meta += 1; names[k] = r; texts[k] = r; continue
    names[k] = (d.get(name_key) or r).strip()
    texts[k] = (build_item_text(d) or names[k]).replace('\n', ' ')

# Test masks from tasks, inference masks from frequency bins
test = collections.defaultdict(set)
for scen, f, label, uid, gt, cands in task_rows: test[uid].add(raw2inner[gt])
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
offsets = [0]; flat = []
for h in histories: flat.extend(x[2] for x in h); offsets.append(len(flat))
np.savez_compressed(OUT / 'histories.npz', items=np.asarray(flat, np.int32), offsets=np.asarray(offsets, np.int64), heldout_pairs=np.asarray(held_pairs, np.int32))

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
row, col = csr.nonzero(); rows = np.r_[row, n_users + col]; cols = np.r_[n_users + col, row]; nodes = n_users + n_items
A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nodes, nodes)).tocsr() + sp.eye(nodes, dtype=np.float32, format='csr')
deg = np.asarray(A.sum(1)).ravel(); D = sp.diags(1 / np.sqrt(deg)); norm = (D @ A @ D).tocoo()
np.savez_compressed(OUT / 'graph.npz', indices=np.vstack([norm.row, norm.col]).astype(np.int64), values=norm.data.astype(np.float32),
                    shape=np.asarray(norm.shape), n_users=n_users, item_num=N)

tasks = []
for scen, f, label, uid, gt, cands in task_rows:
    ui = u2i[uid]; n = len(histories[ui])
    tasks.append({'user_index': ui, 'user_id': uid, 'target': raw2inner[gt], 'target_raw_id': gt, 'candidates': [raw2inner[c] for c in cands],
                  'original_scenario': scen, 'scenario': 'user_cold_start' if n <= 6 else 'classic', 'visible_history_length': n,
                  'task_file': f.name, 'task_sha256': sha(f), 'label_sha256': sha(label)})
(OUT / 'tasks.json').write_text(json.dumps(tasks)); (OUT / 'removed_tasks.json').write_text(json.dumps(removed, indent=1))
(OUT / 'users.json').write_text(json.dumps(uids)); (OUT / 'item_names.json').write_text(json.dumps(names, ensure_ascii=False))
(OUT / 'item_texts.json').write_text(json.dumps(texts, ensure_ascii=False)); (OUT / 'id2rawid.json').write_text(json.dumps([''] + items_sorted))
(OUT / 'split_users.json').write_text(json.dumps({'test': sorted(test), 'inference': sorted(infer), 'sasrec_validation': sorted(val_users)}))
info = {'status': 'PASS', 'dataset': DS, 'raw_users_items_interactions': raw_stats, 'paper_table1': CFG['paper'], 'raw_matches_table1': list(raw_stats) == list(CFG['paper']),
        'eligible_users': n_users, 'item_num_including_padding': N, 'items_without_metadata': no_meta, 'bad_timestamps': bad_ts,
        'visible_interactions_after_mask': len(flat), 'masked_occurrences': removed_occ, 'heldout_pairs': len(held_pairs),
        'test_users': len(test), 'test_fraction': len(test) / n_users, 'test_users_gt_absent_from_raw_history': gt_absent,
        'inference_users': len(infer), 'inference_fraction': len(infer) / n_users, 'inference_bin_quotas': counts,
        'sasrec_training_rows': len(train_uid), 'sasrec_validation_rows': len(val_uid), 'gate_training_rows': len(gate_rows),
        'tasks_evaluated': len(tasks), 'tasks_removed_le3': len(removed), 'paper_scenario_counts': dict(collections.Counter(t['scenario'] for t in tasks)),
        'original_scenario_counts': dict(collections.Counter(t['original_scenario'] for t in tasks)), 'graph_edges': int(csr.nnz), 'graph_nodes': nodes,
        'wall_seconds': time.monotonic() - start}
info['data_hashes'] = {f.name: sha(f) for f in OUT.iterdir() if f.is_file()}
(EXP / 'data-preparation.json').write_text(json.dumps(info, indent=2)); print(json.dumps({k: v for k, v in info.items() if k != 'data_hashes'}), flush=True)
