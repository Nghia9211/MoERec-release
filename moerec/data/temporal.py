"""Global temporal split for one AgentRecBench Amazon dataset (leakage check). Cut-off T = median timestamp of the benchmark test targets.
Every model input (expert training rows, LightGCN graph, co-occurrence / popularity, folds, calibration) uses interactions with time < T only.
Test = benchmark tasks whose target happened at time >= T; their visible history = their interactions before T (target excluded). The
benchmark 50-item test pools are kept. Calibration = 1,000 inference users, last interaction before T masked, target + 49 uniform negatives.
The raw file is a gzipped TSV (user_id, item_id, timestamp_ms, stars), one row per review, extracted from the AgentRecBench review.json.
Also reports the leakage of the user split: the share of training interactions that happen after each test target (median).
Usage: python -m moerec.data.temporal <prepared dataset dir> <raw tsv.gz> <out dir>"""
import sys, os, json, gzip, collections
from pathlib import Path
import numpy as np, scipy.sparse as sp
SRC, RAW, OUT = map(Path, sys.argv[1:4]); D = OUT/'data'; D.mkdir(parents=True, exist_ok=True)
users = json.loads((SRC/'data/users.json').read_text()); u2i = {u: k for k, u in enumerate(users)}
rid = json.loads((SRC/'data/id2rawid.json').read_text()); r2i = {r: k for k, r in enumerate(rid) if k}; N = len(rid)
split = json.loads((SRC/'data/split_users.json').read_text()); tasks = json.loads((SRC/'data/tasks.json').read_text())
H = collections.defaultdict(list); miss = 0
for row, l in enumerate(gzip.open(RAW, 'rt')):
    u, i, t, s = l.rstrip('\n').split('\t')
    if u in u2i and i in r2i: H[u2i[u]].append((int(t), row, r2i[i], float(s or 0)))
    else: miss += 1
for u in H: H[u].sort()
def ts_of(u, item):
    ts = [t for t, _, i, _ in H[u] if i == item]; return max(ts) if ts else None
tt = [ts_of(r['user_index'], r['target']) for r in tasks]; T = int(np.median([x for x in tt if x is not None]))
keep_tasks = []
for r, t in zip(tasks, tt):
    if t is None or t < T: continue
    vis = [x for x in H[r['user_index']] if x[0] < T and x[2] != r['target']]
    if len(vis) == 0: continue
    keep_tasks.append(dict(r, visible_history_length=len(vis)))
test_u = {r['user_index']: r['target'] for r in keep_tasks}
all_test_u = {r['user_index'] for r in tasks}
infer = [u2i[u] for u in split['inference'] if u in u2i]; val = set(split.get('sasrec_validation', []))
hist, held, cal_cand = [], [], {}
for ui in range(len(users)):
    h = [x for x in H.get(ui, []) if x[0] < T]
    if ui in test_u: h = [x for x in h if x[2] != test_u[ui]]; held.append([ui, test_u[ui]])
    elif ui in all_test_u: pass   # test users whose target is before T: plain training users of the past
    elif ui in set(infer) and len(h) >= 2: held.append([ui, h[-1][2]]); cal_cand[ui] = h[-1][2]; h = h[:-1]
    hist.append(h)
flat = [x[2] for h in hist for x in h]; stars = np.array([x[3] for h in hist for x in h], np.float32)
off = np.r_[0, np.cumsum([len(h) for h in hist])].astype(np.int64)
np.savez_compressed(D/'histories.npz', items=np.asarray(flat, np.int32), offsets=off, heldout_pairs=np.asarray(held, np.int32).reshape(-1, 2))
np.save(D/'history_stars.npy', stars)
masked = set(test_u) | set(cal_cand); train_uid, train_pos, val_uid, val_pos, gate_rows = [], [], [], [], []
for ui, u in enumerate(users):
    ids = [x[2] for x in hist[ui]]; n = len(ids); vi = ids[-1] if u in val and n >= 2 and ui not in masked else None
    if vi is not None: val_uid.append(ui); val_pos.append(n-1)
    stride = 1 if n < 15 else 2 if n < 50 else 3
    for t in range(1, n, stride):
        if vi is not None and ids[t] == vi: continue
        idx = len(train_uid); train_uid.append(ui); train_pos.append(t)
        if ui not in masked: gate_rows.append(idx)
np.savez_compressed(D/'samples.npz', train_uid=np.asarray(train_uid, np.int32), train_pos=np.asarray(train_pos, np.int32), val_uid=np.asarray(val_uid, np.int32),
                    val_pos=np.asarray(val_pos, np.int32), gate_rows=np.asarray(gate_rows, np.int64))
nu = len(users); ui_ = np.repeat(np.arange(nu), np.diff(off)); ii_ = np.asarray(flat, np.int64)-1
csr = sp.coo_matrix((np.ones(len(flat), np.float32), (ui_, ii_)), shape=(nu, N-1)).tocsr(); csr.data[:] = 1; csr.eliminate_zeros()
for u, i in held: assert csr[u, i-1] == 0 or i in set(hist[u][k][2] for k in range(len(hist[u])))
sp.save_npz(D/'train_user_item.npz', csr)
row, col = csr.nonzero(); rows = np.r_[row, nu+col]; cols = np.r_[nu+col, row]; nodes = nu+N-1
A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nodes, nodes)).tocsr() + sp.eye(nodes, dtype=np.float32, format='csr')
dg = np.asarray(A.sum(1)).ravel(); Dg = sp.diags(1/np.sqrt(dg)); nm = (Dg @ A @ Dg).tocoo()
np.savez_compressed(D/'graph.npz', indices=np.vstack([nm.row, nm.col]).astype(np.int64), values=nm.data.astype(np.float32), shape=np.asarray(nm.shape), n_users=nu, item_num=N)
(D/'tasks.json').write_text(json.dumps(keep_tasks)); (D/'split_users.json').write_text(json.dumps(split))
rng = np.random.default_rng(303); cu = sorted(np.asarray(list(cal_cand))[rng.permutation(len(cal_cand))[:1000]].tolist()); cal = []
for u in cu:
    tgt = cal_cand[u]; vis = set(x[2] for x in hist[u]); negs = []
    while len(negs) < 49:
        j = int(rng.integers(1, N))
        if j not in vis and j != tgt and j not in negs: negs.append(j)
    pool = np.array([tgt]+negs)[rng.permutation(50)].tolist(); n = len(hist[u])
    cal.append({'user_index': u, 'target': tgt, 'candidates': pool, 'visible_history_length': n, 'scenario': 'user_cold_start' if n <= 6 else 'classic'})
(OUT/'calib.json').write_text(json.dumps(cal))
for f in ('users.json', 'id2rawid.json', 'item_names.json', 'item_texts.json'):
    if not (D/f).exists(): os.symlink((SRC/'data'/f).resolve(), D/f)
if not (OUT/'document_vectors.npy').exists(): os.symlink((SRC/'document_vectors.npy').resolve(), OUT/'document_vectors.npy')
new_items = set(range(1, N)) - set(flat)
info = {'cutoff_ms': T, 'raw_rows_unmatched': miss, 'tasks_total': len(tasks), 'tasks_kept': len(keep_tasks),
        'tasks_target_before_T': sum(1 for t in tt if t is not None and t < T), 'tasks_no_ts': sum(t is None for t in tt),
        'kept_scen': {s: sum(r['scenario'] == s for r in keep_tasks) for s in ('classic', 'user_cold_start')},
        'kept_median_hist': float(np.median([r['visible_history_length'] for r in keep_tasks])),
        'targets_unseen_before_T': sum(r['target'] in new_items for r in keep_tasks), 'interactions_before_T': len(flat),
        'calib_users': len(cal), 'train_rows': len(train_uid)}
# leakage of the ORIGINAL user split: share of other users' training interactions that happen after each test target
allt = np.sort(np.array([x[0] for h in H.values() for x in h])); info['orig_share_train_after_target_median'] = float(np.median([1 - np.searchsorted(allt, t)/len(allt) for t in tt if t is not None]))
(OUT/'temporal-info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info))
