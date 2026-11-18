"""Held-out folds (out-of-sample training users for the Stage 1 gate and the Stage 2 judge).
Pick 20% of the training users (not test, not inference, not SASRec validation; at least 2 visible items) and mask their LAST
item exactly like the inference users: it is removed from the history (hence from the SASRec training rows and from the LightGCN
graph) and kept aside as their target. Test users, inference / calibration users and pools are unchanged.
  fold 1 (gate):  --rng 707
  fold 2 (judge): --rng 808 --exclude <fold-1 dir>/data/gate_holdout.json   (a disjoint group of the same size)
Writes <out>/data/: histories.npz, history_stars.npy, samples.npz, train_user_item.npz, graph.npz (same recipe as prepare_arb.py),
gate_holdout.json ([[user_index, item], ...]); every other file is linked from the source dataset dir. The experts are then
retrained on <out> (moerec.experts.train).
Usage: python -m moerec.data.folds <dataset dir> <out dir> [--rng 707] [--exclude gate_holdout.json]"""
import os, json
from pathlib import Path
import numpy as np, scipy.sparse as sp

import argparse
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--rng', type=int, default=707); ap.add_argument('--exclude')
A = ap.parse_args()
SRC, OUT = Path(A.src), Path(A.out); D = OUT / 'data'; D.mkdir(parents=True, exist_ok=True)
with np.load(SRC / 'data/histories.npz') as f: items, offsets, held = f['items'], f['offsets'], f['heldout_pairs']
stars = np.load(SRC / 'data/history_stars.npy'); assert len(stars) == len(items)
users = json.loads((SRC / 'data/users.json').read_text()); split = json.loads((SRC / 'data/split_users.json').read_text())
u2i = {u: k for k, u in enumerate(users)}; n_users = len(users)
masked = {int(u) for u, _ in held}; val = {u2i[u] for u in split.get('sasrec_validation', []) if u in u2i}
lens = np.diff(offsets); cand = [u for u in range(n_users) if u not in masked and u not in val and lens[u] >= 2]
n_g = int(len(cand) * .2)
if A.exclude:   # fold 2: a second, disjoint group (same size as fold 1), never one of fold 1's users
    ex = {int(u) for u, _ in json.loads(Path(A.exclude).read_text())}; cand = [u for u in cand if u not in ex]
rng = np.random.default_rng(A.rng); G = np.sort(rng.choice(cand, size=n_g, replace=False)); Gs = set(G.tolist())
keep = np.ones(len(items), bool); gate_holdout = []
for u in G: keep[offsets[u + 1] - 1] = False; gate_holdout.append([int(u), int(items[offsets[u + 1] - 1])])
items2, stars2 = items[keep], stars[keep]; lens2 = lens - np.isin(np.arange(n_users), G); offsets2 = np.r_[0, np.cumsum(lens2)].astype(np.int64)
np.savez_compressed(D / 'histories.npz', items=items2.astype(np.int32), offsets=offsets2, heldout_pairs=held)
np.save(D / 'history_stars.npy', stars2)
# SASRec rows: same recipe as prepare_multi.py (stride by length, validation users keep their last item for validation)
val_users_raw = set(split.get('sasrec_validation', [])); train_uid, train_pos, val_uid, val_pos, gate_rows = [], [], [], [], []
for ui in range(n_users):
    ids = items2[offsets2[ui]:offsets2[ui + 1]]; n = len(ids)
    val_item = ids[-1] if users[ui] in val_users_raw and n >= 2 else None
    if val_item is not None: val_uid.append(ui); val_pos.append(n - 1)
    stride = 1 if n < 15 else 2 if n < 50 else 3
    for t in range(1, n, stride):
        if val_item is not None and ids[t] == val_item: continue
        idx = len(train_uid); train_uid.append(ui); train_pos.append(t)
        if ui not in masked and ui not in Gs: gate_rows.append(idx)
np.savez_compressed(D / 'samples.npz', train_uid=np.asarray(train_uid, np.int32), train_pos=np.asarray(train_pos, np.int32),
                    val_uid=np.asarray(val_uid, np.int32), val_pos=np.asarray(val_pos, np.int32), gate_rows=np.asarray(gate_rows, np.int64))
names = json.loads((SRC / 'data/item_names.json').read_text()); N = len(names); n_items = N - 1
u_index = np.repeat(np.arange(n_users), lens2); i_index = items2.astype(np.int64) - 1
csr = sp.coo_matrix((np.ones(len(items2), np.float32), (u_index, i_index)), shape=(n_users, n_items)).tocsr(); csr.data[:] = 1; csr.eliminate_zeros()
for u, it in gate_holdout[:2000]: assert csr[u, it - 1] == 0 or it in set(items2[offsets2[u]:offsets2[u + 1]].tolist())
sp.save_npz(D / 'train_user_item.npz', csr)
row, col = csr.nonzero(); rows = np.r_[row, n_users + col]; cols = np.r_[n_users + col, row]; nodes = n_users + n_items
A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nodes, nodes)).tocsr() + sp.eye(nodes, dtype=np.float32, format='csr')
deg = np.asarray(A.sum(1)).ravel(); Dg = sp.diags(1 / np.sqrt(deg)); norm = (Dg @ A @ Dg).tocoo()
np.savez_compressed(D / 'graph.npz', indices=np.vstack([norm.row, norm.col]).astype(np.int64), values=norm.data.astype(np.float32),
                    shape=np.asarray(norm.shape), n_users=n_users, item_num=N)
(D / 'gate_holdout.json').write_text(json.dumps(gate_holdout))
for f in (SRC / 'data').iterdir():   # link the unchanged files (users, items, texts, split, tasks, ...)
    if not (D / f.name).exists(): os.symlink(f.resolve(), D / f.name)
for name in ('calib.json', 'pools50.json', 'document_vectors.npy'):
    if (SRC / name).exists() and not (OUT / name).exists(): os.symlink((SRC / name).resolve(), OUT / name)
print(json.dumps({'out': str(OUT), 'eligible_train_users': len(cand), 'gate_holdout_users': len(G), 'interactions_before': int(len(items)), 'after': int(len(items2)),
                  'sasrec_rows': len(train_uid), 'gate_rows_in_sample': len(gate_rows)}), flush=True)
