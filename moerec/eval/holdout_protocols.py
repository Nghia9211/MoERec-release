"""Gate-learning protocols without per-fold expert retraining, compared with J1 (FINDINGS.md section 2).

  val_main        J1 recipe on the last item of the SASRec validation users, scored by the MAIN experts
                  (SASRec never trained on it; LightGCN and the co-occurrence / popularity graph did)
  val_cleangraph  the same rows, with those user-item edges removed from the co-occurrence / popularity graph
  fold1_sub       J1 recipe on a random fold-1 subset with as many rows (size control)
  single_experts  the locked J1 gate applied to test pools scored by the FOLD-1 experts (one expert set, LIME-Rec style)
Reference: the locked J1 (stage1-test.npz). Paired bootstrap of J1 minus each variant. CPU.
Usage: python -m moerec.eval.holdout_protocols <dataset dir> <fold-1 dir> <run dir>  -> <run dir>/holdout_protocols.json"""
import sys, json, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import build, graph_side
from moerec.stage1.j1 import fit, J1
from moerec.experts.score import Experts
E, H1, run = map(Path, sys.argv[1:4]); t0 = time.time()
cat = Catalog(E); sE, sH = graph_side(E), graph_side(H1); f1 = np.load(run/'fold1.npz'); P = f1['pools'].shape[1]
EV = {t: np.load(run/f'experts-{t}.npz') for t in ('calib', 'test')}
ev = {t: build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats) for t, z in EV.items()}
S1 = np.load(run/'stage1-test.npz'); tp = EV['test']['tpos']; ref = rank_of(S1['s0'], S1['union'], tp)
X = Experts(E); items, off = X.hist['items'], X.hist['offsets']
with np.load(E/'data/samples.npz') as f: vu, vp = f['val_uid'], f['val_pos']
rng = np.random.default_rng(505); us, ns, pools, tps = [], [], [], []
for u, q in zip(vu, vp):
    vis = set(items[off[u]:off[u]+q].tolist()); tgt = int(items[off[u]+q])
    if tgt in vis: continue
    negs = []
    while len(negs) < P-1:
        j = int(rng.integers(1, X.N))
        if j not in vis and j != tgt and j not in negs: negs.append(j)
    pool = np.array([tgt]+negs); perm = rng.permutation(P)
    us.append(int(u)); ns.append(int(q)); pools.append(pool[perm]); tps.append(int(np.flatnonzero(perm == 0)[0]))
us, ns, pools, tps = np.array(us), np.array(ns), np.stack(pools), np.array(tps)
normed = X.normed(us, ns, pools); print('val rows', len(us), 'fold1 rows', len(f1['tpos']), 'sec', round(time.time()-t0), flush=True)
A = sp.load_npz(E/'data/train_user_item.npz').tolil()
for u, q in zip(us, ns): A[u, int(items[off[u]+q])-1] = 0
A = A.tocsr(); A.eliminate_zeros(); A = A.tocsc().astype(np.float32); sClean = (sE[0], sE[1], A.T.tocsr(), np.asarray(A.sum(0)).ravel())
calib = (ev['calib'][0], ev['calib'][1], ev['calib'][2], EV['calib']['tpos'])
def test_rank(m): Ze, Ue, Fe = ev['test']; return rank_of((m.weights(Fe)[:, :, None]*Ze).sum(1), Ue, tp)
R, info = {'ref_J1': ref}, {}
for nm, side in (('val_main', sE), ('val_cleangraph', sClean)):
    Z, U, F = build(normed, ns, us, pools, side, cat.brand, cat.cats); m, regs = fit(Z, U, tps, F, calib)
    R[nm] = test_rank(m); info[nm] = {'rows': int(U[np.arange(len(tps)), tps].sum()), 'reg_calib': regs, 'w_global': np.round(m.w_global, 3).tolist()}
    print(nm, summarize(R[nm]), 'sec', round(time.time()-t0), flush=True)
Zf, Uf, Ff = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sH, cat.brand, cat.cats)
sub = np.sort(np.random.default_rng(7).choice(len(f1['tpos']), size=min(len(us), len(f1['tpos'])), replace=False))
m, regs = fit(Zf[sub], Uf[sub], f1['tpos'][sub], Ff[sub], calib); R['fold1_sub'] = test_rank(m); info['fold1_sub'] = {'rows': len(sub), 'reg_calib': regs}
print('fold1_sub', summarize(R['fold1_sub']), flush=True)
z = EV['test']; XH = Experts(H1); nH = XH.normed(z['us'], z['ns'], z['pools'])
Zh, Uh, Fh = build(nH, z['ns'], z['us'], z['pools'], sH, cat.brand, cat.cats); j1 = J1.load(run/'j1.pt')
R['single_experts'] = rank_of((j1.weights(Fh)[:, :, None]*Zh).sum(1), Uh, tp); info['single_experts'] = {'recall_U': round(float(Uh[np.arange(len(tp)), tp].mean()), 4)}
res = {'n_test': len(tp), 'info': info, 'scen': {}}; g = np.random.default_rng(0)
for k, r in R.items():
    res[k] = summarize(r, z['scen'])
    if k != 'ref_J1': d = ndcg5(ref)-ndcg5(r); res[k]['ref_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, g)}
res['_seconds'] = round(time.time()-t0); json.dump(res, open(run/'holdout_protocols.json', 'w'), indent=1); print('DONE', json.dumps({k: res[k]['NDCG@5'] for k in R}), flush=True)
