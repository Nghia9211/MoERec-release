"""Stage 1 ablation (J1 recipe): every variant changes ONE thing and is re-learned the same way (joint, neutral start, reg on calibration,
3 seeds), test read once, paired bootstrap of the reference minus the variant.
  ref_J1               the locked Stage 1
  global_only          the 8 global weights alone (no gate), learned on fold 1
  in_sample_rows       rows of training users at training positions, scored by the MAIN experts (they saw these answers), as many rows as fold 1
  rows_50 / rows_25    random 50% / 25% of the fold-1 rows
  no_penalty           reg = 0 on Delta
  feat_paper7 / feat_no_length / feat_no_agreement / feat_no_confidence / feat_signal_only   gate input subsets
  out_experts_only / out_signals_only   the gate moves only the 3 expert weights / only the 5 signal weights
  drop_seq / drop_gcn / drop_sem / drop_signals   channel removed from the fusion (nomination unchanged)
  no_nomination        U = the whole pool (channels min-max over the pool)
Usage: python -m moerec.eval.ablation <dataset dir> <fold-1 dir> <run dir>   -> <run dir>/ablation.json"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import build, graph_side
from moerec.stage1.j1 import fit

FALL, CALL = list(range(12)), list(range(8))


def in_sample_rows(E, n_take, P, side, cat):
    from moerec.experts.score import Experts
    X = Experts(E); items, off = X.hist['items'], X.hist['offsets']
    with np.load(E / 'data/samples.npz') as f: smp = {k: f[k] for k in f.files}
    rows_idx = smp['gate_rows']; take = np.sort(np.random.default_rng(11).choice(len(rows_idx), size=min(n_take, len(rows_idx)), replace=False))
    us, ps = smp['train_uid'][rows_idx[take]], smp['train_pos'][rows_idx[take]]; rng = np.random.default_rng(303); pools, tp = [], []
    for u, q in zip(us, ps):
        vis = set(items[off[u]:off[u + 1]].tolist()); tgt = int(items[off[u] + q]); negs = []
        while len(negs) < P - 1:
            j = int(rng.integers(1, X.N))
            if j not in vis and j != tgt and j not in negs: negs.append(j)
        pool = np.array([tgt] + negs); perm = rng.permutation(P); pools.append(pool[perm]); tp.append(int(np.flatnonzero(perm == 0)[0]))
    pools = np.stack(pools); tp = np.array(tp); normed = X.normed(us, ps, pools)
    Z, U, F = build(normed, ps, us, pools, side, cat.brand, cat.cats); return Z, U, F, tp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('fold1'); ap.add_argument('run'); a = ap.parse_args()
    E, H1, run = Path(a.exp), Path(a.fold1), Path(a.run); t0 = time.time(); cat = Catalog(E); sE, sH = graph_side(E), graph_side(H1)
    f1 = np.load(run / 'fold1.npz'); EV = {t: np.load(run / f'experts-{t}.npz') for t in ('calib', 'test')}; BT = f1['tpos']
    base, evb = {}, {}

    def rows(nomin):
        if nomin not in base: base[nomin] = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sH, cat.brand, cat.cats, nomin)
        return base[nomin]

    def ev(nomin):
        if nomin not in evb: evb[nomin] = {t: build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats, nomin) for t, z in EV.items()}
        return evb[nomin]

    def run_(train=None, feats=FALL, adj=CALL, chans=CALL, regs=(1e-3, 1e-2, 1e-1), nomin=True, gate=True):
        Z, U, F = rows(nomin) if train is None else train[:3]; t = BT if train is None else train[3]; e = ev(nomin)
        Zc, Uc, Fc = e['calib']; model, _ = fit(Z, U, t, F, (Zc, Uc, Fc, EV['calib']['tpos']), chans=chans, feats=feats, adj=adj, regs=regs, gate=gate, copy=True)
        R = {tg: rank_of((model.weights(Fe[:, feats])[:, :, None] * Ze[:, chans]).sum(1), Ue, EV[tg]['tpos']) for tg, (Ze, Ue, Fe) in e.items()}
        return model.reg, R

    res, RK = {}, {}

    def rec(name, out):
        reg, R = out; RK[name] = R; res[name] = {'reg': reg, **{tg: summarize(r) for tg, r in R.items()}}
        print(name, res[name]['test'], 'sec', round(time.time() - t0), flush=True)
    rec('ref_J1', run_())
    rec('global_only', run_(gate=False))
    rec('in_sample_rows', run_(train=in_sample_rows(E, len(BT), f1['pools'].shape[1], sE, cat)))
    Zb, Ub, Fb = rows(True)
    for frac, nm in ((.5, 'rows_50'), (.25, 'rows_25')):
        sub = np.sort(np.random.default_rng(7).choice(len(BT), size=int(len(BT) * frac), replace=False)); rec(nm, run_(train=(Zb[sub], Ub[sub], Fb[sub], BT[sub])))
    rec('no_penalty', run_(regs=[0.]))
    for nm, fs in (('feat_paper7', list(range(7))), ('feat_no_length', [i for i in FALL if i != 0]), ('feat_no_agreement', [i for i in FALL if i not in (1, 2, 3)]),
                   ('feat_no_confidence', [i for i in FALL if i not in (4, 5, 6)]), ('feat_signal_only', list(range(7, 12)))): rec(nm, run_(feats=fs))
    rec('out_experts_only', run_(adj=[0, 1, 2])); rec('out_signals_only', run_(adj=[3, 4, 5, 6, 7]))
    for nm, ch in (('drop_seq', [1, 2, 3, 4, 5, 6, 7]), ('drop_gcn', [0, 2, 3, 4, 5, 6, 7]), ('drop_sem', [0, 1, 3, 4, 5, 6, 7]), ('drop_signals', [0, 1, 2])):
        rec(nm, run_(chans=ch, adj=ch))
    rec('no_nomination', run_(nomin=False))
    rng = np.random.default_rng(0); ref = RK['ref_J1']['test']
    for k in res:
        if k != 'ref_J1': d = ndcg5(ref) - ndcg5(RK[k]['test']); res[k]['ref_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, rng)}
    res['_seconds'] = round(time.time() - t0); json.dump(res, open(run / 'ablation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
