"""Stage 1 steps for one dataset (CPU).

  score <dataset dir> <run dir> [--pool 100]   main experts score the calibration and test pools  -> experts-{calib,test}.npz
  fold1 <fold-1 dir> <run dir> [--pool 100]    fold-1 experts score the fold-1 rows               -> fold1.npz
  fit   <dataset dir> <fold-1 dir> <run dir>   J1 learned on fold 1, reg chosen on calibration     -> j1.pt, stage1-{calib,test}.npz,
                                                                                                       stage1-results.json
Saved arrays: us/users (user index), ns (visible history length), pools (n, P), tpos (target position), scen, normed (n, 3, P)
min-max expert scores, union (n, P) nominated union; stage1-*.npz adds s0 (Stage 1 score, -inf outside U), Wu (n, 8) per-user
weights, wg (8,) global weights, G (n, 3) normalised expert weights."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from moerec.common import Catalog, rank_of, summarize
from moerec.data.pools import enlarge, fold1_rows50, rows50
from moerec.stage1.channels import build, graph_side, nominate
from moerec.stage1.j1 import fit


def score(exp, run, P):
    from moerec.experts.score import Experts
    t0 = time.time(); X = Experts(exp); items, off = X.hist['items'], X.hist['offsets']
    for tag in ('calib', 'test'):
        us, ns, p50, tp, scen = rows50(exp, tag); pools = enlarge(us, ns, p50, tp, items, off, X.N, P) if P > 50 else p50
        normed = X.normed(us, ns, pools); U = nominate(normed)
        np.savez(run / f'experts-{tag}.npz', us=us, ns=ns, pools=pools, tpos=tp, scen=scen, normed=normed, union=U)
        r = [rank_of(normed[:, k], np.ones_like(U), tp) for k in range(3)]
        print('SCORE', tag, len(us), 'single NDCG@5', [summarize(x)['NDCG@5'] for x in r], 'recall U', round(float(U[np.arange(len(tp)), tp].mean()), 4),
              'sec', round(time.time() - t0), flush=True)


def fold1(fold, run, P):
    from moerec.experts.score import Experts
    t0 = time.time(); X = Experts(fold); items, off = X.hist['items'], X.hist['offsets']
    us, ns, p50, tp = fold1_rows50(fold, items, off, X.N)
    pools = enlarge(us, ns, p50, tp, items, off, X.N, P) if P > 50 else p50
    normed = X.normed(us, ns, pools)
    np.savez(run / 'fold1.npz', us=us, ns=ns, pools=pools, tpos=tp, normed=normed)
    print('FOLD1', len(us), 'target in U', round(float(nominate(normed)[np.arange(len(tp)), tp].mean()), 4), 'sec', round(time.time() - t0), flush=True)


def fit_j1(exp, fold, run):
    t0 = time.time(); cat = Catalog(exp); sF, sE = graph_side(fold), graph_side(exp)
    f1 = np.load(run / 'fold1.npz'); Z, U, F = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sF, cat.brand, cat.cats)
    ev = {}
    for tag in ('calib', 'test'):
        z = np.load(run / f'experts-{tag}.npz'); Zt, Ut, Ft = build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats)
        assert np.array_equal(Ut, z['union']); ev[tag] = (z, Zt, Ut, Ft)
    z, Zc, Uc, Fc = ev['calib']; model, regs = fit(Z, U, f1['tpos'], F, (Zc, Uc, Fc, z['tpos']))
    model.save(run / 'j1.pt'); wg = model.w_global
    res = {'rows': int(U[np.arange(len(f1['tpos'])), f1['tpos']].sum()), 'reg_calib': regs, 'reg': model.reg, 'w_global': np.round(wg, 4).tolist()}
    for tag, (z, Zt, Ut, Ft) in ev.items():
        Wu = model.weights(Ft); s0 = np.where(Ut, (Wu[:, :, None] * Zt).sum(1), -np.inf).astype(np.float32); tp = z['tpos']
        R = {'j1': rank_of(s0, Ut, tp), 'j1_global': rank_of((wg[None, :, None] * Zt).sum(1), Ut, tp)}
        for k, nm in enumerate(('SASRec', 'LightGCN', 'Semantic')): R[nm] = rank_of(z['normed'][:, k], np.ones_like(Ut), tp)
        res[tag] = {'recall_U': round(float(Ut[np.arange(len(tp)), tp].mean()), 4), **{k: summarize(r, z['scen']) for k, r in R.items()}}
        g = np.clip(Wu[:, :3], 1e-6, None); g = g / g.sum(1, keepdims=True)
        np.savez(run / f'stage1-{tag}.npz', s0=s0, normed=z['normed'], G=g.astype(np.float32), pools=z['pools'], tpos=tp, ns=z['ns'], users=z['us'],
                 scen=z['scen'], union=Ut, Wu=Wu.astype(np.float32), wg=wg.astype(np.float32))
    json.dump(res, open(run / 'stage1-results.json', 'w'), indent=1)
    print('J1', json.dumps({tag: {k: v['NDCG@5'] for k, v in res[tag].items() if isinstance(v, dict)} for tag in ('calib', 'test')}), 'reg', model.reg, regs,
          'sec', round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('score'); a.add_argument('exp'); a.add_argument('run'); a.add_argument('--pool', type=int, default=100)
    b = sub.add_parser('fold1'); b.add_argument('fold'); b.add_argument('run'); b.add_argument('--pool', type=int, default=100)
    c = sub.add_parser('fit'); c.add_argument('exp'); c.add_argument('fold'); c.add_argument('run')
    x = ap.parse_args(); run = Path(x.run); run.mkdir(parents=True, exist_ok=True)
    if x.cmd == 'score': score(Path(x.exp), run, x.pool)
    elif x.cmd == 'fold1': fold1(Path(x.fold), run, x.pool)
    else: fit_j1(Path(x.exp), Path(x.fold), run)


if __name__ == '__main__':
    main()
