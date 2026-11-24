"""Experiment: a LambdaMART baseline on the same Stage 1 inputs as J1.

LambdaMART (LightGBM lambdarank) on 26 per-candidate features: the 8 channels, 6 item-level features g_ux (rank of x under each of the
3 experts (/P), number of experts that nominate x (top-20, /3), spread of the 3 normalised expert scores (std) and of their ranks
(max - min, /P)) and the 12 user features of the gate. Groups = U of each fold-1 user, label 1 for the answer; num_leaves in {15, 63},
early stopping (50 rounds) on calibration NDCG@5, chosen on calibration.
Usage: python -m moerec.stage1.pairgate <dataset dir> <fold-1 dir> <run dir>  -> <run dir>/pairgate.json"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import build, graph_side

NOM = 20


def item_feats(normed):
    """(n, 3, P) min-max expert scores -> (n, P, 6) item-level features."""
    n, _, P = normed.shape
    r = np.argsort(np.argsort(-normed, axis=2, kind='stable'), axis=2, kind='stable').astype(np.float32)   # 0 = best
    nom = (r < NOM).sum(1) / 3.
    return np.stack([r[:, 0] / P, r[:, 1] / P, r[:, 2] / P, nom, normed.std(1), (r.max(1) - r.min(1)) / P], 2).astype(np.float32)


def flat(Z, U, F, G, t=None):
    """Per-candidate LightGBM rows over U: features (8 channels, 6 item, 12 user), group sizes, labels."""
    rows, grp, lab = [], [], []
    for i in range(len(Z)):
        idx = np.flatnonzero(U[i]); grp.append(len(idx))
        rows.append(np.concatenate([Z[i][:, idx].T, G[i][idx], np.broadcast_to(F[i], (len(idx), F.shape[1]))], 1))
        if t is not None: lab.append((idx == t[i]).astype(np.int32))
    return np.concatenate(rows).astype(np.float32), np.array(grp), (np.concatenate(lab) if t is not None else None)


def unflat(pred, U):
    out = np.full(U.shape, -np.inf, np.float32); k = 0
    for i in range(len(U)):
        idx = np.flatnonzero(U[i]); out[i, idx] = pred[k:k + len(idx)]; k += len(idx)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('fold'); ap.add_argument('run'); a = ap.parse_args()
    E, run = Path(a.exp), Path(a.run); t0 = time.time(); cat = Catalog(E); sF, sE = graph_side(Path(a.fold)), graph_side(E)
    f1 = np.load(run / 'fold1.npz'); Z, U, F = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sF, cat.brand, cat.cats); G = item_feats(f1['normed']); t = f1['tpos']
    keep = U[np.arange(len(t)), t]; Z, U, F, G, t = Z[keep], U[keep], F[keep], G[keep], t[keep]
    ev = {}
    for tag in ('calib', 'test'):
        z = np.load(run / f'experts-{tag}.npz'); Zt, Ut, Ft = build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats)
        ev[tag] = (Zt, Ut, Ft, item_feats(z['normed']), z['tpos'], z['scen'])
    j1 = np.load(run / 'stage1-test.npz'); r_j1 = rank_of(j1['s0'], j1['union'], j1['tpos'])
    res = {'rows': int(len(t)), 'j1_test': summarize(r_j1, j1['scen'])}
    Zc, Uc, Fc, Gc, tc, _ = ev['calib']; Zt, Ut, Ft, Gt, tt, sc = ev['test']

    # ---- LambdaMART
    import lightgbm as lgb
    Xtr, gtr, ytr = flat(Z, U, F, G, t); Xc, gc, yc = flat(Zc, Uc, Fc, Gc, tc)
    best = None; cal = {}
    for leaves in (15, 63):
        mdl = lgb.LGBMRanker(objective='lambdarank', n_estimators=1000, learning_rate=0.05, num_leaves=leaves, min_child_samples=50, subsample=.8, subsample_freq=1,
                             colsample_bytree=.8, random_state=303, verbose=-1, n_jobs=int(__import__('os').environ.get('OMP_NUM_THREADS', '6')))
        mdl.fit(Xtr, ytr, group=gtr, eval_set=[(Xc, yc)], eval_group=[gc], eval_at=[5], callbacks=[lgb.early_stopping(50, verbose=False)])
        v = float(ndcg5(rank_of(unflat(mdl.predict(Xc), Uc), Uc, tc)).mean()); cal[leaves] = round(v, 4)
        if best is None or v > best[0] + 1e-9: best = (v, leaves, mdl)
    Xt, _, _ = flat(Zt, Ut, Ft, Gt); r = rank_of(unflat(best[2].predict(Xt), Ut), Ut, tt); d = ndcg5(r) - ndcg5(r_j1)
    res['lambdamart'] = {'num_leaves': best[1], 'trees': int(best[2].best_iteration_ or best[2].n_estimators), 'calib': cal, 'test': summarize(r, sc),
                         'minus_j1': {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, np.random.default_rng(0))}}
    print('PAIRGATE LAMBDAMART', res['lambdamart']['test']['NDCG@5'], res['lambdamart']['minus_j1'], 'sec', round(time.time() - t0), flush=True)
    res['sec'] = round(time.time() - t0); json.dump(res, open(run / 'pairgate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
