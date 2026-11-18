"""Baseline: Feature-Weighted Linear Stacking (Sill, Takacs, Mackey, Lin, arXiv:0911.0460, 2009) on the Stage 1 channels.

FWLS blends model predictions g_i(x) with weights that are LINEAR in per-example meta-features f_j(u):
    score(u, x) = sum_i sum_j v_ij f_j(u) g_i(x),   f_0(u) = 1 (constant meta-feature)
Here g = the 8 Stage 1 channels (3 experts + 5 signals, min-max within the nominated union U) and f = the 12 gate inputs of J1 (standardised
on fold 1) plus the constant, so 8 x 13 = 104 coefficients. Same rows as J1: fold-1 users (held-out, experts retrained without the target),
candidates = U, rows whose target is outside U dropped. Two fits:
  fwls_ls  : the paper's estimator, ridge least squares on the 0/1 relevance of every candidate (closed form);
  fwls_list: the same linear form trained with J1's listwise softmax cross-entropy (Adam, full batch, learned temperature).
The ridge strength is chosen on calibration NDCG@5 ({1e-2 .. 1e4} for fwls_ls, {0, 1e-4, 1e-3, 1e-2} for fwls_list); test is measured once.
So FWLS differs from J1 only in the gate: linear in x_u (FWLS) instead of w + MLP(x_u) (J1).
Usage: python -m moerec.baselines.fwls <dataset dir> <fold-1 dir> <run dir>   (needs fold1.npz, experts-*.npz, stage1-test.npz)
-> <run dir>/fwls.json"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import build, graph_side

LS_REGS = (1e-2, 1e-1, 1., 10., 1e2, 1e3, 1e4); LIST_REGS = (0., 1e-4, 1e-3, 1e-2)


def design(Z, F):
    """(n, 8, P) channels and (n, 12) standardised meta-features -> (n, P, 104) FWLS inputs."""
    Fc = np.concatenate([np.ones((len(F), 1), np.float32), F.astype(np.float32)], 1)            # (n, 13)
    return np.einsum('ncp,nj->npcj', Z.astype(np.float32), Fc).reshape(Z.shape[0], Z.shape[2], -1)


def fit_ls(X, U, t, reg):
    rows, cols = np.nonzero(U); A = X[rows, cols].astype(np.float64); y = (cols == t[rows]).astype(np.float64)
    G = A.T @ A + reg * len(rows) / 1000. * np.eye(A.shape[1]); return np.linalg.solve(G, A.T @ y)


def fit_list(X, U, t, reg, seed=303, steps=800):
    torch.manual_seed(seed); Xt, Ut, tt = torch.tensor(X), torch.tensor(U), torch.tensor(t)
    v = torch.zeros(X.shape[2], requires_grad=True); lt = torch.tensor(np.log(10.), dtype=torch.float32, requires_grad=True)
    with torch.no_grad(): v.view(8, 13)[:3, 0] = 1 / 3          # neutral start, as J1: equal expert weights, no signals, no meta-feature effect
    opt = torch.optim.Adam([v, lt], lr=1e-2)
    for _ in range(steps):
        s = (Xt @ v) * lt.exp(); s = torch.where(Ut, s, torch.full_like(s, -1e9))
        loss = torch.nn.functional.cross_entropy(s, tt) + reg * (v ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return v.detach().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('fold'); ap.add_argument('run'); a = ap.parse_args()
    E, run = Path(a.exp), Path(a.run); t0 = time.time(); cat = Catalog(E); sF, sE = graph_side(Path(a.fold)), graph_side(E)
    f1 = np.load(run / 'fold1.npz'); Z, U, F = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sF, cat.brand, cat.cats); t = f1['tpos']
    keep = U[np.arange(len(t)), t]; Z, U, F, t = Z[keep], U[keep], F[keep], t[keep]
    mu, sd = F.mean(0), F.std(0); sd[sd < 1e-6] = 1.; X = design(Z, (F - mu) / sd)
    ev = {}
    for tag in ('calib', 'test'):
        z = np.load(run / f'experts-{tag}.npz'); Zt, Ut, Ft = build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats)
        ev[tag] = (design(Zt, (Ft - mu) / sd), Ut, z['tpos'], z['scen'])
    j1 = np.load(run / 'stage1-test.npz'); r_j1 = rank_of(j1['s0'], j1['union'], j1['tpos'])
    res = {'rows': int(len(t))}
    for name, fitter, regs in (('fwls_ls', fit_ls, LS_REGS), ('fwls_list', fit_list, LIST_REGS)):
        best, cal = None, {}
        Xc, Uc, tc, _ = ev['calib']
        for reg in regs:
            v = fitter(X, U, t, reg); s = (Xc @ v); c = float(ndcg5(rank_of(s, Uc, tc)).mean()); cal[reg] = round(c, 4)
            if best is None or c > best[0] + 1e-9: best = (c, reg, v)
        Xt, Ut, tt, sc = ev['test']; r = rank_of(Xt @ best[2], Ut, tt)
        d = ndcg5(r_j1) - ndcg5(r)
        res[name] = {'reg': best[1], 'calib': cal, 'test': summarize(r, sc),
                     'j1_minus': {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, np.random.default_rng(0))},
                     'v': np.round(best[2], 4).reshape(8, 13).tolist()}
        print('FWLS', name, 'reg', best[1], 'test NDCG@5', res[name]['test']['NDCG@5'], 'J1 -', res[name]['j1_minus'], flush=True)
    res['j1_test'] = summarize(r_j1, j1['scen']); res['sec'] = round(time.time() - t0)
    json.dump(res, open(run / 'fwls.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
