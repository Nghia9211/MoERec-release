"""Experiment: pair-level fusion weights (J2) and a LambdaMART baseline on the same Stage 1 inputs.

J1 gives every user ONE weight vector over the 8 channels (w + Delta(x_u)); FWLS does the same with a linear Delta. J2 lets the weights
depend on the (user, item) pair:  score(u, x) = sum_c (w_c + Delta_c(x_u, g_ux)) z_c(x),  with item-level features g_ux (6):
rank of x under each of the 3 experts (/P), number of experts that nominate x (top-20, /3), spread of the 3 normalised expert scores
(std) and of their ranks (max - min, /P). Delta is the J1 gate MLP ([32, 16], zero-initialised output) on [x_u, g_ux] (12 + 6 inputs).
Everything else is J1: fold-1 rows, candidates = U, listwise softmax cross-entropy with a learned temperature, neutral start, Adam
(gate 1e-3, w 1e-2), batch 256, early stopping, reg on ||Delta||^2 in {1e-3, 1e-2, 1e-1} chosen on calibration, 3 seeds averaged.
LambdaMART (LightGBM lambdarank) on the same 26 per-candidate features (8 channels, 6 item features, 12 user features), groups = U of
each fold-1 user, label 1 for the answer; num_leaves in {15, 63}, early stopping (50 rounds) on calibration NDCG@5, chosen on calibration.
Usage: python -m moerec.stage1.pairgate <dataset dir> <fold-1 dir> <run dir>  -> <run dir>/pairgate.json"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import Gate, build, graph_side
from moerec.stage1.j1 import SEEDS, REGS, fit_temp, neutral_start

NOM = 20


def item_feats(normed):
    """(n, 3, P) min-max expert scores -> (n, P, 6) item-level features."""
    n, _, P = normed.shape
    r = np.argsort(np.argsort(-normed, axis=2, kind='stable'), axis=2, kind='stable').astype(np.float32)   # 0 = best
    nom = (r < NOM).sum(1) / 3.
    return np.stack([r[:, 0] / P, r[:, 1] / P, r[:, 2] / P, nom, normed.std(1), (r.max(1) - r.min(1)) / P], 2).astype(np.float32)


class J2:
    def __init__(self, fmu, fsd, gmu, gsd, models): self.fmu, self.fsd, self.gmu, self.gsd, self.models = fmu, fsd, gmu, gsd, models

    def x(self, F, G):
        Fz = (F - self.fmu) / self.fsd; Gz = (G - self.gmu) / self.gsd
        return np.concatenate([np.broadcast_to(Fz[:, None, :], (len(F), G.shape[1], Fz.shape[1])), Gz], 2).astype(np.float32)

    def score(self, Z, F, G, chunk=512):
        out = np.zeros((len(Z), Z.shape[2]), np.float32)
        with torch.no_grad():
            for i in range(0, len(Z), chunk):
                X = torch.tensor(self.x(F[i:i + chunk], G[i:i + chunk])); Zc = torch.tensor(Z[i:i + chunk]); acc = 0
                for net, w in self.models:
                    ww = torch.tensor(w)[None, None, :] + net(X)                     # (b, P, 8)
                    acc = acc + (ww * Zc.transpose(1, 2)).sum(2)
                out[i:i + chunk] = (acc / len(self.models)).numpy()
        return out


def train_j2(X, Z, U, t, w0, reg, seed, lt0, lam=1e-3):
    torch.manual_seed(seed); n = len(t); idx = np.random.default_rng(seed).permutation(n); nv = max(1, n // 10); vi, ti = idx[:nv], idx[nv:]
    net = Gate(X.shape[2], len(w0)); lt = torch.tensor(lt0, requires_grad=True); w = torch.tensor(w0.copy(), requires_grad=True)
    opt = torch.optim.Adam([{'params': net.parameters(), 'weight_decay': 1e-4}, {'params': [lt], 'weight_decay': 0.}, {'params': [w], 'weight_decay': 0., 'lr': 1e-2}], lr=1e-3)

    def step(ii):
        d = net(X[ii]); s = ((w[None, None, :] + d) * Z[ii].transpose(1, 2)).sum(2) * lt.exp(); m = U[ii]
        s = torch.where(m, s, torch.full_like(s, -1e9)); ce = torch.nn.functional.cross_entropy(s, t[ii])
        pen = ((d ** 2).sum(2) * m).sum() / m.sum()
        return ce, ce + reg * pen + lam * (w[3:] ** 2).sum()
    best, state, bad = float('inf'), None, 0
    for ep in range(60):
        net.train(); perm = ti[np.random.default_rng(seed * 1000 + ep).permutation(len(ti))]
        for i in range(0, len(perm), 256):
            loss = step(torch.as_tensor(perm[i:i + 256]))[1]; opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vce = float(step(torch.as_tensor(vi))[0])
        if vce < best - 1e-5: best, bad, state = vce, 0, ({k: v.clone() for k, v in net.state_dict().items()}, w.detach().clone())
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(state[0]); net.eval(); return net, state[1].numpy()


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
    fmu, fsd = F.mean(0), F.std(0); fsd[fsd < 1e-6] = 1.
    Gu = G[U]; gmu, gsd = Gu.mean(0), Gu.std(0); gsd[gsd < 1e-6] = 1.
    j1 = np.load(run / 'stage1-test.npz'); r_j1 = rank_of(j1['s0'], j1['union'], j1['tpos'])
    res = {'rows': int(len(t)), 'j1_test': summarize(r_j1, j1['scen'])}

    # ---- J2
    shell = J2(fmu, fsd, gmu, gsd, []); X = torch.tensor(shell.x(F, G)); TZ, TU, Tt = torch.tensor(Z), torch.tensor(U), torch.tensor(t)
    w0 = neutral_start(); lt0 = fit_temp(TZ, TU, Tt, torch.tensor(w0)); best, cal = None, {}
    Zc, Uc, Fc, Gc, tc, _ = ev['calib']
    for reg in REGS:
        m = J2(fmu, fsd, gmu, gsd, [train_j2(X, TZ, TU, Tt, w0, reg, s, lt0) for s in SEEDS])
        v = float(ndcg5(rank_of(m.score(Zc, Fc, Gc), Uc, tc)).mean()); cal[reg] = round(v, 4)
        if best is None or v > best[0] + 1e-9: best = (v, reg, m)
    Zt, Ut, Ft, Gt, tt, sc = ev['test']; r = rank_of(best[2].score(Zt, Ft, Gt), Ut, tt); d = ndcg5(r) - ndcg5(r_j1)
    res['j2'] = {'reg': best[1], 'calib': cal, 'test': summarize(r, sc), 'minus_j1': {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, np.random.default_rng(0))}}
    print('PAIRGATE J2', res['j2']['test']['NDCG@5'], res['j2']['minus_j1'], 'sec', round(time.time() - t0), flush=True)

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
