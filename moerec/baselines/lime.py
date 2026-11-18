"""LIME-Rec baseline (softmax gate over expert weights, no nomination), re-trained on the same fold-1 rows as J1.
Port: per-user min-max over the pool, softmax gate on 10 features (log history length; per expert: top-1 margin over the mean,
top-1 minus top-2, std), CE on scale * score, AdamW wd 1e-3, initial weights 0.60 / 0.15 / 0.25, batch 256.
Variants: L1 LIME gate on our experts, L2 LIME gate on SASRec + ItemCF + semantic (ItemCF on the fold-1 graph), L3 LIME global weights
on our experts; each with the paper default (scale 20, 4 epochs, lr 0.02) and tuned on calibration (scale x epochs x lr, 20 configs).
J1 (run dir j1.pt) is compared with every variant by paired bootstrap on test.
Usage: python -m moerec.baselines.lime <dataset dir> <fold-1 dir> <run dir>   -> <run dir>/lime.json"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moerec.common import HIST_SIGNALS, Catalog, bootstrap_ci, ndcg5, ndcg10, rank_of
from moerec.stage1.channels import build, graph_side
from moerec.stage1.j1 import J1

GRIDT = [(s, e, l) for s in (5., 10., 20., 40., 80.) for e in (4, 20) for l in (.02, .005)]
REF = 'J1 (locked Stage 1)'


def itemcf(us, ns, pools, items, off, R, deg):
    out = np.zeros(pools.shape, np.float32)
    for b, (u, n, pool) in enumerate(zip(us, ns, pools)):
        Hh = items[off[u]:off[u] + n][-HIST_SIGNALS:].astype(int)
        if len(Hh) == 0: continue
        Hi, Pi = Hh - 1, pool.astype(int) - 1; L = len(Hh); wt = np.arange(1, L + 1, dtype=np.float32) / L
        co = (R[Pi] @ R[Hi].T).toarray() / np.sqrt(np.maximum(deg[Pi], 1)[:, None] * np.maximum(deg[Hi], 1)[None, :]); out[b] = co @ wt
    return out


def mm0(x):
    lo, hi = x.min(-1, keepdims=True), x.max(-1, keepdims=True); return np.where(hi - lo > 0, (x - lo) / np.where(hi - lo > 0, hi - lo, 1), 0.).astype(np.float32)


def feats(Z, ns):
    s = -np.sort(-Z, -1); f = [np.log1p(ns)[:, None] / 5]
    for e in range(Z.shape[1]): f += [(s[:, e, 0] - Z[:, e].mean(-1))[:, None], (s[:, e, 0] - s[:, e, 1])[:, None], Z[:, e].std(-1)[:, None]]
    return np.concatenate(f, 1).astype(np.float32)


def train_gate(Z, X, t, gate=True, scale=20., epochs=4, lr=.02):
    torch.manual_seed(0)
    Wg = torch.zeros(X.shape[1], 3, requires_grad=gate); c = torch.tensor(np.log([.60, .15, .25]), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([c] + ([Wg] if gate else []), lr=lr, weight_decay=1e-3)
    Zt, Xt, tt = torch.tensor(Z), torch.tensor(X), torch.tensor(t)
    for ep in range(epochs):
        perm = torch.randperm(len(tt))
        for i in range(0, len(tt), 256):
            j = perm[i:i + 256]; w = torch.softmax(Xt[j] @ Wg + c, -1); s = (w[:, :, None] * Zt[j]).sum(1)
            loss = torch.nn.functional.cross_entropy(scale * s, tt[j]); opt.zero_grad(); loss.backward(); opt.step()
    return Wg.detach(), c.detach()


def apply(Z, X, Wg, c):
    w = torch.softmax(torch.tensor(X) @ Wg + c, -1).numpy(); return (w[:, :, None] * Z).sum(1), w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('fold1'); ap.add_argument('run'); a = ap.parse_args()
    E, H, run = Path(a.exp), Path(a.fold1), Path(a.run); rng = np.random.default_rng(0); cat = Catalog(E)
    sH, sE = graph_side(H), graph_side(E)
    f1 = np.load(run / 'fold1.npz'); us, ns, pools, tp = f1['us'], f1['ns'], f1['pools'], f1['tpos']; nmH = f1['normed'].astype(np.float32)
    model = J1.load(run / 'j1.pt')
    Zo = mm0(nmH); Zl = mm0(np.stack([nmH[:, 0], itemcf(us, ns, pools, *sH), nmH[:, 2]], 1))
    fit = {'L1 LIME gate, our experts': (Zo, True), 'L2 LIME gate, LIME experts (ItemCF)': (Zl, True), 'L3 LIME global weights, our experts': (Zo, False)}
    PAR = {k: train_gate(Z, feats(Z, ns), tp, g) for k, (Z, g) in fit.items()}
    res = {'rows': int(len(us)), 'j1_reg': model.reg, 'tuning': {}}
    zc = np.load(run / 'experts-calib.npz'); nmc = zc['normed'].astype(np.float32); allc = np.ones_like(zc['union'])
    Zco = mm0(nmc); Zcl = mm0(np.stack([nmc[:, 0], itemcf(zc['us'], zc['ns'], zc['pools'], *sE), nmc[:, 2]], 1))
    for k, (Ztr, g) in list(fit.items()):
        Zc = Zco if Ztr is Zo else Zcl; best = None
        for cfg in GRIDT:
            p = train_gate(Ztr, feats(Ztr, ns), tp, g, *cfg); v = float(ndcg5(rank_of(apply(Zc, feats(Zc, zc['ns']), *p)[0], allc, zc['tpos'])).mean())
            if best is None or round(v, 6) > round(best[0], 6): best = (v, cfg, p)
        kt = k.split()[0] + 't' + k[len(k.split()[0]):] + ' (tuned on calib)'; fit[kt] = (Ztr, g); PAR[kt] = best[2]
        res['tuning'][kt] = {'scale': best[1][0], 'epochs': best[1][1], 'lr': best[1][2], 'calib_ndcg5': round(best[0], 4)}
        print('LIME tuned', kt, res['tuning'][kt], flush=True)
    for tag in ('calib', 'test'):
        z = np.load(run / f'experts-{tag}.npz'); nmt = z['normed'].astype(np.float32)
        Zt8, Ut, Ft = build(z['normed'], z['ns'], z['us'], z['pools'], sE, cat.brand, cat.cats)
        rk = {REF: rank_of((model.weights(Ft)[:, :, None] * Zt8).sum(1), Ut, z['tpos'])}
        allp = np.ones_like(Ut); Zto = mm0(nmt); Ztl = mm0(np.stack([nmt[:, 0], itemcf(z['us'], z['ns'], z['pools'], *sE), nmt[:, 2]], 1))
        for k, (Ztr, g) in fit.items():
            Zx = Zto if Ztr is Zo else Ztl; rk[k] = rank_of(apply(Zx, feats(Zx, z['ns']), *PAR[k])[0], allp, z['tpos'])
        j1 = ndcg5(rk[REF]); res[tag] = {}
        for k, r in rk.items():
            e = {'NDCG@5': round(float(ndcg5(r).mean()), 4), 'NDCG@10': round(float(ndcg10(r).mean()), 4)}
            if k != REF: d = j1 - ndcg5(r); e['j1_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, rng)}
            res[tag][k] = e
    json.dump(res, open(run / 'lime.json', 'w'), indent=1)
    print('LIME', json.dumps({k: (v['NDCG@5'], v.get('j1_minus')) for k, v in res['test'].items()}), flush=True)


if __name__ == '__main__':
    main()
