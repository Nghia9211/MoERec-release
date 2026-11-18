"""Larger candidate sets for the calibration and test users: pool 500, pool 1000 (pool 50 + the fixed negative stream, nested) and the
whole catalogue (every item except the visible history). Top-20 nomination per expert. Methods, all from the main experts:
single experts; early fusion / late CombSUM / late RRF with weights tuned on calibration at the same level (moerec.baselines.fusion);
J1 global weights; J1 (run dir j1.pt, trained at pool 100, applied unchanged; the 12 gate inputs are computed on the actual candidate set).
Also saves the J1 top-10 (ids and Stage 1 scores) and the judge prompts, so Stage 2 can score them with the pool-100 judge
(python -m moerec.stage2.score_judge <ds> <run> --prompts prompts-<level>-{tag}.jsonl.gz --out judge-<level>-{tag}.jsonl).
Usage: python -m moerec.eval.large_pools <dataset dir> <run dir> [--levels p500,p1000,full]
-> <run dir>/big-<level>-{calib,test}.npz, prompts-<level>-{calib,test}.jsonl.gz"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from moerec.baselines.fusion import GRID, channels, grid_ranks
from moerec.common import HIST_SIGNALS, LOST, Catalog, minmax, ndcg5, stable_rank
from moerec.data.pools import extension, rows50
from moerec.experts.score import GCN_RECENCY, SEM_RECENCY, Experts
from moerec.runtime import gcn_user_vectors, normalize_scores, seq_at
from moerec.stage1.channels import graph_side, sig_context
from moerec.stage1.j1 import J1
from moerec.stage2.prompts import evidence, write_top10

BS = 32


def feats7(n, normed):   # moerec.runtime.features7 for one user
    rk = rankdata(normed, method='average', axis=1); c = rk - rk.mean(1, keepdims=True)

    def ag(a, b):
        den = np.linalg.norm(c[a]) * np.linalg.norm(c[b]); return (np.clip((c[a] * c[b]).sum() / den, -1, 1) + 1) / 2 if den > 0 else .5
    return np.array([min(n / 50, 1), ag(0, 1), ag(0, 2), ag(1, 2), *(normed.max(1) - normed.mean(1))], np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('run'); ap.add_argument('--levels', default='p500,p1000,full'); a = ap.parse_args()
    E, run = Path(a.exp), Path(a.run); t0 = time.time(); cat = Catalog(E); N = cat.N; brand, cats = cat.brand, cat.cats
    items, off, R, deg = graph_side(E); X = Experts(E, threads=int(os.environ.get('OMP_NUM_THREADS', '6'))); hist, model, g = X.hist, X.model, X.g
    docs = np.ascontiguousarray(np.load(E / 'document_vectors.npy'), dtype=np.float32); docs_t = torch.from_numpy(docs)
    with torch.no_grad(): itemE = model.item_embeddings.weight[:N].detach().clone()
    j1 = J1.load(run / 'j1.pt'); WG = j1.w_global; ev = evidence(E)

    def signals_on(u, n, ids):
        Hh = items[off[u]:off[u] + n][-HIST_SIGNALS:].astype(int); res = np.zeros((5, len(ids)))
        if len(Hh) == 0: return res
        Hi, Pi = Hh - 1, ids.astype(int) - 1
        co = (R[Pi] @ R[Hi].T).toarray() / np.sqrt(np.maximum(deg[Pi], 1)[:, None] * np.maximum(deg[Hi], 1)[None, :])
        res[0], res[1], res[2] = co.sum(1), co.max(1), np.log1p(deg[Pi])
        hb = [brand[x] for x in Hh]; hc = [cats[x] for x in Hh]
        for p, x in enumerate(ids):
            x = int(x); bx, cx = brand[x], cats[x]
            res[3, p] = np.mean([h == bx for h in hb]) if bx else 0.
            res[4, p] = np.mean([len(cx & c) / max(len(cx | c), 1) for c in hc]) if cx else 0.
        return res

    for lev in a.levels.split(','):
        for tag in ('calib', 'test'):
            out_f = run / f'big-{lev}-{tag}.npz'
            if out_f.exists(): continue
            us, ns, pools, tpos, scen = rows50(E, tag); n = len(us)
            RK = {k: np.full(n, LOST, np.int64) for k in ('SASRec', 'LightGCN', 'Semantic', 'j1', 'j1_global')}
            GR = {k: np.full((n, len(GRID)), LOST, np.int32) for k in ('early_nom', 'late_combsum', 'late_rrf')}
            TOP, TOPS, UN = np.zeros((n, 10), np.int64), np.zeros((n, 10), np.float32), np.zeros(n, np.int32)
            for b0 in range(0, n, BS):
                bi = np.arange(b0, min(b0 + BS, n))
                seqs = torch.from_numpy(np.stack([seq_at(hist, int(us[i]), int(ns[i])) for i in bi])); sq = seqs.numpy()
                hv = docs[sq.ravel()].reshape(len(bi), sq.shape[1], 768); wv = np.exp(SEM_RECENCY * (np.arange(sq.shape[1]) - sq.shape[1]))[None, :] * (sq > 0)
                uq = (hv * wv[:, :, None]).sum(1); uq = (uq / (np.linalg.norm(uq, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
                with torch.no_grad():
                    h = model.encode(seqs); hu = gcn_user_vectors(seqs, g, GCN_RECENCY)
                    if lev == 'full': Sfull = np.stack([(h @ itemE.T).numpy(), (hu @ g.T).numpy(), (torch.from_numpy(uq) @ docs_t.T).numpy()], 1).astype(np.float64)
                Fb, Zb, ib = [], [], []
                for j, i in enumerate(bi):
                    u, nn, tt = int(us[i]), int(ns[i]), int(tpos[i])
                    if lev == 'full':
                        tgt = int(pools[i][tt]); kp = np.ones(N, bool); kp[0] = False; kp[items[off[u]:off[u] + nn]] = False; kp[tgt] = True
                        ids = np.flatnonzero(kp); s = Sfull[j][:, ids]; ti = int(np.searchsorted(ids, tgt))
                    else:
                        P = int(lev[1:]); base = pools[i]; ids = np.r_[base, extension(u, nn, base, base[tt], items, off, N, max(450, P - 50))[:P - 50]]
                        it = torch.from_numpy(ids.astype(np.int64))
                        with torch.no_grad():
                            s = np.stack([(h[j:j + 1] @ itemE[it].T).numpy()[0], (hu[j:j + 1] @ g[it].T).numpy()[0], (torch.from_numpy(uq[j:j + 1]) @ docs_t[it].T).numpy()[0]]).astype(np.float64)
                        ti = tt
                    for k, nm in enumerate(('SASRec', 'LightGCN', 'Semantic')): RK[nm][i] = stable_rank(s[k], ti)
                    normed = normalize_scores(s); Uu, _, Cc, Cr = channels(s); UN[i] = len(Uu)
                    sig = signals_on(u, nn, ids[Uu]); Zn = minmax(normed[:, Uu].astype(np.float64)); Z8 = np.r_[Zn, minmax(sig)]
                    f12 = np.r_[feats7(nn, normed), sig_context(sig[None], np.ones((1, len(Uu)), bool))[0]]
                    Fb.append(f12); Zb.append((Z8, Uu, ids, ti, Cc, Cr)); ib.append(i)
                Wu = j1.weights(np.stack(Fb))
                for wu, (Z8, Uu, ids, ti, Cc, Cr), i in zip(Wu, Zb, ib):
                    s0 = wu @ Z8; o = np.argsort(-s0, kind='stable')[:10]; TOP[i] = ids[Uu[o]]; TOPS[i] = s0[o]
                    if ti not in set(Uu.tolist()): continue
                    tu = int(np.flatnonzero(Uu == ti)[0]); RK['j1'][i] = stable_rank(s0, tu); RK['j1_global'][i] = stable_rank(WG @ Z8, tu)
                    GR['early_nom'][i] = grid_ranks(Z8[:3], tu); GR['late_combsum'][i] = grid_ranks(Cc, tu); GR['late_rrf'][i] = grid_ranks(Cr, tu)
                if b0 // BS % 25 == 0: print(lev, tag, bi[-1] + 1, '/', n, round(time.time() - t0), flush=True)
            np.savez_compressed(out_f, scen=scen, top10=TOP, top10_s0=TOPS, union_size=UN, **{f'rank_{k}': v for k, v in RK.items()}, **{f'grid_{k}': v for k, v in GR.items()})
            write_top10(ev, us, TOP, run / f'prompts-{lev}-{tag}.jsonl.gz')
            print('LARGE', lev, tag, {k: round(float(ndcg5(v).mean()), 4) for k, v in RK.items()}, 'recall U', round(float((RK['j1'] < LOST).mean()), 4),
                  'sec', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
