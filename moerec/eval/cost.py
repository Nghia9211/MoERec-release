"""Inference cost of Stage 1 and the non-LLM baselines as the candidate set grows (RQ4), CPU.

For a fixed random subset of test users, one candidate-set level per process (so the peak RSS belongs to that level):
  experts : SASRec / LightGCN / Semantic scores of every candidate (for the full catalogue: one matrix product per batch)
  late    : CombSUM over the three per-expert top-20 lists (weights fixed; the weight search is offline)
  j1      : nomination (top-20 per expert, union U), the 5 signals on U, the 12 gate inputs, the J1 per-user weights and the final ranking
Times are wall-clock per user (batches of 32 users as in large_pools.py, threads = OMP_NUM_THREADS), after loading; peak RSS covers loading.
Usage: python -m moerec.eval.cost <dataset dir> <run dir> --level p100|p500|p1000|full [--users 300]  -> <run dir>/cost-<level>.json"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from moerec.baselines.fusion import channels
from moerec.common import HIST_SIGNALS, Catalog, minmax
from moerec.data.pools import extension, rows50
from moerec.eval.large_pools import feats7
from moerec.experts.score import GCN_RECENCY, SEM_RECENCY, Experts
from moerec.runtime import gcn_user_vectors, normalize_scores, seq_at
from moerec.stage1.channels import graph_side, sig_context
from moerec.stage1.j1 import J1

BS = 32


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('run'); ap.add_argument('--level', required=True); ap.add_argument('--users', type=int, default=300)
    a = ap.parse_args(); E, run, lev = Path(a.exp), Path(a.run), a.level; th = int(os.environ.get('OMP_NUM_THREADS', '8')); torch.set_num_threads(th)
    t_load = time.perf_counter(); cat = Catalog(E); N = cat.N; brand, cats = cat.brand, cat.cats
    items, off, R, deg = graph_side(E); X = Experts(E, threads=th); hist, model, g = X.hist, X.model, X.g
    docs = np.ascontiguousarray(np.load(E / 'document_vectors.npy'), dtype=np.float32); docs_t = torch.from_numpy(docs)
    with torch.no_grad(): itemE = model.item_embeddings.weight[:N].detach().clone()
    j1 = J1.load(run / 'j1.pt'); t_load = time.perf_counter() - t_load
    us, ns, pools, tpos, _ = rows50(E, 'test'); sub = np.sort(np.random.default_rng(20261118).permutation(len(us))[:a.users])
    T = {'experts': 0., 'late': 0., 'j1': 0.}; usz = []

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

    for b0 in range(0, len(sub), BS):
        bi = sub[b0:b0 + BS]; t0 = time.perf_counter()
        seqs = torch.from_numpy(np.stack([seq_at(hist, int(us[i]), int(ns[i])) for i in bi])); sq = seqs.numpy()
        hv = docs[sq.ravel()].reshape(len(bi), sq.shape[1], 768); wv = np.exp(SEM_RECENCY * (np.arange(sq.shape[1]) - sq.shape[1]))[None, :] * (sq > 0)
        uq = (hv * wv[:, :, None]).sum(1); uq = (uq / (np.linalg.norm(uq, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        with torch.no_grad():
            h = model.encode(seqs); hu = gcn_user_vectors(seqs, g, GCN_RECENCY)
            if lev == 'full': Sfull = np.stack([(h @ itemE.T).numpy(), (hu @ g.T).numpy(), (torch.from_numpy(uq) @ docs_t.T).numpy()], 1).astype(np.float64)
        S, IDS = [], []
        for j, i in enumerate(bi):
            u, nn, tt = int(us[i]), int(ns[i]), int(tpos[i])
            if lev == 'full':
                kp = np.ones(N, bool); kp[0] = False; kp[items[off[u]:off[u] + nn]] = False; kp[int(pools[i][tt])] = True; ids = np.flatnonzero(kp); s = Sfull[j][:, ids]
            else:
                P = int(lev[1:]); base = pools[i]; ids = np.r_[base, extension(u, nn, base, base[tt], items, off, N, max(450, P - 50))[:P - 50]]
                it = torch.from_numpy(ids.astype(np.int64))
                with torch.no_grad():
                    s = np.stack([(h[j:j + 1] @ itemE[it].T).numpy()[0], (hu[j:j + 1] @ g[it].T).numpy()[0], (torch.from_numpy(uq[j:j + 1]) @ docs_t[it].T).numpy()[0]]).astype(np.float64)
            S.append(s); IDS.append(ids)
        t1 = time.perf_counter(); T['experts'] += t1 - t0
        for s in S:                                               # late CombSUM on the per-expert top-20 lists, equal weights
            _, _, Cc, _ = channels(s); np.argsort(-Cc.sum(0) if Cc.ndim > 1 else -Cc, kind='stable')[:10]
        t2 = time.perf_counter(); T['late'] += t2 - t1
        Fb, Zb = [], []
        for s, ids, i in zip(S, IDS, bi):
            normed = normalize_scores(s); Uu, _, _, _ = channels(s); usz.append(len(Uu))
            sig = signals_on(int(us[i]), int(ns[i]), ids[Uu]); Z8 = np.r_[minmax(normed[:, Uu].astype(np.float64)), minmax(sig)]
            Fb.append(np.r_[feats7(int(ns[i]), normed), sig_context(sig[None], np.ones((1, len(Uu)), bool))[0]]); Zb.append(Z8)
        Wu = j1.weights(np.stack(Fb))
        for wu, Z8 in zip(Wu, Zb): np.argsort(-(wu @ Z8), kind='stable')[:10]
        T['j1'] += time.perf_counter() - t2
    n = len(sub)
    res = {'level': lev, 'users': int(n), 'threads': th, 'load_sec': round(t_load, 2), 'union_size': round(float(np.mean(usz)), 1),
           'ms_per_user': {k: round(1000 * v / n, 3) for k, v in T.items()},
           'ms_per_user_stage1': round(1000 * (T['experts'] + T['j1']) / n, 3),
           'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    json.dump(res, open(run / f'cost-{lev}.json', 'w'), indent=1); print('COST', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
