"""Whole-catalogue table over seeds: experts, early / late fusion (weights tuned on calibration at the full level), LIME-Rec, FWLS,
LambdaMART (moerec.eval.full_baselines), BlindFuse without gate and BlindFuse. Mean over datasets x seeds of NDCG@5/10, HR@10/20, and the
number of (seed, dataset) cells where BlindFuse is significantly better / worse (paired bootstrap of NDCG@10, 2,000 resamples).
Usage: python -m moerec.eval.full_report <runs root> 303 304 305   (reads <root>/s<seed>/<ds>/big-full-*.npz and full-baselines-test.npz)"""
import sys
import numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1]); DS = ['video_games','musical','industrial','goodreads','yelp','cell_phones','movies_tv']; SEEDS = [int(s) for s in sys.argv[2:]] or [303, 304, 305]
def nd(r, k): return np.where(r <= k, 1/np.log2(r+1), 0.)
def hr(r, k): return (r <= k).astype(float)
def boot(d, B=2000, seed=0):
    g = np.random.default_rng(seed); s = d[g.integers(0, len(d), (B, len(d)))].mean(1); return np.quantile(s, .025), np.quantile(s, .975)
M = ['SASRec','LightGCN','Semantic','early_nom','late_combsum','late_rrf','lime','fwls','lmart','j1_global','j1']
out = {}
for s in SEEDS:
    rt = ROOT/f's{s}'
    for d in DS:
        c, t = np.load(rt/d/'big-full-calib.npz'), np.load(rt/d/'big-full-test.npz'); fx = np.load(rt/d/'full-baselines-test.npz')
        R = {k: t[f'rank_{k}'] for k in ('SASRec','LightGCN','Semantic','j1','j1_global')}
        for k in ('early_nom','late_combsum','late_rrf'):
            j = int(np.argmax(nd(c[f'grid_{k}'], 5).mean(0))); R[k] = t[f'grid_{k}'][:, j]
        for k in ('lime','fwls','lmart'): R[k] = fx[f'rank_{k}']
        assert (fx['rank_j1'] == t['rank_j1']).all(), (s, d)
        for k in M:
            e = {'ndcg5': nd(R[k], 5).mean(), 'ndcg10': nd(R[k], 10).mean(), 'hr10': hr(R[k], 10).mean(), 'hr20': hr(R[k], 20).mean()}
            if k != 'j1':
                dd = nd(R['j1'], 10) - nd(R[k], 10); lo, hi = boot(dd); e['sig'] = 1 if lo > 0 else (-1 if hi < 0 else 0)
            out[(s, d, k)] = e
rows = []
for k in M:
    v = {m: np.mean([out[(s, d, k)][m] for s in SEEDS for d in DS]) for m in ('ndcg5','ndcg10','hr10','hr20')}
    w = sum(out[(s, d, k)].get('sig', 0) == 1 for s in SEEDS for d in DS); l = sum(out[(s, d, k)].get('sig', 0) == -1 for s in SEEDS for d in DS)
    sd = np.std([np.mean([out[(s, d, k)]['ndcg10'] for d in DS]) for s in SEEDS])
    rows.append((k, v, w, l, sd))
    print(f"{k:13s} N@5 {v['ndcg5']:.4f} N@10 {v['ndcg10']:.4f} (sd {sd:.4f}) HR@10 {v['hr10']:.4f} HR@20 {v['hr20']:.4f}  BlindFuse better/worse (NDCG@10, of 21) {w}/{l}")
print('PER-DATASET NDCG@10 (mean of 3 seeds)')
for k in M: print(f"{k:13s}", ' '.join(f"{np.mean([out[(s, d, k)]['ndcg10'] for s in SEEDS]):.4f}" for d in DS))
