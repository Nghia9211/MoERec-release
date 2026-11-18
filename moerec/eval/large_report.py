"""Report for pool 500 / pool 1000 / whole catalogue: inputs <runs>/<ds>/big-<level>-{calib,test}.npz (moerec.eval.large_pools) and,
when present, the judge scores <runs>/<ds>/judge-<level>-{calib,test}.jsonl. Fusion baselines: the grid weight with the best calibration
NDCG@5 (first maximum). Stage 2: the moerec.stage2.fuse rule. NDCG@5 / NDCG@10 on test, paired bootstrap of J1 and of the final pipeline
against each method. Usage: python -m moerec.eval.large_report <runs root>   -> <runs root>/large-report.json"""
import argparse
import json
from pathlib import Path

import numpy as np

from moerec.common import DATASETS, LOST, SCENARIOS, bootstrap_ci, ndcg5, ndcg10
from moerec.stage2.fuse import GRID, read10

LEVELS = ['p500', 'p1000', 'full']


def rerank(r1, S):   # vectorised moerec.stage2.fuse.rerank for (n, 10) scores
    r = r1.copy(); idx = np.flatnonzero(r1 <= 10)
    if len(idx):
        p = r1[idx] - 1; Ss = S[idx]; st = Ss[np.arange(len(idx)), p][:, None]
        r[idx] = (Ss > st).sum(1) + ((Ss == st) & (np.arange(10)[None, :] < p[:, None])).sum(1) + 1
    return r


def level(root, ds, lev):
    c, t = np.load(root / ds / f'big-{lev}-calib.npz'), np.load(root / ds / f'big-{lev}-test.npz')
    R = {k: t[f'rank_{k}'] for k in ('SASRec', 'LightGCN', 'Semantic', 'j1', 'j1_global')}
    for m in ('early_nom', 'late_combsum', 'late_rrf'):
        gi = int(ndcg5(c[f'grid_{m}']).mean(0).argmax()); R[m] = t[f'grid_{m}'][:, gi].astype(np.int64)
    info = {'recall_U': round(float((t['rank_j1'] < LOST).mean()), 4)}
    JC, JT = root / ds / f'judge-{lev}-calib.jsonl', root / ds / f'judge-{lev}-test.jsonl'
    if JC.exists() and JT.exists():
        jc, jt = read10(JC, len(c['rank_j1'])), read10(JT, len(t['rank_j1']))
        if jc is not None and jt is not None:
            tC, tT = c['top10_s0'].astype(float), t['top10_s0'].astype(float); mu = (tC.mean(), tC.std() + 1e-9, jc.mean(), jc.std() + 1e-9)
            aC, bC, aT, bT = (tC - mu[0]) / mu[1], (jc - mu[2]) / mu[3], (tT - mu[0]) / mu[1], (jt - mu[2]) / mu[3]
            g = [float((ndcg5(rerank(c['rank_j1'], aC + l * bC)) - ndcg5(c['rank_j1'])).mean()) for l in GRID]; lam = float(GRID[int(np.argmax(np.round(g, 6)))])
            R['final'] = rerank(t['rank_j1'], aT + lam * bT); info['lambda'] = lam
    sc = t['scen']; rng = np.random.default_rng(0); res = {'info': info, 'methods': {}}
    for k, r in R.items():
        e = {'NDCG@5': round(float(ndcg5(r).mean()), 4), 'NDCG@10': round(float(ndcg10(r).mean()), 4), **{s: round(float(ndcg5(r[sc == s]).mean()), 4) for s in SCENARIOS}}
        if k != 'j1': d = ndcg5(R['j1']) - ndcg5(r); e['j1_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, rng)}
        if 'final' in R and k != 'final': d = ndcg5(R['final']) - ndcg5(r); e['final_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, rng)}
        res['methods'][k] = e
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs'); a = ap.parse_args(); root = Path(a.runs); out = {}
    for lev in LEVELS:
        out[lev] = {ds: level(root, ds, lev) for ds in DATASETS if (root / ds / f'big-{lev}-test.npz').exists() and (root / ds / f'big-{lev}-calib.npz').exists()}
    json.dump(out, open(root / 'large-report.json', 'w'), indent=1)
    for lev, dd in out.items():
        if not dd: continue
        print(f'\n== {lev} ({len(dd)} datasets)  mean NDCG@5 | mean NDCG@10 | J1 better*/worse* | final better*/worse*')
        for k in ('SASRec', 'LightGCN', 'Semantic', 'early_nom', 'late_combsum', 'late_rrf', 'j1_global', 'j1', 'final'):
            rows = [dd[d]['methods'].get(k) for d in dd]
            if any(r is None for r in rows): continue
            cnt = lambda L: f"{sum(1 for x in L if x and x['ci95'][0] > 0)}/{sum(1 for x in L if x and x['ci95'][1] < 0)}" if any(L) else '-'
            print(f"{k:13s} {np.mean([r['NDCG@5'] for r in rows]):.4f} | {np.mean([r['NDCG@10'] for r in rows]):.4f} | "
                  f"{cnt([r.get('j1_minus') for r in rows])} | {cnt([r.get('final_minus') for r in rows])}")


if __name__ == '__main__':
    main()
