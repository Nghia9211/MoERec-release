"""Stage 2 fusion: the judge re-orders the Stage 1 top-10 of every user,
    final(x) = z(Stage 1 score of x) + lambda * z(judge score of x),   z = standardisation with calibration-wide statistics,
ranks below 10 unchanged. lambda on the grid 0..3 step 0.05 (ties -> smaller) is chosen on calibration NDCG@5; a 2-fold cross-fit on
calibration (6 repeats) is reported; test is evaluated once, with a paired bootstrap (2,000) against Stage 1.
The same rule fuses any reranker that scores the top-10 (LlamaRec, TALLRec in moerec.eval.report).
Usage: python -m moerec.stage2.fuse <run dir> [--scores judge-{tag}.jsonl]   -> <run dir>/stage2.json"""
import argparse
import json
from pathlib import Path

import numpy as np

from moerec.common import ndcg5

GRID = np.round(np.arange(0, 3.0001, 0.05), 2)


def stage1_top10(z):
    s0, t = z['s0'], z['tpos']; o = np.argsort(-s0, 1, kind='stable')
    return (o == t[:, None]).argmax(1) + 1, np.take_along_axis(s0, o[:, :10], 1).astype(float), o


def read10(path, n):
    J = np.full((n, 10), np.nan)
    for d in map(json.loads, open(path)): J[d['row'], np.array(d['ranks']) - 1] = d['scores']
    return None if np.isnan(J).any() else J


def rerank(r1, s, idx=None):
    new = r1.copy()
    for k in np.flatnonzero(r1 <= 10):
        if idx is not None and k not in idx: continue
        o = list(np.argsort(-s[k], kind='stable')); new[k] = o.index(r1[k] - 1) + 1
    return new


def standardise(tC, JC, tT, JT):
    mu = (tC.mean(), tC.std() + 1e-9, JC.mean(), JC.std() + 1e-9)
    return (tC - mu[0]) / mu[1], (JC - mu[2]) / mu[3], (tT - mu[0]) / mu[1], (JT - mu[2]) / mu[3]


def choose_lambda(rC, aC, bC):
    g = [float((ndcg5(rerank(rC, aC + l * bC)) - ndcg5(rC)).mean()) for l in GRID]; best = int(np.argmax(np.round(g, 6)))
    return float(GRID[best]), g[best]


def fuse(zc, zt, JC, JT, crossfit=True):
    """zc / zt: stage1-{calib,test}.npz; JC / JT: (n, 10) scores aligned with the Stage 1 top-10. Returns (test ranks, report)."""
    rng = np.random.default_rng(0); rC, tC, _ = stage1_top10(zc); rT, tT, _ = stage1_top10(zt)
    aC, bC, aT, bT = standardise(tC, JC, tT, JT); lam, gain = choose_lambda(rC, aC, bC)
    res = {'lambda_calib': lam, 'calib_delta': round(gain, 4)}
    if crossfit:
        cf = []; n = len(rC)
        for _ in range(6):
            p = rng.permutation(n); h = (set(p[:n // 2].tolist()), set(p[n // 2:].tolist()))
            for x, y in ((0, 1), (1, 0)):
                ix, iy = np.array(sorted(h[x])), np.array(sorted(h[y]))
                la = GRID[int(np.argmax([np.round(float((ndcg5(rerank(rC, aC + l * bC, h[x]))[ix] - ndcg5(rC)[ix]).mean()), 6) for l in GRID]))]
                cf.append(float((ndcg5(rerank(rC, aC + la * bC, h[y]))[iy] - ndcg5(rC)[iy]).mean()))
        res['calib_crossfit'] = round(float(np.mean(cf)), 4)
    new = rerank(rT, aT + lam * bT); d = ndcg5(new) - ndcg5(rT); bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1); sc = zt['scen']
    res.update({'stage1_test_ndcg5': round(float(ndcg5(rT).mean()), 4), 'final_test_ndcg5': round(float(ndcg5(new).mean()), 4),
                'test': {'delta': round(float(d.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)],
                         'helped': int((d > 0).sum()), 'hurt': int((d < 0).sum()), 'changed': int((new != rT).sum())},
                'by_scenario': {s: {'stage1': round(float(ndcg5(rT)[sc == s].mean()), 4), 'final': round(float(ndcg5(new)[sc == s].mean()), 4)} for s in sorted(set(sc))}})
    return new, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--scores', default='judge-{tag}.jsonl'); ap.add_argument('--out', default='stage2.json')
    a = ap.parse_args(); run = Path(a.run); zc, zt = np.load(run / 'stage1-calib.npz'), np.load(run / 'stage1-test.npz')
    JC, JT = read10(run / a.scores.format(tag='calib'), len(zc['tpos'])), read10(run / a.scores.format(tag='test'), len(zt['tpos']))
    assert JC is not None and JT is not None, 'missing judge scores'
    _, res = fuse(zc, zt, JC, JT); json.dump(res, open(run / a.out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('lambda_calib', 'calib_crossfit', 'stage1_test_ndcg5', 'final_test_ndcg5', 'test')}), flush=True)


if __name__ == '__main__':
    main()
