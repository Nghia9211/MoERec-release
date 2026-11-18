"""Seed table: J1 and the final pipeline (Stage 2 rule) on test for several complete runs (experts, J1 and judge retrained per seed).
Usage: python -m moerec.eval.seeds <runs root seed 303> <runs root seed 304> ...   (each root holds <ds>/stage1-*.npz, judge-*.jsonl)"""
import argparse
import json
from pathlib import Path

import numpy as np

from moerec.common import DATASETS, bootstrap_ci, ndcg5, ndcg10
from moerec.eval.report import stage1_ranks
from moerec.stage2.fuse import fuse, read10


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); a = ap.parse_args(); out = {}
    for root in a.roots:
        out[root] = {}
        for ds in DATASETS:
            P = Path(root) / ds
            if not (P / 'judge-test.jsonl').exists(): continue
            zc, zt = np.load(P / 'stage1-calib.npz'), np.load(P / 'stage1-test.npz')
            JC, JT = read10(P / 'judge-calib.jsonl', len(zc['tpos'])), read10(P / 'judge-test.jsonl', len(zt['tpos']))
            r1, _ = stage1_ranks(zt); fin, res = fuse(zc, zt, JC, JT, crossfit=False); d = ndcg5(fin) - ndcg5(r1)
            out[root][ds] = {'j1': [round(float(ndcg5(r1).mean()), 4), round(float(ndcg10(r1).mean()), 4)],
                             'final': [round(float(ndcg5(fin).mean()), 4), round(float(ndcg10(fin).mean()), 4)],
                             'lambda': res['lambda_calib'], 'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, np.random.default_rng(0))}
    print(json.dumps(out, indent=1))
    dss = [d for d in DATASETS if all(d in out[r] for r in out)]
    for ds in dss:
        j = [out[r][ds]['j1'][0] for r in out]; f = [out[r][ds]['final'][0] for r in out]
        print(f'{ds:12s} J1 {np.mean(j):.4f} +- {np.std(j, ddof=1) if len(j) > 1 else 0:.4f}   final {np.mean(f):.4f} +- {np.std(f, ddof=1) if len(f) > 1 else 0:.4f}')


if __name__ == '__main__':
    main()
