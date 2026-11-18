"""Regression check against the outputs of the original study (the scripts this package was refactored from).
Runs the package on existing expert directories and compares every output with the archived one.

  python tests/reproduce.py <dataset> --exp <main expert dir> --fold1 <fold-1 dir> --fold2 <fold-2 dir> --src20 <pool-20 dataset dir>
         --ref <archived pool-100 run dir> --ref-s2 <archived stage-2 dir> --ref-local <archived local results dir> --out <new run dir>
         [--checks pools,calib,folds,score,fold1,fit,prompts,pairs,fuse,report,lime,large,ablation]

Exact equality is required for data, pools, expert scores and prompts; Stage 1 learning (CPU float32) is compared to 1e-4 on metrics.
Prints one PASS / FAIL line per check and exits 1 on any failure."""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RESULTS = []


def run(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    r = subprocess.run([sys.executable, '-m', *map(str, args)], env=env, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError(f'{args[0]} failed')
    return r.stdout


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok))); print(('PASS' if ok else 'FAIL'), name, detail, flush=True)


def same_npz(a, b, keys, tol=0.):
    za, zb = np.load(a), np.load(b); bad = []
    for ka, kb in keys:
        x, y = za[ka], zb[kb]
        if x.shape != y.shape: bad.append(f'{ka} shape {x.shape} vs {y.shape}'); continue
        if x.dtype.kind in 'fc':
            fin = np.isfinite(x) & np.isfinite(y)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)) or (fin.any() and np.abs(x[fin].astype(float) - y[fin].astype(float)).max() > tol): bad.append(ka)
        elif not np.array_equal(x, y): bad.append(ka)
    return not bad, ','.join(bad)


def same_jsonl(a, b):
    oa = gzip.open(a, 'rt') if str(a).endswith('.gz') else open(a); ob = gzip.open(b, 'rt') if str(b).endswith('.gz') else open(b)
    n = 0
    for la, lb in zip(oa, ob):
        if json.loads(la) != json.loads(lb): return False, f'line {n}'
        n += 1
    return (oa.readline() == '' and ob.readline() == ''), f'{n} lines'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ds')
    for k in ('exp', 'fold1', 'fold2', 'src20', 'ref', 'ref_s2', 'ref_local', 'out'): ap.add_argument('--' + k.replace('_', '-'), dest=k)
    ap.add_argument('--checks', default='pools,calib,folds,score,fold1,fit,prompts,pairs,fuse,report,lime,large'); a = ap.parse_args()
    ds, E, H1, H2, REF, S2, OUT = a.ds, Path(a.exp), Path(a.fold1), Path(a.fold2), Path(a.ref), Path(a.ref_s2), Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    C = a.checks.split(','); tmp = Path(tempfile.mkdtemp(prefix='moerec-repro-', dir=OUT))
    try:
        if 'pools' in C or 'calib' in C:
            T = tmp / 'p20'; T.mkdir(); os.symlink(Path(a.src20, 'data').resolve(), T / 'data')
            if 'calib' in C:
                run('moerec.data.calibration', T); x = json.loads((T / 'calib.json').read_text()); y = json.loads(Path(a.src20, 'calib.json').read_text())
                keys = ('user_index', 'target', 'candidates', 'scenario'); check('calibration users', [[r[k] for k in keys] for r in x] == [[r[k] for k in keys] for r in y], f'{len(x)} users')
            else: shutil.copy(Path(a.src20, 'calib.json'), T / 'calib.json')
            if 'pools' in C:
                run('moerec.data.pools', T); p = json.loads((T / 'pools50.json').read_text())
                ok = all([r['candidates'] for r in p[t]] == [r['candidates'] for r in json.loads(f.read_text())] for t, f in (('test', E / 'data/tasks.json'), ('calib', E / 'calib.json')))
                check('pool 50', ok, f"{len(p['test'])} test / {len(p['calib'])} calib")
        if 'folds' in C:
            for name, ref, extra in (('fold 1', H1, ['--rng', '707']), ('fold 2', H2, ['--rng', '808', '--exclude', H1 / 'data/gate_holdout.json'])):
                F = tmp / name.replace(' ', ''); run('moerec.data.folds', a.src20, F, *extra)
                ok = json.loads((F / 'data/gate_holdout.json').read_text()) == json.loads((ref / 'data/gate_holdout.json').read_text()); det = []
                for f, keys in (('histories.npz', ['items', 'offsets']), ('samples.npz', ['train_uid', 'train_pos', 'val_uid', 'val_pos', 'gate_rows']),
                                ('graph.npz', ['indices', 'values'])):
                    o, d = same_npz(F / 'data' / f, ref / 'data' / f, [(k, k) for k in keys]); ok &= o; det.append(d)
                ok &= np.array_equal(np.load(F / 'data/history_stars.npy'), np.load(ref / 'data/history_stars.npy'))
                check(name + ' data', ok, ' '.join(filter(None, det)))
        if 'score' in C:
            run('moerec.stage1.pipeline', 'score', E, OUT, '--pool', 100)
            for t in ('calib', 'test'):
                check(f'expert scores {t}', *same_npz(OUT / f'experts-{t}.npz', REF / f'stage1p100-{t}.npz', [(k, k) for k in ('us', 'ns', 'pools', 'tpos', 'scen', 'normed', 'union')]))
        if 'fold1' in C:
            run('moerec.stage1.pipeline', 'fold1', H1, OUT, '--pool', 100)
            check('fold-1 rows', *same_npz(OUT / 'fold1.npz', REF / 'fold1-p100.npz', [(k, k) for k in ('us', 'ns', 'pools', 'tpos', 'normed')]))
        if 'fit' in C:
            run('moerec.stage1.pipeline', 'fit', E, H1, OUT); x, y = json.loads((OUT / 'stage1-results.json').read_text()), json.loads((REF / 'j1-results.json').read_text())
            diff = max(abs(x[t][k]['NDCG@5'] - y[t][k]['NDCG@5']) for t in ('calib', 'test') for k in ('j1', 'j1_global', 'SASRec', 'LightGCN', 'Semantic'))
            check('J1 metrics', diff <= 1e-4 and x['reg'] == y['reg'], f"reg {x['reg']} vs {y['reg']}, max |dNDCG@5| {diff:.5f}, test J1 {x['test']['j1']['NDCG@5']} vs {y['test']['j1']['NDCG@5']}")
            for t in ('calib', 'test'):
                o, d = same_npz(OUT / f'stage1-{t}.npz', REF / f'stage1-scores-{t}.npz', [('s0', 's0'), ('Wu', 'Wu'), ('union', 'union')], tol=1e-5); check(f'J1 scores {t} (tol 1e-5)', o, d)
        if 'prompts' in C:
            run('moerec.stage2.prompts', E, OUT)
            for t in ('calib', 'test'): check(f'judge prompts {t}', *same_jsonl(OUT / f'prompts-{t}.jsonl.gz', S2 / f'{ds}-tpairs-{t}10.jsonl.gz'))
        if 'pairs' in C:
            run('moerec.stage2.pairs', E, H1, H2, OUT, '--pool', 100)
            check('judge training pairs', *same_jsonl(OUT / 'judge-pairs.jsonl.gz', S2 / f'{ds}-train-pairs-v2.jsonl.gz'))
            check('fold-2 rows', *same_npz(OUT / 'fold2.npz', REF / 'fold2-p100.npz', [(k, k) for k in ('us', 'pools', 'tpos', 'normed', 'union')]))
        L = Path(a.ref_local) if a.ref_local else None
        if 'fuse' in C and L:
            for t in ('calib', 'test'): shutil.copy(L / 'stage2C' / f'{ds}-tpoint-{t}10.jsonl', OUT / f'judge-{t}.jsonl')
            run('moerec.stage2.fuse', OUT); x = json.loads((OUT / 'stage2.json').read_text()); y = json.loads((L / 'stage2C' / f'b3b-{ds}.json').read_text())[ds]
            check('Stage 2 fusion', all(x[k] == y[k] for k in ('lambda_calib', 'calib_crossfit', 'stage1_test_ndcg5', 'final_test_ndcg5', 'test')), json.dumps(x['test']))
        if 'report' in C and L:
            for f, g in ((L / 'listwise' / f'{ds}-listwise-test.jsonl', OUT / 'listwise-test.jsonl'), (L / 'llm' / ds / f'llamarec-{ds}-test.npz', OUT / 'llamarec/scores-test.npz'),
                         (L / 'llm' / ds / f'llamarec-{ds}-calib.npz', OUT / 'llamarec/scores-calib.npz'), (L / 'llm' / ds / f'{ds}-tallrec-test10.jsonl', OUT / 'tallrec/scores-test.jsonl'),
                         (L / 'llm' / ds / f'{ds}-tallrec-calib10.jsonl', OUT / 'tallrec/scores-calib.jsonl')):
                g.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, g)
            R = tmp / 'runs'; R.mkdir(); os.symlink(OUT.resolve(), R / ds); run('moerec.eval.report', R, '--datasets', ds)
            x = json.loads((R / 'report.json').read_text())[ds]['methods']; y = json.loads((L / 'report.json').read_text())[ds]['methods']
            keys = [k for k in y if k not in ('LIME', 'j1_global')]; bad = [k for k in keys if k not in x or any(x[k][m] != y[k][m] for m in ('NDCG@5', 'NDCG@10'))]
            check('main table', not bad, f'{len(keys)} methods' + (f', differ: {bad}' if bad else ''))
            if bad: print({k: (x.get(k, {}).get('NDCG@5'), y[k]['NDCG@5']) for k in bad})
        if 'lime' in C:
            run('moerec.baselines.lime', E, H1, OUT); x, y = json.loads((OUT / 'lime.json').read_text())['test'], json.loads((REF / 'lime-p100.json').read_text())['test']
            check('LIME-Rec', all(abs(x[k]['NDCG@5'] - y[k]['NDCG@5']) <= 1e-4 for k in y), json.dumps({k: (x[k]['NDCG@5'], y[k]['NDCG@5']) for k in y}))
        if 'large' in C:
            run('moerec.eval.large_pools', E, OUT, '--levels', 'p500')
            for t in ('calib', 'test'):
                keys = ['scen', 'top10', 'union_size'] + [f'rank_{k}' for k in ('SASRec', 'LightGCN', 'Semantic', 'j1', 'j1_global')] + [f'grid_{k}' for k in ('early_nom', 'late_combsum', 'late_rrf')]
                check(f'pool 500 {t}', *same_npz(OUT / f'big-p500-{t}.npz', REF / f'big-p500-{t}.npz', [(k, k) for k in keys] + [('top10_s0', 'top10_s0')]))
                check(f'pool 500 prompts {t}', *same_jsonl(OUT / f'prompts-p500-{t}.jsonl.gz', REF.parent / 'stage2C_p500' / f'{ds}-tpairs-{t}10.jsonl.gz'))
        if 'ablation' in C:
            run('moerec.eval.ablation', E, H1, OUT); x, y = json.loads((OUT / 'ablation.json').read_text()), json.loads((REF / 'abl-p100.json').read_text())
            ks = [k for k in y if not k.startswith('_')]; d = {k: (x[k]['test']['NDCG@5'], y[k]['test']['NDCG@5']) for k in ks}
            check('Stage 1 ablation', all(abs(u - v) <= 1e-4 for u, v in d.values()), json.dumps(d))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} checks passed'); sys.exit(0 if all(ok for _, ok in RESULTS) else 1)


if __name__ == '__main__':
    main()
