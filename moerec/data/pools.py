"""Candidate pools.

Benchmark pool (20): the AgentRecBench task pool (target + 19 negatives) for test users, target + 19 uniform negatives for calibration.
Pool 50  = pool 20 + 30 uniform negatives (outside the visible history and the pool), shuffled, rng [20261112, user, target, 0|1].
Pool P > 50 (100, 500, 1000) = pool 50 + the first P - 50 of a fixed stream of uniform negatives, rng [777, user, target]
           (so pool 100 is nested in pool 500, which is nested in pool 1000); the target keeps its pool-50 position.
Full catalogue = every item except the user's visible history.
Fold rows (held-out users, see moerec.data.folds): fold 1 = target + 49 uniform negatives (rng 505) then the same extension;
fold 2 = target + P - 1 uniform negatives (rng 404, built in moerec.stage2.pairs).

Usage: python -m moerec.data.pools <dataset dir>   -> <dataset dir>/pools50.json  {'test': [...], 'calib': [...]}"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXTRA50, SEED50, SEED_EXT = 30, 20261112, 777


def _history(exp):
    with np.load(Path(exp) / 'data/histories.npz') as f: return f['items'], f['offsets']


def build_pool50(exp):
    exp = Path(exp); items, off = _history(exp); N = len(json.loads((exp / 'data/item_names.json').read_text()))

    def extend(rows, kind):
        out = []
        for r in rows:
            u, tgt, old = int(r['user_index']), int(r['target']), [int(c) for c in r['candidates']]
            seen = set(items[off[u]:off[u + 1]].tolist()); block = seen | set(old) | {tgt}
            rng = np.random.default_rng([SEED50, u, tgt, 0 if kind == 'test' else 1]); extra = []
            while len(extra) < EXTRA50:
                j = int(rng.integers(1, N))
                if j not in block: extra.append(j); block.add(j)
            pool = old + extra; perm = rng.permutation(len(pool)); pool = [pool[i] for i in perm]
            assert len(pool) == 50 and len(set(pool)) == 50 and tgt in pool and set(old) <= set(pool) and not (set(extra) & seen)
            out.append({'user_index': u, 'target': tgt, 'candidates': pool, 'scenario': r['scenario']})
        return out
    tasks = json.loads((exp / 'data/tasks.json').read_text()); calib = json.loads((exp / 'calib.json').read_text())
    assert all(len(t['candidates']) == 20 for t in tasks), 'data/tasks.json must hold the 20-item benchmark pools'
    res = {'test': extend(tasks, 'test'), 'calib': extend(calib, 'calib')}
    (exp / 'pools50.json').write_text(json.dumps(res))
    return {'test': len(res['test']), 'calib': len(res['calib']), 'sha256': hashlib.sha256((exp / 'pools50.json').read_bytes()).hexdigest()}


def rows50(exp, tag):
    """Pool-50 rows of the test ('test') or calibration ('calib') users: us, ns (visible history length), pools (n, 50), tpos, scen."""
    exp = Path(exp); items, off = _history(exp)
    if (exp / 'pools50.json').exists(): rows = json.loads((exp / 'pools50.json').read_text())[tag]
    else:   # layout of the original study: data/tasks.json and calib.json already hold the 50-item pools
        rows = json.loads((exp / ('data/tasks.json' if tag == 'test' else 'calib.json')).read_text())
    us = np.array([r['user_index'] for r in rows]); ns = np.array([int(off[u + 1] - off[u]) for u in us])
    pools = np.array([r['candidates'] for r in rows]); assert pools.shape[1] == 50
    tpos = np.array([r['candidates'].index(r['target']) for r in rows]); scen = np.array([r['scenario'] for r in rows])
    return us, ns, pools, tpos, scen


def extension(u, n, base, tgt, items, off, N, size=450):
    """The fixed stream of uniform negatives that enlarges a 50-item pool (outside the visible history and the pool)."""
    vis = set(items[off[u]:off[u] + n].tolist()) | set(np.asarray(base).tolist()); rng = np.random.default_rng([SEED_EXT, int(u), int(tgt)]); ext = []
    while len(ext) < size:
        x = int(rng.integers(1, N))
        if x not in vis: vis.add(x); ext.append(x)
    return ext


def enlarge(us, ns, pools50, tpos, items, off, N, P):
    size = max(450, P - 50)
    return np.stack([np.r_[b, extension(u, n, b, b[t], items, off, N, size)[:P - 50]] for u, n, b, t in zip(us, ns, pools50, tpos)])


def fold1_rows50(fold_dir, items, off, N, P=50):
    """Fold-1 rows: every held-out user whose masked item is not in the visible history; target + P - 1 uniform negatives (rng 505)."""
    held = {int(u): int(i) for u, i in json.loads((Path(fold_dir) / 'data/gate_holdout.json').read_text())}
    rng = np.random.default_rng(505); us, ns, pools, tpos = [], [], [], []
    for u in sorted(held):
        visible = set(items[off[u]:off[u + 1]].tolist()); tgt = held[u]
        if tgt in visible: continue
        negs = []
        while len(negs) < P - 1:
            j = int(rng.integers(1, N))
            if j not in visible and j != tgt and j not in negs: negs.append(j)
        pool = np.array([tgt] + negs); perm = rng.permutation(P)
        us.append(u); ns.append(int(off[u + 1] - off[u])); pools.append(pool[perm]); tpos.append(int(np.flatnonzero(perm == 0)[0]))
    return np.array(us), np.array(ns), np.stack(pools), np.array(tpos)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); print(json.dumps(build_pool50(ap.parse_args().exp)))
