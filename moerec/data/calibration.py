"""Calibration users: 1,000 inference-split users (last item masked, never test), each with the held-out item + 19 uniform
negatives outside the visible history (the benchmark scheme). Pools are enlarged later (moerec.data.pools).
Usage: python -m moerec.data.calibration <dataset dir>   -> <dataset dir>/calib.json"""
import argparse
import json
from pathlib import Path

import numpy as np

from moerec.runtime import load_data, save_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('--users', type=int, default=1000); a = ap.parse_args()
    E = Path(a.exp); hist, users, names = load_data(E); off, items = hist['offsets'], hist['items']; N = len(names)
    split = json.loads((E / 'data/split_users.json').read_text()); u2i = {u: k for k, u in enumerate(users)}
    held = {}
    for u, i in hist['heldout_pairs']: held.setdefault(int(u), []).append(int(i))
    rng = np.random.default_rng(303)
    inf = [u for u in (u2i[u] for u in split['inference']) if len(held.get(u, [])) == 1]
    pick = sorted(np.asarray(inf)[rng.permutation(len(inf))[:a.users]].tolist())
    rows = []
    for u in pick:
        tgt = held[u][0]; visible = set(items[off[u]:off[u + 1]].tolist()); negs = []
        while len(negs) < 19:
            j = int(rng.integers(1, N))
            if j not in visible and j != tgt and j not in negs: negs.append(j)
        pool = np.array([tgt] + negs)[rng.permutation(20)].tolist(); n = int(off[u + 1] - off[u])
        rows.append({'user_index': u, 'target': tgt, 'candidates': pool, 'visible_history_length': n, 'scenario': 'user_cold_start' if n <= 6 else 'classic'})
    (E / 'calib.json').write_text(json.dumps(rows))
    save_json(E / 'calib-info.json', {'users': len(rows), 'from_inference_users': len(inf),
                                      'scenarios': {s: sum(r['scenario'] == s for r in rows) for s in ['classic', 'user_cold_start']}})
    print('CALIB', len(rows), flush=True)


if __name__ == '__main__':
    main()
