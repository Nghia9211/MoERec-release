"""Judge training data from fold 2 (a second group of held-out users, disjoint from fold 1; experts retrained without their last item).
6,000 users whose masked item is new to them (not in the visible history, no exact title repeat in the last 50 items), each with the
masked item + P - 1 uniform negatives (rng 404). The fold-2 experts score the pool, J1 (trained on fold 1) ranks it. One pair per user:
the answer against a rival = the J1 top negative (40%), the semantic expert's top negative (30%) or a random J1 rank-2..9 negative
(30%); in 20% of the pairs neither finalist is the answer (label N). The pointwise judge turns every pair into two Yes/No examples.
Usage: python -m moerec.stage2.pairs <dataset dir> <fold-1 dir> <fold-2 dir> <run dir> [--pool 100]
-> <run dir>/fold2.npz, judge-pairs.jsonl.gz, fold2-summary.json"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from moerec.common import Catalog, Evidence, ndcg5, rank_of
from moerec.experts.score import Experts
from moerec.stage1.channels import build, graph_side
from moerec.stage1.j1 import J1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('fold1'); ap.add_argument('fold2'); ap.add_argument('run'); ap.add_argument('--pool', type=int, default=100)
    a = ap.parse_args(); E, H1, H2, run, P = Path(a.exp), Path(a.fold1), Path(a.fold2), Path(a.run), a.pool; t0 = time.time()
    assert H2.resolve() != H1.resolve()
    cat = Catalog(E); N = cat.N; model = J1.load(run / 'j1.pt'); X = Experts(H2); off, items = X.hist['offsets'], X.hist['items']
    users = json.loads((H2 / 'data/users.json').read_text()); split = json.loads((H2 / 'data/split_users.json').read_text())
    u2i = {u: i for i, u in enumerate(users)}; excl = {u2i[u] for u in split['test'] + split['inference']}; stars = np.load(H2 / 'data/history_stars.npy')
    HO = {int(u): int(t) for u, t in json.loads((H2 / 'data/gate_holdout.json').read_text())}
    F1 = {int(u) for u, _ in json.loads((H1 / 'data/gate_holdout.json').read_text())}; assert not (F1 & set(HO)), 'fold 1 and fold 2 overlap'
    low = [x.strip().lower() for x in cat.names]; cand = [u for u in sorted(HO) if u not in excl and off[u + 1] - off[u] >= 1]

    def fresh(u):
        ids = items[off[u]:off[u + 1]]; tt = HO[u]; h = ids[-50:]
        return tt not in set(ids.tolist()) and low[tt] not in {low[int(i)] for i in h}
    cand = [u for u in cand if fresh(u)]; rng = np.random.default_rng(404); pick = np.sort(rng.choice(cand, size=min(6000, len(cand)), replace=False)); rows = []
    for u in pick:
        ids = items[off[u]:off[u + 1]]; tgt = HO[int(u)]; own = set(ids.tolist()) | {tgt}; negs = []
        while len(negs) < P - 1:
            j = int(rng.integers(1, N))
            if j not in own and j not in negs: negs.append(j)
        rows.append((int(u), len(ids), np.array([tgt] + negs)[rng.permutation(P)], tgt))
    us = np.array([r[0] for r in rows]); ns = np.array([r[1] for r in rows]); pools = np.stack([r[2] for r in rows]); tp = np.array([list(r[2]).index(r[3]) for r in rows])
    normed = X.normed(us, ns, pools); Z, U, F = build(normed, ns, us, pools, graph_side(H2), cat.brand, cat.cats)
    Wu = model.weights(F); s0 = np.where(U, (Wu[:, :, None] * Z).sum(1), -np.inf); rg = rank_of(s0, U, tp)
    summ = {'users': int(len(us)), 'target_in_U': round(float(U[np.arange(len(tp)), tp].mean()), 4), 'ndcg5_j1': round(float(ndcg5(rg).mean()), 4),
            'target_top1': round(float((rg == 1).mean()), 4)}
    print('FOLD2_STAGE1', json.dumps(summ), flush=True)
    np.savez(run / 'fold2.npz', us=us, ns=ns, pools=pools, tpos=tp, normed=normed, s0=s0.astype(np.float32), union=U, W=Wu.astype(np.float32))
    ev = Evidence(cat, items, off, stars, X.docs)
    stat = {'target_top1': 0, 'target_in_union': int(U[np.arange(len(rows)), tp].sum()), 'rival_top_negative': 0, 'rival_semantic_top': 0, 'neither': 0}
    with gzip.open(run / 'judge-pairs.jsonl.gz', 'wt') as f:
        for kk, (u, n, pool, tgt) in enumerate(rows):
            order = pool[np.argsort(-s0[kk], kind='stable')]; negr = [int(x) for x in order if x != tgt]
            sem = [int(x) for x in pool[np.argsort(-normed[kk, 2], kind='stable')] if x != tgt]; r = rng.random()
            top_neg = r < .4; rival = negr[0] if r < .4 else (sem[0] if r < .7 else negr[int(rng.integers(1, 9))])
            stat['rival_semantic_top'] += int(.4 <= r < .7); neither = None
            if rng.random() < .2: neither = (negr[0], sem[0] if sem[0] != negr[0] else negr[1]); stat['neither'] += 1
            stat['target_top1'] += int(order[0] == tgt); stat['rival_top_negative'] += int(top_neg)
            history, hpos, hi, hv = ev.user(u, n)
            flip = rng.random() < .5; A, B = (rival, tgt) if flip else (tgt, rival)
            if neither is not None: A, B = neither if flip else neither[::-1]
            f.write(json.dumps({'row': kk, 'user': u, 'history': history, 'A': ev.card(A, hpos, hi, hv), 'B': ev.card(B, hpos, hi, hv),
                                'label': 'N' if neither is not None else ('B' if flip else 'A'), 'target_stage1_rank': int(np.flatnonzero(order == tgt)[0]) + 1}) + '\n')
    json.dump({**summ, **stat}, open(run / 'fold2-summary.json', 'w'), indent=1)
    print('PAIRS', len(rows), json.dumps(stat), 'sec', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
