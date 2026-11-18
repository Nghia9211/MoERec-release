"""Main table: every method on the same test users of every dataset, NDCG@5 / NDCG@10, all users and per scenario, paired bootstrap of
the final pipeline against each method. Parts that are missing are skipped (re-run as jobs finish).
Per dataset run dir (<runs>/<ds>/): stage1-{calib,test}.npz, experts-{calib,test}.npz, stage1-results.json, judge-{calib,test}.jsonl,
listwise-test.jsonl, llamarec/scores-{calib,test}.npz, tallrec/scores-{calib,test}.jsonl, lime.json.
A reranker of the top-10 (judge, LlamaRec top-10 logits, TALLRec) is fused with Stage 1 by the Stage 2 rule (moerec.stage2.fuse).
Usage: python -m moerec.eval.report <runs root> [--datasets ...]   -> <runs root>/report.json, report.md"""
import argparse
import json
from pathlib import Path

import numpy as np

from moerec.baselines import fusion
from moerec.common import DATASET_NAMES, DATASETS, LOST, SCENARIOS, bootstrap_ci, ndcg5, ndcg10, rank_of, rerank_top
from moerec.stage2.fuse import fuse, read10

NAMES = [('SASRec', 'SASRec'), ('LightGCN', 'LightGCN'), ('Semantic', 'Semantic'), ('early_nom', 'Early fusion (3 experts)'), ('late_combsum', 'Late CombSUM'),
         ('late_rrf', 'Late RRF'), ('LIME', 'LIME-Rec (gate)'), ('j1_global', 'J1 without gate'), ('j1', 'MoERec Stage 1 (J1)'), ('listwise', 'Listwise zero-shot LLM'),
         ('judge_only', 'Judge only'), ('llamarec_pure', 'LlamaRec'), ('llamarec_fused', 'LlamaRec + Stage 1'), ('tallrec_pure', 'TALLRec'),
         ('tallrec_fused', 'TALLRec + Stage 1'), ('final', 'MoERec Stage 1 + 2')]


def stage1_ranks(z):
    s0, t = z['s0'], z['tpos']; o = np.argsort(-s0, 1, kind='stable'); r = (o == t[:, None]).argmax(1) + 1
    return np.where(np.isfinite(s0[np.arange(len(t)), t]), r, LOST), o


def dataset(P):
    zt = np.load(P / 'stage1-test.npz'); zc = np.load(P / 'stage1-calib.npz'); tp = zt['tpos']; R, info = {}, {}
    for k, nm in enumerate(('SASRec', 'LightGCN', 'Semantic')): R[nm] = rank_of(zt['normed'][:, k], np.ones_like(zt['union']), tp)
    r1, o1 = stage1_ranks(zt); R['j1'] = r1
    if (P / 'experts-calib.npz').exists():
        ec, et = np.load(P / 'experts-calib.npz'), np.load(P / 'experts-test.npz')
        tuned = fusion.tuned(fusion.all_grid_ranks(ec['normed'], ec['tpos']), et['normed'], et['tpos'])
        for m, (w, r) in tuned.items(): R[m] = r; info[f'w_{m}'] = w
    def fused(sc, st): return fuse(zc, zt, sc, st, crossfit=False)
    if (P / 'judge-test.jsonl').exists() and (P / 'judge-calib.jsonl').exists():
        JT, JC = read10(P / 'judge-test.jsonl', len(tp)), read10(P / 'judge-calib.jsonl', len(zc['tpos']))
        if JT is not None and JC is not None: R['final'], res = fused(JC, JT); info['lambda_judge'] = res['lambda_calib']; R['judge_only'] = rerank_top(r1, JT)
    if (P / 'listwise-test.jsonl').exists():
        L = {json.loads(l)['row']: json.loads(l)['perm'] for l in open(P / 'listwise-test.jsonl')}
        if len(L) == len(tp):
            r = r1.copy()
            for i in np.flatnonzero(r1 <= 10): r[i] = L[int(i)].index(int(r1[i])) + 1
            R['listwise'] = r
    LL = P / 'llamarec'
    if (LL / 'scores-test.npz').exists() and (LL / 'scores-calib.npz').exists():
        At, Ac = np.load(LL / 'scores-test.npz'), np.load(LL / 'scores-calib.npz'); assert np.array_equal(At['pos'], o1[:, :20]), 'LlamaRec candidates are not the J1 top-20'
        R['llamarec_pure'] = rerank_top(r1, At['logits'], 20); R['llamarec_fused'], res = fused(Ac['logits'][:, :10], At['logits'][:, :10]); info['lambda_llamarec'] = res['lambda_calib']
    TL = P / 'tallrec'
    if (TL / 'scores-test.jsonl').exists() and (TL / 'scores-calib.jsonl').exists():
        TT, TC = read10(TL / 'scores-test.jsonl', len(tp)), read10(TL / 'scores-calib.jsonl', len(zc['tpos']))
        if TT is not None and TC is not None: R['tallrec_pure'] = rerank_top(r1, TT); R['tallrec_fused'], res = fused(TC, TT); info['lambda_tallrec'] = res['lambda_calib']
    sc = zt['scen']; rng = np.random.default_rng(0); out = {'n': {s: int((sc == s).sum()) for s in SCENARIOS}, 'info': info, 'methods': {}}
    for k, r in R.items():
        e = {'NDCG@5': round(float(ndcg5(r).mean()), 4), 'NDCG@10': round(float(ndcg10(r).mean()), 4)}
        for s in SCENARIOS: m = sc == s; e[s] = {'NDCG@5': round(float(ndcg5(r[m]).mean()), 4), 'NDCG@10': round(float(ndcg10(r[m]).mean()), 4)}
        if k != 'final' and 'final' in R: d = ndcg5(R['final']) - ndcg5(r); e['final_minus'] = {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, rng)}
        out['methods'][k] = e
    if (P / 'stage1-results.json').exists():
        g = json.load(open(P / 'stage1-results.json'))['test']['j1_global']; out['methods']['j1_global'] = {k: g[k] for k in ('NDCG@5', 'NDCG@10', *SCENARIOS)}
    if (P / 'lime.json').exists():
        lm = json.load(open(P / 'lime.json'))['test'].get('L1t LIME gate, our experts (tuned on calib)')
        if lm: out['methods']['LIME'] = {'NDCG@5': lm['NDCG@5'], 'NDCG@10': lm.get('NDCG@10')}
    return out


def markdown(out, dss):
    lines = []
    for metric in ('NDCG@5', 'NDCG@10'):
        for scope in ('all', *SCENARIOS):
            lines += [f'\n### {metric}, {"all users" if scope == "all" else ("Classic" if scope == "classic" else "Cold Start")}\n',
                      '| Method | ' + ' | '.join(DATASET_NAMES[d] for d in dss) + ' | Mean |', '|---|' + '---|' * (len(dss) + 1)]
            for k, nm in NAMES:
                vals = []
                for d in dss:
                    e = out.get(d, {}).get('methods', {}).get(k)
                    vals.append(None if e is None else (e.get(metric) if scope == 'all' else (e.get(scope) or {}).get(metric)))
                if all(v is None for v in vals): continue
                mean = f'{np.mean(vals):.4f}' if None not in vals else '-'
                lines.append(f'| {nm} | ' + ' | '.join('-' if v is None else f'{v:.4f}' for v in vals) + f' | {mean} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs'); ap.add_argument('--datasets', nargs='*', default=DATASETS); a = ap.parse_args(); root = Path(a.runs)
    dss = [d for d in a.datasets if (root / d / 'stage1-test.npz').exists()]; out = {d: dataset(root / d) for d in dss}
    json.dump(out, open(root / 'report.json', 'w'), indent=1); md = markdown(out, dss); (root / 'report.md').write_text(md, encoding='utf-8'); print(md)


if __name__ == '__main__':
    main()
