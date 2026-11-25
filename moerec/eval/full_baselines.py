"""Whole-catalogue ranks for the Stage 1 baselines that moerec.eval.large_pools does not cover: LIME-Rec (L1, tuned on calibration),
FWLS (listwise) and LambdaMART. Each is trained exactly as at pool 100 (fold-1 rows, choices on pool-100 calibration) and applied unchanged
to the whole catalogue (every item except the visible history), like BlindFuse. LambdaMART's rank features are divided by the training pool
size (100). BlindFuse is recomputed as a check against big-full-test.npz.
Usage: python -m moerec.eval.full_baselines <dataset dir> <fold-1 dir> <run dir>   -> <run dir>/full-baselines-test.npz"""
import sys, os, time, json
from pathlib import Path
import numpy as np, torch
from moerec.common import LOST, Catalog, minmax, ndcg5, rank_of, stable_rank
from moerec.baselines.fusion import channels
from moerec.baselines.lime import mm0, feats as lfeats, train_gate, apply as lapply, GRIDT
from moerec.baselines.fwls import design, fit_list, LIST_REGS
from moerec.stage1.pairgate import item_feats, flat, unflat
from moerec.stage1.channels import build, graph_side, sig_context
from moerec.stage1.j1 import J1
from moerec.experts.score import GCN_RECENCY, SEM_RECENCY, Experts
from moerec.runtime import gcn_user_vectors, normalize_scores, seq_at
from moerec.data.pools import rows50
from moerec.eval.large_pools import feats7
import lightgbm as lgb

E, H, RT = map(Path, sys.argv[1:4]); OUT = RT / 'full-baselines-test.npz'; t0 = time.time(); cat = Catalog(E); N = cat.N; brand, cats = cat.brand, cat.cats
sE, sH = graph_side(E), graph_side(H); items, off, R, deg = sE
f1 = np.load(RT/'fold1.npz'); zc = np.load(RT/'experts-calib.npz'); j1 = J1.load(RT/'j1.pt'); thr = int(os.environ.get('OMP_NUM_THREADS', '6'))
# ---- train on fold 1, choose on pool-100 calibration
Z, U, F = build(f1['normed'], f1['ns'], f1['us'], f1['pools'], sH, brand, cats); t = f1['tpos']; G = item_feats(f1['normed'])
Zc, Uc, Fc = build(zc['normed'], zc['ns'], zc['us'], zc['pools'], sE, brand, cats); tc = zc['tpos']; Gc = item_feats(zc['normed'])
Zo = mm0(f1['normed'].astype(np.float32)); Zco = mm0(zc['normed'].astype(np.float32)); allc = np.ones_like(zc['union']); best = None
for cfg in GRIDT:
    p = train_gate(Zo, lfeats(Zo, f1['ns']), t, True, *cfg); v = float(ndcg5(rank_of(lapply(Zco, lfeats(Zco, zc['ns']), *p)[0], allc, tc)).mean())
    if best is None or round(v, 6) > round(best[0], 6): best = (v, cfg, p)
LIME = best[2]; info = {'lime_cfg': best[1], 'lime_calib': round(best[0], 4)}
keep = U[np.arange(len(t)), t]; Zk, Uk, Fk, Gk, tk = Z[keep], U[keep], F[keep], G[keep], t[keep]
mu, sd = Fk.mean(0), Fk.std(0); sd[sd < 1e-6] = 1.; X = design(Zk, (Fk - mu) / sd); Xc = design(Zc, (Fc - mu) / sd); best = None
for reg in LIST_REGS:
    v = fit_list(X, Uk, tk, reg); c = float(ndcg5(rank_of(Xc @ v, Uc, tc)).mean())
    if best is None or c > best[0] + 1e-9: best = (c, reg, v)
FW = best[2]; info.update(fwls_reg=best[1], fwls_calib=round(best[0], 4))
Xtr, gtr, ytr = flat(Zk, Uk, Fk, Gk, tk); Xcc, gcc, ycc = flat(Zc, Uc, Fc, Gc, tc); best = None
for leaves in (15, 63):
    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=1000, learning_rate=0.05, num_leaves=leaves, min_child_samples=50, subsample=.8, subsample_freq=1,
                       colsample_bytree=.8, random_state=303, verbose=-1, n_jobs=thr)
    m.fit(Xtr, ytr, group=gtr, eval_set=[(Xcc, ycc)], eval_group=[gcc], eval_at=[5], callbacks=[lgb.early_stopping(50, verbose=False)])
    v = float(ndcg5(rank_of(unflat(m.predict(Xcc), Uc), Uc, tc)).mean())
    if best is None or v > best[0] + 1e-9: best = (v, leaves, m)
LM = best[2]; info.update(lmart_leaves=best[1], lmart_calib=round(best[0], 4)); print('TRAINED', info, 'sec', round(time.time()-t0), flush=True)
# ---- full catalogue, test users
X_ = Experts(E, threads=thr); hist, model, g = X_.hist, X_.model, X_.g
docs = np.ascontiguousarray(np.load(E/'document_vectors.npy'), dtype=np.float32); docs_t = torch.from_numpy(docs)
with torch.no_grad(): itemE = model.item_embeddings.weight[:N].detach().clone()
def sig_on(u, n, ids):
    Hh = items[off[u]:off[u]+n][-20:].astype(int); res = np.zeros((5, len(ids)))
    if len(Hh) == 0: return res
    Hi, Pi = Hh-1, ids.astype(int)-1
    co = (R[Pi] @ R[Hi].T).toarray() / np.sqrt(np.maximum(deg[Pi], 1)[:, None] * np.maximum(deg[Hi], 1)[None, :])
    res[0], res[1], res[2] = co.sum(1), co.max(1), np.log1p(deg[Pi]); hb = [brand[x] for x in Hh]; hc = [cats[x] for x in Hh]
    for p, x in enumerate(ids):
        x = int(x); bx, cx = brand[x], cats[x]
        res[3, p] = np.mean([h == bx for h in hb]) if bx else 0.; res[4, p] = np.mean([len(cx & c)/max(len(cx | c), 1) for c in hc]) if cx else 0.
    return res
us, ns, pools, tpos, scen = rows50(E, 'test'); n = len(us); RK = {k: np.full(n, LOST, np.int64) for k in ('j1', 'lime', 'fwls', 'lmart')}; BS = 32
for b0 in range(0, n, BS):
    bi = np.arange(b0, min(b0+BS, n)); seqs = torch.from_numpy(np.stack([seq_at(hist, int(us[i]), int(ns[i])) for i in bi])); sq = seqs.numpy()
    hv = docs[sq.ravel()].reshape(len(bi), sq.shape[1], 768); wv = np.exp(SEM_RECENCY*(np.arange(sq.shape[1]) - sq.shape[1]))[None, :] * (sq > 0)
    uq = (hv*wv[:, :, None]).sum(1); uq = (uq/(np.linalg.norm(uq, axis=1, keepdims=True)+1e-8)).astype(np.float32)
    with torch.no_grad():
        h = model.encode(seqs); hu = gcn_user_vectors(seqs, g, GCN_RECENCY)
        Sf = np.stack([(h @ itemE.T).numpy(), (hu @ g.T).numpy(), (torch.from_numpy(uq) @ docs_t.T).numpy()], 1).astype(np.float64)
    for j, i in enumerate(bi):
        u, nn = int(us[i]), int(ns[i]); tgt = int(pools[i][tpos[i]]); kp = np.ones(N, bool); kp[0] = False; kp[items[off[u]:off[u]+nn]] = False; kp[tgt] = True
        ids = np.flatnonzero(kp); s = Sf[j][:, ids]; ti = int(np.searchsorted(ids, tgt)); normed = normalize_scores(s)
        Zl = mm0(normed[None].astype(np.float32)); sl = lapply(Zl, lfeats(Zl, np.array([nn])), *LIME)[0][0]; RK['lime'][i] = stable_rank(sl, ti)
        Uu, _, _, _ = channels(s); sig = sig_on(u, nn, ids[Uu]); Z8 = np.r_[minmax(normed[:, Uu].astype(np.float64)), minmax(sig)]
        f12 = np.r_[feats7(nn, normed), sig_context(sig[None], np.ones((1, len(Uu)), bool))[0]].astype(np.float32)
        if ti not in set(Uu.tolist()): continue
        tu = int(np.flatnonzero(Uu == ti)[0])
        RK['j1'][i] = stable_rank(j1.weights(f12[None])[0] @ Z8, tu)
        RK['fwls'][i] = stable_rank((design(Z8[None].astype(np.float32), ((f12 - mu)/sd)[None]) @ FW)[0], tu)
        r = np.argsort(np.argsort(-normed, axis=1, kind='stable'), axis=1, kind='stable').astype(np.float32)[:, Uu]
        Gi = np.stack([r[0]/100, r[1]/100, r[2]/100, (r < 20).sum(0)/3., normed[:, Uu].std(0), (r.max(0)-r.min(0))/100], 1).astype(np.float32)
        Xi = np.concatenate([Z8.T.astype(np.float32), Gi, np.broadcast_to(f12, (len(Uu), 12))], 1); RK['lmart'][i] = stable_rank(LM.predict(Xi), tu)
    if b0 // BS % 25 == 0: print('full', bi[-1]+1, '/', n, round(time.time()-t0), flush=True)
np.savez_compressed(OUT, scen=scen, **{f'rank_{k}': v for k, v in RK.items()})
print('DONE', json.dumps({**info, **{k: round(float(ndcg5(v).mean()), 4) for k, v in RK.items()}}), 'sec', round(time.time()-t0), flush=True)
