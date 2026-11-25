"""BlindFuse top-20 on the whole catalogue (main experts, locked j1.pt), calibration and test users, written as a Stage 1 file so the LLM
rerankers (LlamaRec top-20, TALLRec / BlindJudge top-10) can score it unchanged: stage1-{tag}.npz with s0 / pools = the top-20 scores / ids in
order, users, ns, scen, tpos (target position in the top-20, -1 if absent) and rank_full (BlindFuse rank of the target in the catalogue).
Usage: python -m moerec.eval.full_top20 <dataset dir> <run dir with j1.pt> <out dir>"""
import sys, os, time
from pathlib import Path
import numpy as np, torch
from moerec.common import LOST, Catalog, minmax, stable_rank
from moerec.baselines.fusion import channels
from moerec.stage1.channels import graph_side, sig_context
from moerec.stage1.j1 import J1
from moerec.experts.score import GCN_RECENCY, SEM_RECENCY, Experts
from moerec.runtime import gcn_user_vectors, normalize_scores, seq_at
from moerec.data.pools import rows50
from moerec.eval.large_pools import feats7
E, RT, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); cat = Catalog(E); N = cat.N; brand, cats = cat.brand, cat.cats
items, off, R, deg = graph_side(E); j1 = J1.load(RT/'j1.pt'); thr = int(os.environ.get('OMP_NUM_THREADS', '4'))
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
for tag in ('calib', 'test'):
    us, ns, pools, tpos, scen = rows50(E, tag); n = len(us); TOP = np.zeros((n, 20), np.int64); TS = np.full((n, 20), -1e9, np.float32)
    RK = np.full(n, LOST, np.int64); TG = np.zeros(n, np.int64); BS = 32
    for b0 in range(0, n, BS):
        bi = np.arange(b0, min(b0+BS, n)); seqs = torch.from_numpy(np.stack([seq_at(hist, int(us[i]), int(ns[i])) for i in bi])); sq = seqs.numpy()
        hv = docs[sq.ravel()].reshape(len(bi), sq.shape[1], 768); wv = np.exp(SEM_RECENCY*(np.arange(sq.shape[1]) - sq.shape[1]))[None, :] * (sq > 0)
        uq = (hv*wv[:, :, None]).sum(1); uq = (uq/(np.linalg.norm(uq, axis=1, keepdims=True)+1e-8)).astype(np.float32)
        with torch.no_grad():
            h = model.encode(seqs); hu = gcn_user_vectors(seqs, g, GCN_RECENCY)
            Sf = np.stack([(h @ itemE.T).numpy(), (hu @ g.T).numpy(), (torch.from_numpy(uq) @ docs_t.T).numpy()], 1).astype(np.float64)
        for j, i in enumerate(bi):
            u, nn = int(us[i]), int(ns[i]); tgt = int(pools[i][tpos[i]]); TG[i] = tgt; kp = np.ones(N, bool); kp[0] = False; kp[items[off[u]:off[u]+nn]] = False; kp[tgt] = True
            ids = np.flatnonzero(kp); s = Sf[j][:, ids]; ti = int(np.searchsorted(ids, tgt)); normed = normalize_scores(s)
            Uu, _, _, _ = channels(s); sig = sig_on(u, nn, ids[Uu]); Z8 = np.r_[minmax(normed[:, Uu].astype(np.float64)), minmax(sig)]
            f12 = np.r_[feats7(nn, normed), sig_context(sig[None], np.ones((1, len(Uu)), bool))[0]].astype(np.float32)
            s0 = j1.weights(f12[None])[0] @ Z8; o = np.argsort(-s0, kind='stable')[:20]; TOP[i, :len(o)] = ids[Uu[o]]; TS[i, :len(o)] = s0[o]
            if ti in set(Uu.tolist()): RK[i] = stable_rank(s0, int(np.flatnonzero(Uu == ti)[0]))
        if b0 // BS % 25 == 0: print(tag, bi[-1]+1, '/', n, round(time.time()-t0), flush=True)
    tp = np.array([int(np.flatnonzero(TOP[i] == TG[i])[0]) if TG[i] in TOP[i] else -1 for i in range(n)])
    np.savez(OUT/f'stage1-{tag}.npz', s0=TS, pools=TOP, users=us, ns=ns, scen=scen, tpos=tp, rank_full=RK, target=TG)
    print('DONE', tag, 'HR@20', round(float((tp >= 0).mean()), 4), 'sec', round(time.time()-t0), flush=True)
