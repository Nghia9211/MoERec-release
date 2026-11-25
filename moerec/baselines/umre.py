"""UMRE baseline (Xu et al., arXiv:2508.07613, "Unified Monotonic Transformation for Ranking Ensemble"), ported to the Stage 1 inputs.
No official code. Port:
  base scores p_k   = the 8 Stage 1 channels over the nominated union U (3 experts + 5 signals, min-max over U), as for FWLS / LambdaMART
  monotone transform t_k = int_0^{p_k} f_k(t, h) dt + b_k, f_k = ELU(MLP_k([t, h])) + 1 (3 x Linear(128)+ReLU, residual), Clenshaw-Curtis Q = 16
                       (paper: Q = 32), h = [user features (12 gate inputs), item features (6, as LambdaMART)]
  weights            w = Linear(U_emb (+) A1), U_emb = MLP(user features), A1 = cross-attention (query U_emb) over the embedded transformed
                       scores T_emb; no softmax (as the paper). The category attention A2 is omitted (no shared category vocabulary).
  score              s = sum_k w_k t_k
  loss               the paper's pointwise MSE to the 0/1 relevance (one positive per user), and a listwise softmax-CE variant (as J1);
                       Adam, weight decay 1e-4, batch 256 users; lr in {1e-3, 3e-4}; early stopping on calibration NDCG@5 (max 15 epochs).
Trained on the same fold-1 users as BlindFuse; the variant / lr chosen on calibration; test read once. CPU.
Usage: python -m moerec.baselines.umre <dataset dir> <fold-1 dir> <run dir>   (needs fold1.npz, experts-{calib,test}.npz, stage1-test.npz)
-> <run dir>/umre.json (GPU if available)"""
import sys, json, time, os
from pathlib import Path
import numpy as np, torch
from moerec.common import Catalog, bootstrap_ci, ndcg5, rank_of, summarize
from moerec.stage1.channels import build, graph_side
from moerec.stage1.pairgate import item_feats
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8'))); DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
E, H, RUN = map(Path, sys.argv[1:4]); OUT = RUN / 'umre.json'; t0 = time.time(); cat = Catalog(E); sE, sH = graph_side(E), graph_side(H)
def cc_nodes(Q):   # Clenshaw-Curtis nodes / weights on [0, 1]
    k = np.arange(Q + 1); x = np.cos(np.pi * k / Q); w = np.zeros(Q + 1)
    for j in range(Q + 1):
        s = 0.
        for m in range(0, Q // 2 + 1):
            b = 1 if m == 0 or 2 * m == Q else 2; s += b / (1 - 4 * m * m) * np.cos(2 * m * j * np.pi / Q)
        w[j] = s * (1 if j in (0, Q) else 2) / Q
    return torch.tensor((x + 1) / 2, dtype=torch.float32), torch.tensor(w / 2, dtype=torch.float32)
XN, XW = cc_nodes(16); XN, XW = XN.to(DEV), XW.to(DEV)
def prep(npz, side):
    z = np.load(npz); Z, U, F = build(z['normed'], z['ns'], z['us'], z['pools'], side, cat.brand, cat.cats); G = item_feats(z['normed']); return Z, U, F, G, z['tpos'], z
Zf, Uf, Ff, Gf, tf, _ = prep(RUN / 'fold1.npz', sH); keep = Uf[np.arange(len(tf)), tf]; Zf, Uf, Ff, Gf, tf = Zf[keep], Uf[keep], Ff[keep], Gf[keep], tf[keep]
mu, sd = Ff.mean(0), Ff.std(0); sd[sd < 1e-6] = 1.; gm, gs = Gf[Uf].mean(0), Gf[Uf].std(0); gs[gs < 1e-6] = 1.
EV = {tag: prep(RUN / f'experts-{tag}.npz', sE) for tag in ('calib', 'test')}
def compact(Z, U, F, G, t=None):   # keep only U columns (padded to max |U|) to save compute
    L = int(U.sum(1).max()); n = len(Z); idx = np.zeros((n, L), np.int64); m = np.zeros((n, L), bool)
    for i in range(n): c = np.flatnonzero(U[i]); idx[i, :len(c)] = c; m[i, :len(c)] = True
    Zc = np.take_along_axis(Z, idx[:, None, :], 2); Gc = np.take_along_axis(G, idx[:, :, None], 1)
    tc = None if t is None else np.array([int(np.flatnonzero(idx[i] == t[i])[0]) if U[i, t[i]] else -1 for i in range(n)])
    return (torch.tensor(Zc, dtype=torch.float32).to(DEV), torch.tensor(m).to(DEV), torch.tensor((F - mu) / sd, dtype=torch.float32).to(DEV),
            torch.tensor((Gc - gm) / gs, dtype=torch.float32).to(DEV), tc, idx)
class Res(torch.nn.Module):
    def __init__(s, d_in, d=128, L=3):
        super().__init__(); s.inp = torch.nn.Linear(d_in, d); s.hid = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(L - 1)]); s.out = torch.nn.Linear(d, 1)
    def forward(s, x):
        h = torch.relu(s.inp(x))
        for l in s.hid: h = h + torch.relu(l(h))
        return s.out(h)
class UMRE(torch.nn.Module):
    def __init__(s, K=8, dh=18, d=32):
        super().__init__(); s.f = torch.nn.ModuleList([Res(1 + dh) for _ in range(K)]); s.b = torch.nn.Parameter(torch.empty(K).uniform_(-.5, .5))
        s.ue = torch.nn.Sequential(torch.nn.Linear(12, d), torch.nn.ReLU(), torch.nn.Linear(d, d)); s.te = torch.nn.Linear(1, d); s.ke = torch.nn.Embedding(K, d)
        s.q, s.k_, s.v = torch.nn.Linear(d, d), torch.nn.Linear(d, d), torch.nn.Linear(d, d); s.w = torch.nn.Linear(2 * d, K); s.d = d
    def forward(s, Z, M, F, G):   # Z (n, K, L), M (n, L), F (n, 12), G (n, L, 6)
        n, K, L = Z.shape; h = torch.cat([F[:, None, :].expand(n, L, 12), G], 2)             # (n, L, 18)
        T = []
        for k in range(K):
            tq = Z[:, k, :, None] * XN[None, None, :]                                          # (n, L, Q)
            x = torch.cat([tq[..., None], h[:, :, None, :].expand(n, L, len(XN), h.shape[2])], 3)
            fk = torch.nn.functional.elu(s.f[k](x)).squeeze(-1) + 1; T.append((fk * XW).sum(-1) * Z[:, k] + s.b[k])
        T = torch.stack(T, 1)                                                                  # (n, K, L)
        ue = s.ue(F); tm = (T * M[:, None, :]).sum(2) / M.sum(1, keepdim=True)                 # user-level summary of each transformed score
        te = s.te(tm[..., None]) + s.ke.weight[None]                                           # (n, K, d)
        a = torch.softmax((s.q(ue)[:, None, :] * s.k_(te)).sum(-1) / s.d ** .5, 1); A1 = (a[..., None] * s.v(te)).sum(1)
        w = s.w(torch.cat([ue, A1], 1)); return (w[:, :, None] * T).sum(1)                    # (n, L)
def score(m, D):
    Z, M, F, G, _, _ = D; out = []
    with torch.no_grad():
        for i in range(0, len(Z), 512): out.append(m(Z[i:i+512], M[i:i+512], F[i:i+512], G[i:i+512]))
    return torch.cat(out).masked_fill(~M, -1e9).cpu().numpy()
def ev_ndcg(m, D, t):
    s = score(m, D); r = np.where(t >= 0, (s > s[np.arange(len(t)), np.maximum(t, 0)][:, None]).sum(1) + 1, 10 ** 6); return float(ndcg5(r).mean()), r
TR = compact(Zf, Uf, Ff, Gf, tf); tt = torch.tensor(TR[4]).to(DEV); CA = compact(*EV['calib'][:4], EV['calib'][4]); TE = compact(*EV['test'][:4], EV['test'][4])
best = None; log = {}
for loss in ('mse', 'listwise'):
    for lr in (1e-3, 3e-4):
        torch.manual_seed(303); m = UMRE().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4); bv, bs, bad = -1, None, 0
        for ep in range(15):
            m.train(); perm = torch.randperm(len(tt), device=DEV)
            for i in range(0, len(tt), 256):
                j = perm[i:i + 256]; s = m(TR[0][j], TR[1][j], TR[2][j], TR[3][j])
                if loss == 'mse':
                    y = torch.zeros_like(s); y[torch.arange(len(j), device=DEV), tt[j]] = 1; l = (((s - y) ** 2) * TR[1][j]).sum() / TR[1][j].sum()
                else: l = torch.nn.functional.cross_entropy(s.masked_fill(~TR[1][j], -1e9), tt[j])
                opt.zero_grad(); l.backward(); opt.step()
            m.eval(); v, _ = ev_ndcg(m, CA, CA[4]); print('  ep', ep + 1, round(v, 4), round(time.time() - t0), flush=True)
            if v > bv + 1e-5: bv, bad, bs = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 3: break
        m.load_state_dict(bs); log[f'{loss}_lr{lr}'] = {'calib': round(bv, 4), 'epochs': ep + 1}; print('UMRE', loss, lr, round(bv, 4), 'sec', round(time.time() - t0), flush=True)
        if best is None or bv > best[0] + 1e-9: best = (bv, f'{loss}_lr{lr}', m)
_, r = ev_ndcg(best[2], TE, TE[4]); z = EV['test'][5]
j1 = np.load(RUN / 'stage1-test.npz'); rj = rank_of(j1['s0'], j1['union'], j1['tpos']); d = ndcg5(rj) - ndcg5(r)
res = {'chosen': best[1], 'calib': log, 'test': summarize(r, z['scen']), 'j1_test': summarize(rj, z['scen']),
       'j1_minus': {'delta': round(float(d.mean()), 4), 'ci95': bootstrap_ci(d, np.random.default_rng(0))}, 'sec': round(time.time() - t0)}
json.dump(res, open(OUT, 'w'), indent=1); print('DONE', res['chosen'], res['test']['NDCG@5'], 'BlindFuse', res['j1_test']['NDCG@5'], res['j1_minus'], flush=True)
