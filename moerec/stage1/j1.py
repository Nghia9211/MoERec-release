"""J1: the global weights w (8 channels) and the per-user gate Delta(x_u) learned JOINTLY on fold 1 (held-out users), from a neutral start.

    score_u(x) = sum_c w_c(u) z_c(x),   w(u) = w + Delta(x_u),   x in U (the nominated union)
    loss = listwise softmax cross-entropy of the answer over U (learned temperature)
           + reg * ||Delta||^2 + 1e-3 * ||w_signals||^2
w starts at (1/3, 1/3, 1/3, 0, 0, 0, 0, 0); the gate output layer starts at 0, so training starts exactly at that fusion.
Adam (gate lr 1e-3 wd 1e-4, w lr 1e-2), batch 256, at most 60 epochs, early stopping on 10% of the rows (patience 6).
reg in {1e-3, 1e-2, 1e-1} chosen on calibration NDCG@5; 3 seeds (303, 304, 305), weights averaged over seeds."""
import numpy as np
import torch

from moerec.common import ndcg5, rank_of
from moerec.stage1.channels import Gate

SEEDS = (303, 304, 305); REGS = (1e-3, 1e-2, 1e-1); ALL8 = list(range(8))


def neutral_start(chans=ALL8):
    w0 = np.array([1 / 3 if c < 3 else 0. for c in chans], np.float32); ne = sum(c < 3 for c in chans); w0[:ne] = 1 / max(ne, 1)
    return w0


def fit_temp(Z, U, t, w):
    """Log-temperature of the softmax for the fixed weights w (starting point of every run)."""
    lt = torch.tensor(np.log(10.), requires_grad=True); opt = torch.optim.Adam([lt], lr=.05); s0 = (w[None, :, None] * Z).sum(1)
    for _ in range(300):
        s = torch.where(U, s0 * lt.exp(), torch.full_like(s0, -1e9)); loss = torch.nn.functional.cross_entropy(s, t)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(lt.detach())


def train_joint(F, Z, U, t, w0, reg, seed, lt0, adj=None, sig_idx=None, lam=1e-3):
    """One seed of J1. `adj`: channels the gate may move (default all); `sig_idx`: channels whose global weight gets the L2 penalty
    (default the 5 signals). Returns the gate (eval mode) and the learned global weights.
    Both paths compute the same loss; they are kept separate because the gradient summation order (broadcast add vs index_add)
    changes the trained weights in the last digits, and each path reproduces the archived runs exactly (J1 / ablation)."""
    full = adj is None; k = len(w0) if full else len(adj); A = None if full else torch.as_tensor(adj)
    torch.manual_seed(seed); n = len(t); idx = np.random.default_rng(seed).permutation(n); nv = max(1, n // 10); vi, ti = idx[:nv], idx[nv:]
    net = Gate(F.shape[1], k); lt = torch.tensor(lt0, requires_grad=True); w = torch.tensor(w0.copy(), requires_grad=True)
    opt = torch.optim.Adam([{'params': net.parameters(), 'weight_decay': 1e-4}, {'params': [lt], 'weight_decay': 0.},
                            {'params': [w], 'weight_decay': 0., 'lr': 1e-2}], lr=1e-3)

    def step(ii):
        d = net(F[ii]); ww = w[None, :] + d if full else w[None, :].repeat(len(ii), 1).index_add(1, A, d)
        s = (ww[:, :, None] * Z[ii]).sum(1) * lt.exp(); s = torch.where(U[ii], s, torch.full_like(s, -1e9))
        ce = torch.nn.functional.cross_entropy(s, t[ii])   # keep this op order: autograd accumulates the gradient of w in graph order
        return ce, ce + reg * (d ** 2).sum(1).mean() + lam * ((w[3:] ** 2).sum() if sig_idx is None else (w[sig_idx] ** 2).sum())
    best, state, bad = float('inf'), None, 0
    for ep in range(60):
        net.train(); perm = ti[np.random.default_rng(seed * 1000 + ep).permutation(len(ti))]
        for i in range(0, len(perm), 256):
            loss = step(torch.as_tensor(perm[i:i + 256]))[1]; opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vce = float(step(torch.as_tensor(vi))[0])
        if vce < best - 1e-5: best, bad, state = vce, 0, ({k: v.clone() for k, v in net.state_dict().items()}, w.detach().clone())
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(state[0]); net.eval(); return net, state[1].numpy()


def learn_global(Z, U, t, w0, steps=600, lam=1e-3):
    """Global weights alone (no gate), full-batch Adam on the same listwise loss."""
    Z, U, t = torch.tensor(Z), torch.tensor(U), torch.tensor(t)
    w = torch.tensor(w0, dtype=torch.float32, requires_grad=True); lt = torch.tensor(np.log(10.), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, lt], lr=.03); nexp = int(min(3, len(w0)))
    for _ in range(steps):
        s = (w[None, :, None] * Z).sum(1) * lt.exp(); s = torch.where(U, s, torch.full_like(s, -1e9))
        loss = torch.nn.functional.cross_entropy(s, t) + lam * (w[nexp:] ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy()


class J1:
    """Trained J1: feature standardisation, one (gate, w) per seed. weights(F) -> (n, 8) per-user weights averaged over seeds."""

    def __init__(self, mu, sd, models, reg, adj=None):
        self.mu, self.sd, self.models, self.reg, self.adj = mu, sd, models, reg, adj

    def weights(self, F):
        Fz = torch.tensor((F - self.mu) / self.sd, dtype=torch.float32); out = []
        with torch.no_grad():
            for net, w in self.models:
                ww = np.tile(w, (len(Fz), 1))
                if net is not None:
                    if self.adj is None: ww = w[None, :] + net(Fz).numpy()
                    else: ww[:, self.adj] += net(Fz).numpy()
                out.append(ww)
        return np.mean(out, 0)

    @property
    def w_global(self): return np.mean([w for _, w in self.models], 0)

    def save(self, path):
        torch.save({'reg': self.reg, 'mu': self.mu, 'sd': self.sd, 'ws': [w for _, w in self.models], 'nets': [n.state_dict() for n, _ in self.models]}, path)

    @classmethod
    def load(cls, path, d=12, k=8):
        ck = torch.load(path, weights_only=False); models = []
        for st, w in zip(ck['nets'], ck['ws']):
            n = Gate(d, k); n.load_state_dict(st); n.eval(); models.append((n, w))
        return cls(ck['mu'], ck['sd'], models, ck['reg'])


def fit(Z, U, t, F, calib, chans=ALL8, feats=None, adj=None, regs=REGS, gate=True, seeds=SEEDS, copy=False):
    """Train on fold-1 rows (Z, U, t, F); choose reg on `calib` = (Z, U, F, tpos). Rows whose target is outside U are dropped.
    copy=True always slices Z / F (the ablation runs were made that way). Returns (J1, {reg: calibration NDCG@5})."""
    # no extra copies when every channel / input is used: the CPU kernels' summation order can depend on array alignment, and the
    # locked runs were made without these copies
    allf = feats is None or list(feats) == list(range(F.shape[1])); feats = list(range(F.shape[1])) if feats is None else feats
    sel_c = (lambda A: A) if list(chans) == ALL8 and not copy else (lambda A: A[:, chans]); sel_f = (lambda A: A) if allf and not copy else (lambda A: A[:, feats])
    Z = sel_c(Z); keep = U[np.arange(len(t)), t]; Zk, Uk, tk, Fk = Z[keep], U[keep], t[keep], sel_f(F[keep])
    mu, sd = Fk.mean(0), Fk.std(0); sd[sd < 1e-6] = 1.
    TZ, TU, Tt, TF = torch.tensor(Zk), torch.tensor(Uk), torch.tensor(tk), torch.tensor((Fk - mu) / sd, dtype=torch.float32)
    w0 = neutral_start(chans); aj = None if adj is None else [chans.index(a) for a in adj if a in chans]
    sig_idx = None if (adj is None and list(chans) == ALL8) else [i for i, c in enumerate(chans) if c >= 3]
    lt0 = fit_temp(TZ, TU, Tt, torch.tensor(w0)); Zc, Uc, Fc, tc = calib; best, scores = None, {}
    for reg in (regs if gate else [0.]):
        if gate: models = [train_joint(TF, TZ, TU, Tt, w0, reg, s, lt0, aj, sig_idx) for s in seeds]
        else: models = [(None, learn_global(Zk, Uk, tk, w0))]
        m = J1(mu, sd, models, reg, aj); m.feats = feats
        v = float(ndcg5(rank_of((m.weights(sel_f(Fc))[:, :, None] * sel_c(Zc)).sum(1), Uc, tc)).mean()); scores[reg] = round(v, 4)
        if best is None or round(v, 6) > round(best[0], 6): best = (v, m)
    return best[1], scores
