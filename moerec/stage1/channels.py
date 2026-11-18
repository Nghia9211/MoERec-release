"""Stage 1 channels: nomination, the 8 fused channels and the 12 gate inputs.

Nomination: every expert keeps its own top-20 of the pool; U = union (a target outside U is lost).
Channels over U (each min-max normalised over U, 0 outside): 3 experts (min-max scores) + 5 signals computed against the last 20
visible items of the user:
  cooc_sum, cooc_max   cosine-normalised co-occurrence with the history items in the user-item graph (sum and max)
  popularity           log(1 + degree)
  brand                share of history items with the same brand
  category             mean Jaccard overlap of category sets with the history items
Gate inputs x_u (12): the paper's 7 context features (history length, 3 rank agreements, 3 confidences) + 5 signal-context features."""
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from moerec.common import HIST_SIGNALS, NOM
from moerec.runtime import features7

# CPU threads for the Stage 1 learning steps (results can differ in the last digits with another thread count)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '6')))


def graph_side(exp):
    """items, offsets, item-user matrix R (CSR) and item degrees of a dataset (or fold) directory."""
    exp = Path(exp)
    with np.load(exp / 'data/histories.npz') as h: items, off = h['items'], h['offsets']
    A = sp.load_npz(exp / 'data/train_user_item.npz').tocsc().astype(np.float32)
    return items, off, A.T.tocsr(), np.asarray(A.sum(0)).ravel()


def signals(us, ns, pools, items, off, R, deg, brand, cats):
    """(B, 5, P) collaborative / content signals of every candidate."""
    out = np.zeros((len(us), 5, pools.shape[1]), np.float32)
    for b, (u, n, pool) in enumerate(zip(us, ns, pools)):
        H = items[off[u]:off[u] + n][-HIST_SIGNALS:].astype(int)
        if len(H) == 0: continue
        Hi, Pi = H - 1, pool.astype(int) - 1
        co = (R[Pi] @ R[Hi].T).toarray() / np.sqrt(np.maximum(deg[Pi], 1)[:, None] * np.maximum(deg[Hi], 1)[None, :])
        out[b, 0], out[b, 1], out[b, 2] = co.sum(1), co.max(1), np.log1p(deg[Pi])
        hb = [brand[h] for h in H]; hc = [cats[h] for h in H]
        for p, x in enumerate(pool):
            x = int(x); bx, cx = brand[x], cats[x]
            out[b, 3, p] = np.mean([h == bx for h in hb]) if bx else 0.
            out[b, 4, p] = np.mean([len(cx & c) / max(len(cx | c), 1) for c in hc]) if cx else 0.
    return out


def chan_z(ch, U):
    """Min-max of every channel over U (constant -> 0.5), 0 outside U."""
    x = np.where(U[:, None], ch, np.nan); lo, hi = np.nanmin(x, 2, keepdims=True), np.nanmax(x, 2, keepdims=True)
    z = np.where(hi - lo > 0, (ch - lo) / np.where(hi - lo > 0, hi - lo, 1), .5); return np.where(U[:, None], z, 0).astype(np.float32)


def nominate(normed, nom=NOM):
    order = np.argsort(-normed, axis=2, kind='stable'); own = np.zeros_like(normed, dtype=bool)
    np.put_along_axis(own, order[:, :, :nom], True, axis=2); return own.any(1)


def sig_context(sig, U):
    """5 per-user features: how much the collaborative / content signals have to say for this user."""
    m = lambda a: np.where(U, a, np.nan)
    return np.column_stack([np.nanmean(m(sig[:, 0] > 0), 1), np.nanmax(m(sig[:, 1]), 1), np.nanmax(m(sig[:, 3]), 1),
                            np.nanmax(m(sig[:, 4]), 1), np.nanmean(m(sig[:, 4]), 1)]).astype(np.float32)


def build(normed, ns, us, pools, side, brand, cats, nominate_=True):
    """Z (n, 8, P) channels, U (n, P) nominated union, F (n, 12) gate inputs."""
    normed = normed.astype(np.float32); U = nominate(normed); sig = signals(us, ns, pools, *side, brand, cats)
    F = np.concatenate([features7(ns, normed), sig_context(sig, U)], 1)
    if not nominate_: U = np.ones_like(U)
    Z = np.concatenate([chan_z(normed, U), chan_z(sig, U)], 1)
    return Z, U, F


class Gate(torch.nn.Module):
    """The paper's gate body [32, 16], Linear -> LayerNorm -> ReLU -> Dropout(0.2), with a zero-initialised output layer."""

    def __init__(self, d, k):
        super().__init__(); L = []
        for a, b in ((d, 32), (32, 16)): L += [torch.nn.Linear(a, b), torch.nn.LayerNorm(b), torch.nn.ReLU(), torch.nn.Dropout(.2)]
        self.body = torch.nn.Sequential(*L); self.out = torch.nn.Linear(16, k)
        torch.nn.init.zeros_(self.out.weight); torch.nn.init.zeros_(self.out.bias)

    def forward(self, x): return self.out(self.body(x))
