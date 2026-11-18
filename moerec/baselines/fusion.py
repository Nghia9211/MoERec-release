"""Score-fusion baselines of the three experts (no signals, no gate), each with its own weights tuned on calibration:
  early_nom     each expert keeps its top-20, U = union, every expert min-max over U, sum_k w_k z_k, rank inside U
  late_combsum  sum_k w_k mm_k, mm_k = min-max of expert k over its OWN top-20 (0 outside it)
  late_rrf      sum_k w_k / (60 + rank_k) over expert k's own top-20
w on the 3-simplex grid (step 0.05, 231 points); the grid point with the best mean calibration NDCG@5 (first maximum) is read on test.
A target outside U is lost (rank LOST) for every method."""
import numpy as np

from moerec.common import LOST, NOM, minmax, ndcg5, stable_rank, topk_stable

GRID = np.array([(a, b, 20 - a - b) for a in range(21) for b in range(21 - a)], np.float64) / 20
METHODS = ('early_nom', 'late_combsum', 'late_rrf')


def grid_ranks(C, t):
    """C (3, M) channel matrix, t target column -> (231,) stable ranks of t under every grid weight."""
    S = GRID @ C; st = S[:, t][:, None]; return (S > st).sum(1) + (S[:, :t] == st).sum(1) + 1


def channels(s, nom=NOM):
    """s (3, P) expert scores -> (U, early (3, |U|), combsum (3, |U|), rrf (3, |U|))."""
    own = [topk_stable(s[k], nom) for k in range(3)]; U = np.unique(np.concatenate(own)); pos = {int(x): q for q, x in enumerate(U.tolist())}
    Crrf, Ccmb = np.zeros((3, len(U))), np.zeros((3, len(U)))
    for k in range(3):
        mk = minmax(s[k, own[k]])
        for r, (x, v) in enumerate(zip(own[k], mk)): q = pos[int(x)]; Crrf[k, q] = 1 / (60 + r + 1); Ccmb[k, q] = v
    return U, minmax(s[:, U]), Ccmb, Crrf


def all_grid_ranks(normed, tpos):
    """(n, 3, P) expert scores -> {method: (n, 231) ranks}."""
    n = len(tpos); out = {m: np.full((n, len(GRID)), LOST, np.int64) for m in METHODS}
    for i in range(n):
        s = normed[i].astype(np.float64); t = int(tpos[i]); U, Ce, Cc, Cr = channels(s)
        if t not in set(U.tolist()): continue
        tu = int(np.flatnonzero(U == t)[0])
        out['early_nom'][i] = grid_ranks(Ce, tu); out['late_combsum'][i] = grid_ranks(Cc, tu); out['late_rrf'][i] = grid_ranks(Cr, tu)
    return out


def ranks_with(normed, tpos, W):
    """Test ranks for one weight vector per method, W = {method: (3,) weights} (vector product, as the locked tables)."""
    n = len(tpos); out = {m: np.full(n, LOST, np.int64) for m in W}
    for i in range(n):
        s = normed[i].astype(np.float64); t = int(tpos[i]); U, Ce, Cc, Cr = channels(s)
        if t not in set(U.tolist()): continue
        tu = int(np.flatnonzero(U == t)[0]); C = {'early_nom': Ce, 'late_combsum': Cc, 'late_rrf': Cr}
        for m, w in W.items(): out[m][i] = stable_rank(np.asarray(w) @ C[m], tu)
    return out


def tuned(calib, test_normed, test_tpos):
    """calib: {method: (n, 231) calibration ranks}. Returns {method: (weights, test ranks)}, weights = first calibration maximum.
    The test ranks use the vector product w @ C (GRID @ C may sum in another order and flip exact ties, frequent for RRF)."""
    W = {m: GRID[int(ndcg5(calib[m]).mean(0).argmax())].tolist() for m in METHODS}; R = ranks_with(test_normed, test_tpos, W)
    return {m: (W[m], R[m]) for m in METHODS}
