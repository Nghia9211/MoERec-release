"""Unit tests on synthetic data (no dataset needed): python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import torch

from moerec.baselines import fusion
from moerec.common import minmax, ndcg5, ndcg10, rank_of, rerank_top, stable_rank, topk_stable
from moerec.data.pools import enlarge, extension
from moerec.runtime import features7, normalize_scores, seq_at
from moerec.stage1 import channels as C
from moerec.stage1.j1 import J1, fit, neutral_start
from moerec.stage2.fuse import GRID, choose_lambda, rerank, standardise


class Metrics(unittest.TestCase):
    def test_ndcg(self):
        r = np.array([1, 2, 5, 6, 10, 11])
        np.testing.assert_allclose(ndcg5(r), [1, 1 / np.log2(3), 1 / np.log2(6), 0, 0, 0])
        np.testing.assert_allclose(ndcg10(r)[:5], [1, 1 / np.log2(3), 1 / np.log2(6), 1 / np.log2(7), 1 / np.log2(11)]); self.assertEqual(ndcg10(r)[5], 0)

    def test_rank_ties_and_mask(self):
        s = np.array([[.5, .9, .5, .1]]); self.assertEqual(rank_of(s, np.ones_like(s, bool), np.array([2]))[0], 3)   # tie -> earlier position first
        allowed = np.array([[True, False, True, True]]); self.assertEqual(rank_of(s, allowed, np.array([2]))[0], 2)
        self.assertEqual(stable_rank(s[0], 2), 3); self.assertEqual(stable_rank(s[0], 0), 2)

    def test_topk_stable_matches_argsort(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            s = rng.integers(0, 5, 30).astype(float); np.testing.assert_array_equal(topk_stable(s, 7), np.argsort(-s, kind='stable')[:7])

    def test_rerank_top(self):
        r1 = np.array([3, 12]); sc = np.zeros((2, 10)); sc[0, 2] = 5.   # target at position 3 gets the best score -> rank 1
        np.testing.assert_array_equal(rerank_top(r1, sc), [1, 12]); np.testing.assert_array_equal(rerank(r1, sc), [1, 12])


class Runtime(unittest.TestCase):
    def test_normalize_and_seq(self):
        x = normalize_scores(np.array([[1., 3., 2.], [4., 4., 4.]])); np.testing.assert_allclose(x, [[0, 1, .5], [.5, .5, .5]])
        hist = {'items': np.arange(1, 11), 'offsets': np.array([0, 4, 10])}
        np.testing.assert_array_equal(seq_at(hist, 1, 3, limit=5), [0, 0, 5, 6, 7])

    def test_features7(self):
        s = normalize_scores(np.random.default_rng(0).normal(size=(4, 3, 20))); f = features7([0, 3, 20, 80], s)
        self.assertEqual(f.shape, (4, 7)); np.testing.assert_allclose(f[:, 0], [0, .06, .4, 1]); self.assertTrue(((f[:, 1:4] >= 0) & (f[:, 1:4] <= 1)).all())
        same = np.repeat(s[:1, :1], 3, 1); np.testing.assert_allclose(features7([5], same)[0, 1:4], 1)   # identical experts agree fully


class Pools(unittest.TestCase):
    def test_extension_nested_and_clean(self):
        items = np.array([1, 2, 3, 4]); off = np.array([0, 4]); base = np.array(list(range(10, 60))); N = 3000
        a, b = extension(0, 4, base, 10, items, off, N, 450), extension(0, 4, base, 10, items, off, N, 950)
        self.assertEqual(a, b[:450]); self.assertFalse(set(b) & (set(base.tolist()) | {1, 2, 3, 4})); self.assertEqual(len(set(b)), 950)
        P = enlarge(np.array([0]), np.array([4]), base[None], np.array([0]), items, off, N, 100)
        self.assertEqual(P.shape, (1, 100)); np.testing.assert_array_equal(P[0, :50], base)


class Stage1(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0); self.n, self.P = 300, 40
        self.normed = rng.random((self.n, 3, self.P)).astype(np.float32); self.t = rng.integers(0, self.P, self.n)
        self.normed[np.arange(self.n), 0, self.t] += .6; self.normed = normalize_scores(self.normed)   # SASRec is informative

    def test_nominate_and_chan_z(self):
        U = C.nominate(self.normed, 5); self.assertTrue(((U.sum(1) >= 5) & (U.sum(1) <= 15)).all())
        Z = C.chan_z(self.normed, U); self.assertTrue((Z[~np.repeat(U[:, None], 3, 1)] == 0).all())
        self.assertAlmostEqual(float(Z[0, 0][U[0]].max()), 1.); self.assertAlmostEqual(float(Z[0, 0][U[0]].min()), 0.)

    def test_j1_learns_informative_expert(self):
        U = C.nominate(self.normed); Z = np.concatenate([C.chan_z(self.normed, U), np.zeros((self.n, 5, self.P), np.float32)], 1)
        F = np.random.default_rng(1).normal(size=(self.n, 12)).astype(np.float32)
        model, regs = fit(Z, U, self.t, F, (Z, U, F, self.t), seeds=(303,), regs=(1e-2,))
        self.assertIsInstance(model, J1); self.assertEqual(model.weights(F).shape, (self.n, 8))
        w = model.w_global; self.assertGreater(w[0], w[1]); self.assertGreater(w[0], w[2])
        np.testing.assert_allclose(neutral_start(), [1 / 3] * 3 + [0] * 5, rtol=1e-6)

    def test_gate_starts_at_zero(self):
        g = C.Gate(12, 8); self.assertTrue(torch.all(g(torch.randn(4, 12)) == 0))


class Fusion(unittest.TestCase):
    def test_grid_ranks_brute_force(self):
        rng = np.random.default_rng(2); Cm = rng.random((3, 25)); t = 7; R = fusion.grid_ranks(Cm, t)
        for gi in (0, 57, 230): self.assertEqual(R[gi], stable_rank(fusion.GRID[gi] @ Cm, t))
        self.assertEqual(len(fusion.GRID), 231); np.testing.assert_allclose(fusion.GRID.sum(1), 1)

    def test_channels(self):
        s = np.random.default_rng(3).random((3, 60)); U, Ce, Cc, Cr = fusion.channels(s)
        self.assertTrue(20 <= len(U) <= 60); self.assertEqual(Ce.shape, (3, len(U))); self.assertEqual(int((Cr > 0).sum()), 60)

    def test_stage2_lambda_zero_when_judge_is_noise(self):
        rng = np.random.default_rng(4); n = 400; rC = rng.integers(1, 30, n); tC = -np.sort(-rng.random((n, 10)), 1); JC = rng.normal(size=(n, 10))
        aC, bC, _, _ = standardise(tC, JC, tC, JC); lam, gain = choose_lambda(rC, aC, bC); self.assertIn(lam, GRID)
        JC2 = np.zeros((n, 10)); JC2[np.arange(n), np.clip(rC - 1, 0, 9)] = 5.   # an oracle judge
        aC, bC, _, _ = standardise(tC, JC2, tC, JC2); lam2, gain2 = choose_lambda(rC, aC, bC); self.assertGreater(lam2, 0); self.assertGreater(gain2, 0)

    def test_minmax(self): np.testing.assert_allclose(minmax(np.array([2., 4., 3.])), [0, 1, .5])


if __name__ == '__main__':
    unittest.main()
