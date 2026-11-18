"""The three experts of Stage 1 score a candidate pool.

  SASRec    h_b . item embedding (moerec.runtime.make_sasrec)
  LightGCN  paper Eq. 4 user vector (recency-weighted history of LightGCN item embeddings, L2-normalised) . raw item embedding
  Semantic  recency-weighted mean (exp(0.2 (t - T))) of the MPNet document vectors of the last 50 visible items, L2-normalised,
            . candidate document vector
Scores are min-max normalised over the pool of each user (runtime.normalize_scores). CPU, batches of 256 users.
Expert directory layout: data/ (prepared data), sasrec_best.pt, gcn_items.pt, document_vectors.npy."""
from pathlib import Path

import numpy as np
import torch

from moerec.runtime import gcn_user_vectors, load_data, make_sasrec, normalize_scores, seq_at

SEM_RECENCY, GCN_RECENCY, BATCH = .2, .1, 256


class Experts:
    def __init__(self, exp, threads=8):
        torch.set_num_threads(threads); self.exp = Path(exp); dev = torch.device('cpu')
        self.hist, _, names = load_data(self.exp); N = len(names); self.N = N
        ck = torch.load(self.exp / 'sasrec_best.pt', map_location=dev, weights_only=False)
        self.model = make_sasrec(N, dev); self.model.load_state_dict(ck['model_state_dict']); self.model.eval()
        self.g = torch.load(self.exp / 'gcn_items.pt', map_location=dev, weights_only=False)['item_emb'].float(); assert self.g.shape == (N, 64)
        self.docs = np.load(self.exp / 'document_vectors.npy', mmap_mode='r'); assert self.docs.shape == (N, 768)

    def raw(self, us, poss, pools):
        """(B, 3, P) raw scores of the candidates `pools` for users `us` with visible history length `poss`."""
        seqs = torch.from_numpy(np.stack([seq_at(self.hist, int(u), int(q)) for u, q in zip(us, poss)]))
        cand = torch.from_numpy(np.asarray(pools, dtype=np.int64))
        with torch.no_grad():
            s_seq = self.model.candidate_scores(seqs, cand)
            hu = gcn_user_vectors(seqs, self.g, GCN_RECENCY); s_gcn = (hu[:, None, :] * self.g[cand]).sum(-1)
        d = np.asarray(self.docs[np.asarray(pools).ravel()]).reshape(len(us), -1, 768)
        sq = seqs.numpy(); hv = np.asarray(self.docs[sq.ravel()], dtype=np.float32).reshape(len(us), sq.shape[1], 768)
        w = np.exp(SEM_RECENCY * (np.arange(sq.shape[1]) - sq.shape[1]))[None, :] * (sq > 0); uq = (hv * w[:, :, None]).sum(1)
        uq = uq / (np.linalg.norm(uq, axis=1, keepdims=True) + 1e-8)
        s_sem = (d * uq[:, None, :]).sum(-1)
        return np.stack([s_seq.numpy(), s_gcn.numpy(), s_sem], 1).astype(np.float32)

    def normed(self, us, ns, pools):
        raw = np.concatenate([self.raw(us[i:i + BATCH], ns[i:i + BATCH], pools[i:i + BATCH]) for i in range(0, len(us), BATCH)])
        return normalize_scores(raw).astype(np.float32)
