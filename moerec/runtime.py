"""Data loading and the expert backbones shared by training and scoring (the paper's Section 3.2 choices)."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); tmp.replace(p)


def seed_all(value=303):
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def configure_torch(gpu=None, fraction=.09, threads=4):
    import torch
    torch.set_num_threads(threads)
    if gpu is None: return torch.device('cpu')
    free, total = torch.cuda.mem_get_info()
    assert free >= fraction * total + 128 * 1024 ** 2, ('not enough free GPU memory for the allocator budget', free)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return torch.device('cuda')


def load_data(exp):
    exp = Path(exp)
    with np.load(exp / 'data/histories.npz', allow_pickle=False) as f: hist = {k: f[k] for k in f.files}
    return hist, json.loads((exp / 'data/users.json').read_text()), json.loads((exp / 'data/item_names.json').read_text())


def seq_at(hist, uid, pos, limit=50):
    """The last `limit` items before position `pos` of user `uid`, left-padded with 0."""
    start = int(hist['offsets'][uid]); past = hist['items'][start:max(start, start + int(pos))]
    out = np.zeros(limit, dtype=np.int64); part = past[-limit:]
    if len(part): out[-len(part):] = part
    return out


def normalize_scores(x):
    """Min-max over the last axis (the candidate pool); a constant row becomes 0.5."""
    x = np.asarray(x, dtype=np.float32); lo = x.min(-1, keepdims=True); hi = x.max(-1, keepdims=True)
    return np.divide(x - lo, hi - lo, out=np.full_like(x, .5), where=(hi - lo) > 0)


def features7(lengths, scores, maxlen=50):
    """The paper's 7 gate context features: history length, 3 pairwise rank agreements, 3 expert confidences."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=np.float32)
    ranked = rankdata(scores, method='average', axis=2); centered = ranked - ranked.mean(-1, keepdims=True)

    def agreement(a, b):
        den = np.linalg.norm(centered[:, a], axis=1) * np.linalg.norm(centered[:, b], axis=1)
        rho = np.divide((centered[:, a] * centered[:, b]).sum(1), den, out=np.zeros(len(scores)), where=den > 0)
        return (np.clip(rho, -1, 1) + 1) / 2
    confid = scores.max(2) - scores.mean(2)
    return np.column_stack([np.minimum(np.asarray(lengths) / maxlen, 1), agreement(0, 1), agreement(0, 2), agreement(1, 2), confid]).astype(np.float32)


def make_sasrec(item_num, device):
    """SASRec of the upstream MoERec code (moerec/sasrec_base.py), scored as h_b . item embedding (no separate classifier)."""
    import torch
    from .sasrec_base import SASRec

    class PaperSASRec(SASRec):
        def __init__(self):
            super().__init__(64, item_num, 50, .1, device, num_heads=2)
            del self.s_fc

        def encode(self, states):
            valid = states.ne(0); mask = valid[:, :, None]   # explicit padding ids (LayerNorm sums are not a padding test)
            x = self.item_embeddings(states) * self.scale + self.positional_embeddings(torch.arange(50, device=states.device))
            q = self.ln_1(self.emb_dropout(x) * mask)
            att = self.mh_attn; B, T, H = q.shape; heads = att.num_heads; d = H // heads
            Q = att.linear_q(q).reshape(B, T, heads, d).transpose(1, 2)
            K = att.linear_k(q).reshape(B, T, heads, d).transpose(1, 2)
            V = att.linear_v(q).reshape(B, T, heads, d).transpose(1, 2)
            score = (Q @ K.transpose(-1, -2)) / math.sqrt(d)
            allow = valid[:, None, None, :] & torch.ones(T, T, device=states.device, dtype=torch.bool).tril()[None, None]
            prob = att.dropout(torch.softmax(score.masked_fill(~allow, -4294967295.), dim=-1)) * valid[:, None, :, None]
            out = (prob @ V).transpose(1, 2).reshape(B, T, H) + q
            return self.ln_3(self.feed_forward(self.ln_2(out)) * mask)[:, -1, :]

        def forward(self, states, lengths=None):
            scores = self.encode(states) @ self.item_embeddings.weight[:self.item_num].T
            scores[:, 0] = -1e9
            return scores

        def candidate_scores(self, states, candidates):
            h = self.encode(states)
            return (h[:, None, :] * self.item_embeddings(candidates)).sum(-1)
    return PaperSASRec().to(device)


def gcn_user_vectors(seqs, raw_items, recency=.1):
    """Paper Eq. 4: recency-weighted mean of the user's LightGCN item embeddings, L2-normalised."""
    import torch
    valid = (seqs > 0)
    w = torch.exp(recency * (torch.arange(seqs.shape[1], device=seqs.device, dtype=torch.float32) - seqs.shape[1]))[None, :] * valid
    h = (raw_items[seqs] * w[:, :, None]).sum(1)
    return torch.nn.functional.normalize(h, dim=1)
