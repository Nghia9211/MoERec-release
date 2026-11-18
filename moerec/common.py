"""Shared helpers: dataset metadata, ranking metrics, item-text parsing and the evidence shown to the LLMs."""
import json
import re
from pathlib import Path

import numpy as np

DATASETS = ['video_games', 'musical', 'industrial', 'goodreads', 'yelp', 'cell_phones', 'movies_tv']
DATASET_NAMES = {'video_games': 'Video Games', 'musical': 'Musical', 'industrial': 'Industrial', 'goodreads': 'Goodreads', 'yelp': 'Yelp',
                 'cell_phones': 'Cell Phones', 'movies_tv': 'Movies & TV'}
PLATFORM = {'goodreads': 'Goodreads', 'yelp': 'Yelp', 'video_games': 'Amazon Video Games', 'musical': 'Amazon Musical Instruments',
            'industrial': 'Amazon Industrial & Scientific', 'cell_phones': 'Amazon Cell Phones & Accessories', 'movies_tv': 'Amazon Movies & TV'}
ITEM = {'goodreads': 'book', 'yelp': 'business', 'video_games': 'product', 'musical': 'musical instrument or accessory',
        'industrial': 'industrial or scientific product', 'cell_phones': 'cell phone or accessory', 'movies_tv': 'movie or TV title'}
SCENARIOS = ('classic', 'user_cold_start')
NOM = 20          # every expert nominates its own top-20
HIST_SIGNALS = 20  # last 20 visible items feed the collaborative / content signals
LOST = 10 ** 6    # rank given to a target that a truncating method dropped

# ------------------------------------------------------------------ metrics
def ndcg5(r): return np.where(r <= 5, 1 / np.log2(r + 1), 0.)
def ndcg10(r): return np.where(r <= 10, 1 / np.log2(r + 1), 0.)


def rank_of(score, allowed, tpos):
    """1-based rank of the target; items not allowed score -inf (stable order on ties)."""
    s = np.where(allowed, score, -np.inf); order = np.argsort(-s, 1, kind='stable')
    return (order == tpos[:, None]).argmax(1) + 1


def stable_rank(s, t):
    """Rank of column t in a 1-d score vector, ties broken by position (as a stable argsort)."""
    return int((s > s[t]).sum() + (s[:t] == s[t]).sum() + 1)


def topk_stable(s, k):
    kth = np.partition(-s, k - 1)[k - 1]; cand = np.flatnonzero(-s <= kth)
    return cand[np.lexsort((cand, -s[cand]))][:k]


def minmax(x):
    lo, hi = x.min(-1, keepdims=True), x.max(-1, keepdims=True)
    return np.where(hi - lo > 0, (x - lo) / np.where(hi - lo > 0, hi - lo, 1), .5)


def bootstrap_ci(d, rng, B=2000):
    s = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(np.quantile(s, .025)), 4), round(float(np.quantile(s, .975)), 4)]


def rerank_top(r1, scores, k=10):
    """Re-order the Stage 1 top-k of every user by `scores` (n, k) aligned with that top-k; ranks below k are unchanged."""
    new = r1.copy()
    for i in np.flatnonzero(r1 <= k):
        new[i] = list(np.argsort(-scores[i], kind='stable')).index(r1[i] - 1) + 1
    return new


def summarize(r, scen=None):
    e = {'NDCG@5': round(float(ndcg5(r).mean()), 4), 'NDCG@10': round(float(ndcg10(r).mean()), 4)}
    if scen is not None:
        for sc in sorted(set(scen)):
            m = scen == sc; e[sc] = {'NDCG@5': round(float(ndcg5(r[m]).mean()), 4), 'NDCG@10': round(float(ndcg10(r[m]).mean()), 4)}
    return e


# ------------------------------------------------------------------ items and evidence
_RX = {'brand': re.compile(r'Brand:\s*([^|]+)'), 'cat': re.compile(r'(?:Category|Categories|genres):\s*([^|]+)', re.I)}


class Catalog:
    """Item names, texts, brand and category sets of one prepared dataset directory."""

    def __init__(self, exp):
        exp = Path(exp)
        self.texts = json.loads((exp / 'data/item_texts.json').read_text())
        self.names = json.loads((exp / 'data/item_names.json').read_text())
        self.N = len(self.texts)   # item ids 1..N-1, 0 = padding
        fx = {k: [(r.search(x).group(1).strip().lower() if r.search(x) else '') for x in self.texts] for k, r in _RX.items()}
        self.brand = fx['brand']; self.cats = [set(c.split(', ')) - {''} for c in fx['cat']]


def body(text):
    """Item text without the title repeated three times at its start (the semantic-index boost)."""
    p = text.split(' | ')
    return ' | '.join(p[3:]) if len(p) > 3 and p[0] == p[1] == p[2] else text


def rate(v): return f'rated {int(v)}/5' if v > 0 else 'not rated'


class Evidence:
    """History block and candidate cards shared by the judge, the listwise baseline and the prompt files."""

    def __init__(self, cat, items, off, stars, docs):
        self.c, self.items, self.off, self.stars, self.docs = cat, items, off, stars, docs

    def user(self, u, n=None):
        a = int(self.off[u]); b = int(self.off[u + 1]) if n is None else a + int(n)
        pos = np.arange(a, b)[::-1][:50]; hi = self.items[pos]
        history = '\n'.join(f'- "{self.c.names[int(self.items[p])]}" ({rate(self.stars[p])}): {body(self.c.texts[int(self.items[p])])[:150]}' for p in pos)
        return history, pos, hi, np.asarray(self.docs[hi], dtype=np.float32)

    def card(self, x, pos, hi, hv):
        c = self.c
        if len(hi) == 0: return {'name': c.names[x][:150], 'desc': body(c.texts[x])[:800], 'near': None, 'near_rate': '', 'near_cos': 0.0}
        sim = hv @ np.asarray(self.docs[x], dtype=np.float32); m = int(np.argmax(sim))
        return {'name': c.names[x][:150], 'desc': body(c.texts[x])[:800], 'near': c.names[int(hi[m])][:150], 'near_rate': rate(self.stars[pos[m]]),
                'near_cos': round(float(sim[m]), 3)}
