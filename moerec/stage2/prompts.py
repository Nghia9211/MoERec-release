"""Judge prompts and the per-user evidence files.

Pointwise judge (Qwen3-8B + LoRA): the user's last 50 history items (most recent first, with the user's own rating and 150 characters
of text, at most 2,500 tokens) and ONE candidate (name, 800-character description, most similar history item by MPNet cosine);
answer Yes / No; score = logit(Yes) - logit(No). Thinking is disabled.

  python -m moerec.stage2.prompts <dataset dir> <run dir>
writes <run dir>/prompts-{calib,test}.jsonl.gz: one line per user {row, history, ranks 1..10, cands} for the Stage 1 top-10."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from moerec.common import ITEM, PLATFORM, Catalog, Evidence

MARK = '<<FINALISTS>>'
POINT_HEAD = """You are an expert recommender for {item}s on {platform}. Decide whether this user will interact with the candidate {item} shown after the history.
Look beyond surface similarity: compatibility or matching specifications with items the user already has, complementary accessories or supplies, brand or author loyalty, series continuation, repeat or replacement purchases, and the user's own ratings (low ratings mean dislike).

User's history (most recent first, with the user's rating):
{history}

""" + MARK


def point_suffix(f, post, item):
    near = (f'"{f["near"]}" ({f["near_rate"]}, similarity {f["near_cos"]:.2f})' if f['near'] else 'none (the user has no history)')
    return (f'Candidate: "{f["name"]}"\nDescription: {f["desc"]}\nMost similar item in the user\'s history: {near}\n\n'
            f'Will the user interact with this {item}? Answer Yes or No.' + post)


def chat_parts(tok, ds, history):
    """Tokenizer chat template split around the candidate block: (prefix shared by all candidates, generation-prompt suffix)."""
    hist = tok.decode(tok(history or '(no history)', add_special_tokens=False)['input_ids'][:2500])
    full = tok.apply_chat_template([{'role': 'user', 'content': POINT_HEAD.format(item=ITEM[ds], platform=PLATFORM[ds], history=hist)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return full.split(MARK)


def evidence(exp):
    exp = Path(exp); cat = Catalog(exp)
    with np.load(exp / 'data/histories.npz') as h: items, off = h['items'], h['offsets']
    return Evidence(cat, items, off, np.load(exp / 'data/history_stars.npy'), np.load(exp / 'document_vectors.npy', mmap_mode='r'))


def write_top10(ev, users, top10, path):
    with gzip.open(path, 'wt') as f:
        for k in range(len(users)):
            history, pos, hi, hv = ev.user(int(users[k]))
            f.write(json.dumps({'row': int(k), 'history': history, 'ranks': list(range(1, 11)), 'cands': [ev.card(int(x), pos, hi, hv) for x in top10[k]]}) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); ap.add_argument('run'); a = ap.parse_args(); run = Path(a.run); ev = evidence(a.exp)
    for tag in ('calib', 'test'):
        z = np.load(run / f'stage1-{tag}.npz'); order = np.argsort(-z['s0'], 1, kind='stable')
        write_top10(ev, z['users'], np.take_along_axis(z['pools'], order[:, :10], 1), run / f'prompts-{tag}.jsonl.gz')
        print('PROMPTS', tag, len(order), flush=True)


if __name__ == '__main__':
    main()
