"""Zero-shot LISTWISE LLM rerank (RankGPT-style) of the Stage 1 top-10: Qwen3-8B, thinking off, greedy decoding.
One prompt per user: the history block of the judge (last 50 items, ratings, 150-character text, 2,500-token cap) and the 10 candidates
in Stage 1 order as [1]..[10] with name and a 400-character description (no Stage 1 scores, no evidence). The model returns
"[i] > [j] > ..."; unparsable or missing ids are appended in Stage 1 order (`parsed` = ids that came from the model). Resumable.
Usage: python -m moerec.baselines.listwise <dataset> <dataset dir> <run dir> [--tags test]   (env LLM_PATH; GPU 0)
-> <run dir>/listwise-{tag}.jsonl rows {row, cm, perm (1-based Stage 1 positions, best first), parsed, raw}"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch

from moerec.common import ITEM, PLATFORM, body, rate
from moerec.runtime import load_data

PROMPT = """You are an expert recommender for {item}s on {platform}. Rank the 10 candidate {item}s below by how likely this user is to interact with them next.

User's history (most recent first, with the user's rating):
{history}

Candidates:
{cands}

Rank all 10 candidates, most likely first. Answer only with the identifiers in order, for example: [3] > [1] > [10] > ... (all 10, no explanation)."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('exp'); ap.add_argument('run'); ap.add_argument('--tags', default='test'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    DS, E, run = a.dataset, Path(a.exp), Path(a.run)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(os.environ['LLM_PATH'])
    lm = AutoModelForCausalLM.from_pretrained(os.environ['LLM_PATH'], dtype=torch.bfloat16).to('cuda:0'); lm.eval()
    hist, users, names = load_data(E); off, items = hist['offsets'], hist['items']
    stars = np.load(E / 'data/history_stars.npy'); texts = json.loads((E / 'data/item_texts.json').read_text())
    for tag in a.tags.split(','):
        z = np.load(run / f'stage1-{tag}.npz'); s0, pools, us = z['s0'], z['pools'], z['users']
        out = run / f'listwise-{tag}.jsonl'; done = set()
        if out.exists():
            for l in open(out):
                try: done.add(json.loads(l)['row'])
                except Exception: pass
        t0 = time.monotonic(); n_new = 0; bad = 0
        with open(out, 'a') as f:
            for k in range(min(len(s0), a.limit) if a.limit else len(s0)):
                if k in done: continue
                ids = pools[k][np.argsort(-s0[k], kind='stable')[:10]]; u = int(us[k]); pos = np.arange(int(off[u]), int(off[u + 1]))[::-1][:50]
                history = '\n'.join(f'- "{names[int(items[p])]}" ({rate(stars[p])}): {body(texts[int(items[p])])[:150]}' for p in pos)
                history = tok.decode(tok(history, add_special_tokens=False)['input_ids'][:2500])
                cands = '\n'.join(f'[{j + 1}] "{names[int(c)][:120]}" - {body(texts[int(c)])[:400]}' for j, c in enumerate(ids))
                msg = tok.apply_chat_template([{'role': 'user', 'content': PROMPT.format(item=ITEM[DS], platform=PLATFORM[DS], history=history, cands=cands)}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
                x = tok(msg, add_special_tokens=False, return_tensors='pt').to('cuda:0')
                with torch.no_grad():
                    g = lm.generate(**x, max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                raw = tok.decode(g[0, x['input_ids'].shape[1]:], skip_special_tokens=True); perm = []
                for m in re.findall(r'\[(\d+)\]', raw):
                    v = int(m)
                    if 1 <= v <= 10 and v not in perm: perm.append(v)
                parsed = len(perm); bad += parsed < 10; perm += [v for v in range(1, 11) if v not in perm]
                f.write(json.dumps({'row': k, 'cm': ids.tolist(), 'perm': perm, 'parsed': parsed, 'raw': raw[:200]}) + '\n'); n_new += 1
                if n_new % 50 == 0: f.flush(); print('LISTWISE', DS, tag, len(done) + n_new, len(s0), 'incomplete', bad, round(time.monotonic() - t0, 1), flush=True)
    print('LISTWISE_DONE', DS, flush=True)


if __name__ == '__main__':
    main()
