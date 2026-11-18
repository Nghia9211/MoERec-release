"""Baseline: TALLRec (Bao et al., RecSys 2023, github.com/SAI990323/TALLRec) ported to our next-item protocol, Qwen3-8B.
TALLRec frames recommendation as binary instruction following: Alpaca prompt, instruction 'Given the user's preference and unpreference,
identify whether the user will like the target {item} by answering "Yes." or "No.".', input 'User Preference: "t1", "t2", ...\nUser Unpreference:
"..."\nWhether the user will like the target {item} "x"?', response 'Yes.' / 'No.'; LoRA r=8, alpha=16, dropout 0.05 on q_proj and v_proj, lr 1e-4,
LM loss on the response tokens only. Preference / unpreference = the last 10 history items split by the user's rating (>= 4 or unrated -> preference,
1-3 -> unpreference), as TALLRec splits by rating.
Port (stated): positives = the training user's next item after a history prefix (last item, plus one random earlier prefix, as moerec.baselines.llamarec),
negatives = one uniform random item not in the prefix per positive (TALLRec's data has like/dislike labels; next-item data has no explicit
dislikes); NEX examples (balanced); one epoch, batch 128 (micro 1 x accumulation 128 / 16 here: see ACC), no early stopping.
Scoring: logit('Yes') - logit('No') at the first response token for each item of the J1 Stage 1 top-10 (stage1-scores-{tag}.npz), written as
{row, ranks 1..10, scores} lines (the judge format, so moerec.stage2.fuse can fuse it with Stage 1).
Usage: python -m moerec.baselines.tallrec <dataset> <dataset dir> <run dir> [--mode build|train|score|all] [--nex 12000]   (env LLM_PATH; GPU 0)
Writes <run dir>/tallrec/: train.jsonl.gz, lora/, scores-<tag>.jsonl."""
import os, json, gzip, time, math, random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from pathlib import Path

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('dataset'); _ap.add_argument('exp'); _ap.add_argument('run'); _ap.add_argument('--mode', default='all')
_ap.add_argument('--nex', type=int, default=12000); _ap.add_argument('--acc', type=int, default=32); _ap.add_argument('--tags', default='calib,test'); _ap.add_argument('--limit', type=int, default=0); _A = _ap.parse_args()
DS = _A.dataset; EXP = Path(_A.exp); RUN = Path(_A.run); OUT = RUN / 'tallrec'; OUT.mkdir(parents=True, exist_ok=True)
MODE = _A.mode; NEX = _A.nex; HL = 10; ACC = _A.acc
ITEM = {'goodreads': 'book', 'yelp': 'business', 'video_games': 'video game product', 'musical': 'musical instrument product',
        'industrial': 'industrial or scientific product', 'cell_phones': 'cell phone product', 'movies_tv': 'movie or TV title'}[DS]
INSTR = f'Given the user\'s preference and unpreference, identify whether the user will like the target {ITEM} by answering "Yes." or "No.".'
TEMPLATE = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'
names = json.loads((EXP / 'data/item_names.json').read_text()); N = len(names)
with np.load(EXP / 'data/histories.npz') as h: items, off = h['items'], h['offsets']
stars = np.load(EXP / 'data/history_stars.npy'); TRAIN = OUT / 'train.jsonl.gz'; LORA = OUT / 'lora'
title = lambda i: (names[int(i)] or 'unknown item')[:80].replace('"', "'")


def prompt(u, n, x):
    a = int(off[u]); pos = list(range(a, a + n))[-HL:]
    like = [title(items[p]) for p in pos if stars[p] == 0 or stars[p] >= 4]; dis = [title(items[p]) for p in pos if 0 < stars[p] < 4]
    inp = ('User Preference: ' + ', '.join(f'"{t}"' for t in like) + '\nUser Unpreference: ' + ', '.join(f'"{t}"' for t in dis) +
           f'\nWhether the user will like the target {ITEM} "{title(x)}"?')
    return TEMPLATE.format(instruction=INSTR, input=inp)


def build():
    users = json.loads((EXP / 'data/users.json').read_text()); split = json.loads((EXP / 'data/split_users.json').read_text())
    u2i = {u: k for k, u in enumerate(users)}; excl = {u2i[u] for u in split['test'] + split['inference'] if u in u2i}
    cand = [u for u in range(len(off) - 1) if u not in excl and off[u + 1] - off[u] >= 2]
    rng = np.random.default_rng(2023); rng.shuffle(cand); rows = []
    for u in cand:
        seq = items[off[u]:off[u + 1]].astype(int).tolist(); ends = [len(seq)] + ([int(rng.integers(2, len(seq)))] if len(seq) > 2 else [])
        for e in ends:
            pre, tgt = seq[:e - 1], seq[e - 1]; seen = set(pre) | {tgt}
            while True:
                j = int(rng.integers(1, N))
                if j not in seen: break
            rows += [{'user': int(u), 'n': e - 1, 'item': tgt, 'label': 1}, {'user': int(u), 'n': e - 1, 'item': j, 'label': 0}]
        if len(rows) >= NEX: break
    with gzip.open(TRAIN, 'wt') as f:
        for r in rows[:NEX]: f.write(json.dumps(r) + '\n')
    print('TALLREC_BUILD', DS, 'examples', min(len(rows), NEX), flush=True)


def load_lm(train):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(os.environ['LLM_PATH']); lm = AutoModelForCausalLM.from_pretrained(os.environ['LLM_PATH'], dtype=torch.bfloat16).to('cuda:0')
    if train: lm.config.use_cache = False; lm.gradient_checkpointing_enable(); lm.enable_input_require_grads()
    return tok, lm


def yn_ids(tok):
    y, n = tok.encode('Yes', add_special_tokens=False), tok.encode('No', add_special_tokens=False); assert len(y) == 1 and len(n) == 1, (y, n); return y[0], n[0]


def train():
    import torch, torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(42); tok, lm = load_lm(True)
    lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']))
    data = [json.loads(l) for l in gzip.open(TRAIN, 'rt')]; random.Random(42).shuffle(data); eos = tok.eos_token_id; Y, Nn = yn_ids(tok)
    total = math.ceil(len(data) / ACC); warm = min(20, max(1, total // 10))
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0., (total - s) / max(1, total - warm)))
    lm.train(); t0 = time.monotonic(); run_l = run_a = 0.; log = []
    for i, d in enumerate(data):
        x = tok(prompt(d['user'], d['n'], d['item']), add_special_tokens=False)['input_ids'][-1500:]
        y = tok(('Yes.' if d['label'] else 'No.'), add_special_tokens=False)['input_ids'] + [eos]
        inp = torch.tensor([x + y], device='cuda:0'); lab = torch.tensor([[-100] * len(x) + y], device='cuda:0')
        lg = lm(input_ids=inp).logits[0, :-1].float(); loss = F.cross_entropy(lg, lab[0, 1:], ignore_index=-100); (loss / ACC).backward(); run_l += loss.item()
        run_a += float((lg[len(x) - 1, Y] > lg[len(x) - 1, Nn]).item() == bool(d['label']))
        if (i + 1) % ACC == 0 or i + 1 == len(data):
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 400 == 0 or i + 1 == len(data):
            k = (i % 400) + 1; rec = {'example': i + 1, 'of': len(data), 'loss': round(run_l / k, 4), 'acc': round(run_a / k, 3), 'sec': round(time.monotonic() - t0)}
            print('TALLREC_TRAIN', DS, json.dumps(rec), flush=True); log.append(rec); run_l = run_a = 0.
    lm.save_pretrained(LORA)
    (OUT / 'lora-training.json').write_text(json.dumps({'dataset': DS, 'examples': len(data), 'minutes': round((time.monotonic() - t0) / 60, 1), 'log': log}, indent=1))
    print('TALLREC_TRAIN_DONE', DS, flush=True)


def score():
    import torch
    from peft import PeftModel
    tok, lm = load_lm(False); lm = PeftModel.from_pretrained(lm, LORA); lm.eval(); Y, Nn = yn_ids(tok); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for tag in _A.tags.split(','):
        z = np.load(RUN / f'stage1-{tag}.npz'); s0, pools, us, ns = z['s0'], z['pools'], z['users'], z['ns']
        out = OUT / f'scores-{tag}.jsonl'; done = set()
        if out.exists():
            for l in open(out):
                try: done.add(json.loads(l)['row'])
                except Exception: pass
        t0 = time.monotonic()
        with open(out, 'a') as f:
            for k in range(min(len(s0), _A.limit) if _A.limit else len(s0)):
                if k in done: continue
                c = pools[k][np.argsort(-s0[k], kind='stable')[:10]]; ps = [prompt(int(us[k]), int(ns[k]), int(x)) for x in c]
                b = tok(ps, add_special_tokens=False, return_tensors='pt', padding=True).to('cuda:0')
                with torch.no_grad(): lg = lm(**b).logits[:, -1].float()
                f.write(json.dumps({'row': k, 'ranks': list(range(1, 11)), 'scores': (lg[:, Y] - lg[:, Nn]).cpu().tolist()}) + '\n')
                if (k + 1) % 200 == 0: f.flush(); print('TALLREC_SCORE', DS, tag, k + 1, len(s0), round(time.monotonic() - t0, 1), flush=True)
    print('TALLREC_DONE', DS, flush=True)


if MODE in ('build', 'all') and not TRAIN.exists(): build()
if MODE in ('train', 'all') and not (LORA / 'adapter_config.json').exists(): train()
if MODE in ('score', 'all'): score()
