"""Baseline: LlamaRec (Yue et al., PGAI@CIKM 2023, github.com/Yueeeeeeee/LlamaRec) ported to our protocol (Stage 1 top-20 as the candidate pool).
Faithful to the official code (train_ranker.py, dataloader/llm.py, trainer/llm.py, config.py):
  * Alpaca prompt, instruction "Given user history in chronological order, recommend an item from the candidate pool with its index letter.",
    input 'User history: (1) t1 \n (2) t2 ...; \n Candidate pool: (A) c1 \n ... (T) c20', titles cut to 32 tokens, last 20 history items, oldest first.
  * Ranker TRAINING uses training users only: target = last item of a train prefix, candidates = target + 19 uniform random negatives
    (not in the prefix), shuffled. Loss = full-vocabulary LM cross-entropy on the answer letter and EOS (labels on the last two tokens only).
  * LoRA r=8, alpha=32, dropout=0.05 on q_proj and v_proj, lr 1e-4, 1 epoch, warmup 100, batch 12 (micro 1 x accumulation 12).
  * Scoring: raw logits of the letter tokens A..T at the last prompt position (no softmax); candidates = retriever top-20 in rank order (A = top-1).
Deviations (stated in the paper): backbone Qwen3-8B in bf16 instead of Llama-2-7B in 4-bit (same backbone as our judge); the retriever is our
Stage 1 J1 (stage1-{tag}.npz, top-20 of U); one or two prefixes per training user (NEX examples) instead of every prefix; no early stopping
(the official run stops on validation NDCG@10; we train exactly one epoch and keep the last adapter). Users whose target is outside the top-20
count as misses in every metric (same as the rest of our tables), and the top-20 tail keeps the Stage 1 order beyond position 20.
Usage: python -m moerec.baselines.llamarec <dataset> <dataset dir> <run dir> [--mode build|train|score|all] [--nex 12000] [--limit N]
(env LLM_PATH; GPU 0). Writes <run dir>/llamarec/: train.jsonl.gz, lora/, scores-<tag>.npz (pos = Stage 1 order (n, 20), logits (n, 20))."""
import os, json, gzip, time, math, random, traceback
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from pathlib import Path

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('dataset'); _ap.add_argument('exp'); _ap.add_argument('run'); _ap.add_argument('--mode', default='all')
_ap.add_argument('--nex', type=int, default=12000); _ap.add_argument('--limit', type=int, default=0); _ap.add_argument('--tags', default='calib,test'); _A = _ap.parse_args()
DS = _A.dataset; EXP = Path(_A.exp); RUN = Path(_A.run); OUT = RUN / 'llamarec'; OUT.mkdir(parents=True, exist_ok=True)
MODE = _A.mode; NEX = _A.nex; LIMIT = _A.limit
K, HL, TL, MAXLEN = 20, 20, 32, 1536; LETTERS = [chr(ord('A') + i) for i in range(K)]
INSTR = 'Given user history in chronological order, recommend an item from the candidate pool with its index letter.'
TEMPLATE = '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'
names = json.loads((EXP / 'data/item_names.json').read_text()); N = len(names)   # index 0 = padding, item ids index directly
with np.load(EXP / 'data/histories.npz') as h: items, off = h['items'], h['offsets']
TRAIN = OUT / 'train.jsonl.gz'; LORA = OUT / 'lora'


def build():
    users = json.loads((EXP / 'data/users.json').read_text()); split = json.loads((EXP / 'data/split_users.json').read_text())
    u2i = {u: k for k, u in enumerate(users)}; excl = {u2i[u] for u in split['test'] + split['inference'] if u in u2i}
    cand = [u for u in range(len(off) - 1) if u not in excl and off[u + 1] - off[u] >= 2]
    rng = np.random.default_rng(2023); rng.shuffle(cand); rows = []
    for u in cand:
        seq = items[off[u]:off[u + 1]].astype(int).tolist()
        ends = [len(seq)] + ([int(rng.integers(2, len(seq)))] if len(seq) > 2 else [])   # the last prefix, plus one random earlier prefix
        for e in ends:
            pre, tgt = seq[:e - 1], seq[e - 1]; seen = set(pre) | {tgt}; negs = []
            while len(negs) < K - 1:
                j = int(rng.integers(1, N))
                if j not in seen and j not in negs: negs.append(j)
            c = [tgt] + negs; rng.shuffle(c); rows.append({'user': int(u), 'hist': pre[-HL:], 'cands': c, 'label': c.index(tgt)})
        if len(rows) >= NEX: break
    with gzip.open(TRAIN, 'wt') as f:
        for r in rows[:NEX]: f.write(json.dumps(r) + '\n')
    print('LLAMAREC_BUILD', DS, 'examples', min(len(rows), NEX), 'from users', len({r['user'] for r in rows[:NEX]}), 'of', len(cand), flush=True)


class Prompter:
    def __init__(self, tok):
        self.tok = tok; self.cut = {}
        ids = [tok.encode(l, add_special_tokens=False) for l in LETTERS]; assert all(len(i) == 1 for i in ids), ids
        self.letter_ids = [i[0] for i in ids]
        p = self.prompt([1], [1] * K); a = tok(p, add_special_tokens=False)['input_ids']; b = tok(p + 'A', add_special_tokens=False)['input_ids']
        assert b[:len(a)] == a and b[len(a):] == [self.letter_ids[0]], 'letter token after the prompt differs from the verbalizer token'

    def title(self, i):
        if i not in self.cut: self.cut[i] = self.tok.convert_tokens_to_string(self.tok.tokenize(names[i] or 'unknown item')[:TL]).strip()
        return self.cut[i]

    def prompt(self, hist, cands):
        h = ' \n '.join(f'({k + 1}) {self.title(i)}' for k, i in enumerate(hist)); c = ' \n '.join(f'({LETTERS[k]}) {self.title(i)}' for k, i in enumerate(cands))
        return TEMPLATE.format(instruction=INSTR, input=f'User history: {h}; \n Candidate pool: {c}')

    def ids(self, hist, cands):
        x = self.tok(self.prompt(hist, cands), add_special_tokens=False)['input_ids']
        return x[-(MAXLEN - 2):]   # left truncation, as the official tokenizer (truncation_side='left')


def load_lm(train):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(os.environ['LLM_PATH'])
    lm = AutoModelForCausalLM.from_pretrained(os.environ['LLM_PATH'], dtype=torch.bfloat16).to('cuda:0')
    if train: lm.config.use_cache = False; lm.gradient_checkpointing_enable(); lm.enable_input_require_grads()
    return tok, lm


def train():
    import torch, torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(42); tok, lm = load_lm(True); P = Prompter(tok)
    lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=32, lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']))
    lm.print_trainable_parameters()
    data = [json.loads(l) for l in gzip.open(TRAIN, 'rt')]; random.Random(42).shuffle(data)
    if LIMIT: data = data[:LIMIT]
    eos = tok.eos_token_id; ACC = 12; total = math.ceil(len(data) / ACC); warm = min(100, max(1, total // 10))
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0., (total - s) / max(1, total - warm)))   # HF default: linear decay
    lm.train(); t0 = time.monotonic(); run_l = run_a = 0.; log = []
    for i, d in enumerate(data):
        x = P.ids(d['hist'], d['cands']); y = [P.letter_ids[d['label']], eos]
        inp = torch.tensor([x + y], device='cuda:0'); lab = torch.tensor([[-100] * len(x) + y], device='cuda:0')
        lg = lm(input_ids=inp).logits[0, :-1].float(); loss = F.cross_entropy(lg, lab[0, 1:], ignore_index=-100)
        (loss / ACC).backward(); run_l += loss.item()
        run_a += float(int(lg[len(x) - 1, P.letter_ids].argmax()) == d['label'])
        if (i + 1) % ACC == 0 or i + 1 == len(data):
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 240 == 0 or i + 1 == len(data):
            n = (i % 240) + 1; rec = {'example': i + 1, 'of': len(data), 'loss': round(run_l / n, 4), 'acc20': round(run_a / n, 3), 'sec': round(time.monotonic() - t0),
                                      'peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            print('LLAMAREC_TRAIN', DS, json.dumps(rec), flush=True); log.append(rec); run_l = run_a = 0.
    lm.save_pretrained(LORA)
    (OUT / 'lora-training.json').write_text(json.dumps({'dataset': DS, 'examples': len(data), 'minutes': round((time.monotonic() - t0) / 60, 1), 'log': log}, indent=1))
    print('LLAMAREC_TRAIN_DONE', DS, flush=True)


def score():
    import torch
    from peft import PeftModel
    tok, lm = load_lm(False); lm = PeftModel.from_pretrained(lm, LORA); lm.eval(); P = Prompter(tok)
    for tag in _A.tags.split(','):
        z = np.load(RUN / f'stage1-{tag}.npz'); s0, pools, us, ns = z['s0'], z['pools'], z['users'], z['ns']
        n = len(s0) if not LIMIT else min(LIMIT, len(s0)); cands = np.zeros((n, K), np.int64); logit = np.zeros((n, K), np.float32); t0 = time.monotonic()
        for k in range(n):
            order = np.argsort(-s0[k], kind='stable')[:K]; c = pools[k][order]; u = int(us[k])
            hist = items[off[u]:off[u] + int(ns[k])].astype(int).tolist()[-HL:]
            with torch.no_grad():
                lg = lm(input_ids=torch.tensor([P.ids(hist, c.tolist())], device='cuda:0')).logits[0, -1].float()
            cands[k] = order; logit[k] = lg[P.letter_ids].cpu().numpy()
            if (k + 1) % 200 == 0: print('LLAMAREC_SCORE', DS, tag, k + 1, n, round(time.monotonic() - t0, 1), flush=True)
        np.savez(OUT / f'scores-{tag}.npz', pos=cands, logits=logit, sec_per_user=(time.monotonic() - t0) / max(n, 1))
        print('LLAMAREC_SCORED', DS, tag, n, 'sec/user', round((time.monotonic() - t0) / max(n, 1), 3), flush=True)
    print('LLAMAREC_DONE', DS, flush=True)


if __name__ == '__main__':
    try:
        if MODE in ('build', 'all') and not TRAIN.exists(): build()
        if MODE in ('train', 'all') and not (LORA / 'adapter_config.json').exists(): train()
        if MODE in ('score', 'all'): score()
    except BaseException:
        traceback.print_exc(); raise
