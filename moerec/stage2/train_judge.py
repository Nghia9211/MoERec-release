"""Train the pointwise judge: Qwen3-8B (bf16) + LoRA r 16, alpha 32, dropout 0.05 on all attention and MLP projections.
Every fold-2 pair gives two examples (A and B), label Yes for the answer, No otherwise ('neither' pairs: both No).
Loss = 2-way cross-entropy on the Yes / No logits at the answer position. AdamW lr 1e-4, accumulation 8, 30 warm-up steps then
cosine, one epoch, gradient checkpointing; 300 examples held out for an accuracy check.
Usage: python -m moerec.stage2.train_judge <dataset> <run dir> [--seed 0]   (env LLM_PATH = local Qwen3-8B dir; GPU 0)
-> <run dir>/lora-judge/, lora-judge-training.json"""
import argparse
import gzip
import json
import math
import os
import random
import time
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import torch.nn.functional as F

from moerec.common import ITEM
from moerec.stage2.prompts import chat_parts, point_suffix

ACC, LR, HOLD = 8, 1e-4, 300


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('run'); ap.add_argument('--seed', type=int, default=int(os.environ.get('JUDGE_SEED', '0')))
    ap.add_argument('--limit', type=int, default=0, help='smoke test: cap the number of examples'); a = ap.parse_args(); DS, run, JS = a.dataset, Path(a.run), a.seed
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(os.environ['LLM_PATH'])
    lm = AutoModelForCausalLM.from_pretrained(os.environ['LLM_PATH'], dtype=torch.bfloat16).to('cuda:0')
    lm.config.use_cache = False; lm.gradient_checkpointing_enable(); lm.enable_input_require_grads()
    torch.manual_seed(JS); torch.cuda.manual_seed_all(JS)
    lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, task_type='CAUSAL_LM',
                                       target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    lm.print_trainable_parameters(); inner = lm.get_base_model()
    ty = tok.encode('Yes', add_special_tokens=False); tn = tok.encode('No', add_special_tokens=False); assert len(ty) == len(tn) == 1; ty, tn = ty[0], tn[0]
    pairs = [json.loads(l) for l in gzip.open(run / 'judge-pairs.jsonl.gz', 'rt')]; random.Random(JS).shuffle(pairs)
    data = []
    for d in pairs: lab = d['label']; data += [(d, 'A', int(lab == 'A')), (d, 'B', int(lab == 'B'))]
    random.Random(JS + 1).shuffle(data); data = data[:a.limit or len(data)]
    hold_n = min(HOLD, len(data) // 5); train, hold = data[:-hold_n], data[-hold_n:]

    def logits2(ex):
        d, which, y = ex; pre, post = chat_parts(tok, DS, d['history'])
        ids = tok(pre, add_special_tokens=False)['input_ids'] + tok(point_suffix(d[which], post, ITEM[DS]), add_special_tokens=False)['input_ids']
        h = inner.model(input_ids=torch.tensor([ids], device='cuda:0')).last_hidden_state[:, -1]
        lg = inner.lm_head(h).float()[0]; return torch.stack([lg[tn], lg[ty]]), y   # index 1 = Yes

    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    total = math.ceil(len(train) / ACC); warm = 30
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    lm.train(); t0 = time.monotonic(); run_l = run_a = 0.; log = []
    for i, ex in enumerate(train):
        z, y = logits2(ex); loss = F.cross_entropy(z[None], torch.tensor([y], device='cuda:0'))
        (loss / ACC).backward(); run_l += loss.item(); run_a += float(int(z.argmax()) == y)
        if (i + 1) % ACC == 0 or i + 1 == len(train):
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 400 == 0:
            rec = {'example': i + 1, 'loss': round(run_l / 400, 4), 'acc': round(run_a / 400, 3), 'sec': round(time.monotonic() - t0),
                   'peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            print('JUDGE_TRAIN', DS, json.dumps(rec), flush=True); log.append(rec); run_l = run_a = 0.
    lm.eval(); ok = 0
    with torch.no_grad():
        for ex in hold: z, y = logits2(ex); ok += int(int(z.argmax()) == y)
    lm.save_pretrained(run / 'lora-judge')
    res = {'dataset': DS, 'seed': JS, 'train_examples': len(train), 'heldout_examples': len(hold), 'heldout_acc': ok / len(hold), 'log': log,
           'minutes': round((time.monotonic() - t0) / 60, 1), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
    (run / 'lora-judge-training.json').write_text(json.dumps(res, indent=1)); print('JUDGE_TRAIN_DONE', json.dumps({k: v for k, v in res.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
