"""Score the Stage 1 top-10 with the trained judge: s = logit(Yes) - logit(No) per candidate. The history prefix is encoded once per
user (KV cache), the 10 candidate suffixes in batches of 4. Resumable (rows already written are skipped; a truncated last line is dropped).
Usage: python -m moerec.stage2.score_judge <dataset> <run dir> [--prompts prompts-{tag}.jsonl.gz pattern] [--lora <dir>] [--tags calib,test]
(env LLM_PATH; GPU 0)  -> <run dir>/judge-{tag}.jsonl rows {row, ranks, scores}"""
import argparse
import copy
import gzip
import json
import os
import time
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch

from moerec.common import ITEM
from moerec.stage2.prompts import chat_parts, point_suffix


def done_rows(out):
    if not out.exists(): return set()
    lines = out.read_text().splitlines(keepends=True); ok = []
    for l in lines:
        try: json.loads(l); ok.append(l)
        except json.JSONDecodeError: break
    if len(ok) != len(lines): out.write_text(''.join(ok))
    return {json.loads(l)['row'] for l in ok}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('run'); ap.add_argument('--lora'); ap.add_argument('--prompts', default='prompts-{tag}.jsonl.gz')
    ap.add_argument('--out', default='judge-{tag}.jsonl'); ap.add_argument('--tags', default='calib,test'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); DS, run = a.dataset, Path(a.run)
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(os.environ['LLM_PATH'])
    torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN attention fails on busy / split GPUs
    lm = AutoModelForCausalLM.from_pretrained(os.environ['LLM_PATH'], dtype=torch.bfloat16).to('cuda:0')
    lm = PeftModel.from_pretrained(lm, a.lora or str(run / 'lora-judge')).merge_and_unload(); lm.eval()
    ta, tb = tok.encode('Yes', add_special_tokens=False)[0], tok.encode('No', add_special_tokens=False)[0]
    for tag in a.tags.split(','):
        src, out = run / a.prompts.format(tag=tag), run / a.out.format(tag=tag); done = done_rows(out); t0 = time.monotonic(); n_new = 0
        with open(out, 'a') as fo:
            for line in gzip.open(src, 'rt'):
                d = json.loads(line)
                if d['row'] in done: continue
                if a.limit and n_new >= a.limit: break
                pre, post = chat_parts(tok, DS, d['history'])
                sufs = [tok(point_suffix(x, post, ITEM[DS]), add_special_tokens=False)['input_ids'] for x in d['cands']]
                pre_ids = tok(pre, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda:0'); sc = []
                with torch.no_grad():
                    base = DynamicCache(); lm.model(pre_ids, past_key_values=base, use_cache=True)
                    for b in range(0, len(sufs), 4):
                        ch = sufs[b:b + 4]; m = len(ch); L = max(map(len, ch)); x = torch.full((m, L), tok.pad_token_id)
                        for r, s in enumerate(ch): x[r, :len(s)] = torch.tensor(s)
                        cache = copy.deepcopy(base); cache.batch_repeat_interleave(m)
                        mask = torch.cat([torch.ones(m, pre_ids.shape[1], dtype=torch.long),
                                          (torch.arange(L)[None] < torch.tensor([len(s) for s in ch])[:, None]).long()], 1).to('cuda:0')
                        hid = lm.model(x.to('cuda:0'), attention_mask=mask, past_key_values=cache, use_cache=True).last_hidden_state
                        lg = lm.lm_head(hid[torch.arange(m), torch.tensor([len(s) - 1 for s in ch])]).float()
                        sc += (lg[:, ta] - lg[:, tb]).cpu().tolist(); del cache, hid
                    del base
                fo.write(json.dumps({'row': d['row'], 'ranks': d['ranks'], 'scores': sc}) + '\n'); n_new += 1
                if n_new % 50 == 0: fo.flush(); print('JUDGE_SCORE', DS, tag, len(done) + n_new, round(time.monotonic() - t0, 1), flush=True)
        print('JUDGE_SCORE_DONE', DS, tag, len(done) + n_new, round(time.monotonic() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
