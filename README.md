# MoERec

Code for **MoERec**: multi-expert recommendation that fuses the experts *before* truncating the candidate list, followed by an LLM judge that is trained on held-out users.

**Stage 1 (J1).** Three experts score a candidate pool: SASRec, LightGCN and a semantic expert (MPNet). Each expert nominates its own top-20, and the union of these lists is re-scored on 8 channels:

- the 3 expert scores;
- 5 collaborative and content signals: co-occurrence sum and max, popularity, brand share and category overlap.

The channels are mixed with per-user weights `w(u) = w + Δ(x_u)`. A small gate (MLP [32, 16]) produces `Δ` from 12 context features of the user. The global weights `w` and the gate are learned **jointly** on held-out users (fold 1) from a neutral start.

**Stage 2.** A pointwise judge (Qwen3-8B + LoRA) scores the Stage 1 top-10. It is trained on a second, disjoint group of held-out users (fold 2). The final score is `z(Stage 1) + λ · z(judge)`, with `λ` chosen on calibration users.

This repository contains code only. No datasets, checkpoints, model weights or results are included.

## Install

```bash
pip install -r requirements.txt
```

- **Stage 1 and the baselines without an LLM:** run on CPU.
- **Expert training:** runs on CPU or GPU.
- **Judge and LLM baselines:** need one GPU with about 24 GB free (bf16 Qwen3-8B). They also need a local copy of the model and of `sentence-transformers/all-mpnet-base-v2`.

## Data

The pipeline uses the AgentRecBench datasets (Amazon Video Games, Musical Instruments, Industrial & Scientific, Goodreads, Yelp) and two subsets of Amazon Reviews 2023 (Cell Phones & Accessories, Movies & TV).

| Step | Command |
|---|---|
| AgentRecBench datasets | `python -m moerec.data.prepare_arb <dataset> --raw <AgentRecBench Dataset dir> --out $WORK` |
| Ratings per history position | `python -m moerec.data.stars <dataset> --raw ... --data $WORK/<dataset>/data` |
| Goodreads | `python -m moerec.data.prepare_goodreads --upstream <MoERec upstream repo> --out $WORK` (then `stars` with `--docstore` for the item texts) |
| Amazon Reviews 2023 | `RAW_DIR=<raw files> python -m moerec.data.prepare_amazon2023 cell_phones $WORK` |
| Calibration users | `python -m moerec.data.calibration $WORK/<dataset>` |
| Item document vectors | `MPNET_PATH=<model dir> python -m moerec.experts.encode $WORK/<dataset>` |

Every user with more than 3 interactions is kept. The users are split as follows:

- **Test users:** the AgentRecBench task users (or generated tasks for the two 2023 datasets), with their target masked.
- **Inference users (5%):** their last item is masked; 1,000 of them form the calibration set.
- **Training users:** all the others. 10% of them provide the SASRec validation item. Two disjoint groups of 20% become fold 1 and fold 2 (`moerec.data.folds`): their last item is masked and the experts are retrained without it.

Candidate pools (`moerec.data.pools`):

- **Pool 50:** the 20-item benchmark pool plus 30 uniform negatives.
- **Pool 100, 500 and 1000:** pool 50 plus a fixed stream of further uniform negatives, so the larger pools are nested.
- **Full catalogue:** every item except the visible history.

## Run

```bash
export WORK=/path/to/work LLM_PATH=/path/to/Qwen3-8B GPU=0
bash scripts/run_dataset.sh industrial          # seed 303
bash scripts/run_dataset.sh industrial 304      # seed variance: experts, J1 and judge retrained with seed 304
python -m moerec.eval.report $WORK/runs/s303    # main table, all datasets
python -m moerec.eval.large_report $WORK/runs/s303
python -m moerec.eval.seeds $WORK/runs/s303 $WORK/runs/s304 $WORK/runs/s305
```

`run_dataset.sh` runs these steps; each one is skipped when its output exists:

1. **Experts:** main, fold 1, fold 2.
2. **Stage 1:** score pools, fold-1 rows, J1.
3. **Stage 2:** judge pairs from fold 2, LoRA training, top-10 scoring, fusion.
4. **Baselines:** LIME-Rec, listwise zero-shot LLM, LlamaRec, TALLRec.
5. **Large pools:** pool 500, pool 1000 and full catalogue, scored with the same judge.
6. **Stage 1 ablation.**

`STEPS="stage1 stage2"` selects a subset of these steps.

| Module | Role |
|---|---|
| `moerec/experts` | Expert training (`train`), document vectors (`encode`), pool scoring (`score`) |
| `moerec/stage1` | Nomination, channels and gate inputs (`channels`), J1 learning (`j1`), CLI (`pipeline`) |
| `moerec/stage2` | Judge prompts, fold-2 training pairs, LoRA training, scoring, fusion (`fuse`) |
| `moerec/baselines` | Early / late score fusion, LIME-Rec, listwise LLM, LlamaRec, TALLRec |
| `moerec/eval` | Main report, larger pools, ablation, seed table |

## Tests

```bash
python -m unittest discover -s tests -v      # unit tests on synthetic data
python tests/reproduce.py <dataset> ...      # regression check against archived outputs (see the docstring)
```

## Notes

- The SASRec backbone (`moerec/sasrec_base.py`) and the item-document builders (`moerec/data/item_text.py`) are copied from the upstream MoERec code.
- Results computed on CPU with float32 can differ in the last digits with a different thread count. `OMP_NUM_THREADS` defaults to 6 for the Stage 1 learning steps.
- The regression check reproduced the archived pool-100 study on all 7 datasets. The following are exactly equal:
  - data, folds, calibration users and pools;
  - expert scores, J1, judge prompts and training pairs;
  - Stage 2 fusion;
  - LIME-Rec, the Stage 1 ablation and the pool-500 outputs (with the same thread count);
  - every NDCG@5 in the main table.

  On GPU, the judge, LlamaRec, TALLRec and listwise scores are identical. The only difference is one NDCG@10 cell (Late RRF, Industrial, 0.3761 on Linux vs 0.3762 on Windows): exact ties in the RRF scores break differently with the platform's float summation.
