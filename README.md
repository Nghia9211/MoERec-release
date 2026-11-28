# MoERec

Code for **MoERec**: multi-expert recommendation in which every learned component is **blind to the answer**. A learned fusion of recommender experts silently breaks when any of its inputs (an expert, the co-occurrence graph, a popularity count) has already seen the interaction it is trained to predict; reusing a split that is held out for one expert only is worse than learning fully in-sample. MoERec therefore learns its two stages on two disjoint groups of held-out users, with every expert and graph signal rebuilt without their answers: **BlindFuse** (Stage 1, fusion of the experts) and **BlindJudge** (Stage 2, an LLM judge on the BlindFuse top-10).

**Stage 1 (BlindFuse).** Three experts score a candidate pool: SASRec, LightGCN and a semantic expert (MPNet). Each expert nominates its own top-20, and the union of these lists is re-scored on 8 channels (the fusion is applied before the list is truncated to the final top-k):

- the 3 expert scores;
- 5 collaborative and content signals: co-occurrence sum and max, popularity, brand share and category overlap.

The channels are mixed with per-user weights `w(u) = w + Δ(x_u)`. A small gate (MLP [32, 16]) produces `Δ` from 12 context features of the user. The global weights `w` and the gate are learned **jointly** on held-out users (fold 1) from a neutral start.

**Stage 2 (BlindJudge).** A pointwise LLM judge (Qwen3-8B + LoRA) scores the Stage 1 top-10. It is trained on a second, disjoint group of held-out users (fold 2). The final score is `z(Stage 1) + λ · z(BlindJudge)`, with `λ` chosen on calibration users.

## Motivation: blindness belongs to the complete input pipeline

Out-of-fold learning is not only a property of the downstream model. Every input to that model must also be computed without the interaction it is asked to predict. This distinction matters in recommendation because a learned component may receive co-occurrence statistics, graph embeddings, popularity features, or ranks from a collaborative-filtering (CF) model in addition to ordinary model predictions. A split held out for one expert does not make these other inputs blind.

The main study establishes the mechanism at Stage 1: learning fusion from inputs that had seen the answer reduced mean test NDCG@5 by 0.0859 across seven datasets, and about 73% of this collapse was attributable to the co-occurrence graph. A follow-up experiment tests the same mechanism in an LLM reranker, reflecting LLM-CF pipelines that train an LLM or alignment module on interactions already used to fit the CF model. `CF-in` supplies training-time ranks from experts that saw the target interaction; `CF-Blind` changes only those training-time ranks to ones produced by fold-2 experts that did not. Both variants use the same Qwen3-8B reranker, rows, labels, candidates, prompts and hyperparameters, and both receive the same production-expert ranks at calibration and test.

| Dataset | CF-in alone | CF-Blind alone | Δ | CF-in + BlindFuse | CF-Blind + BlindFuse | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Goodreads | 0.6147 | 0.6243 | +0.0095 | 0.6481 | 0.6580 | +0.0099 |
| Cell Phones | 0.3483 | 0.4215 | +0.0732 | 0.4050 | 0.4279 | +0.0228 |
| Movies & TV | 0.4632 | 0.4972 | +0.0341 | 0.4890 | 0.5026 | +0.0136 |
| Video Games | 0.5365 | 0.5483 | +0.0118 | 0.5532 | 0.5547 | +0.0016 |
| **Macro mean** | **0.4907** | **0.5228** | **+0.0322** | **0.5238** | **0.5358** | **+0.0120** |

The standalone gain is significant by paired bootstrap on all four datasets. After fusion with BlindFuse, the gain is significant on Goodreads, Cell Phones and Movies & TV, but not on Video Games. This controlled result supports an input-level train-test mismatch; it does not establish that every published LLM-CF system is affected. Full confidence intervals and protocol boundaries are reported in [FINDINGS.md](FINDINGS.md), section 2.

This repository contains code only. No datasets, checkpoints or model weights are included. Summary tables of the results (Stage 1, Stage 2 and all baselines) are in [RESULTS.md](RESULTS.md); follow-up analyses (why the pipeline works, leakage, pair-level weights, LLM reranker variants) are explained in [FINDINGS.md](FINDINGS.md).

## Install

```bash
pip install -r requirements.txt
```

- **Stage 1 and the baselines without an LLM:** run on CPU.
- **Expert training:** runs on CPU or GPU.
- **BlindJudge and LLM baselines:** need one GPU with about 24 GB free (bf16 Qwen3-8B). They also need a local copy of the model and of `sentence-transformers/all-mpnet-base-v2`.

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
bash scripts/run_dataset.sh industrial 304      # seed variance: experts, BlindFuse and BlindJudge retrained with seed 304
python -m moerec.eval.report $WORK/runs/s303    # main table, all datasets
python -m moerec.eval.large_report $WORK/runs/s303
python -m moerec.eval.seeds $WORK/runs/s303 $WORK/runs/s304 $WORK/runs/s305
```

`run_dataset.sh` runs these steps; each one is skipped when its output exists:

1. **Experts:** main, fold 1, fold 2.
2. **Stage 1:** score pools, fold-1 rows, BlindFuse.
3. **Stage 2:** BlindJudge training pairs from fold 2, LoRA training, top-10 scoring, fusion.
4. **Baselines:** LIME-Rec, listwise zero-shot LLM, LlamaRec, TALLRec.
5. **Large pools:** pool 500, pool 1000 and full catalogue, scored with the same BlindJudge.
6. **Stage 1 ablation.**

`STEPS="stage1 stage2"` selects a subset of these steps.

| Module | Role |
|---|---|
| `moerec/experts` | Expert training (`train`), document vectors (`encode`), pool scoring (`score`) |
| `moerec/stage1` | Nomination, channels and gate inputs (`channels`), BlindFuse learning (`j1`; the code keeps the earlier name J1), CLI (`pipeline`) |
| `moerec/stage2` | BlindJudge prompts, fold-2 training pairs, LoRA training, scoring, fusion (`fuse`) |
| `moerec/baselines` | Early / late score fusion, LIME-Rec, Feature-Weighted Linear Stacking (`fwls`), UMRE (`umre`), listwise LLM, LlamaRec, TALLRec |
| `moerec/eval` | Main report, larger pools, ablation, seed table, inference cost (`cost`), gate-learning protocols (`holdout_protocols`), whole-catalogue baselines and table (`full_baselines`, `full_report`, `full_top20`) |

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
  - expert scores, BlindFuse, BlindJudge prompts and training pairs;
  - Stage 2 fusion;
  - LIME-Rec, the Stage 1 ablation and the pool-500 outputs (with the same thread count);
  - every NDCG@5 in the main table.

  On GPU, BlindJudge, LlamaRec, TALLRec and listwise scores are identical. The only difference is one NDCG@10 cell (Late RRF, Industrial, 0.3761 on Linux vs 0.3762 on Windows): exact ties in the RRF scores break differently with the platform's float summation.
