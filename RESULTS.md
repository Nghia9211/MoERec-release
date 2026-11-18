# Results

All numbers come from the archived runs of the pool-100 study (seed 303 unless stated), regenerated from their JSON outputs. Metric: NDCG@5 on the full test set of each dataset (users: Video Games 3,764, Musical Instruments 2,833, Industrial & Scientific 1,419, Goodreads 2,376, Yelp 4,709, Cell Phones 3,548, Movies & TV 3,421). Each user ranks 100 candidates (the 20-item benchmark pool, 30 + 50 uniform negatives). Every tuning choice (Stage 1 penalty, fusion weights, the Stage 2 weight λ) is made on 1,000 calibration users; the test set is read once. `*` = paired bootstrap (2,000 resamples) 95% CI of the stated difference excludes 0.

Stage 1 = J1 (3 experts + 5 signals, global weights and per-user gate learned jointly on held-out fold 1). Stage 2 = pointwise Qwen3-8B LoRA judge trained on held-out fold 2, fused as z(S1) + λ·z(judge) on the Stage 1 top-10.

## 1. Main table (pool 100)

| Method | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV | Mean |
|---|---|---|---|---|---|---|---|---|
| SASRec | 0.5275* | 0.3988* | 0.3226* | 0.5605* | 0.5367* | 0.3698* | 0.4085* | 0.4463 |
| LightGCN | 0.4692* | 0.3856* | 0.3268* | 0.5685* | 0.4955* | 0.3019* | 0.4180* | 0.4236 |
| Semantic (MPNet) | 0.2301* | 0.1766* | 0.1854* | 0.2902* | 0.1688* | 0.1666* | 0.2491* | 0.2095 |
| Early fusion, 3 experts (weights tuned) | 0.5440* | 0.4238* | 0.3626* | 0.6216* | 0.5440* | 0.3941* | 0.4761* | 0.4809 |
| Late CombSUM (tuned) | 0.5417* | 0.4243* | 0.3584* | 0.6205* | 0.5436* | 0.3967* | 0.4676* | 0.4790 |
| Late RRF (tuned) | 0.5290* | 0.4134* | 0.3400* | 0.5891* | 0.5392* | 0.3756* | 0.4445* | 0.4615 |
| LIME-Rec gate (trained on fold 1) | 0.5450 | 0.4222 | 0.3646 | 0.6302 | 0.5390 | 0.3924 | 0.4838 | 0.4825 |
| J1 without gate | 0.5458 | 0.4304 | 0.3689 | 0.6403 | 0.5481 | 0.3976 | 0.4820 | 0.4876 |
| **MoERec Stage 1 (J1)** | 0.5526* | 0.4298 | 0.3738 | 0.6491* | 0.5509* | 0.4000* | 0.4879* | 0.4920 |
| Listwise zero-shot LLM (Qwen3-8B) | 0.4640* | 0.3461* | 0.3095* | 0.5715* | 0.3894* | 0.3289* | 0.4446* | 0.4077 |
| Judge alone (top-10 of J1) | 0.3901* | 0.1905* | 0.1557* | 0.5960* | 0.2734* | 0.3044* | 0.4040* | 0.3306 |
| LlamaRec alone (top-20 of J1) | 0.4530* | 0.2979* | 0.2730* | 0.5221* | 0.3115* | 0.3523* | 0.4441* | 0.3791 |
| TALLRec alone (top-10 of J1) | 0.4585* | 0.3125* | 0.2781* | 0.5414* | 0.3645* | 0.3494* | 0.4324* | 0.3910 |
| LlamaRec + Stage 1 (same λ rule) | 0.5643 | 0.4331 | 0.3799† | 0.6712* | 0.5527* | 0.4366† | 0.5111 | 0.5070 |
| TALLRec + Stage 1 (same λ rule) | 0.5590 | 0.4298 | 0.3756 | 0.6653* | 0.5510* | 0.4294 | 0.5009* | 0.5016 |
| **MoERec Stage 1 + 2** | 0.5628 | 0.4309 | 0.3721 | 0.6861 | 0.5575 | 0.4274 | 0.5094 | 0.5066 |

`*` / `†` on a baseline: Stage 1 + 2 is significantly better / worse (not computed for the LIME-Rec and no-gate rows). NDCG@10 (mean): J1 0.5270, Stage 1 + 2 0.5375, CombSUM 0.5143, LlamaRec + Stage 1 0.5374.

| Stage 2 over J1 | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV |
|---|---|---|---|---|---|---|---|
| Δ NDCG@5 | +0.0103* | +0.0011 | −0.0017 | +0.0370* | +0.0066* | +0.0274* | +0.0215* |
| λ (calibration) | 0.35 | 0.05 | 0.20 | 1.05 | 0.20 | 0.65 | 0.45 |

J1 vs LIME-Rec (trained on the same fold-1 users): J1 better on 6/7 datasets.

## 2. By scenario (mean over 7 datasets)

| Method | Classic | Cold start (short history) |
|---|---|---|
| SASRec | 0.4468 | 0.4495 |
| Late CombSUM | 0.4777 | 0.4818 |
| J1 | 0.4919 | 0.4940 |
| LlamaRec + Stage 1 | 0.5086 | 0.5077 |
| Stage 1 + 2 | 0.5089 | 0.5060 |

## 3. Three seeds (303, 304, 305)

Seeds 304 / 305 retrain every seed-dependent part (main, fold-1 and fold-2 experts, J1, judge pairs, LoRA judge); pools, users and calibration set are fixed.

| Dataset | J1 (mean ± SD) | Stage 1 + 2 (mean ± SD) | Δ Stage 2 (303 / 304 / 305) |
|---|---|---|---|
| Video Games | 0.5550 ± 0.0021 | 0.5637 ± 0.0010 | +0.0103* / +0.0086* / +0.0074* |
| Musical Instr. | 0.4275 ± 0.0027 | 0.4280 ± 0.0036 | +0.0011 / −0.0006 / +0.0008 |
| Industrial & Sci. | 0.3714 ± 0.0039 | 0.3714 ± 0.0039 | −0.0017 / +0.0015 / +0.0003 |
| Goodreads | 0.6491 ± 0.0011 | 0.6870 ± 0.0020 | +0.0370* / +0.0392* / +0.0377* |
| Yelp | 0.5509 ± 0.0021 | 0.5535 ± 0.0035 | +0.0066* / −0.0011 / +0.0024 |
| Cell Phones | 0.4014 ± 0.0015 | 0.4254 ± 0.0018 | +0.0274* / +0.0212* / +0.0232* |
| Movies & TV | 0.4877 ± 0.0013 | 0.5062 ± 0.0029 | +0.0215* / +0.0170* / +0.0173* |
| **Mean** | 0.4919 ± 0.0009 | 0.5050 ± 0.0015 | significant in 13/21 |

## 4. Larger candidate sets (mean over 7 datasets)

Pool 500 / 1000 extend pool 100 with the same random negative stream (nested); full catalogue = every item except the visible history. J1 and the judge are trained at pool 100 and applied unchanged; fusion weights and λ are re-tuned on calibration at each size. (a/b) = datasets where J1 is significantly better / worse.

| Method | Pool 500 | Pool 1000 | Full catalogue |
|---|---|---|---|
| SASRec | 0.2594 (7/0) | 0.1977 (7/0) | 0.0306 (6/0) |
| LightGCN | 0.2345 (7/0) | 0.1745 (7/0) | 0.0193 (7/0) |
| Early fusion | 0.2846 (7/0) | 0.2191 (7/0) | 0.0340 (5/0) |
| Late CombSUM | 0.2807 (7/0) | 0.2147 (7/0) | 0.0316 (6/0) |
| Late RRF | 0.2691 (7/0) | 0.2071 (7/0) | 0.0316 (5/0) |
| J1 without gate | 0.2930 (4/0) | 0.2268 (3/0) | 0.0375 (1/0) |
| J1 | 0.2979 | 0.2319 | 0.0393 |
| Stage 1 + 2 | 0.3077 | 0.2398 | 0.0416 |
| Recall of the nominated set U | 0.683 | 0.574 | 0.142 |

Stage 2 significantly improves J1 on 4/7, 3/7 and 2/7 datasets (pool 500, 1000, full catalogue); never significantly worse. At pool 100 the recall of U is 0.904.

## 5. Further baselines

**RecRanker** (Luo et al., arXiv:2312.16018; ported: Qwen3-8B LoRA, 3,000 sampled users per dataset, hybrid ranking Eq. 9 with the paper defaults), J1 top-10, 1,000 test users per dataset (calibration 500 for λ); every column on the same users:

| Dataset | J1 | Pointwise only | RecRanker (hybrid) | RecRanker + Stage 1 | Stage 1 + 2 |
|---|---|---|---|---|---|
| Video Games | 0.5465 | 0.5003 | 0.4284 | 0.5468 | 0.5587 |
| Musical Instr. | 0.4261 | 0.4063 | 0.3395 | 0.4243 | 0.4281 |
| Industrial & Sci. | 0.3628 | 0.3379 | 0.2968 | 0.3628 | 0.3631 |
| Goodreads | 0.6584 | 0.6029 | 0.4987 | 0.6564 | 0.6979 |
| Yelp | 0.5627 | 0.4998 | 0.4523 | 0.5634 | 0.5716 |
| Cell Phones | 0.3709 | 0.3626 | 0.3086 | 0.3683 | 0.4067 |
| Movies & TV | 0.4611 | 0.4450 | 0.3550 | 0.4577 | 0.4826 |
| **Mean** | 0.4841 | 0.4507 | 0.3828 | 0.4828 | 0.5012 |

Stage 1 + 2 is significantly better than RecRanker (hybrid) on 7/7 datasets and than RecRanker + Stage 1 on 5/7 (worse on 0).

**LLM rerankers run independently** (no Stage 1: they rank all 100 candidates; LlamaRec scores 5 random groups of 20, averaged over 2 partitions), 500 test users per dataset:

| Dataset | J1 | LlamaRec on J1 top-20 | LlamaRec independent | TALLRec on J1 top-10 | TALLRec independent |
|---|---|---|---|---|---|
| Video Games | 0.5385 | 0.4448 | 0.3920 | 0.4377 | 0.3345 |
| Musical Instr. | 0.4331 | 0.3170 | 0.2677 | 0.3041 | 0.2116 |
| Industrial & Sci. | 0.3525 | 0.2504 | 0.1982 | 0.2439 | 0.1817 |
| Goodreads | 0.6493 | 0.5167 | 0.4900 | 0.5417 | 0.4410 |
| Yelp | 0.5800 | 0.3087 | 0.1609 | 0.3552 | 0.1149 |
| Cell Phones | 0.3746 | 0.3343 | 0.2751 | 0.3356 | 0.2421 |
| Movies & TV | 0.4449 | 0.4053 | 0.3469 | 0.4006 | 0.3151 |
| **Mean** | 0.4818 | 0.3682 | 0.3044 | 0.3741 | 0.2630 |

J1 is significantly better than both independent rerankers on 7/7 datasets.

**Feature-Weighted Linear Stacking** (Sill et al., arXiv:0911.0460) on the same 8 channels and 12 user features, same fold-1 users. "Original" = ridge least squares on 0/1 relevance (the paper's estimator); "listwise" = same linear form trained with J1's listwise loss:

| Dataset | J1 | FWLS original | FWLS listwise |
|---|---|---|---|
| Video Games | 0.5526 | 0.5170* | 0.5530 |
| Musical Instr. | 0.4298 | 0.4194* | 0.4295 |
| Industrial & Sci. | 0.3738 | 0.3568* | 0.3705 |
| Goodreads | 0.6491 | 0.6332* | 0.6489 |
| Yelp | 0.5509 | 0.5361* | 0.5499 |
| Cell Phones | 0.4000 | 0.3760* | 0.3985 |
| Movies & TV | 0.4879 | 0.4714* | 0.4910 |
| **Mean** | 0.4920 | 0.4728 | 0.4916 |

`*` J1 significantly better. A linear gate trained like J1 matches it: the gain comes from how the fusion is learned (listwise, out-of-sample, jointly with the global weights), not from the MLP.

## 6. Stage 1 ablation (re-learned like J1; mean over 7 datasets)

| Variant | NDCG@5 | J1 better / worse (datasets) |
|---|---|---|
| J1 (re-run reference) | 0.4923 |  |
| Global weights only (no gate) | 0.4877 | 3/0 |
| Gate learned in-sample (users the experts saw) | 0.4063 | 7/0 |
| 50% of the fold-1 rows | 0.4911 | 0/0 |
| No penalty | 0.4921 | 1/0 |
| Gate inputs: 7 original features only | 0.4912 | 1/0 |
| Gate inputs: 5 signal features only | 0.4902 | 2/0 |
| Gate moves the 3 expert weights only | 0.4904 | 1/0 |
| Drop SASRec | 0.4628 | 7/0 |
| Drop LightGCN | 0.4833 | 5/0 |
| Drop Semantic | 0.4823 | 5/0 |
| Drop the 5 signals | 0.4840 | 6/0 |
| No nomination (U = all 100 items) | 0.4921 | 0/0 |

## 7. Why in-sample learning collapses (mean over 7 datasets)

NDCG@5 of single channels on the rows used for learning: in-sample rows (training users, experts trained on the answer, co-occurrence graph containing it) vs fold-1 rows (experts retrained without it) vs test.

| Channel | In-sample rows | Fold-1 rows | Test |
|---|---|---|---|
| SASRec | 0.621 | 0.443 | 0.446 |
| LightGCN | 0.636 | 0.422 | 0.424 |
| Co-occurrence (sum) | 0.907 | 0.339 | 0.339 |

Dose-response (share of in-sample rows in the learning set, same total): 0.4922 → 0.4811 → 0.4683 → 0.4506 → 0.4063 (0% to 100%).

Collapse (test NDCG@5, fold-1 learning minus in-sample learning): all 8 channels 0.0858; 3 experts only 0.0164; without the 2 co-occurrence channels 0.0233. About 73% of the collapse comes from the co-occurrence graph containing the very interaction to predict (a known form of target leakage; cf. out-of-fold stacking, SpotTarget, CatBoost ordered statistics).

## 8. Inference cost

Stage 1 on CPU (Intel Xeon Silver 4510, 8 threads, 300 test users per dataset), ms per user, mean over 7 datasets:

| Candidates | Experts | J1 (nomination, signals, gate) | Stage 1 total | Peak RSS (MB) |
|---|---|---|---|---|
| 100 | 1.14 | 1.56 | 2.70 | 1880 |
| 500 | 1.53 | 1.87 | 3.40 | 1885 |
| 1000 | 2.40 | 1.92 | 4.33 | 1950 |
| Full catalogue | 3.11 | 18.94 | 22.06 | 2660 |

LLM steps (Qwen3-8B bf16, shared H100, (110 − 10 users) / 100, Industrial and Goodreads; absolute values are noisy because the GPUs were shared):

| Method | s / user | Peak GPU memory (GB) |
|---|---|---|
| LlamaRec (J1 top-20) | 0.101 | 15.8 |
| TALLRec (J1 top-10) | 0.289 | 17.0 |
| Judge, Stage 2 (J1 top-10) | 0.399 | 16.4 |
| LlamaRec independent (100 items) | 0.985 | 15.8 |
| RecRanker (Industrial only) | 1.666 | 17.0 |
| Listwise zero-shot | 2.126 | 15.9 |
| TALLRec independent (100 items) | 2.363 | 19.6 |

## Notes

- Results on sampled candidate sets overstate full-catalogue accuracy; section 4 bounds this.
- Experiments still running at the time of writing (not included): held-out listwise judge variants (LlamaRec-style model trained on fold 2, with and without the Stage 1 score in the loss).
- Reproduction: `scripts/run_dataset.sh` and the modules listed in the README; `tests/reproduce.py` checks a run against archived outputs.
