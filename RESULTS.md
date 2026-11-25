# Results

All numbers come from the archived runs of the pool-100 study (seed 303 unless stated), regenerated from their JSON outputs. Metric: NDCG@5 on the full test set of each dataset (users: Video Games 3,764, Musical Instruments 2,833, Industrial & Scientific 1,419, Goodreads 2,376, Yelp 4,709, Cell Phones 3,548, Movies & TV 3,421). Each user ranks 100 candidates (the 20-item benchmark pool, 30 + 50 uniform negatives). Every tuning choice (Stage 1 penalty, fusion weights, the Stage 2 weight λ) is made on 1,000 calibration users; the test set is read once. `*` = paired bootstrap (2,000 resamples) 95% CI of the stated difference excludes 0.

Stage 1 = BlindFuse (3 experts + 5 signals, global weights and per-user gate learned jointly on held-out fold 1). Stage 2 = BlindJudge (pointwise Qwen3-8B LoRA judge) trained on held-out fold 2, fused as z(S1) + λ·z(BlindJudge) on the Stage 1 top-10.

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
| BlindFuse without gate | 0.5458 | 0.4304 | 0.3689 | 0.6403 | 0.5481 | 0.3976 | 0.4820 | 0.4876 |
| **MoERec Stage 1 (BlindFuse)** | 0.5526* | 0.4298 | 0.3738 | 0.6491* | 0.5509* | 0.4000* | 0.4879* | 0.4920 |
| Listwise zero-shot LLM (Qwen3-8B) | 0.4640* | 0.3461* | 0.3095* | 0.5715* | 0.3894* | 0.3289* | 0.4446* | 0.4077 |
| BlindJudge alone (top-10 of BlindFuse) | 0.3901* | 0.1905* | 0.1557* | 0.5960* | 0.2734* | 0.3044* | 0.4040* | 0.3306 |
| LlamaRec alone (top-20 of BlindFuse) | 0.4530* | 0.2979* | 0.2730* | 0.5221* | 0.3115* | 0.3523* | 0.4441* | 0.3791 |
| TALLRec alone (top-10 of BlindFuse) | 0.4585* | 0.3125* | 0.2781* | 0.5414* | 0.3645* | 0.3494* | 0.4324* | 0.3910 |
| LlamaRec + Stage 1 (same λ rule) | 0.5643 | 0.4331 | 0.3799† | 0.6712* | 0.5527* | 0.4366† | 0.5111 | 0.5070 |
| TALLRec + Stage 1 (same λ rule) | 0.5590 | 0.4298 | 0.3756 | 0.6653* | 0.5510* | 0.4294 | 0.5009* | 0.5016 |
| **MoERec Stage 1 + 2** | 0.5628 | 0.4309 | 0.3721 | 0.6861 | 0.5575 | 0.4274 | 0.5094 | 0.5066 |

`*` / `†` on a baseline: Stage 1 + 2 is significantly better / worse (not computed for the LIME-Rec and no-gate rows). NDCG@10 (mean): BlindFuse 0.5270, Stage 1 + 2 0.5375, CombSUM 0.5143, LlamaRec + Stage 1 0.5374.

| Stage 2 over BlindFuse | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV |
|---|---|---|---|---|---|---|---|
| Δ NDCG@5 | +0.0103* | +0.0011 | −0.0017 | +0.0370* | +0.0066* | +0.0274* | +0.0215* |
| λ (calibration) | 0.35 | 0.05 | 0.20 | 1.05 | 0.20 | 0.65 | 0.45 |

BlindFuse vs LIME-Rec (trained on the same fold-1 users): BlindFuse better on 6/7 datasets.

## 2. By scenario (pool 100)

Same runs and test users as Section 1, split by the benchmark scenario of each test user: **classic** (regular history) and **cold start** (user cold start, short history). Users per dataset are in the first row; Mean = unweighted mean over the 7 datasets. Significance tests are reported for all users only (Section 1). LIME-Rec is not split by scenario in its archived output and is omitted here.

### 2a. Classic

| Method | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV | Mean |
|---|---|---|---|---|---|---|---|---|
| *Users* | 1,239 | 988 | 304 | 1,332 | 1,927 | 1,006 | 1,200 | 7,996 |
| SASRec | 0.5403 | 0.3878 | 0.3593 | 0.5423 | 0.5367 | 0.3705 | 0.3908 | 0.4468 |
| LightGCN | 0.4599 | 0.3564 | 0.3528 | 0.5607 | 0.4781 | 0.3087 | 0.3927 | 0.4156 |
| Semantic (MPNet) | 0.2332 | 0.1679 | 0.1631 | 0.2836 | 0.1614 | 0.1747 | 0.2498 | 0.2048 |
| Early fusion, 3 experts (weights tuned) | 0.5587 | 0.4012 | 0.3910 | 0.6119 | 0.5437 | 0.4039 | 0.4573 | 0.4811 |
| Late CombSUM (tuned) | 0.5536 | 0.4069 | 0.3760 | 0.6091 | 0.5439 | 0.4068 | 0.4476 | 0.4777 |
| Late RRF (tuned) | 0.5395 | 0.3927 | 0.3779 | 0.5813 | 0.5382 | 0.3802 | 0.4194 | 0.4613 |
| BlindFuse without gate | 0.5641 | 0.4095 | 0.3885 | 0.6326 | 0.5525 | 0.3998 | 0.4634 | 0.4872 |
| **MoERec Stage 1 (BlindFuse)** | 0.5718 | 0.4099 | 0.3903 | 0.6388 | 0.5542 | 0.4067 | 0.4717 | 0.4919 |
| Listwise zero-shot LLM (Qwen3-8B) | 0.4775 | 0.3160 | 0.2909 | 0.5622 | 0.3955 | 0.3339 | 0.4349 | 0.4016 |
| BlindJudge alone (top-10 of BlindFuse) | 0.3974 | 0.1855 | 0.1618 | 0.6080 | 0.2625 | 0.3306 | 0.4173 | 0.3376 |
| LlamaRec alone (top-20 of BlindFuse) | 0.4668 | 0.2828 | 0.2590 | 0.5119 | 0.3235 | 0.3782 | 0.4536 | 0.3823 |
| TALLRec alone (top-10 of BlindFuse) | 0.4662 | 0.3025 | 0.2807 | 0.5341 | 0.3664 | 0.3763 | 0.4255 | 0.3931 |
| LlamaRec + Stage 1 (same λ rule) | 0.5832 | 0.4103 | 0.3956 | 0.6649 | 0.5549 | 0.4499 | 0.5014 | 0.5086 |
| TALLRec + Stage 1 (same λ rule) | 0.5775 | 0.4114 | 0.3803 | 0.6597 | 0.5538 | 0.4391 | 0.4878 | 0.5014 |
| **MoERec Stage 1 + 2** | 0.5783 | 0.4111 | 0.3840 | 0.6873 | 0.5590 | 0.4423 | 0.5005 | 0.5089 |

### 2b. Cold start (short history)

| Method | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV | Mean |
|---|---|---|---|---|---|---|---|---|
| *Users* | 2,525 | 1,845 | 1,115 | 1,044 | 2,782 | 2,542 | 2,221 | 14,074 |
| SASRec | 0.5212 | 0.4046 | 0.3126 | 0.5837 | 0.5367 | 0.3695 | 0.4180 | 0.4495 |
| LightGCN | 0.4738 | 0.4013 | 0.3197 | 0.5784 | 0.5076 | 0.2992 | 0.4317 | 0.4302 |
| Semantic (MPNet) | 0.2286 | 0.1812 | 0.1915 | 0.2987 | 0.1739 | 0.1634 | 0.2488 | 0.2123 |
| Early fusion, 3 experts (weights tuned) | 0.5368 | 0.4359 | 0.3548 | 0.6341 | 0.5443 | 0.3902 | 0.4863 | 0.4832 |
| Late CombSUM (tuned) | 0.5359 | 0.4336 | 0.3536 | 0.6351 | 0.5434 | 0.3927 | 0.4785 | 0.4818 |
| Late RRF (tuned) | 0.5239 | 0.4245 | 0.3297 | 0.5990 | 0.5399 | 0.3737 | 0.4580 | 0.4641 |
| BlindFuse without gate | 0.5368 | 0.4416 | 0.3636 | 0.6501 | 0.5451 | 0.3968 | 0.4920 | 0.4894 |
| **MoERec Stage 1 (BlindFuse)** | 0.5431 | 0.4405 | 0.3693 | 0.6622 | 0.5486 | 0.3974 | 0.4967 | 0.4940 |
| Listwise zero-shot LLM (Qwen3-8B) | 0.4574 | 0.3622 | 0.3146 | 0.5833 | 0.3852 | 0.3269 | 0.4499 | 0.4114 |
| BlindJudge alone (top-10 of BlindFuse) | 0.3866 | 0.1931 | 0.1540 | 0.5808 | 0.2809 | 0.2940 | 0.3967 | 0.3266 |
| LlamaRec alone (top-20 of BlindFuse) | 0.4463 | 0.3060 | 0.2767 | 0.5351 | 0.3032 | 0.3420 | 0.4390 | 0.3783 |
| TALLRec alone (top-10 of BlindFuse) | 0.4547 | 0.3179 | 0.2774 | 0.5508 | 0.3632 | 0.3388 | 0.4361 | 0.3913 |
| LlamaRec + Stage 1 (same λ rule) | 0.5550 | 0.4453 | 0.3756 | 0.6793 | 0.5512 | 0.4313 | 0.5163 | 0.5077 |
| TALLRec + Stage 1 (same λ rule) | 0.5500 | 0.4397 | 0.3744 | 0.6725 | 0.5491 | 0.4255 | 0.5080 | 0.5027 |
| **MoERec Stage 1 + 2** | 0.5553 | 0.4415 | 0.3688 | 0.6845 | 0.5565 | 0.4215 | 0.5142 | 0.5060 |


## 3. Three seeds (303, 304, 305)

Seeds 304 / 305 retrain every seed-dependent part (main, fold-1 and fold-2 experts, BlindFuse, BlindJudge training pairs and LoRA); pools, users and calibration set are fixed.

| Dataset | BlindFuse (mean ± SD) | Stage 1 + 2 (mean ± SD) | Δ Stage 2 (303 / 304 / 305) |
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

Pool 500 / 1000 extend pool 100 with the same random negative stream (nested); full catalogue = every item except the visible history. BlindFuse and BlindJudge are trained at pool 100 and applied unchanged; fusion weights and λ are re-tuned on calibration at each size. (a/b) = datasets where BlindFuse is significantly better / worse.

| Method | Pool 500 | Pool 1000 | Full catalogue |
|---|---|---|---|
| SASRec | 0.2594 (7/0) | 0.1977 (7/0) | 0.0306 (6/0) |
| LightGCN | 0.2345 (7/0) | 0.1745 (7/0) | 0.0193 (7/0) |
| Early fusion | 0.2846 (7/0) | 0.2191 (7/0) | 0.0340 (5/0) |
| Late CombSUM | 0.2807 (7/0) | 0.2147 (7/0) | 0.0316 (6/0) |
| Late RRF | 0.2691 (7/0) | 0.2071 (7/0) | 0.0316 (5/0) |
| BlindFuse without gate | 0.2930 (4/0) | 0.2268 (3/0) | 0.0375 (1/0) |
| BlindFuse | 0.2979 | 0.2319 | 0.0393 |
| Stage 1 + 2 | 0.3077 | 0.2398 | 0.0416 |
| Recall of the nominated set U | 0.683 | 0.574 | 0.142 |

Stage 2 significantly improves BlindFuse on 4/7, 3/7 and 2/7 datasets (pool 500, 1000, full catalogue); never significantly worse. At pool 100 the recall of U is 0.904.

Seeds 304 / 305 at pool 500 and 1000 (retrained parts as in Section 3; full catalogue was run for seed 303 only). (a/b) = datasets where Stage 1 + 2 is significantly better / worse than BlindFuse.

| Pool | Seed | BlindFuse | Stage 1 + 2 | Stage 2 vs BlindFuse |
|---|---|---|---|---|
| 500 | 303 | 0.2979 | 0.3077 | 4/0 |
| 500 | 304 | 0.2975 | 0.3059 | 4/1 |
| 500 | 305 | 0.2979 | 0.3067 | 4/0 |
| 1000 | 303 | 0.2319 | 0.2398 | 3/0 |
| 1000 | 304 | 0.2316 | 0.2394 | 4/0 |
| 1000 | 305 | 0.2319 | 0.2406 | 4/0 |

Mean over seeds: pool 500 BlindFuse 0.2978 → 0.3068 (+0.0090); pool 1000 BlindFuse 0.2318 → 0.2399 (+0.0081). BlindFuse beats every single expert and the early/late fusion baselines on 6–7/7 datasets at every seed. The one significant loss is Yelp at pool 500, seed 304 (−0.0088, 95% CI [−0.0137, −0.0041]).


**Whole catalogue, every Stage 1 method, three seeds.** Mean over 7 datasets × seeds 303 / 304 / 305 (21 cells). LIME-Rec, FWLS (listwise) and LambdaMART are trained at pool 100 exactly as in section 5 and applied unchanged, like BlindFuse (`moerec.eval.full_baselines`, table by `moerec.eval.full_report`). (a/b) = cells where BlindFuse is significantly better / worse (paired bootstrap of NDCG@10).

| Method | NDCG@5 | NDCG@10 | HR@10 | HR@20 | (a/b) of 21 |
|---|---|---|---|---|---|
| SASRec | 0.0300 | 0.0369 | 0.0661 | 0.0950 | 18/0 |
| LightGCN | 0.0191 | 0.0251 | 0.0477 | 0.0715 | 21/0 |
| Semantic | 0.0088 | 0.0111 | 0.0209 | 0.0314 | 21/0 |
| Early fusion | 0.0343 | 0.0418 | 0.0742 | 0.1035 | 16/0 |
| Late CombSUM | 0.0313 | 0.0380 | 0.0672 | 0.0945 | 19/0 |
| Late RRF | 0.0313 | 0.0383 | 0.0679 | 0.0957 | 19/0 |
| LIME-Rec | 0.0321 | 0.0398 | 0.0728 | 0.1061 | 15/0 |
| LambdaMART | 0.0368 | 0.0445 | 0.0770 | 0.1070 | 9/0 |
| FWLS (listwise) | 0.0376 | 0.0456 | 0.0798 | 0.1084 | 4/1 |
| BlindFuse without gate | 0.0374 | 0.0451 | 0.0792 | 0.1079 | 10/0 |
| **BlindFuse** | **0.0389** | **0.0468** | **0.0812** | **0.1103** | |

The standard deviation over seeds of the 7-dataset mean NDCG@10 is at most 0.0011 for every method. The ordering of pool 100 holds on the whole catalogue: BlindFuse is never significantly worse than a rank-fusion or learned-fusion baseline except once against FWLS (listwise), which has the same form with a linear gate.

## 5. Further baselines

**RecRanker** (Luo et al., arXiv:2312.16018; ported: Qwen3-8B LoRA, 3,000 sampled users per dataset, hybrid ranking Eq. 9 with the paper defaults), BlindFuse top-10, 1,000 test users per dataset (calibration 500 for λ); every column on the same users:

| Dataset | BlindFuse | Pointwise only | RecRanker (hybrid) | RecRanker + Stage 1 | Stage 1 + 2 |
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

| Dataset | BlindFuse | LlamaRec on BlindFuse top-20 | LlamaRec independent | TALLRec on BlindFuse top-10 | TALLRec independent |
|---|---|---|---|---|---|
| Video Games | 0.5385 | 0.4448 | 0.3920 | 0.4377 | 0.3345 |
| Musical Instr. | 0.4331 | 0.3170 | 0.2677 | 0.3041 | 0.2116 |
| Industrial & Sci. | 0.3525 | 0.2504 | 0.1982 | 0.2439 | 0.1817 |
| Goodreads | 0.6493 | 0.5167 | 0.4900 | 0.5417 | 0.4410 |
| Yelp | 0.5800 | 0.3087 | 0.1609 | 0.3552 | 0.1149 |
| Cell Phones | 0.3746 | 0.3343 | 0.2751 | 0.3356 | 0.2421 |
| Movies & TV | 0.4449 | 0.4053 | 0.3469 | 0.4006 | 0.3151 |
| **Mean** | 0.4818 | 0.3682 | 0.3044 | 0.3741 | 0.2630 |

BlindFuse is significantly better than both independent rerankers on 7/7 datasets.

**Feature-Weighted Linear Stacking** (Sill et al., arXiv:0911.0460) on the same 8 channels and 12 user features, same fold-1 users. "Original" = ridge least squares on 0/1 relevance (the paper's estimator); "listwise" = same linear form trained with BlindFuse's listwise loss:

| Dataset | BlindFuse | FWLS original | FWLS listwise |
|---|---|---|---|
| Video Games | 0.5526 | 0.5170* | 0.5530 |
| Musical Instr. | 0.4298 | 0.4194* | 0.4295 |
| Industrial & Sci. | 0.3738 | 0.3568* | 0.3705 |
| Goodreads | 0.6491 | 0.6332* | 0.6489 |
| Yelp | 0.5509 | 0.5361* | 0.5499 |
| Cell Phones | 0.4000 | 0.3760* | 0.3985 |
| Movies & TV | 0.4879 | 0.4714* | 0.4910 |
| **Mean** | 0.4920 | 0.4728 | 0.4916 |

`*` BlindFuse significantly better. A linear gate trained like BlindFuse matches it: the gain comes from how the fusion is learned (listwise, out-of-sample, jointly with the global weights), not from the MLP.


**UMRE** (Xu et al., arXiv:2508.07613; no official code, ported in `moerec.baselines.umre`): per-channel monotone transforms (integral of a positive MLP, Clenshaw–Curtis, Q = 16) of the same 8 Stage 1 channels over U, mixed with user weights from cross-attention, trained on the same fold-1 users. The paper's pointwise MSE and a listwise variant were both tried (lr ∈ {1e-3, 3e-4}); the listwise variant was chosen on calibration on every dataset. Seed 303, pool 100:

| Dataset | Video Games | Musical Instr. | Industrial & Sci. | Goodreads | Yelp | Cell Phones | Movies & TV | Mean |
|---|---|---|---|---|---|---|---|---|
| BlindFuse | 0.5526 | 0.4298 | 0.3738 | 0.6491 | 0.5509 | 0.4000 | 0.4879 | 0.4920 |
| UMRE | 0.5507 | 0.4298 | 0.3676 | 0.6536 | 0.5433* | 0.4015 | 0.4855 | 0.4903 |

`*` BlindFuse significantly better. Seeds 304 / 305 are running.

## 6. Stage 1 ablation (re-learned like BlindFuse; mean over 7 datasets)

| Variant | NDCG@5 | BlindFuse better / worse (datasets) |
|---|---|---|
| BlindFuse (re-run reference) | 0.4923 |  |
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

| Candidates | Experts | BlindFuse (nomination, signals, gate) | Stage 1 total | Peak RSS (MB) |
|---|---|---|---|---|
| 100 | 1.14 | 1.56 | 2.70 | 1880 |
| 500 | 1.53 | 1.87 | 3.40 | 1885 |
| 1000 | 2.40 | 1.92 | 4.33 | 1950 |
| Full catalogue | 3.11 | 18.94 | 22.06 | 2660 |

LLM steps (Qwen3-8B bf16, shared H100, (110 − 10 users) / 100, Industrial and Goodreads; absolute values are noisy because the GPUs were shared):

| Method | s / user | Peak GPU memory (GB) |
|---|---|---|
| LlamaRec (BlindFuse top-20) | 0.101 | 15.8 |
| TALLRec (BlindFuse top-10) | 0.289 | 17.0 |
| BlindJudge, Stage 2 (BlindFuse top-10) | 0.399 | 16.4 |
| LlamaRec independent (100 items) | 0.985 | 15.8 |
| RecRanker (Industrial only) | 1.666 | 17.0 |
| Listwise zero-shot | 2.126 | 15.9 |
| TALLRec independent (100 items) | 2.363 | 19.6 |

## 9. Additional analyses (seed 303, pool 100, mean over 7 datasets unless stated)

Generated from the archived scores (CPU) plus retrained Stage 1 variants; no new tuning on test. CombSUM3 / RRF3 below use equal weights, so they differ from the tuned rows of section 1.

**9.1 More metrics (test).**

| Method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| SASRec | 0.3103 | 0.5692 | 0.6803 | 0.4463 | 0.4824 | 0.4331 |
| LightGCN | 0.2822 | 0.5514 | 0.6639 | 0.4236 | 0.4600 | 0.4097 |
| Semantic | 0.1245 | 0.2922 | 0.4103 | 0.2095 | 0.2476 | 0.2192 |
| CombSUM3 (equal weights) | 0.3318 | 0.6022 | 0.7097 | 0.4745 | 0.5094 | 0.4586 |
| RRF3 (equal weights) | 0.2609 | 0.5526 | 0.6964 | 0.4117 | 0.4583 | 0.3975 |
| BlindFuse | 0.3500 | 0.6202 | 0.7279 | 0.4920 | 0.5270 | 0.4755 |
| Stage 1 + 2 | 0.3656 | 0.6329 | 0.7279 | 0.5066 | 0.5375 | 0.4891 |

HR@10 is identical for BlindFuse and Stage 1 + 2 because Stage 2 only reorders BlindFuse top-10.

**9.2 User groups (NDCG@5).** By history length (users pooled over datasets):

| History length | n | SASRec | LightGCN | Semantic | CombSUM3 | BlindFuse | Stage 1 + 2 |
|---|---|---|---|---|---|---|---|
| 1-3 | 4,560 | 0.4498 | 0.4255 | 0.1979 | 0.4722 | 0.4869 | 0.5017 |
| 4-7 | 10,946 | 0.4580 | 0.4311 | 0.2088 | 0.4797 | 0.4969 | 0.5084 |
| 8-15 | 4,169 | 0.4635 | 0.4248 | 0.2058 | 0.4823 | 0.5045 | 0.5207 |
| 16-31 | 1,433 | 0.4854 | 0.4561 | 0.2154 | 0.5197 | 0.5417 | 0.5622 |
| 32+ | 962 | 0.4671 | 0.4406 | 0.2107 | 0.5061 | 0.5342 | 0.5742 |

BlindFuse and Stage 1 + 2 beat every expert in every bin; the Stage 2 gain grows with history length (+0.015 for 1-3 items, +0.040 for 32+). By popularity of the target item (tertiles of training degree within each dataset):

| Target group | SASRec | LightGCN | Semantic | CombSUM3 | BlindFuse | Stage 1 + 2 |
|---|---|---|---|---|---|---|
| head | 0.7495 | 0.7560 | 0.2075 | 0.7625 | 0.7810 | 0.7889 |
| mid | 0.4670 | 0.4133 | 0.2125 | 0.4872 | 0.5025 | 0.5147 |
| tail | 0.1363 | 0.1143 | 0.2088 | 0.1864 | 0.2051 | 0.2282 |

On tail targets the Semantic expert alone (0.2088) is better than BlindFuse on average, and per dataset on Industrial & Sci., Musical Instr. and Cell Phones (tail NDCG@5, Semantic vs BlindFuse: 0.1537 vs 0.1098, 0.1935 vs 0.1667, 0.1808 vs 0.0862). The advantage of fusion is concentrated on head and mid items.

**9.3 Gate weights** (test users; mean of w(u) with the global weight in brackets): SASRec 0.403 (0.395), LightGCN 0.236 (0.252), Semantic 0.166 (0.195), co_sum 0.034, co_max 0.060, popularity 0.023, brand 0.051, category 0.046. Spearman correlation of w(u) with history length: popularity −0.31, co_max +0.21, co_sum +0.17, LightGCN −0.15, Semantic −0.11, SASRec +0.10.

**9.4 Recovery** (target in BlindFuse top-k although no single expert has it in its own top-k). Top-5: 1.30% of users, i.e. 2.2% of BlindFuse hits; top-1: 2.09% / 6.2%; top-10: 0.79% / 1.1%. The reverse case (some expert has the target in its top-5, BlindFuse does not) is 10.3% of users, and the best-expert oracle reaches 0.710 HR@5 against 0.620 for BlindFuse. BlindFuse NDCG@5 by number of experts that nominate the target in their top-20 (share of users): 0 → 0.000 (9.6%), 1 → 0.045 (13.4%), 2 → 0.468 (32.2%), 3 → 0.747 (44.8%). Most of the gain therefore comes from agreement between experts, not from rescuing items that no expert ranks highly.

**9.5 Sensitivity.**

| Nominated per expert (N, BlindFuse retrained, penalty fixed) | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| Test NDCG@5 | 0.4910 | 0.4924 | 0.4920 | 0.4916 | 0.4919 |
| Recall of U | 0.710 | 0.816 | 0.904 | 0.944 | 0.980 |

| BlindJudge depth (Stage 1 top-k reranked) | 3 | 5 | 10 |
|---|---|---|---|
| Test NDCG@5 | 0.4967 | 0.5002 | 0.5066 |

| λ | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|
| Δ NDCG@5 vs BlindFuse, calibration | +0.0091 | +0.0086 | +0.0019 | −0.0081 | −0.0304 | −0.0505 |
| Δ NDCG@5 vs BlindFuse, test (post hoc, not used for selection) | +0.0104 | +0.0096 | +0.0026 | −0.0077 | −0.0319 | −0.0521 |

**9.6 Calibration size for λ** (20 random calibration subsets per size; test Δ of Stage 1 + 2 over BlindFuse): 100 users +0.0111 (λ SD 0.29, 21% of draws negative); 200 users +0.0128; 500 users +0.0141; 1,000 users and the full calibration set +0.0146. (For most datasets 1,000 is the full calibration set.)

**9.7 Where Stage 2 helps or hurts** (Δ NDCG@5 over BlindFuse, share of users helped / hurt):

| Group | Δ | Helped | Hurt |
|---|---|---|---|
| BlindFuse rank of target = 1 | −0.0214 | 0.0% | 5.3% |
| BlindFuse rank 2-3 | +0.0485 | 24.4% | 15.3% |
| BlindFuse rank 4-5 | +0.0023 | 38.7% | 22.5% |
| BlindFuse rank 6-10 | +0.1484 | 29.6% | 0.0% |
| BlindFuse top-1 margin: low / mid / high tercile | +0.0269 / +0.0130 / +0.0038 | 16.1 / 10.5 / 4.9% | 9.6 / 6.5 / 3.6% |
| BlindJudge-S1 correlation: low / mid / high tercile | +0.0207 / +0.0154 / +0.0076 | 13.3 / 11.1 / 7.2% | 8.5 / 6.6 / 4.6% |

Stage 2 helps most when BlindFuse is uncertain; when BlindFuse already ranks the target first it hurts 5.3% of those users.

**9.8 BERT4Rec and the choice of experts.** BERT4Rec (Sun et al., CIKM 2019; hidden 64, 2 layers, 2 heads, max length 50, cloze mask 0.2 plus the last position masked with p = 0.5, full softmax, trained on the same samples as the SASRec expert, early stopping on validation next-item cross-entropy). A first version (default N(0,1) embedding initialisation, one sequence per user) did not learn (validation cross-entropy ≈ ln |items|); it was discarded and the initialisation (N(0, 0.02)) and training samples changed before the reported version. That fix was decided on validation loss, but the test NDCG@5 of the failed version had already been seen. BERT4Rec as a single model (test NDCG@5): VG 0.5324, MI 0.3960, I&S 0.3454, GR 0.5443, Yelp 0.5356, CP 0.3548, M&TV 0.3889, mean 0.4425 (SASRec 0.4463, BlindFuse 0.4920).

BlindFuse re-learned (fold 1, same protocol) with different expert sets; the 3-expert set reproduces BlindFuse exactly:

| Experts in BlindFuse | VG | MI | I&S | GR | Yelp | CP | M&TV | Mean | Recall of U |
|---|---|---|---|---|---|---|---|---|---|
| SASRec + LightGCN | 0.5406 | 0.4250 | 0.3639 | 0.6309 | 0.5522 | 0.3941 | 0.4601 | 0.4810 | |
| SASRec + LightGCN + BERT4Rec | 0.5570 | 0.4323 | 0.3885 | 0.6359 | 0.5646 | 0.4007 | 0.4673 | 0.4923 | |
| SASRec + LightGCN + Semantic (BlindFuse) | 0.5526 | 0.4298 | 0.3738 | 0.6491 | 0.5509 | 0.4000 | 0.4879 | 0.4920 | 0.9036 |
| BERT4Rec + LightGCN + Semantic | 0.5578 | 0.4268 | 0.3980 | 0.6452 | 0.5528 | 0.3992 | 0.4814 | 0.4945 | |
| **SASRec + LightGCN + Semantic + BERT4Rec** | 0.5668 | 0.4371 | 0.3947 | 0.6550 | 0.5631 | 0.4103 | 0.4943 | **0.5030** | 0.9219 |

Four experts vs BlindFuse, paired bootstrap, all 7 datasets significant: VG +0.0142, MI +0.0073, I&S +0.0209, GR +0.0059, Yelp +0.0122, CP +0.0103, M&TV +0.0064. This configuration was designed after seeing results and run once (seed 303); it has no Stage 2, other seeds or calibration-only selection yet.

## Notes

- Results on sampled candidate sets overstate full-catalogue accuracy; section 4 bounds this.
- Experiments still running at the time of writing (not included yet):
  - BlindJudge on the whole catalogue for seeds 304 / 305, and LlamaRec / TALLRec + Stage 1 on the whole catalogue (seed 303; BlindFuse top-20 from `moerec.eval.full_top20`);
  - UMRE for seeds 304 / 305;
  - a global temporal split (`moerec.data.temporal`, `scripts/temporal_split.sh`) on the three Amazon datasets with timestamps (Industrial & Sci., Musical Instr., Video Games): every input uses interactions before a cut-off T (median test-target time), test = benchmark tasks whose target is after T. In the user split used elsewhere, a median 24–30% of the training interactions happen after the test target.
- Reproduction: `scripts/run_dataset.sh` and the modules listed in the README; `tests/reproduce.py` checks a run against archived outputs.
