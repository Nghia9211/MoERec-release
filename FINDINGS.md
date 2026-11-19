# Findings from the follow-up experiments

This note explains the experiments run after the main pool-100 study ([RESULTS.md](RESULTS.md)). They test *why* the pipeline works and whether its parts can be replaced by known methods. Unless stated otherwise: pool 100, NDCG@5 on the full test sets, every choice made on the calibration users, `*` = paired bootstrap (2,000) 95% CI of the difference excludes 0.

**Status.** These are follow-up analyses, not part of the locked pipeline. Several designs (J2, the held-out listwise judges) were proposed after test results of the main study had been seen; they still need confirmation on seeds 304 / 305 before any claim. Code: `moerec/baselines/fwls.py`, `moerec/stage1/pairgate.py`, `moerec/eval/cost.py`; the LLM-side scripts are not in the package yet.

## 1. The Stage 1 gate is feature-weighted stacking; what matters is how it is learned

J1 gives each user one weight vector over the 8 channels, w(u) = w + MLP(x_u). Feature-Weighted Linear Stacking (FWLS; Sill et al., 2009) has the same form with a *linear* function of the user features. We fit FWLS on the same channels, features and fold-1 users in two ways.

| Model | Gate | Training objective | NDCG@5 (mean) | vs J1 |
|---|---|---|---|---|
| J1 | MLP | listwise softmax, held-out fold 1 | 0.4920 |  |
| FWLS (original estimator) | linear | least squares on 0/1 relevance | 0.4728 | J1 better on 7/7 |
| FWLS (listwise) | linear | same as J1 | 0.4916 | no significant difference on 7/7 |

**Reading.** The MLP adds nothing: a linear gate trained the way J1 is trained gives the same accuracy. The original FWLS loses because least squares on 0/1 labels is a poor ranking objective. The useful ingredients are the listwise objective and learning on held-out users, not the gate architecture. A paper should present the gate as feature-weighted stacking.

## 2. Why learning the fusion on seen users collapses

Learning the same J1 on users whose answers the experts were trained on ("in-sample") drops test NDCG@5 from 0.4923 to 0.4063 (worse on 7/7 datasets). We measured the cause.

**Channels look far better on the learning rows than they will at test** (NDCG@5 of each channel alone, mean over 7 datasets):

| Channel | In-sample rows | Held-out fold-1 rows | Test |
|---|---|---|---|
| SASRec | 0.621 | 0.443 | 0.446 |
| LightGCN | 0.636 | 0.422 | 0.424 |
| Co-occurrence (sum) | 0.907 | 0.339 | 0.339 |

The co-occurrence channel almost "knows" the answer in-sample because the interaction graph contains the very interaction to be predicted. The fusion learned in-sample puts 0.65 on co-occurrence (sum) and −0.34 on co-occurrence (max) on average, i.e. it learns to exploit the leak.

**Dose-response.** Replacing held-out rows by in-sample rows (same total) lowers test accuracy monotonically on every dataset: 0.4922 → 0.4811 → 0.4683 → 0.4506 → 0.4063 (0%, 25%, 50%, 75%, 100% in-sample).

**Attribution.** Collapse with all 8 channels: 0.0858; without the two co-occurrence channels: 0.0233 (about 73% of the collapse is the graph leak); with the 3 experts only: 0.0164 (the experts trained on the answer). Both are known forms of target leakage (out-of-fold stacking, Wolpert 1992 and Breiman 1996; target links in GNN link prediction, SpotTarget, WSDM 2023; target statistics, CatBoost, NeurIPS 2018). The contribution is measuring them in multi-expert recommender fusion; the fix is the two held-out folds.

## 3. Pair-level fusion weights (J2) and LambdaMART

J2 lets the fusion weights depend on the (user, item) pair: w(u, x) = w + MLP(x_u, g_ux), where g_ux are 6 item-level features (the item's rank under each expert, how many experts nominate it, and how much the experts disagree about it). Everything else is J1. LambdaMART (LightGBM lambdarank) receives the same 26 per-candidate features and the same fold-1 users.

| Dataset | J1 | J2 (vs J1) | LambdaMART (vs J1) |
|---|---|---|---|
| Video Games | 0.5526 | 0.5571 (+0.0045*) | 0.5535 (+0.0010) |
| Musical Instr. | 0.4298 | 0.4334 (+0.0036*) | 0.4257 (−0.0041) |
| Industrial & Sci. | 0.3738 | 0.3742 (+0.0004) | 0.3646 (−0.0092*) |
| Goodreads | 0.6491 | 0.6682 (+0.0192*) | 0.6649 (+0.0158*) |
| Yelp | 0.5509 | 0.5506 (−0.0003) | 0.5443 (−0.0066*) |
| Cell Phones | 0.4000 | 0.4087 (+0.0087*) | 0.4078 (+0.0078*) |
| Movies & TV | 0.4879 | 0.4927 (+0.0048*) | 0.4907 (+0.0028) |
| **Mean** | 0.4920 | 0.4978 | 0.4931 |

**Reading.** J2 is significantly better than J1 on 5/7 datasets and worse on 0, and higher than LambdaMART on 7/7. LambdaMART with the same features only matches J1, so the gain is not just the extra features. This is the first architectural change that helps Stage 1. Caveats: designed after seeing test results (needs seeds 304 / 305), J2 vs LambdaMART not yet tested for significance, and UMRE (arXiv:2508.07613) also learns per user-item fusion weights, so novelty must be checked against it. Stage 2 has not been re-run on top of J2.

## 4. LLM rerankers: what helps and what does not

**Run alone, every LLM reranker is worse than Stage 1.** Re-ordering the J1 top-k by the LLM score alone: judge 0.3306, LlamaRec 0.3791, TALLRec 0.3910, listwise zero-shot 0.4077 vs J1 0.4920. Run independently on all 100 candidates (no Stage 1) they fall further (see RESULTS.md section 5). The value of an LLM comes from fusing its score with a strong Stage 1.

**Fused with Stage 1, fine-tuned LLMs are on par.** With the same rule z(S1) + λ·z(LLM): pointwise judge 0.5066, LlamaRec 0.5070.

**Why LlamaRec alone beats the judge alone.** Not because it sees more candidates (restricting it to the top-10 makes it better, not worse) and not position bias (shuffling the candidate order in the prompt leaves it unchanged: 0.3811 vs 0.3791). It follows the J1 ranking closely: it picks J1's first item as its first choice about twice as often as the judge. The listwise format (comparing the candidates in one prompt) is the main difference.

**Held-out listwise judges.** LJ = the LlamaRec model trained on the judge's data (fold-2 held-out users, J1 top-10 as hard negatives, shuffled order); RJ = the same with the Stage 1 score inside the training loss (a·z(S1) + λ·LLM, a and λ learned).

| Dataset | Stage 1 + judge (current) | LlamaRec + Stage 1 | LJ + Stage 1 | RJ (`*` / `†` = better / worse than current) |
|---|---|---|---|---|
| Video Games | 0.5628 | 0.5643 | 0.5681 | 0.5709* |
| Musical Instr. | 0.4309 | 0.4331 | 0.4319 | 0.4358* |
| Industrial & Sci. | 0.3721 | 0.3799 | 0.3819 | 0.3804* |
| Goodreads | 0.6861 | 0.6712 | 0.6736 | 0.6714† |
| Yelp | 0.5575 | 0.5527 | 0.5532 | 0.5561 |
| Cell Phones | 0.4274 | 0.4366 | 0.4419 | 0.4434* |
| Movies & TV | 0.5094 | 0.5111 | 0.5113 | 0.5076 |
| **Mean** | 0.5066 | 0.5070 | 0.5089 | 0.5094 |

RJ is better than the current pipeline on 4/7 datasets and worse on 1 (Goodreads), better than LlamaRec + Stage 1 on 2/7 and never worse; the learned λ gives the same accuracy as λ tuned on calibration. RJ is architecturally LlamaRec; its edge comes from the held-out, hard-negative training data, while putting the Stage 1 score into the loss adds little (RJ 0.5094 vs LJ 0.5089). It is also about 4 times cheaper than the pointwise judge.

## 5. Cost

Stage 1 costs 2.7 ms per user on CPU at pool 100 and 22.1 ms on the full catalogue (where the original gate features, computed over every candidate, dominate). LLM steps cost 0.10 s (LlamaRec) to 2.13 s (listwise generation) per user on a shared H100. Only the "LLM fused with Stage 1" variants lie on the upper quality-cost frontier, and LlamaRec + Stage 1 matches the current judge at about a quarter of its cost.

## 6. What this means for the paper

- The strongest claims are empirical: fusing several experts and collaborative signals before truncation beats tuned rank fusion, stacking, LIME-Rec and LambdaMART; learning the fusion on held-out users is essential, and we show why (target leakage, mostly through the co-occurrence graph).
- Architectural novelty is weak where it was claimed (MLP gate = feature-weighted stacking; LLM judge ≈ LlamaRec). J2 (pair-level weights) is the one architectural change that helps and needs confirmation.
- LLM reranking adds +0.015 NDCG@5 on average only when fused with a strong Stage 1; its benefit is bounded by the recall of the candidate set (cf. The Recall Ceiling of LLM Recommendation Reranking, CIKM 2026).
