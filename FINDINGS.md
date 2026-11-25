# Findings from the follow-up experiments

This note explains the experiments run after the main pool-100 study ([RESULTS.md](RESULTS.md)). They test *why* the pipeline works and whether its parts can be replaced by known methods. Unless stated otherwise: pool 100, NDCG@5 on the full test sets, every choice made on the calibration users, `*` = paired bootstrap (2,000) 95% CI of the difference excludes 0.

**Status.** These are follow-up analyses, not part of the locked pipeline. Several designs (the held-out listwise judges) were proposed after test results of the main study had been seen; they still need confirmation on seeds 304 / 305 before any claim. Code: `moerec/baselines/fwls.py`, `moerec/stage1/pairgate.py`, `moerec/eval/holdout_protocols.py`, `moerec/eval/cost.py`; the LLM-side scripts are not in the package yet.

## 1. The Stage 1 gate is feature-weighted stacking; what matters is how it is learned

BlindFuse gives each user one weight vector over the 8 channels, w(u) = w + MLP(x_u). Feature-Weighted Linear Stacking (FWLS; Sill et al., 2009) has the same form with a *linear* function of the user features. We fit FWLS on the same channels, features and fold-1 users in two ways.

| Model | Gate | Training objective | NDCG@5 (mean) | vs BlindFuse |
|---|---|---|---|---|
| BlindFuse | MLP | listwise softmax, held-out fold 1 | 0.4920 |  |
| FWLS (original estimator) | linear | least squares on 0/1 relevance | 0.4728 | BlindFuse better on 7/7 |
| FWLS (listwise) | linear | same as BlindFuse | 0.4916 | no significant difference on 7/7 |

**Reading.** The MLP adds nothing: a linear gate trained the way BlindFuse is trained gives the same accuracy. The original FWLS loses because least squares on 0/1 labels is a poor ranking objective. The useful ingredients are the listwise objective and learning on held-out users, not the gate architecture. A paper should present the gate as feature-weighted stacking.

## 2. Why learning the fusion on seen users collapses

Learning the same BlindFuse on users whose answers the experts were trained on ("in-sample") drops test NDCG@5 from 0.4923 to 0.4063 (worse on 7/7 datasets). We measured the cause.

**Channels look far better on the learning rows than they will at test** (NDCG@5 of each channel alone, mean over 7 datasets):

| Channel | In-sample rows | Held-out fold-1 rows | Test |
|---|---|---|---|
| SASRec | 0.621 | 0.443 | 0.446 |
| LightGCN | 0.636 | 0.422 | 0.424 |
| Co-occurrence (sum) | 0.907 | 0.339 | 0.339 |

The co-occurrence channel almost "knows" the answer in-sample because the interaction graph contains the very interaction to be predicted. The fusion learned in-sample puts 0.65 on co-occurrence (sum) and −0.34 on co-occurrence (max) on average, i.e. it learns to exploit the leak.

**Dose-response.** Replacing held-out rows by in-sample rows (same total) lowers test accuracy monotonically on every dataset: 0.4922 → 0.4811 → 0.4683 → 0.4506 → 0.4063 (0%, 25%, 50%, 75%, 100% in-sample).

**Attribution.** Collapse with all 8 channels: 0.0858; without the two co-occurrence channels: 0.0233 (about 73% of the collapse is the graph leak); with the 3 experts only: 0.0164 (the experts trained on the answer). Both are known forms of target leakage (out-of-fold stacking, Wolpert 1992 and Breiman 1996; target links in GNN link prediction, SpotTarget, WSDM 2023; target statistics, CatBoost, NeurIPS 2018). The contribution is measuring them in multi-expert recommender fusion; the fix is the two held-out folds.

**Motivation: every signal source must be blind to the answer, not only the model being tuned.** A cheaper protocol than two held-out folds is to learn the fusion on validation items, as LIME-Rec does. Whether this is safe depends on *which* components never saw those items. We compared four protocols with the same BlindFuse recipe (`moerec/eval/holdout_protocols.py`; seed 303; designed after the main results, so read as diagnostic):

- **Reused validation split:** learn on the last item of the SASRec validation users, scored by the main experts. SASRec never trained on these items, but LightGCN and the co-occurrence / popularity graph did. This is what one gets by reusing an existing early-stopping split.
- **Reused split, clean graph:** the same rows, with those user-item edges removed from the co-occurrence / popularity graph. Only LightGCN has still seen the answer.
- **Fold 1, same size:** BlindFuse learned on a random subset of fold 1 with as many rows as the validation split (size control).
- **One expert set:** the locked BlindFuse gate applied to test pools scored by the fold-1 experts, i.e. a single expert set that never saw the held-out items, with no retrained main experts (LIME-Rec style).

| Dataset | BlindFuse (fold 1) | Reused validation split | Reused split, clean graph | Fold 1, same size | One expert set |
|---|---|---|---|---|---|
| Video Games | 0.5526 | 0.4315 (−0.1211*) | 0.5306 (−0.0220*) | 0.5525 (+0.0000) | 0.5560 (+0.0034) |
| Musical Instr. | 0.4298 | 0.3128 (−0.1170*) | 0.3754 (−0.0544*) | 0.4291 (−0.0007) | 0.4259 (−0.0038) |
| Industrial & Sci. | 0.3738 | 0.2714 (−0.1024*) | 0.2960 (−0.0778*) | 0.3689 (−0.0049*) | 0.3678 (−0.0060) |
| Goodreads | 0.6491 | 0.5763 (−0.0728*) | 0.6005 (−0.0486*) | 0.6471 (−0.0019) | 0.6474 (−0.0017) |
| Yelp | 0.5509 | 0.4621 (−0.0888*) | 0.5389 (−0.0120*) | 0.5506 (−0.0003) | 0.5492 (−0.0017) |
| Cell Phones | 0.4000 | 0.2926 (−0.1074*) | 0.2661 (−0.1339*) | 0.3977 (−0.0023) | 0.3994 (−0.0006) |
| Movies & TV | 0.4879 | 0.3621 (−0.1258*) | 0.3991 (−0.0888*) | 0.4880 (+0.0001) | 0.4877 (−0.0002) |
| **Mean** | 0.4920 | 0.3870 | 0.4295 | 0.4906 | 0.4905 |

Reusing the validation split collapses *further* than learning fully in-sample (0.3870 vs 0.4063, worse on 7/7 datasets). The learned fusion puts 0.52 to 1.16 on co-occurrence (sum): one leaking channel is enough for the fusion to latch onto it. Cleaning the graph recovers only part of the loss (0.4295, still worse on 7/7), because the LightGCN embeddings were trained on the answer edge. Size is not the issue (0.4906 with as few rows). A single expert set that never saw the held-out items matches BlindFuse (0.4905, no significant difference on any dataset), so the protocol LIME-Rec follows is sound and the extra main-expert training can be dropped.

The requirement is therefore stronger than "learn the fusion out-of-fold": every expert *and* every collaborative signal must be computed without the held-out interaction. A split that is held out for one expert only, such as the early-stopping split of SASRec, gives the worst of the protocols we tested. The same trap is reported for LLM rerankers trained on users that overlap with the evaluation users (The Recall Ceiling of LLM Recommendation Reranking, CIKM 2026).

## 3. LambdaMART on the same Stage 1 inputs

LambdaMART (LightGBM lambdarank) is trained on the same fold-1 users and candidates (U) as BlindFuse, with 26 per-candidate features: the 8 channels, 6 item-level features (the item's rank under each expert, how many experts nominate it, and how much the experts disagree about it) and the 12 user features of the gate.

| Dataset | BlindFuse | LambdaMART (vs BlindFuse) |
|---|---|---|
| Video Games | 0.5526 | 0.5535 (+0.0010) |
| Musical Instr. | 0.4298 | 0.4257 (−0.0041) |
| Industrial & Sci. | 0.3738 | 0.3646 (−0.0092*) |
| Goodreads | 0.6491 | 0.6649 (+0.0158*) |
| Yelp | 0.5509 | 0.5443 (−0.0066*) |
| Cell Phones | 0.4000 | 0.4078 (+0.0078*) |
| Movies & TV | 0.4879 | 0.4907 (+0.0028) |
| **Mean** | 0.4920 | 0.4931 |

**Reading.** A standard learning-to-rank model with more features only matches BlindFuse on average (0.4931 vs 0.4920): significantly better on 2/7 datasets (Goodreads, Cell Phones) and worse on 2/7 (Industrial, Yelp).

## 4. LLM rerankers: what helps and what does not

**Run alone, every LLM reranker is worse than Stage 1.** Re-ordering BlindFuse top-k by the LLM score alone: BlindJudge 0.3306, LlamaRec 0.3791, TALLRec 0.3910, listwise zero-shot 0.4077 vs BlindFuse 0.4920. Run independently on all 100 candidates (no Stage 1) they fall further (see RESULTS.md section 5). The value of an LLM comes from fusing its score with a strong Stage 1.

**Fused with Stage 1, fine-tuned LLMs are on par.** With the same rule z(S1) + λ·z(LLM): pointwise BlindJudge 0.5066, LlamaRec 0.5070.

**Why LlamaRec alone beats BlindJudge alone.** Not because it sees more candidates (restricting it to the top-10 makes it better, not worse) and not position bias (shuffling the candidate order in the prompt leaves it unchanged: 0.3811 vs 0.3791). It follows BlindFuse ranking closely: it picks BlindFuse's first item as its first choice about twice as often as BlindJudge. The listwise format (comparing the candidates in one prompt) is the main difference.

**Held-out listwise judges.** LJ = the LlamaRec model trained on the BlindJudge's data (fold-2 held-out users, BlindFuse top-10 as hard negatives, shuffled order); RJ = the same with the Stage 1 score inside the training loss (a·z(S1) + λ·LLM, a and λ learned).

| Dataset | Stage 1 + BlindJudge (current) | LlamaRec + Stage 1 | LJ + Stage 1 | RJ (`*` / `†` = better / worse than current) |
|---|---|---|---|---|
| Video Games | 0.5628 | 0.5643 | 0.5681 | 0.5709* |
| Musical Instr. | 0.4309 | 0.4331 | 0.4319 | 0.4358* |
| Industrial & Sci. | 0.3721 | 0.3799 | 0.3819 | 0.3804* |
| Goodreads | 0.6861 | 0.6712 | 0.6736 | 0.6714† |
| Yelp | 0.5575 | 0.5527 | 0.5532 | 0.5561 |
| Cell Phones | 0.4274 | 0.4366 | 0.4419 | 0.4434* |
| Movies & TV | 0.5094 | 0.5111 | 0.5113 | 0.5076 |
| **Mean** | 0.5066 | 0.5070 | 0.5089 | 0.5094 |

RJ is better than the current pipeline on 4/7 datasets and worse on 1 (Goodreads), better than LlamaRec + Stage 1 on 2/7 and never worse; the learned λ gives the same accuracy as λ tuned on calibration. RJ is architecturally LlamaRec; its edge comes from the held-out, hard-negative training data, while putting the Stage 1 score into the loss adds little (RJ 0.5094 vs LJ 0.5089). It is also about 4 times cheaper than the pointwise BlindJudge.

## 5. Cost

Stage 1 costs 2.7 ms per user on CPU at pool 100 and 22.1 ms on the full catalogue (where the original gate features, computed over every candidate, dominate). LLM steps cost 0.10 s (LlamaRec) to 2.13 s (listwise generation) per user on a shared H100. Only the "LLM fused with Stage 1" variants lie on the upper quality-cost frontier, and LlamaRec + Stage 1 matches the current BlindJudge at about a quarter of its cost.

## 6. What this means for the paper

- The strongest claims are empirical: fusing several experts and collaborative signals before truncation beats tuned rank fusion, stacking, LIME-Rec and LambdaMART; learning the fusion on held-out users is essential, and we show why (target leakage, mostly through the co-occurrence graph).
- Architectural novelty is weak where it was claimed (MLP gate = feature-weighted stacking; BlindJudge ≈ LlamaRec).- LLM reranking adds +0.015 NDCG@5 on average only when fused with a strong Stage 1; its benefit is bounded by the recall of the candidate set (cf. The Recall Ceiling of LLM Recommendation Reranking, CIKM 2026).
