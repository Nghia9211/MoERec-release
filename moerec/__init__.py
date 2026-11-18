"""MoERec: fusion before truncation for multi-expert recommendation, with a held-out LLM judge.

Stage 1 (J1): three experts (SASRec, LightGCN, semantic) each nominate their top-20 of the candidate pool; the union is scored by
8 channels (3 experts + 5 collaborative / content signals) mixed with per-user weights w(u) = w + Delta(x_u), where the global
weights w and the gate Delta are learned jointly on held-out users (fold 1).
Stage 2: a LoRA-tuned LLM judge (trained on a second, disjoint group of held-out users, fold 2) scores the Stage 1 top-10;
final = z(Stage 1) + lambda * z(judge), lambda chosen on calibration users.
"""
__version__ = '1.0.0'
