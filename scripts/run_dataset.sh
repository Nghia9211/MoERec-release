#!/usr/bin/env bash
# Full MoERec pipeline for one prepared dataset, pool 100, one seed. Steps whose output exists are skipped (resume).
#   scripts/run_dataset.sh <dataset> [seed]
# Environment:
#   WORK      work root; the prepared dataset is $WORK/<dataset> (data/, calib.json, document_vectors.npy)
#   LLM_PATH  local Qwen3-8B directory (Stage 2 and the LLM baselines)
#   GPU       GPU index for the expert / LLM steps (default 0)
#   STEPS     subset of: experts stage1 stage2 baselines large ablation (default: all)
# Layout written (seed S, default 303):
#   $WORK/<dataset>/s<S>/main     main experts (all training users)
#   $WORK/<dataset>/s<S>/fold1    fold-1 data + experts (gate training users masked)
#   $WORK/<dataset>/s<S>/fold2    fold-2 data + experts (judge training users masked, disjoint from fold 1)
#   $WORK/runs/s<S>/<dataset>     every Stage 1 / Stage 2 / baseline output of this seed
set -euo pipefail
DS=$1; SEED=${2:-303}; G=${GPU:-0}; STEPS=${STEPS:-"experts stage1 stage2 baselines large ablation"}
D=$WORK/$DS; S=$D/s$SEED; RUN=$WORK/runs/s$SEED/$DS; mkdir -p "$S" "$RUN"
has() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date '+%F %T') [$DS s$SEED] $*"; }
py() { python -m "$@"; }

[ -f "$D/pools50.json" ] || py moerec.data.pools "$D"
if has experts; then
  # main expert dir = links to the prepared data; fold dirs mask a 20% group of training users (seeded independently of SEED)
  mkdir -p "$S/main"; for f in data calib.json pools50.json document_vectors.npy; do [ -e "$S/main/$f" ] || ln -s "$(readlink -f "$D/$f")" "$S/main/$f"; done
  [ -e "$D/fold1/data/gate_holdout.json" ] || py moerec.data.folds "$D" "$D/fold1" --rng 707
  [ -e "$D/fold2/data/gate_holdout.json" ] || py moerec.data.folds "$D" "$D/fold2" --rng 808 --exclude "$D/fold1/data/gate_holdout.json"
  for k in fold1 fold2; do mkdir -p "$S/$k"; for f in data calib.json pools50.json document_vectors.npy; do [ -e "$S/$k/$f" ] || ln -s "$(readlink -f "$D/$k/$f")" "$S/$k/$f"; done; done
  for k in main fold1 fold2; do
    [ -f "$S/$k/sasrec_best.pt" ] || { log "SASRec $k"; py moerec.experts.train sasrec "$S/$k" --gpu "$G" --seed "$SEED"; }
    [ -f "$S/$k/gcn_items.pt" ] || { log "LightGCN $k"; py moerec.experts.train gcn "$S/$k" --gpu "$G" --seed "$SEED"; }
  done
fi
if has stage1; then
  [ -f "$RUN/experts-test.npz" ] || { log "score pools"; py moerec.stage1.pipeline score "$S/main" "$RUN" --pool 100; }
  [ -f "$RUN/fold1.npz" ] || { log "fold-1 rows"; py moerec.stage1.pipeline fold1 "$S/fold1" "$RUN" --pool 100; }
  [ -f "$RUN/j1.pt" ] || { log "J1"; py moerec.stage1.pipeline fit "$S/main" "$S/fold1" "$RUN"; }
  [ -f "$RUN/prompts-test.jsonl.gz" ] || py moerec.stage2.prompts "$S/main" "$RUN"
fi
if has stage2; then
  [ -f "$RUN/judge-pairs.jsonl.gz" ] || { log "judge pairs"; py moerec.stage2.pairs "$S/main" "$S/fold1" "$S/fold2" "$RUN" --pool 100; }
  [ -f "$RUN/lora-judge/adapter_config.json" ] || { log "train judge"; CUDA_VISIBLE_DEVICES=$G py moerec.stage2.train_judge "$DS" "$RUN" --seed $((SEED - 303)); }
  log "score top-10"; CUDA_VISIBLE_DEVICES=$G py moerec.stage2.score_judge "$DS" "$RUN"
  py moerec.stage2.fuse "$RUN"
fi
if has baselines; then
  py moerec.baselines.lime "$S/main" "$S/fold1" "$RUN"
  CUDA_VISIBLE_DEVICES=$G py moerec.baselines.listwise "$DS" "$S/main" "$RUN"
  CUDA_VISIBLE_DEVICES=$G py moerec.baselines.llamarec "$DS" "$S/main" "$RUN"
  CUDA_VISIBLE_DEVICES=$G py moerec.baselines.tallrec "$DS" "$S/main" "$RUN"
fi
if has large; then
  py moerec.eval.large_pools "$S/main" "$RUN"
  for lev in p500 p1000 full; do
    CUDA_VISIBLE_DEVICES=$G py moerec.stage2.score_judge "$DS" "$RUN" --prompts "prompts-$lev-{tag}.jsonl.gz" --out "judge-$lev-{tag}.jsonl"
  done
fi
if has ablation; then py moerec.eval.ablation "$S/main" "$S/fold1" "$RUN"; fi
log DONE
