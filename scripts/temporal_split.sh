#!/usr/bin/env bash
# Global temporal split check for one Amazon dataset (CPU). One expert set: the fold-1 experts also score the test users
# (FINDINGS.md section 2 shows this matches the two-set protocol). Stage 1, LIME-Rec, FWLS and the Stage 1 ablation.
#   RAW_TSV=<user, item, ts, stars tsv.gz> scripts/temporal_split.sh <dataset>     (env WORK as run_dataset.sh)
set -euo pipefail
DS=$1; SRC=$WORK/$DS; T=$WORK/temporal/$DS; F=$T/fold1; RUN=$T/run; mkdir -p "$RUN"
py() { python -m "$@"; }
[ -f "$T/temporal-info.json" ] || py moerec.data.temporal "$SRC" "$RAW_TSV" "$T"
[ -e "$F/data/gate_holdout.json" ] || py moerec.data.folds "$T" "$F" --rng 707
[ -f "$F/sasrec_best.pt" ] || py moerec.experts.train sasrec "$F" --cpu --seed 303
[ -f "$F/gcn_items.pt" ] || py moerec.experts.train gcn "$F" --cpu --seed 303
[ -f "$RUN/experts-test.npz" ] || py moerec.stage1.pipeline score "$F" "$RUN" --pool 100
[ -f "$RUN/fold1.npz" ] || py moerec.stage1.pipeline fold1 "$F" "$RUN" --pool 100
[ -f "$RUN/j1.pt" ] || py moerec.stage1.pipeline fit "$F" "$F" "$RUN"
py moerec.baselines.lime "$F" "$F" "$RUN"; py moerec.baselines.fwls "$F" "$F" "$RUN"; py moerec.eval.ablation "$F" "$F" "$RUN"
