#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/073_poolD_resume920_to_1250_20260421_211734/TeamVsBaselineShapingPPOTrainer_Soccer_18687_00000_0_2026-04-21_21-17-57
CKPTS="1050 1060 1070 1080 1090 1100 1180 1190 1200 1210 1220 1230"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/073_resume_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/073_resume_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/073_resume_baseline1000.log
