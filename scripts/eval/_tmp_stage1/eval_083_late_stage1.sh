#!/bin/bash
# 083 LATE window supplementary eval (iter 800-1250) — inline died at iter 790.
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/083_per_ray_attention_scratch_20260421_210849/TeamVsBaselineShapingPPOTrainer_Soccer_decf1_00000_0_2026-04-21_21-09-11
CKPTS="800 850 900 950 1000 1050 1100 1150 1200 1250"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/083_latewindow_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 62405 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/083_latewindow_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/083_latewindow_baseline1000.log
