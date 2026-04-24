#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/081_aggressive_offense_scratch_20260421_184522/TeamVsBaselineShapingPPOTrainer_Soccer_d3c3b_00000_0_2026-04-21_18-45-42
CKPTS="730 740 750 850 860 870 950 960 970 980 1010 1020 1030 1140 1150 1160 1200 1210 1220 1230 1240"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/081_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 61205 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/081_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/081_baseline1000.log
