#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/072_poolC_cross_axis_distill_warm031B80_20260421_080015/TeamVsBaselineShapingPPOTrainer_Soccer_b5561_00000_0_2026-04-21_08-00-36
CKPTS="930 940 950 960 970 980 1000 1010 1020 1060 1070 1080 1170 1180 1190 1200 1210 1220 1230"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60205 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/072_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/072_baseline1000.log
