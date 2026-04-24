#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/079_055v3_recursive_distill_warm031B80_20260421_081705/TeamVsBaselineShapingPPOTrainer_Soccer_0eb3f_00000_0_2026-04-21_08-17-25
CKPTS="700 710 720 770 780 790 980 990 1000 1050 1060 1070 1120 1130 1140 1180 1190 1200 1210"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60405 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/079_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/079_baseline1000.log
