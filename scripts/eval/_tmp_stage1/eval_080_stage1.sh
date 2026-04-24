#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/080_poolAv2_with_1750_teacher_warm031B80_20260421_184131/TeamVsBaselineShapingPPOTrainer_Soccer_4b60d_00000_0_2026-04-21_18-41-53
CKPTS="600 610 620 630 750 760 770 820 830 840 870 880 890 1120 1130 1140 1150 1160"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/080_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60905 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/080_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/080_baseline1000.log
