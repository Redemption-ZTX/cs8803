#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04
CKPTS="190 200 210 220 230 250 260 270 320 330 340"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/101A_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 61005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/101A_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/101A_baseline1000.log
