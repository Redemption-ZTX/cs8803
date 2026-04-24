#!/bin/bash
# 103B BLIND eval — protocol per 103A/C pattern (inline died early, can't trust).
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103B_defender_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_*
TRIAL=$(ls -d $TRIAL | head -1)
CKPTS="100 200 300 400 500"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103B_blind_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 61505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103B_blind_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/103B_blind_baseline1000.log
