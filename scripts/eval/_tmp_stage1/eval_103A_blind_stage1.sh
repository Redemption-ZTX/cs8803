#!/bin/bash
# 103A BLIND eval — inline died at iter 120, late ckpts 130-500 unmeasured.
# Use thin spaced sampling (iter 100/200/300/400/500) to confirm lane status
# without wasting GPU on 38-ckpt full eval (training appears fundamentally
# broken per inline 0-0.03 vs baseline + 0.14-0.20 vs random).
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_*
TRIAL=$(ls -d $TRIAL | head -1)
CKPTS="100 200 300 400 500"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_blind_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 61305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_blind_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/103A_blind_baseline1000.log
