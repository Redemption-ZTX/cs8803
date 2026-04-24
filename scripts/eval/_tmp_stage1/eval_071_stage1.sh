#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/071_poolA_homogeneous_distill_warm031B80_20260421_073426/TeamVsBaselineShapingPPOTrainer_Soccer_199de_00000_0_2026-04-21_07-34-47
CKPTS="460 470 480 670 680 690 700 710 720 830 840 850 860 970 980 990 1020 1030 1040 1100 1110 1120 1140 1150 1160"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60105 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/071_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/071_baseline1000.log
