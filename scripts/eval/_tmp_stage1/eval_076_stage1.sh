#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/076_wide_student_distill_scratch_20260421_083310/TeamVsBaselineShapingPPOTrainer_Soccer_4ebd4_00000_0_2026-04-21_08-33-32
CKPTS="320 330 340 620 630 640 650 660 670 690 700 710 760 770 780 790 880 890 900 960 970 980 990 1000 1010 1070 1080 1090 1100 1110 1160 1170 1180"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint $TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 60305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/076_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/076_baseline1000.log
