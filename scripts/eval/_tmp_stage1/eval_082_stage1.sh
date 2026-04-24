#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/082_two_stream_siamese_scratch_20260421_202731/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
CKPTS="810 820 830 880 890 900 910 920 1040 1050 1060 1080 1090 1100 1120 1130 1140 1150 1160 1200 1210 1220 1230 1240"
ARGS=""
for c in $CKPTS; do
  ARGS="$ARGS --checkpoint ${TRIAL%/}/checkpoint_$(printf %06d $c)/checkpoint-$c"
done
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/082_baseline1000
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 62505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/082_baseline1000 \
  $ARGS \
  2>&1 | tee docs/experiments/artifacts/official-evals/082_baseline1000.log
