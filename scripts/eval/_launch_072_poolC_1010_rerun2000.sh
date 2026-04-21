#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/072_poolC_1010_rerun2000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 2000 -j 7 --base-port 52705 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/072_poolC_1010_rerun2000 \
  --checkpoint "/storage/ice1/5/1/wsun377/ray_results_scratch/072_poolC_cross_axis_distill_warm031B80_20260421_080015/TeamVsBaselineShapingPPOTrainer_Soccer_b5561_00000_0_2026-04-21_08-00-36/checkpoint_001010/checkpoint-1010" \
  2>&1 | tee docs/experiments/artifacts/official-evals/072_poolC_1010_rerun2000.log
exit ${PIPESTATUS[0]}
