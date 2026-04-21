#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/063_T40_rerun500v3
CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/063_T40_resume_370_to_1250_20260421_053932/TeamVsBaselineShapingPPOTrainer_Soccer_0ec47_00000_0_2026-04-21_05-39-57/checkpoint_001060/checkpoint-1060
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 500 -j 1 --base-port 63905 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/063_T40_rerun500v3 \
  --checkpoint "$CKPT" \
  2>&1 | tee docs/experiments/artifacts/official-evals/063_T40_rerun500v3.log
exit ${PIPESTATUS[0]}
