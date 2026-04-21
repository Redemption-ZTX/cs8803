#!/bin/bash
# 062c Stage 2 rerun: peak ckpt 1090 + 1100 + 1010 (1000ep different port)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/062c_curriculum_noshape_adaptive_*/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/062c_baseline_rerun1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 3 --base-port 35005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/062c_baseline_rerun1000 \
  --checkpoint "${TRIAL}/checkpoint_001010/checkpoint-1010" \
  --checkpoint "${TRIAL}/checkpoint_001090/checkpoint-1090" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  2>&1 | tee docs/experiments/artifacts/official-evals/062c_baseline_rerun1000.log
exit ${PIPESTATUS[0]}
