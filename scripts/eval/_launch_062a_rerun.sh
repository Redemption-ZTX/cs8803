#!/bin/bash
# 062a Stage 2 rerun: peak ckpt 1220 only, 1000ep different port → combined 2000ep with Stage 1
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_*/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/062a_baseline_rerun1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 3 --base-port 37005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/062a_baseline_rerun1000 \
  --checkpoint "${TRIAL}/checkpoint_001180/checkpoint-1180" \
  --checkpoint "${TRIAL}/checkpoint_001210/checkpoint-1210" \
  --checkpoint "${TRIAL}/checkpoint_001220/checkpoint-1220" \
  2>&1 | tee docs/experiments/artifacts/official-evals/062a_baseline_rerun1000.log
exit ${PIPESTATUS[0]}
