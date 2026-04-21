#!/bin/bash
# 055v2 Stage 2 rerun: peak ckpts 1000 + 1190 + 1200 (1000ep different port)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055v2_baseline_rerun1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 3 --base-port 34005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055v2_baseline_rerun1000 \
  --checkpoint "${TRIAL}/checkpoint_001000/checkpoint-1000" \
  --checkpoint "${TRIAL}/checkpoint_001190/checkpoint-1190" \
  --checkpoint "${TRIAL}/checkpoint_001200/checkpoint-1200" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055v2_baseline_rerun1000.log
exit ${PIPESTATUS[0]}
