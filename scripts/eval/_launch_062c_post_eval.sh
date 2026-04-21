#!/bin/bash
# 062c (curriculum + adaptive + no-shape, slower 0/300/700/1100) post-eval
# Smart 10 ckpts: peak 960 + plateau 690-1100
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/062c_curriculum_noshape_adaptive_*/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/062c_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 38005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/062c_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000690/checkpoint-690" \
  --checkpoint "${TRIAL}/checkpoint_000760/checkpoint-760" \
  --checkpoint "${TRIAL}/checkpoint_000900/checkpoint-900" \
  --checkpoint "${TRIAL}/checkpoint_000950/checkpoint-950" \
  --checkpoint "${TRIAL}/checkpoint_000960/checkpoint-960" \
  --checkpoint "${TRIAL}/checkpoint_000970/checkpoint-970" \
  --checkpoint "${TRIAL}/checkpoint_000990/checkpoint-990" \
  --checkpoint "${TRIAL}/checkpoint_001010/checkpoint-1010" \
  --checkpoint "${TRIAL}/checkpoint_001090/checkpoint-1090" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  2>&1 | tee docs/experiments/artifacts/official-evals/062c_baseline1000.log
exit ${PIPESTATUS[0]}
