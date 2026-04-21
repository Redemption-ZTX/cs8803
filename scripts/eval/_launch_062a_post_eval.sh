#!/bin/bash
# 062a (curriculum + adaptive + no-shape, baseline boundaries 0/200/500/1000) post-eval
# Smart 10-ckpt subset from peak/plateau
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_*/TeamVsBaselineShaping*/ | head -1)

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/062a_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 40005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/062a_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000680/checkpoint-680" \
  --checkpoint "${TRIAL}/checkpoint_000890/checkpoint-890" \
  --checkpoint "${TRIAL}/checkpoint_001080/checkpoint-1080" \
  --checkpoint "${TRIAL}/checkpoint_001090/checkpoint-1090" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${TRIAL}/checkpoint_001110/checkpoint-1110" \
  --checkpoint "${TRIAL}/checkpoint_001130/checkpoint-1130" \
  --checkpoint "${TRIAL}/checkpoint_001180/checkpoint-1180" \
  --checkpoint "${TRIAL}/checkpoint_001210/checkpoint-1210" \
  --checkpoint "${TRIAL}/checkpoint_001220/checkpoint-1220" \
  2>&1 | tee docs/experiments/artifacts/official-evals/062a_baseline1000.log
exit ${PIPESTATUS[0]}
