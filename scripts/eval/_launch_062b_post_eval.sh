#!/bin/bash
# 062b (curriculum + adaptive + no-shape, faster boundaries 0/100/300/800) post-eval
# Smart 10 ckpts from peak (1040/1060) + plateau (920/1090/1190)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/062b_curriculum_noshape_adaptive_*/TeamVsBaselineShaping*/ | tail -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/062b_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 33005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/062b_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000920/checkpoint-920" \
  --checkpoint "${TRIAL}/checkpoint_001040/checkpoint-1040" \
  --checkpoint "${TRIAL}/checkpoint_001050/checkpoint-1050" \
  --checkpoint "${TRIAL}/checkpoint_001060/checkpoint-1060" \
  --checkpoint "${TRIAL}/checkpoint_001070/checkpoint-1070" \
  --checkpoint "${TRIAL}/checkpoint_001090/checkpoint-1090" \
  --checkpoint "${TRIAL}/checkpoint_001120/checkpoint-1120" \
  --checkpoint "${TRIAL}/checkpoint_001180/checkpoint-1180" \
  --checkpoint "${TRIAL}/checkpoint_001190/checkpoint-1190" \
  --checkpoint "${TRIAL}/checkpoint_001200/checkpoint-1200" \
  2>&1 | tee docs/experiments/artifacts/official-evals/062b_baseline1000.log
exit ${PIPESTATUS[0]}
