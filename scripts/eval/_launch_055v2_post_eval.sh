#!/bin/bash
# 055v2 (Tier 2: 5-teacher recursive distill + LR=3e-4) post-eval
# Smart 10-ckpt subset: peak window + late
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055v2_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 39005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055v2_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000900/checkpoint-900" \
  --checkpoint "${TRIAL}/checkpoint_000940/checkpoint-940" \
  --checkpoint "${TRIAL}/checkpoint_001000/checkpoint-1000" \
  --checkpoint "${TRIAL}/checkpoint_001040/checkpoint-1040" \
  --checkpoint "${TRIAL}/checkpoint_001140/checkpoint-1140" \
  --checkpoint "${TRIAL}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${TRIAL}/checkpoint_001160/checkpoint-1160" \
  --checkpoint "${TRIAL}/checkpoint_001190/checkpoint-1190" \
  --checkpoint "${TRIAL}/checkpoint_001200/checkpoint-1200" \
  --checkpoint "${TRIAL}/checkpoint_001210/checkpoint-1210" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055v2_baseline1000.log
exit ${PIPESTATUS[0]}
