#!/bin/bash
# 055lr3e4 (Tier 1a: 055 distill + LR=3e-4) post-eval
# Smart 10-ckpt subset: top 3 peaks (1030, 1140, 1210) + windows + plateau anchor
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL_GLOB="/storage/ice1/5/1/wsun377/ray_results_scratch/055lr3e4_distill_034e_ensemble_to_031B_scratch_20260420_134647/TeamVsBaselineShaping*"
TRIAL=$(ls -d $TRIAL_GLOB | head -1)

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055lr3e4_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 41005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055lr3e4_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000810/checkpoint-810" \
  --checkpoint "${TRIAL}/checkpoint_001020/checkpoint-1020" \
  --checkpoint "${TRIAL}/checkpoint_001030/checkpoint-1030" \
  --checkpoint "${TRIAL}/checkpoint_001040/checkpoint-1040" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${TRIAL}/checkpoint_001130/checkpoint-1130" \
  --checkpoint "${TRIAL}/checkpoint_001140/checkpoint-1140" \
  --checkpoint "${TRIAL}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${TRIAL}/checkpoint_001210/checkpoint-1210" \
  --checkpoint "${TRIAL}/checkpoint_001220/checkpoint-1220" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055lr3e4_baseline1000.log
exit ${PIPESTATUS[0]}
