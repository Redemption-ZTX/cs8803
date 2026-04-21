#!/bin/bash
# 054M (Tier 1b: MAT-medium) post-eval — inline eval was 100% failed (deployment registration missing).
# Pick spread of 10 ckpts (no peak guidance), favor late ckpts where training stabilized.
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/054M_mat_medium_scratch_v2_512x512_*/TeamVsBaselineShaping*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/054M_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 --base-port 36005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/054M_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000300/checkpoint-300" \
  --checkpoint "${TRIAL}/checkpoint_000500/checkpoint-500" \
  --checkpoint "${TRIAL}/checkpoint_000700/checkpoint-700" \
  --checkpoint "${TRIAL}/checkpoint_000900/checkpoint-900" \
  --checkpoint "${TRIAL}/checkpoint_001050/checkpoint-1050" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${TRIAL}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${TRIAL}/checkpoint_001200/checkpoint-1200" \
  --checkpoint "${TRIAL}/checkpoint_001230/checkpoint-1230" \
  --checkpoint "${TRIAL}/checkpoint_001250/checkpoint-1250" \
  2>&1 | tee docs/experiments/artifacts/official-evals/054M_baseline1000.log
exit ${PIPESTATUS[0]}
