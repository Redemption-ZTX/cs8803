#!/bin/bash
# 054 post-train eval: smart 10-ckpt subset, baseline 1000ep
# Selected iters: 460, 720, 910, 930, 940, 970, 1050, 1100, 1230, 1240
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

ARCHIVE_TRIAL="/storage/ice1/5/1/wsun377/ray_results_archive/054_mat_min_scratch_v2_512x512_20260419_185019/TeamVsBaselineShapingPPOTrainer_Soccer_31190_00000_0_2026-04-19_18-50-41"
SCRATCH_TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/054_mat_min_scratch_v2_512x512_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c660d_00000_0_2026-04-20_09-21-01"

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 64005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/054_baseline1000 \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000460/checkpoint-460" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000720/checkpoint-720" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000910/checkpoint-910" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000930/checkpoint-930" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000940/checkpoint-940" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000970/checkpoint-970" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_001050/checkpoint-1050" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001230/checkpoint-1230" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001240/checkpoint-1240" \
  2>&1 | tee docs/experiments/artifacts/official-evals/054_baseline1000.log
exit ${PIPESTATUS[0]}
