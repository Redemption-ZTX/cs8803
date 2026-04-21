#!/bin/bash
# 056D post-train eval: smart 10-ckpt subset, baseline 1000ep
# Selected iters: 630, 710, 720, 730, 770, 1060, 1110, 1140, 1190, 1200
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

ARCHIVE_TRIAL="/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55"
SCRATCH_TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06"

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 62005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/056D_baseline1000 \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000630/checkpoint-630" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000710/checkpoint-710" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000720/checkpoint-720" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000730/checkpoint-730" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000770/checkpoint-770" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_001060/checkpoint-1060" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001110/checkpoint-1110" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001140/checkpoint-1140" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001190/checkpoint-1190" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001200/checkpoint-1200" \
  2>&1 | tee docs/experiments/artifacts/official-evals/056D_baseline1000.log
exit ${PIPESTATUS[0]}
