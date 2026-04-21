#!/bin/bash
# 055 post-train eval: smart 10-ckpt subset, baseline 1000ep
# Selected iters: 430, 1000, 1070, 1100, 1130, 1150, 1200, 1210, 1230, 1240
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

ARCHIVE_TRIAL="/storage/ice1/5/1/wsun377/ray_results_archive/055_distill_034e_ensemble_to_031B_scratch_20260419_193252/TeamVsBaselineShapingPPOTrainer_Soccer_24fb7_00000_0_2026-04-19_19-33-18"
SCRATCH_TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01"

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 60005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055_baseline1000 \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_000430/checkpoint-430" \
  --checkpoint "${ARCHIVE_TRIAL}/checkpoint_001000/checkpoint-1000" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001070/checkpoint-1070" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001130/checkpoint-1130" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001200/checkpoint-1200" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001210/checkpoint-1210" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001230/checkpoint-1230" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001240/checkpoint-1240" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055_baseline1000.log
exit ${PIPESTATUS[0]}
