#!/bin/bash
# 055 baseline rerun v2 — brings rerun total to 1000ep (matching 051A precedent)
# Ckpts 1130/1150/1200, port 58005
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

SCRATCH_TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01"

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055_baseline_rerun500_v2

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 500 \
  -j 3 \
  --base-port 58005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055_baseline_rerun500_v2 \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001130/checkpoint-1130" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${SCRATCH_TRIAL}/checkpoint_001200/checkpoint-1200" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055_baseline_rerun500_v2.log
exit ${PIPESTATUS[0]}
