#!/bin/bash
# 053Dmirror post-train eval (PBRS-only warm from 031B@80)
# Smart 10-ckpt subset: peak 680 ± 1, secondary 540 ± 1, late 770-790, early sanity 100/130
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/053Dmirror_pbrs_only_warm031B80_20260420_094739/TeamVsBaselineShapingPPOTrainer_Soccer_8c3d4_00000_0_2026-04-20_09-48-01"

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/053Dmirror_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 43005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/053Dmirror_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000100/checkpoint-100" \
  --checkpoint "${TRIAL}/checkpoint_000130/checkpoint-130" \
  --checkpoint "${TRIAL}/checkpoint_000530/checkpoint-530" \
  --checkpoint "${TRIAL}/checkpoint_000540/checkpoint-540" \
  --checkpoint "${TRIAL}/checkpoint_000550/checkpoint-550" \
  --checkpoint "${TRIAL}/checkpoint_000670/checkpoint-670" \
  --checkpoint "${TRIAL}/checkpoint_000680/checkpoint-680" \
  --checkpoint "${TRIAL}/checkpoint_000690/checkpoint-690" \
  --checkpoint "${TRIAL}/checkpoint_000780/checkpoint-780" \
  --checkpoint "${TRIAL}/checkpoint_000790/checkpoint-790" \
  2>&1 | tee docs/experiments/artifacts/official-evals/053Dmirror_baseline1000.log
exit ${PIPESTATUS[0]}
