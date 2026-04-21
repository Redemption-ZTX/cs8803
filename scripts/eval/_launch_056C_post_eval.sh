#!/bin/bash
# 056C post-eval: smart 10-ckpt subset, lr=1.5e-4 LR sweep mid-point
# Selected from pick_top: peak 940 + window, plateau 990-1040, late
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/056C_pbt_lr0.00015_scratch_20260420_092048/TeamVsBaselineShapingPPOTrainer_Soccer_d00f3_00000_0_2026-04-20_09-21-17"

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/056C_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 42005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/056C_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_000490/checkpoint-490" \
  --checkpoint "${TRIAL}/checkpoint_000750/checkpoint-750" \
  --checkpoint "${TRIAL}/checkpoint_000930/checkpoint-930" \
  --checkpoint "${TRIAL}/checkpoint_000940/checkpoint-940" \
  --checkpoint "${TRIAL}/checkpoint_000950/checkpoint-950" \
  --checkpoint "${TRIAL}/checkpoint_000990/checkpoint-990" \
  --checkpoint "${TRIAL}/checkpoint_001020/checkpoint-1020" \
  --checkpoint "${TRIAL}/checkpoint_001040/checkpoint-1040" \
  --checkpoint "${TRIAL}/checkpoint_001100/checkpoint-1100" \
  --checkpoint "${TRIAL}/checkpoint_001110/checkpoint-1110" \
  2>&1 | tee docs/experiments/artifacts/official-evals/056C_baseline1000.log
exit ${PIPESTATUS[0]}
