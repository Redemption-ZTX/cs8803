#!/bin/bash
# 083 Stage 2 — rerun ckpt-1000 (peak) for combined 2000ep verdict.
# Single-shot 1000ep = 0.919 (Δ vs 1750 SOTA +0.004 within SE) → need rerun
# to confirm true value vs single-shot luck.
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/083_per_ray_attention_scratch_20260421_210849/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/083_rerun1000_stage2
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 62605 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/083_rerun1000_stage2 \
  --checkpoint ${TRIAL%/}/checkpoint_001000/checkpoint-1000 \
  2>&1 | tee docs/experiments/artifacts/official-evals/083_rerun1000_stage2.log
