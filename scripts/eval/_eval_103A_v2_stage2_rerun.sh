#!/bin/bash
# Stage 2 rerun for 103A-warm-distill v2 BUG-fix peak ckpt 400.
# v2 Stage 1 = 0.920 → combined 2000ep with rerun = SOTA confirmation.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_v2_bugfix_stage2_rerun
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_bugfix_stage2

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_v2_bugfix_20260422_202340/TeamVsBaselineShapingPPOTrainer_Soccer_bcfb1_00000_0_2026-04-22_20-24-06

LOG=docs/experiments/artifacts/official-evals/103A_v2_bugfix_stage2_rerun.log

# Rerun ckpt 400 (peak) + ckpt 467 (terminal, also strong) for triangulation
exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 2 \
  --base-port 64805 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_bugfix_stage2 \
  --checkpoint $TRIAL/checkpoint_000400/checkpoint-400 \
  --checkpoint $TRIAL/checkpoint_000467/checkpoint-467 \
  2>&1 | tee "$LOG"
