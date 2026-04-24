#!/bin/bash
# Stage 2 rerun for 103A-warm-distill Stone Layered L2 lane.
# Stage 1 peak 0.923 @ ckpt 300 (single-shot) + 0.913 @ ckpt 400. Rerun both
# at 1000ep each → combined 2000ep confirms if truly above 1750 SOTA 0.9155.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_warm_distill_stage2
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_warm_distill_stage2

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55

LOG=docs/experiments/artifacts/official-evals/103A_warm_distill_stage2_rerun.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 2 \
  --base-port 63305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_warm_distill_stage2 \
  --checkpoint $TRIAL/checkpoint_000300/checkpoint-300 \
  --checkpoint $TRIAL/checkpoint_000400/checkpoint-400 \
  2>&1 | tee "$LOG"
