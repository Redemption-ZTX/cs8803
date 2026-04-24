#!/bin/bash
# Stage 2 rerun for 103A extend ckpt 590 (peak Stage 1 0.916).
# After 1750 fresh n=5000 = 0.9066 correction, ckpt 590's 0.916 may be real
# +0.009 above 1750 true value. Combined 2000ep tightens SE to 0.011 → can
# detect Δ=0.005 with reasonable power.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_extend_stage2
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_extend_stage2

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_extend_20260422_120331/TeamVsBaselineShapingPPOTrainer_Soccer_dd220_00000_0_2026-04-22_12-03-55

LOG=docs/experiments/artifacts/official-evals/103A_extend_stage2_rerun.log

# Single ckpt × 1000ep rerun (Stage 1 ckpt 590 already gave 0.916; Stage 2 = independent 1000ep)
exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 1 \
  --base-port 64505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_extend_stage2 \
  --checkpoint $TRIAL/checkpoint_000590/checkpoint-590 \
  2>&1 | tee "$LOG"
