#!/bin/bash
# Fresh n=5000 baseline eval on 115A@250 — same protocol as 1750/111A 5000ep validation.
# 5 parallel workers × 1000ep each on the same ckpt. Verify single-shot 0.920 against high-side noise.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=115A_250_fresh_n5000
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/115A_250_fresh_n5000
RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/115A_no_distill_warm_only_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL=${TRIAL%/}
CKPT=$TRIAL/checkpoint_000250/checkpoint-250

LOG=docs/experiments/artifacts/official-evals/115A_250_fresh_n5000_baseline.log

# 5 × 1000ep parallel on same ckpt
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 61005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/115A_250_fresh_n5000 \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  2>&1 | tee "$LOG"
