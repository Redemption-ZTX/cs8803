#!/bin/bash
# Fresh n=5000 baseline eval on 111A@180 — same protocol as 1750 5000ep validation.
# 5 parallel workers × 1000ep each, all on ckpt 180. Compare aggregate to 1750's 0.9066.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=111A_180_fresh_n5000
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/111A_180_fresh_n5000
RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/111A_strong_opp_test_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL=${TRIAL%/}
CKPT=$TRIAL/checkpoint_000180/checkpoint-180

LOG=docs/experiments/artifacts/official-evals/111A_180_fresh_n5000_baseline.log

# 5 × 1000ep parallel on same ckpt (mirrors 1750 fresh n=5000 protocol)
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 60005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/111A_180_fresh_n5000 \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  2>&1 | tee "$LOG"
