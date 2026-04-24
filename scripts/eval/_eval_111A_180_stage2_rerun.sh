#!/bin/bash
# Stage 2 rerun on 111A ckpt 180 (peak) — combined 2000ep tightens SE.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=111A_180_stage2_rerun
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/111A_180_stage2
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

LOG=docs/experiments/artifacts/official-evals/111A_180_stage2_rerun.log

# Rerun ckpt 180 + 120 (top 2 peaks) for 2nd sample on each → combined 2000ep
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 2 \
  --base-port 54505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/111A_180_stage2 \
  --checkpoint $TRIAL/checkpoint_000180/checkpoint-180 \
  --checkpoint $TRIAL/checkpoint_000120/checkpoint-120 \
  2>&1 | tee "$LOG"
