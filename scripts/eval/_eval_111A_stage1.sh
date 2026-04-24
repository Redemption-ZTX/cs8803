#!/bin/bash
# Stage 1 1000ep eval for 111A strong-opp test — KEY EXPERIMENT.
# Inline 200ep peaks: 0.940@120 / 0.935@140/180/210 / 0.930@240 / late-window cluster 0.92.
# Real 1000ep verdict tells if stronger opponent training breaks 0.91 ceiling.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=111A_stage1_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/111A_stage1
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

LOG=docs/experiments/artifacts/official-evals/111A_stage1_baseline.log

# Pick top 8 ckpts: peak 120 + cluster 140/180/210/220/240 + ±1 + final 250
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 53505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/111A_stage1 \
  --checkpoint $TRIAL/checkpoint_000120/checkpoint-120 \
  --checkpoint $TRIAL/checkpoint_000140/checkpoint-140 \
  --checkpoint $TRIAL/checkpoint_000180/checkpoint-180 \
  --checkpoint $TRIAL/checkpoint_000210/checkpoint-210 \
  --checkpoint $TRIAL/checkpoint_000220/checkpoint-220 \
  --checkpoint $TRIAL/checkpoint_000240/checkpoint-240 \
  --checkpoint $TRIAL/checkpoint_000250/checkpoint-250 \
  2>&1 | tee "$LOG"
