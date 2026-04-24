#!/bin/bash
# Stage 1 1000ep eval for 115A no-distill warm-only — KEY anchor-removal verdict.
# Inline peaks: 0.950@210 / 0.940@30/120/190 / 0.935@150 / late-window mean 0.91.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=115A_stage1_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/115A_stage1
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

LOG=docs/experiments/artifacts/official-evals/115A_stage1_baseline.log

# Top 7 ckpts: peak 210 + cluster 30/120/150/190 + late 240/250
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 54005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/115A_stage1 \
  --checkpoint $TRIAL/checkpoint_000030/checkpoint-30 \
  --checkpoint $TRIAL/checkpoint_000120/checkpoint-120 \
  --checkpoint $TRIAL/checkpoint_000150/checkpoint-150 \
  --checkpoint $TRIAL/checkpoint_000190/checkpoint-190 \
  --checkpoint $TRIAL/checkpoint_000210/checkpoint-210 \
  --checkpoint $TRIAL/checkpoint_000240/checkpoint-240 \
  --checkpoint $TRIAL/checkpoint_000250/checkpoint-250 \
  2>&1 | tee "$LOG"
