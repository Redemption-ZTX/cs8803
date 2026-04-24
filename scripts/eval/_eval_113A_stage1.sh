#!/bin/bash
# Stage 1 1000ep eval for 113A wide encoder — KEY architecture-ceiling test.
# Inline 200ep peaks: 0.895 @ 420/500/600 cluster; plateau 0.85-0.90 at iters 400-620.
# 1000ep Stage 1 verdict: does wide encoder (FCNET 1024,1024 + Siamese 384,384) break 0.91 ceiling?
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=113A_stage1_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/113A_stage1
RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/113A_wide_encoder_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL=${TRIAL%/}

LOG=docs/experiments/artifacts/official-evals/113A_stage1_baseline.log

# top 5% + ties + ±1 = 9 ckpts: 400/420/440/480/500/520/580/600/620
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 62005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/113A_stage1 \
  --checkpoint $TRIAL/checkpoint_000400/checkpoint-400 \
  --checkpoint $TRIAL/checkpoint_000420/checkpoint-420 \
  --checkpoint $TRIAL/checkpoint_000440/checkpoint-440 \
  --checkpoint $TRIAL/checkpoint_000480/checkpoint-480 \
  --checkpoint $TRIAL/checkpoint_000500/checkpoint-500 \
  --checkpoint $TRIAL/checkpoint_000520/checkpoint-520 \
  --checkpoint $TRIAL/checkpoint_000580/checkpoint-580 \
  --checkpoint $TRIAL/checkpoint_000600/checkpoint-600 \
  --checkpoint $TRIAL/checkpoint_000620/checkpoint-620 \
  2>&1 | tee "$LOG"
