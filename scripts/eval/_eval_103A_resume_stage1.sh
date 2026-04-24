#!/bin/bash
# Stage 1 baseline 1000ep for 103A resume past 500 (interceptor specialist upgrade).
# Inline peak 0.660 @ ckpt 640, 0.655 @ 760 — several 0.60+ cluster iter 550-770.
# Training stopped at iter 787 (SLURM wall).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_resume_stage1
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_resume_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_resume_20260422_102921/TeamVsBaselineShapingPPOTrainer_Soccer_b46b6_00000_0_2026-04-22_10-29-43

# Top 9 ckpts: inline peak 0.660@640 + ±1 + runner-ups
CKPTS="560 630 640 650 710 760 770 780 787"
ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  fi
done

LOG=docs/experiments/artifacts/official-evals/103A_resume_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 63605 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_resume_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
