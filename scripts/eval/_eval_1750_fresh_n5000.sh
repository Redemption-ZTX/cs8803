#!/bin/bash
# Track A.3: Fresh n=5000 rerun of 1750 SOTA to test selection-effect hypothesis.
# Per evaluation audit: 1750 was promoted from 8-ckpt max → expected ~+0.016 over true mean
# under iid noise. If true SOTA-tier mean is 0.908-0.910, 1750 actual at fresh sample
# should regress to ~0.910 not 0.9155. This rerun is the proof.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=1750_fresh_n5000_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/1750_fresh_n5000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

LOG=docs/experiments/artifacts/official-evals/1750_fresh_n5000_baseline.log

# Single ckpt × n=5000 episodes. Use parallel-5 to split into 5 × 1000ep tasks (different ports → different seeds).
# That gives effective n=5000 with 5 independent runs we can mean+SE.
# But evaluate_official_suite_parallel only takes one --checkpoint per task; need a different approach.
# Hack: pass same ckpt 5 times — parallel-5 will run 5 tasks at different ports.
# We then aggregate the 5 wins/losses for combined verdict.

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 64305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/1750_fresh_n5000 \
  --checkpoint $CKPT_1750 \
  --checkpoint $CKPT_1750 \
  --checkpoint $CKPT_1750 \
  --checkpoint $CKPT_1750 \
  --checkpoint $CKPT_1750 \
  2>&1 | tee "$LOG"
