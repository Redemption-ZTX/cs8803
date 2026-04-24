#!/bin/bash
# 3rd sample 1000ep rerun on 103A-warm-distill@400.
# Stage 1 = 0.913, Stage 2 = 0.915 → combined 2000ep = 0.914
# vs 1750 true (fresh 5000ep) = 0.9066 → Δ=+0.007 above 1750 true
# Combined 3000ep tightens SE to ~0.005. If 3rd sample ≥ 0.910 → confirmed marginal SOTA shift.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_wd_400_3rd_sample
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_wd_400_3rd

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55/checkpoint_000400/checkpoint-400

LOG=docs/experiments/artifacts/official-evals/103A_wd_400_3rd_sample.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 1 \
  --base-port 64605 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_wd_400_3rd \
  --checkpoint $CKPT \
  2>&1 | tee "$LOG"
