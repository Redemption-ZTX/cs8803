#!/bin/bash
# Samples 3, 4, 5 of v2 ckpt 400 — to reach combined 5000ep parity with 1750 SOTA validation.
# After Stage 1 (sample 1) + Stage 2 rerun (sample 2), this adds 3 more independent runs.
# Total: 5 × 1000ep = 5000ep combined, SE ≈ 0.004 (matches 1750 fresh validation rigor).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_v2_400_samples_3to5
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_400_samples_3to5

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_v2_bugfix_20260422_202340/TeamVsBaselineShapingPPOTrainer_Soccer_bcfb1_00000_0_2026-04-22_20-24-06/checkpoint_000400/checkpoint-400

LOG=docs/experiments/artifacts/official-evals/103A_v2_400_samples_3to5.log

# Pass same ckpt 3 times → parallel-3 = 3 independent 1000ep tasks at different ports
# (per memory: ports DON'T give independent seeds for Unity, but at least we can run sequentially OR
# accept that "3 identical samples" is fine if eval already gave 0.920 reproducibly — Stage 2 will tell)
exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 3 \
  --base-port 65005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_400_samples_3to5 \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  --checkpoint $CKPT \
  2>&1 | tee "$LOG"
