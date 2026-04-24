#!/bin/bash
# DIR-A Wave 3 narrow-trigger eval — baseline 1000ep vs SOTA anchor.
# Selector: NEAR-GOAL → 081 aggressive (only when nearest<0.10 AND centroid>0.5);
# BALL_DUEL/POS/MID → 1750 SOTA. Narrow trigger bounds NEAR-GOAL phase freq ~10%
# of Wave 2's mapping so even if 081 is worse the ensemble damage is minimal.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=wave3_narrow_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/wave3_narrow_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export SELECTOR_PHASE_MAP_PRESET=wave3_narrow

# No --checkpoint needed for zero-training selector agent — the agent self-loads
# specialist ckpts per _PRESETS. Pass a dummy but valid path to satisfy eval CLI,
# OR use the --checkpoint-optional mode.
# evaluate_official_suite_parallel.py's --checkpoint is required. For selector
# agent we use a dummy path (the SOTA path) since the agent ignores it (resolves
# its own via preset). Check if script has a no-ckpt mode instead.

LOG=docs/experiments/artifacts/official-evals/wave3_narrow_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v_selector_phase4 \
  --opponents baseline \
  -n 1000 -j 1 \
  --base-port 62805 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/wave3_narrow_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750 \
  2>&1 | tee "$LOG"
