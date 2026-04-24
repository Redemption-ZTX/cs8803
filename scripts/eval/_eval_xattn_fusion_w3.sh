#!/bin/bash
# DIR-H W3 (untrained 1750-anchored cross-attention fusion) eval — baseline 1000ep.
# 8 experts: 1750 (anchor, +3.0 bias) + 055/029B generalists + 081/101A/103A/103B/103C specialists.
# Expected: ~1750 baseline WR (0.91+) with small specialist perturbation.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=xattn_fusion_w3_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/xattn_fusion_w3_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

# DIR-H W3 hyperparams (defaults in agent.py, exposed as env vars for sweep later)
export XATTN_ANCHOR_BIAS=${XATTN_ANCHOR_BIAS:-3.0}
export XATTN_D_KEY=${XATTN_D_KEY:-64}
export XATTN_SEED=${XATTN_SEED:-0}

LOG=docs/experiments/artifacts/official-evals/xattn_fusion_w3_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v_xattn_fusion_w3_anchored \
  --opponents baseline \
  -n 1000 -j 1 \
  --base-port 62905 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/xattn_fusion_w3_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750 \
  2>&1 | tee "$LOG"
