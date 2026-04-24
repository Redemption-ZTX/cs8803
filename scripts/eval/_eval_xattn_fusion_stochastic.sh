#!/bin/bash
# DIR-H W3 anchored fusion with STOCHASTIC sampling (audit SUSP-2 fix verification).
# Original eval used greedy=True; SUSP-2 said this could flip argmax under
# smeared fusion distributions. Stochastic should preserve probability mass intent.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=xattn_fusion_stochastic_eval
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/${LANE_TAG}

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export XATTN_ANCHOR_BIAS=${XATTN_ANCHOR_BIAS:-3.0}
export XATTN_D_KEY=${XATTN_D_KEY:-64}
export XATTN_SEED=${XATTN_SEED:-0}
export XATTN_GREEDY=0  # Stochastic sampling (audit SUSP-2 fix)

LOG=docs/experiments/artifacts/official-evals/${LANE_TAG}_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v_xattn_fusion_w3_anchored \
  --opponents baseline \
  -n 1000 -j 1 \
  --base-port 64405 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/${LANE_TAG} \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750 \
  2>&1 | tee "$LOG"
