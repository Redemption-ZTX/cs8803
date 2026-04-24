#!/bin/bash
# DIR-A ablation A-F RETEST with 103A-wd v2@400 SOTA-tier specialist (per snapshot-109 §5 top recommendation).
# Wave 3 §7C closed "0/5 specialists improve ensemble" but used v1 broken specialists.
# Now retest with v2 specialists to see if framework can extract value when specialist quality fixed.
# Two presets: ablation_v2_103Awd_balldul (single-slot v2 in BALL_DUEL) + ablation_v2_strong3 (v2+081+101A 3-slot).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=dir_a_v2_retest
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/dir_a_v2_retest

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

# Run 2 presets sequentially (each is its own selector config with different specialist mix).
for preset in ablation_v2_103Awd_balldul ablation_v2_strong3; do
  echo "===== preset=$preset =====" | tee -a docs/experiments/artifacts/official-evals/dir_a_v2_retest.log

  SELECTOR_PHASE_MAP_PRESET=$preset \
  $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module agents.v_selector_phase4 \
    --opponents baseline \
    -n 1000 -j 1 \
    --base-port 65305 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/dir_a_v2_retest \
    --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750 \
    2>&1 | tee -a docs/experiments/artifacts/official-evals/dir_a_v2_retest.log
done
