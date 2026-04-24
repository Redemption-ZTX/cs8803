#!/bin/bash
# Stage 3 H2H: 103A-warm-distill peak ckpt vs 055v2_extend@1750 SOTA.
# Per snapshot-106 peer-axis doctrine: SOTA shift requires direct H2H ≥ 0.55, z > 2.24.
# Uses --team0-module + --m1/m2 env-var pattern. Ckpt auto-selected from Stage 2 top.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

# Peak ckpt will be picked from Stage 2 rerun — default to ckpt 300, override via CKPT_103A env
PEAK_CKPT=${CKPT_103A:-300}

LANE_TAG=103A_warm_distill_stage3_h2h
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/headtohead

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55
CKPT_103A_PATH=$TRIAL/checkpoint_$(printf %06d $PEAK_CKPT)/checkpoint-$PEAK_CKPT
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

if [[ ! -f "$CKPT_103A_PATH" ]]; then
  echo "ERROR: 103A-warm-distill ckpt missing: $CKPT_103A_PATH" >&2
  exit 1
fi

LOG=docs/experiments/artifacts/official-evals/headtohead/103A_warm_distill_${PEAK_CKPT}_vs_1750.log

PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=$CKPT_103A_PATH \
TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_1750 \
exec $PYTHON_BIN scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 63505 \
  2>&1 | tee "$LOG"
