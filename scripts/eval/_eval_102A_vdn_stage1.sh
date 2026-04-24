#!/bin/bash
# Stage 1 baseline 1000ep for 102A VDN (DIR-F) — full 1250 iter complete after restore fix.
# Inline trajectory: plateau 0.87-0.90 iter 980-1240; isolated peak 0.930 @ 1070.
# Top 10 ckpts selected by inline + ±1 window on peak.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=102A_vdn_stage1
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/102A_vdn_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

# 102A VDN has 2 trial dirs — original scratch (through 930) + restore (from 930 to 1250)
# Restore trial holds all ckpts 930-1250
TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/102A_vdn_restore_20260422_103346/TeamVsBaselineShapingPPOTrainer_Soccer_51f94_00000_0_2026-04-22_10-34-08

# Top 10 by inline: 1070 (peak 0.930) + plateau + ±1
CKPTS="970 980 990 1000 1010 1060 1070 1080 1140 1210 1250"

ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  fi
done

LOG=docs/experiments/artifacts/official-evals/102A_vdn_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 63805 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/102A_vdn_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
