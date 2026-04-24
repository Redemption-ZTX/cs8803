#!/bin/bash
# Stage 1 baseline 1000ep eval for 103A-warm-distill (Stone Layered L2 ULTIMATE).
# Training: warm from 1750 + KL distill from 1750 + INTERCEPTOR scenario + light aux reward.
# Inline 200ep trajectory: peak 0.950 @ iter 350, 0.945 @ 100/200, 0.940 @ 280, 0.935 @ 150.
# Blind-backfill strategy: inline top 6 + late window + ±1 = 13 ckpts.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_warm_distill_stage1
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_warm_distill_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55

# 13 ckpts: top 6 inline + ±1 on peak + late window
CKPTS="100 150 200 260 280 300 340 350 360 400 430 460 480 489"

ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  else
    echo "[WARN] ckpt $c not yet saved, skipping" >&2
  fi
done

if [[ -z "$ARGS" ]]; then
  echo "ERROR: no ckpts available yet for 103A-warm-distill" >&2
  exit 1
fi

LOG=docs/experiments/artifacts/official-evals/103A_warm_distill_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 63005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_warm_distill_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
