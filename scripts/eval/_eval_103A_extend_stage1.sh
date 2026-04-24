#!/bin/bash
# Stage 1 baseline 1000ep for 103A-warm-distill EXTEND (iter 489→750).
# Note: ran with BUG-1 (opp pool no resample) + BUG-2 (alpha decay too fast) — same
# bugs as original 103A-warm-distill. Use as "bugged-but-extended" comparison.
# Inline top peaks: 0.940 @ 540/490, 0.925 @ 680/590/550, plateau 0.91-0.94.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_extend_stage1
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_extend_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_extend_20260422_120331/TeamVsBaselineShapingPPOTrainer_Soccer_dd220_00000_0_2026-04-22_12-03-55

CKPTS="490 500 540 550 580 590 650 680 700 750"
ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  fi
done

LOG=docs/experiments/artifacts/official-evals/103A_extend_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 64205 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_extend_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
