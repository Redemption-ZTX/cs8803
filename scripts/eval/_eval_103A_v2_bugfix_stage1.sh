#!/bin/bash
# Stage 1 baseline 1000ep for 103A-warm-distill v2 BUG-fix.
# Inline trajectory: peak 0.945@120, cluster 0.92-0.94 across iter 120-460.
# Top 10 ckpts: 110/120/130 + 160/170 + 270 + 340 + 400/410/440 + 467 (terminal).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_v2_bugfix_stage1
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_bugfix_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_v2_bugfix_20260422_202340/TeamVsBaselineShapingPPOTrainer_Soccer_bcfb1_00000_0_2026-04-22_20-24-06

# Top 11 ckpts: peak 120 + ±1 + cluster + late 400/410/440 + terminal
CKPTS="110 120 130 160 170 260 270 340 400 410 440 467"
ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  fi
done

LOG=docs/experiments/artifacts/official-evals/103A_v2_bugfix_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 64705 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103A_v2_bugfix_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
