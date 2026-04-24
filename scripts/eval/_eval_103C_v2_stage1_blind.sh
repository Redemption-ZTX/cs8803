#!/bin/bash
# Stage 1 blind-backfill for 103C v2 shot_reward (inline eval died iter 130 due to
# join-timeout bug, see memory feedback_inline_eval_death_root_cause.md).
# Blind 5-ckpt backfill at iter 100/200/300/400/500.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103C_v2_stage1_blind
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/103C_v2_baseline1000

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

TRIAL=/storage/ice1/5/1/wsun377/ray_results_scratch/103C_v2_dribble_shot_20260422_103018/TeamVsBaselineShapingPPOTrainer_Soccer_d64b5_00000_0_2026-04-22_10-30-40

CKPTS="100 200 300 400 500"
ARGS=""
for c in $CKPTS; do
  ck=$TRIAL/checkpoint_$(printf %06d $c)/checkpoint-$c
  if [[ -f "$ck" ]]; then
    ARGS="$ARGS --checkpoint $ck"
  fi
done

LOG=docs/experiments/artifacts/official-evals/103C_v2_baseline1000.log

exec $PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 5 \
  --base-port 64105 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/103C_v2_baseline1000 \
  $ARGS \
  2>&1 | tee "$LOG"
