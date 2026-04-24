#!/bin/bash
# Stage 1 1000ep eval combo for 4 today's lanes:
# 104A PASS restore (peak ckpt 490) / 104B DEFENDER restore (peak ckpt 480) /
# 110A NEAR-GOAL bottleneck (peak ckpt 250/280) / 110B MID-FIELD bottleneck (peak ckpt 290).
# Single parallel-7 run on idle node (~10min).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=today_lanes_stage1_combo
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/today_lanes_stage1_combo

RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

# 104A PASS restore (peak inline @490 = 0.895)
TRIAL_104A=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/104A_layered_p2_passdecision_restore_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL_104A=${TRIAL_104A%/}
# 104B DEFENDER restore (peak @480 = 0.900)
TRIAL_104B=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/104B_layered_p2_defender_restore_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL_104B=${TRIAL_104B%/}
# 110A NEAR-GOAL bottleneck (peak @250 = 0.895, 280 = 0.895)
TRIAL_110A=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/110A_bottleneck_neargoal_striker_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL_110A=${TRIAL_110A%/}
# 110B MID-FIELD bottleneck (peak @290 = 0.755 — need to verify regression)
TRIAL_110B=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/110B_bottleneck_midfield_dribble_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | sort | tail -1)
TRIAL_110B=${TRIAL_110B%/}

LOG=docs/experiments/artifacts/official-evals/today_lanes_stage1_combo.log

# NOTE: NOT using exec — that loses bash trap so .done flag never written on python error.
# Use direct invocation; bash trap fires after $PYTHON_BIN exits with any code.
$PYTHON_BIN scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 -j 7 \
  --base-port 53005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/today_lanes_stage1_combo \
  --checkpoint $TRIAL_104A/checkpoint_000490/checkpoint-490 \
  --checkpoint $TRIAL_104A/checkpoint_000500/checkpoint-500 \
  --checkpoint $TRIAL_104B/checkpoint_000480/checkpoint-480 \
  --checkpoint $TRIAL_104B/checkpoint_000500/checkpoint-500 \
  --checkpoint $TRIAL_110A/checkpoint_000250/checkpoint-250 \
  --checkpoint $TRIAL_110A/checkpoint_000280/checkpoint-280 \
  --checkpoint $TRIAL_110B/checkpoint_000290/checkpoint-290 \
  2>&1 | tee "$LOG"
