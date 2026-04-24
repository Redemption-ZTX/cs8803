#!/bin/bash
# Stage 3 H2H @ n=1500 for 103A-warm-distill@400 vs 055v2_extend@1750.
# Per evaluation audit: n=500 H2H has 13% power for Δ=0.02. n=1500 → 80% power.
# Run as 3 parallel 500ep tasks (different ports → independent seeds).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_wd_400_vs_1750_h2h_n1500
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
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

CKPT_103A_WD=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55/checkpoint_000400/checkpoint-400
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

# 3 sequential 500ep H2H runs at different ports → aggregate to n=1500
LOG_BASE=docs/experiments/artifacts/official-evals/headtohead/103A_wd_400_vs_1750_h2h_n1500
mkdir -p $(dirname $LOG_BASE)

for i in 1 2 3; do
  port=$((64900 + i * 10))
  log="${LOG_BASE}_run${i}.log"
  echo "===== H2H run ${i}/3 (port $port) =====" | tee -a "${LOG_BASE}.log"
  PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
  TRAINED_RAY_CHECKPOINT=$CKPT_103A_WD \
  TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_1750 \
  $PYTHON_BIN scripts/eval/evaluate_headtohead.py \
    -m1 cs8803drl.deployment.trained_team_ray_agent \
    -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
    -e 500 \
    -p $port \
    2>&1 | tee "$log" | tail -30 | tee -a "${LOG_BASE}.log"
done

echo "===== ALL DONE =====" | tee -a "${LOG_BASE}.log"
