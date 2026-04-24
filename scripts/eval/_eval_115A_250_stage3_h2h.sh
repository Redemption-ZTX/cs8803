#!/bin/bash
# Stage 3 H2H: 115A@250 vs 1750 SOTA + 103A-wd v2@467 + 111A@180 (peer-axis triangulation).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=115A_250_stage3_h2h
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

TRIAL_115A=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/115A_no_distill_warm_only_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL_115A=${TRIAL_115A%/}
CKPT_115A_250=$TRIAL_115A/checkpoint_000250/checkpoint-250
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750
CKPT_103AWD_V2_467=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_v2_bugfix_20260422_202340/TeamVsBaselineShapingPPOTrainer_Soccer_bcfb1_00000_0_2026-04-22_20-24-06/checkpoint_000467/checkpoint-467
TRIAL_111A=$(ls -d /storage/ice1/5/1/wsun377/ray_results_scratch/111A_strong_opp_test_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/ | head -1)
TRIAL_111A=${TRIAL_111A%/}
CKPT_111A_180=$TRIAL_111A/checkpoint_000180/checkpoint-180

# Matchup 1: 115A@250 vs 1750 (primary peer-axis verdict)
LOG1=docs/experiments/artifacts/official-evals/headtohead/115A_250_vs_1750.log
echo "===== H2H matchup 1: 115A@250 vs 1750 SOTA =====" | tee -a $LOG1
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=$CKPT_115A_250 \
TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_1750 \
$PYTHON_BIN scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 61205 \
  2>&1 | tee -a $LOG1

# Matchup 2: 115A@250 vs 103A-wd v2@467 (paradigm comparison)
LOG2=docs/experiments/artifacts/official-evals/headtohead/115A_250_vs_103Awd_v2_467.log
echo "===== H2H matchup 2: 115A@250 vs 103A-wd v2@467 =====" | tee -a $LOG2
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=$CKPT_115A_250 \
TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_103AWD_V2_467 \
$PYTHON_BIN scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 61305 \
  2>&1 | tee -a $LOG2

# Matchup 3: 115A@250 vs 111A@180 (companion-paradigm direct comparison)
LOG3=docs/experiments/artifacts/official-evals/headtohead/115A_250_vs_111A_180.log
echo "===== H2H matchup 3: 115A@250 vs 111A@180 =====" | tee -a $LOG3
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=$CKPT_115A_250 \
TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_111A_180 \
$PYTHON_BIN scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 61405 \
  2>&1 | tee -a $LOG3
