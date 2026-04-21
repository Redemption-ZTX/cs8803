#!/bin/bash
# 053D-mirror: PBRS-only from weak base — fair test of "v2 as ceiling" hypothesis.
# Mirrors 051D's design philosophy but swaps reward path:
#   Base: 031B@80 (weak, ~50% baseline WR, ~80 iter v2 prior)
#   Reward: outcome-PBRS only (no v2 shaping, no learned-bucket reward)
#   Budget: 800 iter (long, mirror 051D)
# Tests:
#   1. PBRS path vs learned-bucket path direct (053D-mirror vs 051D)
#   2. v2 as ceiling vs accelerator (053D-mirror vs 053Acont 0.898)
# See snapshot-053D-mirror-pbrs-only-from-weak-base.md for full design + rationale.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
export LOCAL_DIR=/storage/ice1/5/1/wsun377/ray_results_scratch

# Running-flag convention
LANE_TAG=053Dmirror
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() {
  local rc=$?
  rm -f $RUNNING_FLAG
  echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG
}
trap cleanup EXIT

PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export QUIET_CONSOLE=${QUIET_CONSOLE:-0}

unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_ENSEMBLE_KL
unset RND_ENABLED CURRICULUM_ENABLED

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# PORT_SEED far from active lanes (054=31, 055=37, 053Acont=25 done, 056ABCD=41-44, 057=35, 058=39)
PORT_SEED=${PORT_SEED:-29}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

# Warmstart: 031B@80 (weak ckpt, ~50% baseline WR, mirror 051D's choice)
export WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# A2 PBRS predictor (same as 053A — A3 calibrated)
export OUTCOME_PBRS_PREDICTOR_PATH=/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
export OUTCOME_PBRS_WEIGHT=0.01
export OUTCOME_PBRS_WARMUP_STEPS=10000
export OUTCOME_PBRS_MAX_BUFFER_STEPS=80

# v2 shaping OFF (KEY ablation — no v2 during training)
export USE_REWARD_SHAPING=0
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0 SHAPING_BALL_PROGRESS=0 SHAPING_GOAL_PROXIMITY_SCALE=0
export SHAPING_GOAL_PROXIMITY_GAMMA=0.99 SHAPING_GOAL_CENTER_X=15.0 SHAPING_GOAL_CENTER_Y=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0 SHAPING_POSSESSION_BONUS=0 SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=0 SHAPING_DEEP_ZONE_OUTER_PENALTY=0
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=0 SHAPING_DEEP_ZONE_INNER_PENALTY=0
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0

export RUN_NAME=053Dmirror_pbrs_only_warm031B80_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r53Dm_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture (cross-attention, same as 031B base + 053A)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0

# Budget: 800 iter mirror 051D (TIMESTEPS_TOTAL=32M)
export TIMESTEPS_TOTAL=32000000 MAX_ITERATIONS=800 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/053Dmirror_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
