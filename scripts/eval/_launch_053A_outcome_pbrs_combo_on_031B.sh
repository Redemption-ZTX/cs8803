#!/bin/bash
# 053A: combo (v2 + outcome PBRS) warmstart from 031B@1220 cross-attention SOTA.
# A2 PBRS uses calibrated A3 outcome predictor (direction_1b_v3, val_acc 0.835,
# per-prefix gap 0.015→0.240). Reward = v2 shaping + λ·(V(s_t+1) - V(s_t)).
#
# Hypothesis: PBRS provides dense per-step gradient that complements v2 shaping.
# vs 051A combo (v2 + multi-head learned reward, peak 0.888): tests if outcome-prediction
# PBRS gives different signal than failure-bucket multi-head.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export QUIET_CONSOLE=${QUIET_CONSOLE:-0}

unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Hard-coded PORT_SEED far from active lanes
PORT_SEED=${PORT_SEED:-23}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

# Warmstart: 031B@1220 cross-attention SOTA (1000ep 0.882)
export WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220

# A2 PBRS predictor (A3 calibrated)
export OUTCOME_PBRS_PREDICTOR_PATH=/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
export OUTCOME_PBRS_WEIGHT=0.01
export OUTCOME_PBRS_WARMUP_STEPS=10000
export OUTCOME_PBRS_MAX_BUFFER_STEPS=80

# v2 shaping ON (combo)
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01 SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_GOAL_PROXIMITY_GAMMA=0.99 SHAPING_GOAL_CENTER_X=15.0 SHAPING_GOAL_CENTER_Y=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002 SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0

export RUN_NAME=053A_outcome_pbrs_combo_on_031B_512x512_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r53A_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture (cross-attention same as 031B base for warmstart compat)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0

# Budget: 200 iter same as 051A/B (direct comparison)
export TIMESTEPS_TOTAL=8000000 MAX_ITERATIONS=200 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=50
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

mkdir -p docs/experiments/artifacts/slurm-logs
LOG=docs/experiments/artifacts/slurm-logs/053A_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
