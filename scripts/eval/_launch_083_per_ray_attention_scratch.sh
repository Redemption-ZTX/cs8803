#!/bin/bash
# 083: Per-Ray Attention Siamese encoder scratch (DIR ③ architecture test).
# Replaces the per-agent monolithic 336 → MLP(256,256) encoder with a shared
# ray embedding + Transformer-style self-attention over 14 ray tokens
# (Option A with per-ray temporal collapse: 336 = 3 × 14 × 8 → 14 × 24).
# NOT a distillation lane — pure arch revision from scratch to test whether
# per-ray spatial inductive bias breaks the 0.91 tie-saturation plateau.
# See snapshot-083-per-ray-attention.md for design + thresholds.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

# Running-flag convention (auto-manage .running / .done)
LANE_TAG=083_per_ray
SLURM_LOG_DIR=docs/experiments/artifacts/slurm-logs
mkdir -p $SLURM_LOG_DIR
RUNNING_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.running
DONE_FLAG=$SLURM_LOG_DIR/${LANE_TAG}.done
rm -f $DONE_FLAG
touch $RUNNING_FLAG
cleanup() { local rc=$?; rm -f $RUNNING_FLAG; echo "EXIT_CODE=$rc at $(date)" > $DONE_FLAG; }
trap cleanup EXIT
export LOCAL_DIR=/storage/ice1/5/1/wsun377/ray_results_scratch
PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export QUIET_CONSOLE=${QUIET_CONSOLE:-0}

unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset WARMSTART_CHECKPOINT TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH OUTCOME_PBRS_PREDICTOR_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_TEACHER_POLICY_ID
unset TEAM_DISTILL_ENSEMBLE_KL TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS
unset TEAM_DISTILL_ALPHA_INIT TEAM_DISTILL_ALPHA_FINAL TEAM_DISTILL_DECAY_UPDATES TEAM_DISTILL_TEMPERATURE
unset TEAM_DISTILL_TEACHER_WEIGHTS
unset TEAM_SIAMESE_TWO_STREAM TEAM_SIAMESE_SELF_HIDDENS TEAM_SIAMESE_ENV_HIDDENS TEAM_SIAMESE_SELF_SLICE_DIM

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Hard-coded PORT_SEED — far from active lanes (082=82, 055=37, 054=31, 053A=23, 051D=51, 031B-noshape=13).
PORT_SEED=${PORT_SEED:-83}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=083_per_ray_attention_scratch_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r083_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture: 031B-arch foundation + NEW per-ray attention encoder (083).
# Per-agent 336 → reshape (3 frames × 14 rays × 8 feat) → permute→flatten →
# (14 rays × 24 feat) → linear embed (24→64) + pos embed → 2× pre-LN
# Transformer block (4 heads, ffn=128) → mean pool → Linear(64→256, ReLU) →
# 256-d agent feature (= 4 tokens × 64-dim, matches 031B cross-attention).
export TEAM_SIAMESE_ENCODER=1
export TEAM_SIAMESE_ENCODER_HIDDENS=256,256
export TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64
export TEAM_SIAMESE_PER_RAY_ATTN=1
export TEAM_PER_RAY_N_RAYS=14
export TEAM_PER_RAY_FEAT_DIM=24
export TEAM_PER_RAY_EMBED_DIM=64
export TEAM_PER_RAY_ATTN_LAYERS=2
export TEAM_PER_RAY_ATTN_HEADS=4
export TEAM_PER_RAY_FFN_HIDDEN=128
export TEAM_PER_RAY_POOL=mean

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0

# v2 shaping (same as 031B / 055 / 082, snapshot-083 §2.5)
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

# Budget: same as 031B/055/082 (1250 iter scratch, 50M steps, ~12h)
export TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

mkdir -p docs/experiments/artifacts/slurm-logs
LOG=docs/experiments/artifacts/slurm-logs/083_per_ray_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
