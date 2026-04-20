#!/bin/bash
# 031B-noshape ablation: scratch + cross-attention + sparse env reward only.
# Tests fundamental question: does v2 shaping contribute meaningfully on cross-attention arch?
# All other config identical to original 031B (which got 1000ep 0.882 with v2).
#
# Hypothesis A: 031B-noshape ≥ 0.85 → v2 shaping marginal, architecture is everything
# Hypothesis B: 031B-noshape ≤ 0.65 → v2 shaping essential, learned reward replacement matters
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export QUIET_CONSOLE=${QUIET_CONSOLE:-0}

unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset WARMSTART_CHECKPOINT TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Hard-coded PORT_SEED far from any active lane
PORT_SEED=${PORT_SEED:-13}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=031B_noshape_ablation_cross_attention_512x512_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r31Bns_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture (same as 031B cross-attention scratch)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0

# === KEY DIFF: NO v2 shaping (sparse env reward only) ===
export USE_REWARD_SHAPING=0
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.0
export SHAPING_BALL_PROGRESS=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.0
export SHAPING_POSSESSION_BONUS=0.0
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=0
export SHAPING_DEEP_ZONE_OUTER_PENALTY=0.0
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=0
export SHAPING_DEEP_ZONE_INNER_PENALTY=0.0
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0
export SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0
export SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_GOAL_PROXIMITY_GAMMA=0.99
export SHAPING_GOAL_CENTER_X=15.0
export SHAPING_GOAL_CENTER_Y=0.0

# Budget: same as 031B (1250 iter scratch)
export TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=50
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

mkdir -p docs/experiments/artifacts/slurm-logs
LOG=docs/experiments/artifacts/slurm-logs/031B_noshape_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
