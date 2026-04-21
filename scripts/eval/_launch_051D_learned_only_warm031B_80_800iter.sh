#!/bin/bash
# 051D: learned-only (no v2) on 031B cross-attention warmstart from EARLY ckpt 80 (50ep WR=0.50).
# Same family as 051A/B (learned-reward × cross-attention). Different warmstart axis:
#   - 051A combo on 031B@1220 (frontier)
#   - 051B learned-only on 031B@1220 (frontier)
#   - 051D learned-only on 031B@80 (early, weak baseline)
# Hypothesis: 051D 1000ep peak ≫ 0.50 → learned reward CAN drive incremental improvement
#   on cross-attention arch from early weak start (covers part of "learned reward standalone driver"
#   question, but warmstart still has 80 iter v2 prior baked in — true zero-prior test = B-2 BC).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHON_BIN=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python

export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export QUIET_CONSOLE=${QUIET_CONSOLE:-0}

unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset TEAM_OPPONENT_CHECKPOINT  # train vs baseline only (default), not vs frozen

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Hard-coded ports far from leftover port allocation on 015-2-0
# (047B used 58310, 51C-v1 used 57074, leftover 57225/57239/61363/63090 still listening)
PORT_SEED=${PORT_SEED:-51}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

# Warmstart from 031B@80 (cross-attention SAME arch, internal 50ep WR=0.50)
export WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# Reward: learned-only (no v2 shaping)
export LEARNED_REWARD_MODEL_PATH=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/reward_models/051_strong_vs_strong/reward_model.pt
export LEARNED_REWARD_SHAPING_WEIGHT=0.003
export LEARNED_REWARD_WARMUP_STEPS=10000
export LEARNED_REWARD_APPLY_TO_TEAM1=0
export USE_REWARD_SHAPING=0

# All v2 shaping disabled
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.0 SHAPING_BALL_PROGRESS=0.0 SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.0 SHAPING_POSSESSION_BONUS=0.0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=0 SHAPING_DEEP_ZONE_OUTER_PENALTY=0
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=0 SHAPING_DEEP_ZONE_INNER_PENALTY=0
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0

export RUN_NAME=051D_learned_only_warm031B80_cross_attention_512x512_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r51D_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture (must match 031B@80's arch exactly for warmstart weight load)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0  # 100% baseline opponent (no frozen team opp)

# Budget: 200 iter same as 051A/B for direct comparison
export TIMESTEPS_TOTAL=32000000 MAX_ITERATIONS=800 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

mkdir -p docs/experiments/artifacts/slurm-logs
LOG=docs/experiments/artifacts/slurm-logs/051D_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
