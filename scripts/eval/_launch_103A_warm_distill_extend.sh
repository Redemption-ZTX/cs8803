#!/bin/bash
# 103A-warm-distill EXTEND: resume from ckpt 489 (wall-terminated at iter 489/500) and
# extend to iter 750 with same recipe. If Stage 2 combined 0.914 @ ckpt 400 is tied with
# 1750 SOTA, extending past current window may push past 1750 (or validate plateau).
#
# Per memory feedback_ray_restore_time_total.md: TIME_TOTAL_S=0 for restore lanes.
# Per memory feedback_slurm_wall_budget.md: need ~8h wall for 260 more iter × 30s/iter ≈ 2.2h + buffer.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_warm_distill_extend
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

unset BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH OUTCOME_PBRS_PREDICTOR_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_TEACHER_POLICY_ID
unset TEAM_SIAMESE_TWO_STREAM TEAM_SIAMESE_PER_RAY_ATTN TEAM_SIAMESE_VDN
unset WARMSTART_CHECKPOINT  # use RESTORE_CHECKPOINT instead; mutually exclusive

# RESTORE from 103A-warm-distill final ckpt 489
export RESTORE_CHECKPOINT=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_20260422_073726/TeamVsBaselineShapingPPOTrainer_Soccer_b44ce_00000_0_2026-04-22_07-37-55/checkpoint_000489/checkpoint-489

# 1750 SOTA for KL distill (still anchored, but α=0 this phase — restore from decay schedule)
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

# KL distill continues with fresh schedule (don't inherit from restored state because
# α should already have decayed to 0 — set α_init=0.01, α_final=0, decay 4000 updates).
# This provides a residual anchor to prevent late-window drift seen in Stage 1 (300→489: -0.028).
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_1750
export TEAM_DISTILL_ALPHA_INIT=0.01
export TEAM_DISTILL_ALPHA_FINAL=0.0
export TEAM_DISTILL_DECAY_UPDATES=10000
export TEAM_DISTILL_TEMPERATURE=1.0

# Scenario init + BASELINE_PROB continue unchanged
export SCENARIO_RESET=interceptor_subtask
export BASELINE_PROB=0.7

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-197}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=103A_warm_distill_extend_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r103Awdex_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture: same as 103A-warm-distill (required to match restore ckpt)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# Reward shape same as 103A-warm-distill
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001
export SHAPING_BALL_PROGRESS=0.01
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.02
export SHAPING_POSSESSION_BONUS=0.002
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_EVENT_SHOT_REWARD=0.0
export SHAPING_EVENT_TACKLE_REWARD=0.05
export SHAPING_EVENT_CLEARANCE_REWARD=0.03
export SHAPING_EVENT_COOLDOWN_STEPS=5

# Extend budget: 489 → 750 iter (261 more × 30s ≈ 2.2h). Wall 4h conservative.
# TIME_TOTAL_S=0 per feedback_ray_restore_time_total.md — disable stop criterion
# since ckpt persists _time_total accumulated from Stage 1 run (~4h already).
export TIMESTEPS_TOTAL=40000000 MAX_ITERATIONS=750 TIME_TOTAL_S=0 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/103A_warm_distill_extend_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
