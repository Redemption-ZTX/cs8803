#!/bin/bash
# 103A-refined: Stone Layered Learning Layer 2 fix for 103A INTERCEPTOR lane.
#
# ROOT CAUSE (identified 2026-04-22, snapshot-106):
#   (1) 103A v1 used BASELINE_PROB=0.0 → specialist never saw baseline during
#       training → distribution mismatch at eval (baseline 0.548 vs random 0.19)
#   (2) Stone Layered Learning bootstrap skipped — specialist trained in
#       isolation without team-coordination context
#
# v1 FIX (this lane):
#   - Warm-start from 103A@500 (retain INTERCEPTOR scenario knowledge)
#   - Continue training in STANDARD 2v2 env (no scenario init) with
#     BASELINE_PROB=0.7 (70% vs baseline, 30% vs random)
#   - Keep INTERCEPTOR-flavored aux reward (tackle + clearance +
#     opp_progress_penalty) but combine with v2 base shape
#   - Budget 500 iter / 20M steps / 4h
#
# Note: did NOT include frozen-teammate bootstrap (Stone 2000 Layer 2 full
# spec) — requires per-agent policy split in env which is non-trivial.
# Distribution fix alone should resolve #1; if v1 successful we'll add
# teammate-freeze as v2 later.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_refined
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
unset TEAM_DISTILL_ENSEMBLE_KL TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS
unset TEAM_DISTILL_ALPHA_INIT TEAM_DISTILL_ALPHA_FINAL TEAM_DISTILL_DECAY_UPDATES TEAM_DISTILL_TEMPERATURE
unset TEAM_SIAMESE_TWO_STREAM TEAM_SIAMESE_PER_RAY_ATTN TEAM_SIAMESE_VDN

# IMPORTANT: no scenario init (Stone Layered Learning Layer 2 = standard env)
unset SCENARIO_RESET RESTORE_CHECKPOINT

# Warm-start from 103A@500 (retain INTERCEPTOR scenario knowledge as prior)
export WARMSTART_CHECKPOINT=/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_78989_00000_0_2026-04-22_02-28-27/checkpoint_000500/checkpoint-500

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-199}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=103A_refined_stone_layered_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r103Aref_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# KEY FIX: see baseline 70% of time (vs v1's 0%)
export BASELINE_PROB=0.7

# INTERCEPTOR-flavored reward (lower weights) + v2 base shape
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001
export SHAPING_BALL_PROGRESS=0.01       # v2 base
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.02  # halved from v1's 0.05 (less aggressive)
export SHAPING_POSSESSION_BONUS=0.002   # v2 base
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_EVENT_SHOT_REWARD=0.0
export SHAPING_EVENT_TACKLE_REWARD=0.05  # halved from v1's 0.10
export SHAPING_EVENT_CLEARANCE_REWARD=0.03  # reduced from v1's 0.05
export SHAPING_EVENT_COOLDOWN_STEPS=5

export TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/103A_refined_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
