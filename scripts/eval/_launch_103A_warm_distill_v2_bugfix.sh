#!/bin/bash
# 103A-warm-distill v2: Stone Layered L2 ULTIMATE rerun with TWO bug fixes.
#
# BUG-1 (per-episode opponent resample): fixed in core code 2026-04-22 — pool
# init now triggers TeamVsPolicyWrapper reset hook. v1 ran with each worker
# locked to one opponent for life; this run gets proper per-episode resample.
# (No launcher change needed — fix is in cs8803drl/core/utils.py.)
#
# BUG-2 (KL distill α schedule): v1 used DECAY_UPDATES=4000 → alpha decayed
# to 0 by iter ~53/500 (only 10% of training had anchor). v2 uses
# DECAY_UPDATES=39000 (covers ~500 iter ≈ full training) AND alpha_final=0.005
# (residual anchor throughout), so Stone Layered L2 anchor truly persists.
#
# Expected result if both bugs were load-bearing:
# - v1 result: combined 2000ep 0.914 / H2H vs 1750 0.496 (TIED)
# - v2 expected: > 0.92 baseline / H2H > 0.52 (positive direction at least)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=103A_warm_distill_v2_bugfix
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
unset RESTORE_CHECKPOINT

# 1750 SOTA path (anchor + warm + distill teacher)
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

# (1) Warm-start from 1750
export WARMSTART_CHECKPOINT=$CKPT_1750

# (2) BUG-2 FIX: KL distill α schedule extended to cover full training.
# Math: TRAIN_BATCH_SIZE=40000 / SGD_MINIBATCH_SIZE=2048 × NUM_SGD_ITER=4 ≈ 78 PPO updates per training iter.
# 500 training iter × 78 = 39000 updates total → DECAY_UPDATES=39000 spans full first run.
# alpha_final=0.005 keeps residual anchor (vs v1 alpha_final=0 hard zero).
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_1750
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.005          # v2: residual anchor (was 0.0 in v1)
export TEAM_DISTILL_DECAY_UPDATES=39000        # v2: covers 500 iter (was 4000 = 53 iter)
export TEAM_DISTILL_TEMPERATURE=1.0

# (3) Scenario init — focus learning on BALL_DUEL state distribution
export SCENARIO_RESET=interceptor_subtask

# (4) Mixed baseline+random opponent — fix distribution mismatch from v1
# BUG-1 FIX is automatic via core code change; this var is unchanged.
export BASELINE_PROB=0.7

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-217}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=103A_warm_distill_v2_bugfix_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r103Av2_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# (5) Light INTERCEPTOR aux reward + v2 base shape (same as v1)
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

# Per memory feedback_slurm_wall_budget.md: 4h conservative for fresh node (5037135 newly allocated)
export TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/103A_warm_distill_v2_bugfix_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
