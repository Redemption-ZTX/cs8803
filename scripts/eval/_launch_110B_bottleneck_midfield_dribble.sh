#!/bin/bash
# 110B: State/action bottleneck specialist — MID-FIELD-DRIBBLE per Wang/Stone/Hanna 2025 (snapshot-110).
#
# Paradigm verification: train ONE specialist with bottleneck (obs masked + action constrained)
# to validate the wrapper infra + paradigm before scaling to 4 specialists.
#
# Recipe (snapshot-110 §2.4):
# (1) Warm + KL anchor from 101A@460 (Layer 1 ball-control specialist) — Stone Layered methodology
# (2) SCENARIO_RESET=dribble_subtask (existing 103C scenario)
# (3) OBS_BOTTLENECK_RAY_TYPES=0,4 — only ball (type 0) + opp_goal (type 4) rays kept
# (4) OBS_BOTTLENECK_RAY_INDICES=0,1,2,3,4,5,6,7,8,9,10 — only forward 11 of 14 rays
# (5) ACTION_BOTTLENECK_FREE_DIMS=2 — only kick dim varies
# (6) ACTION_BOTTLENECK_FIXED=0:2,1:1 — forward fixed=2 (always forward), rotate fixed=1 (no turn)
# (7) Reward = ball_progress 0.05 + possession 0.005 + small shot_reward 0.02 (allow occasional finish)
# (8) BASELINE_PROB=0.7 (BUG-1 fix automatic)
#
# Hypothesis: bottleneck specialist achieves sub-task success ≥0.85 in DRIBBLE scenario.
# If HIT → scale to 4 specialists + ensemble compose for ceiling-break attempt.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=110B_bottleneck_midfield_dribble
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

# Layer 1 specialist: 101A ball-control @ ckpt 460 (warm + KL anchor source)
CKPT_101A=/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04/checkpoint_000460/checkpoint-460

export WARMSTART_CHECKPOINT=$CKPT_101A

# Stone Layered KL anchor (BUG-2 fix lesson)
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_101A
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.005
export TEAM_DISTILL_DECAY_UPDATES=39000
export TEAM_DISTILL_TEMPERATURE=1.0

# Scenario: DRIBBLE (mid-field, my agent has ball, opp not immediately threatening)
export SCENARIO_RESET=dribble_subtask
export BASELINE_PROB=0.7

# *** SNAPSHOT-110 BOTTLENECK CONFIG ***
# Obs: keep only ball (type 0) + opp_goal (type 4) rays + only forward 11 ray indices
export OBS_BOTTLENECK_RAY_TYPES=0,4
export OBS_BOTTLENECK_RAY_INDICES=0,1,2,3,4,5,6,7,8,9,10
export OBS_BOTTLENECK_MASKED_DISTANCE=1.0   # "max far / not visible" sentinel

# Action: only kick (dim 2) free; forward fixed=2 (always forward), rotate fixed=1 (face fwd)
export ACTION_BOTTLENECK_FREE_DIMS=2
export ACTION_BOTTLENECK_FIXED=0:2,1:1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-227}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=110B_bottleneck_midfield_dribble_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r110B_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture: same as 101A (031B Siamese + cross-attn)
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# Reward: dribble + small shot allowance (per snapshot-110 §2.4)
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001
export SHAPING_BALL_PROGRESS=0.05            # main: dribble forward
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.0      # not defender, no penalty
export SHAPING_POSSESSION_BONUS=0.005        # encourage hold
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=1
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.0
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.0
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_EVENT_SHOT_REWARD=0.02         # small finish allowance
export SHAPING_EVENT_TACKLE_REWARD=0.0
export SHAPING_EVENT_CLEARANCE_REWARD=0.0
export SHAPING_EVENT_PASS_REWARD=0.0
export SHAPING_EVENT_COOLDOWN_STEPS=10

export TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/110B_bottleneck_midfield_dribble_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
