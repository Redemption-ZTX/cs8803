#!/bin/bash
# 111A: Stronger non-growing opponent test (snapshot-111 §5/§6 recommendation).
#
# Hypothesis: 0.91 baseline WR ceiling may be "ceiling vs baseline opponent specifically",
# not "architecture-imposed". If we train vs STRONGER fixed opponent (1750 SOTA), policy
# must learn harder → may push baseline WR > 0.91 at eval time.
#
# Recipe:
# (1) Warm from 1750 (start at known-good 0.9066 baseline)
# (2) KL distill anchor from 1750 (BUG-2 fixed: DECAY_UPDATES=39000)
# (3) TEAM_OPPONENT_CHECKPOINT=1750 (full game vs frozen 1750 — much stronger than ceia_baseline_agent)
# (4) NO scenario_reset (standard 2v2 game, just stronger opp)
# (5) Standard v2 reward
# (6) Inline eval every 10 iter on baseline (track baseline WR even while training vs 1750)
#
# Decision rule (after 250 iter / ~3h):
# - inline baseline WR ≥ 0.92 + H2H vs 1750 ≥ 0.55 → opp-weakness was binding (scale up)
# - inline baseline WR ∈ [0.88, 0.92] + H2H ≈ 0.50 → ceiling holds (drop axis)
# - inline baseline WR < 0.88 → catastrophic forgetting (anchor failed to hold)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=111A_strong_opp_test
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
unset LEARNED_REWARD_MODEL_PATH OUTCOME_PBRS_PREDICTOR_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_TEACHER_POLICY_ID
unset TEAM_SIAMESE_TWO_STREAM TEAM_SIAMESE_PER_RAY_ATTN TEAM_SIAMESE_VDN
unset RESTORE_CHECKPOINT
unset OBS_BOTTLENECK_RAY_TYPES OBS_BOTTLENECK_RAY_INDICES OBS_BOTTLENECK_MASKED_DISTANCE
unset ACTION_BOTTLENECK_FREE_DIMS ACTION_BOTTLENECK_FIXED
unset SCENARIO_RESET

CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

# (1) Warm from 1750
export WARMSTART_CHECKPOINT=$CKPT_1750

# (2) KL distill anchor from 1750 (BUG-2 fix throughout 250 iter)
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_1750
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.005
export TEAM_DISTILL_DECAY_UPDATES=19500   # 250 iter × 78 PPO updates ≈ 19500 (covers full)
export TEAM_DISTILL_TEMPERATURE=1.0

# (3) STRONGER OPPONENT: 1750 itself as frozen team opponent
# (overrides BASELINE_PROB; uses team-level frozen policy via FrozenTeamPolicy)
export TEAM_OPPONENT_CHECKPOINT=$CKPT_1750

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-257}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=111A_strong_opp_test_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r111A_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# Standard v2 reward
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001
export SHAPING_BALL_PROGRESS=0.01
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.01
export SHAPING_POSSESSION_BONUS=0.002
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_EVENT_SHOT_REWARD=0.0
export SHAPING_EVENT_TACKLE_REWARD=0.0
export SHAPING_EVENT_CLEARANCE_REWARD=0.0
export SHAPING_EVENT_PASS_REWARD=0.0
export SHAPING_EVENT_COOLDOWN_STEPS=10

# Budget: 250 iter ~3h decisive test
export TIMESTEPS_TOTAL=10000000 MAX_ITERATIONS=250 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

# Inline eval STILL ON baseline (track ceiling-vs-baseline transfer)
export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/111A_strong_opp_test_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
