#!/bin/bash
# 104B: Stone Layered Learning Phase 2 — defender specialist variant (paradigm-generalize test #2, snapshot-107 §8.1).
#
# Companion to 104A (pass-decision). Tests if Stone Layered Phase 2 paradigm
# generalizes across MULTIPLE sub-tasks: pass + defender + (later) attack.
# If both 104A AND 104B succeed, Phase 2 paradigm strongly validated.
# If only one succeeds, paradigm is sub-task-dependent (which is itself a finding).
#
# Recipe (same as 104A, only scenario + reward changes):
# (1) WARMSTART_CHECKPOINT=101A@460 (Layer 1 ball-control specialist, 0.851 baseline-competent)
# (2) KL distill anchor from 101A@460 with α 0.05→0.005 over 39000 updates
# (3) SCENARIO_RESET=defender_subtask — ball own half, teammate has ball, our agent positions defensively
# (4) BASELINE_PROB=0.7 (BUG-1 fix automatic)
# (5) Defender-flavored aux reward: opp_progress_penalty 0.05, clearance event 0.05, NO pass reward (different specialty)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=104B_layered_p2_defender
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

# Layer 1 specialist: 101A ball-control @ ckpt 460 (Phase 1 verdict 0.851 baseline)
CKPT_101A=/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04/checkpoint_000460/checkpoint-460

# (1) Warm-start from 101A@460 (Layer 1 specialist)
export WARMSTART_CHECKPOINT=$CKPT_101A

# (2) KL distill anchor from 101A@460 — Stone L2 anchor throughout training (BUG-2 fix)
# Math: TRAIN_BATCH_SIZE=40000 / SGD_MINIBATCH_SIZE=2048 × NUM_SGD_ITER=4 ≈ 78 updates per iter
# 500 iter × 78 = 39000 updates → DECAY_UPDATES=39000 spans full training
# alpha_final=0.005 keeps residual anchor (not zero) per Stone L2 lessons
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_101A
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.005
export TEAM_DISTILL_DECAY_UPDATES=39000
export TEAM_DISTILL_TEMPERATURE=1.0

# (3) Scenario init — focus learning on defender state distribution
export SCENARIO_RESET=defender_subtask

# (4) Mixed opp distribution — BUG-1 fix in core makes per-episode resample work
export BASELINE_PROB=0.7

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-249}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=104B_layered_p2_defender_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r104B_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Architecture: same as 101A (031B Siamese + cross-attn) — required to match warm ckpt shape
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# (5) Reward — v2 base + DEFENDER specialty (opp_progress_penalty + clearance event)
export USE_REWARD_SHAPING=1
export SHAPING_FIELD_ROLE_BINDING=0
export SHAPING_TIME_PENALTY=0.001
export SHAPING_BALL_PROGRESS=0.01           # v2 base
export SHAPING_GOAL_PROXIMITY_SCALE=0.0
export SHAPING_OPP_PROGRESS_PENALTY=0.05    # 104B SPECIALTY: heavy opp progress penalty (defender)
export SHAPING_POSSESSION_BONUS=0.002       # v2 base
export SHAPING_POSSESSION_DIST=1.25
export SHAPING_PROGRESS_REQUIRES_POSSESSION=0
export SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.005   # 104B SPECIALTY: heavier deep-zone penalty
export SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.005
export SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=0 SHAPING_DEFENSIVE_SURVIVAL_BONUS=0
export SHAPING_FAST_LOSS_THRESHOLD_STEPS=0 SHAPING_FAST_LOSS_PENALTY_PER_STEP=0
export SHAPING_EVENT_SHOT_REWARD=0.0
export SHAPING_EVENT_TACKLE_REWARD=0.03     # 104B SPECIALTY: small tackle bonus
export SHAPING_EVENT_CLEARANCE_REWARD=0.05  # 104B SPECIALTY: main defender signal
export SHAPING_EVENT_PASS_REWARD=0.0        # 104B: NOT pass specialty
export SHAPING_EVENT_COOLDOWN_STEPS=10

# Per memory feedback_slurm_wall_budget.md: 4h conservative for fresh node
export TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((59505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((59605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

LOG=docs/experiments/artifacts/slurm-logs/104B_layered_p2_defender_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
