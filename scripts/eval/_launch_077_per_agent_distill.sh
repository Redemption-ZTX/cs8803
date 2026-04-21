#!/bin/bash
# 077 DIR-B: Per-agent shared-policy student distilled from team-level
# teacher ensemble {031B@1220 + 045A@180 + 051A@130} — snapshot-077.
# Hypothesis: per-agent slot-symmetric rollout (2x sample efficiency) +
# cross-arch distill probes whether the current ~0.91 plateau is an arch
# bottleneck of the team-level Siamese+cross-attn path.
# See docs/experiments/snapshot-077-per-agent-student-distill.md.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

# Running-flag convention (auto-manage .running / .done)
LANE_TAG=077
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

# Clean slate: unset all conflicting env vars that sibling lanes might set.
unset RESTORE_CHECKPOINT BC_WARMSTART_CHECKPOINT TEAMMATE_CHECKPOINT TEAMMATE_BASE_CHECKPOINT
unset AUX_TEAM_ACTION_HEAD AUX_TEAM_ACTION_WEIGHT AUX_TEAM_ACTION_HIDDEN
unset WARMSTART_CHECKPOINT TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH OUTCOME_PBRS_PREDICTOR_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_TEACHER_POLICY_ID
unset TEAM_DISTILL_ENSEMBLE_KL TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS
unset AUX_TEAMMATE_HEAD AUX_TEAMMATE_WEIGHT AUX_TEAMMATE_HIDDEN

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Hard-coded PORT_SEED — far from active lanes (076=76 / 070-073 Pool / 068=68)
PORT_SEED=${PORT_SEED:-77}
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=077_per_agent_distill_from_034E_ensemble_scratch_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r077_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5

# Per-agent FC hiddens — matches 029B/036D (per-agent SOTA precedent)
export FCNET_HIDDENS=512,512

# Per-agent student distill from 034E 3-teacher team-level ensemble
export PER_AGENT_STUDENT_DISTILL=1
export PER_AGENT_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_resume1080/TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220,/home/hice1/wsun377/Desktop/cs8803drl/ray_results/045A_team_combo_on_031A1040_formal_rerun1/TeamVsBaselineShapingPPOTrainer_Soccer_be409_00000_0_2026-04-19_05-47-13/checkpoint_000180/checkpoint-180,/home/hice1/wsun377/Desktop/cs8803drl/ray_results/051A_combo_on_031B_with_051reward_512x512_20260419_110852/TeamVsBaselineShapingPPOTrainer_Soccer_b9914_00000_0_2026-04-19_11-09-13/checkpoint_000130/checkpoint-130"
export PER_AGENT_DISTILL_ALPHA_INIT=0.05      # per snapshot-077 §2.3, matches 055
export PER_AGENT_DISTILL_ALPHA_FINAL=0.0
export PER_AGENT_DISTILL_DECAY_UPDATES=8000
export PER_AGENT_DISTILL_TEMPERATURE=1.0

# Rollout / SGD config — aligned with 055 for isolation
export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15

# v2 shaping (same as 055 / 031B)
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

# Budget: match 055 (1250 iter scratch, ~12h on H100)
export TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_shared_cc_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((55605 + PORT_OFFSET))
export LOG_LEVEL=INFO LOG_SYS_USAGE=0

mkdir -p docs/experiments/artifacts/slurm-logs
LOG=docs/experiments/artifacts/slurm-logs/077_per_agent_distill_train_$(date +%Y%m%d_%H%M%S).log
$PYTHON_BIN -u -m cs8803drl.training.train_ray_mappo_vs_baseline 2>&1 | tee "$LOG"
