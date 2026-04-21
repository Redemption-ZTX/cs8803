#!/bin/bash
# 072 Pool C: Cross-axis 4-teacher ensemble distill (no 055 anchor — differentiates from Pool A)
# Teachers: 055v2@1000 (recursive) + 056D@1140 (HP) + 054M@1750 (MAT arch) + 062a@1220 (curriculum)
# Each represents a different "axis" that reached ~0.89-0.91 single-model.
# Student: 031B Siamese+cross-attn warm from 031B@80. Uniform 1/4 weighting for simplicity.
# See snapshot-072-pool-C-newcomer-frontier.md for full design + KL-conflict risk analysis.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

LANE_TAG=072_poolC
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
unset TEAM_OPPONENT_CHECKPOINT
unset LEARNED_REWARD_MODEL_PATH OUTCOME_PBRS_PREDICTOR_PATH
unset TEAM_TRANSFORMER TEAM_TRANSFORMER_MIN TEAM_TRANSFORMER_MHA TEAM_CROSS_AGENT_ATTN
unset TEAM_DISTILL_KL TEAM_DISTILL_TEACHER_CHECKPOINT TEAM_DISTILL_TEACHER_POLICY_ID

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PORT_SEED=${PORT_SEED:-57}  # 53=Pool A, 57 distinct
PORT_OFFSET=$(( (PORT_SEED % 60) * 50 ))

export RUN_NAME=072_poolC_cross_axis_distill_warm031B80_$(date +%Y%m%d_%H%M%S)
export RAY_ADDRESS_OVERRIDE=local
export RAY_SESSION_TMPDIR_OVERRIDE=/tmp/r072_${PORT_SEED}
export NUM_GPUS=1 NUM_WORKERS=8 NUM_ENVS_PER_WORKER=5
export FCNET_HIDDENS=512,512

# Student arch = 031B Siamese + cross-attn
export TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
export TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

export WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# 3-teacher cross-axis ensemble (DROPPED 054M due to MAT arch incompat with _FrozenTeamEnsembleTeacher)
# 055v2@1000: recursive distill (Tier 2), combined 3000ep 0.909
# 056D@1140: HP sweep (lr=3e-4), 0.891 single-shot
# 062a@1220: curriculum + no-shape + adaptive phase gate, combined 2000ep 0.892
# NOTE: original 4-teacher design had 054M@1750 but its ckpt has _ca_* keys that
# _build_frozen_team_siamese_from_checkpoint rejects; MAT teacher needs code patch (deferred).
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001000/checkpoint-1000,/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140,/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220"
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.0
export TEAM_DISTILL_DECAY_UPDATES=8000
export TEAM_DISTILL_TEMPERATURE=1.0

export ROLLOUT_FRAGMENT_LENGTH=1000 TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048 NUM_SGD_ITER=4
export LR=0.0001 CLIP_PARAM=0.15
export BASELINE_PROB=1.0

# v2 shaping (same as 055/Pool A)
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

export TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

export EVAL_INTERVAL=10 EVAL_EPISODES=200
export EVAL_BASE_PORT=$((55505 + PORT_OFFSET))
export EVAL_MAX_STEPS=1500
export EVAL_OPPONENTS=baseline,random
export EVAL_TEAM0_MODULE=cs8803drl.deployment.trained_team_ray_agent
export EVAL_PYTHON_BIN=$PYTHON_BIN

export BASE_PORT=$((60605 + PORT_OFFSET))

LOG=docs/experiments/artifacts/slurm-logs/072_poolC_train_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] 072 Pool C launching: 4-teacher cross-axis, BASE_PORT=$BASE_PORT" >&2

$PYTHON_BIN -u -m cs8803drl.training.train_ray_team_vs_baseline_shaping 2>&1 | tee "$LOG"
