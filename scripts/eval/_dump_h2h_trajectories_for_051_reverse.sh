#!/usr/bin/env bash
# snapshot-051 Stage 0a REVERSE direction: same 5 pairs but team0/team1 swapped.
# Goal: get BOTH agents' winning trajectories as positive reward samples.
# Forward (e.g. team0=031B vs team1=031A) gives only 031B's wins as W class.
# Reverse (team0=031A vs team1=031B) gives 031A's wins as W class.
# Combined → reward model learns "what winning-against-strong-opp looks like" from BOTH sides.
#
# Usage: bash scripts/eval/_dump_h2h_trajectories_for_051_reverse.sh <pair_index>
# pair_index ∈ {1..5}

set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PAIR="${1:-1}"
PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
SAVE_BASE=docs/experiments/artifacts/trajectories/051_strong_vs_strong
mkdir -p "$SAVE_BASE"

T031A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T031B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220
T036D=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150
T029B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190
T025B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80
T028A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060

TEAM=cs8803drl.deployment.trained_team_ray_agent
TEAMOPP=cs8803drl.deployment.trained_team_ray_opponent_agent
PERAGENT=cs8803drl.deployment.trained_shared_cc_agent
PERAGENTOPP=cs8803drl.deployment.trained_shared_cc_opponent_agent

case "$PAIR" in
  1)  # REVERSE of pair 1: team0=031A vs team1=031B
    PAIR_NAME=031A_vs_031B_REV
    PORT=58702
    M0_MOD=$TEAM; M0_CKPT=$T031A
    M1_MOD=$TEAMOPP; M1_CKPT=$T031B
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ;;
  2)  # REVERSE of pair 2: team0=029B vs team1=036D
    PAIR_NAME=029B_vs_036D_REV
    PORT=58722
    M0_MOD=$PERAGENT; M0_CKPT=$T029B
    M1_MOD=$PERAGENTOPP; M1_CKPT=$T036D
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ;;
  3)  # REVERSE of pair 3: team0=029B vs team1=025b
    PAIR_NAME=029B_vs_025b_REV
    PORT=58742
    M0_MOD=$PERAGENT; M0_CKPT=$T029B
    M1_MOD=$PERAGENTOPP; M1_CKPT=$T025B
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ;;
  4)  # REVERSE of pair 4: team0=031A vs team1=036D
    PAIR_NAME=031A_vs_036D_REV
    PORT=58762
    M0_MOD=$TEAM; M0_CKPT=$T031A
    M1_MOD=$PERAGENTOPP; M1_CKPT=$T036D
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ;;
  5)  # REVERSE of pair 5: team0=028A vs team1=031A
    PAIR_NAME=028A_vs_031A_REV
    PORT=58782
    M0_MOD=$TEAM; M0_CKPT=$T028A
    M1_MOD=$TEAMOPP; M1_CKPT=$T031A
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ;;
  *)
    echo "ERROR: unknown PAIR=$PAIR (use 1..5)" >&2
    exit 1 ;;
esac

SAVE_DIR=$SAVE_BASE/$PAIR_NAME
mkdir -p "$SAVE_DIR"
LOG=$SAVE_BASE/${PAIR_NAME}.log

echo "[051-stage0-REV] PAIR=$PAIR_NAME  m0=$M0_MOD($M0_CKPT)  m1=$M1_MOD($M1_CKPT)  port=$PORT"

env \
  "PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$M0_ENV=$M0_CKPT" \
  "$M1_ENV=$M1_CKPT" \
  "$PY" -u scripts/eval/dump_trajectories.py \
    --team0-module "$M0_MOD" \
    --team1-module "$M1_MOD" \
    --episodes 200 \
    --max-steps 1500 \
    --base-port "$PORT" \
    --save-dir "$SAVE_DIR" \
    --filename-prefix "${PAIR_NAME}_" \
    2>&1 | tee "$LOG"
