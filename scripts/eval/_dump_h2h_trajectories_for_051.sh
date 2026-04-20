#!/usr/bin/env bash
# snapshot-051 Stage 0a: dump per-step H2H trajectories for strong-vs-strong
# learned reward training. 5 pairs × 200ep each (single direction; sample
# diversity comes from pair diversity, not from running both directions).
#
# Usage on a free GPU node:
#   bash scripts/eval/_dump_h2h_trajectories_for_051.sh <pair_index>
#
# Where pair_index ∈ {1..5}. This lets us run multiple pairs in parallel on
# different nodes (one pair per node, ~10-20 min each at 200ep).

set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PAIR="${1:-1}"
PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
SAVE_BASE=docs/experiments/artifacts/trajectories/051_strong_vs_strong
mkdir -p "$SAVE_BASE"

# Frontier checkpoints
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
  1)  # 031B vs 031A — same team-level family
    PAIR_NAME=031B_vs_031A
    PORT=58701
    M0_MOD=$TEAM; M0_CKPT=$T031B
    M1_MOD=$TEAMOPP; M1_CKPT=$T031A
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ;;
  2)  # 036D vs 029B — per-agent intra (cross-reward)
    PAIR_NAME=036D_vs_029B
    PORT=58721
    M0_MOD=$PERAGENT; M0_CKPT=$T036D
    M1_MOD=$PERAGENTOPP; M1_CKPT=$T029B
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ;;
  3)  # 025b vs 029B — per-agent intra (BC vs PBRS-handoff)
    PAIR_NAME=025b_vs_029B
    PORT=58741
    M0_MOD=$PERAGENT; M0_CKPT=$T025B
    M1_MOD=$PERAGENTOPP; M1_CKPT=$T029B
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ;;
  4)  # 036D vs 031A — failure mode complementary (defensive_pin vs wasted_possession)
    PAIR_NAME=036D_vs_031A
    PORT=58761
    M0_MOD=$PERAGENT; M0_CKPT=$T036D
    M1_MOD=$TEAMOPP; M1_CKPT=$T031A
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ;;
  5)  # 031A vs 028A — architecture step (Siamese vs flat MLP team-level)
    PAIR_NAME=031A_vs_028A
    PORT=58781
    M0_MOD=$TEAM; M0_CKPT=$T031A
    M1_MOD=$TEAMOPP; M1_CKPT=$T028A
    M0_ENV=TRAINED_RAY_CHECKPOINT; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ;;
  *)
    echo "ERROR: unknown PAIR=$PAIR (use 1..5)" >&2
    exit 1 ;;
esac

SAVE_DIR=$SAVE_BASE/$PAIR_NAME
mkdir -p "$SAVE_DIR"
LOG=$SAVE_BASE/${PAIR_NAME}.log

echo "[051-stage0] PAIR=$PAIR_NAME  m0=$M0_MOD($M0_CKPT)  m1=$M1_MOD($M1_CKPT)  port=$PORT"

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
