#!/bin/bash
# Migrate 3 v3all pairs (p14/p21/p28) abandoned on slow node 7
# Usage: bash _migrate_v3all_3pairs.sh <PAIR_NAME>  (one per call)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PAIR=${1:?need pair name like p14}
PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
SAVE_BASE=docs/experiments/artifacts/trajectories/v3_all_30pair

T031A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T029B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190
T043A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/043Aprime_warm_vs_frontier_pool_031B_formal/TeamVsBaselineShapingPPOTrainer_Soccer_06257_00000_0_2026-04-19_11-25-40/checkpoint_000080/checkpoint-80

TEAM=cs8803drl.deployment.trained_team_ray_agent
TEAMOPP=cs8803drl.deployment.trained_team_ray_opponent_agent
PERAGENT=cs8803drl.deployment.trained_shared_cc_agent
PERAGENTOPP=cs8803drl.deployment.trained_shared_cc_opponent_agent
ENSEMBLE034E=agents.v034e_frontier_031B_3way.agent

case "$PAIR" in
  p14)
    NAME=p14_031A_vs_029B
    PORT=62100
    M0_MOD=$TEAM; T0_CKPT=$T031A
    M1_MOD=$PERAGENTOPP; T1_CKPT=$T029B; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
    ENV_T0="TRAINED_RAY_CHECKPOINT=$T0_CKPT"
    ;;
  p21)
    NAME=p21_029B_vs_043A
    PORT=62200
    M0_MOD=$PERAGENT; T0_CKPT=$T029B
    M1_MOD=$TEAMOPP; T1_CKPT=$T043A; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ENV_T0="TRAINED_RAY_CHECKPOINT=$T0_CKPT"
    ;;
  p28)
    NAME=p28_034E_vs_043A
    PORT=62300
    M0_MOD=$ENSEMBLE034E; T0_CKPT=NONE
    M1_MOD=$TEAMOPP; T1_CKPT=$T043A; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
    ENV_T0=""  # no env var for ensemble
    ;;
  *)
    echo "ERROR: unknown PAIR=$PAIR (use p14|p21|p28)" >&2; exit 1 ;;
esac

SAVE_DIR=$SAVE_BASE/$NAME
LOG=docs/experiments/artifacts/official-evals/failure-capture-logs/dump_v3all_${NAME}_migrated.log
mkdir -p "$SAVE_DIR"

ENV_ARGS=("PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}" "$M1_ENV=$T1_CKPT")
[ -n "$ENV_T0" ] && ENV_ARGS+=("$ENV_T0")

echo "[migrate] $NAME on port $PORT"
env "${ENV_ARGS[@]}" \
  "$PY" -u scripts/eval/dump_trajectories.py \
  --team0-module "$M0_MOD" \
  --team1-module "$M1_MOD" \
  --episodes 500 \
  --max-steps 1500 \
  --base-port "$PORT" \
  --save-dir "$SAVE_DIR" \
  --filename-prefix "${NAME}_" \
  2>&1 | tee "$LOG"
