#!/bin/bash
# H2H peer-vs-peer failure capture batch — for v3 bucket dataset
# Usage: bash _capture_h2h_batch.sh <NODE_INDEX>
# NODE_INDEX in {1..4}, splits 17 pairs across 4 nodes
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
NODE_IDX=${1:-1}

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
mkdir -p docs/experiments/artifacts/failure-cases/h2h_v3 docs/experiments/artifacts/official-evals/failure-capture-logs

# Frontier checkpoints
T051A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/051A_combo_on_031B_with_051reward_512x512_20260419_110852/TeamVsBaselineShapingPPOTrainer_Soccer_b9914_00000_0_2026-04-19_11-09-13/checkpoint_000130/checkpoint-130
T031B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220
T031A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T028A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060
T036D=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150
T029B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190

# Module mapping
TEAM=cs8803drl.deployment.trained_team_ray_agent
TEAMOPP=cs8803drl.deployment.trained_team_ray_opponent_agent
PERAGENT=cs8803drl.deployment.trained_shared_cc_agent
PERAGENTOPP=cs8803drl.deployment.trained_shared_cc_opponent_agent

# Pair definitions: NAME T0_KIND T0_CKPT T1_KIND T1_CKPT
# T0_KIND in {team_ray, shared_cc}
# All 17 pairs: 15 cross + 2 self-play
declare -a PAIRS=(
  "p01_051A_vs_031B  team_ray $T051A team_ray $T031B"
  "p02_051A_vs_031A  team_ray $T051A team_ray $T031A"
  "p03_051A_vs_028A  team_ray $T051A team_ray $T028A"
  "p04_051A_vs_036D  team_ray $T051A shared_cc $T036D"
  "p05_051A_vs_029B  team_ray $T051A shared_cc $T029B"
  "p06_031B_vs_031A  team_ray $T031B team_ray $T031A"
  "p07_031B_vs_028A  team_ray $T031B team_ray $T028A"
  "p08_031B_vs_036D  team_ray $T031B shared_cc $T036D"
  "p09_031B_vs_029B  team_ray $T031B shared_cc $T029B"
  "p10_031A_vs_028A  team_ray $T031A team_ray $T028A"
  "p11_031A_vs_036D  team_ray $T031A shared_cc $T036D"
  "p12_031A_vs_029B  team_ray $T031A shared_cc $T029B"
  "p13_028A_vs_036D  team_ray $T028A shared_cc $T036D"
  "p14_028A_vs_029B  team_ray $T028A shared_cc $T029B"
  "p15_036D_vs_029B  shared_cc $T036D shared_cc $T029B"
  "p16_031B_vs_031B_selfplay  team_ray $T031B team_ray $T031B"
  "p17_031A_vs_031A_selfplay  team_ray $T031A team_ray $T031A"
)

TOTAL=${#PAIRS[@]}
# Split into 4 nodes: round-robin assignment
for i in "${!PAIRS[@]}"; do
  if (( (i % 4) + 1 != NODE_IDX )); then continue; fi
  IFS=' ' read -ra P <<< "${PAIRS[$i]}"
  NAME=${P[0]}
  T0_KIND=${P[1]}
  T0_CKPT=${P[2]}
  T1_KIND=${P[3]}
  T1_CKPT=${P[4]}
  PORT=$((60000 + i * 5))

  if [ "$T0_KIND" = "team_ray" ]; then
    M0_MOD=$TEAM
  else
    M0_MOD=$PERAGENT
  fi
  if [ "$T1_KIND" = "team_ray" ]; then
    M1_MOD=$TEAMOPP
    M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT
  else
    M1_MOD=$PERAGENTOPP
    M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT
  fi

  SAVE_DIR=docs/experiments/artifacts/failure-cases/h2h_v3/${NAME}_n500
  LOG=docs/experiments/artifacts/official-evals/failure-capture-logs/h2h_${NAME}.log
  mkdir -p "$SAVE_DIR"

  echo "[node=$NODE_IDX] launching $NAME (port=$PORT) m0=$M0_MOD m1=$M1_MOD"
  env "TRAINED_RAY_CHECKPOINT=$T0_CKPT" "$M1_ENV=$T1_CKPT" \
    "$PY" scripts/eval/capture_failure_cases.py \
    --checkpoint "$T0_CKPT" \
    --team0-module "$M0_MOD" \
    --opponent "$M1_MOD" \
    -n 500 \
    --max-steps 1500 \
    --base-port "$PORT" \
    --save-dir "$SAVE_DIR" \
    --save-mode losses \
    --max-saved-episodes 500 \
    --trace-stride 10 \
    --trace-tail-steps 30 \
    --reward-shaping-debug \
    --time-penalty 0.001 \
    --ball-progress-scale 0.01 \
    --goal-proximity-scale 0.0 \
    --progress-requires-possession 0 \
    --opponent-progress-penalty-scale 0.01 \
    --possession-dist 1.25 \
    --possession-bonus 0.002 \
    --deep-zone-outer-threshold -8 \
    --deep-zone-outer-penalty 0.003 \
    --deep-zone-inner-threshold -12 \
    --deep-zone-inner-penalty 0.003 \
    --defensive-survival-threshold 0 \
    --defensive-survival-bonus 0 \
    --fast-loss-threshold-steps 0 \
    --fast-loss-penalty-per-step 0 \
    --event-shot-reward 0.0 \
    --event-tackle-reward 0.0 \
    --event-clearance-reward 0.0 \
    --event-cooldown-steps 10 \
    2>&1 | tee "$LOG"
  echo "[node=$NODE_IDX] $NAME done"
done
echo "[node=$NODE_IDX] all assigned pairs done"
