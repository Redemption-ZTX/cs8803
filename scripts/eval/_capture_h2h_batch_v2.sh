#!/bin/bash
# v2: 8-model H2H + 034E ensemble + 043A' inclusion. C(8,2)=28 cross + 2 self-play = 30 pairs.
# --save-mode all (W+L 都存). New save_dir: failure-cases/h2h_v3_all_v2/
# Usage: bash _capture_h2h_batch_v2.sh <NODE_INDEX 1..4>
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
NODE_IDX=${1:-1}

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
mkdir -p docs/experiments/artifacts/failure-cases/h2h_v3_all_v2 docs/experiments/artifacts/official-evals/failure-capture-logs

# Frontier checkpoints
T051A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/051A_combo_on_031B_with_051reward_512x512_20260419_110852/TeamVsBaselineShapingPPOTrainer_Soccer_b9914_00000_0_2026-04-19_11-09-13/checkpoint_000130/checkpoint-130
T031B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220
T031A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T028A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060
T036D=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150
T029B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190
T043A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/043Aprime_warm_vs_frontier_pool_031B_formal/TeamVsBaselineShapingPPOTrainer_Soccer_06257_00000_0_2026-04-19_11-25-40/checkpoint_000080/checkpoint-80

# Module mapping for ckpt-based models
TEAM=cs8803drl.deployment.trained_team_ray_agent
TEAMOPP=cs8803drl.deployment.trained_team_ray_opponent_agent
PERAGENT=cs8803drl.deployment.trained_shared_cc_agent
PERAGENTOPP=cs8803drl.deployment.trained_shared_cc_opponent_agent
# Ensemble module (loads internally, no env var needed). 034E always TEAM0 in ensemble pairs.
ENSEMBLE034E=agents.v034e_frontier_031B_3way.agent

# 30 pairs: NAME T0_KIND T0_CKPT T1_KIND T1_CKPT
# T0_KIND in {team_ray, shared_cc, ensemble}; ensemble has CKPT="" (loads internally)
declare -a PAIRS=(
  "p01_051A_vs_031B  team_ray $T051A team_ray $T031B"
  "p02_051A_vs_031A  team_ray $T051A team_ray $T031A"
  "p03_051A_vs_028A  team_ray $T051A team_ray $T028A"
  "p04_051A_vs_036D  team_ray $T051A shared_cc $T036D"
  "p05_051A_vs_029B  team_ray $T051A shared_cc $T029B"
  "p06_051A_vs_043A  team_ray $T051A team_ray $T043A"
  "p07_031B_vs_031A  team_ray $T031B team_ray $T031A"
  "p08_031B_vs_028A  team_ray $T031B team_ray $T028A"
  "p09_031B_vs_036D  team_ray $T031B shared_cc $T036D"
  "p10_031B_vs_029B  team_ray $T031B shared_cc $T029B"
  "p11_031B_vs_043A  team_ray $T031B team_ray $T043A"
  "p12_031A_vs_028A  team_ray $T031A team_ray $T028A"
  "p13_031A_vs_036D  team_ray $T031A shared_cc $T036D"
  "p14_031A_vs_029B  team_ray $T031A shared_cc $T029B"
  "p15_031A_vs_043A  team_ray $T031A team_ray $T043A"
  "p16_028A_vs_036D  team_ray $T028A shared_cc $T036D"
  "p17_028A_vs_029B  team_ray $T028A shared_cc $T029B"
  "p18_028A_vs_043A  team_ray $T028A team_ray $T043A"
  "p19_036D_vs_029B  shared_cc $T036D shared_cc $T029B"
  "p20_036D_vs_043A  shared_cc $T036D team_ray $T043A"
  "p21_029B_vs_043A  shared_cc $T029B team_ray $T043A"
  "p22_034E_vs_051A  ensemble NONE team_ray $T051A"
  "p23_034E_vs_031B  ensemble NONE team_ray $T031B"
  "p24_034E_vs_031A  ensemble NONE team_ray $T031A"
  "p25_034E_vs_028A  ensemble NONE team_ray $T028A"
  "p26_034E_vs_036D  ensemble NONE shared_cc $T036D"
  "p27_034E_vs_029B  ensemble NONE shared_cc $T029B"
  "p28_034E_vs_043A  ensemble NONE team_ray $T043A"
  "p29_031B_vs_031B_selfplay  team_ray $T031B team_ray $T031B"
  "p30_031A_vs_031A_selfplay  team_ray $T031A team_ray $T031A"
)

TOTAL=${#PAIRS[@]}
echo "[node=$NODE_IDX] total pairs in queue: $TOTAL (this node will run $((TOTAL/4 + 1)))"

for i in "${!PAIRS[@]}"; do
  if (( (i % 8) + 1 != NODE_IDX )); then continue; fi
  IFS=' ' read -ra P <<< "${PAIRS[$i]}"
  NAME=${P[0]}
  T0_KIND=${P[1]}
  T0_CKPT=${P[2]}
  T1_KIND=${P[3]}
  T1_CKPT=${P[4]}
  PORT=$((59000 + i * 7))  # +7 spacing to leave room for NUM_ENVS=5

  case $T0_KIND in
    team_ray)  M0_MOD=$TEAM ;;
    shared_cc) M0_MOD=$PERAGENT ;;
    ensemble)  M0_MOD=$ENSEMBLE034E ;;
  esac
  case $T1_KIND in
    team_ray)  M1_MOD=$TEAMOPP; M1_ENV=TRAINED_TEAM_OPPONENT_CHECKPOINT ;;
    shared_cc) M1_MOD=$PERAGENTOPP; M1_ENV=TRAINED_SHARED_CC_OPPONENT_CHECKPOINT ;;
  esac

  SAVE_DIR=docs/experiments/artifacts/failure-cases/h2h_v3_all_v2/${NAME}_n500
  LOG=docs/experiments/artifacts/official-evals/failure-capture-logs/v2_${NAME}.log
  mkdir -p "$SAVE_DIR"

  # Build env + checkpoint args
  ENV_ARGS=()
  CKPT_ARGS=()
  if [ "$T0_KIND" != "ensemble" ]; then
    ENV_ARGS+=("TRAINED_RAY_CHECKPOINT=$T0_CKPT")
    CKPT_ARGS+=(--checkpoint "$T0_CKPT")
  fi
  ENV_ARGS+=("$M1_ENV=$T1_CKPT")

  echo "[node=$NODE_IDX] launching $NAME (port=$PORT) m0=$M0_MOD m1=$M1_MOD"
  env "${ENV_ARGS[@]}" \
    "$PY" scripts/eval/capture_failure_cases.py \
    "${CKPT_ARGS[@]}" \
    --team0-module "$M0_MOD" \
    --opponent "$M1_MOD" \
    -n 500 \
    --max-steps 1500 \
    --base-port "$PORT" \
    --save-dir "$SAVE_DIR" \
    --save-mode all \
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
