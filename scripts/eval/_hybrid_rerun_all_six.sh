#!/usr/bin/env bash
# Hybrid eval rerun — 6 conditions parallel on a SINGLE node, with -u (unbuffered).
# New artifact dir: hybrid-eval-rerun/ to not overwrite the original 010-30-0 run.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/hybrid-eval-rerun

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
S031A=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
S036D=ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150

declare -A SPECS=(
  ["C0_031A_none"]="cs8803drl.deployment.trained_team_ray_agent $S031A none 53401"
  ["C1_031A_alpha"]="cs8803drl.deployment.trained_team_ray_agent $S031A alpha 53421"
  ["C2_031A_beta"]="cs8803drl.deployment.trained_team_ray_agent $S031A beta 53441"
  ["C0_036D_none"]="cs8803drl.deployment.trained_shared_cc_agent $S036D none 53461"
  ["C3_036D_alpha"]="cs8803drl.deployment.trained_shared_cc_agent $S036D alpha 53481"
  ["C4_036D_beta"]="cs8803drl.deployment.trained_shared_cc_agent $S036D beta 53501"
)

for cond in "${!SPECS[@]}"; do
  read -r MOD CKPT TRG PORT <<< "${SPECS[$cond]}"
  LOG=docs/experiments/artifacts/hybrid-eval-rerun/${cond}.log
  JSON=docs/experiments/artifacts/hybrid-eval-rerun/${cond}.json
  echo "[launch] $cond  mod=$MOD trigger=$TRG port=$PORT"
  nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
    --student-module "$MOD" \
    --student-checkpoint "$CKPT" \
    --trigger "$TRG" \
    --episodes 1000 \
    --base-port "$PORT" \
    --print-every 100 \
    --json-out "$JSON" \
    > "$LOG" 2>&1 &
  echo "  PID=$!"
done

wait
echo "[done] all 6 conditions complete"
