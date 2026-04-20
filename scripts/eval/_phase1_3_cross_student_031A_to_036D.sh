#!/usr/bin/env bash
# Phase 1.3: 036D as student + 031A as teacher takeover.
# 036D's main failure mode = defensive_pin (55%). Trigger fires in defensive_pin states.
# 031A's failure mode = wasted_possession → 031A is presumably stronger at handling defensive_pin.
# This is the theoretically most promising cross-student pair (different failure modes).
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/cross-student-hybrid

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
S036D=ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150
T031A=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040

declare -A SPECS=(
  ["P13_C0_036D_solo"]="none 65401"
  ["P13_C1_036D_alpha_teacher031A"]="alpha 65421"
  ["P13_C2_036D_beta_teacher031A"]="beta 65441"
)

for cond in "${!SPECS[@]}"; do
  read -r TRG PORT <<< "${SPECS[$cond]}"
  LOG=docs/experiments/artifacts/cross-student-hybrid/${cond}.log
  JSON=docs/experiments/artifacts/cross-student-hybrid/${cond}.json
  if [ "$TRG" = "none" ]; then
    echo "[launch] $cond  trigger=none (036D solo)  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_shared_cc_agent \
      --student-checkpoint "$S036D" \
      --trigger "$TRG" --episodes 100 --base-port "$PORT" \
      --print-every 20 --json-out "$JSON" > "$LOG" 2>&1 &
  else
    echo "[launch] $cond  trigger=$TRG  teacher=031A@1040  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_shared_cc_agent \
      --student-checkpoint "$S036D" \
      --takeover-module cs8803drl.deployment.trained_team_ray_agent \
      --takeover-checkpoint "$T031A" \
      --trigger "$TRG" --episodes 100 --base-port "$PORT" \
      --print-every 20 --json-out "$JSON" > "$LOG" 2>&1 &
  fi
  echo "  PID=$!"
done
wait
echo "[done] Phase 1.3 (031A → 036D) all 3 complete"
