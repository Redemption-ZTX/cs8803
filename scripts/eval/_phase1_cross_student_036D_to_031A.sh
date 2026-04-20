#!/usr/bin/env bash
# Phase 1 sanity: 031A as student + 036D as teacher takeover.
# 3 conditions × 100ep on a SINGLE node (-u for unbuffered).
# Goal: see if Δ vs 031A_solo ≥ +0.02 to justify Phase 2 (real DAGGER training).
#
# Trigger semantic: window-based ball-in-our-half (alpha/beta from snapshot-048 §2.2).
# 031A solo (C0) baseline = 0.880@100ep from snapshot-048 §7.2.
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/cross-student-hybrid

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
S031A=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T036D=ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150

declare -A SPECS=(
  ["C0_031A_solo"]="none 63401"
  ["C1_031A_alpha_teacher036D"]="alpha 63421"
  ["C2_031A_beta_teacher036D"]="beta 63441"
)

for cond in "${!SPECS[@]}"; do
  read -r TRG PORT <<< "${SPECS[$cond]}"
  LOG=docs/experiments/artifacts/cross-student-hybrid/${cond}.log
  JSON=docs/experiments/artifacts/cross-student-hybrid/${cond}.json

  if [ "$TRG" = "none" ]; then
    # C0 = student solo, no takeover (skip teacher flags)
    echo "[launch] $cond  trigger=none (student solo)  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_team_ray_agent \
      --student-checkpoint "$S031A" \
      --trigger "$TRG" \
      --episodes 100 \
      --base-port "$PORT" \
      --print-every 20 \
      --json-out "$JSON" \
      > "$LOG" 2>&1 &
  else
    echo "[launch] $cond  trigger=$TRG  teacher=036D  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_team_ray_agent \
      --student-checkpoint "$S031A" \
      --takeover-module cs8803drl.deployment.trained_shared_cc_agent \
      --takeover-checkpoint "$T036D" \
      --trigger "$TRG" \
      --episodes 100 \
      --base-port "$PORT" \
      --print-every 20 \
      --json-out "$JSON" \
      > "$LOG" 2>&1 &
  fi
  echo "  PID=$!"
done

wait
echo "[done] all 3 conditions complete"
