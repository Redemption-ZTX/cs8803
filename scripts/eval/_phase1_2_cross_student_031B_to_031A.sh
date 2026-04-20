#!/usr/bin/env bash
# Phase 1.2: 031A as student + 031B (project SOTA 0.882) as teacher takeover.
# 3 conditions × 100ep on a SINGLE node.
# Question: stronger team-level teacher → larger Δ vs Phase 1 (036D teacher gave +1pp)?
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/cross-student-hybrid

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
S031A=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040
T031B=ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220

declare -A SPECS=(
  ["P12_C0_031A_solo"]="none 64401"
  ["P12_C1_031A_alpha_teacher031B"]="alpha 64421"
  ["P12_C2_031A_beta_teacher031B"]="beta 64441"
)

for cond in "${!SPECS[@]}"; do
  read -r TRG PORT <<< "${SPECS[$cond]}"
  LOG=docs/experiments/artifacts/cross-student-hybrid/${cond}.log
  JSON=docs/experiments/artifacts/cross-student-hybrid/${cond}.json
  if [ "$TRG" = "none" ]; then
    echo "[launch] $cond  trigger=none (student solo, will reuse Phase 1 C0 if seeded same; rerun for sample)  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_team_ray_agent \
      --student-checkpoint "$S031A" \
      --trigger "$TRG" --episodes 100 --base-port "$PORT" \
      --print-every 20 --json-out "$JSON" > "$LOG" 2>&1 &
  else
    echo "[launch] $cond  trigger=$TRG  teacher=031B@1220  port=$PORT"
    nohup "$PY" -u -m cs8803drl.evaluation.evaluate_hybrid \
      --student-module cs8803drl.deployment.trained_team_ray_agent \
      --student-checkpoint "$S031A" \
      --takeover-module cs8803drl.deployment.trained_team_ray_agent \
      --takeover-checkpoint "$T031B" \
      --trigger "$TRG" --episodes 100 --base-port "$PORT" \
      --print-every 20 --json-out "$JSON" > "$LOG" 2>&1 &
  fi
  echo "  PID=$!"
done
wait
echo "[done] Phase 1.2 (031B → 031A) all 3 complete"
