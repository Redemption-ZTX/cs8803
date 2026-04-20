#!/usr/bin/env bash
# 045A baseline 1000ep eval — top 10% fallback (3 → 8 ckpts)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/045A_baseline1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/045A_team_combo_on_031A1040_formal_rerun1/TeamVsBaselineShapingPPOTrainer_Soccer_be409_00000_0_2026-04-19_05-47-13

CKPTS=()
for it in 10 20 130 140 150 170 180 190; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponents baseline \
    -n 1000 \
    -j 7 \
    --base-port 62005 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/045A_baseline1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/045A_baseline1000.log
