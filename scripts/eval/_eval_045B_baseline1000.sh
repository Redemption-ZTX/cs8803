#!/usr/bin/env bash
# 045B baseline 1000ep eval — top 10% fallback (6 ckpts: 100/110/120/140/150/160)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/045B_baseline1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/045B_learned_only_on_031A1040_512x512_20260419_095729/TeamVsBaselineShapingPPOTrainer_Soccer_c0f88_00000_0_2026-04-19_09-57-50

CKPTS=()
for it in 100 110 120 140 150 160; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponents baseline \
    -n 1000 \
    -j 6 \
    --base-port 56305 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/045B_baseline1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/045B_baseline1000.log
