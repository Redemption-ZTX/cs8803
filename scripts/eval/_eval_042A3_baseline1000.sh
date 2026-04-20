#!/usr/bin/env bash
# 042A3 baseline 1000ep eval — top 5% + ties + ±1 (5 ckpts)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/042A3_baseline1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T1=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/042A3_team_kl_distill_from_029B_on_031A1040_merged/042A3_team_kl_distill_from_029B_on_031A1040_formal__TeamVsBaselineShapingPPOTrainer_Soccer_49907_00000_0_2026-04-19_04-32-23
T2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/042A3_team_kl_distill_from_029B_on_031A1040_merged/042A3_team_kl_distill_from_029B_on_031A1040_resume170__TeamVsBaselineShapingPPOTrainer_Soccer_c6211_00000_0_2026-04-19_05-47-27

CKPTS=()
for it in 80 90 100; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T1}/checkpoint_${p}/checkpoint-${it}" )
done
for it in 180 190; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T2}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponents baseline \
    -n 1000 \
    -j 5 \
    --base-port 52005 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/042A3_baseline1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/042A3_baseline1000.log
