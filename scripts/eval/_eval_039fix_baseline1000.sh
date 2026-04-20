#!/usr/bin/env bash
# 039fix baseline 1000ep eval — top 5% + ties + ±1 (16 ckpts)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/039fix_baseline1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T1=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/039fix_mappo_airl_adaptive_on_029B190_formal/MAPPOVsBaselineTrainer_Soccer_0a654_00000_0_2026-04-19_04-09-08
T2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/039fix_mappo_airl_adaptive_on_029B190_formal/039fix_mappo_airl_adaptive_on_029B190_resume140__MAPPOVsBaselineTrainer_Soccer_c1aaa_00000_0_2026-04-19_05-47-19

CKPTS=()
# Trial 1 (formal): 1-143
for it in 90 100 110 120 140; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T1}/checkpoint_${p}/checkpoint-${it}" )
done
# Trial 2 (resume): 141-300
for it in 150 160 170 180 190 230 240 250 270 280 290; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T2}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_shared_cc_agent \
    --opponents baseline \
    -n 1000 \
    -j 7 \
    --base-port 61005 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/039fix_baseline1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/039fix_baseline1000.log
