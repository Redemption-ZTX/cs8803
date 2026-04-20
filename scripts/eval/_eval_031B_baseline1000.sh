#!/usr/bin/env bash
# 031B baseline 1000ep eval — top 5% + ties + ±1 (24 ckpts)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/031B_baseline1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T1=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13
T2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38

CKPTS=()
for it in 420 430 440 500 510 520 580 590 600 680 690 700 740 750 760 860 870 880; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T1}/checkpoint_${p}/checkpoint-${it}" )
done
for it in 1090 1100 1110 1220 1230 1240; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T2}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponents baseline \
    -n 1000 \
    -j 7 \
    --base-port 51005 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/031B_baseline1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/031B_baseline1000.log
