#!/usr/bin/env bash
# 031B random 1000ep verify — grading 第二条 (vs Random 9/10) 验证
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/031B_random1000

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38

# Top 3 baseline 1000ep peaks: 1220 (0.882), 1230 (0.881), 1240 (0.872)
CKPTS=()
for it in 1220 1230 1240; do
    p=$(printf "%06d" "$it")
    CKPTS+=( "--checkpoint" "${T2}/checkpoint_${p}/checkpoint-${it}" )
done

"$PY" scripts/eval/evaluate_official_suite_parallel.py \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponents random \
    -n 1000 \
    -j 3 \
    --base-port 58005 \
    --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/031B_random1000 \
    "${CKPTS[@]}" \
    2>&1 | tee docs/experiments/artifacts/official-evals/031B_random1000.log
