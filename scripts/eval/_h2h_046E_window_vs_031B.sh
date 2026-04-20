#!/bin/bash
# H2H: 046E@<ITER> vs 031B@1220 (the training opponent), 500ep
# Tests if 046E learned to beat its training opp at each training window
# Usage: bash _h2h_046E_window_vs_031B.sh <ITER> <PORT>
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
ITER=${1:?need iter}
PORT=${2:?need port}

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T046E=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/046E_scratch_vs_frozen_031B_cross_attention_512x512_20260419_095358/TeamVsBaselineShapingPPOTrainer_Soccer_4223e_00000_0_2026-04-19_09-54-17/checkpoint_$(printf "%06d" $ITER)/checkpoint-$ITER
T031B=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220

mkdir -p docs/experiments/artifacts/official-evals/headtohead

env "PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    "TRAINED_RAY_CHECKPOINT=$T046E" \
    "TRAINED_TEAM_OPPONENT_CHECKPOINT=$T031B" \
  "$PY" scripts/eval/evaluate_headtohead.py \
    -m1 cs8803drl.deployment.trained_team_ray_agent \
    -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
    -e 500 \
    -p "$PORT" \
    2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/046E_iter${ITER}_vs_031B_1220.log
