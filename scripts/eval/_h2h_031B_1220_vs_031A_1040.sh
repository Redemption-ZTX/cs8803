#!/usr/bin/env bash
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
M1=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220
M2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040

PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="$M1" \
TRAINED_TEAM_OPPONENT_CHECKPOINT="$M2" \
"$PY" scripts/eval/evaluate_headtohead.py \
    -m1 cs8803drl.deployment.trained_team_ray_agent \
    -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
    -e 500 \
    -p 55205 \
    2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/031B_1220_vs_031A_1040.log
