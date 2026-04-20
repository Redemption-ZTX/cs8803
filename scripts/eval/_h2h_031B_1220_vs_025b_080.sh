#!/usr/bin/env bash
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
M1=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220
M2=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80

PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="$M1" \
TRAINED_SHARED_CC_OPPONENT_CHECKPOINT="$M2" \
"$PY" scripts/eval/evaluate_headtohead.py \
    -m1 cs8803drl.deployment.trained_team_ray_agent \
    -m2 cs8803drl.deployment.trained_shared_cc_opponent_agent \
    -e 500 \
    -p 59205 \
    2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/031B_1220_vs_025b_080.log
