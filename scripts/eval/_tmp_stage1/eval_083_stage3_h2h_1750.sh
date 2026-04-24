#!/bin/bash
# 083 Stage 3 — H2H 083@ckpt-1000 vs 1750 SOTA (n=500).
# Direct comparison; if 083 wins > 0.55 z=2.24 sig p<0.05 → real SOTA shift from arch axis.
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead

CKPT_083=/storage/ice1/5/1/wsun377/ray_results_scratch/083_per_ray_attention_scratch_20260421_210849/TeamVsBaselineShapingPPOTrainer_Soccer_decf1_00000_0_2026-04-21_21-09-11/checkpoint_001000/checkpoint-1000
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750

PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=$CKPT_083 \
TRAINED_TEAM_OPPONENT_CHECKPOINT=$CKPT_1750 \
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 62705 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/083_1000_vs_1750.log
