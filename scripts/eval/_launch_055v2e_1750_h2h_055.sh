#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750" \
TRAINED_TEAM_OPPONENT_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150" \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 -p 52805 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/055v2e_1750_vs_055_1150.log
exit ${PIPESTATUS[0]}
