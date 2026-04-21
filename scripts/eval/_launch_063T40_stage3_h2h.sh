#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/063_T40_resume_370_to_1250_20260421_053932/TeamVsBaselineShapingPPOTrainer_Soccer_0ec47_00000_0_2026-04-21_05-39-57/checkpoint_001060/checkpoint-1060" \
TRAINED_TEAM_OPPONENT_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150" \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 -p 64805 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/063T40_1060_vs_055_1150.log
exit ${PIPESTATUS[0]}
