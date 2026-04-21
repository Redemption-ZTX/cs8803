#!/bin/bash
# H2H: 062a_1220 vs 055_1150 (500ep, port 33005)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
export TRAINED_RAY_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220"
export TRAINED_TEAM_OPPONENT_CHECKPOINT="/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150"
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 -p 33005 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/062a_1220_vs_055_1150.log
exit ${PIPESTATUS[0]}
