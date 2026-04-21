#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/055temp_stage3_h2h_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 055temp Stage 3 H2H vs 055@1150 starting"
mkdir -p docs/experiments/artifacts/official-evals/headtohead
CKPT_055TEMP=/storage/ice1/5/1/wsun377/ray_results_scratch/055temp_distill_034e_ensemble_to_031B_scratch_20260420_155212/TeamVsBaselineShapingPPOTrainer_Soccer_78f4d_00000_0_2026-04-20_15-52-33/checkpoint_001030/checkpoint-1030
CKPT_055=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="$CKPT_055TEMP" \
TRAINED_TEAM_OPPONENT_CHECKPOINT="$CKPT_055" \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 \
  -p 64205 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/055temp_1030_vs_055_1150.log
echo "[$(date)] 055temp Stage 3 H2H EXIT=$?"
