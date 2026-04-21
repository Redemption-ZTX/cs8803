#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/054M_stage3_h2h_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 054M Stage 3 H2H @1460+1750 vs 055@1150 starting"
mkdir -p docs/experiments/artifacts/official-evals/headtohead
CKPT_054M_1460=/storage/ice1/5/1/wsun377/ray_results_scratch/054M_extend_resume_1250_to_1750_20260421_030244/TeamVsBaselineShapingPPOTrainer_Soccer_25766_00000_0_2026-04-21_03-03-06/checkpoint_001460/checkpoint-1460
CKPT_055_1150=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT="$CKPT_054M_1460" \
TRAINED_TEAM_OPPONENT_CHECKPOINT="$CKPT_055_1150" \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 500 -p 64405 \
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/054M_1460_vs_055_1150.log
echo "[$(date)] 054M Stage 3 H2H EXIT=$?"
