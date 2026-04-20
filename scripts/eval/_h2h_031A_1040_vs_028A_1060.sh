#!/bin/bash
# H2H: 031A@1040 vs 028A@1060
# Cleanest team-level architecture ablation: same reward (v2 shaping), same warmstart base lineage,
# only difference is flat MLP [512,512] (028A) vs Siamese dual encoder [256,256] + merge [256,128] (031A).
# Goal: separate "architecture contribution" from "training-budget contribution".
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
tmux new -d -s h2h_031A1040_028A1060 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHONPATH=\$PWD\${PYTHONPATH:+:\$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040 \
TRAINED_TEAM_OPPONENT_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060 \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \
  -e 1000 \
  -p 64405 \
  2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/031A_1040_vs_028A_1060.log
read
'"
while tmux has-session -t h2h_031A1040_028A1060 2>/dev/null; do sleep 60; done
echo "H2H_031A1040_028A1060_DONE"
