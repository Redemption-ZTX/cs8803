#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
tmux new -d -s h2h_031A1040_025b080 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHONPATH=\$PWD\${PYTHONPATH:+:\$PYTHONPATH} \
TRAINED_RAY_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040 \
TRAINED_SHARED_CC_OPPONENT_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80 \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_team_ray_agent \
  -m2 cs8803drl.deployment.trained_shared_cc_opponent_agent \
  -e 1000 \
  -p 64405 \
  2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/031A_1040_vs_025b_080.log
read
'"
while tmux has-session -t h2h_031A1040_025b080 2>/dev/null; do sleep 60; done
echo "H2H_031A1040_025b080_DONE"
