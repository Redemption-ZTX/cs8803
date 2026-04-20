#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
tmux new -d -s h2h_036D150_029B190 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
PYTHONPATH=\$PWD\${PYTHONPATH:+:\$PYTHONPATH} \
TRAINED_SHARED_CC_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150 \
TRAINED_SHARED_CC_OPPONENT_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190 \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 cs8803drl.deployment.trained_shared_cc_agent \
  -m2 cs8803drl.deployment.trained_shared_cc_opponent_agent \
  -e 1000 \
  -p 65305 \
  2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/036D_150_vs_029B_190.log
read
'"
while tmux has-session -t h2h_036D150_029B190 2>/dev/null; do
  sleep 60
done
echo "H2H_036D150_029B190_DONE"
