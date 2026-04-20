#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/036D_baseline1000
tmux new -d -s eval_036D_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_shared_cc_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63105 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/036D_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000040/checkpoint-40 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000070/checkpoint-70 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000130/checkpoint-130 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000160/checkpoint-160 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000250/checkpoint-250 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000010/checkpoint-10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/036D_baseline1000.log
read
'"
while tmux has-session -t eval_036D_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_036D_DONE"
