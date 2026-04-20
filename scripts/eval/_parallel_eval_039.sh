#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/039_baseline1000
tmux new -d -s eval_039_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_shared_cc_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/039_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000040/checkpoint-40 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000170/checkpoint-170 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000010/checkpoint-10 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000140/checkpoint-140 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000150/checkpoint-150 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000230/checkpoint-230 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607/MAPPOVsBaselineTrainer_Soccer_bbfeb_00000_0_2026-04-18_13-26-28/checkpoint_000190/checkpoint-190 \
  2>&1 | tee docs/experiments/artifacts/official-evals/039_baseline1000.log
read
'"
while tmux has-session -t eval_039_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_039_DONE"
