#!/bin/bash
# Temporary eval launcher for 038B; deleted after run.
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/038B_baseline1000
tmux new -d -s eval_038B_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/038B_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728/TeamVsBaselineShapingPPOTrainer_Soccer_c2661_00000_0_2026-04-18_12-07-55/checkpoint_000090/checkpoint-90 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728/TeamVsBaselineShapingPPOTrainer_Soccer_c2661_00000_0_2026-04-18_12-07-55/checkpoint_000170/checkpoint-170 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728/TeamVsBaselineShapingPPOTrainer_Soccer_c2661_00000_0_2026-04-18_12-07-55/checkpoint_000110/checkpoint-110 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728/TeamVsBaselineShapingPPOTrainer_Soccer_c2661_00000_0_2026-04-18_12-07-55/checkpoint_000150/checkpoint-150 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728/TeamVsBaselineShapingPPOTrainer_Soccer_c2661_00000_0_2026-04-18_12-07-55/checkpoint_000030/checkpoint-30 \
  2>&1 | tee docs/experiments/artifacts/official-evals/038B_baseline1000.log
read
'"
while tmux has-session -t eval_038B_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_038B_DONE"
