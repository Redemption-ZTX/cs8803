#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/038A_baseline1000
tmux new -d -s eval_038A_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/038A_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346/TeamVsBaselineShapingPPOTrainer_Soccer_9fbb8_00000_0_2026-04-18_12-14-06/checkpoint_000050/checkpoint-50 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346/TeamVsBaselineShapingPPOTrainer_Soccer_9fbb8_00000_0_2026-04-18_12-14-06/checkpoint_000180/checkpoint-180 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346/TeamVsBaselineShapingPPOTrainer_Soccer_9fbb8_00000_0_2026-04-18_12-14-06/checkpoint_000020/checkpoint-20 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346/TeamVsBaselineShapingPPOTrainer_Soccer_9fbb8_00000_0_2026-04-18_12-14-06/checkpoint_000160/checkpoint-160 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346/TeamVsBaselineShapingPPOTrainer_Soccer_9fbb8_00000_0_2026-04-18_12-14-06/checkpoint_000170/checkpoint-170 \
  2>&1 | tee docs/experiments/artifacts/official-evals/038A_baseline1000.log
read
'"
while tmux has-session -t eval_038A_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_038A_DONE"
