#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/038C_baseline1000
tmux new -d -s eval_038C_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/038C_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730/TeamVsBaselineShapingPPOTrainer_Soccer_c16c0_00000_0_2026-04-18_12-07-53/checkpoint_000050/checkpoint-50 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730/TeamVsBaselineShapingPPOTrainer_Soccer_c16c0_00000_0_2026-04-18_12-07-53/checkpoint_000080/checkpoint-80 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730/TeamVsBaselineShapingPPOTrainer_Soccer_c16c0_00000_0_2026-04-18_12-07-53/checkpoint_000090/checkpoint-90 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730/TeamVsBaselineShapingPPOTrainer_Soccer_c16c0_00000_0_2026-04-18_12-07-53/checkpoint_000120/checkpoint-120 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730/TeamVsBaselineShapingPPOTrainer_Soccer_c16c0_00000_0_2026-04-18_12-07-53/checkpoint_000130/checkpoint-130 \
  2>&1 | tee docs/experiments/artifacts/official-evals/038C_baseline1000.log
read
'"
while tmux has-session -t eval_038C_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_038C_DONE"
