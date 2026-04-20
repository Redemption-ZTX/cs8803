#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/038D_baseline1000
tmux new -d -s eval_038D_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 63005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/038D_baseline1000 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000100/checkpoint-100 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000040/checkpoint-40 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000090/checkpoint-90 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000060/checkpoint-60 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000020/checkpoint-20 \
  2>&1 | tee docs/experiments/artifacts/official-evals/038D_baseline1000.log
read
'"
while tmux has-session -t eval_038D_b1000 2>/dev/null; do
  sleep 60
done
echo "EVAL_038D_DONE"
