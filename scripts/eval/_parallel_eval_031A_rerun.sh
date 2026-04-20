#!/bin/bash
# 031A@1040 1000ep rerun — confirm 0.867 isn't sampling upper bound
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/031A_baseline1000_rerun
tmux new -d -s eval_031A_rerun "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 2 \
  --base-port 64605 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/031A_baseline1000_rerun \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001170/checkpoint-1170 \
  2>&1 | tee docs/experiments/artifacts/official-evals/031A_baseline1000_rerun.log
read
'"
while tmux has-session -t eval_031A_rerun 2>/dev/null; do sleep 60; done
echo "EVAL_031A_RERUN_DONE"
