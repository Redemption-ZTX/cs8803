#!/bin/bash
# 045B learned-only (0.003·learned_reward only on 031A@1040, no v2) failure capture @ ckpt 150.
# Top 1000ep peak ckpt. v2 bucket comparison vs 031A@1040 / 045A@180.
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
tmux new -d -s fail_045B_150 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/045B_learned_only_on_031A1040_512x512_20260419_095729/TeamVsBaselineShapingPPOTrainer_Soccer_c0f88_00000_0_2026-04-19_09-57-50/checkpoint_000150/checkpoint-150 \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponent baseline \
  -n 500 \
  --max-steps 1500 \
  --base-port 64545 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/045B_checkpoint150_baseline_500 \
  --save-mode losses \
  --max-saved-episodes 500 \
  --trace-stride 10 \
  --trace-tail-steps 30 \
  --reward-shaping-debug \
  --time-penalty 0.001 \
  --ball-progress-scale 0.01 \
  --goal-proximity-scale 0.0 \
  --progress-requires-possession 0 \
  --opponent-progress-penalty-scale 0.01 \
  --possession-dist 1.25 \
  --possession-bonus 0.002 \
  --deep-zone-outer-threshold -8 \
  --deep-zone-outer-penalty 0.003 \
  --deep-zone-inner-threshold -12 \
  --deep-zone-inner-penalty 0.003 \
  --defensive-survival-threshold 0 \
  --defensive-survival-bonus 0 \
  --fast-loss-threshold-steps 0 \
  --fast-loss-penalty-per-step 0 \
  --event-shot-reward 0.0 \
  --event-tackle-reward 0.0 \
  --event-clearance-reward 0.0 \
  --event-cooldown-steps 10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/045B_checkpoint150.log
read
'"
while tmux has-session -t fail_045B_150 2>/dev/null; do sleep 60; done
echo "FAIL_045B_150_DONE"
