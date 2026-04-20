#!/bin/bash
# 038D @ ckpt 40 failure capture (team-level lane, top 1000ep = 0.806)
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
tmux new -d -s fail_038D_40 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000040/checkpoint-40 \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponent baseline \
  -n 500 \
  --max-steps 1500 \
  --base-port 64805 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/038D_checkpoint040_baseline_500 \
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
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/038D_checkpoint040.log
read
'"
while tmux has-session -t fail_038D_40 2>/dev/null; do
  sleep 60
done
echo "FAIL_038D_40_DONE"
