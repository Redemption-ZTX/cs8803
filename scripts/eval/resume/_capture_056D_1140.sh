#!/bin/bash
# Failure capture: 056D_1140 vs baseline 500ep (save losses for bucket analysis)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/failure-cases/056D_1140_baseline_500
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint "/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140" \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponent baseline \
  -n 500 \
  --max-steps 1500 \
  --base-port 36005 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/056D_1140_baseline_500 \
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
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/056D_1140.log
exit ${PIPESTATUS[0]}
