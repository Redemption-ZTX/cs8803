#!/bin/bash
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
T031A=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040

mkdir -p docs/experiments/artifacts/failure-cases/h2h_v3_all_v2/p24_034E_vs_031A_migrated_n500

env "TRAINED_TEAM_OPPONENT_CHECKPOINT=$T031A" \
  "$PY" scripts/eval/capture_failure_cases.py \
  --team0-module agents.v034e_frontier_031B_3way.agent \
  --opponent cs8803drl.deployment.trained_team_ray_opponent_agent \
  -n 500 --max-steps 1500 --base-port 60100 \
  --save-dir docs/experiments/artifacts/failure-cases/h2h_v3_all_v2/p24_034E_vs_031A_migrated_n500 \
  --save-mode all --max-saved-episodes 500 --trace-stride 10 --trace-tail-steps 30 \
  --reward-shaping-debug --time-penalty 0.001 --ball-progress-scale 0.01 --goal-proximity-scale 0.0 \
  --progress-requires-possession 0 --opponent-progress-penalty-scale 0.01 --possession-dist 1.25 \
  --possession-bonus 0.002 --deep-zone-outer-threshold -8 --deep-zone-outer-penalty 0.003 \
  --deep-zone-inner-threshold -12 --deep-zone-inner-penalty 0.003 --defensive-survival-threshold 0 \
  --defensive-survival-bonus 0 --fast-loss-threshold-steps 0 --fast-loss-penalty-per-step 0 \
  --event-shot-reward 0.0 --event-tackle-reward 0.0 --event-clearance-reward 0.0 --event-cooldown-steps 10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/v2_p24_034E_vs_031A_migrated.log
