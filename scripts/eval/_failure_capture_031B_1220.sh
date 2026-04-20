#!/usr/bin/env bash
# Stage 2: failure capture for 031B@1220 (top 1000ep peak 0.882)
# Shaping v2 family (matches 031A/031B training config: deep_zone -8/-12 + outer/inner penalty 0.003)
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
CKPT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/031B_team_cross_attention_scratch_v2_resume1080__TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/checkpoint_001220/checkpoint-1220

"$PY" scripts/eval/capture_failure_cases.py \
    --checkpoint "$CKPT" \
    --team0-module cs8803drl.deployment.trained_team_ray_agent \
    --opponent baseline \
    -n 500 \
    --max-steps 1500 \
    --base-port 54005 \
    --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/031B_checkpoint1220_baseline_500 \
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
    2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/031B_checkpoint1220.log
