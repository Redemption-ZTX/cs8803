#!/bin/bash
# 058 curriculum Stage 1 post-eval baseline 1000ep
# pick_top_ckpts output: 850 860 930 940 950 1150 1160 1170 1220 1230 1240 (11 ckpts)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/058_curriculum_scratch_v2_512x512_20260420_092046"
TRIAL_DIR=$(ls -d ${TRIAL}/TeamVsBaselineShaping*/ | head -1)

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/058_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 49005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/058_baseline1000 \
  --checkpoint "${TRIAL_DIR}/checkpoint_000850/checkpoint-850" \
  --checkpoint "${TRIAL_DIR}/checkpoint_000860/checkpoint-860" \
  --checkpoint "${TRIAL_DIR}/checkpoint_000930/checkpoint-930" \
  --checkpoint "${TRIAL_DIR}/checkpoint_000940/checkpoint-940" \
  --checkpoint "${TRIAL_DIR}/checkpoint_000950/checkpoint-950" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001150/checkpoint-1150" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001160/checkpoint-1160" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001170/checkpoint-1170" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001220/checkpoint-1220" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001230/checkpoint-1230" \
  --checkpoint "${TRIAL_DIR}/checkpoint_001240/checkpoint-1240" \
  2>&1 | tee docs/experiments/artifacts/official-evals/058_baseline1000.log
exit ${PIPESTATUS[0]}
