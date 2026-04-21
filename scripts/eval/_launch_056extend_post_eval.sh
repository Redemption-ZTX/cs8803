#!/bin/bash
# 056extend post-eval — verify inline 200ep 0.930 @ 1280 with 1000ep
# pick_top substitute: top 10 ckpts by inline 200ep
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

TRIAL="/storage/ice1/5/1/wsun377/ray_results_scratch/056extend_resume_056D_1250_to_2000_20260420_135527/TeamVsBaselineShapingPPOTrainer_Soccer_2a326_00000_0_2026-04-20_13-55-49"

mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/056extend_baseline1000

/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port 44005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/056extend_baseline1000 \
  --checkpoint "${TRIAL}/checkpoint_001260/checkpoint-1260" \
  --checkpoint "${TRIAL}/checkpoint_001280/checkpoint-1280" \
  --checkpoint "${TRIAL}/checkpoint_001300/checkpoint-1300" \
  --checkpoint "${TRIAL}/checkpoint_001320/checkpoint-1320" \
  --checkpoint "${TRIAL}/checkpoint_001380/checkpoint-1380" \
  --checkpoint "${TRIAL}/checkpoint_001410/checkpoint-1410" \
  --checkpoint "${TRIAL}/checkpoint_001430/checkpoint-1430" \
  --checkpoint "${TRIAL}/checkpoint_001480/checkpoint-1480" \
  --checkpoint "${TRIAL}/checkpoint_001500/checkpoint-1500" \
  2>&1 | tee docs/experiments/artifacts/official-evals/056extend_baseline1000.log
exit ${PIPESTATUS[0]}
