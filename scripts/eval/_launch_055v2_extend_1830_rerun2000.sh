#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055v2_extend_1830_rerun2000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 2000 -j 7 --base-port 51005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055v2_extend_1830_rerun2000 \
  --checkpoint "/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001830/checkpoint-1830" \
  2>&1 | tee docs/experiments/artifacts/official-evals/055v2_extend_1830_rerun2000.log
exit ${PIPESTATUS[0]}
