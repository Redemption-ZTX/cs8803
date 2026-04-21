#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/074F_baseline1000
DUMMY_CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v074f_weighted_1750_sota.agent \
  --opponents baseline -n 1000 -j 1 --base-port 52905 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/074F_baseline1000 \
  --checkpoint "$DUMMY_CKPT" \
  2>&1 | tee docs/experiments/artifacts/official-evals/074F_baseline1000.log
exit ${PIPESTATUS[0]}
