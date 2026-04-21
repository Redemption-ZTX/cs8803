#!/bin/bash
# 074E — Outcome-predictor-enhanced ensemble (top-K re-rank on 074A members).
# Members: {055@1150, 053Dmirror@670, 062a@1220}
# Augmentation: calibrated v3 outcome predictor at
#   docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
# Only active when ensemble top-1/top-2 margin < 0.10. Otherwise falls back
# to plain mean-of-probs (= 074A).
# Stage 1: baseline 1000ep (§3.1-§3.4 of snapshot-074E).
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/074E_baseline1000

# Let the wrapper pick its own device. Explicit export for clarity.
export OUTCOME_RERANK_ENABLE=1
export OUTCOME_RERANK_TOPK=3
export OUTCOME_RERANK_DEVICE=auto

DUMMY_CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
$PY scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v074e_predictor_rerank.agent \
  --opponents baseline \
  -n 1000 \
  -j 1 \
  --base-port 65405 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/074E_baseline1000 \
  --checkpoint "$DUMMY_CKPT" \
  2>&1 | tee docs/experiments/artifacts/official-evals/074E_baseline1000.log
exit ${PIPESTATUS[0]}
