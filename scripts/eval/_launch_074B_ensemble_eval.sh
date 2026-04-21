#!/bin/bash
# 074B — Architectural-diversity ensemble:
#   {055@1150 (Siamese cross-attn) + 054M@1230 (MAT cross-AGENT attn) + 062a@1220 (Siamese)}
# Stage 1: baseline 1000ep (§3.1-§3.4 of snapshot-074B)
# WARN: MAT model registration in ensemble_agent.py may need fixup —
# see snapshot-074B §4.1 / §6 step 2 (smoke load test first).
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/074B_baseline1000

DUMMY_CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
$PY scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v074b_frontier_arch_diversity.agent \
  --opponents baseline \
  -n 1000 \
  -j 1 \
  --base-port 65105 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/074B_baseline1000 \
  --checkpoint "$DUMMY_CKPT" \
  2>&1 | tee docs/experiments/artifacts/official-evals/074B_baseline1000.log
exit ${PIPESTATUS[0]}
