#!/bin/bash
# 074C — H2H-least-correlated ensemble:
#   {055@1150 + 056D@1140 + 053Dmirror@670}
# Selected because 055 vs 056D H2H = 0.536 NOT sig (marginally tied, most orthogonal
# pair by direct measurement); 053Dmirror added for PBRS blood. 053Dmirror vs 055/056D
# H2H is NOT YET MEASURED — see snapshot-074C §4.1.
# Stage 1: baseline 1000ep (§3.1-§3.4 of snapshot-074C)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl

PY=/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/074C_baseline1000

DUMMY_CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150
$PY scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v074c_h2h_orthogonal.agent \
  --opponents baseline \
  -n 1000 \
  -j 1 \
  --base-port 65205 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/074C_baseline1000 \
  --checkpoint "$DUMMY_CKPT" \
  2>&1 | tee docs/experiments/artifacts/official-evals/074C_baseline1000.log
exit ${PIPESTATUS[0]}
