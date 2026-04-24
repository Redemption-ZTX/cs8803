#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/dir-A
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python -m cs8803drl.evaluation.evaluate_matches \
  --team0_module agents.v_selector_phase4 \
  --team1_module ceia_baseline_agent \
  -n 200 \
  --base_port 61605 \
  2>&1 | tee docs/experiments/artifacts/official-evals/dir-A/v_selector_phase4_wave2_baseline200.log
