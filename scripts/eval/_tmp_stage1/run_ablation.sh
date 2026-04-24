#!/bin/bash
# Usage: PRESET=ablation_X PORT=62005 LOG=ablation_X bash run_ablation.sh
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/ablation
exec env SELECTOR_PHASE_MAP_PRESET=$PRESET \
  /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python -m cs8803drl.evaluation.evaluate_matches \
    --team0_module agents.v_selector_phase4 \
    --team1_module ceia_baseline_agent \
    -n 200 \
    --base_port $PORT \
    2>&1 | tee docs/experiments/artifacts/official-evals/ablation/${LOG}.log
