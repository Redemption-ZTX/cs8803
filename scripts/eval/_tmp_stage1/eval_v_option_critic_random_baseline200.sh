#!/bin/bash
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/dir-E
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python -m cs8803drl.evaluation.evaluate_matches \
  --team0_module agents.v_option_critic_random \
  --team1_module ceia_baseline_agent \
  -n 200 \
  --base_port 60805 \
  2>&1 | tee docs/experiments/artifacts/official-evals/dir-E/v_option_critic_random_wave1_baseline200.log
