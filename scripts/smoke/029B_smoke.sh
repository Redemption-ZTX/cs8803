#!/bin/bash
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
export MAX_ITERATIONS=1
export EVAL_INTERVAL=1
export EVAL_EPISODES=50
export TIMESTEPS_TOTAL=0
export TIME_TOTAL_S=0
export CHECKPOINT_FREQ=1
export RUN_NAME=smoke_029B_$(date +%Y%m%d_%H%M%S)
export QUIET_CONSOLE=1
export BASE_PORT=63505
export EVAL_BASE_PORT=63405
bash scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_029B_bwarm170_to_v2_512x512.batch
