#!/bin/bash
set -euo pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
export MAX_ITERATIONS=1
export EVAL_INTERVAL=1
export EVAL_EPISODES=50
export TIMESTEPS_TOTAL=0
export TIME_TOTAL_S=0
export CHECKPOINT_FREQ=1
export RUN_NAME=smoke_029C_$(date +%Y%m%d_%H%M%S)
export QUIET_CONSOLE=1
export BASE_PORT=63705
export EVAL_BASE_PORT=63605
bash scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_029C_025b80_oppool_peers_512x512.batch
