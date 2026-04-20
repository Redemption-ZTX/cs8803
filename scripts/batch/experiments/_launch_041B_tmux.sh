#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/slurm-logs
tmux new -d -s 041B "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
bash scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_041B_pbrs_on_036D150_512x512.batch \
2>&1 | tee docs/experiments/artifacts/slurm-logs/041B_local-$(date +%Y%m%d_%H%M%S).log
read
'"
while tmux has-session -t 041B 2>/dev/null; do sleep 60; done
echo "041B_TRAINING_DONE"
