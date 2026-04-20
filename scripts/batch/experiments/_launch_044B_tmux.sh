#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/slurm-logs
tmux new -d -s 044B "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
bash scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_044B_shield_scratch_512x512.batch \
2>&1 | tee docs/experiments/artifacts/slurm-logs/044B_local-$(date +%Y%m%d_%H%M%S).log
read
'"
while tmux has-session -t 044B 2>/dev/null; do sleep 60; done
echo "044B_TRAINING_DONE"
