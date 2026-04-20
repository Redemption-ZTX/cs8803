#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/slurm-logs
tmux new -d -s 044A "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
bash scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_044A_spear_scratch_512x512.batch \
2>&1 | tee docs/experiments/artifacts/slurm-logs/044A_local-$(date +%Y%m%d_%H%M%S).log
read
'"
while tmux has-session -t 044A 2>/dev/null; do sleep 60; done
echo "044A_TRAINING_DONE"
