#!/bin/bash
# Wrapper to start 031B training inside a persistent tmux session on the
# already-allocated SLURM node. Mirrors the pattern used by H2H eval scripts.
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/slurm-logs
tmux new -d -s 031B "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
bash scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch 2>&1 | tee docs/experiments/artifacts/slurm-logs/031B_local-$(date +%Y%m%d_%H%M%S).log
read
'"
while tmux has-session -t 031B 2>/dev/null; do sleep 60; done
echo "031B_TRAINING_DONE"
