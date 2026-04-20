#!/bin/bash
# Launch 039 training in tmux and keep srun alive until training ends.
# This is needed because srun-launched tmux gets killed when srun bash exits.
cd /home/hice1/wsun377/Desktop/cs8803drl
tmux kill-session -t exp_039 2>/dev/null
sleep 2
tmux new -d -s exp_039 'cd /home/hice1/wsun377/Desktop/cs8803drl && bash scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_039_airl_adaptive_on_029B190_512x512.batch 2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/039_airl_adaptive_$(date +%Y%m%d_%H%M%S).log; read'
echo "[launch_039] tmux created, keeping srun alive until session ends"
while tmux has-session -t exp_039 2>/dev/null; do
  sleep 300
done
echo "EXP_039_TMUX_EXITED"
