#!/bin/bash
# Resume 031B training from checkpoint 50 (segment 1 was killed at iter 50
# when SLURM job 4992078 was replaced).
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/slurm-logs
tmux new -d -s 031B "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
RESTORE_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_225904/TeamVsBaselineShapingPPOTrainer_Soccer_c69d9_00000_0_2026-04-18_22-59-26/checkpoint_000050/checkpoint-50 \
bash scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch \
2>&1 | tee docs/experiments/artifacts/slurm-logs/031B_resume-$(date +%Y%m%d_%H%M%S).log
read
'"
while tmux has-session -t 031B 2>/dev/null; do sleep 60; done
echo "031B_RESUME_DONE"
