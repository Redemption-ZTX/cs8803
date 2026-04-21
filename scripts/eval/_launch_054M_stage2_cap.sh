#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/054M_stage2_cap_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 054M Stage 2 capture @1460+1750 starting"
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
CKPT_1460=/storage/ice1/5/1/wsun377/ray_results_scratch/054M_extend_resume_1250_to_1750_20260421_030244/TeamVsBaselineShapingPPOTrainer_Soccer_25766_00000_0_2026-04-21_03-03-06/checkpoint_001460/checkpoint-1460
CKPT_1750=/storage/ice1/5/1/wsun377/ray_results_scratch/054M_extend_resume_1250_to_1750_20260421_030244/TeamVsBaselineShapingPPOTrainer_Soccer_25766_00000_0_2026-04-21_03-03-06/checkpoint_001750/checkpoint-1750

# cap@1460 (peak)
mkdir -p docs/experiments/artifacts/failure-cases/054M_checkpoint1460_baseline_500
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint "$CKPT_1460" \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponent baseline -n 500 --max-steps 1500 --base-port 63405 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/054M_checkpoint1460_baseline_500 \
  --save-mode losses --max-saved-episodes 500 --trace-stride 10 --trace-tail-steps 30 \
  --reward-shaping-debug \
  --time-penalty 0.001 --ball-progress-scale 0.01 --goal-proximity-scale 0.0 --progress-requires-possession 0 \
  --opponent-progress-penalty-scale 0.01 --possession-dist 1.25 --possession-bonus 0.002 \
  --deep-zone-outer-threshold -8 --deep-zone-outer-penalty 0.003 --deep-zone-inner-threshold -12 --deep-zone-inner-penalty 0.003 \
  --defensive-survival-threshold 0 --defensive-survival-bonus 0 --fast-loss-threshold-steps 0 --fast-loss-penalty-per-step 0 \
  --event-shot-reward 0.0 --event-tackle-reward 0.0 --event-clearance-reward 0.0 --event-cooldown-steps 10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/054M_checkpoint1460.log
echo "[$(date)] 054M cap1460 EXIT=$?"
