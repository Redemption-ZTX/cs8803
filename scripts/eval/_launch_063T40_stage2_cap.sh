#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/063T40_stage2_cap_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] T40@1060 Stage 2 capture starting"
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
mkdir -p docs/experiments/artifacts/failure-cases/063T40_checkpoint1060_baseline_500
CKPT=/storage/ice1/5/1/wsun377/ray_results_scratch/063_T40_resume_370_to_1250_20260421_053932/TeamVsBaselineShapingPPOTrainer_Soccer_0ec47_00000_0_2026-04-21_05-39-57/checkpoint_001060/checkpoint-1060
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint "$CKPT" \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponent baseline -n 500 --max-steps 1500 --base-port 51805 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/063T40_checkpoint1060_baseline_500 \
  --save-mode losses --max-saved-episodes 500 --trace-stride 10 --trace-tail-steps 30 \
  --reward-shaping-debug \
  --time-penalty 0.001 --ball-progress-scale 0.01 --goal-proximity-scale 0.0 --progress-requires-possession 0 \
  --opponent-progress-penalty-scale 0.01 --possession-dist 1.25 --possession-bonus 0.002 \
  --deep-zone-outer-threshold -8 --deep-zone-outer-penalty 0.003 --deep-zone-inner-threshold -12 --deep-zone-inner-penalty 0.003 \
  --defensive-survival-threshold 0 --defensive-survival-bonus 0 --fast-loss-threshold-steps 0 --fast-loss-penalty-per-step 0 \
  --event-shot-reward 0.0 --event-tackle-reward 0.0 --event-clearance-reward 0.0 --event-cooldown-steps 10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/063T40_checkpoint1060.log
echo "[$(date)] T40@1060 cap EXIT=$?"
