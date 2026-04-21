#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/056E_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50605 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/056E_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000690/checkpoint-690 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000700/checkpoint-700 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000710/checkpoint-710 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000910/checkpoint-910 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000920/checkpoint-920 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000930/checkpoint-930 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000960/checkpoint-960 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000970/checkpoint-970 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_000980/checkpoint-980 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001020/checkpoint-1020 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001030/checkpoint-1030 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001040/checkpoint-1040 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001050/checkpoint-1050 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001060/checkpoint-1060 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001110/checkpoint-1110 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001120/checkpoint-1120 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001130/checkpoint-1130 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001140/checkpoint-1140 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001150/checkpoint-1150 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001160/checkpoint-1160 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001170/checkpoint-1170 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001180/checkpoint-1180 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001190/checkpoint-1190 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260421_041510/TeamVsBaselineShapingPPOTrainer_Soccer_433d6_00000_0_2026-04-21_04-15-31/checkpoint_001200/checkpoint-1200 \
  2>&1 | tee docs/experiments/artifacts/official-evals/056E_baseline1000.log
exit ${PIPESTATUS[0]}
