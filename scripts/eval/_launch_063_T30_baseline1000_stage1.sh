#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/063_T30_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50105 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/063_T30_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000660/checkpoint-660 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000670/checkpoint-670 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000680/checkpoint-680 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000790/checkpoint-790 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000800/checkpoint-800 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000810/checkpoint-810 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000840/checkpoint-840 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000850/checkpoint-850 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_000860/checkpoint-860 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001040/checkpoint-1040 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001050/checkpoint-1050 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001060/checkpoint-1060 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001160/checkpoint-1160 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001170/checkpoint-1170 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001180/checkpoint-1180 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001190/checkpoint-1190 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001200/checkpoint-1200 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001210/checkpoint-1210 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T30_resume_590_to_1250_20260421_025714/TeamVsBaselineShapingPPOTrainer_Soccer_61772_00000_0_2026-04-21_02-57-37/checkpoint_001220/checkpoint-1220 \
  2>&1 | tee docs/experiments/artifacts/official-evals/063_T30_baseline1000.log
exit ${PIPESTATUS[0]}
