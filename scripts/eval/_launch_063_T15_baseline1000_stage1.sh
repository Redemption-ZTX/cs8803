#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/063_T15_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50205 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/063_T15_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000640/checkpoint-640 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000650/checkpoint-650 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000660/checkpoint-660 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000760/checkpoint-760 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000770/checkpoint-770 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000780/checkpoint-780 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_000990/checkpoint-990 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001000/checkpoint-1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001010/checkpoint-1010 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001040/checkpoint-1040 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001050/checkpoint-1050 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001060/checkpoint-1060 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001070/checkpoint-1070 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001080/checkpoint-1080 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001090/checkpoint-1090 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001100/checkpoint-1100 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001110/checkpoint-1110 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001170/checkpoint-1170 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001180/checkpoint-1180 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001190/checkpoint-1190 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001210/checkpoint-1210 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001220/checkpoint-1220 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/063_T15_resume_600_to_1250_20260421_025720/TeamVsBaselineShapingPPOTrainer_Soccer_64927_00000_0_2026-04-21_02-57-42/checkpoint_001230/checkpoint-1230 \
  2>&1 | tee docs/experiments/artifacts/official-evals/063_T15_baseline1000.log
exit ${PIPESTATUS[0]}
