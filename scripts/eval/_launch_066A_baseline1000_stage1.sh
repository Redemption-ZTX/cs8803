#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/066A_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50405 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/066A_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000840/checkpoint-840 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000850/checkpoint-850 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000860/checkpoint-860 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000870/checkpoint-870 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000880/checkpoint-880 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_000890/checkpoint-890 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001090/checkpoint-1090 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001100/checkpoint-1100 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001110/checkpoint-1110 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001120/checkpoint-1120 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001130/checkpoint-1130 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001140/checkpoint-1140 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001150/checkpoint-1150 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001160/checkpoint-1160 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001170/checkpoint-1170 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001180/checkpoint-1180 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001190/checkpoint-1190 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001200/checkpoint-1200 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001210/checkpoint-1210 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001220/checkpoint-1220 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001230/checkpoint-1230 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066A_resume_810_to_1250_20260421_032638/TeamVsBaselineShapingPPOTrainer_Soccer_85d23_00000_0_2026-04-21_03-27-16/checkpoint_001240/checkpoint-1240 \
  2>&1 | tee docs/experiments/artifacts/official-evals/066A_baseline1000.log
exit ${PIPESTATUS[0]}
