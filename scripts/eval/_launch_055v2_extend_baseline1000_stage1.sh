#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/055v2_extend_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/055v2_extend_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001140/checkpoint-1140 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001150/checkpoint-1150 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001160/checkpoint-1160 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001250/checkpoint-1250 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001260/checkpoint-1260 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001270/checkpoint-1270 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001450/checkpoint-1450 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001460/checkpoint-1460 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001470/checkpoint-1470 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001500/checkpoint-1500 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001510/checkpoint-1510 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001520/checkpoint-1520 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001570/checkpoint-1570 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001580/checkpoint-1580 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001590/checkpoint-1590 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001660/checkpoint-1660 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001670/checkpoint-1670 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001680/checkpoint-1680 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001690/checkpoint-1690 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001700/checkpoint-1700 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001710/checkpoint-1710 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001720/checkpoint-1720 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001740/checkpoint-1740 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001750/checkpoint-1750 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001760/checkpoint-1760 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001810/checkpoint-1810 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001820/checkpoint-1820 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001830/checkpoint-1830 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001840/checkpoint-1840 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001850/checkpoint-1850 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001860/checkpoint-1860 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001940/checkpoint-1940 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001950/checkpoint-1950 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_001960/checkpoint-1960 \
  2>&1 | tee docs/experiments/artifacts/official-evals/055v2_extend_baseline1000.log
exit ${PIPESTATUS[0]}
