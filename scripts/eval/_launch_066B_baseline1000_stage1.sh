#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/066B_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50505 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/066B_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000810/checkpoint-810 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000820/checkpoint-820 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000830/checkpoint-830 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000890/checkpoint-890 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000900/checkpoint-900 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000910/checkpoint-910 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000920/checkpoint-920 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000930/checkpoint-930 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000940/checkpoint-940 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000950/checkpoint-950 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000960/checkpoint-960 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_000970/checkpoint-970 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001070/checkpoint-1070 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001080/checkpoint-1080 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001090/checkpoint-1090 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001220/checkpoint-1220 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001230/checkpoint-1230 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/066B_resume_770_to_1250_20260421_043711/TeamVsBaselineShapingPPOTrainer_Soccer_57fa9_00000_0_2026-04-21_04-37-34/checkpoint_001240/checkpoint-1240 \
  2>&1 | tee docs/experiments/artifacts/official-evals/066B_baseline1000.log
exit ${PIPESTATUS[0]}
