#!/bin/bash
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/070_poolB_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module cs8803drl.deployment.trained_team_ray_agent \
  --opponents baseline -n 1000 -j 7 --base-port 50305 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/070_poolB_baseline1000 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000710/checkpoint-710 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000720/checkpoint-720 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000730/checkpoint-730 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000780/checkpoint-780 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000790/checkpoint-790 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000800/checkpoint-800 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000870/checkpoint-870 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000880/checkpoint-880 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000890/checkpoint-890 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000910/checkpoint-910 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000920/checkpoint-920 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_000930/checkpoint-930 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001070/checkpoint-1070 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001080/checkpoint-1080 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001090/checkpoint-1090 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001110/checkpoint-1110 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001120/checkpoint-1120 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001130/checkpoint-1130 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001170/checkpoint-1170 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001180/checkpoint-1180 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001190/checkpoint-1190 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001210/checkpoint-1210 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001220/checkpoint-1220 \
  --checkpoint /storage/ice1/5/1/wsun377/ray_results_scratch/070_poolB_divergent_distill_scratch_20260421_043336/TeamVsBaselineShapingPPOTrainer_Soccer_d68be_00000_0_2026-04-21_04-33-57/checkpoint_001230/checkpoint-1230 \
  2>&1 | tee docs/experiments/artifacts/official-evals/070_poolB_baseline1000.log
exit ${PIPESTATUS[0]}
