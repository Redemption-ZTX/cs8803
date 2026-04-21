#!/usr/bin/env python3
"""Generate resume launchers for all 10 salvage/extend lanes.

Each resume launcher:
- Copies original launcher
- Injects RESTORE_CHECKPOINT=<last ckpt path>
- Sets TIME_TOTAL_S=86400 (24h, avoid wall)
- Sets MAX_ITERATIONS per target (usually same, 2000 for 055v2 extend, 1750 for 054M extend)
- Sets new PORT_SEED (fresh, no collision)
- Adjusts RUN_NAME with resume_ prefix
"""
import os, re, sys

# Priority order (top = launch first)
RESUMES = [
    # (lane_tag, source_launcher, restore_ckpt, max_iterations, new_port_seed, new_run_prefix)
    ("055v2_extend", "_launch_055v2_recursive_distill.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001210/checkpoint-1210",
     2000, 81, "055v2_extend_resume_1210_to_2000_"),
    ("055temp_resume", "_launch_055temp_distill_T2.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/055temp_distill_034e_ensemble_to_031B_scratch_20260420_155212/TeamVsBaselineShapingPPOTrainer_Soccer_78f4d_00000_0_2026-04-20_15-52-33/checkpoint_001030/checkpoint-1030",
     1250, 83, "055temp_resume_1030_to_1250_"),
    ("063_T30_resume", "_launch_063_temp_T30.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/063_temp_distill_T30_on_034e_ensemble_031B_20260420_194828/TeamVsBaselineShapingPPOTrainer_Soccer_79b88_00000_0_2026-04-20_19-48-48/checkpoint_000590/checkpoint-590",
     1250, 85, "063_T30_resume_590_to_1250_"),
    ("063_T15_resume", "_launch_063_temp_T15.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/063_temp_distill_T15_on_034e_ensemble_031B_20260420_195004/TeamVsBaselineShapingPPOTrainer_Soccer_b2942_00000_0_2026-04-20_19-50-23/checkpoint_000600/checkpoint-600",
     1250, 87, "063_T15_resume_600_to_1250_"),
    ("054M_extend", "_launch_054M_mat_medium_scratch.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/054M_mat_medium_scratch_v2_512x512_20260420_135128/TeamVsBaselineShapingPPOTrainer_Soccer_a0dde_00000_0_2026-04-20_13-51-59/checkpoint_001250/checkpoint-1250",
     1750, 89, "054M_extend_resume_1250_to_1750_"),
    ("066A_resume", "_launch_066A_pure_selfdistill_from_055.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/066A_pure_selfdistill_from_055_1150_scratch_20260420_173612/TeamVsBaselineShapingPPOTrainer_Soccer_00637_00000_0_2026-04-20_17-36-33/checkpoint_000810/checkpoint-810",
     1250, 91, "066A_resume_810_to_1250_"),
    ("066B_resume", "_launch_066B_weighted_4teacher_from_055.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/066B_weighted_4teacher_055plus034e_scratch_20260420_180834/TeamVsBaselineShapingPPOTrainer_Soccer_858e8_00000_0_2026-04-20_18-08-55/checkpoint_000770/checkpoint-770",
     1250, 93, "066B_resume_770_to_1250_"),
    ("056E_resume", "_launch_056_pbt_lr_sweep.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/056E_pbt_lr0.00050_scratch_20260420_164845/TeamVsBaselineShapingPPOTrainer_Soccer_5ee7f_00000_0_2026-04-20_16-49-05/checkpoint_000970/checkpoint-970",
     1250, 95, "056E_resume_970_to_1250_"),
    ("056F_resume", "_launch_056_pbt_lr_sweep.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/056F_pbt_lr0.00070_scratch_20260420_164850/TeamVsBaselineShapingPPOTrainer_Soccer_62935_00000_0_2026-04-20_16-49-12/checkpoint_000970/checkpoint-970",
     1250, 97, "056F_resume_970_to_1250_"),
    ("063_T40_resume", "_launch_063_temp_T40.sh",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/063_temp_distill_T40_on_034e_ensemble_031B_20260420_214137/TeamVsBaselineShapingPPOTrainer_Soccer_53a2f_00000_0_2026-04-20_21-42-16/checkpoint_000370/checkpoint-370",
     1250, 99, "063_T40_resume_370_to_1250_"),
]

SRC_DIR = "/home/hice1/wsun377/Desktop/cs8803drl/scripts/eval"
OUT_DIR = "/home/hice1/wsun377/Desktop/cs8803drl/scripts/eval/resume"
os.makedirs(OUT_DIR, exist_ok=True)

for lane, src, ckpt, max_iter, port, run_prefix in RESUMES:
    src_path = os.path.join(SRC_DIR, src)
    if not os.path.exists(src_path):
        print(f"MISS: {src_path}")
        continue
    with open(src_path) as f:
        content = f.read()

    # Insert RESTORE_CHECKPOINT (after LOCAL_DIR line), bump TIME_TOTAL_S, fix MAX_ITER, fix PORT_SEED default
    # 1. Add RESTORE_CHECKPOINT after the initial env-var unset block (safest: after "export LOCAL_DIR=")
    if 'export RESTORE_CHECKPOINT=' not in content:
        content = content.replace(
            'export LOCAL_DIR=',
            f'export RESTORE_CHECKPOINT="{ckpt}"\nexport LOCAL_DIR=',
            1
        )
    else:
        content = re.sub(r'export RESTORE_CHECKPOINT=.*', f'export RESTORE_CHECKPOINT="{ckpt}"', content)

    # Remove unset RESTORE_CHECKPOINT line if it exists (would clobber our set)
    content = re.sub(r'^unset (.*\b)?RESTORE_CHECKPOINT(\b.*)?$\n?', '', content, flags=re.MULTILINE)

    # 2. Bump TIME_TOTAL_S=43200 -> 86400
    content = re.sub(r'TIME_TOTAL_S=43200', 'TIME_TOTAL_S=86400', content)

    # 3. MAX_ITERATIONS
    content = re.sub(r'MAX_ITERATIONS=\d+', f'MAX_ITERATIONS={max_iter}', content)

    # 4. PORT_SEED default
    content = re.sub(r'PORT_SEED=\$\{PORT_SEED:-\d+\}', f'PORT_SEED=${{PORT_SEED:-{port}}}', content)
    # Also for 056 sweep lane/port mapping — for 056E/F skip since they pick PORT_SEED from LANE case

    # 5. RUN_NAME prefix adjustment (add resume prefix)
    content = re.sub(
        r'export RUN_NAME=(\$\{RUN_NAME:-)?([^_}]+_[^}]*)',
        lambda m: f'export RUN_NAME={m.group(1) or ""}{run_prefix}$(date +%Y%m%d_%H%M%S)',
        content, count=1
    )

    # 6. For 056 sweep lane: it reads $LANE — need to set LANE=056E/F in resume if source is sweep script
    # But we can't easily do that here; user must pass LANE env at launch.

    out = os.path.join(OUT_DIR, f"_resume_{lane}.sh")
    with open(out, 'w') as f:
        f.write(content)
    os.chmod(out, 0o755)
    print(f"Created: {out}")

print("\nDONE.")
