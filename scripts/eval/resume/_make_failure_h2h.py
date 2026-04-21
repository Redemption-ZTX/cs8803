#!/usr/bin/env python3
"""Generate 5 failure-capture + 4 H2H launchers for plateau-family failure-mode analysis.

Peak ckpts (combined 1000ep or 2000ep verified where available):
  055 @1150 — SKIP (already captured)
  055v2 @1000 (combined 3000ep 0.909)
  062a @1220 (combined 2000ep 0.892)
  056D @1140 (combined 2000ep 0.891)
  062c @1090 (combined 2000ep 0.886)
"""
import os

OUT_DIR = "/home/hice1/wsun377/Desktop/cs8803drl/scripts/eval/resume"
os.makedirs(OUT_DIR, exist_ok=True)

CKPTS = {
    "055v2_1000": "/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001000/checkpoint-1000",
    "062a_1220": "/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220",
    "056D_1140": "/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140",
    "062c_1090": "/storage/ice1/5/1/wsun377/ray_results_scratch/062c_curriculum_noshape_adaptive_0_300_700_1100_20260420_142916/TeamVsBaselineShapingPPOTrainer_Soccer_e37fc_00000_0_2026-04-20_14-29-38/checkpoint_001090/checkpoint-1090",
    "055_1150": "/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150",
}

# --- 4 failure capture launchers ---
CAPTURE_V2_FLAGS = """
  --time-penalty 0.001 \\
  --ball-progress-scale 0.01 \\
  --goal-proximity-scale 0.0 \\
  --progress-requires-possession 0 \\
  --opponent-progress-penalty-scale 0.01 \\
  --possession-dist 1.25 \\
  --possession-bonus 0.002 \\
  --deep-zone-outer-threshold -8 \\
  --deep-zone-outer-penalty 0.003 \\
  --deep-zone-inner-threshold -12 \\
  --deep-zone-inner-penalty 0.003 \\
  --defensive-survival-threshold 0 \\
  --defensive-survival-bonus 0 \\
  --fast-loss-threshold-steps 0 \\
  --fast-loss-penalty-per-step 0 \\
  --event-shot-reward 0.0 \\
  --event-tackle-reward 0.0 \\
  --event-clearance-reward 0.0 \\
  --event-cooldown-steps 10"""

# 062a/062c used USE_REWARD_SHAPING=0, 055v2/056D used v2. 
# For capture, just match each's own shaping for consistency
CAPTURES = [
    ("055v2_1000", 38005),
    ("062a_1220", 37005),
    ("056D_1140", 36005),
    ("062c_1090", 35005),
]

for tag, port in CAPTURES:
    ckpt = CKPTS[tag]
    save_dir = f"docs/experiments/artifacts/failure-cases/{tag}_baseline_500"
    log = f"docs/experiments/artifacts/official-evals/failure-capture-logs/{tag}.log"
    content = f"""#!/bin/bash
# Failure capture: {tag} vs baseline 500ep (save losses for bucket analysis)
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/failure-cases/{tag}_baseline_500
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \\
  --checkpoint "{ckpt}" \\
  --team0-module cs8803drl.deployment.trained_team_ray_agent \\
  --opponent baseline \\
  -n 500 \\
  --max-steps 1500 \\
  --base-port {port} \\
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/{tag}_baseline_500 \\
  --save-mode losses \\
  --max-saved-episodes 500 \\
  --trace-stride 10 \\
  --trace-tail-steps 30 \\
  --reward-shaping-debug \\
{CAPTURE_V2_FLAGS} \\
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/{tag}.log
exit ${{PIPESTATUS[0]}}
"""
    path = os.path.join(OUT_DIR, f"_capture_{tag}.sh")
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, 0o755)
    print(f"Capture: {path}")

# --- 4 H2H launchers ---
# Each: m1 = first ckpt, m2 = second (all team-level so use trained_team_ray_agent for m1 and trained_team_ray_opponent_agent for m2)
H2HS = [
    ("055v2_1000_vs_055_1150", "055v2_1000", "055_1150", 34005),
    ("062a_1220_vs_055_1150", "062a_1220", "055_1150", 33005),
    ("062c_1090_vs_055_1150", "062c_1090", "055_1150", 32005),
    ("062a_1220_vs_056D_1140", "062a_1220", "056D_1140", 31005),
]

for name, m1_tag, m2_tag, port in H2HS:
    m1 = CKPTS[m1_tag]
    m2 = CKPTS[m2_tag]
    content = f"""#!/bin/bash
# H2H: {m1_tag} vs {m2_tag} (500ep, port {port})
set -o pipefail
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
export TRAINED_RAY_CHECKPOINT="{m1}"
export TRAINED_TEAM_OPPONENT_CHECKPOINT="{m2}"
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \\
  -m1 cs8803drl.deployment.trained_team_ray_agent \\
  -m2 cs8803drl.deployment.trained_team_ray_opponent_agent \\
  -e 500 -p {port} \\
  2>&1 | tee docs/experiments/artifacts/official-evals/headtohead/{name}.log
exit ${{PIPESTATUS[0]}}
"""
    path = os.path.join(OUT_DIR, f"_h2h_{name}.sh")
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, 0o755)
    print(f"H2H: {path}")

print("\nDONE.")
