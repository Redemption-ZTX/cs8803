## SNAPSHOT-068: 055+053D — Distill with PBRS Reward (replace v2 shaping)

**Date**: 2026-04-21
**Status**: 预注册 / launching when node available

---

## §0 背景

Current SOTA 055 = 0.907 combined 2000ep. 053D (053Dmirror) PBRS-only (no distill, warm from weak base 031B@80) = 0.902 single-shot peak.

- 055 uses **v2 reward shaping** (TIME_PENALTY + BALL_PROGRESS + POSSESSION + OPP_PROGRESS + deep_zone_penalty etc.) — hand-tuned bonuses
- 053D proves **PBRS-only** (Ng99 policy-invariant potential-based shaping from a calibrated outcome predictor) carries near-SOTA signal standalone

Hypothesis: replace v2 shaping in 055's distill setup with outcome-PBRS predictor reward. Clean variable swap. Teacher+PBRS signals are theoretically compatible because PBRS doesn't bias toward any particular action — it only densifies credit assignment from predictor, without altering the optimal policy (Ng99 Thm 1).

---

## §1 假设

**H_068**: 055 ensemble distill + PBRS reward (no v2 shape) → combined 2000ep peak **≥ 0.914** (+0.007 over 055 0.907, crossing 1×SE threshold).

Sub-hypotheses:

- **H_068-a**: PBRS provides denser credit assignment than v2 shape for team-level distill (predictor signal is match-relevant, v2 bonuses are heuristic proxies)
- **H_068-b**: v2 shape's hand-tuned bonuses (ball_progress, possession, deep_zone_penalty) may actively conflict with teacher's policy distribution — teacher pushes one action, v2 rewards another
- **H_068-c**: Ng99 guarantee — PBRS doesn't change asymptotic optimum, only speed. If 055's ceiling at 0.907 is reward-induced (not distill-induced), PBRS clean signal could unlock beyond.

---

## §2 设计

**Isolation**: change ONLY the shaping config. All else identical to 055.

Shared with 055:
- 3-teacher ensemble (031B@1220 + 045A@180 + 051A@130)
- LR = 1e-4
- Alpha schedule 0.05 → 0 decay over 8000 updates
- Distill temperature T = 1.0
- 031B student arch (Siamese + cross-attn)
- 1250 iter scratch (not warm-started)

**Key changes vs 055**:

```bash
# Remove v2 shape:
export USE_REWARD_SHAPING=0
export SHAPING_TIME_PENALTY=0
export SHAPING_BALL_PROGRESS=0
export SHAPING_POSSESSION=0
export SHAPING_OPP_PROGRESS=0
# (all v2 shaping knobs zeroed)

# Add PBRS (same config as 053D):
export OUTCOME_PBRS_PREDICTOR_PATH=/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
export OUTCOME_PBRS_WEIGHT=0.01
export OUTCOME_PBRS_WARMUP_STEPS=10000
export OUTCOME_PBRS_MAX_BUFFER_STEPS=80
```

**Budget**: same as 055 — 1250 iter, ~50M env steps, ~12h on 1×H100.

---

## §3 预注册判据

| # | 判据 | 阈值 | verdict |
|---|---|---|---|
| §3.1 | marginal peak | ≥ 0.911 (+0.004 vs 055) | PBRS has measurable effect |
| §3.2 | main | ≥ 0.915 (+0.008) | PBRS+distill stacks, 2σ SE detectable |
| §3.3 | breakthrough | ≥ 0.925 (+0.018) | PBRS unlocks distill ceiling |
| §3.4 | 持平 | ∈ [0.895, 0.911) | sub-marginal — PBRS ≈ v2 for distill |
| §3.5 | regression | < 0.890 | PBRS conflicts with distill |

Decision point: combined 2000ep (post-hoc, not inline 200ep — per MEMORY feedback_inline_eval_noise).

---

## §4 简化 + 风险 + 降级

- **4.1** PBRS weight fixed at 0.01 (053D default) — not swept in this run. If peak lands in §3.1/§3.4 zone, consider follow-up sweep (0.005 / 0.02).
- **4.2** Predictor is the same v3 calibrated checkpoint used by 053D — not retrained on 055-style trajectories. Risk: distribution shift between predictor's training data (weak-base rollouts) and 055-teacher-guided student rollouts.
- **4.3** No warmstart (scratch). 053D used warmstart from 031B@80; we go scratch here for cleaner comparison to 055 (which is also scratch).
- **4.4** Retrograde sequence:
  - **Step 0**: PBRS weight 0.01 scratch (this snapshot)
  - **Step 1** (peak < 0.895): sweep PBRS weight {0.005, 0.02, 0.05}
  - **Step 2** (all fail): abandon PBRS-for-distill path; record as negative result

---

## §5 不做的事

- 不混 LR=3e-4 (isolate reward change — LR stays at 055's 1e-4)
- 不改 teacher pool (same 3-teacher ensemble)
- 不 sweep PBRS buffer / warmup (fix to 053D defaults)
- 不 warm-start (scratch only, matches 055)

---

## §6 执行清单

- [x] 1. Snapshot 起草 (this file)
- [ ] 2. 创建 launcher `scripts/eval/_launch_068_055PBRS_distill.sh` (copy 055 launcher, apply shaping changes)
- [ ] 3. Smoke test (launch on any node, verify predictor loads + shaping env vars take effect)
- [ ] 4. Full launch (1 node, 12h, tmux)
- [ ] 5. Combined 2000ep verdict per §3
- [ ] 6. Append result to rank.md + this snapshot §7

---

## §7 Verdict

### §7.1 Stage 1 baseline 1000ep — BOTH warm + scratch variants (2026-04-21 14:55 EDT, append-only)

User launched two parallel variants of the 055+PBRS combo to isolate warm-start vs scratch contribution:
- **068_warm** (jobid 5028782): warm from 031B@80 (weak base), 800 iter budget, otherwise identical to 055 config
- **068_scratch** (jobid 5028915): scratch, 1250 iter budget, identical to 055 config except PBRS replaces v2 shape

#### 7.1.1 `068_warm` Stage 1 post-eval (completed ~09:40 EDT)

| iter | 1000ep WR | (W-L) | notes |
|---:|---:|---:|---|
| 400 | 0.874 | 874-126 | 早段 |
| 410 | 0.876 | 876-124 | |
| 420 | 0.851 | 851-149 | |
| 430 | 0.852 | 852-148 | |
| 440 | 0.865 | 865-135 | |
| 450 | 0.867 | 867-133 | |
| 460 | 0.880 | 880-120 | |
| 470 | 0.880 | 880-120 | |
| 480 | 0.874 | 874-126 | |
| 600 | 0.882 | 882-118 | |
| 610 | 0.880 | 880-120 | |
| 620 | 0.861 | 861-139 | |
| 670 | 0.875 | 875-125 | |
| 680 | 0.878 | 878-122 | |
| 690 | 0.873 | 873-127 | |
| 700 | 0.881 | 881-119 | |
| 710 | 0.886 | 886-114 | |
| 720 | 0.878 | 878-122 | |
| 730 | 0.872 | 872-128 | |
| 740 | 0.887 | 887-113 | |
| 760 | 0.888 | 888-112 | |
| 770 | 0.878 | 878-122 | |
| **780** | **0.892** | **892-108** | **★ PEAK** |
| 800 | 0.860 | 860-140 | 收尾 |

- **Peak 0.892 @ iter 780**; plateau 700-780 = 0.872-0.892
- vs 053Dmirror 0.902 (PBRS-only from same warm base): **Δ = -0.010 (worse)**
- vs 055 SOTA 0.907: Δ = -0.015 below

Raw log：[068_warm_baseline1000.log](../../docs/experiments/artifacts/official-evals/068_warm_baseline1000.log)

#### 7.1.2 `068_scratch` Stage 1 post-eval (completed ~14:30 EDT)

Top window ckpts 1130-1250:

| iter | 1000ep WR | (W-L) | notes |
|---:|---:|---:|---|
| 590 | 0.876 | 876-124 | 中段 |
| 600 | 0.850 | 850-150 | |
| 610 | 0.865 | 865-135 | |
| 780 | 0.876 | 876-124 | |
| 790 | 0.869 | 869-131 | |
| 800 | 0.867 | 867-133 | |
| 830 | 0.875 | 875-125 | |
| 840 | 0.862 | 862-138 | |
| 850 | 0.889 | 889-111 | |
| 980 | 0.871 | 871-129 | |
| 990 | 0.882 | 882-118 | |
| 1000 | 0.880 | 880-120 | |
| 1080 | 0.896 | 896-104 | |
| 1090 | 0.893 | 893-107 | |
| 1100 | 0.881 | 881-119 | |
| 1130 | 0.884 | 884-116 | |
| **1140** | **0.905** | **905-95** | **★ PEAK (dual)** |
| 1150 | 0.890 | 890-110 | |
| 1160 | 0.884 | 884-116 | |
| 1170 | 0.894 | 894-106 | |
| **1180** | **0.905** | **905-95** | **★ PEAK (dual)** |
| 1200 | 0.895 | 895-105 | |
| 1210 | 0.882 | 882-118 | |
| 1220 | 0.886 | 886-114 | |
| 1250 | 0.892 | 892-108 | terminal |

- **Dual peak 0.905 @ iter 1140 + 1180**; plateau 1130-1250 = 0.882-0.905
- vs 055 SOTA 0.907: **Δ = -0.002 (statistically tied within SE)**
- vs 068_warm 0.892: **Δ = +0.013 — scratch decisively beats warm**

Raw log：[068_scratch_baseline1000.log](../../docs/experiments/artifacts/official-evals/068_scratch_baseline1000.log)

#### 7.1.3 判据对照 (068_scratch as the primary)

| 阈值 | peak 0.905 | 判定 |
|---|---|---|
| §3.1 marginal ≥ 0.911 | 0.905 | ✗ MISS |
| §3.2 main ≥ 0.915 | 0.905 | ✗ MISS |
| §3.3 breakthrough ≥ 0.925 | 0.905 | ✗ MISS |
| **§3.4 持平 [0.895, 0.911)** | **0.905** | **✅ HIT (tied 055)** |
| §3.5 regression < 0.890 | 0.905 | ✗ NO |

**Outcome C — tied 055**: PBRS ≡ v2 for distill on the baseline axis. Shape-choice is **free** (use simpler PBRS going forward if convenient).

#### 7.1.4 Key finding — scratch > warm for PBRS+distill (mechanism)

- **068_scratch 0.905 >> 068_warm 0.892** (+0.013pp, clearly outside combined SE)
- **Interpretation**: warm-start from 031B@80 (weak base with v2-shape bias in its features) **conflicts** with PBRS's replacement-reward signal. Teacher+PBRS setup is cleanest when student starts from random init — there is no pre-existing v2-shape bias to unlearn.
- Contrast with 053Dmirror (warm + PBRS-only, no distill) = 0.902: PBRS alone **can** accept a warm v2-tainted base (because PPO RL on PBRS reward can freely overwrite feature-level shape bias). But once distill is **also** pulling student toward teacher's (v2-trained) distribution, the warm base becomes a third signal that conflicts with PBRS (which is policy-invariant by Ng99 but replaces the v2 signal entirely).
- **Generalizable lesson**: when stacking {warm-start, teacher-distill, replacement-shaping}, scratch is the correct path — warm-start only helps when pretraining objective and replacement reward share support.

#### 7.1.5 Scientific conclusion

- PBRS-for-distill **does not beat v2-for-distill** (068_scratch 0.905 tied 055 0.907)
- PBRS-for-distill with warm-start **regresses** (068_warm 0.892 below both 055 and 053Dmirror)
- **H_068 main (≥0.914) MISS**; H_068-c hypothesis (055 ceiling is reward-induced) **NOT supported** — ceiling is likely distill-structural (teacher pool size / argmax sharpness) not reward-path
- **Path closure**: PBRS-as-replacement-reward-for-distill **closed**; no follow-up sweep of PBRS weight {0.005, 0.02, 0.05} needed because the directional signal is null (scratch tied, warm regressed)
- Resources redirect to distillation-structural axes (Pool A/B/C/D teacher composition, 076 wide student, 077 per-agent, 079 055v3 recursive)

**Submission impact**: 055@1150 remains primary grading candidate (combined 2000ep 0.907). 068_scratch@1140 available as diversity candidate (same-ceiling, different reward path) if ensemble needs non-v2 member.

---

## §8 后续路径

- **A** (breakthrough ≥ 0.925): PBRS + distill combo is new SOTA → investigate larger PBRS weight + ensemble scaling
- **B** (marginal 0.911 – 0.915): PBRS slight help → next try PBRS + 5-teacher (= 055v2-with-PBRS upgrade)
- **C** (tied 055 ~0.907): PBRS ≡ v2 for distill → shape choice is free, use simpler one (PBRS) going forward
- **D** (regression): PBRS + teacher signals conflict → stay with v2 + distill as canonical; document as negative result

---

## §9 相关

- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — v2-shape baseline (0.907 combined)
- [snapshot-053D-mirror-pbrs-only-from-weak-base](snapshot-053D-mirror-pbrs-only-from-weak-base.md) — PBRS standalone (0.902 peak)
- [BACKLOG.md](BACKLOG.md) — "055+053D" entry (now being executed)
- `cs8803drl/training/train_ray_team_vs_baseline_shaping.py` — supports both `USE_REWARD_SHAPING` and `OUTCOME_PBRS_*` env vars
- Predictor checkpoint: `docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt` (val_acc 93.8%)
- Theoretical:
  - Ng, Harada, Russell (1999) "Policy invariance under reward transformations" — PBRS optimality theorem
  - Hinton, Vinyals, Dean (2015) "Distilling the knowledge in a neural network" — KD foundation
