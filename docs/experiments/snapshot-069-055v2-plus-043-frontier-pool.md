## SNAPSHOT-069: 055v2 + 043 — Recursive Distill + Frontier Pool Opponent

**Date**: 2026-04-21
**Status**: 预注册 / launching

## Context

055v2 (5-teacher recursive distill + LR=3e-4, combined 3000ep 0.909, TIED with 055 SOTA) used BASELINE_PROB=1.0 (100% baseline opponent). 043A' lane showed that mixing frontier pool opponents (OPPONENT_POOL_FRONTIER_SPECS + OPPONENT_POOL_BASELINE_PROB=0.50) with warm-start gave 043A'@080 = 0.900.

**Hypothesis**: stack 055v2's recursive distill + 043A's frontier pool opponent → student learns from:
1. 5-teacher ensemble output (KL distill signal)
2. Mix of baseline + frontier opponents (diverse exploration, frontier attack patterns)
Expected: combo may unlock above current 0.91 ceiling.

### §0 背景
- 055v2 combined 3000ep 0.909 (5-teacher distill + LR=3e-4, 100% baseline opp)
- 043A' combined 0.900 (031B@1220 warm-start vs frontier pool, 50% baseline + 50% frontier)
- Both paths ~SOTA-tier (~0.91 plateau). Stacking distill + frontier pool = untried combo.
- Hypothesis: H2H showed 055 beats 029B @0.696 (cross-arch), 031B @0.620. Adding these as opponents during training may teach student to exploit their failure modes → higher WR on ultimate baseline eval.

### §1 假设
**H_069**: 055v2 + frontier pool opponent → combined 2000ep peak ≥ 0.920 (+0.011 vs 055v2 0.909).

Sub-hypotheses:
- H_069-a: frontier opponents force student to learn more robust strategies (Supervised by teacher KL + diverse actual play)
- H_069-b: 50% baseline preserves grading-target skill
- H_069-c: frontier pool contains strong team-level SOTA ckpts → student gets "adversarial training" against same-arch opponents

### §2 设计

**Isolation**: take 055v2 config, add OPPONENT_POOL_FRONTIER_SPECS + change BASELINE_PROB mechanism.

**Changes from 055v2 base**:
```bash
# 055v2 original:
# BASELINE_PROB=1.0

# 069 changes:
export BASELINE_PROB=1.0                        # base per-env opponent type selection
export OPPONENT_POOL_BASELINE_PROB=0.50         # 50% baseline, 50% frontier-pool
export OPPONENT_POOL_FRONTIER_SPECS="\
team_ckpt|team_frozen_team_ray|0.4|<path to 028A@1060>,\
team_ckpt|team_frozen_team_ray|0.3|<path to 031A@1040>,\
team_ckpt|team_frozen_team_ray|0.3|<path to 029B@190>"
```

(Format: `name|kind|weight|ckpt_path`, weights sum to 1 for the frontier pool portion)

**Frontier pool** (4 team-level champions):
- 028A@1060 (team-level BC bootstrap): weight 0.3 (older/weaker)
- 031A@1040 (team-level dual-encoder scratch): weight 0.3
- 031B@1220 (team-level 031B base): weight 0.4 (strongest team frontier)
- (Could also add 029B@190 but per-agent not team-level — skip for now)

All else identical to 055v2:
- 5-teacher ensemble (055@1150 + 031B@1220 + 045A@180 + 051A@130 + 056D@1140)
- LR=3e-4, clip 0.15
- v2 shaping (as 055v2 uses)
- 1250 iter scratch
- Team-level 031B arch

### §3 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal peak ≥ 0.914 | +0.005 vs 055v2 0.909 | frontier pool adds small gain |
| §3.2 main ≥ 0.920 | +0.011 | combo detectable 2σ SE |
| §3.3 breakthrough ≥ 0.930 | +0.021 | 055v2+043 is SOTA path |
| §3.4 持平 [0.895, 0.914) | sub-marginal | pool doesn't help recursive distill |
| §3.5 regression < 0.890 | pool hurts | frontier opponents distract student from baseline-target |

### §4 简化 + 风险 + 降级

#### 4.1 简化 — 单 pool config
- Only one frontier pool set (3 ckpts with fixed weights). Could sweep weights.
- Downgrade: if marginal, sweep weight distribution 0.4/0.4/0.2 vs 0.5/0.3/0.2 etc.

#### 4.2 简化 — fixed opponent pool baseline ratio 0.50
- 043A' used this default. Could be 0.3 or 0.7.
- Downgrade: sweep ratio if main hypothesis missed.

#### 4.3 简化 — scratch (not warm-start from 031B@80)
- User's directive was 055v2+043. 055v2 was scratch. Keep scratch for cleaner comparison.
- 043A' was WARM from 031B@1220. If needed, try warm variant later.

#### 4.4 风险 — teacher+opponent signal conflict
- 5-teacher ensemble KL pulls student toward teachers' distributions
- Frontier opponents try to defeat student (reward-pulling)
- If teachers' policies inflexible vs diverse opponents → conflict, student learns poorly
- Mitigation: α schedule decays distill weight over training; late game dominated by PPO on actual (baseline + frontier) rewards.

#### 4.5 Retrograde sequence
- Step 0: 069 scratch with 3-frontier pool (031B@1220 + 031A@1040 + 028A@1060)
- Step 1 (peak < 0.914): sweep pool composition (add 043A'@080, remove 028A)
- Step 2 (step 1 fails): sweep pool baseline ratio (0.3, 0.7)
- Step 3 (all fail): 069 path closed, frontier pool doesn't help distill

### §5 不做的事
- 不混 PBRS (isolate frontier-pool from reward changes)
- 不 warm-start (keep scratch for cleaner vs 055v2 baseline)
- 不改 5-teacher ensemble composition (055@1150 + 034e originals + 056D)

### §6 执行清单
- [x] 1. Snapshot 起草 (this file)
- [ ] 2. Launcher `_launch_069_055v2_plus_043_frontier.sh` creation
- [ ] 3. Launch on free R node
- [ ] 4. Verdict per §3

### §7 Verdict

#### 7.1 Stage 1 baseline 1000ep (completed 2026-04-21 14:40 EDT, append-only)

Trial **5028750** (launched ~04:27 EDT, completed ~12:30 EDT); 18 ckpts selected (top plateau + terminal):

| iter | 1000ep WR | (W-L) | notes |
|---:|---:|---:|---|
| 550 | 0.873 | 873-127 | |
| 560 | 0.896 | 896-104 | |
| 570 | 0.880 | 880-120 | |
| 690 | 0.894 | 894-106 | |
| 700 | 0.867 | 867-133 | |
| 710 | 0.868 | 868-132 | |
| 720 | 0.886 | 886-114 | |
| 730 | 0.879 | 879-121 | |
| 740 | 0.875 | 875-125 | |
| 750 | 0.863 | 863-137 | |
| 760 | 0.881 | 881-119 | |
| 980 | 0.885 | 885-115 | |
| 990 | 0.898 | 898-102 | |
| 1000 | 0.870 | 870-130 | |
| 1090 | 0.904 | 904-96 | 次 peak |
| 1100 | 0.887 | 887-113 | |
| 1110 | 0.886 | 886-114 | |
| **1250** | **0.908** | **908-92** | **★ PEAK (terminal)** |

- **Peak 0.908 @ terminal ckpt 1250**; plateau 1000-1250 = 0.870-0.908
- Trajectory: steady mid-0.87 through 700-1000, late-window climb to 0.90+ at 1090/1250

Raw log：[069_baseline1000.log](../../docs/experiments/artifacts/official-evals/069_baseline1000.log)

#### 7.2 判据对照

| 阈值 | peak 0.908 | 判定 |
|---|---|---|
| §3.1 marginal ≥ 0.914 | 0.908 | ✗ MISS (-0.006) |
| §3.2 main ≥ 0.920 | 0.908 | ✗ MISS |
| §3.3 breakthrough ≥ 0.930 | 0.908 | ✗ MISS |
| **§3.4 持平 [0.895, 0.914)** | **0.908** | **✅ HIT (tied 055)** |
| §3.5 regression < 0.890 | 0.908 | ✗ NO |

**vs 055@1150 combined 2000ep SOTA 0.907**: **Δ = +0.001 (essentially tied within SE 0.009 single-shot)**
**vs 055v2 combined 3000ep 0.909**: Δ = -0.001 (tied, Δ < SE 0.005)

#### 7.3 科学结论 — frontier-opponent hypothesis partially validated, no breakthrough

- 055v2 recipe (5-teacher recursive distill + LR=3e-4 + v2) **+ 40%baseline/60%frontier pool opponent** → single-shot 1000ep 0.908
- Matches 055 SOTA on baseline axis → hypothesis **H_069-b (50% baseline preserves grading-target skill)** CONFIRMED
- But NO breakthrough over 055v2's 0.909 pure-baseline-opp setting → hypothesis **H_069-a (frontier opponents force robust strategies)** NOT supported at combined-sample level — frontier exposure did not push peak above recursive-distill ceiling
- "Frontier opponent = general strength = baseline strength" conjecture **partially validated** (no regression from mixing frontier opp) **but no actionable uplift**
- Consistent with 055v2 family plateau ceiling at ~0.907-0.909 and 074 family ceiling ~0.900-0.903 → **current project SOTA ceiling is distillation-structural, not opponent-composition-driven**

#### 7.4 Path decision

- 069 path **closed** as "tied SOTA"; no Step 1 pool-composition sweep, no Step 2 ratio sweep
- Optional: keep 069@1250 as diversity candidate if ensemble needs a non-pure-baseline-opp member (zero-training ensemble like 074 family already closed but re-entry pending new members)
- Stage 2 capture (500ep) + Stage 3 H2H vs 055 → will queue as part of 7-parallel post-eval batch (see task-queue-20260421 §6)
- Combined 2000ep pending (would need +1000ep rerun on ckpt 1250); ROI low since single-shot already lands mid-persistent plateau with 055 — unlikely to shift verdict

#### 7.5 Submission impact

055@1150 remains primary grading candidate. 069@1250 is a tied-SOTA independent-recipe candidate for diversity (frontier-pool-trained vs pure-baseline-trained) if needed for mixed-member ensemble reconsideration. No resubmission required.

### §8 后续路径
- Outcome A (≥0.920): frontier-pool + distill is new SOTA path. Extend with more frontier ckpts.
- Outcome B (persistent ~0.909): pool doesn't help recursive distill (teacher dominates signal)
- Outcome C (regression <0.890): pool conflicts with teacher, close path

### §9 相关
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 base
- [snapshot-061](snapshot-061-055v2-recursive-distill.md) — 055v2 parent
- [snapshot-043-frontier-selfplay-pool](snapshot-043-frontier-selfplay-pool.md) — frontier pool base
- [BACKLOG.md](BACKLOG.md) — "055v2+043" entry now executing
- Code: `cs8803drl/training/train_ray_team_vs_baseline_shaping.py` reads `OPPONENT_POOL_FRONTIER_SPECS` (format "name|kind|weight|ckpt" comma-separated) and `OPPONENT_POOL_BASELINE_PROB`
- Theoretical: self-play + KD combo — Silver 2017 (AlphaZero self-play), Rusu 2016 (PD from experts); Allen-Zhu & Li 2020 (ensemble distill stability)
