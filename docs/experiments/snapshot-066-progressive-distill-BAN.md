## SNAPSHOT-066: Progressive Distill (BAN-style multi-stage) (Tier 3b)

- **Date**: 2026-04-20
- **Status**: 预注册 / code pending (deferred to next day after 055-temp/055v2 verdict)

### §0 背景

- 055 is current Project SOTA (combined 2000ep 0.907, 031B arch + 3-teacher 034e ensemble + T=1.0 + LR=1e-4).
- Tier 3a (055-temp, T=2.0) tests temperature — expected +0.005–0.015pp yield, running now (~10h left).
- 056-PBT-full **PAUSED** due to 60h GPU cost + uncertain yield.
- **Progressive Distill / BAN (Born Again Networks, Furlanello et al. 2018)**: student_t becomes teacher for student_{t+1}. Iterative distillation with no new external teacher.
- Theoretically: self-distillation smooths output distribution, improves generalization. Furlanello 2018 shows +2–5% on CIFAR-100 / language modeling over single-stage KD.
- For RL policy distillation: Rusu 2016 showed iterative distillation maintains policy quality; combined with ensemble teaching (055) this could stack.

### §1 假设

**H_066** (primary): 2-stage progressive distill where 055@1150 (current SOTA) is used as the SOLE teacher (or mixed with 034e ensemble) for a new student → combined 2000ep peak ≥ 0.920 (+0.013pp vs 055 0.907, above 2σ SE).

**Sub-hypotheses**:

- **H_066-a**: Self-distillation from 055@1150 alone (single-teacher path) produces student that inherits + refines policy. Hinton 2015 + Furlanello 2018.
- **H_066-b**: Mixing 055@1150 + 034e 3-teacher ensemble into 4-teacher pool (weighted by recency: 055 gets 0.5, 3× 034e gets 0.5/3 each) may beat pure 4-teacher equal average.
- **H_066-c**: Temperature T=1.0 is used for stage 2 (since stage-1 model 055 is itself softer than raw baseline); can sweep T in later stage.

### §2 设计

#### 2.1 Infrastructure

- Same training script as 055 (`train_ray_team_vs_baseline_shaping.py`) with `TEAM_DISTILL_ENSEMBLE_KL=1`.
- **No new engineering needed** — leverages existing ensemble-distill infrastructure + Option B teacher-ckpt sanitization (already in place).
- Single node training, 1 GPU.

#### 2.2 Design of 2-stage schedule

**Stage 1 (055 itself, already done)**: 055@1150 = product of distill from 034e ensemble (031B@1220 + 045A@180 + 051A@130), T=1.0, LR=1e-4, 1250 iter. Peak combined 2000ep 0.907.

**Stage 2 (this snapshot)**: 2 variants to test.

**Variant A (066A "pure self-distill")**:

- Teacher: ONLY 055@1150 (single model)
- Student arch: 031B (same as 055)
- LR: 1e-4 (same as 055 stage 1)
- T: 1.0 (baseline); later L2 can try T=2.0 if 066A tied 055
- Alpha schedule: same as 055 (init 0.05, decay to 0 by 8000 updates)
- 1250 iter scratch

**Variant B (066B "weighted ensemble refresh")**:

- Teacher: 4-teacher ensemble {055@1150 weight 0.5, 031B@1220 weight 0.5/3, 045A@180 weight 0.5/3, 051A@130 weight 0.5/3}
- **Weighted averaging**: not equal mean — 055 dominates (recency prior)
- Requires small code tweak: `TEAM_DISTILL_TEACHER_WEIGHTS="0.5,0.166,0.166,0.166"` env var
- LR=1e-4, T=1.0
- 1250 iter scratch

#### 2.3 Code additions required

- **Variant A**: no code change — existing ensemble infrastructure supports single-teacher. Launcher just lists 1 ckpt in `TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS`.
- **Variant B**:
  - Add `TEAM_DISTILL_TEACHER_WEIGHTS` env var to `team_siamese_distill.py` (parse comma-sep floats, default equal).
  - Modify `_FrozenTeamEnsembleTeacher.forward` to do weighted factor-prob average instead of equal average.
  - ~30–60 min code + smoke test.
- **Time total**: 1h code + 2 lanes × ~12h train = 25h GPU.

### §3 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.911 | +0.004 vs 055 | BAN has measurable effect |
| §3.2 main: peak ≥ 0.920 | +0.013 | BAN detectable at 2σ SE; **H_066 met** |
| §3.3 breakthrough: peak ≥ 0.930 | +0.023 | iterative distillation is major SOTA path |
| §3.4 持平 [0.895, 0.911) | sub-marginal | BAN doesn't help vs single-stage |
| §3.5 regression: peak < 0.890 | 自 distill 反伤 | teacher 055's own biases amplified in student |

Compare 066A vs 066B:

- 066A > 066B: self-distillation alone is cleanest path
- 066A ≈ 066B: teacher pool structure doesn't matter much
- 066B > 066A: recency-weighted ensemble better than pure self-distill

### §4 简化 + 风险 + 降级

#### 4.1 简化 A — Only 2 variants (not 3–4 stage cascade)

- Full BAN original paper does 4+ stages (student_1 → student_2 → … → student_N).
- Due to GPU budget, only 1 iteration (stage 1 = 055, stage 2 = 066A/B).
- **降级**: If one of 066A/B breaks SOTA, try stage 3 (066C = distill from 066A-or-B).

#### 4.2 简化 B — No temperature sweep in stage 2

- T=1.0 used for isolation from 055-temp (Tier 3a) experiment.
- **降级**: If 066A/B tied 055, L2 sweep T=2.0.

#### 4.3 简化 C — No LR sweep in stage 2

- LR=1e-4 same as 055.
- If 056E/F shows higher LR wins, consider 066A/B with LR=3e-4 as L3.

#### 4.4 Risk — teacher-copy problem

- If student 055@1150 has idiosyncratic biases, self-distillation may amplify them (H_066 null hypothesis).
- Mitigation: 066B's ensemble mix averages out single-teacher bias.
- **降级**: If both 066A/B regress, progressive distill path closed for this arch/config.

#### 4.5 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | 066A (pure) + 066B (weighted) parallel | base 25h |
| 1 | Both < 0.911 | Temp sweep on best variant (T=1.5/2.0/3.0) | +36h × 3 |
| 2 | Step 1 fail | LR sweep on best variant (LR=3e-4) | +12h |
| 3 | Step 2 fail | Progressive stage 3 (iterate once more) | +12h |
| 4 | Step 3 fail | BAN path closed; return to ensemble/temperature |

### §5 不做的事

- 不与 055v2 同时 launch 新 variant (055v2 已是 5-teacher including 055@1150,结果等出再决定 066 叠加策略).
- 不做 temperature sweep in stage 2 (isolate 变量).
- 不改 architecture (保持 031B).

### §6 执行清单

**Phase 1: 066A (pure self-distill, 0 code change)**

- [ ] Create launcher `_launch_066A_pure_selfdistill_from_055.sh` (copy 055 launcher, teacher = only 055@1150)
- [ ] Smoke test (model import, teacher load via Option B)
- [ ] Launch (1 GPU, ~12h)

**Phase 2: 066B (weighted ensemble)**

- [ ] Add `TEAM_DISTILL_TEACHER_WEIGHTS` parsing to `team_siamese_distill.py` `_FrozenTeamEnsembleTeacher`
- [ ] Smoke test weighted averaging (verify weights sum to 1, probs correctly weighted)
- [ ] Create launcher `_launch_066B_weighted_4teacher_from_055.sh`
- [ ] Launch (1 GPU, ~12h)

**Phase 3: Verdict**

- [ ] Stage 1 1000ep post-eval per variant
- [ ] Combined 2000ep rerun if peak near marginal threshold
- [ ] H2H vs 055@1150 (direct self-improvement check)

### §7 Verdict

#### §7.1 2026-04-21 15:30 EDT — 066A (pure self-distill from 055@1150) Stage 1 single-shot 1000ep — Outcome B (TIED 055), **LANE CLOSED**

**Setup**: Pure single-teacher self-distillation, teacher = 055@1150 only (no ensemble), student = 031B arch scratch, all other hyperparameters match 055 Pool A exactly. Standard BAN (Born Again Network) pattern — student shares teacher architecture.

**Raw Recap top-8 (official evaluator parallel, single-shot 1000ep each)**:

```
=== Official Suite Recap top-8 ===
ckpt 1180 vs baseline: win_rate=0.909   [★ peak]
ckpt 890  vs baseline: win_rate=0.894
ckpt 1210 vs baseline: win_rate=0.894
ckpt 1100 vs baseline: win_rate=0.894
ckpt 1090 vs baseline: win_rate=0.894
ckpt 860  vs baseline: win_rate=0.893
ckpt 1190 vs baseline: win_rate=0.891
ckpt 1160 vs baseline: win_rate=0.889
```

Log: `docs/experiments/artifacts/official-evals/066A_baseline1000.log`

**预注册判据 (§3/§8) 对照**:

| Outcome | 阈值 | single-shot @1180 | verdict |
|---|---:|---:|---|
| Outcome A (breakthrough ≥ 0.920) | progressive distill is the path | 0.909 | **MISS** (-0.011) |
| Outcome B (tied 0.907-0.919) | moderate gain, stack 066 with 055-temp/LR | 0.909 | **HIT** |
| Outcome C (regression < 0.895) | self-distill unhelpful, close path | 0.909 | NO |

**Δ vs reference**:
- vs **055@1150 SOTA combined 2000ep 0.907**: Δ = **+0.002 statistically tied within SE**
- BAN literature promises student > teacher by ~1pp on image classification; **on RL policy distillation with Multi-Discrete 6×3 action space, self-distill preserves but does not improve**

**Verdict**: Pure single-teacher self-distill **saturates at teacher's own WR**. Distill paradigm does not produce BAN-style student-beats-teacher uplift for this policy setup. Consistent with theoretical expectation: self-distillation with identical arch relies on teacher having suboptimal soft-target noise that student can denoise — but 055@1150 is already near this family's WR ceiling, so there is no residual noise to denoise.

**Lane 066A CLOSED** (Outcome B). Not worth combined rerun or H2H (tied with 055 on baseline axis — any H2H diff would be within noise of coin-flip 0.50).

#### §7.2 2026-04-21 15:30 EDT — 066B (weighted 4-teacher progressive distill) Stage 1 single-shot 1000ep — Outcome B (TIED 055), **LANE CLOSED**

**Setup**: Weighted 4-teacher progressive distill, teacher = {055@1150 + 034E-frontier members}, weights tuned so 055 dominates (weight ~0.55) with 034E ensemble members supplying residual diversity. Student = 031B arch scratch, same base config as 066A.

**Raw Recap top-7 (official evaluator parallel, single-shot 1000ep each)**:

```
=== Official Suite Recap top-7 ===
ckpt 940  vs baseline: win_rate=0.905   [★ peak]
ckpt 1070 vs baseline: win_rate=0.904
ckpt 820  vs baseline: win_rate=0.903
ckpt 950  vs baseline: win_rate=0.899
ckpt 910  vs baseline: win_rate=0.899
ckpt 830  vs baseline: win_rate=0.898
ckpt 1240 vs baseline: win_rate=0.896
```

Log: `docs/experiments/artifacts/official-evals/066B_baseline1000.log`

Plateau 820-1240 spans 420 iter at [0.895, 0.905] — structurally stable.

**预注册判据 (§3/§8) 对照**:

| Outcome | 阈值 | single-shot @940 | verdict |
|---|---:|---:|---|
| Outcome A (breakthrough ≥ 0.920) | progressive distill is the path | 0.905 | **MISS** (-0.015) |
| Outcome B (tied 0.907-0.919) | moderate gain, stack 066 with 055-temp/LR | 0.905 | **HIT boundary** (just below lower edge) |
| Outcome C (regression < 0.895) | self-distill unhelpful, close path | 0.905 | NO |

**Δ vs reference**:
- vs **055@1150 SOTA combined 2000ep 0.907**: Δ = **-0.002 statistically tied within SE**
- vs **066A (pure self-distill) 0.909**: Δ = -0.004 **tied**, weighted multi-teacher no lift over pure self-distill

**Verdict**: Weighted multi-teacher progressive distill (Rusu 2016 kickstarting / BAN variant) **also fails to break 0.91**. Adding 034E ensemble members as weighted soft targets on top of 055 teacher neither helps nor hurts — matches the pattern observed in snapshot-061 (055v2 5-teacher recursive distill saturated at same plateau).

**Combined result of §7.1 + §7.2 — Progressive distill paradigm CLOSED**:
- Pure self-distill (066A) = 0.909 tied
- Weighted 4-teacher (066B) = 0.905 tied
- Both saturate at 055 SOTA plateau — **teacher-count expansion and reweighting are not the path above 0.91**
- Mechanism implication: future distillation improvements must come from (a) architectural diversity (054M, wide-student, per-agent student), (b) DAGGER-style online correction (078), or (c) extending the training horizon past the canonical 1250 iter (055v2_extend's unique plateau ≥ 0.911 at iter 1710-1860 is preliminary evidence)
- Resources redirected to 076 (wide-student) / 077 (per-agent student) / 078 (DAGGER distill) / 055v2_extend rerun

### §8 后续路径

- **Outcome A (breakthrough ≥ 0.920)**: progressive distill is the path. Iterate: stage 3 using 066 winner, combine with temperature (066C = BAN + T=2.0).
- **Outcome B (066A/B persistent tied 0.907–0.919)**: moderate gain, stack 066 winner with 055-temp or LR sweep.
- **Outcome C (regression < 0.895)**: self-distillation not helpful for this arch. Close progressive path.

### §9 相关

- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — stage 1 base (055@1150 is 066's teacher)
- [snapshot-061](snapshot-061-055v2-recursive-distill.md) — concurrent 5-teacher variant (055v2 = 055 + 034e + 056D, parallel direction)
- [snapshot-063](snapshot-063-055-temp-sharpening.md) — concurrent temperature variant (Tier 3a)
- [snapshot-064](snapshot-064-056-pbt-full.md) — PAUSED alternative
- [BACKLOG.md](BACKLOG.md) — 055v3 / progressive was listed
- **Theoretical**: Furlanello et al. 2018 "Born Again Neural Networks" (ICML); Hinton 2015 "Distilling the Knowledge in a Neural Network" (§5 self-distillation setup); Rusu et al. 2016 "Policy Distillation" (iterative PD on Atari).
- **Code targets**:
  - `scripts/eval/_launch_066A_pure_selfdistill_from_055.sh`
  - `scripts/eval/_launch_066B_weighted_4teacher_from_055.sh`
  - `cs8803drl/branches/team_siamese_distill.py` (add `TEAM_DISTILL_TEACHER_WEIGHTS` env var support)
