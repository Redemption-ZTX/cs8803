## SNAPSHOT-058: 真启动 Curriculum Learning (Tier A4)

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); pending free node + S1/S2 完成

## 0. 背景

`cs8803drl/training/train_ray_curriculum.py` + `curriculum.yaml` 已经存在但**没正经实验**。这是 project 已经写过的 infrastructure, 但 curriculum 在主线 lane (031A/B、029B、043 等) 上**从未跑过**。

curriculum 的核心思想:
- 早期训弱对手 (random / weak baseline)
- 中期训中等对手
- 后期训强对手 (baseline / 自身 frontier pool)

理论上能解决 "强对手稀疏 reward + 早期 policy 太弱" 的问题。

## 1. 假设

### H_058

> 用 curriculum (random → weak baseline → real baseline → frontier) 训 031B-arch from scratch, **1000ep peak ≥ 0.886** (+0.4pp vs 031B 0.882)。

### 子假设

- **H_058-a**: 早期 random opponent 让 student 学会基本 control 而不被 baseline 压得 entropy collapse
- **H_058-b**: 中期切到 weak baseline 让 student 学打 baseline 早期 mistake pattern
- **H_058-c**: 后期 real baseline 让 student 学会 baseline 的 stable behavior
- **H_058-d**: 跟 raw scratch + baseline_prob=1 比, curriculum 的 sample efficiency 更高

## 2. 设计

### 2.1 Curriculum schedule (4 phase)

| Phase | iter range | opponent | baseline_prob | 持续 |
|---|---|---|---|---|
| 1 (warmup) | 0-200 | random | 0.0 (100% random) | 200 iter |
| 2 (transition) | 200-500 | mixed: 0.3 baseline + 0.7 random | 0.3 | 300 iter |
| 3 (consolidate) | 500-1000 | mixed: 0.7 baseline + 0.3 random | 0.7 | 500 iter |
| 4 (final) | 1000-1250 | 100% baseline | 1.0 | 250 iter |

(总 1250 iter, 跟 031B 一字 budget)

### 2.2 实现机制

需要的工程:
- 在 train script 添加 `CURRICULUM_ENABLED=1` env var
- 在每 N iter 通过 callback 改 `env_config.opponent_mix.baseline_prob`
- 修改 `RewardShapingWrapper` 或 worker setup 让 baseline_prob 可动态更新

**或者**: 使用现有 `train_ray_curriculum.py` 入口 + 写新 curriculum.yaml

让我先看现有 curriculum infrastructure 再决定。

### 2.3 训练超参

```bash
# 同 031B
TEAM_SIAMESE_ENCODER=1 TEAM_CROSS_ATTENTION=1 ...
USE_REWARD_SHAPING=1 ...
LR=1e-4 CLIP_PARAM=0.15 ...
MAX_ITERATIONS=1250 ...

# Curriculum
CURRICULUM_ENABLED=1
CURRICULUM_PHASES="0,200,random;200,500,baseline_0.3;500,1000,baseline_0.7;1000,1250,baseline_1.0"
```

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.886 | +0.4pp | curriculum 有真增益 |
| §3.2 主: peak ≥ 0.890 | +0.8pp | sample efficiency 显著 |
| §3.3 突破: peak ≥ 0.900 | grading | declare success |
| §3.4 持平: peak ∈ [0.875, 0.886) | sub-marginal | curriculum 不影响 final WR |
| §3.5 退化: peak < 0.870 | curriculum 反伤 | warmup 让 student 学到错误 prior |

## 4. 简化点 + 风险 + 降级 + 预案

### 4.1 简化 A4.A — 4 phase fixed schedule

| 简化项 | 完整 | 当前 |
|---|---|---|
| Curriculum schedule | adaptive (eval WR-based phase transition) | fixed iter-based |

**风险**:
- 200/500/1000 iter 切点是 ad-hoc, 没数据 backing
- 如果 student 在 iter 200 还没学会基本 control, 切到 baseline 太早会崩
**降级**: -0.5~-1pp。
**预案**:
- L1: 加 condition: phase transition 需要 50ep WR ≥ 0.6 才进下一 phase
- L2: 完全 adaptive curriculum (RL teacher, 复杂)

### 4.2 简化 A4.B — Phase boundaries 没 sweep

**风险**: 200/500/1000 是猜的; 真 best 可能 100/300/800。
**预案**: 若主 lane 持平, sweep [phase1_end, phase3_end] grid。

### 4.3 简化 A4.C — Opponent pool 简单 (random + baseline only)

| 简化项 | 完整 | 当前 |
|---|---|---|
| Opponent diversity | self-play frontier pool | random + baseline |

**风险**: random 与 baseline 之间可能有 difficulty gap, 中间没 weak self-play 填补。
**预案**: 若持平, 加 phase 0.5 用 frontier early ckpt (e.g., 028A@200) 作为 weak self-play opponent。

### 4.4 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | 4-phase fixed | base 12h |
| 1 | peak < 0.880 | adaptive WR-gated phase transition (L1) | +12h |
| 2 | step 1 失败 | sweep phase boundaries (L2) | +12h × 3 |
| 3 | step 2 失败 | 加 frontier-pool opponent (L3) | +12h |
| 4 | step 3 失败 | curriculum 路径关闭 | — |

## 5. 不做的事

- 不在 implementation 完成 + smoke pass 之前 launch
- 不混入架构 / reward 改动
- 不和 RND (snapshot-057) 同时开
- 不立刻投资 adaptive teacher (太复杂)

## 6. 执行清单

- [ ] 1. 阅读 `train_ray_curriculum.py` + `curriculum.yaml` (~30 min)
- [ ] 2. 决定: 用现有 curriculum 入口 vs 给 team-level train script 加 curriculum hook (~1h)
- [ ] 3. 实现 curriculum hook (~3h)
- [ ] 4. Smoke test
- [ ] 5. 写 launch script
- [ ] 6. 找 free node, launch
- [ ] 7. Verdict

## 7. Verdict

### 7.1 2026-04-20 13:55 EDT — Stage 1 baseline 1000ep verdict — **FAIL §3.1 marginal threshold, Outcome C (curriculum path closes)**

**Append-only**。

训练数据：058 Curriculum Stage 1 baseline 1000ep，11 ckpts (pick_top +11 因 §3.3 training 是 short-range variable window)。

**Stage 1 1000ep 结果表**:

| ckpt | 1000ep | W-L |
|---:|---:|---:|
| 850 | 0.813 | 813-187 |
| 860 | 0.812 | 812-188 |
| 930 | 0.824 | 824-176 |
| 940 | 0.837 | 837-163 |
| **950** | **0.847** | 847-153 |
| 1150 | 0.845 | 845-155 |
| 1160 | 0.818 | 818-182 |
| 1170 | 0.827 | 827-173 |
| 1220 | 0.845 | 845-155 |
| 1230 | 0.825 | 825-175 |
| 1240 | 0.833 | 833-167 |

**Peak = 0.847 @ iter 950** (847W-153L, n=1000, SE ±0.011). Plateau mean 约 0.83, ±SE 0.010。

**Raw recap** (`evaluate_official_suite.py` parallel run，total_elapsed=577.3s, 7-way parallel):

```
=== Official Suite Recap (parallel) ===
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-850 vs baseline: 0.813 (813W-187L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-860 vs baseline: 0.812 (812W-188L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-930 vs baseline: 0.824 (824W-176L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-940 vs baseline: 0.837 (837W-163L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-950 vs baseline: 0.847 (847W-153L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1150 vs baseline: 0.845 (845W-155L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1160 vs baseline: 0.818 (818W-182L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1170 vs baseline: 0.827 (827W-173L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1220 vs baseline: 0.845 (845W-155L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1230 vs baseline: 0.825 (825W-175L-0T)
058_curriculum_scratch_v2_512x512_20260420_092046/.../checkpoint-1240 vs baseline: 0.833 (833W-167L-0T)
[suite-parallel] total_elapsed=577.3s tasks=11 parallel=7
```

**对预注册判据评估**:

| §3.N 判据 | 阈值 | 实测 | verdict |
|---|---|---|---|
| §3.1 marginal | peak ≥ 0.886 | 0.847 | **FAIL** |
| §3.2 主 | peak ≥ 0.890 | 0.847 | FAIL |
| §3.3 突破 | peak ≥ 0.900 | 0.847 | FAIL |
| §3.4 持平 | peak ∈ [0.875, 0.886) | 0.847 | miss (below) |
| §3.5 退化 | peak < 0.870 | 0.847 | **命中退化区** |

**结论**:
- Peak 1000ep = 0.847 @ iter 950 — 明确低于 031B SOTA 0.882, **Δ = -0.035 (-3.5pp), 约 2.2× SE 下方**
- 命中 §3.5 "退化" 区 (peak < 0.870), 也同时 fail §3.1 marginal threshold (≥ 0.886)
- 强化 "curriculum reverse-harm" 读法: warmup 阶段 student 对着 random opponent 学到的 prior 跟 baseline opponent 的 real policy 不对齐, 后期 transfer 效率 < 纯 scratch vs baseline
- **50ep 内部 eval 0.92 是 overly optimistic** — curriculum training 产生的 50ep WR 信号 (对着 mixed-difficulty opponent 测) 不 translate 到 1000ep baseline skill
- 距离 055 distill (combined 0.907) = **-6.0pp**, 距离 031B 单模 baseline = **-3.5pp**
- **Curriculum learning path (random → weak → mixed → baseline) 对 2v2 soccer 这个 seed = SUB-SOTA**, 在任何公平比较下都没有优势

**§4.5 retrograde sequence 初判** (2026-04-20 13:55): §3.5 退化阈值命中,但 **user 2026-04-20 14:xx review revision**: 不关闭 curriculum 路径。理由:
- 50ep (0.92) 与 1000ep (0.847) gap = **0.073pp** — 异常大,平均 lane 只有 0.02-0.03。说明 058 训练内部方差大,可能真实 peak 未被 1000ep draw 抓到
- 未做 H2H vs frontier → 可能有 niche strength (尤其对强对手,非 baseline)
- 未 re-eval 另一 port → 可能受 Unity port-determinism 影响
- Curriculum 路径未与 LR=3e-4 / distill 结合测试

**修订决定**: 058 进入 **Stage 2 follow-up 探索**:
- 058b H2H 500ep vs {031B@1220, 055@1150, 029B@190} — 测 niche strength
- 058d re-eval 1000ep @ 不同 port — 排除 seed unlucky
- 根据 58b/58d 结果决定:
  - 若 H2H 有意外胜场 (vs frontier → 058 > 0.50): 走 **Outcome A** 路径 (adaptive curriculum / frontier pool opponent)
  - 若 re-eval 回 0.87+: 说明之前 1000ep 是 unlucky draw, 走 **058a extend with LR=3e-4**
  - 两者都无信号: 确认 Outcome C, 但保留 058 作为 ensemble candidate 不 prune

## 8. 后续路径

### Outcome A — 突破
- adaptive curriculum (L1 → L2 → L3)
- combine with PBT (056) / RND (057)

### Outcome B — 持平
- sweep phase boundaries
- 加 self-play opponent

### Outcome C — 退化
- curriculum 路径关闭
- 退回 baseline_prob=1 的 standard training

## 9. 相关

### 理论支撑
- **Bengio et al. 2009** "Curriculum Learning" — 原始 paper
- **Soviany et al. 2022** "Curriculum Learning: A Survey" — RL curriculum survey
- **POET (Wang 2019)**, **PAIRED (Dennis 2020)** — adaptive teacher curriculum

### 代码
- [train_ray_curriculum.py](../../cs8803drl/training/train_ray_curriculum.py) — 现有 curriculum 入口
- [curriculum.yaml](../../curriculum.yaml) — 现有 yaml 配置

---

## 7.2 Re-eval 1000ep (2026-04-20 15:20 EDT, port 48005, append-only)

user 2026-04-20 14:xx review 不关闭 curriculum 路径, 对 Stage 1 的 3 个候选 ckpt (940 / 950 / 1150) 在**不同 port (48005, 跟 Stage 1 port 49005 不同 seed)** 重跑 1000ep 并 combine 为 2000ep:

| ckpt | Stage 1 1000ep (port 49005) | Re-eval 1000ep (port 48005) | Combined 2000ep | ±SE |
|---:|---:|---:|---:|---:|
| 940 | 0.837 | 0.829 | 1666/2000 = 0.833 | ±0.008 |
| 950 | 0.847 (prior peak) | 0.826 | 1673/2000 = 0.837 | ±0.008 |
| 1150 | 0.845 | 0.839 | **1684/2000 = 0.842** | ±0.008 |

Raw recap:
```
=== Official Suite Recap (parallel) ===
/storage/ice1/5/1/wsun377/ray_results_scratch/058_curriculum_scratch_v2_512x512_20260420_092046/TeamVsBaselineShapingPPOTrainer_Soccer_cb48b_00000_0_2026-04-20_09-21-09/checkpoint_000940/checkpoint-940 vs baseline: win_rate=0.829 (829W-171L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/058_curriculum_scratch_v2_512x512_20260420_092046/TeamVsBaselineShapingPPOTrainer_Soccer_cb48b_00000_0_2026-04-20_09-21-09/checkpoint_000950/checkpoint-950 vs baseline: win_rate=0.826 (826W-174L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/058_curriculum_scratch_v2_512x512_20260420_092046/TeamVsBaselineShapingPPOTrainer_Soccer_cb48b_00000_0_2026-04-20_09-21-09/checkpoint_001150/checkpoint-1150 vs baseline: win_rate=0.839 (839W-161L-0T)
[suite-parallel] total_elapsed=285.0s tasks=3 parallel=3
```

**关键发现**: Stage 1 的 peak (0.847 @ 950) 是 **positive fluctuation** — combined 2000ep 下真 peak shift 到 **iter 1150 = 0.842** (不是 950)。

- §7.1 记录的 "50ep (0.92) 与 1000ep (0.847) gap = 0.073pp" 的 **unlucky sample 假设被部分反驳**: re-eval 下 950 掉到 0.826, 1150 稳到 0.839, combined true peak 0.842 (1150) 不是 0.837 (950)。Stage 1 把 ckpt 950 偶然抬到 0.847, 被 re-eval 拉回来。
- 但 gap 本质 (internal eval 0.92 vs official 1000ep 0.842) 仍存在 — 不是 1000ep unlucky, 而是 **058 curriculum training 真实内部方差 large**, peak ckpt identity 在不同 sample 之间会 shift (940/950/1150 在不同 port 下各有擡升/跌落)。

---

## 7.3 H2H 3-way (2026-04-20 15:15-15:20 EDT, n=500 each, append-only)

测试 §7.1 verdict revision 中"niche strength hypothesis" — 058 curriculum 对 frontier / peer 是否有 peer-axis 优势 (即使 baseline 轴 sub-SOTA)。

| matchup | 058 wins | opp wins | 058 rate | z | p | sig |
|---|---:|---:|---:|---:|---:|:---:|
| 058@950 vs 031B@1220 | 187 | 313 | **0.374** | -5.64 | <0.001 | `***` (LOSE) |
| 058@950 vs 055@1150 | 144 | 356 | **0.288** | -9.48 | <0.001 | `***` (LOSE) |
| 058@950 vs 029B@190 | 239 | 261 | **0.478** | -0.98 | 0.16 | — (TIE) |

Raw recaps:
```
---- H2H Recap (058 vs 031B) ----
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: cs8803drl.deployment.trained_team_ray_opponent_agent
episodes: 500
team0_overall_record: 187W-313L-0T
team0_overall_win_rate: 0.374
team0_edge_vs_even: -0.126
```
```
---- H2H Recap (058 vs 055) ----
team0_overall_record: 144W-356L-0T
team0_overall_win_rate: 0.288
team0_edge_vs_even: -0.212
```
```
---- H2H Recap (058 vs 029B) ----
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: cs8803drl.deployment.trained_shared_cc_opponent_agent
episodes: 500
team0_overall_record: 239W-261L-0T
team0_overall_win_rate: 0.478
team0_edge_vs_even: -0.022
```

**H2H 读法**:
- **LOSE `***` vs 031B (frontier team-level SOTA)** — 058 curriculum 对 team-level strongest single-model 给出 -12.6pp 决定性劣势。
- **LOSE `***` vs 055 (project SOTA distill)** — 对 project SOTA 给出 -21.2pp 更大劣势, 完全无 niche strength。
- **TIE vs 029B (per-agent SOTA, cross-architecture)** — 唯一未显著的 matchup; 跨架构下 058 稳住 0.478 (NS), 但仍 <0.5 方向性劣势。

---

## 7.4 Final verdict (2026-04-20 15:20 EDT, 058 simplified curriculum, append-only)

综合 §7.1 + §7.2 + §7.3, **058 simplified curriculum (4-phase fixed schedule random → 0.3 baseline → 0.7 baseline → 1.0 baseline)**:

1. **Combined 2000ep peak = 0.842 @ iter 1150** (supersedes §7.1 single-shot 0.847 @ 950 — Stage 1 是 positive fluctuation; true peak 在 re-eval 下 shift 到 1150)。
2. **vs 031B (0.880): Δ = -0.038 (≈3.5× SE)** — sub-SOTA **statistically significant** (1.96σ 外)。
3. **vs 031B-noshape (0.875): Δ = -0.033** — 即使拿掉 v2 作 fair anchor, 仍 sub-SOTA。
4. **H2H portfolio** (§7.3):
   - LOSE `***` vs 031B (frontier team-level SOTA, -12.6pp)
   - LOSE `***` vs 055 (project SOTA, -21.2pp)
   - TIE vs 029B (per-agent SOTA, cross-architecture, -2.2pp NS)
5. **Niche strength hypothesis**: **PARTIALLY validated** — 058 对 cross-arch per-agent (029B) 打成 tie, 但**不**对任何 frontier team-level 产生正号。无 actionable upside。
6. **50ep/1000ep gap 的真解释**: 058 curriculum 训练**内部方差大** (re-eval 下 peak identity 从 950 shift 到 1150 确认), 不是 Stage 1 1000ep unlucky sample。
7. **Directive**: 058 **simplified path CONFIRMED sub-SOTA**。但 **DO NOT close curriculum path overall** — 062a/b/c (adaptive gate + boundary sweep + no-shape) 正在测 upgraded 版本。Final curriculum verdict pending 062 results。058 保留作为 curriculum family baseline reference, 不进 ensemble / submission。
