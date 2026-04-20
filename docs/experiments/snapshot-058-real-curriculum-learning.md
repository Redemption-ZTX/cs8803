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

_Pending_

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
