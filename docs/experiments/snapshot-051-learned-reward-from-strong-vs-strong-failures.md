# SNAPSHOT-051: Learned Reward from Strong-vs-Strong Failure Captures

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 预注册 / 待数据采集 + 模型训练

## 0. 为什么做

### 0.0 设计理念 / Design Rationale (051C/051D 回填 — 2026-04-19)

> 本节回填 051C/051D 的设计动机. 原文档落地时只记录了 lane 配置, 漏写了从 051A/B → 051C → 051D 的完整推理链. 内容来自 2026-04-19 设计对话 (sessionId `2f87bb87…`). 用户原话: "之前 051 的设计理念也是从此出发的, 但是文档忘记更新了".

#### 设计动机 (motivation)

051A/B 跑完后暴露一个本质问题: 两者都从 **031B@1220** (cross-attn SOTA, baseline 1000ep 0.882) warmstart, v2 reward 已在 1220 iter 训练里被 policy **完全内化**. 此时 learned reward 加进来, 无论 combo (051A 0.888) 还是 learned-only (051B 0.872 退化), 都无法分离两个混淆变量:
1. learned reward 是否真有信号? 还是 v2 prior 仍在主导?
2. learned reward model 本身的训练数据 (5 对 frontier H2H failures) 也全部来自 v2-trained students → reward model 的 ID 已被 v2 prior 染过.

用户原话: *"对 051C 其实没有那么决定性, 因为 learned reward 本身还是来自于大部分 v2 的训练出来的模型, 可能已经完全内化了."*

要 falsify "learned reward 是有效独立信号源" 这个假设, 必须让 policy 的 warmstart **尽量少受 v2 prior 污染**, 但又不能纯 scratch (前 300-500 iter 完全 OOD, learned reward heads 给不出梯度, 本质就退化成"scratch + 纯 sparse" — 本项目从未跑通过).

#### 选 weak base (031B@80) 的理由

中间地带候选:
- **(B-1) Scratch on 031B 架构** — 严格但不可行 (OOD)
- **(B-2) BC pretrain + learned-only** — rigorous 但需 1-2h 工程
- **(B-3) 复用 031B 早期 ckpt @80** — 0 工程, 务实选择

031B@80 关键属性:
- cross-attn 架构 ✓ (跟 051A/B 同家族, 控制架构变量)
- 内部 50ep WR ≈ 0.50 (跟 BC matches baseline 同档起点)
- 仅含 80 iter 的 v2 prior (vs 051A/B 的 1220 iter, 减弱 ~15×)
- 已脱离 random init basin, 处在"对抗 baseline 但还很弱"的早期 basin

Trade-off 老实承认: 起点不是 v2-free 的 (BC 才是), 但 **80 iter 的 v2 影响远小于 1220 iter**, 足以让 learned reward 的边际信号显形. 用户接受 B-3 的"务实严格性": *"我觉得你选取的这个 031B 80 点还真可以, 这个做法不错."*

#### 选 learned-only (no v2) 的理由

跟 051B 同 reward 配置 (USE_REWARD_SHAPING=0, LEARNED_REWARD λ=0.003). 理由是要在已经"减少 v2 baked-in 起点"的基础上, **继续断掉 v2 在训练中的实时贡献** — 否则 v2 在前期 0-200 iter 仍会主导梯度, learned reward 的"独立驱动能力"判据被冲淡.

预期结果分档 (训练前预注册):
- 1000ep peak ≪ 0.50 → learned-only 完全没信号
- 1000ep peak ≈ 0.50 → 弱到连 incremental 都做不到
- 1000ep peak >> 0.50 → 在 cross-attn 上能 incremental 改进
- 1000ep peak >> 0.882 → 强证据 learned-only 是真信号源 (能在 80→saturate 内复现 v2 在 80→1220 内做的事)

#### 选 800 iter 长 budget 的理由

051C (200 iter 试跑) 50ep eval 显示: iter 120 peak 0.88 → iter 130-190 oscillate 0.74-0.84, **没有 saturate 信号也没有稳定 plateau**. 趋势提示长 budget 可能 break 0.88.

051D = 051C config + MAX_ITERATIONS=800 (同 warmstart, 同 reward, 同节点), 目的:
- 给 learned reward 充分时间从 0.46 weak base 收敛
- 直接对比 "v2 在 80→1220 iter 把 031B 训到 0.882" vs "learned reward 在 80→880 iter 能否做出同等量级提升"
- 单 8h 节点成本可承受

#### 跟 051A/051B 的对比意义

| Lane | warmstart | warmstart v2-prior 量 | Reward (训练时) | iter | 测的问题 |
|---|---|---|---|---|---|
| **051A** | 031B@1220 (0.88) | 大量 (1220 iter) | v2 + learned combo | 200 | combo 在 saturated 起点的边际值 |
| **051B** | 031B@1220 (0.88) | 大量 (1220 iter) | learned-only | 200 | learned 在 saturated 起点的边际值 |
| **051C** | 031B@80 (0.50) | 少量 (80 iter) | learned-only | 200 | learned 在 weak 起点的 incremental 能力 (短跑) |
| **051D** | 031B@80 (0.50) | 少量 (80 iter) | learned-only | 800 | learned 在 weak 起点能否长跑 break 0.88 |

051D 的设计是 051C 的 "同条件长 budget extension", **不是新变量** — 把"learned reward 单独能做多少"这个问题问到底.

> Caveat (用户在设计阶段已点出, 回填时必须保留): 031B@80 仍含 80 iter v2 prior, 加上 reward model 训练数据本身也来自 v2-trained frontier, 所以 051D 不是 v2-free 的纯净测试, 只是"v2 影响显著减弱"的近似测试. 真正 v2-free 的 BC pretrain 路径 (B-2) 没启动, 作为 051D 的潜在 follow-up 保留.

### 0.1 现有 learned reward (036/045) 的 data 来源

snapshot-036 系列 reward model (`ray_results/reward_models/036_stage2/reward_model.pt`) 训练于：

- **trajectory pool**: 029B@190 / 025b@080 / 017@2100 / 028A@1220 在 **vs baseline 的失败** episode
- **labels**: v2 buckets (defensive_pin / wasted_possession / possession_stolen / etc.)
- **意图**: 教 reward model "强 student 输给 baseline 时长什么样" → 加 dense signal 让训练时避开这些 state

### 0.2 这个数据源的硬限制

baseline self-play WR ≈ 0.50 ([snapshot-013](snapshot-013-baseline-weakness-analysis.md))。这意味着：

- **baseline 是个 0.5-WR 对手**——它自己也经常输自己
- 我们看到的 "strong-vs-baseline 失败" 主要是 "强 agent 偶尔翻车被弱对手抓"，不是真正的 difficult-state-strong-policy 表现
- learned reward 学到的 signal = "如何避免给弱对手机会"，不是 "如何在 hard state 转化成 win"

**实证 evidence**:
- 036D 在 per-agent 拿了 +1.4pp on max ([snapshot-036D §10](snapshot-036d-learned-reward-stability-fix.md#10-首轮结果2026-04-18-1700-edt))
- 045A 在 team-level (Siamese) 仅拿 +0.7pp ([snapshot-045 §11.4](snapshot-045-architecture-x-learned-reward-combo.md#114-严格按-4-判据))
- 040 系列 4 个 reward variants + 042A3 KL distill 全 saturate 在 0.86-0.87 ceiling ([snapshot-045 §11.6](snapshot-045-architecture-x-learned-reward-combo.md#116-跟-0404-lanes-合并起来的更大-picture))
- learned reward 边际值在 0.86+ student 上 ≤ +1pp，**不能 cross 架构 ceiling**

### 0.3 新假设: strong-vs-strong failure data 是更强 signal

如果 reward model 的训练数据从 "strong-vs-baseline failure" 切换到 **"strong-vs-strong failure"** (e.g., 031A 输给 031B 的 episode、029B 输给 025b 的 episode), reward model 会学到更**深层/更可迁移**的 signal:

- **vs baseline failure**: 偶发的 "强 policy 一时疏忽"，pattern 噪声大
- **strong-vs-strong failure**: "高水平对抗中真正决定胜负的 state"，信号密度高
- 训练得到的 reward 给 student 的 dense signal = "如何赢下高水平对抗中的关键 state"
- 这种 signal **更可能 transfer 到 vs baseline** (因为 baseline 难以制造的"关键 state" vs strong opp 频繁出现)

### 0.4 跟 snapshot-050 (cross-student DAGGER) 的关系

snapshot-050 是**部署侧**用 strong teacher 接管 student 的失败 state。snapshot-051 是**训练侧**让 reward model 用 strong-vs-strong 数据训练后给 dense signal。**两者正交**:

- 050: deploy time, 不需重训 student
- 051: train time, 需要重训 student + 先采 strong-vs-strong 数据 + 重训 reward model

如果 050 验证 "cross-student 信号有用" (Phase 1.3b 100ep 已 +14pp), 051 是把这个信号 internalize 到 student 自己的 policy 里，无需 deploy time switching。

## 1. 核心假设

### H_051

> reward model 训练数据从 "strong-vs-baseline failure" 切换到 "**strong-vs-strong failure**" 后，用同样 recipe (036D 配方: weight 0.003, warmup 10000) 加到 031A/031B 训练上，**1000ep peak 比 045A (0.867) 高 ≥ +1.5pp**，超过 0.882 (031B) 接近 0.90。

预期机制：
- strong-vs-strong failure 数据 = baseline 难以制造的「关键 state」
- reward model 学到的 dense signal 在那些 state 上 reward 更精确
- 训练时 student 收到更准确的 "我现在该做什么" guidance

### 子假设

- **H_051a**: data quality matters > recipe — 同 weight/warmup, 数据换了, 结果显著不同
- **H_051b**: data 多样性重要 — 收集 ≥ 4 个 H2H pair (e.g., 031A vs 031B, 029B vs 025b, 036D vs 029B, 028A vs 027A)，模型学到的 signal 比 single-pair 更通用
- **H_051c**: 训练 student 也是 SOTA → reward model 才能 push student 进 advanced state；如果 student 是 baseline 级，reward model 给的 advanced signal 是错的方向

## 2. 设计

### 2.1 Stage 0: 采集 strong-vs-strong failure capture

**Pairs to capture** (对应 snapshot-050 v2 桶证据，失败模式互补优先):

| pair | 期望失败结构差异 | 数据量 (target) |
|---|---|---|
| **031B vs 031A** | 同架构系列，wasted_possession 主导 | 200 episode (peer H2H 的 ~40-60 loss episodes) |
| **036D vs 029B** | 跨 reward (learned vs static) | 200 ep |
| **025b vs 029B** | per-agent 内部 | 200 ep |
| **036D vs 031A** | failure mode 互补 (defensive_pin vs wasted) | 200 ep |
| **031A vs 028A** | 架构 step (Siamese vs flat) | 200 ep |

每个 pair 跑 **400ep H2H + save losses 双向**：
- 取胜方角度：胜方做对了什么 (positive sample)
- 取败方角度：败方做错了什么 (negative sample)
- 实质上每 pair 给 ~200 W + ~200 L data

**总数据量**: 5 pairs × 400 ep ≈ 2000 episodes (vs 036 系列原数据 313 episodes), **约 6x 数据量**。

### 2.2 Stage 1: 重训 reward model

复用 [`cs8803drl/imitation/learned_reward_trainer.py`](../../cs8803drl/imitation/learned_reward_trainer.py) 但：
- 数据源换成 §2.1 收集的 strong-vs-strong dump
- v2 buckets 同 ([failure_buckets_v2.py](../../cs8803drl/imitation/failure_buckets_v2.py))
- 训练 epoch 同 (默认)
- 输出: `ray_results/reward_models/051_strong_vs_strong/reward_model.pt`

### 2.3 Stage 2: 用新 reward model 训 student

在 031B 架构上跑两条 lane:
- **051A** = 031B warmstart + v2 + new learned reward (recipe 同 045A)
- **051B** = 031B warmstart + new learned reward only (no v2, recipe 同 045B if 045B verdict is interesting)

200 iter, ~4h GPU each。

## 3. 预注册判据

| 判据 | 阈值 | 含义 |
|---|---|---|
| §3.1 主: 1000ep peak ≥ 0.890 | 突破 031B base 0.882 | strong-vs-strong reward 真有用 |
| §3.2 主: 1000ep mean ≥ 0.870 | 跟 031B 整体提升 | 不只是单点 spike |
| §3.3 失败: peak ≤ 0.882 | 跟 031B 持平 | new reward 不增益, lane 关闭 |
| §3.4 退化: peak < 0.86 | 比 031A base 还低 | strong-vs-strong data 反而干扰, 数据质量假设错 |

## 4. 风险

### R1 — 数据收集成本

5 pairs × 400 ep ≈ 2000 ep × ~5s/ep ≈ 2.8h GPU on free node. 不算大但需要 evaluator 改一下让它能 `--save-mode losses` AND `--save-mode wins` 两边都 dump。可能需要小 patch on `evaluate_matches.py` (已支持 `wins / losses` save_mode 单独, 需要 dual-save 或 跑两次)。

### R2 — Reward model overfit on hard pairs

如果 5 pairs 之间 strategy distribution 重叠少, reward model 可能过拟合 specific opponent style。**缓解**: 多 pair 提供 diversity (H_051b)。

### R3 — Student-reward distribution mismatch

031B@1220 已经在 strong play 区间，但用 029B vs 025b 失败数据训的 reward 可能给它 "回到弱者状态" 的 signal。**缓解**: 测 student = 031B (current SOTA) AND student = 029B (中等强度) 两条变种, 看效果是否依赖 student 强度。

### R4 — 045B 同时也是测试

[045B (learned only, no v2) 现在跑](snapshot-045-architecture-x-learned-reward-combo.md) 用的还是**旧 reward model** (036_stage2). 如果 045B verdict 也跟 045A 一样 saturate, 间接证明 reward model 数据质量比 recipe 关键, 强化 051 的假设。

## 5. 不做的事

- **不重新设计 v2 桶** — 复用 failure_buckets_v2.py
- **不改 reward model 架构** — 只换数据
- **不在 045 saturation 之前启动 051** — 等 045B verdict 后决定优先级
- **不在 snapshot-050 Phase 1.3b verdict 之前启动 Stage 2** — 51 跟 50 互补，先看 50 verdict 决定 51 student 选哪个

## 6. 执行清单

1. 等 045B (in-flight) 1000ep verdict — 数据点
2. 等 Phase 1.3b (in-flight) 500ep verdict — 决定 student 选 031A or 031B
3. **Stage 0**: 改 evaluator 支持 H2H + 双侧 save (~1h 工程)
4. **Stage 0**: 跑 5 pairs × 400 ep ≈ 2.8h GPU
5. **Stage 1**: 重训 reward model (~30 min CPU)
6. **Stage 2 launch 051A**: 200 iter on 031B base, ~4h GPU
7. **post-train-eval skill**: 1000ep + capture + H2H

## 7. 相关

- [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) — 旧 reward model 来源
- [SNAPSHOT-045](snapshot-045-architecture-x-learned-reward-combo.md) — 045A combo saturate 的证据 (motivation 主源)
- [SNAPSHOT-050](snapshot-050-cross-student-dagger-probe.md) — cross-student DAGGER probe (concept 互补)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A/031B base
- [failure_buckets_v2.py](../../cs8803drl/imitation/failure_buckets_v2.py) — bucket 定义复用
- [learned_reward_trainer.py](../../cs8803drl/imitation/learned_reward_trainer.py) — reward model trainer 代码复用

## 8. Stage 2 verdict — 051A/051B 1000ep (2026-04-19, append-only)

### 8.1 训练完成

两条 lane 各 200 iter，clean (无 inf 报告)。

- **051A combo**: warmstart 031B@1220, USE_REWARD_SHAPING=1, LEARNED_REWARD λ=0.003 from `reward_models/051_strong_vs_strong/reward_model.pt`
- **051B learned-only**: warmstart 031B@1220, USE_REWARD_SHAPING=0 (所有 SHAPING_*=0), LEARNED_REWARD λ=0.003

run dirs:
- `ray_results/051A_combo_on_031B_with_051reward_512x512_20260419_110852/`
- `ray_results/051B_learned_only_on_031B_with_051reward_512x512_20260419_110853/`

### 8.2 1000ep 官方 eval (top 10% per skill, 7 ckpts each, parallel-7, ~4.5 min/each)

**051A combo on 011-28-0, port 56355**

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 130 | **🏆 0.888** | 888-112 | 0.80 | **+0.088** ⚡ |
| 140 | 0.866 | 866-134 | 0.86 | +0.006 |
| 150 | 0.875 | 875-125 | 0.90 | -0.025 |
| 160 | 0.869 | 869-131 | 0.88 | -0.011 |
| 170 | 0.870 | 870-130 | 0.88 | -0.010 |
| 180 | 0.882 | 882-118 | 0.92 | -0.038 |
| 190 | 0.870 | 870-130 | 0.88 | -0.010 |

**peak = 0.888 @ 130, mean = 0.875, range [0.866, 0.888]**

**051B learned-only on 013-19-0, port 56455**

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 130 | 0.843 | 843-157 | 0.86 | -0.017 |
| 140 | 0.865 | 865-135 | 0.86 | +0.005 |
| 150 | 0.849 | 849-151 | 0.90 | -0.051 |
| 160 | 0.861 | 861-139 | 0.90 | -0.039 |
| **170** | **🏆 0.872** | 872-128 | 0.92 | -0.048 |
| 180 | 0.854 | 854-146 | 0.92 | -0.066 |
| 190 | 0.862 | 862-138 | 0.82 | +0.042 |

**peak = 0.872 @ 170, mean = 0.858, range [0.843, 0.872]**

### 8.3 严格按 [§4 判据](#4-判据)

| 阈值 | 051A combo | 051B learned-only | verdict |
|---|---|---|---|
| §4.1 主: peak ≥ 0.890 (vs 034E ensemble) | ❌ 0.888 (-0.002) | ❌ 0.872 | 051A 边缘 tied with ensemble, 051B 落后 |
| §4.2 突破 0.882 (031B base) | ✅ 0.888 (+0.006) | ❌ 0.872 (-0.010) | 051A 边缘 over base, 051B 反而退化 |
| §4.3 falsify reward model 数据源 hypothesis | — | — | 见 §8.4 |

### 8.4 关键发现 — **reward model 数据源不是 leverage 点**

跨 045 / 051 直接对比, base × reward model 4 组：

| Base × Reward | base 1000ep | combo peak | learned-only peak | combo Δ vs base | learned-only Δ vs base |
|---|---:|---:|---:|---:|---:|
| **031A × 045 reward** (baseline failures) | 0.860 | 0.867 | 0.870 | +0.007 | +0.010 |
| **031B × 051 reward** (strong-vs-strong) | 0.882 | **0.888** | 0.872 | **+0.006** | **-0.010** |

**两个对照同时落地**：
1. **Combo lane 边际增益恒为 sub-noise +0.6-0.7pp** — 不管 reward model 训自 baseline failures 还是 strong-vs-strong, marginal gain 几乎相同。**reward model 数据源不是杠杆**。
2. **Learned-only lane 在弱 base (031A) 上 +1pp, 在强 base (031B) 上 -1pp** — 相反方向!

**Strong base + learned-only 反而退化的解读**：
- 031B (cross-attention) 的 policy 已经 internalize 了 learned reward 想 inject 的 coordination signal（从 031B 比 031A baseline +2.2pp 的 architecture-axis 跃升来推断）
- 在已 internalize 的 base 上叠 learned reward → reward 试图 push policy 去到它已经在的方向 → 实际产生 drift away from optimum
- **比 045B 的 +1pp 反向, 但跟 045A combo saturation 的逻辑一致**：reward signal 对架构 ceiling 的边际值随 base 强度递减

### 8.5 051A peak @ ckpt 130 — early-peak signature

- 051A 在 warmstart 后**仅 130 iter 就达到 1000ep peak 0.888**
- 比 045A peak @ 180 还早 50 iter
- 训练后续 70 iter 没贡献 (mean 0.875, 浮动在 [0.866, 0.888])
- **暗示**: 在 SOTA architecture (031B) base 上 + combo reward, saturation 在 ~iter 130 就锁定, 后续训练 no marginal value
- 后续 Stage 2 训练 budget 可以从 200 iter 收紧到 150 iter 节省 GPU (但已 sunk cost, 不动)

### 8.6 对 ensemble 034F 的 implication — 051A 是新候选

| Ensemble candidate | 1000ep peak | base | reward path | 失败模式 fingerprint (待 capture 验证) |
|---|---:|---|---|---|
| 031B@1220 | 0.882 | scratch | v2 | progress_deficit + unclear_loss |
| 036D@150 | 0.860 | warmstart | learned (per-agent) | defensive_pin + territorial_dominance |
| 029B@190 | 0.846 | warmstart | v2 | similar to 036D |
| **新: 045A@180** | 0.867 | 031A | combo (v2 + 045 learned) | **wasted_possession 主导 (55.4%, +13pp orthogonal)** |
| **新: 051A@130** | **0.888** | 031B | combo (v2 + 051 learned) | **TBD - capture 进行中 on 010-20-0** |

**预测 (基于 045A 的 wasted_possession signature pattern)**：
- combo lane 在 031A → wasted_possession 主导
- combo lane 在 031B → 大概率继承 wasted_possession 倾向（因为 v2 push possession + learned reward 的不修 conversion）
- **如果 051A capture 验证 wasted_possession 主导**, **034F = {031B, 036D, 051A@130}** 期望 ≈ 034E 但 single-model SOTA 提升到 0.888

### 8.7 后续行动

1. ✅ **051A@130 capture 已 launch** (010-20-0, tmux fail_051A_130, port 64605)
2. ⏳ **045A H2H** vs 031B / 029B / 036D — 验证 045A combo 是否真有 peer-axis 优势 (snapshot-045 的遗漏)
3. ⏳ 051A H2H vs 031B (同 base 不同 reward 直连)
4. **051C / 051D / 051C-annealed**：基于 §8.4 verdict (reward source 不是 leverage), **新建 lane 的 ROI 极低, 不投**

### 8.8 Raw recap

```
=== 051A combo (5014236 / 011-28-0 port 56355) ===
ckpt-130 vs baseline: win_rate=0.888 (888W-112L-0T)
ckpt-180 vs baseline: win_rate=0.882 (882W-118L-0T)
ckpt-150 vs baseline: win_rate=0.875 (875W-125L-0T)
ckpt-170 vs baseline: win_rate=0.870 (870W-130L-0T)
ckpt-190 vs baseline: win_rate=0.870 (870W-130L-0T)
ckpt-160 vs baseline: win_rate=0.869 (869W-131L-0T)
ckpt-140 vs baseline: win_rate=0.866 (866W-134L-0T)
total_elapsed=270.8s tasks=7 parallel=7

=== 051B learned-only (5014119 / 013-19-0 port 56455) ===
ckpt-170 vs baseline: win_rate=0.872 (872W-128L-0T)
ckpt-140 vs baseline: win_rate=0.865 (865W-135L-0T)
ckpt-190 vs baseline: win_rate=0.862 (862W-138L-0T)
ckpt-160 vs baseline: win_rate=0.861 (861W-139L-0T)
ckpt-180 vs baseline: win_rate=0.854 (854W-146L-0T)
ckpt-150 vs baseline: win_rate=0.849 (849W-151L-0T)
ckpt-130 vs baseline: win_rate=0.843 (843W-157L-0T)
total_elapsed=270.5s tasks=7 parallel=7
```

完整 log: [051A](../../docs/experiments/artifacts/official-evals/051A_baseline1000.log) / [051B](../../docs/experiments/artifacts/official-evals/051B_baseline1000.log)

## 9. 051D Stage 1 verdict + 051 family 横向对比 (2026-04-19)

### 9.1 051D 训练完成 + Stage 1

- **051D** = warmstart 031B@**80** (weak early ckpt, NOT @1220 strong) + learned reward only (no v2) + cross_attention + **800 iter** (4× 051A budget)
- Run: `ray_results/051D_learned_only_warm031B80_cross_attention_512x512_20260419_164416/`
- Full 800/800 iter on `atl1-1-03-015-2-0` (5015754), 16:44-23:06 EDT

13 ckpts (top 5%+ties+±1) × 1000ep parallel-7, 528s:

| ckpt | 1000ep WR | NW-ML | window |
|---:|---:|---|---|
| 290 | 0.859 | 859-141 | early |
| 300 | 0.834 | 834-166 | early |
| 310 | 0.849 | 849-151 | early |
| 490 | 0.856 | 856-144 | mid |
| 500 | 0.878 | 878-122 | mid |
| 510 | 0.876 | 876-124 | mid |
| 720 | 0.897 | 897-103 | late |
| 730 | 0.877 | 877-123 | late |
| **740** | **0.900** 🥇 | 900-100 | late |
| 760 | 0.874 | 874-126 | late |
| 770 | 0.882 | 882-118 | late |
| 780 | 0.892 | 892-108 | late |
| 790 | 0.892 | 892-108 | late |

**Peak 0.900 @ 740** (single shot), **mean 0.874**, **late-window (720-790, n=7) mean 0.888**.

### 9.2 051D rerun verify (top 2 ckpts)

| ckpt | orig 1000ep | rerun 1000ep | **combined 2000ep** | SE | 95% CI |
|---:|---:|---:|---:|---:|---:|
| **740** | 0.900 | 0.878 | **0.889** | 0.007 | [0.875, 0.903] |
| 720 | 0.897 | 0.873 | **0.885** | 0.007 | [0.871, 0.899] |

**Verdict**: 真值 ~**0.889**, single-shot 高估 ~1pp (符合项目 calibration norm)。**没 decisively 越过 0.900** (CI 上界 0.903)。

### 9.3 051 Family 横向对比 (Combined where available)

| Lane | Warmstart | Reward Composition | Budget | Peak 1000ep | Combined 2000ep | vs 031B base 0.880 |
|---|---|---|---:|---:|---:|---:|
| 051A | 031B@**1220** (strong) | v2 + learned (combo λ=0.003) | 200 iter | 0.888 (single) | not reverified | +0.8pp (within SE) |
| 051B | 031B@**1220** (strong) | learned only (no v2) | 200 iter | 0.872 (single) | not reverified | **-1.0pp** 退化 |
| 051C | 031B@**80** (weak) | learned only (no v2) | 200 iter | not 1000ep evaled | — | TBD |
| **051D** | 031B@**80** (weak) | learned only (no v2) | **800 iter** | 0.900 (single) | **0.889** (n=2000) | **+0.9pp** (within SE) |

### 9.4 跨 lane 关键洞察

#### 9.4.1 Reward composition 维度
- **v2 + learned (combo)** > **learned only**: 051A combo > 051B learned-only (-1.6pp on weak verify)
- learned reward **不能完全替代 v2** — 它在 combo 中是 marginal +; 单独不足以 carry 训练
- 这跟 053A combo 0.891 vs 053A 同 PBRS-only (假设无 v2, 没测过) 的差别预期类似

#### 9.4.2 Budget × warmstart 维度
- **051A (strong base, 200 iter combo) ≈ 051D (weak base, 800 iter learned-only)**: 两者 1000ep 真值都在 ~0.888
  - 解读: 增加 4× budget 用 learned-only 仅能"追平"强 base 200 iter combo
  - learned reward 提供的 signal 比 v2 dense shaping **slow 4×**
  - 不是「不工作」而是「同等增益需要更多 iter」

#### 9.4.3 vs 053A family (同 base 不同 reward path)
- **051A combo (200 iter) = 0.888** vs **053A combo (200 iter) = 0.891** → similar (within noise)
- **053Acont continue (500 iter) = 0.898** > **051D (800 iter) = 0.889** → **PBRS combo 续训 > learned-only 续训**
- 解读: PBRS 信号比 multi-head bucket reward 更 dense, 同等 budget 更高 peak

#### 9.4.4 Peak 0.900 现象
- 051D@740 single 0.900 + 053Acont@430 single 0.903 + 053Acont@310 single 0.901 + 043A'@080 single 0.901 — 多个 lane 都触及 0.900
- 但都是 single-shot, rerun 都回到 [0.889, 0.900] 区间
- **0.900 似乎是一个 noise ceiling**: 1000ep × current method 栈很难真正 decisively 越过

### 9.5 Combined leaderboard (post 051D rerun)

| Rank | Lane | combined 2000ep | n | Notes |
|---|---|---:|---:|---|
| 🥇 | 043A'@080 | 0.900 | 2000 | 唯一严格 verified ≥ 0.900 |
| 🥈 | 053Acont@430 | **0.898** | 2500 | + plateau stability ⭐ |
| 🥈 | 043C'@480 | 0.8965 | 2000 | |
| 🥈 | 053Acont@310 | 0.896 | 2000 | second plateau peak |
| 🥉 | 043B'@440 | 0.893 | 2000 | rerun -1.1pp |
| 4 | 034E ensemble | 0.892 | 2000 | |
| 5 | 053A@190 | 0.891 | 2500 | |
| 6 | **051D@740** | **0.889** | 2000 | learned-only weak-base 800 iter |
| 6 | 031B@1220 | 0.880 | 2000 | base |

### 9.6 051 family 整体 verdict

**051 path 没突破 saturation ceiling** (~0.89):
- 051A combo (0.888) 略 over 031B base 但 sub-noise
- 051B 学只 (-1pp) 退化, learned 反向 hurt
- 051D 用 4× budget 把 weak base 拉到 0.889 — 仅追上
- **PBRS path (053) 严格优于 learned-bucket path (051)** 在同等 base + budget 上

**结论**: learned-bucket reward (snapshot-021/036/045/051 series) 已 saturate; 后续投资应转向 PBRS reward family (snapshot-053) 或全新 paradigm (distillation 055 / curriculum 058 / RND 057).
