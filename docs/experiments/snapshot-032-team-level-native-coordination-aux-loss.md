# SNAPSHOT-032: Team-Level Native Coordination Aux Loss

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 已完成首轮 verdict（032-A / 032-A-control）

## 0. 方法论位置

team-level native 系列的第二条——**训练目标层面**。

| snapshot | 维度 | 改动位置 |
|---|---|---|
| [031](snapshot-031-team-level-native-dual-encoder-attention.md) | 网络架构 | forward pass 结构 |
| **032（本）** | 训练目标 | PPO loss += λ × coordination aux loss |
| [033](snapshot-033-team-level-native-coordination-reward.md) | 环境/reward | shaping 信号 |

和 031 正交：032 不改网络结构，只在 PPO 主 loss 之外加一个辅助监督信号。

## 0.5 首轮可执行口径

首轮 runnable 版本先按最小可归因路径落地：

1. **base 从 `028A@1220` 改为 `028A@1060`**
   - `1060 = 0.810 official / 0.806 capture`，比 `1220` 更稳，更适合作为 fine-tune 起点
2. **首轮先不用 weight ramping**
   - `032-A` 直接固定 `aux_loss_weight=0.05`
   - `032-A-control` 走完全同配置，但 `aux_loss_weight=0.0`
   - 先回答“显式 coordination aux loss 本身是否有增益”，再决定要不要加 scheduler / gap monitor

## 0.6 首轮结果摘要

首轮两条线均已完整跑到 `300 iter`，且内部 eval 不需要像 `027A` 那样做断点回填：`checkpoint_eval.csv` 中 `baseline/random` 都稳定覆盖到 `290`。

### 0.6.1 训练与 internal eval

| lane | run dir | internal best baseline | 备注 |
|---|---|---|---|
| `032-A` | `032A_team_action_aux_on_028A1060_512x512_20260418_053238` | `0.920 @ checkpoint-260` | 次强点 `checkpoint-170 = 0.900` |
| `032-A-control` | `032Acontrol_team_action_aux0_on_028A1060_512x512_20260418_053246` | `0.860 @ checkpoint-190` | `170/230/290 = 0.840` |

### 0.6.2 Official `baseline 500`

| lane | main candidate | official `baseline 500` | 读法 |
|---|---|---|---|
| `032-A` | `checkpoint-170` | `0.826 (413W-87L-0T)` | 比 `260` 更稳，`260 = 0.820` |
| `032-A-control` | `checkpoint-200` | `0.836 (418W-82L-0T)` | 首轮 baseline-oriented winner |

### 0.6.3 Failure capture `500ep`

| lane | candidate | capture WR | official→capture |
|---|---|---|---|
| `032-A` | `checkpoint-170` | `0.826 (413W-87L-0T)` | `0.826 -> 0.826` |
| `032-A-control` | `checkpoint-200` | `0.828 (414W-86L-0T)` | `0.836 -> 0.828` |

### 0.6.4 H2H

| matchup | result | 结论 |
|---|---|---|
| `032-A@170 vs 032-A-control@200` | `0.528` | aux 线小胜 control |
| `032-A-control@200 vs 028A@1060` | `0.558` | continuation 本身已明显优于 base |
| `032-A@170 vs 028A@1060` | `0.536` | aux 线也真实优于 base |

### 0.6.5 首轮 verdict

这轮最稳的口径不是“aux 在 baseline score 上大胜”，而是：

1. `032-A-control` 在 **baseline-oriented 指标** 上略优于 `032-A`
2. `032-A` 在 **direct H2H** 上小胜 `032-A-control`
3. 两条线都已经超过 `028A@1060`

因此首轮结论应收成：

- `032-A-control@200` 是首轮 **baseline-oriented winner**
- `032-A@170` 是首轮 **H2H-oriented winner / 更值得保留的主候选**
- `032` 整体属于 **neutral-to-positive**：aux loss 没有把 `vs baseline` ceiling 明显抬高，但在策略质量/对抗质量上给出了真实正信号

一个尚未补齐的机制缺口是：当前 `progress.csv` 里没有成功落出可读的 `aux_*` 指标列，因此“aux head 学到了什么”还不能像 `WR/H2H` 那样直接定量解释。

## 1. 核心假设

team-level policy 在 forward pass 中看到两个 agent 的 obs。但只要 PPO loss 只优化 win/reward，policy 没有显式动机去"**理解队友是什么样的 agent、会做什么**"——只要联合动作碰巧让 team 赢就够了。

假设：**加一个显式的 coordination aux loss，强制 policy 在内部表征层面建模队友**，可以：

1. 让 forward pass 学到更结构化的联合表征
2. 加速探索（aux signal 比 PPO 的 sparse reward 密集得多）
3. 间接提升 H2H 表现（对等对抗下"理解队友"比"碰巧协调"更稳）

## 2. 三个 aux loss 候选

### 2.1 候选 A — Teammate Action Prediction（最推荐）

aux head 从 policy 的中间层特征预测**队友当前这一步的 action**。

```
shared_features → policy_head → own_action_logits
shared_features → aux_head    → teammate_action_logits

aux_loss = cross_entropy(teammate_action_logits, actual_teammate_action)

total_loss = ppo_loss + λ * aux_loss
```

**为什么有用**：forward pass 必须在内部表征中编码"给定当前局面，队友会做什么"，这是无显式通信下协调的前置条件。

**实现复杂度**：低——加一个 aux head + 一个 loss term。

### 2.2 候选 B — Teammate Future-Reward Prediction

```
shared_features → aux_head → predicted_teammate_future_reward (regression)

aux_loss = MSE(predicted, teammate_reward[t+1..t+k] 的加权累积)
```

比 action prediction 更高阶：要求 policy 内部建模"我现在的动作如何影响队友未来的表现"。

**风险**：team-level 用的是 `TeamVsPolicyWrapper.step()` 把两人 reward 求和。严格意义上没有独立的"队友 reward"。需要环境 info 里保留 per-agent reward（这个信息 wrapper 有，但需要确保不被丢弃）。

### 2.3 候选 C — Inverse Dynamics Prediction

```
shared_features[t] + shared_features[t+1] → aux_head → predicted_joint_action[t]

aux_loss = cross_entropy(predicted, actual_joint_action[t])
```

从 `(s_t, s_{t+1})` 预测 `a_t`。这个 aux 在 per-agent RL（比如 ICM curiosity）里用过，强制表征保留动作相关信息。

**风险**：team-level 下 joint action 空间大（3^6=729），信号可能散。

## 3. 首轮策略

**只跑候选 A (Teammate Action Prediction)**。理由：

- 实现最简单（最低工程风险）
- 假设最清晰（"forward pass 是否内部建模队友"）
- 候选 B 需要额外 wrapper 改动（保留 per-agent reward），工程延期
- 候选 C 在 team-level 上信号散，不是首选

如果 候选 A 出正结果，下一轮再考虑 B/C 作为扩展。

## 4. 候选 A 设计

### 4.1 网络结构（基于 028A flat MLP）

```
obs (672) → shared trunk (512→512) ──┬── policy head → MultiDiscrete logits
                                     └── aux head   → 6 × Categorical(3) logits
                                                      (预测队友 3 个 action 子维度)
```

**Aux head 预测哪些动作**：team-level action = 6 维 MultiDiscrete，前 3 维是 agent 0，后 3 维是 agent 1。**Aux head 从 policy 头分支出来**，预测**另一个 agent 的 3 维子动作**。

但 team-level policy 本身输出完整 6 维。所以 aux loss 本质上是**"让 policy 预测自己输出的后 3 维"**，听起来是自指的。

更合理的设计：**从中间层分支出 aux head，stop-gradient 阻断 aux head 对 policy head 的直接影响**。让 aux loss 只训练 shared trunk，不污染 policy 本身的决策。

```
obs → shared trunk → feat
                    ├── policy head (no stop-grad)
                    └── aux head (stop-grad on policy output direction)
```

### 4.2 Aux loss 实现细节

**关键设计选择**：

1. **Aux head 位置**：从 shared trunk 倒数第二层分支（不是最后一层），避免直接干扰 policy head 的梯度
2. **Aux target**：**actual 队友 action**（trajectory 里实际执行的），不是当前 policy 的输出
3. **首轮 runnable 版本先固定 `λ = 0.05`**
   - 预注册原案中的 weight ramping 保留为后续增强项
   - 首轮通过 `032-A-control (λ=0)` 提供严格对照
4. **Aux accuracy 监控**：每 10 iter 计算 aux head 的 argmax accuracy，写进 progress.csv

### 4.3 超参

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 028A@1060 | 更稳的 team-level base |
| aux_loss_weight final | 0.05 | 保守起手 |
| aux head hidden | 256 → 6×3 | 小 head |
| 其他 PPO 参数 | 同 028A | 保持最小扰动 |

## 5. 预声明判据

### 5.1 主判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 500 | ≥ 0.85（突破 028A 0.844）| aux signal 让 policy 真正学到更强策略 |
| H2H vs 028A@1060 | ≥ 0.55 | 对等对抗下显式超越 base |
| H2H vs 029B@190 | ≥ 0.50 | 追上当前 SOTA |

### 5.1 首轮对照

| 项 | 首轮结果 | 是否达到 |
|---|---|---|
| official 500 | `032-A@170 = 0.826`, `control@200 = 0.836` | 否 |
| H2H vs 028A@1060 | `032-A = 0.536`, `control = 0.558` | `control` 达到，`032-A` 接近 |
| H2H vs 029B@190 | 未做 | 待补 |

### 5.2 机制判据（诊断 aux 是否 work）

| 项 | 期望 |
|---|---|
| aux head argmax accuracy | 单调上升，最终 ≥ 30%（随机是 1/27 ≈ 3.7%）|
| main WR curve | 不应该因为加 aux 而掉到 028A 以下 |
| `low_possession`（failure capture）| 期望下降（理解队友 → 更好配合持球）|

### 5.3 失败判据（Gaming / Over-fit 防护）

| 条件 | 解读 |
|---|---|
| aux accuracy 持续升，main WR 停滞或降 | aux 侵蚀 main loss——降 `λ` 或止损 |
| official 500 ≥ 0.85 但 H2H vs 028A < 0.50 | baseline-specific exploit（类似 026 B-warm 情况）|
| aux accuracy < 10% after 200 iter | aux signal 没学到东西，aux head 死掉 |
| internal-official gap > 0.10 | 经典 shaping/loss over-optimization 信号 |

### 5.4 Gaming 防护具体措施

1. **Stop-gradient 安排**（§4.1）：aux head 的梯度不污染 policy head
2. **对照 lane**：同配置 + `aux_loss_weight=0`（即 028A vanilla 续训）必须跑——比较“加了 aux”和“没加 aux”的 H2H
3. **Fixed-λ 起手**：首轮先用小权重 `0.05`，不在第一批就引入 scheduler 变量
4. **H2H 是 ground truth**：official 500 WR 只是辅助指标，最终判定以 H2H 为准
5. **Aux-primary gap monitor**：保留为第二轮增强项，不作为首轮 batch 前置依赖

## 6. 执行矩阵

| lane | 改动 | 预算 | 优先级 |
|---|---|---|---|
| **032-A (teammate action pred)** | aux loss = action prediction, `λ=0.05` | ~6h (300 iter fine-tune) | **已完成首轮；H2H-oriented winner** |
| 032-A-control | 同配置但 `λ=0`（= 028A 续训）| ~6h | 已完成首轮；baseline-oriented winner |
| 032-B (future reward pred) | aux loss = reward regression | ~6h | 条件启动（A 成立才做）|
| 032-C (inverse dynamics) | aux loss = IDM | ~6h | 条件启动 |

## 7. 工程依赖

### 7.1 已落实的首轮实现

- 自定义 `TeamActionAuxTorchModel`：在 team-level flat `TorchFC` 上加 action-prediction aux head，并通过 `custom_loss()` 叠加 aux term
- `train_ray_team_vs_baseline_shaping.py` 已支持：
  - `AUX_TEAM_ACTION_HEAD`
  - `AUX_TEAM_ACTION_WEIGHT`
  - `AUX_TEAM_ACTION_HIDDEN`
- `trained_team_ray_agent.py` 已注册 custom model，因此后续 eval / official / H2H 可以直接加载 `032` checkpoint

### 7.2 可以复用

- `cs8803drl/branches/teammate_aux_head.py` 已经有 per-agent 的 aux head 框架（snapshot-021c-B 用过）
- 028A batch 作为模板

### 7.3 需要确认

- `TeamVsPolicyWrapper` 传回的 trajectory 里是否保留完整 joint action（6 维）
- 对 agent 0 的 aux target 应该是 agent 1 的 action 子维度，反之亦然——具体 slot 分配需要在 policy forward 里拆分

## 8. 和其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [021c-B](snapshot-021-actor-teammate-obs-expansion.md) | per-agent 版 aux head（teammate state prediction）；032 是 team-level 版 action prediction |
| [028A](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | base |
| [031](snapshot-031-team-level-native-dual-encoder-attention.md) | 正交 native 方向；如果 031 成立，未来可以叠加 |
| [033](snapshot-033-team-level-native-coordination-reward.md) | 正交 native 方向 |

## 9. 不做的事

- 不做 policy gradient 通过 aux head 的改动（aux 是监督信号，不修改 policy 目标）
- 不做 adversarial / discriminator-style aux loss（复杂度太高）
- 不做同时启动 B/C（避免混淆归因）

## 10. 执行清单

1. 1-iter smoke：确认 aux loss 可以 backprop、`λ=0` 时数值和 028A 一致
2. 起 032-A + 032-A-control 并行 batch
3. 训练中监控 aux accuracy 和 WR
4. 300 iter 后做 official 500 + H2H vs 028A / 029B
5. 如首轮为正，再补 weight ramping / gap monitor

## 12. 首轮结果后的下一步

如果继续沿 `032` 迭代，最值的方向不是立刻扩大 batch，而是先补“为什么 aux 线 H2H 更强但 baseline score 没明显拉开”的机制证据：

1. 把 `aux accuracy / aux loss` 真正写进 `progress.csv`
2. 做 `032-A@170` 对 `029B@190` 的 H2H，确认它是否只是 team-level 内部优势，还是已经接近当前最强 continuation 线
3. 再决定是否要上第二轮 `weight ramping` / `gap monitor`

本轮更具体的后续执行已经独立拆到：

- [SNAPSHOT-032-next: symmetric action aux refinement](snapshot-032-next-symmetric-action-aux.md)

## 13. [2026-04-18] 500-ep 官评 + v2 失败桶归因

300 iter 训完后做的 honest readout。50-ep 的"eye-catching peaks"全回归均值。

### 13.1 500-ep official baseline WR

| iter | 032A (aux=0.05) | 032Ac (aux=0.00) |
|---:|:---:|:---:|
| 160 | 0.814 | — |
| 170 | **0.826** ← 032A peak | 0.794 |
| 180 | 0.806 | 0.802 |
| 190 | — | 0.820 |
| **200** | — | **0.836** ← Ctrl peak |
| 220 | — | 0.830 |
| 230 | — | 0.814 |
| 250 | 0.814 | — |
| 260 | 0.820 | — |
| 270 | 0.808 | — |
| 280 | 0.812 | — |
| 290 | — | 0.776 |
| **mean** | **0.8143** | **0.8103** |

- 50-ep peak **0.92 @iter 260** → 500-ep 重测 **0.820**（回归 10pp）
- 50-ep 漏掉了 Ctrl 的真 peak：**iter 200** 在 50-ep 只有 0.82（看不出来），500-ep 是 **0.836**
- Ctrl peak re-run 一次 500-ep 又给了 0.828（414-86） → 单次 500-ep 噪声 ~0.8pp，1σ SE ≈ 0.017

**Aux 主效应**：mean +0.004pp，**z ≈ 0.25，完全没信号**。

### 13.2 v2 失败桶分布（failure capture 500 ep）

用 [`failure_buckets_v2.classify_failure_v2`](../../cs8803drl/imitation/failure_buckets_v2.py) 离线重分（metric 在旧 failure JSON 里都有，无需重采）：

| v2 Bucket | 032A (aux, n_L=87) | 032Ac (no aux, n_L=86) | Δ (pp) |
|---|---:|---:|---:|
| defensive_pin | 47.1% | 41.9% | **+5.3** |
| wasted_possession | 46.0% | 41.9% | **+4.1** |
| territorial_dominance | 47.1% | 44.2% | +2.9 |
| progress_deficit | 23.0% | 20.9% | +2.1 |
| possession_stolen | 32.2% | 36.0% | **−3.9** |
| unclear_loss | 11.5% | 12.8% | −1.3 |

**统计性**：Δ 的二项 SE ≈ 7.5pp，**没有 Δ 过 1σ**。但 pattern 内部一致（见 13.4）。

### 13.3 Episode length（来自 failure capture summary）

| Run | W mean steps | L mean steps | W / L 比 |
|---|---:|---:|---:|
| 032A (aux=0.05) | 46.4 | **40.5** | 1.15 |
| 032Ac (aux=0.00) | 45.9 | **34.5** | 1.33 |

- W episode 长度几乎一样 (46.4 vs 45.9)
- L episode **aux 变体多撑 +6 步 (+17.4%)**，中位数 30 → 26

### 13.4 解释：aux 学到了什么 —— "海龟 (turtle) 失败模式"

把 13.2 和 13.3 拼起来是一个高度内部自洽的图像：

1. `possession_stolen ↓ -3.9pp` —— aux 让队伍更少被抢球 → **球权纪律改善**
2. `wasted_possession ↑ +4.1pp` —— 但持球没转化为进球 → **进攻效率没变**
3. `defensive_pin ↑ +5.3pp` —— 更多时间被压在自己 1/5 场 → **防守吃压更深**
4. L episode +17% 长 —— **崩盘延迟但终究崩**

**aux 学到的是短时 scale 协作（看队友位置 → 不丢球），没学到长时 scale 战略（主动创造破门）**。这符合 §1 理论假设的**下界**：aux 只强迫建模 "teammate action at t"，没强迫建模 "teammate behavior over episode"，所以只改善瞬时协作，不改善回合战略。

### 13.5 对报告的表述（null on WR, directional positive on robustness）

> Team-action auxiliary loss (λ=0.05) did not significantly improve win rate vs the λ=0.0 control at 500-ep evaluation (0.826 vs 0.828 ± 0.017, z ≈ 0.25). However, paired analysis of 86-87 loss episodes per variant suggests a consistent **possession-holding but low-conversion** pattern: the aux variant was dispossessed 3.9pp less often, yet got pinned in its defensive third 5.3pp more often, and wasted possession 4.1pp more often. Loss episodes were 17% longer on average (40.5 vs 34.5 steps). Interpretation: aux loss improved short-timescale positional coordination at the cost of offensive transitions, though effect sizes remain within 1σ at this sample size.

### 13.6 对 SOTA 链的影响

- **029B@190 (0.868) 仍是 SOTA**；032A peak 0.826 比 029B 低 4.2pp，032Ac peak 0.836 低 3.2pp
- **036C 的 warmstart 不换**（仍用 029B@190）—— 没有证据支持切过去

## 14. [2026-04-18] H2H 三角矩阵 + 非传递性

### 14.1 500-ep H2H 矩阵

| ↓ vs → | 028A@1060 | 032A@170 | 032Ac@200 | baseline (official) |
|:---:|:---:|:---:|:---:|:---:|
| **032A @170** | **0.536** (268-232) | — | **0.528** (264-236) | 0.826 |
| **032Ac @200** | **0.558** (279-221) | 0.472 (236-264) | — | 0.828 |

### 14.2 非传递性

```
032Ac  beats  028A  by +11.6pp     ← Ctrl 对 warmstart 进步最大
032A   beats  028A  by +7.2pp      ← aux 对 warmstart 进步较小
032A   beats  032Ac by +5.6pp      ← 但 aux 对 Ctrl 直接赢 ←←← 传递律应相反
```

按传递律应 Ctrl > 028A > Ctrl（因为 Ctrl 对 028A 赢得比 A 还多）→ Ctrl 总体最强。但直接 peer 是 **A 赢 Ctrl**。这是多 agent policy 没有全序的典型表现。

### 14.3 统计性

| matchup | W-L | z | p | 显著 |
|---|---|---:|---:|---|
| 032A vs 032Ac | 264-236 | 1.27 | 0.10 | 边缘 |
| 032A vs 028A | 268-232 | 3.22 | 0.0006 | 显著 |
| 032Ac vs 028A | 279-221 | 5.19 | < 1e-6 | 极显著 |

"**A 战胜 Ctrl**" 是**边缘信号**，"两者都战胜 028A" 是**硬事实**。

### 14.4 解释：两条策略轴而非一条强弱轴

- **Ctrl 学了"general improvement over warmstart"** —— 对 028A +11.6pp，对 baseline 0.828
- **A 学了"exploit Ctrl 的反制"** —— 对 Ctrl +5.6pp peer，但对 028A 只 +7.2pp（比 Ctrl 少 4.4pp）

aux loss 把 032A 推到**对 Ctrl 有克制但不一定更通用**的方向。结合 §13.4 的 turtle 失败模式：A 在**短时协调**上比 Ctrl 强（所以同为 team-level policy 对抗时略占上风），但这**没转化为对 baseline 的通用增益**。

### 14.5 Side asymmetry（环境自带，不是 policy 特性）

| matchup | 某方 blue WR | 某方 orange WR | 差 |
|---|---:|---:|---:|
| 032A vs 032Ac | 53.6 | 52.0 | +1.6pp |
| 032A vs 028A | **58.8** | 48.4 | **+10.4pp** |
| 032Ac vs 028A | 58.0 | 53.6 | +4.4pp |

所有 3 对 H2H 都显示 blue > orange。这是 `soccer_twos` 的 starter 环境本身不对称（可能开球/位置优势在 blue），**不是 032 的特性**。官评两边各 250 ep，没被污染。

### 14.6 对报告的升级表述

> Although 032A did not outperform the control on baseline WR (0.826 vs 0.828), direct head-to-head evaluation (500 ep, z=1.27) showed 032A winning 0.528 against 032Ac. Critically, both variants significantly beat the common warmstart 028A@1060 (A: +7.2pp z=3.22, Ctrl: +11.6pp z=5.19), but the control improved more over the base than the aux variant did — a violation of transitivity. This combined with the failure-bucket analysis (§13.2) suggests aux loss moved 032A onto a distinct strategy axis (short-timescale positional discipline) rather than a strictly stronger version of the control.

## 15. [2026-04-18] 事后诊断：这是 concept 问题还是 implementation 问题？

诊断基于 §13-14 所有观测数据。**我的判断：concept ≈ 60%，implementation ≈ 40%**。

### 15.1 concept 侧的证据（为什么 60%）

1. **aux 目标不击中瓶颈**。要从 0.868 打到 0.90 差的 3pp 是**进攻转化** (`wasted_possession` 在 029B 数据上占 64%)。"Teammate action prediction" 强迫的是**瞬时位置协调**，理论上限就是"少丢球 + 多堵位"。失败桶实测完美印证：possession_stolen ↓3.9pp 但 wasted_possession ↑4.1pp，WR 持平。**aux 做的事和想拿的分不是同一轴**。

2. **信息论上 aux 冗余**。team-level policy 的每个 agent 的 obs **已包含**队友位置/速度（ray casting + 近距离队友特征）。"用自己 obs 预测队友动作"的信息增益很小 —— encoder 不需要改变就能满足 aux。这解释了为什么效果量级小且只 turtle 化：aux 没给新的表征压力，只是让 encoder 对当前协调更敏感。

3. **同策略 self-supervised 问题**。team-level 两个 agent 共享 policy π。aux 在学 `π(a_teammate | obs_self)`，目标是 `actual_a_teammate = π(a_teammate | obs_teammate)`。本质是**自蒸馏**，非稳态目标（policy 变 → target 变），收敛到"什么都改变不了"不奇怪。

### 15.2 implementation 侧的证据（为什么 40% 而不是 0）

4. **λ=0.05 未调**（§0.5 明写"首轮先不用 weight ramping"）。aux loss 绝对值大小和 PPO loss 差了多少量级**我们不知道**（下一条）。

5. **aux loss / accuracy 没写进 `progress.csv`**（§12.1 明列为"未完成"）。这是最硬伤 —— 连 aux 有没有真的下降、accuracy 有没有超 1/27 chance (3.7%) 都不清楚。**但行为差异（turtle pattern + peer H2H 边缘赢）说明 aux 至少是有 gradient flow 且在影响 policy 的**，不是死的实验。

6. **aux 可能非对称**。`032-next: symmetric action aux refinement` 的存在暗示当前是单向 —— 对称双向预测会翻倍信号。

### 15.3 为什么不是 50/50 —— 关键判据

**反事实**：即使完美修 implementation（λ swept、aux 写入 progress、symmetric、深 aux head），**结果最多到 0.84-0.86**，不会到 0.90。因为 aux 优化的是"防守协调"，0.90 要的是"进攻转化"。

**反向反事实**：如果 concept 对（"完美短时协调"就能到 0.90），那即使 implementation 简陋，**至少能看到 baseline WR 显著超 Ctrl**。我们看到的是 peer H2H +5.6pp 但 baseline 持平 → 机制有效果但作用面不对。

### 15.4 对后续路线的含义（谨慎版）

| 路线 | 当前证据 | 优先级判断 |
|---|---|---|
| 032-next (symmetric aux) | 未跑，只是 §0.5 说过 "先不用 weight ramping" + aux acc 未落盘两个 impl gap | 如能跑得起 aux_acc logging，价值在于**把 §15.3 的 60/40 推测替换成硬数据**（参 §12 的诊断判据表）；不把它当刷分主线 |
| 033 (team reward shaping — PBRS spacing/coverage) | [snapshot-033 §13](snapshot-033-team-level-native-coordination-reward.md#13-首轮实现与结果033-a) 给出"narrow positive signal"：80/130 两 checkpoint 官评 0.826（warmstart +2pp），H2H vs 028A 一胜一负 | 单 lane 没 control，不能定性归因。下一步看 033-A-control / 033-B，**暂时不能武断判定路线强弱** |
| 036 (learned reward from W/L contrast) | Stage 3 W/L 排序 AUC 0.9772，signal 强且**直接定向 offense**（`wasted_possession` 是训练时的一个 head） | **主线**（训练中） |

**结论**：aux loss 实验不白做 —— 给出 null on WR + directional positive on robustness + 非传递性发现。对"coordination 路线够不够打到 9/10"的判断，当前**我只有 032 一条 lane 的直接证据**，外加 033 的"narrow positive / 需 control 归因"，不足以对"coordination-focused 整类路线"盖棺。

这类路线**短时间内最值得投入的诊断动作**是 033-A-control（同训 budget，关 shaping），因为它能同时回答:

1. 033A 的 0.826 官评高点是 PBRS 效应还是 028A@1060 continuation 本身；
2. 如果关 shaping 的 continuation 也能做到 0.82-0.83，那就是 028A 的空间还没跑干，而不是 PBRS/aux 错；
3. 反之，如果 control 明显低于 0.82，033A 的 narrow positive 就站住，值得再投第二轮。

## 11. 相关

- [SNAPSHOT-021: per-agent teammate aux head](snapshot-021-actor-teammate-obs-expansion.md)
- [SNAPSHOT-028: team-level BC base](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-031: dual encoder](snapshot-031-team-level-native-dual-encoder-attention.md)
- [SNAPSHOT-032-next: symmetric action aux refinement](snapshot-032-next-symmetric-action-aux.md)
- [SNAPSHOT-033: team reward shaping](snapshot-033-team-level-native-coordination-reward.md)
- [SNAPSHOT-036 §12.9: failure-bucket v2 redesign](snapshot-036-learned-reward-shaping-from-demonstrations.md)（本次 v2 归因所用）
