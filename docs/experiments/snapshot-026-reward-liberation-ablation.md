# SNAPSHOT-026: Reward Liberation Ablation

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 首轮 `A/B/C/D-warm` 内评 / official `baseline 500` / failure capture / 关键 head-to-head 已完成

## 1. 动机

截至 snapshot-025，项目的 shaping 进化一直是**加约束**方向：

```
v1 (ball_progress + possession + time_penalty)
→ v2 (+deep_zone + opp_progress_penalty)
→ v4 (+survival_bonus + fast_loss_penalty)
→ 022/024 (+per-agent role-diff shaping)
```

每一轮都在告诉 agent"不许做 X"或"做 Y 才对"。但 baseline-vs-baseline 的 failure 分析（[snapshot-013](snapshot-013-baseline-weakness-analysis.md)）显示 baseline 的赢法可能远比我们 shaping 假设的灵活：
- BvB 的 `unclear_loss` 占 47%——baseline 的比赛有近一半局面**不符合我们分类器预设的任何失败模式**
- BvB 的 `low_possession` 和 `poor_conversion` 都是 0%——baseline 从不犯这些"我们独有的"错

这暗示：**baseline 的策略不是"一直往前推 + 控球"，而是更灵活的混合打法**。而我们所有 shaping 设计都隐式假设"持续推进 + 控球 = 赢"——这套理论可能恰好**限制了 agent 发现 baseline 真正用的那种灵活打法**。

具体约束：
- `opp_progress_penalty` 惩罚"让对手推进"→ 限制**反击打法**（先退后再快速反击）
- `deep_zone_penalty` 惩罚"球在自家深区"→ 限制**诱敌深入**策略
- `time_penalty` 催促 agent 行动→ 限制**耐心等待 / 伺机而动**的打法
- `possession_bonus` 鼓励靠近球→ 限制**离球跑位拉空间**的配合

本 snapshot 测的是：**如果反过来——减约束、明确正向奖励、给 agent 更大的策略自由度——能不能突破 0.84 天花板？**

## 2. 三条路径

| 路径 | 核心思路 | 策略自由度 | 风险 |
|---|---|---|---|
| **A — 去惩罚** | 保留 ball_progress + possession_bonus，去掉所有 penalty 项 | 中 | scratch 可能太松散不收敛；warm-start 可能释放被压制的能力 |
| **B — Goal-Proximity (PBRS)** | 用 potential-based 位置价值替代 per-step dx 推进奖励 | 中高 | 改变 reward 信号结构，需要重新学习 |
| **C — Event-Based** | 去掉所有 per-step shaping，只奖励离散事件（射门/抢断/解围）| 最高 | 信号可能太稀疏，scratch 尤其可能学不动 |

**关键原则**：三条路径的共同方向是"**减约束、明确奖励**"——不是在加新 shaping term，是在**解放 agent 的策略空间**。

## 3. 路径 A：去惩罚版 shaping

### 3.1 设计

**保留**（正向信号）：
- `ball_progress_scale = 0.01`（保留推进的正向导向）
- `possession_bonus = 0.002 @ 1.25m`（保留控球的正向导向）
- sparse ±3 goal reward

**去掉**（所有惩罚性 term）：
- ~~`time_penalty`~~ → 0
- ~~`deep_zone_outer/inner_penalty`~~ → 0
- ~~`opp_progress_penalty_scale`~~ → 0
- ~~`defensive_survival_bonus`~~ → 0
- ~~`fast_loss_penalty`~~ → 0

**rationale**：agent 仍然知道"推进 + 控球 = 好"，但不再被告知"球在深区 / 对手推进 / 时间流逝 = 坏"。**防守策略完全由 agent 自己发现**。

### 3.2 只做 warm-start，不做 scratch

从 scratch 训时，没有任何惩罚 → agent 可能在 early-training 停在"随便乱踢 + 偶尔进球拿 +3"的 local optimum，因为没有足够 gradient 推它改进防守。

从 **BC @2100 warm-start**：agent 已经会踢球、会防守。去掉惩罚只是**放松对它的微观管理**——它可以继续用已有的防守行为，也可以发现"退防反击"等被 v2 shaping 压制的新策略。

## 4. 路径 B：Goal-Proximity (Potential-Based)

### 4.1 设计

**替换**：
- `ball_progress = scale × dx` → **`goal_proximity = γ × Φ(s') - Φ(s)`**
- 其中 `Φ(s) = -distance(ball, opponent_goal_center)`
- `opponent_goal_center` 对 team0 ≈ `(+15, 0)`（按 field 布局）

这是 **PBRS (Potential-Based Reward Shaping)**——理论保证不改变最优策略（[Ng, Harada, Russell 1999](../references/papers.md)）。

`ball_progress = dx` 不是 PBRS（它只看 delta-x，不看绝对位置到目标的距离）。PBRS 版本允许横传、回传后再突破——只要最终球比之前更接近对方门，就有正 reward。

**保留**：
- `possession_bonus = 0.002`（正向）
- sparse ±3

**去掉**：
- ~~`time_penalty`~~
- ~~`deep_zone_penalty`~~
- ~~`opp_progress_penalty`~~

### 4.2 参数

- `goal_proximity_scale = 0.01`（和 `ball_progress_scale` 同量级）
- `gamma_pbrs = 0.99`（和训练 gamma 对齐）
- `goal_x = 15.0`（field 半宽，需要 1-iter smoke 确认实际 goal 位置）

### 4.3 scratch + warm-start 都做

- **B-scratch**：500 iter from scratch → 测 PBRS 本身的收敛能力
- **B-warm**：300 iter from BC @2100 → 测在强底座上 PBRS 能否释放更多策略空间

## 5. 路径 C：Event-Based Reward

### 5.1 设计

**完全去掉所有 per-step shaping**。只奖励离散事件：

| 事件 | 检测条件 | reward | rationale |
|---|---|---|---|
| **shot_on_goal** | `ball_x > 10 AND ball_dx > 0`（球在攻击区且朝门移动）| +0.05 | 奖励"制造射门机会" |
| **tackle** | possessing_team 从对方翻转到我方（需要连续两步 state 对比）| +0.03 | 奖励"抢回球权" |
| **clearance** | `prev_ball_x < -8 AND curr_ball_x > -4`（球从深区到中场）| +0.03 | 奖励"解围成功" |
| **goal** | env 原生 | ±3（不变）| |

**不保留** `ball_progress / possession_bonus / time_penalty / deep_zone / opp_progress`。

**effect**：agent 被告知"做到这些事件 = 好"，但**不被指导如何到达这些事件**。策略自由度最大化。

### 5.2 工程需求

`soccer_info.py` 需要新增：
- `prev_ball_x` / `prev_possessing_team` 的状态跟踪（`RewardShapingWrapper` 已有 `_prev_ball_x`，可复用；possession tracking 需新增）
- `compute_event_shaping()` 函数返回 per-agent event reward
- 事件检测有一次性 cooldown（避免同一事件被连续多步重复触发）

### 5.3 scratch + warm-start 都做

- **C-scratch**：500 iter → event-based 信号可能太稀疏（每局只有几个事件），scratch 风险最高
- **C-warm**：300 iter from BC @2100 → agent 已经会制造 shot / tackle / clearance，event reward 只是**量化确认它已会的行为**，可能比 scratch 稳很多

## 5b. 路径 D：Entropy Regularization（正交对照）

### 5b.1 动机

路径 A/B/C 从 **reward 侧** 释放策略自由度。路径 D 从 **optimization 侧** 做同一件事：通过 entropy bonus 直接阻止 policy 过早坍缩到单一打法。

当前所有训练的 `entropy_coeff = 0.0`（PPO 默认值），意味着 agent 没有任何显式激励去维持策略多样性。如果 0.84 天花板部分来自 policy entropy 过低（agent 过早收敛到某种固定打法），那么加 entropy bonus 可以比改 reward shaping 更直接地解决问题。

### 5b.2 设计

**不改 reward shaping**——完全沿用 v2 shaping（ball_progress + possession_bonus + time_penalty + deep_zone + opp_progress_penalty），只改一个参数：

- `ENTROPY_COEFF=0.01`

### 5b.3 参数选择

- `entropy_coeff=0.01` 是 OpenAI Five 使用的值，属于 PPO entropy bonus 的标准起步值
- 对于 Discrete(27) action space，max entropy = ln(27) ≈ 3.3，`0.01 × 3.3 = 0.033` 的 entropy bonus 和当前 shaping scale（~0.02-0.05）在同一量级，不会淹没 shaping signal
- 只做 warm-start（BC @2100），不做 scratch

### 5b.4 为什么和 A/B/C 正交

A/B/C 改的是"给 agent 什么信号"。D 改的是"agent 怎么用信号学"。理论上可以组合（例如 B + D = PBRS + entropy bonus），但首轮先单独测，避免混淆归因。

## 6. 执行矩阵

| lane | shaping | warm-start | iter | GPU 预估 |
|---|---|---|---|---|
| **A-warm** | 去惩罚 v2 | BC @2100 | 300 | ~3h |
| **B-scratch** | goal-proximity PBRS | scratch | 500 | ~8h |
| **B-warm** | goal-proximity PBRS | BC @2100 | 300 | ~3h |
| **C-scratch** | event-based | scratch | 500 | ~8h |
| **C-warm** | event-based | BC @2100 | 300 | ~3h |
| **D-warm** | v2 shaping 不变 + entropy_coeff=0.01 | BC @2100 | 300 | ~3h |
| **总计** | | | | **~28h** |

**推荐跑序**：
1. **先跑 4 条 warm-start**（A/B/C/D-warm, 合计 ~12h）——warm-start 出结果最快、最有可能突破
2. 看 warm-start 结果决定 scratch 的优先级：
   - B-warm 过 0.86 → B-scratch 值得跑（验证 PBRS 从零也能 work）
   - C-warm 过 0.86 → C-scratch 值得跑
   - 都 ≤ 0.84 → scratch 预期也 ≤ 0.80，优先级降低

## 7. 预声明判据

### 7.1 主判据

| lane 类型 | 阈值 | 逻辑 |
|---|---|---|
| warm-start 四条 | **500-ep ≥ 0.86** | 从 BC @2100 (0.842) 起步，liberation shaping / entropy 应能至少推 +0.02 |
| scratch 两条 | **500-ep ≥ 0.81** | 追平 Pool 018 (0.812) = 新 shaping 设计不劣于 v2 |

### 7.2 机制判据

**failure capture 里"我方特有桶"(low_poss + poor_conv) 合计占失败比 ≤ 25%**

这比之前的 "low_poss ≤ 15%" 更务实——我们现在知道 low_poss 在所有 shaping 下都 ≥ 20%，设 15% 是不可达阈值。25% 对应的是"low_poss + poor_conv 合起来不恶化"。

### 7.3 "liberation" 特有判据

**failure distribution 更接近 BvB 形状**，但这里把 `unclear_loss` 明确降为**辅助诊断**而不是强成功判据。

原因：

- `unclear_loss` 一部分确实可能反映“策略更灵活、失败不再落入既有模板”
- 但它也受当前 failure classifier 覆盖范围影响
- 因此它适合作为 supporting signal，不适合作为单独的 hard gate

本 snapshot 更看重的 liberation 机制判据仍然是：

- `low_poss + poor_conv` 这两个“我方特有桶”是否下降
- `late_defensive_collapse` 是否至少不恶化

而 `unclear_loss` 的升高只作为：

- **“策略空间可能更开放了”** 的加分证据

| 桶 | 当前最强 lane (BC @2100) | 期望 liberation lane |
|---|---|---|
| late_defensive_collapse | 46.5% | ≤ 50%（不恶化）|
| low_poss + poor_conv | 39.5% 合计 | **≤ 30%** |
| **unclear_loss** | 11.6% | 若上升到 `~20%` 可视为 supporting signal，但不作为单独成败门槛 |

### 7.4 失败情形

| 条件 | 解读 |
|---|---|
| warm-start 任一条 500-ep < 0.80 | liberation shaping 破坏了 BC 底座能力 → 该路径 reward shock 太猛 |
| scratch 任一条 500-ep < 0.70 | shaping 信号太弱，从零学不起来 |
| warm-start WR 保持但 failure structure 和 v2 完全一样 | liberation 没有改变策略，只是"换了一种等价的约束" |

## 8. 路径 A/B/C 的互关系

三条路径不是三个独立假设——它们是**策略自由度的递进梯度**：

| 路径 | per-step 信号 | event 信号 | 策略约束程度 |
|---|---|---|---|
| **当前 v2** | ball_progress + possession + time + deep_zone + opp_progress | — | 最强约束 |
| **A (去惩罚)** | ball_progress + possession（只正向）| — | 中等 |
| **B (PBRS)** | goal_proximity（potential-based，理论无偏）+ possession | — | 低 |
| **C (event)** | — | shot + tackle + clearance | 最低约束 |
| **D (entropy)** | 同 v2 | — | 同 v2（但 optimization 侧增加多样性）|

如果**A-warm > B-warm > C-warm**（约束越强 WR 越高）→ agent 确实需要指导，当前 v2 方向对
如果**C-warm > B-warm > A-warm**（约束越少 WR 越高）→ **当前 v2 一直在限制 agent**
如果**B-warm 最高**（中间最优）→ 有一个 sweet spot，v2 约束过度但完全不约束也不行
如果**D-warm > A/B/C-warm**（reward 不变但 entropy 有效）→ **瓶颈不在 reward 设计而在 policy 坍缩**

**这就是为什么四条都跑：A/B/C 测"约束程度的最优点在哪"，D 测"瓶颈是不是根本不在 reward 侧"。**

## 9. 风险

### R1 — Warm-start 的 reward shock

BC @2100 的 value function 是在 v2 shaping 下训的。切到 B 或 C 后 value 估计全错 → 前 50 iter 可能 WR 暴跌。

缓解：
- 监控 iter 10/20/30 的 50-ep baseline WR
- 若 iter 30 还在 ≤ 0.60 → reward shock 太猛，需要渐进切换（例如前 100 iter 混 50% v2 + 50% 新 shaping，后 200 iter 纯新 shaping）

### R2 — C-scratch 可能信号太弱

Event-based reward 每局可能只触发 2-5 个事件（每个 +0.03-0.05）。和 sparse ±3 goal 相比，这些小 bonus 可能被 noise 淹没。

缓解：
- C-scratch 如果 iter 200 仍在 0.50 以下 → 调高 event reward（×2 或 ×3）
- 或接受 C 只在 warm-start 条件下有效

### R3 — Goal position 需要确认

Path B 的 PBRS 需要知道 opponent goal 的精确坐标。当前假设 `goal_x ≈ 15`，但需要 1-iter smoke 从 env info 或 obs 确认实际 goal 位置。如果 goal 不在 x=15 → proximity 计算偏移 → PBRS 不准。

### R4 — Event 检测的 false positive

Path C 的 "shot_on_goal" 检测 (`ball_x > 10 AND ball_dx > 0`) 可能 false-positive：球恰好在前场向前滚但没有射门意图。如果 false positive 太多 → agent 学到"把球放在 x>10 附近来回弹 = 拿 bonus"的 exploit。

缓解：加 minimum `ball_dx` 阈值（例如 `ball_dx > 0.5` 才触发）；或加 cooldown（同一事件每 10 步只触发一次）。

## 10. 不做的事

- **不改 obs / model / CC 结构**（纯 reward 侧实验）
- **不改 per-agent asymmetry**（本轮只测"约束程度"，不叠 022/024 的 role-diff）
- **不做 A-scratch**（用户判断 scratch 去惩罚可能不收敛；A 只做 warm-start）

## 11. 与既有 lane 的关系

| snapshot | 和 026 的关系 |
|---|---|
| [010 v2 shaping](snapshot-010-shaping-v2-deep-zone-ablation.md) | 026 的**逆操作**——010 加约束，026 减约束 |
| [017 BC @2100](snapshot-017-bc-to-mappo-bootstrap.md) | warm-start 底座来源 |
| [022 agent-id role-diff](snapshot-022-role-differentiated-shaping.md) | 如果 026 某路径 > 0.86，后续可叠加 role-diff |
| [024 field-role](snapshot-024-striker-defender-role-binding.md) | 同上 |
| [025 BC-champion field-role](snapshot-025-bc-champion-field-role-binding.md) | 并行跑，可能先出结果 |

## 12. 相关

- [SNAPSHOT-013: baseline weakness analysis](snapshot-013-baseline-weakness-analysis.md)（BvB failure 分布 → "为什么要 liberation"的核心依据）
- [SNAPSHOT-017: BC→MAPPO](snapshot-017-bc-to-mappo-bootstrap.md)（warm-start 底座）
- [code-audit-001 §2.2.1](../architecture/code-audit-001.md)（reward 对称性分析）
- [code-audit-000 §1.2](../architecture/code-audit-000.md)（原始 shaping 设计的"consider potential-based version"建议——从未执行过，现在 Path B 正是在补这个遗憾）

## 13. 执行清单

1. `soccer_info.py` 新增：
   - `compute_goal_proximity_shaping()`（Path B: PBRS delta）
   - `compute_event_shaping()`（Path C: shot/tackle/clearance 事件检测 + cooldown）
   - event tracking state（prev_possessing_team 等）
2. 1-iter smoke 确认 goal position（从 env info 或 obs 读 goal center 坐标）
3. 新 batch 脚本 × 5：
   - `soccerstwos_h100_cpu32_mappo_liberation_A_warm_bc2100_512x512.batch` ✅
   - `soccerstwos_h100_cpu32_mappo_liberation_B_scratch_512x512.batch`
   - `soccerstwos_h100_cpu32_mappo_liberation_B_warm_bc2100_512x512.batch` ✅
   - `soccerstwos_h100_cpu32_mappo_liberation_C_scratch_512x512.batch`
   - `soccerstwos_h100_cpu32_mappo_liberation_C_warm_bc2100_512x512.batch` ✅
   - `soccerstwos_h100_cpu32_mappo_liberation_D_warm_bc2100_512x512.batch` ✅
4. 先启动 4 条 warm-start（A/B/C/D-warm, ~12h GPU 并行）
5. 按 §7 判据做 official `baseline 500` 选模
6. 对 best ckpt 做 failure capture，重点看 §7.3 的"liberation 特有判据"
7. 按 warm-start 结果决定 scratch 优先级
8. verdict 落本文件 §14+（append-only）

## 14. 首轮 warm-start 初步结果

### 14.1 先说最重要的结论

四条 warm-start 都打出了**真实高点**，但它们支持的机制解释并不一样：

- **A-warm**：当前 ceiling 最高，最像“reward liberation 真能释放被 v2 压住的能力”
- **B-warm**：最像宽平台，PBRS 是目前最稳健的 liberation 路线
- **C-warm**：有正信号，但最抖，事件型 shaping 仍有工程/优化风险
- **D-warm**：最关键的对照臂。它在**完全不改 reward** 的前提下也打到 `0.90`，而且 `bad_count = 0`

因此，`026` 当前阶段不能简单收口为“reward liberation 已被证明”。更准确的判断是：

- **reward liberation 很可能有效**
- **但 exploration / entropy 也很可能是当前 0.84 天花板的重要主因之一**

### 14.2 四条 lane 的首轮内评

| lane | best internal `baseline 50` | best checkpoint | `best_reward_mean` | `bad_count` (`kl/total_loss inf/nan`) |
|---|---:|---:|---:|---:|
| `A-warm` | `0.96` | `290` | `-0.0879 @ 300` | `3` |
| `B-warm` | `0.90` | `260` | `+0.1865 @ 300` | `7` |
| `C-warm` | `0.90` | `210` | `+0.1444 @ 300` | `14` |
| `D-warm` | `0.90` | `220` | `-0.7991 @ 300` | `0` |

注意：

- 这四条 reward 定义不同，所以 `best_reward_mean` **不能横向比较**
- 当前真正可比的是：
  - `baseline 50` 曲线形态
  - 数值稳定性
  - 后续 official `baseline 500`

### 14.3 曲线形态初读

#### A-warm

- `baseline 50` 高点：`100 = 0.92`, `260 = 0.90`, `290 = 0.96`
- `random 50` 后段仍有 `0.96~1.00`
- `bad_count = 3`

当前四条里，**A 的 ceiling 最高**，而且不是单点噪声。

#### B-warm

- `baseline 50` 高点：`130/160 = 0.88`, `180/260 = 0.90`
- `140/170/200 = 0.86`
- `bad_count = 7`

**B 的峰值不如 A，但平台更宽、更稳。**

#### C-warm

- `baseline 50` 高点：`10/100/210 = 0.90`
- `50/280 = 0.88`
- `bad_count = 14`

**C 有真实正信号，但抖动最大。**

#### D-warm

- `baseline 50` 高点：`140/220 = 0.90`
- `280 = 0.84`
- `bad_count = 0`

这是目前最值得重视的对照结果：

- **在 reward 完全不变时，只加 entropy 就已经追平 A/B/C 的内评上限**

### 14.4 当前解释优先级

只看第一轮 warm-start 内评，我会把机制解释优先级排成：

1. **A-warm**：最强“reward liberation 可能有效”的证据
2. **D-warm**：最强“exploration 本身就是主因”的证据
3. **B-warm**：最稳的 reward-side 替代方案
4. **C-warm**：保留，但优先级最低

换句话说，当前最值得继续做 official `baseline 500` 的不是“只看 A/B/C”。
而是：

- **A-warm**：测 reward liberation 的冲顶能力
- **D-warm**：测单纯 entropy control 是否已经足够解释高点
- **B-warm**：测 PBRS 这条更干净、更理论化的 reward 路线
- **C-warm**：作为事件型最自由路径的保留项

### 14.5 official `baseline 500` 复核窗口

按 `top 5% + ties + 前后 2 点`，当前建议窗口如下：

- **A-warm**: `270 280 290`
- **B-warm**: `160 170 180 190 200 240 250 260 270 280`
- **C-warm**: `10 20 30 80 90 100 110 120 190 200 210 220 230`
- **D-warm**: `120 130 140 150 160 200 210 220 230 240`

这些窗口是当前最值得做 official `baseline 500` 的最小集合。

## 15. 首轮 official `baseline 500` 结果

按 §14.5 的窗口扫描完成。每个 lane 的最高点如下：

### 15.1 A-warm (去惩罚)

| checkpoint | official 500 |
|---|---:|
| 270 | 0.798 |
| **280** | **0.810** |
| 290 | 0.808 |

峰值 `0.810 @ checkpoint-280`，**低于 BC@2100 (0.842)**。内评 `0.960` 到官方 `0.810` 的 gap 是 `-0.150`——A-warm 的内评严重 overfit 新 reward 定义，在真实比赛里并没有突破。

### 15.2 B-warm (PBRS goal-proximity)

| checkpoint | official 500 |
|---|---:|
| 170 | 0.842 |
| 250 | **0.864** |
| 280 | 0.836 |

峰值 `0.864 @ checkpoint-250`。只看 official `baseline 500`，这是本项目第一次明显超过 `0.842` 天花板；但后续 `500ep` failure capture 与 head-to-head 表明，这个峰值**不能直接等价为新的总冠军**。

### 15.3 C-warm (event-based)

| checkpoint | official 500 |
|---|---:|
| **30** | **0.846** |
| 90 | 0.840 |
| 200 | 0.834 |
| 220 | 0.822 |

峰值 `0.846 @ checkpoint-30`，**略高于 BC@2100 的 official 峰值**。但它在 `500ep` failure capture 上回落到 `0.808`，说明这条线更像“早期高点 + 较大波动”，而不是稳定平台。

### 15.4 D-warm (entropy=0.01)

| checkpoint | official 500 |
|---|---:|
| 130 | 0.816 |
| **140** | **0.824** |
| 150 | 0.820 |

峰值 `0.824 @ checkpoint-140`，**低于 BC@2100**。D-warm 的内评 `0.900` 到官方 `0.824` 的 gap 是 `-0.076`——entropy 单独作用确实能抬升这条线，但不足以形成突破。

### 15.5 汇总表

| Lane | 内评 50 peak | Official 500 peak | @ ckpt | Δ vs BC@2100 (0.842) | 内评-官方 gap |
|---|---:|---:|---:|---:|---:|
| A-warm (去惩罚) | 0.960 | 0.810 | 280 | **-0.032** | -0.150 |
| **B-warm (PBRS)** | 0.900 | **0.864** | **250** | **+0.022** | -0.036 |
| C-warm (event) | 0.900 | 0.846 | 30 | +0.004 | -0.054 |
| D-warm (entropy) | 0.900 | 0.824 | 140 | -0.018 | -0.076 |

**内评-官方 gap 自己就是信号**：B-warm 的 gap 最小（`-0.036`），意味着它的 reward signal 和真实胜负目标在四条里最接近；A-warm 最大（`-0.150`），说明去惩罚后 policy 找到了“在 shaping 定义下看似赢但实际上不稳”的 exploit。

## 16. 失败桶（best ckpt）

### 16.1 对齐 BC@2100 / 025b@80 的统一表

| checkpoint | official | capture | total | late_def | low_poss | poor_conv | unclear | low_poss+poor_conv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B-warm @170** (更可信主候选) | 0.842 | 0.822 | 89 | 51.7% | **20.2%** | 7.9% | 14.6% | **28.1%** |
| B-warm @250 (official 峰值) | 0.864 | 0.810 | 95 | **47.4%** | 27.4% | 7.4% | 14.7% | 34.8% |
| B-warm @280 | 0.836 | 0.812 | 94 | 43.6% | 25.5% | 8.5% | 19.1% | 34.0% |
| C-warm @30 | 0.846 | 0.808 | 96 | 51.0% | 22.9% | 9.4% | 10.4% | 32.3% |
| D-warm @140 | 0.824 | 0.810 | 95 | 55.8% | 23.2% | 10.5% | 6.3% | 33.7% |
| **017 BC@2100** (ref) | 0.842 | — | — | 46.5% | — | — | 11.6% | 39.5% |
| **025b@80** (prev champ) | 0.842 | 0.836 | 82 | 50.0% | 34.1% | 4.9% | 9.8% | 39.0% |

### 16.2 对 §7 预声明判据的对照

| 判据 | 阈值 | B-warm @170 |
|---|---|---|
| §7.1 主判据（warm-start）| 500-ep ≥ 0.86 | 0.842 ❌ FAIL 严格阈值 |
| §7.2 机制判据 | low_poss + poor_conv ≤ 25% | 28.1% ❌ 略高于严格阈值 |
| §7.2 机制判据（松）| 比 017 (39.5%) 低 | **28.1% < 39.5% PASS 相对改善** |
| §7.3 `unclear_loss` ≤ 20% 上升可加分 | 14.6% | PASS（在"策略开放"区间）|
| §7.3 `late_def_collapse` 不恶化 | ≤ 50% | 51.7% ❌ 略高于阈值 |

如果只看更可信主候选 `170`，`026B` 并没有满足我们最激进的预声明阈值；但它依然给出了一个重要结论：PBRS 的价值更多体现在**失败结构改善和稳健性**，而不是把总体胜率稳定推到新冠军位。

### 16.3 机制解读

更细一点地看，`026B` 现在应当分成两个读法：

- **`250`**：official 峰值最高，但更像尖峰，small-sample / window effect 更重
- **`170`**：official/capture 对齐最好，`low_possession` 最低，是更可信的主候选

和 `025b@80` 相比，`170` 最有信息量的地方是：
- `low_poss + poor_conv` 明显更低（`28.1%` vs `39.0%`）
- 但 `late_defensive_collapse` 没有同步压下去

所以 `026B` 更像是在**把“我方特有失败桶”压掉**，而不是把整体比赛统治力推到冠军级。

### 16.4 训练轨迹：PBRS 在 warm-start 下的三个相位

把 `170 / 250 / 280` 三点按训练顺序串起来，可以看到一条典型的相位曲线：

| 相位 | ckpt | 特征 | 解读 |
|---|---|---|---|
| **持球改善相位** | `170` | `low_poss 20.2%` 最低、WR `0.842`、`late_def 51.7%` | PBRS 的稠密正向信号帮 policy 学到"更稳地控球" |
| **激进 exploit 相位** | `250` | WR 峰 `0.864`、`low_poss 27.4%` 回涨、`unclear 14.7%` | policy 放弃一部分持球稳定性，转而在 baseline 特定弱点上堆胜率 |
| **策略开放 / 退化相位** | `280` | `unclear 19.1%` 最高、WR 回落到 `0.836` | 策略进一步偏离 BC 的模板分布，但收益也开始消失 |

这不是三个"独立 best 候选"的选择问题，而是**同一训练轨迹被 PPO 带过三个 phase 的采样**。重要启示：

- 从 *失败结构* 角度，PBRS 的真收益在 `170` 附近
- 从 *baseline 胜率* 角度，被 PPO 继续优化到 `250` 时已进入 baseline-specific exploit（H2H 证明它不可迁移）
- 继续训练到 `280` 已经开始退化

这条相位曲线本身就是 PBRS 在 warm-start 下的**行为签名**。对未来类似 lane 的 checkpoint selection 有两个直接推论：

1. **official 峰值点未必是机制最干净的点**——应该把 official 曲线和 failure 桶曲线一起看
2. **"PBRS warm-start 的 sweet spot 出现得很早"**——这条线不值得训太久，`200-300 iter` 范围内基本已经把能榨取的正收益吃完

### 16.5 项目级别的机制突破：`low_possession` 不变量首次被打破

这一条单独拎出来，因为它的重要性超出 `026` 自身：

**`B-warm @170` 的 `low_possession = 20.2%`，是项目自启动以来第一次把这个桶压到 22% 以下。**

历史对照：

| lane | low_poss 占比 |
|---|---:|
| 14+ 条 per-agent lane（017 / 018 / 019 / 020 / 022 / 024 / 025 / 025b / v2 / v4 等）| 22%-34% 区间内浮动，从未跌破 22% |
| 021b-norm（teammate-obs expansion） | 34.5% |
| 025b@80（之前冠军位） | 34.1% |
| **B-warm @170** | **20.2%** |

之前我们把 `low_possession 22-28%` 当作 **CTDE + symmetric reward 架构的结构性下限**，这个假设现在被部分推翻：

- `low_possession` 不是纯架构问题——**通过修改 reward 设计可以将它压到 20% 附近**
- `B-warm @170` 的证据表明：`ball_progress = scale × dx` 这种 non-potential-based shaping 可能是 low_poss 不变量的**部分成因**（它只奖励 x 方向推进，对 agent 持球但不推进的状态不给正反馈）
- PBRS（`γ·Φ(s') - Φ(s)`）因为对"球离门近"本身赋值，即使 agent 在横向或静止持球也能获得非零信号，这可能正是 `low_poss` 被压低的机制

**对 027/028 的影响**：

- 如果 `low_poss` 在 reward 侧就能被打破一次（→20%），那么 team-level 架构是否能把它再压一档（→15% 甚至更低）就成了一个可量化的检验点
- 027/028 的机制判据可以在 "比 017 (39.5%) 低" 的软阈值之外，再加一个**"比 026B@170 (28.1%, low_poss+poor_conv) 更低"** 的更严格目标，作为"架构改善 + reward 改善叠加"的检验
- 这个发现也说明：`low_poss` 不是协调问题的唯一信号，它也受 reward 的 potential shape 影响；027/028 在做协调实验时要控制 reward 侧（保持 v2 shaping），才能把"架构 vs reward" 两个因素分离

## 17. 首轮 verdict（加入 H2H 后）

### 17.1 先说最重要的结论

`026` 的首轮结果**不支持“新冠军已经出现”**。

- `B-warm @250` 的 official `baseline 500 = 0.864` 很强，但 `500ep` failure capture 回落到 `0.810`
- `B-warm @170` 才是更可信的主候选：`0.842 -> 0.822`，而且 `low_possession` 压到 `20.2%`
- `B-warm @250` 在 H2H 里：
  - 对 [017@2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 仅 `244W-256L = 0.488`
  - 对 [025b@80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) 为 `227W-273L = 0.454`
- `D-warm @140` 对 `017@2100` 也只有 `230W-270L = 0.460`

所以 `026` 当前最稳的定位不是“新 SOTA”，而是：

**reward liberation / exploration 改写能造出很强的 baseline specialist，但这一轮还没有超过当前冠军链。**

### 17.2 四条路径的明确排序

| 路径 | 首轮结论 |
|---|---|
| **B (PBRS)** | `026` 内部最强；official 峰值在 `250`，但更可信主候选是 `170`；failure structure 有真实改善，但 H2H 仍输 `017` 与 `025b` |
| C (event) | 有真信号，但 capture 回落明显，更像高波动早期强线 |
| D (entropy) | 明确有效，但 ceiling 不够高，且 H2H 输 `017` |
| A (去惩罚) | **负结果**——内评严重 overfit，官方掉到 0.810；v2 的 penalty 不是多余约束 |

这里有一个重要的**修正**：

- 只看 official `baseline 500`，会很容易把 `B-warm @250` 误读成新冠军
- 加上 `500ep` failure capture 和 H2H 之后，这个结论站不住

所以 `026` 目前最准确的故事是：

- **exploration 是重要因素**
- **reward 重新设计也带来了 failure-structure 改善**
- 但两者在当前版本上都还不足以把 `026` 推到 `025b` 之上

### 17.3 A-warm 的反向证据

A-warm 的负结果非常有信息量：它说明 v2 shaping 的 `time_penalty / deep_zone / opp_progress_penalty` **不是多余的约束**，而是**真的在防止 policy 作弊**。去掉它们后，policy 学到"在新 reward 定义下看起来赢但实际上不稳"的打法——0.960 的内评是这种 exploit 的直接证据。

这同时也部分修正了 §1 的动机预设："shaping 限制策略空间" 这个断言需要细化：**penalty 型 shaping 的限制不等于有害约束**，PBRS 才是真正应当替换 `ball_progress = scale × dx` 的那一项。

### 17.4 H2H 结果

| matchup | result |
|---|---:|
| `026B@250 vs 017@2100` | `244W-256L = 0.488` |
| `026B@250 vs 025b@80` | `227W-273L = 0.454` |
| `026D@140 vs 017@2100` | `230W-270L = 0.460` |

这三组对打一起说明：

- `026B` 很会打 baseline，但还**不是**当前最强通用策略
- `026D` 证明 entropy-only 不是主突破口
- `026` 的收益不能简化成“只要加 entropy 就够了”，但也不能被解释成“reward liberation 已经登顶”

### 17.5 当前归属

- **`026` internal official 峰值**：[B-warm @checkpoint-250](../../ray_results/PPO_mappo_liberation_B_warm_bc2100_512x512_20260417_062526/MAPPOVsBaselineTrainer_Soccer_d3698_00000_0_2026-04-17_06-25-59/checkpoint_000250/checkpoint-250) = `0.864`
- **`026` 更可信主候选**：[B-warm @checkpoint-170](../../ray_results/PPO_mappo_liberation_B_warm_bc2100_512x512_20260417_062526/MAPPOVsBaselineTrainer_Soccer_d3698_00000_0_2026-04-17_06-25-59/checkpoint_000170/checkpoint-170) = `0.842 official / 0.822 capture`
- **`026` 当前最佳结论**：强 baseline-oriented 正结果，但不是当前总冠军
- **当前总排序**：`025b` 仍在前面；`026B` 尚未通过 H2H 挤掉 `017/025b`

## 18. 下一步优先级

### 18.1 短期（把 026 收成更稳的实验结论）

1. **若继续追 H2H，优先 `B@170 vs 024@270` 或 `B@170 vs 017@2100`**：这会告诉我们“更可信主候选”是否比 `250` 更接近通用强线
2. **保留 `250` 作为 baseline specialist 参考点**：如果课程评分最终极度偏向 baseline-only 指标，`250` 仍然有单独讨论价值

### 18.2 B/C-scratch：**当前不跑**

原预算里包含 `B-scratch` / `C-scratch`（合计 ~16h GPU）。基于 §17 的 H2H 结果，这两条 scratch 当前**降级为可选**，不再列为中期主线：

- H2H 已经表明 PBRS 产生的是 **baseline-specific profile**，不是"更强的 agent"
- B-scratch 不论落在 0.70 / 0.81 / 0.86 哪一档，都不会改变我们的主线决策（不会回去用 PBRS 作默认 shaping）
- 16h GPU 放到 `027-A` 长训或 `028` 的 Stage 1+2 上信息量更高

**保留作为可选**：如果 report 阶段需要完整的 `warm-start × scratch × A/B/C` 对照表来支持 "reward liberation 路径分析"这一节，再补跑。

### 18.3 中期（PBRS 的使用方式）

3. **不把 `goal_proximity_pbrs` 设为 v2 的默认替换** — H2H 证据表明 PBRS 的收益不是普适性技能提升
4. **保留 B-warm @250 作为"baseline specialist"候选** — 如果作业最终评分以 `baseline 9/10` 为绝对门槛，B-warm 可能仍是最好的提交点（0.864 最接近 0.90 阈值）
5. **B-warm 的 `768x512` 宽度对照** — 当前 512x512 已经 0.864，更宽网络是否能进一步上推？**优先级中等**，等 027/028 出结果后再决定

### 18.4 长期（和 027/028 的交叉）

6. 027/028 的 team-level 路线是真正针对"`low_possession` 不变量 + 协调瓶颈"的架构级实验——优先级高于 026 的任何 scratch 补实验
7. 如果 027/028 出正结果，可以在 team-level 架构上**单独测一次 PBRS**，看看 PBRS 的"baseline-specific" 性质是否和架构相关
8. A-warm 的负结果暗示 "更激进的 reward liberation" 是错方向，后续不再沿此路线加新 A-lane

### 18.5 保留但降低优先级

- C-warm 的 early peak 值得再做一次 rerun 确认是不是 seed-specific（低优先级）
- D-warm 整体降低优先级，entropy 调优可以在 B-warm 之上作为 secondary knob（如 B-warm + ENTROPY_COEFF=0.005）
