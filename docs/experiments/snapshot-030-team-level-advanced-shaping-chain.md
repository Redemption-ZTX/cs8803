# SNAPSHOT-030: Team-Level Advanced Shaping Chain

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册 / 待实现

## 0. 方法论前置（必读）

### 0.1 现状

per-agent 路径已经形成完整的 **base → advanced shaping** 链：

```
017 (BC + v2)                                  = 0.842 baseline (ref)
  → 025b (BC + v2 + field-role binding)        = 0.842 + H2H 赢 017
  → 029B (BC + B-warm@170 init + v2 handoff)   = 0.868 + H2H 平 025b ← per-agent SOTA
```

team-level 路径目前只跑到 **base**：

```
027 (scratch + v2)                             = 0.804
028A (BC + v2)                                 = 0.844 baseline
```

最近 H2H 数据显示 028A@1220 vs 017 = `0.466`（输 0.068），看上去 team-level 落后 per-agent。但**这是不公平的对比**。

### 0.2 不公平在哪里

当前所有方法栈——v2 shaping 系数、BC pipeline 设计、网络架构（512×512）、超参（lr=1e-4 / sgd_iter=4 / clip=0.15）、training trick——都是为 **single / per-agent multi-agent setup** 设计、调优、验证的。把它们 port 到 team-level 后，得到的"team-level 表现"实际上是：

> **"为 per-agent 优化过的方法栈，在 team-level 架构上的表现"**

这不等于"team-level 架构本身的上限"。

### 0.3 信息论直觉

team-level 架构 actor 看到 672 维联合 obs，per-agent 架构 actor 只看 336 维 own obs。**信息量更大的输入理论上应该有更高上限**——因为 per-agent 架构能学到的策略集是 team-level 能学到的策略集的真子集（team-level policy 可以 ignore 队友 obs 退化为 per-agent policy）。

如果 team-level 在所有指标上都被 per-agent 打败，**只能说明我们的方法栈没用上多出来的那一半信息**，不能说明"team-level 架构没用"。

### 0.4 本 snapshot 的范围与限制

本 snapshot **只做 port 实验**：把 per-agent 的 advanced shaping trick 直接搬到 team-level，作为：

1. **最低门槛检查**：team-level 架构能否从已知有效的 per-agent trick 中吃到收益
2. **基线对比**：为后续 team-level-native 方法实验（snapshot-031+）提供对照
3. **架构判断的下界**：如果 port 实验都能让 team-level 显著进步，说明架构有 headroom；如果 port 全失败，架构的 headroom 必须靠 native 方法挖

**不**做 team-level-native 方法（独立 snapshot 处理）：
- 双 encoder 网络（两人 obs 分别编码后融合）
- 跨 agent attention
- 团队协作 auxiliary loss
- 联合动作的 hierarchical decomposition
- 团队级 reward shaping（基于 formation / 区域分工的奖励）

## 1. 三条 lane 的设计

按 per-agent advanced 的 029 模板，但适配 team-level base。

| 030 lane | 对应 per-agent | warm-start | shaping | opponent |
|---|---|---|---|---|
| **030-A** | 025b (field-role) | 028A@1220 | v2 + field-role binding | baseline |
| **030-B** | 029-C (opp pool) | 028A@1220 | v2 | pool |
| **030-C** | 029-B (PBRS handoff) | （需 030-D 先跑出 PBRS mech 点）| — | — |
| **030-D** | 029-A (PBRS on base) | 028A@1220 | PBRS only | baseline |

030-D 同时是 030-C 的前置：必须先有 team-level PBRS 训练，才知道有没有"team-level B-warm @170 等价物"，才能做 handoff。

按风险/收益排序，**首轮先跑 030-A 和 030-B 并行**：
- 030-A 风险最低（25b 在 per-agent 上明确成功，H2H 赢 017）
- 030-B 工程最复杂（需要 4-slot opp pool，但基础设施已存在）
- 030-D 是开放探索（PBRS 在 per-agent 029-A 上失败，但 team-level 上可能不同）
- 030-C 等 030-D 出结果再决定

## 2. 路径 A — `028A + team-level field-role binding`

### 2.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 028A@1220 | team-level base SOTA |
| shaping 类型 | v2 + spawn-depth field-role binding | 025b 配置 |
| network | 512x512 | 匹配 028A |
| `lr` | 1e-4 | 025b/029 同款 |
| `num_sgd_iter` | 4 | 同上 |
| `clip_param` | 0.15 | 同上 |
| opponent | baseline 100% | 纯对照 |
| `SHAPING_FIELD_ROLE_BINDING` | 1 | 启用 |
| `SHAPING_FIELD_ROLE_BINDING_MODE` | spawn_depth | 025b 同款 |

### 2.2 工程检查点

- [ ] 确认 `RewardShapingWrapper` 在 `team_vs_policy` 模式下正确路由 per-agent role-bound reward
- [ ] 1-iter smoke：训练 log 应该看到 `role_by_agent = {0: 'striker', 1: 'defender'}`（或反之）

### 2.3 假设

**H_A 成立**：team-level 架构能从 per-agent 验证过的 field-role binding 中吃到同等收益。

预期：
- 028A→030-A 的 official 500 提升 +0.00 到 +0.02（类比 017→025b 在 per-agent 上的收益）
- H2H vs 028A@1220 ≥ 0.52
- H2H vs 017@2100 改善（从输 0.068 拉到 ≥ 0.50）

**H_A 不成立**：
- field-role binding 在 team-level 上无效或负效应
- 暗示 team-level 架构对"per-agent 风格的 reward asymmetry"不响应
- 进一步暗示需要 team-level-native asymmetry 设计（如 sub-policy / hierarchical action）

## 3. 路径 B — `028A + opponent pool`

### 3.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 028A@1220 | team-level base |
| shaping | v2（不变）| 028A 同款 |
| network | 512x512 | 同上 |
| PPO | tight params 同 029-C | |
| opponent pool | baseline 40% / 017@2100 20% / 025b@80 20% / 028A@1220-self 20% | 复用 029-C 思路但 self 替换 |

注意：029-C 的 pool 是 `baseline / 017 / 024 / 025b-self`。030-B 的 pool 是 `baseline / 017 / 025b / 028A-self`——把 024 换成 025b（因为 025b 比 024 更强，且 028A 还没和 025b 在 H2H 中赢过）。

### 3.2 工程关键点

team-level policy 用的是 `trained_team_ray_agent` deployment wrapper，per-agent SOTA (017/025b) 用的是 `trained_shared_cc_agent`。**opponent pool 现有基础设施只支持 frozen per-agent checkpoint**，需要确认能否混合 team-level 和 per-agent opponent。

如果不能，030-B 需要先解决 deployment wrapper 互通问题，或者 pool 限定为 per-agent opponent only。

### 3.3 假设

**H_B 成立**：team-level policy 在 per-agent peer pool 中训练后，对同级 per-agent 的 H2H 表现改善（从输 0.07 拉到 ≥ 0.50）。

如果 H_B 成立但 H_A 失败，说明 team-level 的核心弱点在 **out-of-distribution against per-agent style**，而 opponent pool 训练能直接修复这个弱点。

## 4. 路径 D — `028A + PBRS only`（开放探索）

### 4.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 028A@1220 | team-level base |
| shaping | PBRS only（同 029-A 配置） | |
| network | 512x512 | |

### 4.2 假设

**H_D 成立**：team-level 架构上 PBRS 比 v2 更适合，能压低 `low_poss`。

注意 029-A（per-agent + PBRS）失败了。但 team-level 的 reward landscape 不一样——PBRS 的"球离门近就好"信号在 team-level 联合视角下可能引发不同行为。

如果 H_D 成立且产生明显 `low_poss` 改善（≤ 25%），030-D 的 best ckpt 就是 030-C 的 PBRS handoff 起点。

如果 H_D 失败，030-C 也不必跑——直接说明 PBRS 在 team-level 上也不是钥匙。

## 5. 执行矩阵

| lane | 复杂度 | 预算 | 优先级 |
|---|---|---|---|
| **030-A** field-role binding | 低 | ~6h | **首轮 #1** |
| **030-B** opp pool | 中（需确认 wrapper 兼容性） | ~8h | **首轮 #2** |
| **030-D** PBRS only | 低 | ~6h | **首轮 #3** |
| **030-C** PBRS handoff | 中（依赖 030-D 结果） | ~6h | 第二轮 |

GPU 同时跑 3 条 lane（A/B/D），16h 内完成首轮。

## 6. 预声明判据

### 6.1 主判据

| 030 lane | official 500 阈值 | H2H 阈值 |
|---|---|---|
| 030-A | ≥ 0.86（追平 026B@250 / 接近 029B@190）| H2H vs 017 ≥ 0.50 + H2H vs 028A 略赢 |
| 030-B | ≥ 0.84（不损失 baseline WR）| H2H vs 017/025b/029B 都 ≥ 0.48 |
| 030-D | ≥ 0.84 | failure capture 中 `low_poss` ≤ 25% |

**任一 lane 触发 H2H ≥ 0.50 vs 029B**，team-level 架构就证明了它的潜力，下一步明确转向 team-level + 更多 trick。

### 6.2 失败情形

| 条件 | 解读 |
|---|---|
| 三条 lane 全部 official < 0.83 | per-agent advanced trick 在 team-level 上不通用 → 必须做 team-level-native 方法（snapshot-031+）|
| 030-A 显著 +0.02 但 030-D 失败 | 架构感知 reward asymmetry 但不感知 PBRS 差异 → 优先继续 field-role 方向 |
| 030-A/B/D 都接近 028A 水平（≈ 0.84） | per-agent trick 完全没在 team-level 上 transfer → 转 native 方法 |

## 7. 与现有 lane 的关系

| snapshot | 关系 |
|---|---|
| [025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) | 030-A 的 per-agent 模板 |
| [029-A](snapshot-029-post-025b-sota-extension.md) | 030-D 的 per-agent 模板 |
| [029-B](snapshot-029-post-025b-sota-extension.md) | 030-C 的 per-agent 模板 |
| [029-C](snapshot-029-post-025b-sota-extension.md) | 030-B 的 per-agent 模板 |
| [028A](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | 030 所有 lane 的 warm-start 源 |

## 8. 不做的事

- **不做 team-level-native 方法**（双 encoder / cross-agent attention / 团队级 reward）—— 这是 snapshot-031+ 的范围
- **不做架构改动**（FCNet 层数 / actor-critic 拆分）—— 030 只测 reward & opponent 改动
- **不并行做 030-C**（必须等 030-D 出结果决定有无必要）

## 9. 工程依赖

### 9.1 已存在

- 028A@1220 checkpoint
- 027/028 team-level 训练入口
- per-agent field-role binding env vars（`SHAPING_FIELD_ROLE_BINDING=*`）
- per-agent PBRS env vars（`SHAPING_GOAL_PROXIMITY_*`）
- per-agent opp pool 基础设施（`POOL_*_CHECKPOINT`）

### 9.2 需要确认

- field-role binding 的 spawn-depth role 计算在 `team_vs_policy` env 下是否正确
- `RewardShapingWrapper` 的 per-agent reward 在 `TeamVsPolicyWrapper.step()` sum 之前是否仍然 role-aware
- opp pool 能否混合 team-level 和 per-agent opponent（或限定 per-agent only）

### 9.3 可能需要新增

- 如果 opp pool 不兼容混合，需要为 team-level policy 写一个独立的"frozen per-agent opponent" adapter（类似 snapshot-023 但反向）

## 10. 执行清单

1. 确认工程依赖 §9.2 的三项
2. 起 batch：030-A / 030-B / 030-D
3. 1-iter smoke 验证：
   - 030-A：log 中能看到 `role_by_agent = {...}`
   - 030-B：log 中能看到 `opponent_pool: baseline:0.40, ...`
   - 030-D：log 中能看到 `shaping_goal_prox: 0.01`
4. 提交首轮 3 条并行（~16h）
5. 按 §6 判据做 verdict
6. 如果有 lane 触发 030-C 条件，下一轮起 030-C
7. 决定是否进入 snapshot-031（team-level-native 方法）

## 11. 相关

- [SNAPSHOT-025b: per-agent field-role binding](snapshot-025b-bc-champion-field-role-binding-stability-tune.md)
- [SNAPSHOT-027: team-level scratch](snapshot-027-team-level-ppo-coordination.md)
- [SNAPSHOT-028: team-level BC bootstrap](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-029: per-agent advanced shaping trio](snapshot-029-post-025b-sota-extension.md)
