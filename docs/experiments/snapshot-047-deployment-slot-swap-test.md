# SNAPSHOT-047: Deployment 0/1 Slot Swap Test (Train-Deploy Symmetry Validation)

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 已完成最小 sanity subset / 已形成可执行 verdict

## 0. 为什么现在做

[SNAPSHOT-022 §R1](snapshot-022-role-differentiated-shaping.md#r1) 早在 2026-04 中就明确写出：

> **Side-swap eval**：本 P0 只验证了训练环境的 spawn 稳定性，**不能**替代 eval 时 team0 ↔ team1 交换后 agent_id 映射的独立检查
> **Inference 层的 agent_id 语义一致性**：如果 deployment wrapper 在某些路径里把 team0 slot 0/1 反过来塞给本地 `{0,1}`，训练时学到的 agent-id 行为差异会被打乱

[SNAPSHOT-024 §10](snapshot-024-striker-defender-role-binding.md#10) 也把 "side-swap 分析" 列为 TODO：
> 5. 若有正信号，再补 failure capture 与 side-swap 分析

但**没有任何 lane 实际跑过 swap test 把数据落下来**——只有 risk/TODO 笔记。当时 022/024 的 WR (0.83-0.84) 还有上行空间，没有人逼着把这个 risk 转成数据。

2026-04-19 第一性原理复盘后情况变了：
- 5+ lane saturate 在 [0.852, 0.865]，单点优化已停滞
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) ensemble 是当前最高 ROI 突破口
- **ensemble 收益严重依赖 train-deploy slot 一致性**——如果 031A 的 "agent 0" 在部署时被反喂，与 036D 的 "agent 0" 不对齐，ensemble averaging 会把两个 policy 的不同行为强行平均，失去 PETS 的"互补"性质

所以 047 现在不是 "验证 022 的旧 risk"，而是 **034 ensemble 的前置 sanity check**。这是**1-2h 工作量解锁 1 天工程的 ensemble**——ROI 极高，必须做。

## 1. 核心假设

### H1（slot 对称）

> Deployment wrapper 对 team0 内部 slot 0/1 的映射是**确定性、与训练时一致**的。即：训练时学到的 "agent 0 行为" = 部署时 wrapper 提供的 obs[0] 对应的策略输出。

如果 H1 成立 → ensemble 安全，可以直接做 probability averaging
如果 H1 不成立 → ensemble wrapper 必须显式做 slot 对齐 / 或选 slot-invariant 的策略组合（如 031A Siamese 共享 encoder）

### H2（swap 后 WR 不变）

> 强制把 deployment 时 `obs[0]` 与 `obs[1]` 互换喂给 policy，然后把输出 action 也互换回来——单独评估每个 SOTA checkpoint 的 1000ep WR 应该 **不变**（在 SE ±0.016 范围内）。

如果 H2 fail（swap 后 WR 显著下降）→ checkpoint **学了 slot-binding 行为**，agent 0 与 agent 1 不对称
如果 H2 pass → checkpoint 是 **slot-invariant**，wrapper 无 slot 序问题

## 2. 测试设计

### 2.1 实验矩阵

原始预注册矩阵包含四个代表性 checkpoint，每个跑 normal + swap 两个条件：

| checkpoint | 架构 | 预期 |
|---|---|---|
| **031A@1040** (Siamese) | shared encoder | swap-invariant by design → WR 不变 |
| **036D@150** (per-agent + learned reward) | 独立 per-agent | 可能 slot-binding → WR 可能变 |
| **029B@190** (per-agent v2) | 独立 per-agent | per-agent baseline |
| **025b@80** (per-agent + field-role binding) | 显式 role binding | **预测 swap 后 WR 显著下降**（如果 swap 真的改变行为） |

025b 是关键 positive control——它是 explicit field-role binding 的 lane，如果连 025b 都对 swap 不敏感，说明 wrapper 层把 slot 序统一了，所有 ensemble 都安全。如果 025b 对 swap 敏感而 031A 不敏感，符合架构差异预期。

**2026-04-19 实际执行范围（最小 sanity subset）**

首轮先完成了最有信息量的两组：

| checkpoint | normal | swap | Δ(swap-normal) |
|---|---:|---:|---:|
| **031A@1040** | 0.851 | 0.839 | -0.012 |
| **025b@80** | 0.824 | 0.832 | +0.008 |

也就是说，当前 verdict 来自 **031A + 025b 的最小 sanity 子集**，而不是原始 4-checkpoint 全矩阵。`036D / 029B` 的扩展 swap test 仍可在后续需要时追加。

### 2.2 实现方式

**Option A（推荐，零代码改动）**: 写一个 wrapper agent 类继承现有 `trained_*_ray_agent`，重写 `act()` 方法：

```python
class SlotSwappedAgent(TrainedRayAgent):
    def act(self, observation):
        # observation = {0: obs_a, 1: obs_b}
        swapped_obs = {0: observation[1], 1: observation[0]}
        swapped_action = super().act(swapped_obs)
        # swap action back so env sees correct mapping
        return {0: swapped_action[1], 1: swapped_action[0]}
```

放在 `cs8803drl/deployment/_swap_test/` 下，仅用于测试，不进 submission 路径。

**Option B（环境侧 swap）**: 在 evaluator 调用前 swap obs dict 的 key——但这需要改 evaluator 代码，污染主路径。**不采用**。

### 2.3 评估流程

每个 checkpoint × {normal, swap} 两个条件：

1. 跑 1000ep vs baseline（用 `evaluate_official_suite.py`，与 standard eval 完全一致）
2. 同种 seed 序列（确保唯一变量是 swap）
3. 记录 WR、tie 率、failure structure（v2 桶）

总 GPU 时间: 4 ckpts × 2 conditions × 1000ep ≈ 4-6h（可并行 -j 3）

### 2.4 判据

| 条件 | 解读 |
|---|---|
| ckpt × {normal, swap} 差 ≤ 0.022 (1000ep SE 2σ) | swap-invariant，slot 不影响该 ckpt |
| ckpt × {normal, swap} 差 ∈ (0.022, 0.05] | 弱 slot-binding，但实际影响小 |
| ckpt × {normal, swap} 差 > 0.05 | **强 slot-binding**，ensemble 必须显式对齐该 ckpt |

原始计划中的最终输出（数据回填后）：

| ckpt | normal WR | swap WR | Δ | binding 判定 |
|---|---:|---:|---:|---|
| 031A@1040 | 0.851 | 0.839 | -0.012 | 未观察到强 binding |
| 036D@150 | TBD | TBD | TBD | TBD |
| 029B@190 | TBD | TBD | TBD | TBD |
| 025b@80 | 0.824 | 0.832 | +0.008 | 未观察到强 binding |

## 3. 对 [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) ensemble 的指导

根据 047 verdict，034 的 wrapper 设计有三种分支：

### 3.1 全 swap-invariant（最理想）

如果 4 ckpt 都 |Δ| ≤ 0.022 → 直接 probability averaging，无需对齐。这意味着 deployment wrapper 已经统一了 slot 序，且训练得到的 policy 也是对称的。**ensemble 工程量最小**。

### 3.2 部分 binding

如果某些 ckpt slot-binding 强（典型: 025b、036D 之一）但 031A invariant → ensemble 用 "slot-canonical" 策略：
- 调用每个 policy 时，按 wrapper 的 sorted player_ids 顺序固定喂入
- 不同 policy 不需要在它们之间统一 slot 0/1，只需要每个 policy 自己稳定

这是当前 wrapper 已经做的（`sorted(player_ids)`）。**ensemble 工程量中等**——增加单元测试验证。

### 3.3 全 binding

如果 4 ckpt 都对 swap 敏感（不太可能） → ensemble 必须给每个 policy 学一个 slot-mapping：
- 用 spawn-x 符号判断当前 episode 是否需要 swap，re-route obs
- 工程量大，需要在 wrapper 里维护 episode-level state

## 4. 与 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 的关系

022 的 P0 验证了**训练时**的 spawn 稳定性（agent 0 在 blue 方天然偏深）。但训练 ≠ 部署：

- 训练时 env 会自然返回 spawn-conditioned obs[0/1]
- 部署时 evaluator 调用 wrapper.act(observation)，observation 的 key 顺序由 evaluator 决定
- 中间还有可能经过 ActionFlattener、reward wrapper、单/多 worker 路径

047 是在**部署侧**直接 A/B test，不依赖任何中间假设。

## 5. 风险

### R1 — Swap 实现 bug 让 swap test 自身无效

如果 SlotSwappedAgent 的 swap 写错（swap obs 但忘记 swap action），结果会无意义。

**缓解**: 实现后先做 sanity check：
- 用一个明显有 left/right 偏好的 toy policy 验证 swap 真的改变了 evaluator 看到的 action 序
- 单元测试 `obs[0] != obs[1]` 时，wrapper.act() 输出的 dict 是否和未 swap 不同

### R2 — Sample size

1000ep × SE 0.016 × 2σ = 0.032。如果 ckpt 的真实 swap-effect 是 ±0.025，1000ep 可能区分不出来。

**缓解**: 如果 1000ep 落在 (0.022, 0.05] 模糊带，加跑到 2000ep 缩窄 SE。

### R3 — Deployment wrapper 的 sorted() 已经把 swap 抹掉了

`trained_team_ray_agent.py` 的 `act()` 用 `sorted(player_ids)`。如果 evaluator 永远以 {0, 1} 顺序喂入，sorted() 是 no-op，swap test 直接展示 policy 真实 sensitivity。如果 evaluator 反向喂 {1, 0}，sorted() 又把它捋直了。

**这正是 047 要回答的**——sorted() 是不是真的让 wrapper 对 evaluator 的 key 顺序 invariant。R3 不是缓解，是 047 的核心问题之一。

## 6. 不做的事

- **不重新训练**任何 ckpt——047 是纯部署测试
- **不改主 deployment 路径**（`trained_*_ray_agent.py` 不动）
- **不在 047 范围内做 ensemble** ——那是 034 的事
- **不测 4 ckpt 之外的 lane**（保持范围最小，回填后再决定要不要扩）

## 7. 执行清单

1. 创建 `cs8803drl/deployment/_swap_test/swap_wrapper.py`（30min）
2. 写单元测试验证 swap 实现正确（30min）
3. 在 GPU 节点 launch 4 ckpts × 2 conditions 1000ep（4-6h）
4. 用 v2 桶分析 failure structure 是否也跟着变（30min）
5. 回填 §2.4 表格 + §8 verdict
6. 把 verdict 同步给 [SNAPSHOT-034 §1.3](snapshot-034-deploy-time-ensemble-agent.md#13-ensemble-的潜在收益) 残留风险段，决定 ensemble wrapper 的实现分支

## 8. Verdict

### 8.1 最小 sanity subset 的结论

当前 `031A@1040 / 025b@80` 的 `normal vs swap` 结果都没有显示出强 slot-binding：

- `031A@1040`: `0.851 -> 0.839`，差值 `-1.2pp`
- `025b@80`: `0.824 -> 0.832`，差值 `+0.8pp`

这两个量级都落在 `1000ep` 单次 official eval 的常见波动范围内，不足以支持“deployment swap 会显著改变 checkpoint 行为”的结论。也就是说，在当前 deployment wrapper + baseline-1000 口径下，**没有观察到强 slot-swap 敏感性**。

### 8.2 对原始假设的修正读法

这轮最重要的发现不是“验证了 031A 比 025b 更稳定”，而是：

- `031A` 的确没有表现出明显 swap 敏感性
- 但预设的 positive control `025b` 也**没有**表现出明显敏感性

这意味着当前最小 sanity test 更支持：

> deployment 层至少没有明显把 team0 的 0/1 slot 语义搞乱到会在 baseline WR 上造成可观测退化

而不是：

> 已经证明 031A 完全 slot-invariant、025b 强 slot-binding

### 8.3 对 [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) 的直接影响

047 的最小 sanity subset 已经足够回答 034 的前置问题：

- **目前没有证据表明** `034-A` 必须先加入更复杂的 slot-canonical handling 才能继续
- 因此 `034` 不再被 047 阻塞，可以先按现有 wrapper 往前推进

如果后续 `034-A` 出现异常不稳、或在 H2H 口径下出现可疑退化，再补 `036D / 029B` 的 normal/swap 或更强的 H2H swap test 即可。

## 9. 相关

- [SNAPSHOT-022 §R1](snapshot-022-role-differentiated-shaping.md) — 提出 side-swap risk 的源头
- [SNAPSHOT-024 §10](snapshot-024-striker-defender-role-binding.md) — TODO 列出但从未执行
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — Siamese 共享 encoder, slot-invariant by design
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) — 047 的下游消费者（前置依赖）
- [trained_team_ray_agent.py:154](../../cs8803drl/deployment/trained_team_ray_agent.py#L154) — 当前 wrapper 的 `sorted()` 处理点
