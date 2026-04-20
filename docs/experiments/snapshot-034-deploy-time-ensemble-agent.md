# SNAPSHOT-034: Deploy-Time Ensemble Agent

- **日期**: 2026-04-18 (创建) / 2026-04-19 (优先级提升 + v2 证据)
- **负责人**:
- **状态**: 旧版 per-agent triad 已完成并否决；frontier mixed（`031A` / `031B` anchors）已完成首轮与 follow-up

## 0. 2026-04-19 第一性原理复盘后的优先级提升

把 040(4 lanes) + 041B + 044 + 045/046 全 launch 完之后做 5-lane saturation 复盘，得到一个关键认知：

> **0.86 不是 PPO 上限，而是「单点优化天花板」。033 之后我们一直在猜「再加一勺 X」，但跨 5 个 lane 全 saturate 在 [0.852, 0.865]——证据强烈指向「单 checkpoint 已经到顶，必须从 ensemble / 部署侧突破」。**

更重要的是，**v2 失败桶**（snapshot-036 §12）重新分类后，**两个最强单 checkpoint 的失败模式相反**——这正好是 ensemble 期望生效的条件（详见 §1.3 v2 evidence）。

本 snapshot 从「预注册」推到「优先实施」。优先级理由：
- **零新训练**（用现有 ≥ 5 个 ≥ 0.84 资产）
- **理论支撑**（PETS 经验 + 失败模式正交）
- **工程可控**（1-2 天完成 wrapper + eval）
- **路径独立**（与 045 / 046 / 039-fix 并行不冲突 GPU）

### 0.1 当前文档的两个时间层

这份 snapshot 现在同时包含两套历史口径，必须分开读：

1. **创建时的 v1 执行矩阵**
   - 以 `029B + 025b + 017` 这组纯 per-agent triad 作为首轮 `034-A`
   - 这是本文档 `§2.3 / §5` 最早落地的那组执行计划

2. **2026-04-19 的 v2 frontier 动机**
   - 基于 failure-buckets-v2，真正最强的正交证据来自 `031A / 036D / 029B`
   - 这组 frontier 资产才是当前 snapshot 想继续追的 ensemble 方向

因此，后文中已经跑完的 `034-A` 结果，只能解读为：

> **旧版 per-agent triad (`029B + 025b + 017`) 的首轮 verdict**

而不能被误读成：

> **整个 034 frontier ensemble 假设已经被否决**

## 0. 灵感来源

HW3（Model-Based RL / PETS）的核心方法论：**多个独立训练的模型 ensemble，在期望上做 planning**。PETS 用 5 个独立 dynamics model 处理 epistemic uncertainty。

> "Ensembled modeling helped performance ... A likely reason is that averaging over multiple independently trained models reduced overconfidence and made planning more robust to model error."

迁移到我们：**多个独立训练的 policy ensemble，在 deploy 时做投票/平均**。

## 1. 为什么现在做

### 1.1 当前局面

我们手里有 ≥ 5 个 official 500 ≥ 0.84 的 checkpoint，**来自完全不同的 path / 架构 / shaping / trick**：

| checkpoint | 架构 | bootstrap | shaping | official 500 |
|---|---|---|---|---|
| 017 BC@2100 | per-agent MAPPO | BC | v2 | 0.842 |
| 025b@80 | per-agent MAPPO | BC | v2 + field-role binding | 0.842 (+ H2H 强) |
| **029B@190** | per-agent MAPPO | BC + B-warm@170 init | v2 handoff | **0.868** |
| 026B@250 | per-agent MAPPO | BC | PBRS | 0.864 (baseline-specific) |
| 028A@1220 | **team-level PPO** | team BC | v2 | 0.844 |

这是天然 diverse 的资产组合。

### 1.2 距离作业 9/10 阈值还差 +0.032

最强单 checkpoint 是 029B@190 = `0.868`。距离 `0.90 = 9/10` 还有 **+0.032**。

我之前测过的 mechanism-side 突破方向：
- field-role binding (025b) → +0.000 baseline
- PBRS handoff (029B) → +0.026 baseline
- opp pool (029C) → +0.006 baseline
- team-level (028A) → +0.002 baseline

每一条**单 checkpoint 改进**都在 +0.006 ~ +0.026 区间。**单点优化看起来已经接近天花板**。

### 1.3 Ensemble 的潜在收益

#### 旧版 v1 失败桶证据（保留）

不同 path 学到的策略**有不同的失败模式**：
- 029B@190 失败结构：`low_poss 26.2%, late_def 51.2%`
- 025b@80：`low_poss 34.1%, late_def 50.0%`
- 028A@1220：`low_poss 29.8%, late_def 47.1%, poor_conv 14.4%`

这些 weakness **正交不重叠**——029B 控球弱、025b 控球更弱、028A 转化差。Ensemble 应该把 weakness 投票稀释掉。

按 PETS 经验，独立 model ensemble 通常 +0.02-0.05 over best single。如果在 baseline WR 上能拿 +0.03，**029B 的 0.868 → 0.90**，正好打到 9/10 门槛。

#### 2026-04-19 v2 失败桶重新分类（强化证据）

v1 桶有已知偏见（snapshot-036 §12）：阈值 `mean_ball_x < -0.15` 是把 raw Unity 单位 (~±15) 当 normalized 处理，几乎所有"球微微偏左"都被标 `late_defensive_collapse`，false positive 严重。

用 [v2 桶](../../cs8803drl/imitation/failure_buckets_v2.py) 重新对 SOTA checkpoint 的 failure capture 分类（multi-label）：

| ckpt | n | defensive_pin | wasted_possession | possession_stolen | tail_ball_x median | poss median |
|---|---:|---:|---:|---:|---:|---:|
| **031A@1040** (Siamese, 0.860) | 85 | 40 (47%) | 36 (42%) | 28 (33%) | **+1.37** | 0.500 |
| **036D@150** (per-agent + learned reward, 0.860) | 88 | 48 (55%) | 34 (39%) | 34 (39%) | **-4.90** | 0.448 |
| 029B@190 (per-agent v2, 0.868) | 136 | 63 (46%) | 65 (48%) | 44 (32%) | +3.27 | 0.527 |

**两个 0.86 的 SOTA 失败模式相反**：
- **031A**: tail_x=+1.37（球留在对方半场）+ poss median 0.50 → **wasted_possession 主导**：有球但转化失败
- **036D**: tail_x=-4.90（球深陷自方半场）+ poss median 0.45 → **defensive_pin 主导**：被压制无法突围

这是比 v1 旧证据**更强**的 ensemble 论据——两个 SOTA 不是「都犯同样的错只是程度不同」，而是「在不同情境下犯错」。Probability averaging 的期望收益会高于 v1 估计：

- 当 ball 在对方半场 → 031A weight 高（它会主动找射门窗口），036D weight 低（它倾向 over-defensive）
- 当 ball 在自方半场 → 036D weight 高（它学了如何 clear），031A weight 低（它转化能力强但防守相对弱）

**预测**: PETS 经验 +2-5pp 的下界更可能成立。**031A (0.860) 或 029B (0.868) + ensemble +0.03 → 0.890-0.898**，跨 0.90 门槛概率显著提升。

#### 残留风险

ensemble 收益的前提是 deployment 阶段 agent 0/1 的 slot 语义稳定（snapshot-022 §R1 提出过 risk，但长期没有数据化验证）。如果训练时 agent 0/1 与部署时映射不一致，ensemble 不同 policy 的 action 概率分布会"对不齐"，averaging 反而注入噪声。

**2026-04-19 更新**: [SNAPSHOT-047](snapshot-047-deployment-slot-swap-test.md) 的最小 sanity subset（`031A@1040 / 025b@80` 的 normal/swap `baseline 1000`）未观察到强 slot-binding：`031A Δ=-0.012`，`025b Δ=+0.008`。这说明 047 已经完成了解锁 034 所需的最小前置检查，**034 不再被 slot-swap 风险阻塞**；后续只有在 ensemble 结果异常时，才需要回头补更强的 swap/H2H 验证。

### 1.4 对未知 agent2（bonus +5）的天然防御

bonus 题是 **vs 未公布的 agent2**。单 checkpoint 是"针对 baseline 优化的 specialist"，对未知对手风险高。Ensemble 是 **rounded policy**，对策略空间均匀覆盖，更抗未知 exploit。

## 2. 设计

### 2.1 Ensemble 类型对比

| 类型 | 描述 | 优 | 劣 |
|---|---|---|---|
| **A. Action voting** | 每 policy 各自 sample 一个 action，多数票 | 简单，离散动作直接做 | 信息浪费（只看最终 action）|
| **B. Logit averaging** | 每 policy 输出 logits，平均后再 sample | 用上完整概率分布 | 不同 policy 的 logit scale 可能不同 |
| **C. Probability averaging** | 每 policy softmax 后平均，再 sample | scale 一致；离散版本最 principled | 计算多一步 |
| **D. Random checkpoint** | 每 act() 随机选一个 policy | 最简单；天然多样性 | 同一局内策略不连贯，可能很糟 |

**推荐 C（probability averaging）** 作为主线。理由：
- 离散动作的 PETS 等价物是"每 model 给一个分布，期望上 plan"
- scale 不变性：每个 policy 输出归一化概率，无论 logit 大小
- 退化情形优雅：所有 policy 同分布时，ensemble = single policy

D（random checkpoint）作为对照——验证 C 的收益是否来自 averaging 还是仅来自 stochasticity。

### 2.2 Cross-architecture 兼容性

挑战：**team-level policy 和 per-agent policy 的接口不同**。

| 类型 | inference 输入 | inference 输出 |
|---|---|---|
| per-agent (017/025b/029B/026B) | 336 维 single agent obs | Discrete(27) per agent |
| team-level (028A) | 672 维 team obs | MultiDiscrete([3,3,3,3,3,3]) joint |

**解决方案**：在 ensemble 内部为每个 policy 调用合适的 wrapper：

```python
def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    # observation = {0: obs_336, 1: obs_336}

    # 收集每 policy 的 per-agent action 概率分布
    per_agent_probs_0 = []
    per_agent_probs_1 = []

    for policy in self.policies:
        if policy.is_per_agent:
            p0 = policy.action_probs(observation[0])  # 27 dim
            p1 = policy.action_probs(observation[1])
            per_agent_probs_0.append(p0)
            per_agent_probs_1.append(p1)
        elif policy.is_team_level:
            joint_obs = np.concatenate([observation[0], observation[1]])
            joint_probs = policy.action_probs(joint_obs)  # 6×3 MultiDiscrete
            # 转换成 per-agent Discrete(27)
            p0 = multi_discrete_to_discrete(joint_probs[:3])
            p1 = multi_discrete_to_discrete(joint_probs[3:])
            per_agent_probs_0.append(p0)
            per_agent_probs_1.append(p1)

    # ensemble: 平均概率，sample
    avg_probs_0 = np.mean(per_agent_probs_0, axis=0)
    avg_probs_1 = np.mean(per_agent_probs_1, axis=0)

    return {
        0: sample_from(avg_probs_0),
        1: sample_from(avg_probs_1),
    }
```

**关键工程问题**：
- team-level policy 的 6 维 MultiDiscrete 联合分布投影到 per-agent 27 维 Discrete 时，**两个 agent 的 action 不再 jointly distributed**——失去了 team-level 联合决策的优势
- 折中方案：team-level 输出可以被视为"已经 marginalized 的 per-agent 概率"，再和 per-agent 平均

这个折中**减弱了 team-level 在 ensemble 中的贡献**，但保留了 architectural diversity。如果 team-level 在 ensemble 里完全无效，可以从 ensemble 里去掉。

### 2.3 子 ensemble 候选

不是所有组合都跑——计算量爆炸。下面这组是 **创建时的 v1 执行矩阵**：

| lane | ensemble 成员 | 假设 |
|---|---|---|
| **034-A** | `029B@190 + 025b@80 + 017@2100` | 同架构 (per-agent) ensemble，最干净的 lifting test |
| **034-B** | 034-A + `028A@1220` | 加跨架构成员，测 cross-arch 是否 lift |
| **034-C** | 034-A + `026B@250` | 加 baseline specialist，测 specialist 是否帮助 |

**首轮先跑 034-A**。如果 034-A 显著 > 029B@190 → 验证 ensemble 方向，再跑 B/C。

**2026-04-19 frontier 读法更新**

结合 `§1.3` 的 v2 失败桶证据，当前更值得继续的 frontier 资产并不是旧版 `017/025b/029B` 三元组，而是：

- `031A@1040`
- `036D@150`
- `029B@190`（可选 anchor）

也就是说，`§2.3` 这组矩阵现在应理解为：

- **旧版首轮执行矩阵**
- 不是当前 frontier 假设的完整代表

### 2.4 Inference 成本

每个 act() 需要：
- 同时加载 N 个 policy 到 GPU (~50MB each = 250MB for N=5)
- 每步 N 次 forward pass

实测预估：
- 每 forward pass ≈ 5-10ms on H100
- N=5 ensemble: 25-50ms / step
- Soccer-Twos step interval：~100ms（Unity binary 渲染速度限制）
- **延迟充足，不会卡 env**

## 3. 工程实现

### 3.1 新文件

```
agents/v034a_ensemble_pa3/         # 034-A (3 per-agent policies)
  ├── __init__.py
  ├── agent.py                     # EnsembleAgent class
  ├── README.md
  └── checkpoints/
      ├── 017_bc2100/              # symlinks or copies
      ├── 025b_ckpt80/
      └── 029B_ckpt190/
```

### 3.2 EnsembleAgent 实现要点

- 复用 `cs8803drl/core/checkpoint_utils.py` 的 weight 加载
- 每个子 policy 实例化为 lightweight inference-only torch model（不需要 RLlib trainer overhead）
- `act()` 内部并行 forward（torch batched）
- 缓存 trick：如果同一步两个 agent 的 obs 不同但都 query 同一个 policy，可以打 batch=2 一次 forward

### 3.3 评估流程

1. 实现 EnsembleAgent + smoke test (10 episodes vs baseline)
2. 验证：
   - 所有 N 个 policy 加载成功
   - 推理延迟 ≤ 80ms / step（不阻塞 Unity）
   - 输出的 action 分布合理（不全 0 / 不退化为某单 policy）
3. official 500 vs baseline
4. official 500 vs random（防 deploy-time bug 退化）

## 4. 预声明判据

### 4.1 主判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **034-A vs baseline 500** | **≥ 0.88** | 至少 +0.012 over 029B@190；接近 9/10 |
| **034-A vs random 500** | ≥ 0.99 | basic competence 不退化 |
| 034-A H2H vs 029B@190 | ≥ 0.50 | ensemble 在对等对抗下不弱于最强 single |

### 4.2 突破判据

| 项 | 阈值 | 含义 |
|---|---|---|
| 034-A vs baseline 500 | **≥ 0.90** | **达到 9/10 门槛**，Performance 25 分满分 |
| 034-A vs baseline 500 | ≥ 0.88 + 034-B/C 进一步推高 | 架构组合给 ensemble 加成 |

### 4.3 失败判据

| 条件 | 解读 |
|---|---|
| 034-A vs baseline 500 < 0.86 | ensemble 不如 best single (029B = 0.868) → averaging 损害策略 |
| 034-A vs random < 0.95 | ensemble 退化（可能是 logit scale 不一致 / sample bug）|
| 034-B 加入 028A 后掉分 > 0.02 | 跨架构组合损害 → 028A 排除出 ensemble |
| inference 延迟 > 100ms | Unity step 阻塞，无法部署 |

### 4.4 Gaming 防护

ensemble 主要风险**不是 reward gaming**（没有训练阶段），而是 **deployment 实现 bug**：

1. **Sanity check 必跑**：vs random 必须 ≥ 0.99——任何不到的情况都是 bug 或 cross-arch 转换错误
2. **Logit scale 监控**：log 每 policy 的 logit 范数，差异 > 10x 时 ensemble 可能被某 policy 主导
3. **失败时降级**：如果 034-B/C 引入 028A 后 vs baseline 显著掉，立刻去掉 team-level 成员
4. **同分布 baseline**：实现一个 `EnsembleAgent(policies=[029B, 029B, 029B])`（3 份相同 policy）作为 sanity——这种"假 ensemble"应该 = 029B single 的水平，验证 ensemble 框架本身没有引入退化

## 5. 执行矩阵

| lane | 成员 | 预算 | 优先级 |
|---|---|---|---|
| **034-A** | 029B + 025b + 017 | inference only, ~30min | **首轮主线** |
| 034-A-control | 3 × 029B (same policy) | ~30min | sanity baseline |
| 034-B | 034-A + 028A | ~30min | 条件启动 |
| 034-C | 034-A + 026B | ~30min | 条件启动 |

**没有训练阶段**，全是 inference-time evaluation。从开始到 verdict ≤ 1 天。

### 5.1 2026-04-19 旧版 034-A 首轮 official 500 结果

首轮先完成了预注册中的两条最高信息量 lane：

| lane | baseline 500 | random 500 | 解读 |
|---|---:|---:|---|
| **034-A** (`029B + 025b + 017`) | **0.806** (403W-97L) | **0.984** (492W-8L) | ensemble 本体明显低于 best single |
| **034-A-control** (`3 × 029B`) | **0.850** (425W-75L) | **0.992** (496W-4L) | sanity 通过，框架未明显损坏 |

对照当前单 checkpoint 参考：

- `029B@190 official 500 = 0.868`
- `029B@190 official 1000 = 0.846`

因此对 **旧版 per-agent triad** 的最稳读法是：

- `034-A-control = 0.850` 与 `029B` 的高置信度 baseline 水平基本一致，说明 **ensemble inference 框架本身没有被明显写坏**
- `034-A = 0.806` 则比 `034-A-control` 低 `4.4pp`，也明显低于当前 best single

也就是说，问题不在“能不能把多个 policy 一起跑起来”，而在于：

> **当前这版等权 probability averaging (`029B + 025b + 017`) 会伤害 baseline 轴表现，而不是带来 lift。**

这条结论只覆盖：

- 旧版 `034-A`
- 以及它所代表的 “纯 per-agent triad + 等权平均” 这一种 recipe

**它并不自动否定** `031A / 036D / 029B` 这组由 `§1.3` 推出来的 frontier ensemble 假设。

### 5.2 2026-04-19 frontier mixed 首轮结果

在把 `047` 的最小 slot-swap sanity 跑完、并把 ensemble 默认决策口径从“平均后随机采样”改回与单 checkpoint deployment 对齐的 **deterministic / greedy** 之后，frontier mixed follow-up 的首轮 official `500` 给出如下结果：

| lane | 成员 | baseline 500 | random 500 | 读法 |
|---|---|---:|---:|---|
| `034B-control` | `2 × 031A@1040` | `0.836` (418W-82L) | `0.992` (496W-4L) | team-level handle sanity |
| **`034B-frontier`** | `031A@1040 + 036D@150` | **`0.872`** (436W-64L) | **`0.996`** (498W-2L) | 最小 frontier 双组合为真阳性 |
| `034C-control` | `3 × 031A@1040` | `0.822` (411W-89L) | `0.996` (498W-2L) | 三成员 team-level control |
| **`034C-frontier`** | `031A@1040 + 036D@150 + 029B@190` | **`0.890`** (445W-55L) | **`0.996`** (498W-2L) | 当前 mixed ensemble 主候选 |

更关键的是 frontier lane 相对各自 control 的提升：

- `034B-frontier - 034B-control = +0.036`
- `034C-frontier - 034C-control = +0.068`

这说明 frontier 组合不是“侥幸没坏”，而是 **在同口径 control 之上给出了实质 lift**。同时，`034C-frontier` 相比 `034B-frontier` 再多加一个 `029B` 后，baseline `500` 从 `0.872` 进一步推到 `0.890`，说明 `029B` 在这个 mixed ensemble 里是加分项，而不是干扰项。

### 5.3 2026-04-19 `034C-frontier` follow-up：official 1000 / failure capture / H2H

围绕当前最强 mixed lane `034C-frontier = 031A + 036D + 029B`，补做的 follow-up 结果如下：

| 评测 | 结果 | 读法 |
|---|---:|---|
| official `baseline 1000` | `0.843` (843W-157L) | baseline 轴明显低于它的 `500ep` 峰值 |
| failure capture `500` | `0.872` (436W-64L) | 与 `500ep official` 更接近 |
| H2H vs `031A@1040` | `0.580` (290W-210L) | 明显优于当前 team-level 单模型 SOTA |
| H2H vs `029B@190` | `0.624` (312W-188L) | 对当前 per-agent 强线给出更强优势 |

因此，`034C-frontier` 当前最稳的读法不是“新 baseline 冠军已坐实”，而是：

> **baseline 轴处于中高 `0.8x`，但 peer-H2H 很强，是一条明显偏向 peer-play 的 mixed ensemble 主候选。**

也就是说，它不像 `029B/031A` 那样能被简单压缩成单一 baseline 排名，而更像：

- 对 baseline：强，但不一定最稳
- 对强 peer：明显更强

这类组合在 bonus / 未知 agent 对抗场景里反而可能更有价值。

### 5.4 `034C-frontier` failure structure

`034C-frontier` 的 [summary.json](artifacts/failure-cases/034C_frontier_3way_baseline_500/summary.json) 给出：

- `team0_win_rate = 0.872`
- `fast_win_rate = 0.808`
- 总体均步数 `44.5`
- 败局均步数 `33.2`

按 `summary.json` 中 `saved_episode_paths` 统计的 64 个保存败局，主失败桶为：

- `late_defensive_collapse = 32`
- `low_possession = 18`
- `unclear_loss = 13`
- `opponent_forward_progress = 1`

这说明它的主要短板是：

- **后程防守掉线**
- **控球 / 控场不稳**

而不是系统性 finishing 崩坏，也不是 territory / forward-progress 被长期压穿。再结合 `fast_win_rate = 0.808`，这条线更像是：

> **攻击性很强、赢得快，但输的时候也往往是早早掉线的“尖锐型” ensemble。**

一个小的文档严谨点：保存目录里实际存在少量旧的 partial JSON，后续统计应以 `summary.json.saved_episode_paths` 为准，而不是直接按目录文件数计数。

### 5.4.1 `failure_buckets_v2` 对照：`034C` 到底更像谁

为了把 `034C` 的失败模式和当前最重要的单模型参照线放到同一口径下，我们用 [failure_buckets_v2.py](../../cs8803drl/imitation/failure_buckets_v2.py) 对各自 `summary.json.saved_episode_paths` 中保存的 baseline 败局重新分类：

| model | n | defensive_pin | territorial_dominance | wasted_possession | possession_stolen | progress_deficit | unclear_loss | tail_ball_x median | poss median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `031A@1040` | 85 | 47.1% | 50.6% | 42.4% | 32.9% | 15.3% | 12.9% | `+1.372` | `0.500` |
| `036D@150` | 88 | 54.5% | 55.7% | 38.6% | 38.6% | 21.6% | 10.2% | `-4.895` | `0.448` |
| `029B@190` | 80 | 51.2% | 47.5% | 48.7% | 32.5% | 23.7% | 8.7% | `-3.629` | `0.543` |
| **`034C-frontier`** | **64** | **45.3%** | **42.2%** | **31.2%** | **35.9%** | **25.0%** | **20.3%** | **`-0.707`** | **`0.433`** |

这张表给出一个比 v1 桶更有解释力的结论：

- `034C` **不像 `031A` 那样“有球但转化失败”**
  - `wasted_possession = 31.2%`
  - 明显低于 `031A = 42.4%`
  - 也明显低于 `029B = 48.7%`

- `034C` 也**不像 `036D / 029B` 那样长期被钉在自家半场**
  - `defensive_pin = 45.3%`
  - 低于 `036D = 54.5%`、`029B = 51.2%`
  - `tail_ball_x median = -0.707`，已经明显比 `036D / 029B` 更接近中场

- 所以 `034C` 更像一种 **rounded compromise**
  - 同时削弱了 `031A` 的 `wasted_possession`
  - 也削弱了 `036D / 029B` 的 `defensive_pin / territorial_dominance`

但这个折中的代价也很清楚：

- `progress_deficit = 25.0%` 是这组里最高
- `unclear_loss = 20.3%` 也是这组里最高

也就是说，`034C` 不是靠“把某一个病灶修到极致”变强，而是：

> **把几条强单模型各自的主要短板同时压下去，但把一部分失败模式换成了更分散、更难一句话概括的 mixed loss。**

这和它现在的整体画像是一致的：

- baseline 轴并不稳定地压过所有单模型
- 但 peer-H2H 很强
- 更像一条 **更全面、更难被单点 exploit 的 rounded ensemble**

### 5.5 2026-04-19 `031B` anchor 第二轮 frontier mixed：`034D / 034E`

在把 `031B@1220` 纳入 failure-v2 对照后，一个更自然的问题变成了：

> `031B` 已经部分吸收了 `031A` 的 failure 弱点，那 `034` 下一步更该围绕 `031B` 而不是 `031A` 做新 anchor 吗？

为回答这个问题，我们实现并评测了两组新的 mixed ensemble：

| lane | 成员 | baseline 500 | random 500 | 读法 |
|---|---|---:|---:|---|
| `034D-control` | `2 × 031B@1220` | `0.914` (457W-43L) | `0.996` (498W-2L) | `031B` anchor 的最小 control；projection tax 明显更小 |
| `034D-frontier` | `031B@1220 + 036D@150` | `0.880` (440W-60L) | `0.994` (497W-3L) | **低于** `2 × 031B` control，说明 `036D` 单独接入 `031B` 会伤 baseline |
| `034E-control` | `3 × 031B@1220` | `0.876` (438W-62L) | `0.996` (498W-2L) | 三成员 `031B` control |
| **`034E-frontier`** | `031B@1220 + 036D@150 + 029B@190` | **`0.904`** (452W-48L) | **`0.998`** (499W-1L) | 当前 `034` 主候选；首次在 ensemble 线上稳定站到 `0.90` 附近 |

这组结果把两件事说得很清楚：

- `031B` 当 anchor 是对的
  `2 × 031B` control 已经达到 `0.914`，明显高于之前 `2 × 031A = 0.836`、`3 × 031A = 0.822`，说明 mixed ensemble 里的 team-level projection tax 在 `031B` 身上小得多。

- `029B` 是关键补强项
  `034D-frontier = 0.880` 明显低于 `034D-control = 0.914`，说明 **`031B + 036D` 这一组本身不成立**；但把 `029B` 加回后，`034E-frontier = 0.904` 不仅高于 `034E-control = 0.876`，也重新打到当前 ensemble 线的最高 `baseline 500`。

因此，`034` 的下一步主线已经从：

- `031A + 036D (+029B)`

切换为：

- **`031B + 036D + 029B`**

### 5.6 `034E-frontier` follow-up：official 1000 / peer H2H

围绕当前最强 ensemble lane `034E-frontier = 031B + 036D + 029B`，继续补做的 follow-up 结果如下：

| 评测 | 结果 | 读法 |
|---|---:|---|
| official `baseline 1000` | `0.890` (890W-110L) | 相比 `500ep 0.904` 只回落 `1.4pp`，比 `034C` 稳定得多 |
| `034E-control` official `baseline 1000` | `0.869` (869W-131L) | `034E-frontier` 相对同 anchor control 仍保留 `+2.1pp` |
| H2H vs `031B@1220` | `0.596` (298W-202L) | 对当前 strongest single model 给出真实正优势 |
| H2H vs `029B@190` | `0.590` (295W-205L) | 对 per-agent frontier 也有稳定优势 |
| H2H vs `034C-frontier` | `0.544` (272W-228L) | 方向上支持 `034E > 034C`，但优势仍属中等幅度 |

当前最稳的综合读法是：

> **`034E-frontier` 已经是当前 `034` 的新主候选。它不是单纯的 baseline specialist，也不是只会打 peer 的奇技，而是一条同时在 baseline 与 peer 两条轴上都站住的 ensemble frontier。**

和 `034C` 相比：

- `baseline 1000`: `0.890 > 0.843`
- `vs 029B`: `0.590` 仍为硬正号
- `vs strongest single (031B)`: `0.596`
- `vs 034C`: `0.544`

因此，`034C` 现在更适合保留为“第一代 frontier mixed 参考”，而 `034E` 应升格为：

- **`034` 主候选**
- **当前 ensemble 主候选**

## 6. 工程依赖

### 6.1 已存在

- 5 个 best checkpoints（017/025b/029B/026B/028A）
- `cs8803drl/core/checkpoint_utils.py` 的 weight loading 工具
- `cs8803drl/deployment/trained_shared_cc_agent.py`（per-agent 推理）
- `cs8803drl/deployment/trained_team_ray_agent.py`（team-level 推理）
- 官方 evaluator (`scripts/eval/evaluate_official_suite.py`)

### 6.2 需要新增

- `cs8803drl/deployment/ensemble_agent.py`：通用 EnsembleAgent 框架
- `agents/v034a_ensemble_pa3/agent.py`：具体 034-A 配置
- batch 脚本（如果需要在 GPU 节点跑 official 500）

### 6.3 关键技术问题

- team-level → per-agent 概率投影（§2.2 折中方案）
- 多 policy 同 GPU 的内存与并行 forward
- AgentInterface dict 接口的 cross-arch 路由

## 7. 和其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [017 / 025b / 029B / 026B / 028A] | ensemble 成员来源 |
| [029](snapshot-029-post-025b-sota-extension.md) | 提供 best single 029B@190 |
| [030/031/032/033] | team-level native 探索；如出 SOTA 也可加入 ensemble |
| [035](snapshot-035-ppo-stability-backlog.md) | 平行的 PPO-side stability 实验（advantage norm / entropy / twin value）|

## 8. 不做的事

- **不做训练阶段 ensemble**（如 distillation、joint training）—— 这是另一个 snapshot 的事
- **不做超过 5 个成员的 ensemble**——边际收益递减且 inference 延迟逼近 100ms
- **不做 weighted voting**（按 official 500 加权）——首轮等权，避免引入新超参

## 8.1 首轮 verdict

### 8.1.1 我们现在知道了什么

首轮结果已经足够回答 `034` 的核心第一问：

- `047` 最小 swap sanity 没发现强 slot 问题
- `034-A-control` 说明 deployment-time ensemble 框架基本可用
- 但 `034-A` 本体没有 lift，反而显著掉分

所以当前更稳的结论不是“ensemble 方向错了”，而是：

> **简单的等权概率平均不是旧版 asset set (`029B + 025b + 017`) 的可行组合策略。**

### 8.1.2 这意味着什么

这次失败更像是 **policy interference / calibration mismatch**，而不是实现 bug：

- 三个 strong checkpoint 的动作分布虽然都强，但并不代表它们的逐步概率可以直接安全平均
- 不同 policy 的偏好可能在关键状态下互相抵消，导致平均后的行为变钝
- `034-A-control` 之所以没明显崩，是因为 `3 × 029B` 不存在跨-policy 语义冲突

### 8.1.3 对后续 034 分支的影响

因此当前不建议直接继续启动旧矩阵里的：

- `034-B`
- `034-C`

因为在旧版 `034-A` 已经明显掉分的前提下，继续往旧矩阵里加更多异构成员大概率只会让读法更混。

更合理的 follow-up 应该是另开新 lane，例如：

- `031A + 036D`
- `031A + 036D + 029B`
- `top-2` 组合而不是 `top-3`
- confidence / entropy / state-dependent gating
- weighted averaging 或 winner-take-most

也就是说，当前 snapshot 更合适的下一步是：

> **把旧版 `034-A` 记为 legacy negative result，然后为 `031A / 036D / 029B` 的 frontier 组合单独开第二轮矩阵。**

### 8.1.4 已落实的 frontier follow-up 实现

第二轮 frontier follow-up 现在已经有可运行实现，不再停留在口头计划：

| 实现名 | 成员 | 目的 |
|---|---|---|
| `v034b_frontier_2way` | `031A@1040 + 036D@150` | 最小 frontier 组合，直接测试 v2 桶里最正交的双 checkpoint |
| `v034b_control_2x031A` | `2 × 031A@1040` | team-level handle sanity control |
| `v034c_frontier_3way` | `031A@1040 + 036D@150 + 029B@190` | 在 frontier 双组合上再加 `029B` 作为 per-agent anchor |
| `v034c_control_3x031A` | `3 × 031A@1040` | 三成员 team-level control |

对应代码：

- [ensemble_agent.py](../../cs8803drl/deployment/ensemble_agent.py)
- [v034b_frontier_2way](../../agents/v034b_frontier_2way/agent.py)
- [v034b_control_2x031A](../../agents/v034b_control_2x031A/agent.py)
- [v034c_frontier_3way](../../agents/v034c_frontier_3way/agent.py)
- [v034c_control_3x031A](../../agents/v034c_control_3x031A/agent.py)

当前已完成的最小 runtime 验证：

- `2×031A` vs random (`n=2`) = `1.000`
- `031A+036D` vs random (`n=2`) = `1.000`
- `3×031A` vs random (`n=2`) = `1.000`
- `031A+036D+029B` vs random (`n=2`) = `1.000`

这说明 frontier mixed ensemble 的 **team-level member 接入 / mixed probability averaging / evaluator 接口** 已全部打通，可以直接进入正式 official eval。

### 8.1.5 当前 frontier mixed verdict

截至 2026-04-19，这个 snapshot 的更稳总结已经不再是“legacy triad 失败”那么简单，而是：

1. **legacy triad (`029B + 025b + 017`) 已被否决**
   - 它是旧版 recipe 的 negative result
   - 不值得继续按这个配方扩成员

2. **frontier mixed (`031A + 036D (+029B)`) 已经翻正**
   - `034B-frontier = 0.872`
   - `034C-frontier = 0.890`
   - 并且都显著优于各自 control

3. **`031B` 是更好的 ensemble anchor**
   - `2 × 031B = 0.914`
   - `3 × 031B = 0.876`
   - 相比 `2 × 031A / 3 × 031A`，projection tax 明显更小

4. **`034E-frontier = 031B + 036D + 029B` 已成为当前 `034` 主候选**
   - `official 500 = 0.904`
   - `official 1000 = 0.890`
   - `H2H vs 031B = 0.596`
   - `H2H vs 029B = 0.590`
   - `H2H vs 034C = 0.544`

5. **`034C-frontier` 当前更像第一代 peer-play ensemble，而 `034E-frontier` 是更完整的第二代 frontier ensemble**
   - `official 1000 baseline = 0.843`
   - `034C` 的 H2H 对 `031A` / `029B` 分别达到 `0.580 / 0.624`
   - 但 `034E` 在更强的 `031B` anchor 上拿到了更稳的 baseline 与仍然为正的 peer 结果

因此当前最合理的方向不是退回“ensemble 整体失败”，而是：

> **承认 mixed frontier ensemble 已经成立，而且 `034E-frontier` 已经把这条线从“peer-play 很强的 rounded policy”推进到了“baseline 与 peer 两条轴都成立的 ensemble 主候选”。**

### 8.1.6 `034F-router` 已实现并完成首轮 eval：negative result

在确认 `034E-frontier = 031B + 036D + 029B` 已经成为当前 ensemble 主候选后，下一步最自然的问题不再是“再换谁进成员表”，而是：

> **什么时候更该信 `031B`，什么时候更该临时放大 `036D` / `029B`？**

为此，当前已补出一个最小可跑的 heuristic router 版本：

- [ensemble_agent.py](../../cs8803drl/deployment/ensemble_agent.py)
  - 新增 `HeuristicRoutingMixedEnsembleAgent`
  - 默认 `031B` 为 anchor，`036D` 为 specialist，`029B` 为 stabilizer
  - 根据 anchor entropy、anchor/specialist 的 relative confidence、以及两者的 JS disagreement，动态调整三者权重
- [v034f_router_031B_3way](../../agents/v034f_router_031B_3way/agent.py)
  - `031B@1220` `base_weight=0.50`
  - `036D@150` `base_weight=0.20`
  - `029B@190` `base_weight=0.30`

这条线当前的定位是：

- **不是 learned router**
- **不是新的训练线**
- 而是 `034E` 之上的最小 deploy-time 决策升级

当前已完成的首轮结果如下：

| experiment | result | read |
|---|---:|---|
| `034F-router` official `500` vs baseline | `0.884` (442W-58L) | 看起来尚可，但仍低于 `034E-frontier = 0.904` |
| `034F-router` official `500` vs random | `0.998` (499W-1L) | 推理链路正常 |
| `034F-router` official `1000` vs baseline | `0.862` (862W-138L) | 明显低于 `034E-frontier = 0.890`，且也低于 `034E-control = 0.869` |

因此这条线现在最稳的 verdict 是：

- heuristic router **没有**把 `034E` 往上推
- 而且 `1000 ep` 下出现明显回落，说明这不是“还差一点”的主候选
- `034F-router v0` 应降级为 **negative result**

也就是说，当前证据支持：

> **`034` 的下一跳如果还要重做 routing，应考虑更强的 learned / distilled router，而不是继续在 heuristic v1/v2 上小修小补。**

在此之前，`034` 主线保持为 `034E-frontier` 不变。

## 9. 执行清单

1. 实现 `cs8803drl/deployment/ensemble_agent.py` 通用框架
2. 实现 034-A 子类（3 个 per-agent policies）
3. 1-iter smoke：10 episodes vs baseline，确认推理通畅、action 分布合理
4. 实现 034-A-control（3 × 029B）做 sanity check
5. 起 official 500 vs baseline + vs random
6. 按 §4 判据 verdict
7. 决定 034-B / 034-C 是否启动
8. 如果 034-A vs baseline ≥ 0.88，**优先准备 submission**（这是当前最接近 9/10 的路径）

## 10. 相关

- [SNAPSHOT-029: per-agent SOTA](snapshot-029-post-025b-sota-extension.md)
- [SNAPSHOT-035: PPO stability backlog](snapshot-035-ppo-stability-backlog.md)
- HW3 ensemble dynamics 模型方法论

## 11. 034ea / 034eb 反向验证 — 架构 diversity > v2 桶 fingerprint (2026-04-19, append-only)

### 11.1 motivation

snapshot-051 §8.6 提出基于 v2 桶 orthogonality 的新 ensemble 候选：
- **034ea = {031B, 045A, 051A}** (3-way, 全 team-level, 各自 v2 桶 fingerprint 不同)
- **034eb = {031B, 036D, 045A, 051A}** (4-way, 加回 036D)

假设：v2 桶 fingerprint 不同 → policy 决策 diverse → ensemble averaging 应给 +2-3pp lift over avg member。

### 11.2 实测结果

| Ensemble | 成员 | 架构组成 | 平均 indiv 1000ep | Ensemble 1000ep | vs avg lift |
|---|---|---|---|---|---|
| **034E (baseline)** | {031B, 036D, 029B} | 1× team-level + **2× per-agent** | 0.863 | **0.890** | **+2.7pp** |
| **034ea** | {031B, 045A, 051A} | **3× team-level (全部)** | 0.879 | **0.878** | **-0.1pp anti-lift!** |
| **034eb** | {031B, 036D, 045A, 051A} | 3× team-level + 1× per-agent | 0.874 | **0.882** | +0.8pp |

### 11.3 假设被反驳 — 双重证据

**证据 1: 034ea anti-lift (-0.1pp 比成员均值还低)**
- 个体平均 0.879, ensemble 0.878 — 三个 team-level 模型 prob-average **没产生任何 ensemble 增益**
- 即使其个体平均 (0.879) 比 034E 个体平均 (0.863) 高 1.6pp, ensemble 数字 (0.878) 反而比 034E (0.890) 低 1.2pp

**证据 2: 045A vs 031A H2H = 0.492 (NS)**
- 045A 的 v2 桶 wasted_possession 55% (vs 031A 42%, +13pp diff)
- 但 H2H peer-axis 几乎 50%: blue 0.508 / orange 0.476
- → **v2 桶 fingerprint 不同 ≠ 实际 policy 决策不同**
- 045A 是 031A 的 "noisy clone", 在决策空间不构成新方向

### 11.4 关键 lesson — naive prob-averaging ensemble 的 lift 来源

**034E 0.890 lift 的真原因不是 v2 桶 fingerprint, 是架构 family diversity**:
- 031B: team-level + cross-attention (672 dim concat obs + 双 encoder)
- 036D: per-agent shared CC (336 dim ego obs + per-agent forward)
- 029B: per-agent shared CC (同 036D 架构 family)
- → 不同 obs encoding + 不同 forward path → **决策真正 diverse**

**034ea 失败的根因**: 三个 team-level (031B 是 cross-attention scratch v2, 045A 是 031A Siamese warmstart + 045 reward, 051A 是 031B cross-attention warmstart + 051 reward) 共享 obs encoding 路径 → policy 决策空间高度相关 → averaging 没增益。

**取巧空间已耗尽**: 想突破 034E 0.890 必须**真正引入新架构 family**, 不能继续在 team-level 内部找 v2 桶看似 orthogonal 的 candidates。

### 11.5 重要 meta-lesson — Ensemble ≠ 智力提升

**用户 2026-04-19 反复指出**: ensemble (034 系列) 是 stability + cost optimization, **不是真正的模型智力变化**。本节实证支持这个判断:
- ensemble lift 上限 ≈ 个体最强成员 + 1-3pp (来自架构 family 多样性)
- 没有 learned gating (MoE 至少有 router 学到 input-dependent specialization)
- prob-averaging = vote, 没新决策能力涌现
- 034 系列穷尽: 当前 architecture pool ({team-level, per-agent}) 的最优 ensemble = 034E 0.890

**项目 0.90 突破的真正路径必须是单 model 架构突破, 不是 ensemble 调优**:
- ✅ **052A (031C transformer)** in-flight on 011-18-0 — 可能引入第三个架构 family
- ❌ 找新 v2 桶 orthogonal 的 team-level candidate — 已实证无效
- ❌ 加更多 per-agent 成员 (017/025b/028A 都更弱, 加进 ensemble 拖低)

### 11.6 决定 — 034 系列 freeze, 不再加 lane

- **034ea / 034eb**: 已落地 verdict, 不再 launch H2H 或 capture (没价值)
- **034ec 候选** (051A 替 031B): 不 launch — 个体 +0.6pp marginal, 跟 034ea 失败的逻辑相同, ROI 极低
- **034F router** (snapshot-034 §5 提过的 learned router): 暂不实施 — 若要做也需等 052A 出来后, 用 cross-architecture pool
- 当前 SOTA 锁定: **single model 051A@130 = 0.888 / ensemble 034E = 0.890**, 034ea/eb 都退役

### 11.7 Raw recap

```
=== 034ea_frontier_031B_045A_051A vs baseline ===
win_rate=0.878 (878W-122L-0T)

=== 034eb_frontier_031B_036D_045A_051A vs baseline ===
win_rate=0.882 (882W-118L-0T)
```

完整 log: [034ea](../../docs/experiments/artifacts/official-evals/034ea_frontier_baseline1000.log) / [034eb](../../docs/experiments/artifacts/official-evals/034eb_frontier_baseline1000.log)
- [作业要求 — Performance 25+25 (+5)](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points)
