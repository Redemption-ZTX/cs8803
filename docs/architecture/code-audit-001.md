# 代码审计 001：隐藏约束与默认假设审查

> 审计索引见 [code-audit.md](code-audit.md)。上一版审计见 [code-audit-000](code-audit-000.md)（接手时模块逐项分析）。
> 诊断依据见 [snapshot-013 §11–§12](../experiments/snapshot-013-baseline-weakness-analysis.md) 和 [snapshot-017 §11](../experiments/snapshot-017-bc-to-mappo-bootstrap.md)。

- **日期**: 2026-04-15
- **负责人**:
- **审计类型**: 针对性审计（focused audit），不同于 000 的全模块线性审查
- **相比 000 的变更点**: 000 按模块列功能/问题/改进；本 001 按"**默认假设 → 实际代码行为 → 差距**"结构找**隐藏约束**

---

## 0. 为什么做这次审计

### 0.1 触发事件

截至 2026-04-15，项目跑完 9 条 RL lane（v1/v2/v4 PPO + MAPPO{no-shape, v1, v2} + Pool 018 + BC→MAPPO），发现一个**无法用 reward / architecture / warm-start 解释的跨 lane 不变量**：

> **`low_possession` 在所有 9 条 lane 上占失败比稳定在 22.6% - 28.1%**

见 [snapshot-013 §11](../experiments/snapshot-013-baseline-weakness-analysis.md#11-mappo-数据对-10-推论的反馈2026-04-15-后补) / [snapshot-017 §11](../experiments/snapshot-017-bc-to-mappo-bootstrap.md#11-failure-bucket-深度分析与判据-verdict)。

在追查根因的过程中，**意外发现一个长期被默认但从未验证的约束**：

> Actor (policy network) 的输入**只包含 own_obs**，完全不含 teammate 的 state。team 协调所需的队友位置/速度信息**从未进入 policy 的决策输入**。

这个约束**之前 0 个 snapshot、0 个 code-audit-000 章节明确点出**。之所以没被发现，是因为：

1. Soccer-Twos 是 2v2 coordination 任务，直觉上"policy 当然要知道队友在哪才能配合"
2. MAPPO 架构加入了 centralized critic，critic 看得到 teammate_obs + teammate_action，**但 actor 不看**
3. 没人在代码审计里问过"actor 到底吃什么"这种默认成立的问题

这是一个**经典的默认假设型 bug**——它不是代码 bug，是**研究设计假设与代码实现的断层**。

### 0.2 审计目标

**找出所有类似的"默认成立但未验证"假设**，避免新一轮实验（[snapshot-021 obs expansion](../experiments/snapshot-021-actor-teammate-obs-expansion.md) / [snapshot-022 role-differentiated shaping](../experiments/snapshot-022-role-differentiated-shaping.md)）建立在错误前提上。

具体要回答的问题：

1. Obs 层：actor 还有什么信号在"我们以为有但实际没有"
2. Reward 层：reward 的对称性、事件定义、信号漏斗
3. Action 层：action space 转换是否有信息损失
4. Identity 层：agent_id 与物理角色的绑定是否稳定
5. Episode 层：时间、终止条件的可见性
6. Train vs Eval 一致性：训练环境和推理环境是否真的一样

---

## 1. 审计方法论

### 1.1 审计范围

**纳入审计的文件**（按优先级）：

| 优先级 | 文件 | 审计重点 |
|---|---|---|
| P0 | [cs8803drl/core/utils.py](../../cs8803drl/core/utils.py) | env factory, RewardShapingWrapper, opponent_mix |
| P0 | [cs8803drl/core/soccer_info.py](../../cs8803drl/core/soccer_info.py) | info 解析，shaping 纯逻辑 |
| P0 | [cs8803drl/branches/shared_central_critic.py](../../cs8803drl/branches/shared_central_critic.py) | MAPPO CC 模型 + obs_fn 实现 |
| P1 | [cs8803drl/training/train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py) | MAPPO 主训练入口 |
| P1 | [cs8803drl/training/train_ray_mappo_vs_opponent_pool.py](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py) | pool 变体 + frozen dispatch |
| P1 | [cs8803drl/training/train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) | PPO shaping (single_player) lane |
| P2 | [cs8803drl/deployment/trained_shared_cc_agent.py](../../cs8803drl/deployment/trained_shared_cc_agent.py) | 推理 wrapper |
| P2 | [cs8803drl/evaluation/evaluate_matches.py](../../cs8803drl/evaluation/evaluate_matches.py) | eval 流程 |
| P2 | [cs8803drl/branches/role_specialization.py](../../cs8803drl/branches/role_specialization.py) | 已有的 role 机制（和 snapshot-022 相关）|

**排除在本次审计外的文件**：
- `train_ray_curriculum.py` / `train_ray_selfplay.py` — 当前 main line 不用
- 其他 deployment wrapper — 用法和 shared_cc_agent 类似

### 1.2 审计方法

采用 **"assumption-driven"** 结构化审计：

1. **列出可能的默认假设**（按 obs/reward/action/identity/episode/train-eval 六类）
2. 对每条假设，**读对应代码段定位其真实行为**
3. 对比"假设"与"实际"，记录偏差
4. 对每条偏差，判断 severity（🔴 可能解释已观察到的失败模式 / 🟡 潜在未来风险 / ⚪ 无害但值得记录）
5. 提出**验证方法**（能跑 smoke 就给 smoke；需要额外数据就说清要看什么）

### 1.3 审计工具

本次审计通过**委派 Explore agent 做结构化问答**完成，主 agent 给出假设清单 → 子 agent 逐条按 file:line 查实际代码行为。人工 review 后把判断分类为 🔴/🟡/⚪。

审计输出**不修改任何代码**，仅产出文档。

---

## 2. 审计结果（按主题分组）

### 2.1 观测层（obs）

#### 2.1.1 🔴 Actor 只看 own_obs（已知但未正式落档）

**默认假设**：Actor 看得到队友 state。

**实际代码行为**：
- [shared_central_critic.py:186-192](../../cs8803drl/branches/shared_central_critic.py#L186-L192) — actor 用 `input_dict["obs"]["own_obs"]`，**不包含** teammate_obs / teammate_action
- [shared_central_critic.py:187-195](../../cs8803drl/branches/shared_central_critic.py#L187-L195) — value_model 才用 `input_dict["obs_flat"]`（完整的 Dict 被 flatten）

**对 v1/v2/v4 PPO lane**（single_player=True）：
- 同一个 shared policy 被 env 对 team0 两个 agent 分别 query
- 每次 query 只传 own_obs（从 [utils.py:316 起的 create_rllib_env](../../cs8803drl/core/utils.py#L316)）
- 结果：PPO lane 的 actor 也只看 own_obs

**前因后果**：
- 前因：MAPPO 框架的常见实现就是 **Centralized Training with Decentralized Execution (CTDE)**——critic 能看全局，actor 只看本地。这是正确的 CTDE 范式
- 后果：但"只看本地"的约束**从未在项目内部讨论文档里出现**，导致后续 9 条 lane 的实验设计都**隐式假设 teammate coord 是可学的**，实际上 actor 根本没机会学
- 对 `low_possession` 的影响：`low_possession` state 下，actor 需要判断"是我去抢还是队友去抢"，但它**看不到队友在哪**，无法做这种判断 → 两个 agent 按相同 obs 做相同决策 → positional collapse

**验证方法**：已验证。代码行号如上。

**严重程度**：🔴 **已确认是 `low_possession` 跨 lane 不变量的主要候选根因 A**。[snapshot-021](../experiments/snapshot-021-actor-teammate-obs-expansion.md) 将直接测试该根因。

---

#### 2.1.2 🟡 Policy 对 episode time 完全盲

**默认假设**：Policy 可以通过 obs 感知到"episode 过了多久"或"还剩多少时间"。这是足球类任务的常识假设。

**实际代码行为**：
- Obs 结构里**没有** episode step count、remaining time、elapsed fraction 或任何时间类信号
- [utils.py:172](../../cs8803drl/core/utils.py#L172) 和 [utils.py:232](../../cs8803drl/core/utils.py#L232) 内部跟踪 `_episode_steps`，但**只用于计算 terminal shaping**（`fast_loss_penalty`）
- `_episode_steps` **从未被传进 policy 的 obs**

**前因后果**：
- 前因：soccer_twos 原生 env 的 obs 设计里没有时间信号，前任/上游代码全盘继承，没人补
- 后果：policy 无法学"比赛后段保守 / 前段激进"之类的时间相关策略
- 对 `late_defensive_collapse` 的影响：**这个桶占失败 46-52%，被命名为"后段崩盘"——但 policy 压根不知道什么是"后段"**。每一步对它都是 Markov state，没有时间上下文。它**不知道现在是比赛第 1 步还是第 40 步**。这可能是 late_collapse 无法被 shaping / CC / BC 任何手段完全修复的**另一条结构性原因**

**验证方法**：
- 打印 eval 时 obs shape 和字段，确认无时间维度
- 读 env.observation_space 确认维度 = 336（无时间字段）

**严重程度**：🟡 **在 022 之后可能成为一条独立的 obs-expansion 候选**（+1 维 `episode_step / max_steps`，和 snapshot-021 的 teammate obs 正交，可叠加到同一 obs 扩展实验里）。

---

#### 2.1.3 ⚪ Train 时 team0 / team1 的 obs 结构不对称（有意设计）

**默认假设**：所有 agent 在训练时看到同种结构的 obs。

**实际代码行为**：
- [shared_central_critic.py:110-124](../../cs8803drl/branches/shared_central_critic.py#L110-L124) 的 `shared_cc_observer`：
  - Team0 agent（0, 1）拿到 **Dict obs** = `{own_obs, teammate_obs, teammate_action}`
  - Team1 agent（2, 3）拿到 **原始 flat obs**
- [shared_central_critic.py:127-137](../../cs8803drl/branches/shared_central_critic.py#L127-L137) 的 `shared_cc_observer_all`：4 个 agent 都拿 Dict obs，**只在 opponent_pool lane 用**

**前因后果**：
- 前因：team1 是 frozen baseline（[FrozenBaselinePolicy](../../cs8803drl/branches/role_specialization.py#L42-L103)），baseline 不是我们训的，它不认识 Dict obs → 给它 raw flat obs 对齐它原始接口
- 后果：**没有** bug，但 lane 之间的 obs 契约不同：
  - [train_ray_mappo_vs_baseline.py:277](../../cs8803drl/training/train_ray_mappo_vs_baseline.py#L277) 用 `shared_cc_observer`（不对称）
  - [train_ray_mappo_vs_opponent_pool.py:383](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py#L383) 用 `shared_cc_observer_all`（全对称），配合 `strip_obs_tail_dims` 让 baseline 能吃 CC obs 后再 strip
- 这个差异不影响训练正确性，但**如果未来加新 lane**，作者需要明确选哪种 observer_fn

**验证方法**：代码已读清楚。

**严重程度**：⚪ 设计合理，文档空白需要补齐。

---

#### 2.1.4 ⚪ possession_bonus 是 proximity-based 不是 touch-based

**默认假设**：possession = 实际持球。

**实际代码行为**：
- [soccer_info.py:261-277](../../cs8803drl/core/soccer_info.py#L261-L277)：任何 agent 距球 `<= 1.25` 单位就拿 possession_bonus
- 不区分哪个 agent 实际碰过球、不区分 agent 朝向

**前因后果**：
- 前因：soccer_twos info 里没暴露"触球事件"字段，只能用距离近似
- 后果：两个 team0 agent 同时在 1.25m 内**同时拿 bonus**，encouraging 扎堆抢球
- 这**不一定**是 bug——对 sparse reward 任务，鼓励靠近球本身合理。但是它**不提供角色分工信号**

**验证方法**：代码已读清楚。snapshot-013 / 010 已经讨论过这个。

**严重程度**：⚪ 已知设计选择，不是隐藏约束。

---

### 2.2 奖励层（reward）

#### 2.2.1 🟡 默认 shaping 对两个 team0 agent **完全对称**

**默认假设**：shaping reward 对两个 team0 agent 对称（"这是 team-level 信号"）。

**实际代码行为**：
- [soccer_info.py:186-282](../../cs8803drl/core/soccer_info.py#L186-L282)：
  - `ball_progress_scale`：team0 两个 agent 都加 `+scale·dx`
  - `possession_bonus`：任一 agent 在 1.25m 内都加 bonus（独立）
  - `time_penalty`：所有 agent 每步都扣
  - `opponent_progress_penalty`：defending 方全队都扣
  - `deep_zone_penalty`：ball 进深区时本方全队都扣
- **但同时**：这三项都支持 `*_by_agent` dict 形式的**per-agent 覆盖**（[soccer_info.py:194-196](../../cs8803drl/core/soccer_info.py#L194-L196)），允许 agent 0 和 agent 1 收到**不同**的 scale

**前因后果**：
- 前因：现代 shaping 设计通常保留 per-agent override 作为 role-specialization 接口。我们的代码继承了这个灵活性但**从未实际用过**
- 后果：9 条已跑的 lane 全部使用"对称默认"配置，两个 agent 收到完全一样的 reward signal → 学到同构 policy → positional collapse → 解释部分 `low_possession`
- 对 [snapshot-022](../experiments/snapshot-022-role-differentiated-shaping.md) 的影响：022 **就是专门用这套 per-agent override 做 role-diff shaping**。infrastructure 已就位，不需要改核心代码

**验证方法**：snapshot-022 跑完后看 low_poss 是否下降 ≤ 15%。

**严重程度**：🟡 **是 `low_possession` 跨 lane 不变量的候选根因 B，snapshot-022 直接测**。

---

#### 2.2.2 ⚪ `aggregate_scalar_shaping` 对两 agent 相加（不是平均）

**默认假设**：team reward 是两 agent reward 的某种聚合。

**实际代码行为**：
- [soccer_info.py:313-317](../../cs8803drl/core/soccer_info.py#L313-L317)：`aggregate_scalar_shaping` 是**求和**，不是平均
- 这意味着：如果 per-agent shaping 都是 +0.01，team-level 拿到的是 +0.02

**前因后果**：
- 前因：sum 语义让 "double coverage" 自然加倍（两个 agent 都近球 → 两倍 bonus）
- 后果：per-agent override 的 shaping 设计要注意：如果 agent 0 给 0.02，agent 1 给 0.01，team 拿到 0.03——不是 max，也不是 mean
- 对 snapshot-022 的影响：role-diff shaping 的 magnitude 要按这个 sum 语义预留

**验证方法**：1-iter smoke 里打印 `team_reward_sum` 确认等于 per-agent sum。

**严重程度**：⚪ 已知语义，要记住但不是 bug。

---

#### 2.2.3 ⚪ env 原生 reward 在 shaping 开关之外

**默认假设**：我们 "关 shaping" = 关所有外部 reward signal。

**实际代码行为**：
- Env 原生还发 sparse ±3 goal reward（进球/失球时）——**这个 reward 我们从未关过也无法关**
- 所谓"关 shaping" 只关了我们加的 4-5 项 shaping，**不关 env 本身的进球奖励**

**前因后果**：
- 前因：sparse goal reward 是任务定义本身，不是我们加的
- 后果："MAPPO no-shape" lane 不是"完全无 reward signal"，是"只有 sparse goal reward"——这改变了"no-shape 还是 22-26% low_poss"的解读：不是"完全无 reward 信号也有 low_poss"，而是"只有对称的 sparse reward 也有 low_poss"，这**不否决 reward 对称性假设 (根因 B)**

**验证方法**：直接读 [utils.py:316](../../cs8803drl/core/utils.py#L316) 的 `create_rllib_env`，`use_reward_shaping=False` 时**完全不 wrap** `RewardShapingWrapper`——env 只剩原生 reward。

**严重程度**：⚪ 不影响现状，但 [snapshot-013 §12](../experiments/snapshot-013-baseline-weakness-analysis.md#122-113-根因诊断被更新) 之前的"no-shape 也有 low_poss → reward 不是主因"推理需要**被修正**：应该改为"even under sparse-only reward, symmetry persists and low_poss 继续出现"——**仍然支持根因 B**。

---

### 2.3 Action 层

#### 2.3.1 ⚪ ActionFlattener MultiDiscrete([3,3,3]) → Discrete(27) 是 bijective

**默认假设**：ActionFlattener 无损。

**实际代码行为**：
- [shared_central_critic.py:206-212](../../cs8803drl/branches/shared_central_critic.py#L206-L212) 用 `ActionFlattener` 或 `itertools.product` 枚举全部 27 组合 → 双射
- 逆映射 `lookup_action()` 把 Discrete 0-26 还原到 MultiDiscrete

**前因后果**：
- 前因：RLlib PPO 需要 Discrete action space，env 给的是 MultiDiscrete
- 后果：无损转换，但 **27 个 action 里可能有明显劣的组合**（例如"同时向前、向后、横向"这种自相矛盾指令）——policy 需要自己把它们 down-weight
- 目前没有 action masking，policy capacity 的一部分花在学"不选废 action"上

**验证方法**：已读。

**严重程度**：⚪ 已知设计。

---

### 2.4 Identity 层（agent_id vs 物理角色）

#### 2.4.1 🔴 Agent_id 与 spawn position 的绑定稳定性**从未被验证**

**默认假设**：`agent_id 0` 永远是同一个物理 agent，每局从同一个位置开球。[role_specialization.py](../../cs8803drl/branches/role_specialization.py) 和 [snapshot-022](../experiments/snapshot-022-role-differentiated-shaping.md) 都依赖这个假设。

**实际代码行为**：
- [shared_central_critic.py:68-80](../../cs8803drl/branches/shared_central_critic.py#L68-L80) 定义 `TEAM0_AGENT_IDS = (0, 1)`，但**这是我们的约定**，不是 env 保证
- **没有任何代码在 env reset 时检查 agent_id 0 的实际 spawn 位置是否稳定**
- env 变量 `CC_TEAM0_AGENT_IDS` 允许 runtime 覆盖这个 tuple（[shared_central_critic.py:48-55](../../cs8803drl/branches/shared_central_critic.py#L48-L55)）——如果 train 和 eval 用了不同值，teammate_id 映射**直接 break**

**前因后果**：
- 前因：soccer_twos env 的内部行为没有显式文档说明 agent_id 0 是否总在同一物理位置 spawn
- 后果（如果 spawn 实际不稳定）：
  - [role_specialization.py:147-159](../../cs8803drl/branches/role_specialization.py#L147-L159) 的 `attacker_id=0, defender_id=1` 变成**随机标签**——两个 agent 角色每局乱换
  - [snapshot-022](../experiments/snapshot-022-role-differentiated-shaping.md) 的 role-diff shaping 失去意义：agent 0 这局是 striker 位置（吃 striker shaping），下局是 goalie 位置（仍然吃 striker shaping，但物理角色不匹配）
  - 整个实验等于"把两个 agent 的 shaping 在 episode 间随机打散"，**完全无法学到 role specialization**

**验证方法**（**必做，snapshot-022 开训前**）：
- 写 5 行 Python 脚本
- 创建一个 env
- 连续 reset 20 次
- 每次 reset 后立刻读 env 的 agent 位置（从 info 或 obs）
- 打印 agent_id 0 的初始 (x, y)
- 若 20 次都一样 → spawn 稳定，snapshot-022 前提成立
- 若位置每 episode 随机 → **必须改 role-diff 代码**把 role 绑定到 **spawn 位置 x 的符号** 而不是 agent_id

**验证结果（2026-04-15 P0 完成）**：

spawn **位置本身稳定**（20/20 一致），但与"严格 striker/defender role"的映射**不干净**。完整 spawn 数据：

| 队 | agent_id | 初始位置 |
|---|---|---|
| Blue (team0) | 0 | `(-9.03, +1.20)` |
| Blue (team0) | 1 | `(-6.24, -1.20)` |
| Orange (team1) | 0 | `(6.45, +1.20)` |
| Orange (team1) | 1 | `(6.66, -1.20)` |

分三层：
1. **API 本地 agent_id 重映射** ✅ 对称（两队模块都看到本地 `{0, 1}`）
2. **上下路 (y 轴)** ✅ 对称（两队 agent_id 0 = +y, agent_id 1 = -y）
3. **前锋/后卫 (x 轴深度)** ❌ **非对称**：
   - Blue: agent 0 (x=-9.03) **更深** vs agent 1 (x=-6.24) **更前**——按足球直觉 agent 0 像 defender, agent 1 像 striker
   - Orange: agent 0/1 几乎同深（6.45 vs 6.66）——无 spawn-level role 先验
   - 本 snapshot-022 配置把"进攻 pole" 给 agent 0，对 blue 方反而是"深度位置 ↔ 进攻 reward"的倒置

**对 snapshot-022 的修订**：
- 标题和内部叙述从 "Role-Differentiated Shaping (striker/defender)" 收紧为 "**Agent-ID Asymmetric Shaping**"
- 测试的本质问题是 "**打破 agent_id 对称性本身是否让 shared policy 学出分化 behavior**"，而非"严格 striker/defender role 是否有效"
- §6.3 行为判据加"必须和 scratch MAPPO+v2 的 |Δx| 基线对比"——防止 spawn 天然偏移（blue 方 2.8 单位）被误认为是 reward 分化的效果
- 若 022 过，下一轮可做"按 spawn x 符号 runtime 绑定 role"的严格版

**严重程度**：🟡（**从 🔴 降级**）——spawn 稳定，022 可以以收紧 claim 的方式执行。但 strict role-mapping 仍是未解问题，留待后续实验。

---

#### 2.4.2 🟡 Eval 从未测试 team0 ↔ team1 side swap

**默认假设**：我方 policy 在"team0 还是 team1"都能工作。

**实际代码行为**：
- [evaluate_matches.py:74-91](../../cs8803drl/evaluation/evaluate_matches.py#L74-L91) 实现了 side remapping 逻辑（`team0_ids=(2,3)` 可以用）——**但默认关闭、从未启用**
- 所有训练 + eval 把"我方 = team0 (0,1)"写死

**前因后果**：
- 前因：train-eval 在一侧保持一致最简单，没人改
- 后果：如果 env 存在 left-right 不对称（例如 field geometry 或 goal position 的 obs 编码本身不对称），我们的 policy **可能 side-specialized** — 学到一套 left-side playbook，右侧会变弱
- 对 WR 数字的直接影响：**没有**，因为 train 和 eval 都固定一侧
- 对 **report 诚实性**的影响：我们**从来不知道** policy 真实的 side-agnostic 能力

**验证方法**：
- 取 BC @2100 (current SOTA)
- 用 `team0_ids=(2,3)` 跑 200-ep eval（即 policy 放在 team1 位置）
- 对比默认 500-ep 的 0.842
- 如果对半分或轻微下降（<5%）→ side-agnostic，没事
- 如果大幅回退（>10%）→ side 严重 specialized，report 里要诚实写

**严重程度**：🟡 **不影响现状 WR 数字，但影响 report analysis 的完整性**。

---

#### 2.4.3 ⚪ FrozenBaselinePolicy 的 obs 契约

**默认假设**：Frozen baseline 吃的 obs 和训练时我方 agent 吃的一样。

**实际代码行为**：
- 在 `train_ray_mappo_vs_baseline` lane：baseline 吃 raw flat obs（通过 `shared_cc_observer` 对 team1 不加 Dict 包装）
- 在 `train_ray_mappo_vs_opponent_pool` lane：baseline 吃 cc_obs_space，通过 `strip_obs_tail_dims` ([train_ray_mappo_vs_opponent_pool.py:371-376](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py#L371-L376)) **剥掉尾部维度**后传给 baseline
- 两种契约都 work，但不一致

**前因后果**：
- 前因：两个 lane 的 observation_fn 选择不同（§2.1.3）
- 后果：baseline 的决策行为**应该等价**（它不在乎多余尾部维度，因为它不认识），但实际需要**1-iter smoke 验证** baseline 在两种 lane 下返回相同 action distribution

**验证方法**：1-iter smoke 跑两个 lane，对同一 obs 打印 baseline action → 应相同。

**严重程度**：⚪ 大概率无害，但未验证。

---

### 2.5 Episode 层

#### 2.5.1 🟡 Episode 长度和终止条件**对 policy 不可见**

见 §2.1.2（policy 时间盲）——episode 层和 obs 层是同一问题的两面。

**补充**：
- [soccer_info.py:160-179](../../cs8803drl/core/soccer_info.py#L160-L179) 的 `extract_winner_from_info` 只在 episode 结束时使用，policy **从不**收到"即将结束"的信号
- Terminal shaping (`fast_loss_penalty`) 只在最后一 step 一次性施加，前面 step 的 policy 决策无法考虑剩余时间

---

#### 2.5.2 ⚪ `rewards` 在两个 team0 agent 间共享基础结构（sparse goal 相同）

**默认假设**：两个 teammate 的 sparse reward 等价。

**实际代码行为**：
- Env 原生的进球奖励：team0 所有 agent 同时 +3（或 -3）
- 终局 shaping (`per_agent_bonus` in [soccer_info.py:351-360](../../cs8803drl/core/soccer_info.py#L351-L360))：两个 agent 平分

**前因后果**：
- 前因：team-level reward 默认对所有 team member 对称发放，这是 soccer_twos 的设计
- 后果：即使我们做 per-agent shaping overrides，**sparse goal 这一项仍然对称**——任何 role-diff shaping 只能通过 shaping 层制造不对称，底层 sparse signal 永远对称

**验证方法**：读 [utils.py:280-290](../../cs8803drl/core/utils.py#L280-L290) 的 `_broadcast_reward`。

**严重程度**：⚪ 已知设计。

---

### 2.6 Train vs Eval 一致性

#### 2.6.1 🟡 训练时 FillInTeammateActions callback 让 critic 看到 teammate action；推理时 teammate_action 硬编码 0

**默认假设**：critic 用的信息在推理时也可获得。

**实际代码行为**：
- Train：[shared_central_critic.py:140-176](../../cs8803drl/branches/shared_central_critic.py#L140-L176) 的 `FillInTeammateActions` callback 在 rollout 收集后**回填** teammate 上一步的 action（one-hot 到 obs 尾部 27 维）
- Inference：[trained_shared_cc_agent.py:208](../../cs8803drl/deployment/trained_shared_cc_agent.py#L208) 硬编码 `teammate_action = 0`

**前因后果**：
- 前因：teammate action 在 inference 时**不可得**（你不知道队友下一步会做什么）。这是 CTDE 的标准限制
- 后果：乍看是 train-eval mismatch，但**实际上 actor 在训练和推理都不用 teammate_action**（actor 只用 own_obs）
  - Train 时：teammate_action 只进 critic 的 value estimation，用来算 advantage → 影响 policy gradient 的数值
  - Inference 时：只 call actor，critic 完全不用 → `teammate_action=0` 这个硬编码**被无视**
- 所以这个 mismatch **对 inference 行为无影响**，但**会影响训练的 advantage 估计质量**——critic 学到的 value 函数依赖了一个 inference 时不存在的 causal signal，可能导致 advantage biased

**验证方法**：难验证，需要 careful training signal analysis。影响量级大概率小。

**严重程度**：🟡 技术上 train-eval 信号不对齐，但**不影响 inference 行为**，只影响 training gradient 质量。不作为当前瓶颈的主要候选。

---

#### 2.6.2 ⚪ Shaping 在 eval 是否生效

**默认假设**：eval 时 shaping 和 train 一致（这样 policy 看到的 reward signal 一样）。

**实际代码行为**：
- Eval 通过 [evaluate_matches.py](../../cs8803drl/evaluation/evaluate_matches.py) 调用 env，env 构造仍然走 `create_rllib_env`
- 若传的 env_config 里有 `reward_shaping` → shaping 生效；没有 → 不生效
- Eval 里 reward 本身**不影响 agent 决策**（agent 只用 obs 算 action），所以这个 toggle 实际**只影响 eval 时打印/记录的 reward 数字**

**前因后果**：
- 前因：evaluate_matches 的 code path 已经通过 env_config 传参，默认跟训练配置一致
- 后果：无功能性 bug，但 eval log 里看到的 reward 数字需要注意"这是真实胜负信号 vs shaping 叠加的混合"

**严重程度**：⚪ 无害但要记得。

---

#### 2.6.3 🔴 Official API 推理时无 `info` 访问

**默认假设**：只要 agent 在本地持有 env 引用，deployment wrapper 就可以在评测时直接读 `info` / 真实 teammate state。

**实际代码行为**：
- 官方 `AgentInterface.act()` 只接收 `observation: Dict[int, np.ndarray]`
- 官方 evaluator 主循环只调用：
  - `team_order[0].act({0: obs[0], 1: obs[1]})`
  - `team_order[1].act({0: obs[2], 1: obs[3]})`
- `load_agent()` 会在构造 agent 后立即 `env.close()`，因此 agent 不能靠持久 env 引用在推理时回查真实 state

**前因后果**：
- 前因：official submission API 的设计就是“agent 只根据 observation 决策”
- 后果：
  - 任何训练时依赖 env `info` 真值的 obs 扩展，默认都**不可直接提交**
  - [snapshot-021](../experiments/snapshot-021-actor-teammate-obs-expansion.md) 首轮踩到的正是这类坑：训练用真实 teammate state，deployment 只能用 decoder 补 → train/eval 语义严重分裂
  - 之后所有新 lane 在预注册时都必须先回答：“我用的信号在 official API 下可得吗？”

**验证方法**：
- 已直接核对官方 `soccer_twos` 的 `AgentInterface` 与 `evaluate.py`
- 这是接口级硬约束，不依赖具体 checkpoint 或 seed

**严重程度**：🔴 **submission 级别硬约束**。它不只影响 snapshot-021，也影响之后所有试图把 env `info` 直接塞进 policy 输入的实验设计。

---

#### 2.6.4 🟡 Train 用 env `info` 真值，eval 用 lossy decoder——真实风险，但不是当前首轮失败的唯一主因

**默认假设**：obs 扩展（如 teammate state）在训练和推理时使用**等价信号**。即使具体来源不同（训练读 `info` / 推理读 decoder），数值分布应该接近。

**实际代码行为**（snapshot-021 首轮原型）：

- **训练时**：[utils.py 的 TeammateStateObsWrapper](../../cs8803drl/core/utils.py#L467-L500) 在 env `info` 提供时，把**真实** teammate position/velocity 拼到 actor obs 尾部
  - 数据来源：`info[agent_id]["player_info"]["position"]` + `["velocity"]`
  - 数值范围：position ~ ±15 单位，velocity ~ ±5 单位，**精确**
- **推理时**（official-style deployment）：[trained_shared_cc_teammate_obs_agent.py](../../cs8803drl/deployment/trained_shared_cc_teammate_obs_agent.py) 调用 [`fit_observation_state_decoder` + `decode_player_state_from_observation`](../../cs8803drl/core/obs_teammate.py#L124-L182)，从 raw rays **线性回归**估计 teammate state
  - 数据来源：raw observation 336 维通过 lstsq 拟合的 (336+1) → 4 矩阵
  - 实测 decoder 误差（snapshot-021 首轮，用 256 sample 拟合后离线评估）：
    - `mean_abs_err_per_dim = [15.85, 9.36, 20.18, 15.10]`
    - `mean_l2_err = 35.71`
    - `median_l2_err = 32.30`
    - `p90_l2_err = 61.73`

**误差量级判读**：teammate position 真实范围约 ±15 单位（field 半径），velocity 约 ±5。decoder mean abs err **15.85 = field radius 量级**——**预测值与真值不相关**。线性投影从 ray-based obs 抽 teammate xyz 在数学上不可行（rays 编码距离/碰撞类型，不是绝对坐标的线性函数）。

**前因后果**：
- 前因：snapshot-021 设计时**没意识到** §2.6.3 的 official API 约束，把"训练用 info、推理 wrapper 自己想办法补"当成可行方案
- 后果：actor 在训练时学到 condition on **真实 teammate state**，到 inference 时被喂 **L2=35 的噪声**，等效于让 policy 在最关键的 5 维上看 adversarial garbage
- 训练与推理的输入分布**统计上不重叠**——网络参数在 train 分布上得到的 mapping 在 eval 分布上几乎全错
- 结果：snapshot-021 首轮 internal eval baseline WR = **0.18**, vs random 出现 24/50 ties——**功能性异常**，policy 行为退化到随机/僵化
- 这个 bug **不是模型 bug 也不是 RL bug**，是一个**训练-推理表征契约破裂** bug，比 §2.6.1（teammate_action 硬编码 0）严重得多——后者 inference 不调用 critic 所以无害；这里 actor 的 forward 路径**真的吃**了 decoder 的噪声

**验证方法**（已完成）：
- 离线对 500 个 (raw_obs, real_teammate_state) 样本跑 decoder，按维度计算 abs error → 上述数字
- 训练 run 的 internal eval WR 长期低于 0.20，与 decoder 噪声严重相符

**2026-04-16 再修正**：
- `021b` 的 local true-info 诊断只有 `0.135 @ 200 ep`，说明即使把评测改回真实 teammate state，策略也没有恢复
- 因此 decoder 分裂**不是当前首轮失败的唯一主因**
- 它仍然是一个真实且严重的 engineering risk，但更准确的定位应该是：**存在的 train-eval 风险，被首轮异常暴露出来，但主训练失败还需要别的解释**

**严重程度**：🟡 **存在、且未来必须避免，但当前更像次因而非唯一主因**。任何"训练用真值、推理用代理信号"的 obs 扩展设计仍然需要先证明代理信号的保真度。

**修复路径**（详见 [snapshot-021 §4.2](../experiments/snapshot-021-actor-teammate-obs-expansion.md) 的 021c-A/B/C）：

| 修复路径 | 描述 | 适用性 |
|---|---|---|
| (1) 训练也用 decoder | train 与 eval 都跑同一 decoder → 一致但保真度仍差 | low_poss 假设没机会被检验 |
| (2) 训出更好的 decoder | 非线性 / 多步 ray 历史，让 L2 误差降到 < 5 | 工程量大，回报不确定 |
| (3) Auxiliary prediction head | 训练时不扩 obs，加辅助头从 raw obs 预测 teammate state；推理时 actor 主干已内化感知 | **推荐**，符合 official API 约束（snapshot-021 §4.2 021c-B）|
| (4) 不扩 obs，改模型结构 | rays 已含 teammate 信息，让 actor 自己学 | 不直接测 A 假设 |

**对 future lane 的预防性约束**：
- 任何引入 obs 扩展或 critic-only 信号的 lane，**预注册时必须答**：
  1. 这个信号在 official `act(obs)` 时**字面可得吗**？
  2. 如果不可得，推理时用什么代理？代理的保真度**已离线测过**吗？
- 把这两条作为 [engineering-standards.md 实验迭代 § 检查表](../architecture/engineering-standards.md#实验迭代) 的预注册必填项

#### 2.6.5 🔴 Obs 扩展字段未归一化，first-layer activation 被高量级维度主导

**默认假设**：只要 teammate state 真值被正确拼到 actor obs 末尾，网络就能自己学会处理不同量纲，不会因为新增 4-5 个 field-unit 特征而功能性退化。

**实际代码行为**（snapshot-021 首轮原型）：

- [obs_teammate.py](../../cs8803drl/core/obs_teammate.py) 的 `extract_own_player_state()` 直接返回 field 单位的 `x, y, vx, vy`
- [utils.py 的 TeammateStateObsWrapper](../../cs8803drl/core/utils.py#L467-L520) 再把这 4 维直接拼到原始 `own_obs` 末尾
- `normalized time` 虽然在 `[0,1]`，但 `max_steps=1500` 时前几十步只有 `0.0007 ~ 0.03`，量级远小于其他 tail 字段

**Smoke 证据**（2026-04-16）：

- 原始 `336` 维 `own_obs` 抽样全部落在 `[0,1]`
  - `global min/max = 0.0 / 1.0`
  - `frac_abs_gt_1 = 0.0`
  - `frac_abs_gt_5 = 0.0`
  - `frac_abs_gt_10 = 0.0`
- 第一次 `step()` 时，wrapper 打印出的真实 teammate tail 典型量级：
  - `x ≈ ±8~15`
  - `y ≈ ±7`
  - `vx/vy ≈ ±8`
  - `time ≈ 0.00067`

**前因后果**：
- 前因：snapshot-021 首轮只验证了“能不能把 teammate 真值塞进去”，没有验证“塞进去之后量纲是否和原始 obs 匹配”
- 后果：新增 tail 的 4 个位置/速度维度在第一层前向时会天然拥有远大于其余 336 维的数值幅度，导致：
  - 第一层 activation 更容易被高量级 tail 主导
  - 原始 rays / ball / opponent / goal 相关信号在 scratch PPO 的早期训练里被压制
  - 训练看起来像“拿到了更多信息却更不会踢球”

**为什么这是当前最强主因假设**：
- §2.6.4 的 decoder 分裂只能解释“official-style eval 为什么坏”
- 但 [snapshot-021](../experiments/snapshot-021-actor-teammate-obs-expansion.md) 的 `021b` 本地 true-info 诊断也只有 `0.135 @ 200 ep`
- 这说明：即使评测也直接使用真实 teammate state，策略仍然非常弱
- 因此，**训练期自身就存在更强的问题**；当前证据最支持的就是“unnormalized tail dominance”

**修复方向**：
- `021b` 重跑时，为 teammate tail 增加固定 field-scale 归一化
  - 初始保守建议：`scale = [15.0, 7.0, 8.0, 8.0]`
- time 维先保持原逻辑，必要时再单独调大其有效量级
- `021c` 的 auxiliary path 不直接扩 obs，因此天然绕开这一类 first-layer dominance 问题

**严重程度**：🔴 **当前 snapshot-021 首轮最强主因假设**。在 future lanes 里，任何 obs 扩展除了检查“能不能拿到信号”，还必须检查“新增信号与原始 obs 的量纲是否匹配”。

---

## 3. 风险严重程度总表

| Finding | 类别 | Severity | 对当前实验的影响 |
|---|---|---|---|
| Actor only own_obs | obs | 🔴 | **`low_possession` 根因 A 候选**；snapshot-021 直接测 |
| Policy 时间盲 | obs | 🟡 | **可能是 `late_collapse` 机制之一**；潜在 obs-expansion 候选 |
| Team0/team1 obs 不对称 | obs | ⚪ | 设计合理，文档补齐 |
| possession_bonus proximity-based | reward | ⚪ | 已知 |
| **Shaping 对两 agent 对称** | reward | 🟡 | **`low_possession` 根因 B 候选**；snapshot-022 直接测 |
| aggregate 是 sum 不是 mean | reward | ⚪ | 已知语义 |
| env 原生 reward 不受 shaping toggle 控 | reward | ⚪ | 修正 snapshot-013 §12 的"no-shape 否决 reward 假设"推理 |
| ActionFlattener bijective | action | ⚪ | 已知 |
| Agent_id spawn 稳定性（已验证：稳定 + role-mapping 不干净）| identity | 🟡（原 🔴 已降级）| P0 已完成；spawn 位置稳定但 agent_id ↔ field role 不是严格对应，snapshot-022 claim 已相应收紧 |
| Eval 从未 side swap | identity | 🟡 | 影响 report 完整性；BC @2100 应补测 |
| FrozenBaseline obs 契约差异 | identity | ⚪ | 大概率无害，smoke 验证 |
| Episode 终止对 policy 不可见 | episode | 🟡 | 和 "policy 时间盲"是同一问题 |
| Sparse goal reward 强制对称 | reward | ⚪ | 已知任务定义 |
| CC 看 teammate_action / inference 硬编码 0 | train-eval | 🟡 | **不影响 inference 行为**，轻微影响 training advantage 质量 |
| Shaping 在 eval 是否生效 | train-eval | ⚪ | 无害 |
| Official API 推理时无 `info` 访问 | train-eval / API | 🔴 | **submission 级硬约束**；021 必须拆成本地诊断与 official-aligned 两条 |
| **Train 用 info 真值 / eval 用 lossy decoder（snapshot-021 首轮）** | train-eval | 🟡 | **真实风险，但当前更像次因**；decoder L2=35 噪声仍让 actor 输入分布在 train/eval 间不重叠；预防方法见 §2.6.4 |
| **Obs 扩展字段未归一化，first-layer activation 被高量级维度主导** | train-eval / obs | 🔴 | **当前 snapshot-021 首轮最强主因假设**；raw teammate tail `±15/±7/±8` 直接拼到 `[0,1]` own_obs 上，预防方法见 §2.6.5 |

**总计**：3 🔴 + 6 🟡 + 8 ⚪

**2026-04-16 更新**：snapshot-021 首轮原型暴露出两个相关但不同的问题：
- §2.6.4：**decoder 分裂** 是真实存在的 train-eval 风险，但 `021b` 数据显示它不是唯一主因
- §2.6.5：**obs tail 未归一化** 是当前更强的训练期主因假设

因此当前 🔴 数仍为 3，但第三个 🔴 已从“decoder 分裂”替换为“unnormalized teammate tail dominance”。

---

## 4. 对当前实验的即时影响

### 4.1 Snapshot-022 的 spawn 验证（已完成）

**开训前 gate**：验证 `agent_id 0/1` 的 spawn 位置稳定性。

**执行结果（2026-04-15）**：

20 次 reset + 第一步零动作后读取 `info["player_info"]["position"]`，**位置完全稳定**：

```
Blue  agent 0: (-9.03, +1.20)     <- 更深，后场
Blue  agent 1: (-6.24, -1.20)     <- 更前，中前
Orange agent 0: (6.45, +1.20)      <- 和 agent 1 深度几乎同
Orange agent 1: (6.66, -1.20)
a0_left_of_a1_count = 20/20
```

**决策（已采纳）**：
- spawn 位置本身稳定 → snapshot-022 可以开，**但 claim 收紧**
- Agent_id 与 field role 不是干净映射（blue 方 agent 0 比 agent 1 深 2.8 单位，orange 方几乎同深）
- snapshot-022 标题从 "Role-Differentiated (striker/defender)" 改为 **"Agent-ID Asymmetric Shaping"**
- 测试问题从 "严格 role shaping 有没有用" 改为 "**打破 agent_id 对称性本身是否让 shared policy 学出分化 behavior**"
- §6.3 行为判据加"必须和 scratch MAPPO+v2 的 |Δx| 基线对比"，避免把 blue 方天然 2.8 单位 spawn 偏移误认为 reward 分化效果
- 若 022 过，下一轮可做"按 spawn x 符号 runtime 绑定 role"的严格版

### 4.2 Snapshot-021 可以考虑顺便验证"policy 时间盲"假设

原计划 [snapshot-021](../experiments/snapshot-021-actor-teammate-obs-expansion.md) 加 +4 维 teammate (pos+vel) 到 actor obs。**可以顺便加 +1 维 `episode_step / max_steps`** 作为副产物——代价几乎为零。

两个 obs 扩展**正交**（teammate info 修 coordination，time info 修 temporal awareness），一起加可以测试两个假设。

建议在 snapshot-021 §4.1 最小版改成 "+5 维"（4 teammate + 1 time），在 failure bucket 分析时**分别看** `low_poss` 和 `late_collapse` 的变化：

- low_poss 降 + late_collapse 不变 → teammate obs 有效
- low_poss 不变 + late_collapse 降 → time obs 有效
- 都降 → 都有效
- 都不降 → 都不是根因

### 4.3 SOTA checkpoint 的 side-swap 验证（🟡 建议）

取 BC @2100（current SOTA），补跑 200-ep `team0_ids=(2,3)` eval。

- 如果胜率保持 0.80+ → side-agnostic，report 可写"policy 在两侧均能工作"
- 如果明显下降 → report 要诚实写"policy 仅在 train 一侧测试"，避免过度声明

这对 WR 数字没影响，但对 **report 诚实性** 有加分。

---

## 5. 对 snapshot-013 §12 推理的修正

§2.2.3 的发现影响了 [snapshot-013 §12.2](../experiments/snapshot-013-baseline-weakness-analysis.md#122-113-根因诊断被更新) 当时对根因 B 的判断：

**原推理**：
> "训练分布层面：低占球起始罕见" ❌ 被 BC 否决——BC 见了 teacher 全部轨迹也修不到

**修正**：
- env 原生的 sparse goal reward **对称**
- 我们"关 shaping" 不关它
- 所以"MAPPO no-shape 也有 low_poss"不否决 reward 对称假设——因为 sparse goal reward 本身**也对称**
- **根因 B (reward 对称性) 的证据实际上被"关 shaping 也有 low_poss"这个数据点继续支持**，不是反驳

这不改变 snapshot-013 §12.2 整体判断（根因 A obs 仍是最强候选），但把 B 的可信度**上调**。和 snapshot-022 的实验设计一致。

---

## 6. 推荐的后续工作

| 优先级 | 动作 | 成本 | 触发后做什么 |
|---|---|---|---|
| P0 | Spawn 稳定性验证 | 30 秒 | 决定 snapshot-022 是否需要改代码 |
| P0 | Snapshot-021 加 time obs（+1 维）| 0 额外成本（顺便）| 多获得"time awareness"的 ablation 数据 |
| P1 | BC @2100 side-swap eval | 15 分钟 | Report analysis 诚实素材 |
| P1 | Snapshot-013 §12.2 的"reward 对称"证据修正 | 5 分钟文档 | 和 snapshot-022 结论对齐 |
| P2 | 把 FrozenBaseline obs 契约差异写成工程笔记 | 15 分钟 | 未来新 lane 加入时避免踩坑 |
| P2 | 记录"env 原生 reward 在 shaping toggle 外" | 5 分钟 | 避免未来误解 no-shape 语义 |

---

## 7. 关键教训

### 7.1 这次审计方法本身的价值

"**按假设反向查代码**"比"逐模块正向查功能"发现的问题**完全不一样**。code-audit-000 是标准的模块审计，没发现本轮 3 个 🔴。

**假设驱动审计** 的核心是**列出"默认成立但没人验证"的清单**。future 版本的 audit 如果保持这种 prompt 结构，能继续发现新隐藏约束。

### 7.2 为什么 own_obs 问题之前没被发现

- 前任代码 + 上游代码（soccer_twos-starter）的 CTDE 实现是正确的
- 没人在项目文档里**明确说过**"actor 不看 teammate"——相反，文档里反复提"coordination"、"teammate"
- 这种**代码实现正确 + 文档默认错误**的组合，在 snapshot 级别的叙述里看不出来
- 只有**明确问"actor 到底吃什么 shape 的 tensor"才能看出**

这个模式值得记在工程规范里：**每条新 lane 的 snapshot 应该在第一段写清楚 "actor input / critic input / reward dict / agent_id 绑定" 四条基础事实**，不要默认。

### 7.3 对 Report 的含义

9-lane 数据 × 本次审计发现的 2 个 obs 约束（own_obs only + time-blind）+ 1 个 reward 约束（对称） = **报告里 Analysis and Discussion 30 分的完整素材**：

> "我们运行了 9 条 intervention 的 ablation，发现 `low_possession` 占失败比在所有 lane 上稳定在 22-28%。进一步的代码审计揭示这可能源于三个结构性约束：(1) actor 的 obs 空间不含队友 state，(2) actor 对 episode time 完全盲，(3) shaping reward 对两个 agent 完全对称。这三个约束不由任何 reward/architecture/data 调整所能单独修复，指向 obs-space 扩展和 role-differentiated reward 作为未来工作的直接方向。SNAPSHOT-021/022 是这两个方向的预注册直接检验。"

这一段**比单纯 report SOTA 0.842 更有 technical reasoning 价值**，应作为 report analysis 的主体之一。

---

## 8. 相关文档

- [code-audit-000](code-audit-000.md) — 接手时全模块线性审计（本审计的前置）
- [overview.md](overview.md) — 架构总览
- [SNAPSHOT-013 §11/§12](../experiments/snapshot-013-baseline-weakness-analysis.md) — `low_possession` 不变量诊断
- [SNAPSHOT-017 §11](../experiments/snapshot-017-bc-to-mappo-bootstrap.md) — 9-lane failure bucket 数据
- [SNAPSHOT-021](../experiments/snapshot-021-actor-teammate-obs-expansion.md) — 根因 A 直接验证
- [SNAPSHOT-022](../experiments/snapshot-022-role-differentiated-shaping.md) — 根因 B 直接验证
