# SNAPSHOT-027: Team-Level PPO with Native Coordination

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 已完成首轮结果

## 0. 续跑说明

`027-A` 首轮正式 run 在 home quota 打满后中断，原始 run root 为：

- [PPO_team_level_v2_scratch_768x512_20260417_095059](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059)

断点与恢复信息：

- 原计划续跑点：`checkpoint-700`
- 实际恢复点： [checkpoint-690](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_8384d_00000_0_2026-04-17_09-51-20/checkpoint_000690/checkpoint-690)
- 原因： [checkpoint-700](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_8384d_00000_0_2026-04-17_09-51-20/checkpoint_000700/checkpoint-700) 本身损坏，restore 会触发 `_pickle.UnpicklingError`
- 成功续跑 trial： [TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02)
- 失败的中间重试 trial：`27175 / 7b8b9 / 9047e`

当前应以 run root 下的 **merged summary** 作为正式总摘要，而不是只看单个 trial 的末尾打印。这里的 canonical 输出包含 **文本摘要 + 合并曲线图** 两部分：

- [merged_training_summary.txt](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/merged_training_summary.txt)
- [training_loss_curve_merged.png](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/training_loss_curve_merged.png)

该 merged summary 由 [print_merged_training_summary.py](../../scripts/tools/print_merged_training_summary.py) 生成，已经确认能拼出：

- 原始 trial：`it 1 -> 699`
- 续跑 trial：`it 691 -> 1250`

此外，`027-A` 的训练内 checkpoint eval 曾被坏掉的 `checkpoint-700` 卡住：旧 monitor 会不断重试同一个失败 matchup，导致 root [checkpoint_eval.csv](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/checkpoint_eval.csv) 一度停在 `690`。后续已通过改进后的 [backfill_run_eval.py](../../scripts/eval/backfill_run_eval.py) 对 `710+` 完成回填，因此现在的 root `checkpoint_eval.csv` 已可作为完整选模输入，但内部 eval 仍应视为**候选过滤器**，正式结论以 official `baseline 500` 为准。

因此，`027-A` 后续结果解读必须明确区分“原始段”和“quota 中断后的续跑段”，但最终统计以 merged summary 为准；图形上也应优先查看 `training_loss_curve_merged.png`，而不是任一单段 trial 自己的旧 `training_loss_curve.png`。

## 1. 动机

截至当前，所有训练 lane 都使用 `multiagent_player` 模式：每个 agent 有独立 policy，只看自己的 336 维 obs，独立输出动作。即使在 MAPPO 架构下，centralized critic 也只在训练时让 value function 看到队友信息——actor 仍然是盲的。

这导致了一个贯穿项目始终的根本限制：**两个 agent 无法协调**。

- [SNAPSHOT-006](snapshot-006-fixed-teammate-and-dual-expert-rethink.md) 的 dual-expert + 部署协调器 → 0.780，没超过 single-policy 平台
- [SNAPSHOT-021](snapshot-021-actor-teammate-obs-expansion.md) 试图在训练时注入 teammate state → 训练-部署 obs gap 导致部署掉分
- [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) / [024](snapshot-024-striker-defender-role-binding.md) 用 reward asymmetry 做隐式分工 → 有效但上限受限于 agent 看不到队友
- `low_possession` 22-28% 不变量跨 14 条 lane 从未被攻破 → 可能正是因为 agent 无法感知队友位置来配合控球

但回到上游 starter code 和 SSOT 的设计：

- `AgentInterface.act()` 接收 `Dict[int, np.ndarray]`——**同时包含整队所有成员的 obs**
- starter 提供了 `multiagent_team` variation 和 `team_vs_policy` variation——都是 **team-level 训练**
- `MultiagentTeamWrapper` 和 `TeamVsPolicyWrapper` 把两人 obs 拼接为 672 维，输出联合动作
- starter 的 `example_ray_ma_teams.py` 就是用这个模式训练的

也就是说：**上游设计者明确预期 team-level training 是一条合理路线**。而我们从 [SNAPSHOT-008](snapshot-008-starter-aligned-base-model-lane.md) Base-C 提出后，这条线从未被正式训练和评估过。

## 2. 核心假设

### 主假设

如果 0.84 天花板的主要瓶颈之一是"两个 agent 无法协调"（而非纯粹的 reward shaping 或 obs 设计问题），那么 team-level PPO——一个 policy 同时看到两人 obs、同时输出两人动作——应当能：

1. 打破 `low_possession` 22-28% 不变量（因为 policy 能感知两人位置，做出传球/跑位/补位等配合）
2. 减少 `late_defensive_collapse`（因为 policy 能调度一人回防、一人控球）
3. official `baseline 500 >= 0.84`（追平或超过当前冠军）

### 备择假设

如果 team-level PPO 在 v2 shaping 下仍然 ≤ 0.80：

- 672 维 obs + MultiDiscrete([3,3,3,3,3,3]) 的搜索空间可能太大，500 iter 不够收敛
- 或者：协调能力不是当前瓶颈，reward / obs 设计才是

## 2b. 对标基准：MAPPO+v2 scratch，不是当前 SOTA

本 snapshot 必须明确说明对标的是什么，不然容易被误读。

当前 SOTA 路径是：

```
baseline teacher 轨迹（015）
  → player-level BC（015/017）
  → BC→MAPPO warm-start（017）→ BC@2100 = 0.842
  → field-role binding stable（025b@80）= 0.842 + H2H 压过 BC
```

**整条 SOTA 都依赖 BC bootstrap**。而本 snapshot 的 027-A/B 是 **team-level scratch**——跳过了让 017 成为 SOTA 的那一步。因此：

- **027 对标的是 `MAPPO + v2 shaping scratch`**（per-agent scratch baseline），不是 BC@2100 或 025b@80
- 即 027 测的是：**把 per-agent scratch 换成 team-level scratch（v2 shaping 不变），架构转换本身的价值**
- 这个假设下 027 的合理目标是"追平或小幅超过 per-agent scratch"，不是"追平 SOTA"

公平的 SOTA 挑战版本（team-level IL→BC→RL）在 [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) 中独立做。027 的结果不论正负，都必须在这个对标下解读。

## 3. 设计

### 3.1 env variation

使用 `team_vs_policy`（不是 `multiagent_team`）：

- `team_vs_policy`：team0 由一个 policy 控制（672 维 obs → 联合 action），team1 由 baseline 控制
- 这和我们所有 per-agent 实验的对手设定一致（vs baseline）
- 是 **single-agent PPO**（只有一个 policy），不是 multi-agent

### 3.2 obs space

`TeamVsPolicyWrapper` 自动 concat：

```
obs = np.concatenate((obs[agent_0], obs[agent_1]))  # shape = (672,)
```

672 维 = 两个 agent 各自的 336 维 egocentric obs。policy 可以从中提取：
- agent 0 看到的球位置/速度、自己相对球门的位置、raycast 物体检测
- agent 1 看到的同样信息

两人的 obs 是不同坐标系（各自 egocentric），但 neural network 应当能学会对齐。

### 3.3 action space

**MultiDiscrete([3, 3, 3, 3, 3, 3])**：

- 前 3 维 = agent 0 的 [forward/back, left/right, rotate]
- 后 3 维 = agent 1 的 [forward/back, left/right, rotate]
- 6 个独立的 3-选-1 决策

不用 Discrete(729)，原因：
- 729 个 action 的初始 entropy = ln(729) ≈ 6.6，太高
- MultiDiscrete 可以分解为 6 ��独立子问题，更容易学
- `TeamVsPolicyWrapper` 在底层 env 是 MultiDiscrete 时直接用 slice 分配，不需要额外转换

**注意**：这意味着 `create_rllib_env` 在创建 env 时 **不能用 `flatten_branched=True`**。需要确认当前代码是否在 `team_vs_policy` 下默认 flatten。

### 3.4 reward shaping

沿用 v2 shaping 配置。`TeamVsPolicyWrapper.step()` 返回 `reward[0] + reward[1]`，所以 team-level reward 自动是两人 shaping reward 之和。

需要确认 `RewardShapingWrapper` 在 `team_vs_policy` 模式下能正确工作——它应该包在底层 `MultiAgentUnityWrapper` 之上、`TeamVsPolicyWrapper` 之下，这样每个 agent 的 shaping 先独立计算，再由 team wrapper 求和。

### 3.5 网络架构

- 标准 PPO（不需要 centralized critic——actor 已经看到全队信息）
- `fcnet_hiddens = [768, 512]`（672 维输入比 336 维大一倍，第一层适当加宽）
- `vf_share_layers = True`（team-level 没有 actor-critic obs 分裂问题）
- 备选对照：`[512, 512]` 保持和 per-agent 一致

### 3.6 warm-start

**不能从 BC@2100 warm-start**：

- BC@2100 的 policy 输入是 336 维（per-agent MAPPO，含 CC obs space）
- team-level policy 输入是 672 维
- 网络第一层 weight shape 不匹配，无法直接迁移

因此本 snapshot **只做 scratch**。如果首轮出正结果，后续可以考虑：
- 在 team-level policy 上做 BC（从 baseline team-level 轨迹学习）
- 或设计 adapter 把两个 per-agent policy 的 weight 拼接成 team-level

### 3.7 训练预算

从 scratch 训，搜索空间远大于 per-agent（672 维 obs + MultiDiscrete 联合动作 + 协调是全新学习目标），需要充足预算：

- `TIMESTEPS_TOTAL = 50,000,000`
- `TIME_TOTAL_S = 57,600`（16h）
- `MAX_ITERATIONS = 0`（不限 iter，按 steps 停）
- SLURM 单次最多申请 16h（`-t 16:00:00`），如果被 preempt 或超时，用 `RESTORE_CHECKPOINT` 续跑

## 4. 和 per-agent MAPPO 的本质差异

| | per-agent MAPPO | team-level PPO |
|---|---|---|
| policy 数量 | 1 shared policy × 2 agents | 1 team policy |
| actor 输入 | 336 维（own obs only）| 672 维（both agents' obs）|
| critic 输入 | 672+ 维（team concat + teammate action）| 672 维（same as actor）|
| 训练时协调 | **不能**（actor 看不到队友）| **能**（同一 forward pass）|
| 部署时协调 | 不能（各自独立 inference）| **能**（`act()` 拿到 dict → concat → 一次 forward）|
| action 空间 | Discrete(27) per agent | MultiDiscrete([3,3,3,3,3,3]) |
| 优势 | 更小搜索空间，可 warm-start | **天然协调**，训练-部署语义一致 |
| 劣势 | 无法协调 | 更大搜索空间，不能从现有 ckpt warm-start |

核心不同：MAPPO 的 centralized critic 是一个 **训练时 trick**——它让 value function 估计更准，但不改变 actor 的决策能力。Team-level PPO 是从根本上让 **actor 看到全局信息并做出联合决策**。

## 5. 风险

### R1 — 搜索空间膨胀

672 维 obs × MultiDiscrete([3,3,3,3,3,3]) 的联合搜索空间远大于 336 × Discrete(27)。从 scratch 训 500 iter 可能不够收敛。

缓解：监控 entropy 和 reward 曲线。如果 iter 200 时 WR < 0.50，考虑加 entropy_coeff 或扩大训练预算。

### R2 — Agent 顺序敏感性

`TeamVsPolicyWrapper` 固定 concat 为 `(obs[0], obs[1])`。但 agent 0 和 1 的 spawn 位置每局可能不同。如果 policy 对"obs 前 336 维是哪个 agent"过度敏感，泛化可能差。

缓解：如果首轮结果显示 WR 方差异常大，后续可以做 data augmentation（随机 swap obs[0] 和 obs[1] 的顺序）。

### R3 — flatten_branched 干扰

当前 `create_rllib_env` 可能在 `team_vs_policy` 模式下默认 `flatten_branched=True`，把 MultiDiscrete 压成 Discrete(729)。需要在代码实现时确认并修正。

### R4 — Shaping wrapper 层序

`RewardShapingWrapper` 需要在 `TeamVsPolicyWrapper` 之前（即包在底层 multiagent env 上），这样每个 agent 的 shaping 独立计算后再被 team wrapper 求和。当前代码的 wrapping 层序需要确认。

## 6. 执行矩阵

| lane | shaping | network | iter | GPU 预估 |
|---|---|---|---|---|
| **027-A** | v2 shaping | [768, 512] | 50M steps | ~16h |
| **027-B**（可选对照）| v2 shaping | [512, 512] | 50M steps | ~16h |

首轮只跑 **027-A**。如果 027-A 正结果，再跑 027-B 做网络宽度消融。

## 7. 预声明判据

### 7.1 主判据

按照 §2b，对标的是 **MAPPO+v2 scratch (per-agent)**，不是 SOTA：

| 阈值 | 逻辑 |
|---|---|
| **official 500 ≥ 0.78** | 追平 SNAPSHOT-014 `MAPPO+v2 scratch` 首轮水平 = team-level scratch 架构转换不劣化 |
| **official 500 ≥ 0.81** | 追平 Pool 018 (0.812) = team-level scratch 小幅优于 per-agent scratch 最佳 |
| **official 500 ≥ 0.84** | 意外地追平 SOTA = team-level 本身强到可以不靠 BC bootstrap（极强信号，预期不会）|

### 7.2 机制判据

**failure capture 中 `low_possession` < 20%**

当前所有 per-agent lane 的 `low_possession` 在 22-28% 不��。如果 team-level 能把它压到 20% 以下，说明协调确实是 low_poss 的主因。

### 7.3 失败情形

| 条件 | 解读 |
|---|---|
| official 500 < 0.70 | 672 维 + MultiDiscrete 从 scratch 没学到足够策略 → 需要 [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) 的 BC bootstrap |
| official 500 在 0.70-0.78 | 架构转换损失性能 → 027 方向本身风险大，优先看 028 结果 |
| official 500 在 0.78-0.81 且 low_poss 不变 | 搜索空间更大但协调没带来本质变化 → 备择假设方向 |
| official 500 ≥ 0.81 但 low_poss 仍 ≥ 22% | team-level 有效但 low_poss 的原因不是协调 → 需要重新分析 low_poss 根因 |

## 8. 工程需求

### 8.1 实际落地文件

`027-A` 现阶段不需要新写 team-level 训练/部署代码，仓库里已有闭环可直接复用：

1. **训练脚本（复用）**: [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)
   - 已使用 `variation = team_vs_policy`
   - 已支持 v2 shaping
   - 已支持 `fcnet_hiddens`、`entropy_coeff` 等 env var
   - 与本 snapshot 的 scratch 训练需求一致

2. **部署 wrapper（复用）**: [trained_team_ray_agent.py](../../cs8803drl/deployment/trained_team_ray_agent.py)
   - `act(observation)` 接收 `{0: obs0, 1: obs1}`
   - concat 为 672 维 team obs
   - 输出 `MultiDiscrete([3,3,3,3,3,3])` 后再 slice 给两人

3. **batch 脚本（新增）**: [soccerstwos_h100_cpu32_team_level_v2_scratch_768x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_v2_scratch_768x512.batch)

### 8.2 需要确认的代码

1. `create_rllib_env` 在 `team_vs_policy` 下是否 `flatten_branched=True`
2. `RewardShapingWrapper` 在 `team_vs_policy` 下的 wrapping 层序
3. `TeamVsPolicyWrapper` 底层 action space 是 MultiDiscrete 还是 Discrete

## 9. 和既有线的关系

| snapshot | 和 027 的关系 |
|---|---|
| [008 Base-C](snapshot-008-starter-aligned-base-model-lane.md) | 同一方向（multiagent_team），但 008 没有 shaping、从未跑过 |
| [006 dual-expert](snapshot-006-fixed-teammate-and-dual-expert-rethink.md) | 006 是部署期协调，027 是训练期协调——本质不同 |
| [021 teammate obs](snapshot-021-actor-teammate-obs-expansion.md) | 021 试图在 per-agent 架构里注入 teammate info，027 直接换架构 |
| [014 MAPPO+v2 scratch](snapshot-014-mappo-fair-ablation.md) | **本 snapshot 真正的对标**——同 shaping 不同架构（per-agent vs team-level）|
| [017 BC@2100](snapshot-017-bc-to-mappo-bootstrap.md) | 当前 SOTA，但依赖 BC bootstrap；**不是 027 的公平对照** |
| [025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) | 当前冠军位（0.842 + H2H 压 BC），同样不是 027 对标 |
| [026 reward liberation](snapshot-026-reward-liberation-ablation.md) | 并行实验，026 从 reward 侧突破，027 从 architecture 侧突破 |
| [028 team-level BC→PPO](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | 027 的 SOTA 挑战版本——补齐 BC bootstrap，才是公平对比 025b 的那条线 |

## 10. 不做的事

- **不做 warm-start**（维度不兼容，见 §3.6）
- **不做 centralized critic**（actor 已看全局，CC 无意义）
- **不做 role-diff shaping**（首轮只测 team-level 架构本身的价值，用标准 v2 shaping）
- **不做 self-play**（先对齐 baseline 目标）

## 11. 执行清单

1. 确认 `create_rllib_env` + `TeamVsPolicyWrapper` 的 action space 和 flatten 行为
2. 确认 `RewardShapingWrapper` 在 `team_vs_policy` 模式下的 wrapping 层序
3. 创建 `027-A` batch 脚本
4. 1-iter smoke：
   - 验证 obs shape = 672
   - 验证 action space = MultiDiscrete([3,3,3,3,3,3])
   - 验证 shaping reward 在训练 log 中可见
   - 验证 iter-1 的 `baseline 50` 不低于 0.10（scratch sanity）
5. 提交首轮长训（50M steps, ~16h）
6. 按 §7 判据做 official `baseline 500` 选模
7. 对 best ckpt 做 failure capture，重点看 `low_possession`
8. verdict 落本文件 §12+（append-only）

## 12. Preflight（已完成）

`027-A` 的关键工程前提已经用真实 env smoke 验证过：

- `create_rllib_env(... variation=team_vs_policy, multiagent=False)` 的 reset obs shape = `(672,)`
- action space 保持为 `MultiDiscrete([3,3,3,3,3,3])`，没有被压成 `Discrete(729)`
- `team_vs_policy` 的 wrapper 链路为：
  - `EnvChannelWrapper -> TeamVsPolicyWrapper -> MultiAgentUnityWrapper`
- 在开启 v2 shaping 时：
  - 顶层 env 变为 `RewardShapingWrapper`
  - `step()` 返回 scalar team reward
  - `info` 中可见 `_reward_shaping`

因此 `027` 目前没有发现会阻塞首轮训练的工程问题。

## 12. 首轮正式结果（已完成）

`027-A` 最终完整跑到 `50M steps / 1250 iter`，merged summary 显示 training reward 在后段仍保持健康：

- merged best reward：`+2.2141 @ iter 922`
- final reward：`+2.1969 @ iter 1250`
- canonical summary： [merged_training_summary.txt](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/merged_training_summary.txt)

但在回填完 `710+` 的 internal eval 后，我们没有直接相信 `checkpoint-830 = 0.900` 这一类 late-window internal peak，而是按与 [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) 相同的 **top 5% + ties + `±2 checkpoint window`** 协议重新做了 official `baseline 500`。

official `baseline 500` 的真实结果是：

- best official： [checkpoint-650](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_8384d_00000_0_2026-04-17_09-51-20/checkpoint_000650/checkpoint-650) `= 0.804`
- 并列 best official： [checkpoint-1230](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02/checkpoint_001230/checkpoint-1230) `= 0.804`
- 次优窗口： [checkpoint-1210](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02/checkpoint_001210/checkpoint-1210) `= 0.792`、 [checkpoint-1130](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02/checkpoint_001130/checkpoint-1130) `= 0.788`

这说明：

1. `027A` 不是负结果。team-level scratch 在 official `500` 下可以稳定站到 `0.80` 左右，说明“一个 policy 同时看两人 obs 并输出联合动作”这条架构路线本身是可行的。
2. `027A` 也不是冠军线。即使在回填后使用更系统的 window 选点，best official 仍只有 `0.804`，没有出现被漏掉的 `0.84+` 高点。
3. late-window internal eval 明显高估了真实表现。典型例子是 [checkpoint-830](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_a2de0_00000_0_2026-04-17_19-32-02/checkpoint_000830/checkpoint-830)：internal `0.900`，但 official `500` 只有 `0.758`。

因此，`027A` 的最稳收口是：

- **它是一个成立的 team-level scratch 正结果线**
- **它的 true official ceiling 大约在 `0.80`**
- **它明显低于当前 per-agent 冠军线，也低于后续的 [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) team-level BC warm-start 结果**

这个结果也反过来加强了 `028` 的机制解释：如果 team-level scratch 只能到 `~0.80`，而 team-level BC warm-start 能把 official best 拉到 `0.844`，那么 BC bootstrap 对 team-level PPO 的提升就是实打实的，而不是单纯来自“team-level 架构本身已经足够”。

## 13. 相关

- [SNAPSHOT-006: dual-expert](snapshot-006-fixed-teammate-and-dual-expert-rethink.md)
- [SNAPSHOT-008: Base-C multiagent_team](snapshot-008-starter-aligned-base-model-lane.md)
- [SNAPSHOT-021: teammate obs expansion](snapshot-021-actor-teammate-obs-expansion.md)
- [code-audit-001](../architecture/code-audit-001.md)（hidden constraints: actor own_obs only, official API）
