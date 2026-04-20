# SNAPSHOT-017: BC -> MAPPO Bootstrap

- **日期**: 2026-04-14
- **负责人**:
- **状态**: 已完成首轮结果

## 1. 背景

到当前为止，我们已经把 `BC` 和 `MAPPO` 两条主线分别跑通：

- `BC` 最小闭环已落地，见 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md)
- 纯 `BC` 官方评估结果：
  - [BC checkpoint-30](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033/checkpoint_000030)
  - `vs baseline = 0.554`
  - `vs random = 0.974`
- `MAPPO` 首轮公平对照已完成，见 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md)
- 当前最强 `MAPPO` 候选：
  - [MAPPO + shaping-v2 checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
  - 官方 `baseline 500 = 0.786`

因此，当前最自然的下一步不是继续单独强化 `BC` 或单独强化 `MAPPO`，而是回答：

> baseline teacher 学到的行为先验，能否作为更好的初始化接入当前最强的 `MAPPO + shaping-v2` 主线？

## 2. 主假设

如果我们能把 `BC` 学到的 baseline-level 行为先验，以稳定的方式接入 `MAPPO` 的共享 actor，那么：

1. 训练前期不需要再从随机策略重新学基础控球与推进；
2. `MAPPO` 可以把更多预算放在：
   - teammate coordination
   - failure bucket 中尚未完全修掉的 `low_possession`
   - `poor_conversion`
3. 最终性能有机会超过当前 scratch `MAPPO + shaping-v2 = 0.786`。

一句话：

> `BC` 负责给出 teacher-aligned 起点，`MAPPO` 负责把它推到 baseline 之上。

## 3. 当前最大的技术约束

这里不能假设“BC checkpoint 可以直接塞进 MAPPO”。

当前两边的表征并不完全同构：

### 3.1 当前 BC 是 team-level

- 模型定义：[imitation_bc.py](../../cs8803drl/branches/imitation_bc.py)
- 当前 checkpoint：
  - [BC checkpoint-30](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033/checkpoint_000030)
- 输入：
  - `team obs = 672`
- 输出：
  - `team action = 6 branches`

### 3.2 当前 MAPPO actor 是 player-level shared actor

- 训练入口：[train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)
- 模型定义：[shared_central_critic.py](../../cs8803drl/branches/shared_central_critic.py)
- actor 只吃：
  - `own_obs`
- critic 才额外看：
  - `own_obs + teammate_obs + teammate_action`

所以：

- 现有 team-level BC checkpoint 不能直接作为 MAPPO actor 的“同构 state_dict”来加载；
- 真正的工程问题不是“要不要 warm-start”，而是“怎么桥接 team-level BC 与 player-level shared actor”。

这也是本 snapshot 要先解决的第一件事。

## 4. 本轮范围

本 snapshot 先只覆盖：

1. 明确 `BC -> MAPPO` 的兼容桥接方案；
2. 做最小 warm-start smoke；
3. 只在 smoke 成功后，再进入正式短跑或长跑训练。

本轮先**不**做：

- `BC -> PPO`
- MARWIL
- ensemble
- 多条 bridge 同时大规模对照

## 5. 候选桥接方案

### 5.1 首选：新增 player-level BC bridge

思路：

- 复用现有 baseline teacher 数据采集链路
- 再产一份 player-level teacher dataset
- 训练一个与 shared actor 更对齐的 player-level BC policy
- 再把这个 player-level BC actor warm-start 到 MAPPO shared actor

优点：

- 表征对齐最干净
- 权重映射最自然
- 最容易解释实验结果

代价：

- 需要新增 player-level BC 数据与 trainer / 或最小变体

### 5.2 次选：从 team-level BC 做启发式部分映射

思路：

- 只拷贝后半部分 trunk / action head 中可对上的部分
- 或对第一层做手工切片适配

优点：

- 不用重采数据

缺点：

- 风险大
- 解释性弱
- 很容易出现“能加载但不是合理初始化”

当前判断：

- 这个方案只作为备选，不作为首选主线。

### 5.3 当前已选定的实现方向

本 snapshot 当前已明确采用 **5.1 player-level BC bridge** 作为执行路线。

原因：

- teacher trajectory collector 已经原生支持 `--mode player`
- 这条线和 shared actor 的表征最接近
- 相比从 team-level BC checkpoint 做手工切片，它的解释性和稳定性都更好

因此，后续实现默认沿以下顺序推进：

1. 采集 `player-level baseline self-play` teacher dataset
2. 训练 `BC player policy`
3. 只把 `BC player` 的 actor 侧 trunk / logits warm-start 到 MAPPO shared actor
4. 再做短跑 smoke，最后再进入正式 A/B

### 5.4 当前已完成的桥接落地

本轮已经把 `BC -> MAPPO` 的最小 bridge 真正接起来，而不再停留在设计阶段。

新增 / 扩展的关键实现：

- player-level BC trainer：
  - [train_bc_player_policy.py](../../cs8803drl/training/train_bc_player_policy.py)
- shared centralized-critic warm-start helper：
  - [shared_central_critic.py](../../cs8803drl/branches/shared_central_critic.py)
- MAPPO 训练入口新增 `BC_WARMSTART_CHECKPOINT`：
  - [train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)

这里有一个对工程推进很关键的改动：

- 如果 teacher dataset 目录下暂时没有独立的 `player` shards，
- [train_bc_player_policy.py](../../cs8803drl/training/train_bc_player_policy.py) 现在会回退到现有 `team` shards，
- 并在读取阶段把每条 team sample 拆成两个 player sample。

这意味着：

- 不需要重新等待一轮 player-only 采集，
- 就可以先验证 `player-level BC -> MAPPO actor` 这条桥。

## 6. 当前推荐执行顺序

### 6.1 Phase A: 兼容性桥接确认

目标：

- 确认最终采用哪条 bridge
- 明确 warm-start 插入点

当前最可能的插入点：

- [warmstart_shared_cc_policy](../../cs8803drl/branches/shared_central_critic.py)

但这部分现阶段只支持：

- 从 RLlib checkpoint 向 shared CC policy 拷贝

因此，本阶段需要先决定：

- 是扩展该 helper 支持 BC checkpoint
- 还是先补一条 player-level BC 训练线，再沿现有 helper 逻辑接入

### 6.2 Phase B: Warm-start Smoke

成功标准：

1. BC 权重能被稳定加载到 MAPPO actor 侧；
2. 短跑训练不报错、不数值发散；
3. 训练前几十 iter 的 baseline 50 表现不低于 scratch MAPPO 同阶段。

### 6.2.1 当前 smoke 结果

本轮已经完成两步最小烟测：

1. `player-BC` smoke
   运行目录：
   - [BC_player_smoke_20260414](../../ray_results/BC_player_smoke_20260414)
   输出 checkpoint：
   - [checkpoint_000001](../../ray_results/BC_player_smoke_20260414/checkpoint_000001)

   关键元数据：
   - `format = bc_player_policy_v1`
   - `obs_dim = 336`
   - `action_nvec = [3, 3, 3]`
   - `flat_action_dim = 27`

   这一步确认：
   - player-level BC checkpoint 的结构已经和 MAPPO shared actor 对齐。

2. `BC -> MAPPO` warm-start smoke
   运行目录：
   - [PPO_mappo_bc_player_smoke_20260414](../../ray_results/PPO_mappo_bc_player_smoke_20260414)
   输出 checkpoint：
   - [checkpoint-1](../../ray_results/PPO_mappo_bc_player_smoke_20260414/MAPPOVsBaselineTrainer_Soccer_65d97_00000_0_2026-04-14_19-12-25/checkpoint_000001/checkpoint-1)

   关键日志信号：
   - 训练启动时正常打印：
     - `[bc-warmstart] copied player-level BC weights into shared centralized-critic policy (copied=8, adapted=3, skipped=0)`
   - `1 iter` 训练完整结束并正常写出 checkpoint

   这一步确认：
   - `BC_WARMSTART_CHECKPOINT` 接线已打通
   - actor trunk / logits 的桥接不会在 trainer 初始化、采样、学习或 checkpoint 保存阶段崩掉

因此，`BC -> MAPPO` 这条线当前已经通过了“能不能接起来”的工程门槛。

### 6.3 Phase C: 正式 A/B

如果 smoke 成功，再做真正的对照：

- `scratch MAPPO + shaping-v2`
- `BC -> MAPPO + shaping-v2`

正式选模仍沿用：

- `top 5% + ties`
- `baseline 500`
- `random 500` 只补最终 shortlist

## 7. 判据

### 7.1 主判据

如果 `BC -> MAPPO + shaping-v2` 的官方 `baseline 500` 最优 checkpoint：

- 超过当前 scratch `MAPPO + shaping-v2 = 0.786`

则说明 teacher bootstrap 对当前最强主线有实质增益。

### 7.2 次判据

即使最终 `win_rate` 还没超过 `0.786`，如果出现以下任一信号，也说明方向有效：

1. 训练前期更快进入高位窗口；
2. `low_possession` 明显下降；
3. `poor_conversion` 明显下降；
4. 最终 best checkpoint 的选择更稳定，不再严重依赖 `50` 局尖峰。

## 8. 当前判断

当前我方对 `BC -> MAPPO` 的立场已经更新为：

- 值得做
- 而且优先级够高
- 工程桥接已经证明可行
- 下一步不再是“能不能接”，而是“接上之后是否优于 scratch MAPPO”

因此，本 snapshot 现阶段的重点已经从“桥接方式设计”切换为：

> 在 bridge smoke 已通过的前提下，进入第一条正式 short-run / fair-run 对照。

## 9. 与总路线的关系

这条线是 [PLAN-002](../plan/plan-002-il-mappo-dual-mainline.md) 里 Mainline-A 与 Mainline-B 的第一次真正汇合：

- Mainline-A 提供 teacher-aligned initialization
- Mainline-B 提供当前最强的训练结构

如果这条线打通，它会成为当前阶段最值得期待的组合路线。

## 10. 首轮正式结果

### 10.1 正式 run

首轮正式 `BC -> MAPPO + shaping-v2` 运行目录：

- [PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2)

对应 trial：

- [MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18)

训练正常结束：

- `training_iteration = 2335`
- `done = True`
- `timesteps_total = 18.68M`
- `time_total_s ≈ 28008`

### 10.2 训练内窗口

这条线的训练内 `baseline 50` 出现过多次高点：

- [checkpoint-1410](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_001410/checkpoint-1410) = `0.92`
- [checkpoint-2240](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002240/checkpoint-2240) = `0.92`

但正式 `baseline 500` 复核表明：

- 这条线真正兑现的强窗口不在最早的尖峰；
- 而是在后段 `1870-2250` 一带形成了真实高平台。

### 10.3 官方 `baseline 500` 复核

按 `top 5% + ties` 并扩展到“前后各 2 个 checkpoint”后，正式复核出的当前最强点为：

- [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) = **`0.842`**
- [checkpoint-2250](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002250/checkpoint-2250) = `0.834`
- [checkpoint-2170](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002170/checkpoint-2170) = `0.832`
- [checkpoint-1870](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_001870/checkpoint-1870) = `0.830`

这意味着：

- `BC -> RL` 已正式超过 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 当前冠军 [checkpoint-290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290) 的 `0.812`
- 在当前已完成的主线中，`BC -> MAPPO + shaping-v2` 成为新的 `baseline 500` 最佳结果

### 10.4 Failure Capture（4 个高点）

进一步对 4 个最强点做 `baseline 500` failure capture：

- [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)
  - capture `win_rate = 0.828`
- [checkpoint-2250](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002250/checkpoint-2250)
  - capture `win_rate = 0.788`
- [checkpoint-2170](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002170/checkpoint-2170)
  - capture `win_rate = 0.824`
- [checkpoint-1870](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_001870/checkpoint-1870)
  - capture `win_rate = 0.794`

从失败结构看：

- [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 是当前最均衡、最可信的冠军点
- [checkpoint-2170](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002170/checkpoint-2170) 是很强的第二名，但输局更短、更锋利
- [checkpoint-2250](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002250/checkpoint-2250) 虽然官方分高，但 capture 明显更脆
- [checkpoint-1870](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_001870/checkpoint-1870) 稍弱一档

### 10.5 当前结论

首轮 `BC -> MAPPO + shaping-v2` 已经给出明确正结果：

1. `BC` 不是只在训练前期加速；
2. 它最终把强窗口推到了 scratch MAPPO 与 opponent-pool 线之上；
3. 当前最值得保留的正式冠军点是：
   - [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)
4. 强备选为：
   - [checkpoint-2170](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002170/checkpoint-2170)

## 11. Failure Bucket 深度分析与判据 Verdict

### 11.1 4 个高点的失败桶明细（500-ep save-all）

| ckpt | 总失败 | late_collapse | low_poss | poor_conv | unclear | territory | opp_forward | 总失败率 |
|---|---|---|---|---|---|---|---|---|
| **2100** | **86** | 40 | 23 | 11 | 10 | 1 | 1 | **17.2%** |
| 2170 | 88 | 49 | 21 | **6** | 10 | 2 | 0 | 17.6% |
| 2250 | 106 | 51 | 24 | 10 | 20 | 1 | 0 | 21.2% |
| 1870 | 103 | 52 | 24 | 12 | 11 | 2 | 2 | 20.6% |

**ckpt 2100 总失败 86/500 = 17.2%**——项目史上最低总失败率。比 [snapshot-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的 Pool 018 ckpt290（102/500 = 20.4%）再降 16%。

### 11.2 占比（% of losses）对照表

| ckpt | late_collapse | **low_poss** | poor_conv | unclear | territory |
|---|---|---|---|---|---|
| 2100 | 46.5% | **26.7%** | 12.8% | 11.6% | 1.2% |
| 2170 | 55.7% | **23.9%** | 6.8% | 11.4% | 2.3% |
| 2250 | 48.1% | **22.6%** | 9.4% | 18.9% | 0.9% |
| 1870 | 50.5% | **23.3%** | 11.7% | 10.7% | 1.9% |

**4 个 BC ckpt 的 `low_possession` 占比全在 22.6-26.7% 区间**。

### 11.3 跨 lane 对比（failure bucket share）

合并所有已 capture 的 lane：

| Lane | 总失败 | late_col % | **low_poss %** | poor_conv % |
|---|---|---|---|---|
| v1 PPO @430 | 157 | 49.0% | 23.6% | 12.1% |
| v2 PPO @440 fresh | 127 | 48.8% | 23.6% | 9.4% |
| v4 PPO @400 | 139 | 41.7% | **28.1%** | 13.7% |
| MAPPO+v2 @470 | 125 | 49.6% | 24.0% | 8.8% |
| Pool 018 @290 | 102 | 48.0% | **26.5%** | 12.7% |
| **BC @1870** | 103 | 50.5% | **23.3%** | 11.7% |
| **BC @2100** | **86** | 46.5% | **26.7%** | 12.8% |
| **BC @2170** | 88 | 55.7% | **23.9%** | 6.8% |
| **BC @2250** | 106 | 48.1% | **22.6%** | 9.4% |

**9 个 lane × `low_possession` 占失败比 = 22.6% - 28.1%** 的极窄带。跨越：

- 2 种 critic 结构（vanilla actor-critic / centralized critic）
- 5 种不同 shaping（v1 / v2 / v4 / no-shape / Pool 配比变体）
- 2 种初始化（scratch / BC warm-start）
- 多种训练时长（300 iter / 500 iter / 2335 iter）

**`low_possession` 在所有 intervention 下都稳定在 22-28%**，表现为**完全的跨 intervention 不变量**。

### 11.4 §7 判据 Verdict

| 类别 | 判据 | 阈值 | 实测 | 结果 |
|---|---|---|---|---|
| **§7.1 主判据** | 500-ep WR 超过 scratch MAPPO+v2 的 0.786 | | **0.842 @ ckpt2100** | ✅ **PASS** (+0.056) |
| §7.2 次判据 A | 前期进入高位窗口更快 | | iter 960 首次 ≥0.85 (scratch MAPPO 500 iter 未达到) | ✅ PASS |
| §7.2 次判据 B | `low_possession` 明显下降 | ≤ 约 10% (snapshot-013 §11.4) | **22.6-26.7%** | ❌ **FAIL** |
| §7.2 次判据 C | `poor_conversion` 明显下降 | | 6.8-12.8%（BC @2170 极低 6.8%, 其他持平）| ⚠️ 部分（仅 ckpt2170） |
| §7.2 次判据 D | 最终 best 选择更稳定 | | 50-ep best=2240 (0.92→500ep 0.792), 500-ep best=2100 (50ep 0.86) | ❌ **FAIL**（50-ep 继续不可靠）|

**主判据 PASS，4 个次判据 1 PASS + 1 部分 + 2 FAIL**。其中最关键的 **B 判据 FAIL**——BC 没有按假设降低 `low_possession`。

### 11.5 结构性发现：9-lane low_possession 不变量

[snapshot-013 §11.3](snapshot-013-baseline-weakness-analysis.md#113-101-的诊断被强化low_possession-是我们特有所有干预都修不到的-bug) 曾提出三个候选根因：

| 根因候选 | BC 数据验证 |
|---|---|
| (1) obs 层面：single_player 看不到队友具体位置 | ✅ **强化**——BC 见了 teacher 全部轨迹也修不到 |
| (2) 环境层面：2v2 单策略被压扁 | 可能但不充分 |
| (3) 训练分布层面：低占球起始罕见 | ❌ **被 BC 否决**——BC 数据里包含 baseline 在所有 state 的行为，包括低占球 state |

**根因 (1) obs 表征瓶颈**是最可能的解释。

### 11.6 BC 的实际机制：**uniform 减损 + teacher-aligned 起点**

拆解 BC 真正改善的地方：

**改善了的（相对 Pool 018 @290）**：
- 总失败数 102 → **86**（−16%）
- late_collapse 49 → 40（−18%）
- poor_conv 13 → 11（−15%）
- unclear_loss 12 → 10
- win median steps 48 → 43（进攻更干脆利索）

**没改善的**：
- `low_possession` 占比 26.5% → 26.7%（几乎完全相同）
- `low_possession` 绝对数 27 → 23（按比例同步下降，不是选择性修复）

**结论**：BC 的增益来自"**更好的起点让整个训练过程更高效**"，而不是"**修复了某个特定结构缺陷**"。具体：

- BC teacher 让 student 在训练第 0 步就已接近 baseline-level 行为（0.554 起点 vs 0 起点）
- 2335 iter RL fine-tune 在这个高起点上继续爬升，avoiding 了 scratch RL 前几百 iter 的"从零学会踢球"
- 最终 policy **质量更高（WR +0.056）**，但**结构失败模式和其他 lane 相同**

### 11.7 对项目的硬决定

**Performance agent 候选升级**：从 [Pool 018 @290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290)（0.812）切换到 [BC @2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)（**0.842**）。

**Novel concept +5 bonus**：BC 本身是 imitation learning，作业明确列为 bonus 来源，**已 delivered**。

**0.842 是当前天花板**，距离 9/10 稳过的 0.95 仍差 **0.108**。9 个 intervention 都撞在同一面墙——这指向 **obs 空间改造是唯一未试的直接路径**（teammate 位置/速度拼进 obs + 重训 from scratch）。

**Report narrative upgrade**：把 "9 种 intervention 下 `low_possession` 稳定在 22-28%，指向 obs 表征瓶颈" 作为 Analysis and Discussion 30 分的核心素材——**比单纯报告 0.842 SOTA 更有 technical reasoning 价值**。

### 11.8 和 [snapshot-013](snapshot-013-baseline-weakness-analysis.md) 的反馈

[snapshot-013 §11](snapshot-013-baseline-weakness-analysis.md#11-mappo-数据对-10-推论的反馈2026-04-13-后补) 建立了"跨 lane `low_possession` 不变量"诊断，本次 BC 数据为该诊断加上了**9-lane 规模**的实证。snapshot-013 §11.4 当时的预声明判据：

> "BC→RL 路线的 500-ep 胜率 ≥ 0.816 + `low_possession` 占失败比 ≤ 10%"

实测：

- 主判据 ≥ 0.816 → 0.842 ✅ **PASS**
- 机制判据 `low_poss` ≤ 10% → 22.6-26.7% ❌ **FAIL**

**这个 FAIL 本身是高信息量的负结果**，它把"表征瓶颈"假设从**推测**升级为**实证支持**。snapshot-013 的 §11 需要相应 append 更新。
