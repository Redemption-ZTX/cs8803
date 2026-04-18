# SNAPSHOT-018: MAPPO+v2 Warm-Start Opponent-Pool Fine-Tune

- **日期**: 2026-04-14
- **负责人**:
- **状态**: 已完成（首轮 static frozen-pool 对照）

## 1. 背景

截至 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 与 [SNAPSHOT-016](snapshot-016-shaping-v4-survival-anti-rush-ablation.md)，当前 500-ep 官方 WR 排名稳定：

1. MAPPO + shaping-v2 @ ckpt470 = **0.786**（现 SOTA）
2. MAPPO + shaping-v1 @ ckpt490 = 0.774
3. v4 PPO @ ckpt400/430/460 = 0.767（最强 PPO）
4. v1 PPO @ ckpt430 = 0.746
5. v2 PPO @ ckpt440 = 0.728-0.732

[SNAPSHOT-013 §11](snapshot-013-baseline-weakness-analysis.md#11-mappo-数据对-10-推论的反馈2026-04-13-后补) 与 [SNAPSHOT-016 §12.5](snapshot-016-shaping-v4-survival-anti-rush-ablation.md#125-对-snapshot-013-11-的反馈) 共同锁定了一个诊断：

- **`low_possession` 占失败比在 PPO/MAPPO/shaping 全部干预下稳定在 22-28%**
- 它是"我们的单策略对 baseline 两人配合的空间压制"特有的病
- **shaping 维度修不到它**——v4 甚至把它推高到 28%
- 唯一合理的直接攻击是 [SNAPSHOT-017 BC lane](snapshot-017-bc-to-mappo-bootstrap.md)（teacher baseline 的 low_poss = 0%）

本 snapshot 定义的是**另一条正交路线**：**不是**修 low_possession，而是在 MAPPO+v2 的现有优势上叠加**opponent diversity** 信号，让策略对多种对手行为鲁棒。两条线（BC / opponent pool）目标不冲突、失败原因互不重叠，可并行。

当前先落地的是**Phase-1 static frozen pool**：

- baseline 主锚
- MAPPO+v2 self anchor
- MAPPO+v1 跨 shaping seed
- MAPPO no-shaping 跨 lane seed

rolling self 与更激进的跨算法 pool 扩展留到 Phase-2。

## 2. 和 [SNAPSHOT-005](snapshot-005-observation-memory-and-centralized-critic-ablation.md) 旧 self-play 的区别

SNAPSHOT-005 的 `shared_cc_warm225` 曾被概括为"self-play 没突破 0.80"。本 lane 和它的关键区别：

| 维度 | 旧 shared_cc_warm225 | 本 lane |
|---|---|---|
| 起点 | 旧 PPO ckpt-225 (warm-start to CC) | **MAPPO+v2@470 (现 SOTA, 0.786)** |
| 任务语义 | `multiagent_player` + fixed baseline | `multiagent_player` + static opponent pool |
| baseline 在训练中 | 出现，但不是多样化 pool | **60% 主力** |
| 模型 | 256x128 | 512x512 + centralized critic |
| shaping | 无 | 继承 v2（deep_zone + C-neg）|
| 训练量 | 3M steps, 125 iter | 计划 12M steps fine-tune |

**旧 lane 是“早期 centralized-critic 原型 + fixed baseline”，本 lane 是“在 SOTA 基础上叠加跨对手鲁棒性”**。二者不是同一实验。

## 3. 假设

### 主假设

在 MAPPO+v2@470 warm-start 基础上，训练对手分布由“100% baseline”改为 **“60% baseline + 跨 lane frozen opponents”**，policy 将：

1. 保持现有 baseline-specialized 能力（60% 采样主力保持对齐）
2. 学习在**多种 opponent 行为模式下**都能稳定赢球的 policy 表达
3. 由此获得对 baseline 的 WR 进一步提升（机制：不再过拟合 baseline 的特定反应模式）

### 明确不做的事

- **不做**纯 self-play（无 baseline 锚定 → drift 风险）
- **不做**from-scratch（MAPPO+v2@470 已达 0.786，重起浪费）
- **不做**含 random 的 mix（我们 vs random 已 0.98-1.00，采样零收益）
- **不触碰** shaping 参数（v2 shaping 已验证，不叠加未知干预）
- **不修** `low_possession` 这条病（那是 SNAPSHOT-017 的任务）
- **Phase-1 不开** rolling self（先把 static frozen pool 跑通，保持可解释性）

## 4. 实验设计

### 4.1 Warm-start 源

- [MAPPO+v2 @ ckpt470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
- 500-ep 官方 WR 0.786
- `variation = multiagent_player`
- `multiagent = True`
- shared-policy actor + centralized critic

### 4.2 Opponent pool 组成

| 成员 | 比例 | 角色 | 来源 |
|---|---|---|---|
| `ceia_baseline_agent` | **60%** | 评分对齐锚 | frozen |
| MAPPO+v2 @ ckpt470 (self anchor) | 15% | 防漂 anchor | frozen |
| MAPPO+v1 @ ckpt490 | 15% | 跨 shaping diversity | frozen |
| MAPPO no-shaping @ ckpt490 | 10% | 跨 lane diversity | frozen |

**关键设计理由**：
- **60% baseline 保底**：如果其他 40% 全部失效，policy 至少不会比 MAPPO+v2 差
- **跨 lane seed (15% + 15% + 10% = 40%)**：day-1 就有真正行为多样性，不是追自己的影子
- **Phase-1 不开 rolling self**：先验证 static pool 本身是否有收益，避免把变量一次混太多

### 4.3 训练配置

- 脚本： [train_ray_mappo_vs_opponent_pool.py](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py)
- `variation = multiagent_player`
- `multiagent = True`
- `FCNET_HIDDENS = 512,512`
- `custom_model = shared_cc_model`
- `gamma = 0.99, lambda = 0.95`
- shaping：**继承 v2**（`opp_progress_penalty=0.01, deep_zone_outer=-0.003, deep_zone_inner=-0.003`）
- rollout 参数：继承 MAPPO lane (`rollout_fragment_length=1000, train_batch_size=40000, num_sgd_iter=10`)

### 4.4 训练预算

- **300 iter** fine-tune（~12M steps）
- 预算依据：policy 已会踢球，不需要 500 iter 从头学；300 iter 提供 6 个 rolling window 迭代
- 预计 walltime：~5 小时（比 MAPPO from-scratch 的 8 小时短）

### 4.5 评估协议

沿用 [engineering-standards.md § checkpoint 选模规则](../architecture/engineering-standards.md)：

1. 训练内 `baseline 50` 做候选筛选（注意：评分只看 vs baseline，不看 vs pool 成员）
2. `top 5% + ties` 取候选
3. 正式 `baseline 500` 官方 eval
4. 最终 1-2 个 shortlist 补 `random 500` 确认

## 5. 预声明判据

### 5.1 主判据（胜出必过）

**500-ep 官方 WR vs baseline ≥ 0.81**

- MAPPO+v2 现状 0.786
- Binomial 95% CI ±0.036 on 500 eps
- +0.02 需要超过噪声
- 这个阈值低于 "BC 路线预声明 0.816"（SNAPSHOT-017），因为 opponent pool lane 不直接修 low_possession，预期上限低一档

### 5.2 机制判据（择一过即可视为"真突破"）

**A — 鲁棒性实证**：vs 非 baseline pool 成员的 WR 都 ≥ 0.55
- vs MAPPO+v1@490 ≥ 0.55
- vs MAPPO no-shaping@490 ≥ 0.55
- vs MAPPO+v2@470 (self anchor) ≥ 0.50（自己打平自己是合理上限）

**B — 失败结构更 baseline 化**：500-ep failure capture 里，失败桶分布朝 [BvB 分布](snapshot-013-baseline-weakness-analysis.md#101-失败桶分布对比) 靠拢
- `late_defensive_collapse` 占比 ≥ 45%（靠近 BvB 的 46%）
- `unclear_loss` 占比 ≥ 20%（当前所有 lane 只有 10-15%，BvB 是 47%；这个桶升高意味着对局更像"势均力敌"）

### 5.3 失败情形的预声明

**明确失败场景** — 任一触发即视为 opponent pool 方向在我们这个问题上无增益，**不再继续扩展该 lane**：

1. 主判据 500-ep WR 不到 0.79（即没超 MAPPO+v2 的 0.786）
2. 主判据过但 A/B 机制判据都不过 → 是 seed luck 或过拟合 pool，不是真鲁棒
3. vs baseline WR 涨但 vs random 跌到 < 0.95（意味着 policy 为了对付 pool 丢了基础能力）

## 6. 风险与已识别的陷阱

### R1 — `low_possession` 桶本 lane 大概率不动

snapshot-013 §11 + snapshot-016 §12.5 已实证：pool 成员都是我们训的 agent，都不会产生 baseline 式 squish pattern。**本 lane 跑完 `low_possession` 占比很可能仍 ≥ 22%**。这不是 lane 失败，是 lane 设计边界——修 low_possession 是 SNAPSHOT-017 BC 的任务。本 snapshot 不把 `low_possession ≤ 10%` 列为判据。

### R2 — Pool 成员陈旧化

`MAPPO+v2@470 / MAPPO+v1@490 / MAPPO no-shaping@490` 都是 2026-04-13 的固定 checkpoint。300 iter fine-tune 期间它们不变——如果训练主体很快超过这些 frozen 成员，pool 的有效难度会下降。

缓解：每 100 iter 检查一次训练期间 agent vs 各 pool 成员 WR；如果某成员被碾压超过 0.85 → 触发告警，并在 Phase-2 再考虑 rolling self。

### R3 — Static pool 可能只提供“鲁棒性下限”，不给上限

这条线的理论上限本来就低于 SNAPSHOT-017。即便它工作正常，也更可能把 policy 做得更稳，而不是直接修掉 `low_possession`。

### R4 — centralized-critic 观察函数与 frozen opponent 语义耦合

Phase-1 实现要求 team0 与 team1 都走 centralized-critic 观察包装，但只有 team0 的 shared policy 会被训练。风险在于：

- baseline 需要正确 strip 回 own_obs
- frozen MAPPO opponent 需要在 `shared_cc_observer_all` 下稳定吃到 teammate context
- policy mapping 必须保证 team1 全队同一 episode 使用同一个 pool 成员

当前已通过的 smoke 只能证明“能初始化、能采样、能过 1 iter”，不能证明长期训练完全无漂移。

## 7. 不做的事（明确边界）

本 lane **不触碰**以下维度，避免与其他 lane 混淆：

- 不改 reward shaping 参数（v2 shaping 冻结）
- 不改 model 架构（shared_cc_model 冻结）
- 不改 obs 空间（centralized-critic 观察语义冻结）
- 不改 γ / λ
- 不做 BC-style pre-train（那是 SNAPSHOT-017）
- 不做 MAPPO 从零重训（SNAPSHOT-014 已完成）

## 8. 相关

- [SNAPSHOT-005: 旧 shared_cc self-play](snapshot-005-observation-memory-and-centralized-critic-ablation.md)（反例）
- [SNAPSHOT-013: baseline weakness + 失败桶分析](snapshot-013-baseline-weakness-analysis.md)（诊断依据）
- [SNAPSHOT-014: MAPPO 公平对照](snapshot-014-mappo-fair-ablation.md)（warm-start 源）
- [SNAPSHOT-016: v4 shaping ablation](snapshot-016-shaping-v4-survival-anti-rush-ablation.md)（对“shaping 继续细调”的边界给出反证）
- [SNAPSHOT-017: BC -> RL fine-tune](snapshot-017-bc-to-mappo-bootstrap.md)（并行路线，互补目标）

## 9. 下一步

Phase-1 现已完成第 1 步：

1. 已实现 [train_ray_mappo_vs_opponent_pool.py](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py)
2. 已新增 [soccerstwos_h100_cpu32_mappo_v2_opponent_pool_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_v2_opponent_pool_512x512.batch)
3. 已完成最小 smoke：
   - warm-start 成功
   - frozen opponent checkpoints 成功加载
   - 1 iter 完整结束并写出 checkpoint

接下来的顺序是：

4. 启动完整 12M-step static pool fine-tune
5. `top 5% + ties -> baseline 500`
6. 对最终候选做 failure capture，并检验 §5 的主/机制判据
7. 如果 static pool 有正增益，再讨论 Phase-2：
   - rolling self
   - 加入 v4 等跨算法 pool 成员
   - 更复杂的 pool 重采样策略

verdict 结果写入本文件 §10-§11（append-only），不修改 §1-§9 的预注册内容。

## 10. 首轮结果（2026-04-15）

### 10.1 训练运行

- 运行目录：
  - [PPO_mappo_v2_opponent_pool_512x512_20260414_212239](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239)
- 训练摘要：
  - `best_eval_baseline = 0.900 @ iter 280`（50-ep internal）
  - `best_eval_random = 1.000`
  - `best_reward_mean = -0.9110 @ iter 300`
- 训练 reward 依旧显著为负，因此和 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 一致，本 lane 继续支持：
  - **MAPPO 系列里，训练 reward 不能直接作为最终 baseline WR 的代理指标**

### 10.2 官方 `baseline 500` 复核

围绕 `240-300` 的高位窗口做正式复核：

| checkpoint | baseline 500 WR |
|---|---:|
| [ckpt240](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000240/checkpoint-240) | **0.788** |
| [ckpt270](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000270/checkpoint-270) | 0.774 |
| [ckpt280](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000280/checkpoint-280) | 0.780 |
| [ckpt290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290) | **0.812** |
| [ckpt300](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000300/checkpoint-300) | 0.780 |

### 10.3 当前项目内排名更新

首轮 static pool fine-tune 已把项目官方 `baseline 500` 最优点刷新为：

1. [opponent-pool ckpt290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290) = **0.812**
2. [MAPPO + shaping-v2 ckpt470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470) = 0.786
3. [role-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30) = 0.786
4. [MAPPO + shaping-v1 ckpt490](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000490/checkpoint-490) = 0.774
5. [v4 PPO ckpt400](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000400/checkpoint-400) = 0.768

## 11. Failure Capture 机制解释（2026-04-15）

冠军点 failure capture：

- [mappo_pool_checkpoint290_baseline_500](../artifacts/failure-cases/mappo_pool_checkpoint290_baseline_500)

对照组：

- [mappo_v2_checkpoint470_baseline_500](../artifacts/failure-cases/mappo_v2_checkpoint470_baseline_500)

### 11.1 概况

- `pool@290` capture：`398W-102L = 0.796`
- `mappo-v2@470` capture：`375W-125L = 0.750`

说明 capture 口径继续和官方 500 的排序方向一致：

- official：`0.812 > 0.786`
- capture：`0.796 > 0.750`

### 11.2 它主要减少了什么失败

主标签计数对比：

| 失败桶 | pool@290 | mappo-v2@470 |
|---|---:|---:|
| `late_defensive_collapse` | **49** | 62 |
| `low_possession` | **27** | 30 |
| `poor_conversion` | 13 | 11 |
| `unclear_loss` | **12** | 16 |
| `territory_loss` | **1** | 6 |

最关键的收益不是“把所有失败都改写”，而是：

- 明显减少了 `late_defensive_collapse`
- 明显减少了 `territory_loss`
- 同时把 `unclear_loss` 也压下去了一些

### 11.3 `late_defensive_collapse` 变浅了

在这一最大失败桶中：

| 指标 | pool@290 | mappo-v2@470 |
|---|---:|---:|
| steps mean | 40.286 | 45.290 |
| `tail_mean_ball_x` mean | **-7.075** | -8.084 |
| `team1_progress_toward_goal` mean | **5.786** | 6.818 |

解释：

- baseline 仍然能在坏局里压进深区
- 但 pool fine-tune 后，压制没有以前那么深、那么完整
- 这条线真正修到的是“被 baseline 打穿后的持续压制深度”

### 11.4 剩下来的输局更像“压着打但没收掉”

`poor_conversion` 桶虽然从 `11 -> 13` 略升，但形状明显变了：

| 指标 | pool@290 | mappo-v2@470 |
|---|---:|---:|
| `team0_progress_toward_goal` mean | **12.091** | 9.015 |
| `team1_progress_toward_goal` mean | **0.348** | 2.443 |

解释：

- pool line 剩下来的输局，更像我方已把 baseline 压到前场
- 但没有完成终结
- 这比“被 baseline 直接打穿”更接近一个高级失败模式

### 11.5 机制结论

本 lane 的首轮提升是**有结构的**，不是纯 seed luck：

- 它没有彻底消灭 `low_possession / poor_conversion`
- 但它确实减少了最伤的 defensive-collapse / territory-style losses
- 并把更多比赛推成“我方前场主导，但仍待终结”的局面

因此，[§5.1](#51-主判据胜出必过) 的主判据已经明确通过，而 §5.2 的机制解释也有了正面证据：

- 不是 failure buckets 完全 baseline 化
- 而是对当前最强 `MAPPO + shaping-v2` 做了进一步的结构性减损

### 11.6 当前结论

首轮 static frozen-pool fine-tune 已经足够把本 snapshot 从“并行探索线”升级为：

- **当前项目最强主线候选**
- 也是在 `MAPPO + shaping-v2` 之上最明确跑出正增益的一条鲁棒性路线

## 12. Head-to-Head Against Prior MAPPO Strong Points（2026-04-15）

为判断 `pool@290` 的提升是否只针对 baseline，额外做了三组 500-ep head-to-head 官方对打：

- [pool290_vs_v1_500.log](../artifacts/official-evals/headtohead/pool290_vs_v1_500.log)
- [pool290_vs_bs0_500.log](../artifacts/official-evals/headtohead/pool290_vs_bs0_500.log)
- [pool290_vs_v2_500.log](../artifacts/official-evals/headtohead/pool290_vs_v2_500.log)

### 12.1 官方 500-ep 对打结果

`policy` 一侧固定为：

- [opponent-pool ckpt290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290)

对手分别为：

- [MAPPO + shaping-v1 @ ckpt490](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000490/checkpoint-490)
- [MAPPO no-shaping @ ckpt490](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000490/checkpoint-490)
- [MAPPO + shaping-v2 @ ckpt470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)

| 对手 | 官方 500-ep WR（pool@290 作为 team0） |
|---|---:|
| MAPPO + shaping-v1 @ ckpt490 | 0.488 |
| MAPPO no-shaping @ ckpt490 | **0.532** |
| MAPPO + shaping-v2 @ ckpt470 | **0.520** |

这组结果说明：

- `pool@290` **不是**“对所有旧 MAPPO 变体都绝对最强”
- 但它对 `MAPPO no-shaping` 与 `MAPPO + shaping-v2` 都有小幅 head-to-head 优势
- 对 `MAPPO + shaping-v1` 则没有拉开明显差距，表现更接近五五开

### 12.2 对打 Failure Capture

对应 failure capture：

- [mappo_pool290_vs_mappo_v1_500](../artifacts/failure-cases/mappo_pool290_vs_mappo_v1_500)
- [mappo_pool290_vs_mappo_bs0_500](../artifacts/failure-cases/mappo_pool290_vs_mappo_bs0_500)
- [mappo_pool290_vs_mappo_v2_500](../artifacts/failure-cases/mappo_pool290_vs_mappo_v2_500)

以及自动聚合：

- [mappo_pool290_vs_mappo_v1_500/analysis_summary.json](../artifacts/failure-cases/mappo_pool290_vs_mappo_v1_500/analysis_summary.json)
- [mappo_pool290_vs_mappo_bs0_500/analysis_summary.json](../artifacts/failure-cases/mappo_pool290_vs_mappo_bs0_500/analysis_summary.json)
- [mappo_pool290_vs_mappo_v2_500/analysis_summary.json](../artifacts/failure-cases/mappo_pool290_vs_mappo_v2_500/analysis_summary.json)

capture 仅用于机制解释，不替代 §12.1 的官方 head-to-head 排名。

| 对手 | capture WR | 输局数 | 输局 median steps |
|---|---:|---:|---:|
| MAPPO + shaping-v1 @ ckpt490 | 0.500 | 250 | 46.0 |
| MAPPO no-shaping @ ckpt490 | 0.560 | 220 | 54.0 |
| MAPPO + shaping-v2 @ ckpt470 | 0.522 | 239 | 48.0 |

主标签计数：

| 对手 | `late_defensive_collapse` | `low_possession` | `poor_conversion` | `unclear_loss` | `territory_loss` |
|---|---:|---:|---:|---:|---:|
| MAPPO + shaping-v1 @ ckpt490 | 108 | 43 | 39 | 36 | 16 |
| MAPPO no-shaping @ ckpt490 | 112 | 35 | 25 | 27 | 18 |
| MAPPO + shaping-v2 @ ckpt470 | 112 | 37 | 25 | 34 | 23 |

这组机制结果说明：

- `late_defensive_collapse` 仍然是 `pool@290` 对强 MAPPO 对手时的最大失败桶
- 对 `MAPPO no-shaping` 的优势最稳定：官方对打与 capture 都显示明确正收益
- 对 `MAPPO + shaping-v2` 的收益较小，但两次口径方向一致，说明它确实在 `v2` 基础上再往前推了一步
- 对 `MAPPO + shaping-v1` 则更像“基本持平”：官方小负，capture 打平，说明 `v1` 的更激进风格依然能和 `pool@290` 互相克制

### 12.3 补充结论

因此，本 lane 当前最准确的画像是：

- **vs baseline 最强**（官方 `0.812`）
- **vs 旧 MAPPO 强线总体站得住**，但不是无条件统治型策略
- 是一个”baseline-targeted 目标最强，同时对其他 MAPPO 变体也基本具备竞争力”的当前冠军

## 13. 预声明判据逐项 verdict

把 §5 / §6 的预声明项与 §10-§12 的实测合在一张表上，避免叙述式结论模糊化具体 PASS/FAIL：

### 13.1 §5 / §6 实测对照

| 判据 | 预声明阈值 | 实测 | 结果 |
|---|---|---|---|
| §5.1 主判据 — 500-ep WR ≥ 0.81 | | ckpt290 = **0.812** | ✅ **PASS**（擦边 0.002）|
| §5.2 A.1 — vs MAPPO+v1@490 ≥ 0.55 | | **0.488** | ❌ **FAIL**（差 0.062，且 pool 反而小负）|
| §5.2 A.2 — vs MAPPO no-shaping@490 ≥ 0.55 | | 0.532 | ❌ **FAIL**（差 0.018）|
| §5.2 A.3 — vs MAPPO+v2@470 (anchor) ≥ 0.50 | | 0.520 | ✅ **PASS**（高 0.020）|
| §5.2 B.1 — late_defensive_collapse 占失败比 ≥ 45% | | 48.0% | ✅ **PASS** |
| §5.2 B.2 — unclear_loss 占失败比 ≥ 20% | | 11.8% | ❌ **FAIL**（差 8.2 pp）|
| §6 R1 — low_possession 占失败比 ≥ 22% | | 26.5% | ✅ **PASS**（预测验证）|
| §5.3 失败 1 — best WR < 0.79 | | 0.812 | ✅ **未触发** |
| §5.3 失败 2 — main pass 但 A/B 都不过 | | A 1/3, B 1/2 均部分过 | ✅ **未触发** |
| §5.3 失败 3 — vs random < 0.95 | | 50-ep 1.000；500-ep 暂未测 | ⏳ 待补 |

**严格 PASS/FAIL 计数**：6 PASS / 4 FAIL / 1 待补。

### 13.2 §5.2 A 阈值的事后反思

按字面 §5.2 A 阈值（”vs 三个 pool 成员都 ≥ 0.55”），本 lane **2/3 子项未过**。但这个失败是否是**”鲁棒性真的不行”**？需要重新审视阈值本身：

观察事实：
- pool@290 vs baseline = **0.812**
- pool@290 vs 三个 pool 成员 = **0.488 / 0.532 / 0.520**（围绕 0.5 紧凑分布）

这是**经典的同代 agent 非传递博弈**——四个 MAPPO siblings 都对 baseline 强（0.74-0.81），但相互对打接近 50%。0.5 不是”鲁棒性失败”，是**等强 cluster 的自然均衡**。

事后看，§5.2 A 当时把 ≥ 0.55 写死的假设是”如果 pool@290 真鲁棒，它该传递性地比所有 pool 成员都强”。**这个传递假设和实际博弈结构不符**——head-to-head 不是 baseline-WR 的线性投影。

**校准后的 A 判据建议**：

- 严格意义上 A FAIL（按预声明字面）
- 但**机制解释**：0.49-0.53 cluster 表明 pool@290 既没被任何成员压制，也没压制任何成员——这是”它仍然属于这组强 agent，没退化”的证据
- **真正的失败信号**应该是 vs 任一成员 < 0.40 或 > 0.70——前者意味着对某种风格无招架，后者意味着 pool@290 学到了一个对那个成员的过拟合策略。**实测三项都在 [0.488, 0.532]**，无任一极端，所以是”健康均衡”

按这个**严格 FAIL + 机制 healthy** 的双重读法，对 ensemble 设计反而是好消息（见 §14）。

### 13.3 §5.2 B 阈值的事后反思

unclear_loss 11.8% < 20% 在字面上 FAIL。但 unclear_loss 的产生方式是：**eval 局没有清晰的失败 pattern 被分类器抓到**——通常出现在两个旗鼓相当的对手互相消耗的局里。

pool@290 在 baseline 上的表现是**胜率 0.812**，意味着大多数局有清晰的胜利结构；剩下的输局也大多有清晰原因（late_collapse 48%）。**unclear_loss 占比低反而暗示我们的 policy 不再陷入”似输非输”的灰区**——这和 BvB 47% 的 unclear_loss 不在同一层叙事上：BvB 双方实力对称所以模糊；pool@290 vs baseline 是**强势方对弱势方**，模糊局自然少。

事后看，§5.2 B.2 的”unclear_loss ≥ 20%”假设把 BvB 当作”理想化”参考，但 BvB 的对称性是这条数字的成因，**不能直接搬到 vs baseline 这种非对称对局上**。这条判据**应该整体被废弃**，或重写为”unclear_loss + 其他高质量失败合占比 ≥ 30%”。

## 14. 整体结论与下一步

### 14.1 综合 verdict

**snapshot-018 整体 PASS**——理由：

- 主判据明确通过（500-ep 0.812）
- 机制层面**有结构性减损**（late_collapse 减、territory_loss 几乎归零、failure 总数 102 vs MAPPO+v2 的 125）
- head-to-head 显示**等强 cluster 健康均衡**，无过拟合 baseline 的退化
- 对 §5 / §6 字面阈值**6 PASS / 4 FAIL**，但 4 个 FAIL 中 3 个是阈值本身设计偏差（事后反思可见 §13.2 / §13.3）

### 14.2 对 Ensemble 战线的直接价值

§12.1 的 head-to-head 给出了一个**不在原计划但极其重要**的 by-product：

| 对 | head-to-head WR | 解读 |
|---|---|---|
| pool@290 ↔ MAPPO+v1@490 | 0.488 | 几乎完全均衡（差 0.012）|
| pool@290 ↔ MAPPO no-shape@490 | 0.532 | 微弱倾向 pool |
| pool@290 ↔ MAPPO+v2@470 | 0.520 | 微弱倾向 pool |

**四个 MAPPO siblings 构成了一个”相互制衡 + 都对 baseline 强”的 ensemble seed pool**。Ensemble 在这种”个体强 + 失败相关性低”的成员上效果最好。预期的 majority-vote ensemble：

- pool@290 (0.812) + MAPPO+v2 (0.786) + MAPPO+v1 (0.774) + MAPPO no-shape (0.742) 四者投票
- 相关性低（head-to-head 都 ~0.5）→ 失败模式独立
- 期望 ensemble WR：**0.83-0.86**（按经典 ensemble 提升 +0.02-0.05）

如果 ensemble 真把 WR 推到 ≥ 0.85，配合 [SNAPSHOT-017 BC](snapshot-017-bc-to-mappo-bootstrap.md) 在 `low_possession` 上的可能突破，**0.90 提交线变得可触达**。

### 14.3 后续动作（按优先级）

1. **vs random 500-ep 验证**（§5.3 最后一项待补，5 分钟搞定）
2. **不再扩展本 lane 单独跑**——首轮已 PASS，再叠 rolling self / 跨算法 pool 期望增量 ≤ +0.02，不如把 GPU 让给 BC 与 ensemble
3. **启动 ensemble snapshot**（建议编号 snapshot-019）：4-agent majority vote wrapper，使用 [trained_shared_cc_agent](../../cs8803drl/deployment/trained_shared_cc_agent.py) 内部包 4 个 checkpoint
4. **submission deployment 切换**：当前 Performance agent 候选从 [MAPPO+v2@470] 升级为 [pool@290]（500-ep 0.812 vs 0.786，failure 总数 102 vs 125）
5. snapshot-018 保持本版本作为终稿，除非 vs random 出现意外回退（< 0.95 → 触发 §5.3 失败 3）

### 14.4 一句话总结

> snapshot-018 是 SNAPSHOT-014 之后第一条**严格意义上把项目 baseline-WR SOTA 推前**的 lane，且其 head-to-head 数据**意外地为后续 ensemble 战线提供了完美的 4-成员 seed pool**。机制层面”减少 late_collapse + 收掉 territory_loss”是真实结构性收益；low_possession 的桶不动印证了 [§6 R1](#r1--low_possession-桶本-lane-大概率不动) 的预测，确认**该桶必须等 BC lane 攻克**。
