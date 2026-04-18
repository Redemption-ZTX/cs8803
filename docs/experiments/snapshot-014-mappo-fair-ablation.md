# SNAPSHOT-014: MAPPO / Centralized-Critic 公平对照

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 已完成（首轮对照）

## 1. 背景

在 [SNAPSHOT-005](snapshot-005-observation-memory-and-centralized-critic-ablation.md) 中，`shared_central_critic` 方向曾被和 `summary observation / LSTM` 一起归纳为“没有突破 `~0.80` 平台”。

2026-04-13 复核后确认：

- 旧 `shared_cc_warm225` run 确实拿到过 `0.80 @ 200`
- 但它使用的是：
  - `multiagent_player`
  - `256x128`
  - `reward_shaping = off`
  - `warm225`
  - `3M steps`
- 它和当前 `team_vs_baseline + 512x512 + 500 iter` 主线不是公平对照

因此，旧 run 不能作为“MAPPO / centralized critic 已被否定”的证据。

本 snapshot 的目标，就是在与当前主线尽量公平的前提下，重新验证：

> centralized critic / MAPPO 到底能不能提供真正的性能增益？

## 2. 实验设计

### 2.1 训练入口与 batch

- 训练脚本：[train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)
- 首轮 batch：
  - [mappo no shaping](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_vs_baseline_noshaping_512x512.batch)
  - [mappo shaping-v1](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v1_512x512.batch)
  - [mappo shaping-v2](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v2_512x512.batch)

### 2.2 共同设置

- 对手固定为 baseline
- `512x512`
- `500 iter / 20M steps`
- 正式选模沿用：
  - `top 5% + ties`
  - `baseline 500`
  - `random 500` 只对最终 shortlist 再补

### 2.3 三条线分别回答什么

1. `MAPPO + no shaping`
   - centralized critic 的裸效应
2. `MAPPO + shaping-v1`
   - centralized critic 与当前最熟悉的 baseline-targeted shaping 是否协同
3. `MAPPO + shaping-v2`
   - centralized critic 与已知“更稳但不上限”的防守型 shaping 是否协同

## 3. 首轮实际运行

### 3.1 no shaping

- run: [PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113)
- 训练内 best：
  - [checkpoint-350](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000350/checkpoint-350)
  - `39/50 = 0.78`

### 3.2 shaping-v1

- run: [PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616)
- 训练内 best：
  - [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000490/checkpoint-490)
  - `40/50 = 0.80`

### 3.3 shaping-v2

- run: [PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545)
- 训练内 best：
  - [checkpoint-290](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000290/checkpoint-290)
  - `43/50 = 0.86`

注意：
- 这个 `0.86` 后来被官方 `500` 证明是被 `50` 局尖峰夸大了
- 但它不是假信号，因为同一条线上 `460/470` 仍然给出更强的正式结果

## 4. 正式复核

### 4.1 no shaping：top 5% + ties -> baseline 500

复核点：

- [checkpoint-350](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000350/checkpoint-350): `0.708`
- [checkpoint-360](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000360/checkpoint-360): `0.712`
- [checkpoint-450](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000450/checkpoint-450): `0.742`
- [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000490/checkpoint-490): `0.742`

最优：

- [checkpoint-450](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000450/checkpoint-450)
- [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/MAPPOVsBaselineTrainer_Soccer_9be63_00000_0_2026-04-13_03-01-35/checkpoint_000490/checkpoint-490)
- `0.742`

### 4.2 shaping-v1：top 5% + ties -> baseline 500

复核点：

- [checkpoint-360](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000360/checkpoint-360): `0.716`
- [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000490/checkpoint-490): `0.774`
- [checkpoint-320](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000320/checkpoint-320): `0.702`
- [checkpoint-410](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000410/checkpoint-410): `0.754`
- [checkpoint-430](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000430/checkpoint-430): `0.746`
- [checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000470/checkpoint-470): `0.750`
- [checkpoint-480](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000480/checkpoint-480): `0.756`

最优：

- [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/MAPPOVsBaselineTrainer_Soccer_e97b0_00000_0_2026-04-13_03-46-42/checkpoint_000490/checkpoint-490)
- `0.774`

### 4.3 shaping-v2：top 5% + ties -> baseline 500

复核点：

- [checkpoint-290](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000290/checkpoint-290): `0.752`
- [checkpoint-460](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000460/checkpoint-460): `0.770`
- [checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470): `0.786`

最优：

- [checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
- `0.786`

## 5. 与既有主线的对比

当前关键基线：

- 旧 PPO warm-start 强点 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225): `0.764`
- 旧 role 强点 [checkpoint-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30): `0.786`
- 纯 PPO shaping 强点 [v1 checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000430/checkpoint-430): `0.746`

MAPPO 首轮结果：

- `MAPPO + no shaping`: `0.742`
- `MAPPO + shaping-v1`: `0.774`
- `MAPPO + shaping-v2`: `0.786`

因此：

1. `MAPPO` 裸效应就已经接近当前 PPO shaping 最强点；
2. shaping 叠在 MAPPO 上仍然有用；
3. `MAPPO + shaping-v2` 已经追平当前项目里的最好官方 `500` 成绩。

## 6. Failure Capture 机制分析

### 6.1 采样说明

本轮做了三组 `baseline 500` failure capture：

- [mappo_bs0_checkpoint490_baseline_500](artifacts/failure-cases/mappo_bs0_checkpoint490_baseline_500)
- [mappo_v1_checkpoint490_baseline_500](artifacts/failure-cases/mappo_v1_checkpoint490_baseline_500)
- [mappo_v2_checkpoint470_baseline_500](artifacts/failure-cases/mappo_v2_checkpoint470_baseline_500)

注意：

- 这三组采样主要用于**诊断失败模式**
- 不用于替代官方 `500` 的最终排名
- 例如 `v1 / v2` 在这轮 capture 中都得到 `0.750`，但最终强弱仍以上面的官方 `500` 为准

### 6.2 no shaping：拖得更久，但不够高效

在 [mappo_bs0_checkpoint490_baseline_500](artifacts/failure-cases/mappo_bs0_checkpoint490_baseline_500) 中：

- primary `late_defensive_collapse = 59/129`
- `low_possession = 34/129`
- `unclear_loss = 19/129`
- 输局 `median steps = 34`

解读：

- 裸 MAPPO 已经能把很多局拖长
- 但仍有较高比例的 `low_possession`
- 且不少败局停留在 `unclear_loss`，说明“拖住了，但没稳定转成赢局”

### 6.3 shaping-v1：更激进，也更容易快输

在 [mappo_v1_checkpoint490_baseline_500](artifacts/failure-cases/mappo_v1_checkpoint490_baseline_500) 中：

- primary `late_defensive_collapse = 63/125`
- `poor_conversion = 15/125`
- `low_possession = 30/125`
- 输局 `median steps = 27`

解读：

- `shaping-v1` 比 `no shaping` 更积极
- 但坏局结束得更快
- `poor_conversion` 也更明显

它更像是在推进和压上上更激进，但代价是快攻快崩。

### 6.4 shaping-v2：三条里最平衡

在 [mappo_v2_checkpoint470_baseline_500](artifacts/failure-cases/mappo_v2_checkpoint470_baseline_500) 中：

- primary `late_defensive_collapse = 62/125`
- `low_possession = 30/125`
- `poor_conversion = 11/125`
- 输局 `median steps = 34`

和另外两条比：

- 它不像 `no shaping` 那样把很多失败留在 `unclear_loss`
- 也不像 `v1` 那样容易快输和快攻快崩
- `poor_conversion` 相比 `v1` 有所下降

所以 `v2` 当前最像：

- 保住了 MAPPO 的协调收益
- 同时没有把策略推成过度激进

### 6.5 机制判据的真实结论

若按 [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md) 预想的“failure buckets 是否 baseline 化”来判断：

- `low_possession`
- `poor_conversion`

这两类失败并没有被完全修掉。

因此，本轮更准确的结论不是：

> MAPPO 已经完全修好协调问题

而是：

> MAPPO 明显改善了主线表现；而 `shaping-v2` 提供了目前最好的“稳定性与有效性平衡”。

## 7. 结论

### 7.1 主判据

首轮主判据成立。

因为：

- `MAPPO + shaping-v1 = 0.774`，已经超过旧 `checkpoint-225 = 0.764`
- `MAPPO + shaping-v2 = 0.786`，已经超过纯 PPO shaping 主线，并追平旧 role 强点

### 7.2 更细的判断

1. `MAPPO` 不是伪信号，centralized critic 这条线真实有效。
2. shaping 叠在 MAPPO 上仍然有增益。
3. `MAPPO + shaping-v2` 是当前 Mainline-B 的最佳候选。
4. 机制层面仍未彻底“baseline 化”：
   - `low_possession`
   - `poor_conversion`
   依然存在，因此这条线还有继续改进空间。

## 8. 下一步

当前最自然的后续不是继续在 `MAPPO no shaping / v1 / v2` 之间反复摆动，而是：

1. 保留 [checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470) 作为 Mainline-B 当前 best
2. 继续推进 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) 的 BC 方向
3. 优先考虑 `BC -> MAPPO`，而不是直接再开更多单纯 shaping 变体

也就是说，首轮公平 MAPPO 对照给出的答案是：

> centralized critic 值得保留，而且当前最好的组合是 `MAPPO + shaping-v2`。

## 9. 定量 Ablation 合表与机制判据

### 9.1 2×2 胜率 ablation matrix (500-ep 官方)

| | no shaping | v1 shaping | v2 shaping |
|---|---|---|---|
| **PPO** | (Base-D ~0.70) | 0.746 @ ckpt430 | 0.728–0.732 @ ckpt440 |
| **MAPPO** | 0.742 @ ckpt490 | **0.774 @ ckpt490** | **0.786 @ ckpt470** |

行间增量：`PPO → MAPPO` 贡献 **+0.03~0.05**（在任何 shaping 配置上都成立）
列间增量：`no shaping → shaping` 贡献 **+0.01~0.05**（PPO 下更明显，MAPPO 下更平）

**两个维度近似正交可叠加**。从 PPO baseline (~0.70) 到 MAPPO+v2 (0.786) = 累计 **+0.086**。

### 9.2 Failure bucket 横向对比（500-ep）

每一格为该 bucket 在 500 局里的**绝对失败数**（括号内为占失败总数的百分比）：

| 失败类型 | v1 PPO (157 losses) | v2 PPO (134 losses) | MAPPO no-shape (130) | MAPPO v1 (126) | MAPPO v2 (125) |
|---|---|---|---|---|---|
| **late_defensive_collapse** | **77** (49%) | 70 (52%) | **59** (46%) | 63 (50%) | 62 (50%) |
| **low_possession** | 37 (24%) | 29 (22%) | **34 (26%)** | 30 (24%) | 30 (24%) |
| poor_conversion | 19 (12%) | 11 (8%) | 11 (9%) | 15 (12%) | 11 (9%) |
| unclear_loss | 19 (12%) | 17 (13%) | 19 (15%) | 13 (10%) | 16 (13%) |
| territory_loss | 4 (3%) | 4 (3%) | 4 (3%) | 1 (1%) | 6 (5%) |
| opponent_forward_progress | 1 (1%) | 3 (2%) | 2 (2%) | 3 (2%) | 0 (0%) |

### 9.3 [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md) §10.5 预声明机制判据的 verdict

预声明：

> "机制判据 A（关键）: `low_possession` 在失败桶里占比从 22% → ≤ 10%"
> "如果只有 WR 涨但 `low_possession` 没降 → MAPPO 只是在表面上赢球，没修根本协调问题"

实测：

| Lane | low_possession 占失败% |
|---|---|
| v1 PPO | 24% |
| v2 PPO | 22% |
| MAPPO no-shape | **26%** |
| MAPPO v1 | 24% |
| MAPPO v2 | 24% |

**判据 FAIL**——五条 lane 全在 22-26% 区间，MAPPO no-shape 甚至比 PPO 还高。**credit-assignment 假设被数据否决**。

### 9.4 MAPPO 实际修复的机制

看绝对失败数变化：

- `late_defensive_collapse`：v1 PPO **77** → MAPPO no-shape **59**（−18 局，-23%），其他 MAPPO lane 也在 62-63
- `low_possession`：v1 PPO **37** → MAPPO no-shape **34**（-3 局，几乎不动）
- 其他桶变化 ≤ 4 局

**MAPPO 的 +0.04 胜率增量，几乎全部来自减少后防崩塌。** 失败分布的**形状**不变，只是**平移**了一点。

这与 Yu 2022 SMAC 观察一致：**centralized critic 主要价值不在"协调"而在"稀有高价值状态下降低 advantage 方差"**——后防紧急状态罕见但关键，CC 在这些状态上给出更干净的梯度，policy 学到更稳的防守响应。

### 9.5 Loss-episode median steps

| Lane | loss median |
|---|---|
| v1 PPO | 31 |
| v2 PPO | 35 (+4) |
| MAPPO no-shape | 34 |
| MAPPO v1 | 27 (−4) |
| MAPPO v2 | 34 |

**v2 PPO 当时的"降节奏 +4 步"效应在 MAPPO lanes 上不一致**（MAPPO v1 甚至更短）——这进一步说明 MAPPO 的机制不是"延迟失败"而是"直接减少失败频率"。

### 9.6 新暴露的关键问题：`low_possession` 是跨 lane 不变量

五条 lane 的 `low_possession` 占比恒定在 22-26%，baseline-vs-baseline（[snapshot-013 §10.1](snapshot-013-baseline-weakness-analysis.md#101-失败桶分布对比)）里这个桶是 **0%**。

结合预声明判据 9.3 FAIL：**`low_possession` 不是 credit-assignment 的病，否则 CC 该起作用。**

三个更可能的原因：

1. **obs 层面**：single_player 模式下 policy 看不到队友的具体位置和意图
2. **环境层面**：2v2 设置下单策略控双人时偶尔被两个 baseline 在空间上"压扁"
3. **训练分布层面**：低占球率的起始状态在 on-policy rollout 里罕见，policy 没学过怎么抢回球

这三个原因 **PPO/MAPPO/shaping 都修不到**——需要换维度：

- **obs 改造**（teammate 信息放进 obs）
- **BC/DAgger**（baseline 的 low_possession = 0%，可以把"不放弃抢球"的行为模式直接传染给 student）
- **Curriculum 改训练分布**（人为提高低占球率初始局面的比例）

[SNAPSHOT-012](snapshot-012-imitation-learning-bc-bootstrap.md) / [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) 的 BC 方向由此**获得针对性的成功判据**：BC→RL 路线的核心验证指标应该是 **`low_possession` 占失败比能否降到 ≤ 10%**，而不只是 WR 数字。

### 9.7 本轮判据与结论的最终 verdict

| 判据 | 状态 |
|---|---|
| 主判据（WR ≥ 当前 PPO 最强点 0.746）| **PASS**（MAPPO+v2 = 0.786，+0.040）|
| 机制判据 A（`low_possession` 占比 → ≤ 10%）| **FAIL**（五条 lane 稳定在 22-26%）|
| 机制判据 B（`late_collapse` 减少）| **PASS**（绝对数 −18~-23%，未预声明但实际发生）|

**综合判断**：MAPPO 线成功——WR 上去了、机制部分可解释（后防状态的 value 估计改善）。但它**无法单独**把我们带到 0.95，因为最顽固的 `low_possession` 桶没被触动。后续必须换维度（BC / obs / curriculum）。

## 10. 对后续路线的修订

1. **Mainline-B best = [mappo-v2 @ ckpt470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470) (0.786)** 确认为新 SOTA
2. MAPPO+v2 两条 lane 在 iter 490-500 均仍在爬坡（`v1 ckpt490 = 0.774` 是本 run 最高点；`v2 ckpt470 = 0.786` 后半段未全部 500-ep eval），建议续训到 iter 1000 或对 ckpt 480/490/500 补 500-ep，可能再推 +0.02~0.04
3. **不再做**"MAPPO no-shape / v1 / v2 之间继续摆动"的变体实验
4. **BC / DAgger 路线（[SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md)）获得明确的 target metric**：`low_possession` 占失败 ≤ 10%，而不只是 WR
5. Report 叙事上 MAPPO 部分可按 §9.1–§9.6 组织，完整呈现"两条正交干预 + 一个跨 lane 不变量"的 ablation 结构——对 technical reasoning 15 分极有价值
