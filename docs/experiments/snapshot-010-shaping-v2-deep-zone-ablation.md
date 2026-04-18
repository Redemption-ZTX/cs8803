# SNAPSHOT-010: Team-vs-Baseline Shaping v1 复盘、v2 Deep-Zone Ablation 与 500 局复核

- **日期**: 2026-04-12
- **负责人**:
- **状态**: 已完成

## 1. 目的

在 `team_vs_baseline` 主线上，我们已经完成一轮 `reward_shaping=True` 的 `scratch vs restore` 对照，并补上了：

- 官方 `500` 局 baseline 复核
- 失败 episode 的结构化采样
- episode 长度与 horizon 检查

本 snapshot 的目标是：

1. 正式收口 shaping-v1 的结果与失败模式
2. 明确撤回 `gamma / lambda` 是主问题的假设
3. 预先声明下一轮 `v2 shaping` 的改动边界、判据与对照方式

## 2. Shaping-v1 实验线

### 2.1 训练入口

- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)

### 2.2 v1 训练运行

#### Scratch 512x512

- 初始 run: [PPO_team_vs_baseline_shaping_scratch_512x512_20260412_002150](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_20260412_002150)
- 续跑 run: [PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353)
- merged 产物: [team_vs_baseline_shaping_scratch_512x512_merged_20260412](artifacts/merged-runs/team_vs_baseline_shaping_scratch_512x512_merged_20260412)

#### Restore 512x512

- run: [PPO_team_vs_baseline_shaping_restore_512x512_20260412_002744](../../ray_results/PPO_team_vs_baseline_shaping_restore_512x512_20260412_002744)

### 2.3 训练内最佳 checkpoint

- scratch:
  - [checkpoint-400](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353/TeamVsBaselineShapingPPOTrainer_Soccer_639ea_00000_0_2026-04-12_06-14-28/checkpoint_000400/checkpoint-400)
  - [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353/TeamVsBaselineShapingPPOTrainer_Soccer_639ea_00000_0_2026-04-12_06-14-28/checkpoint_000430/checkpoint-430)
- restore:
  - [checkpoint-530](../../ray_results/PPO_team_vs_baseline_shaping_restore_512x512_20260412_002744/TeamVsBaselineShapingPPOTrainer_Soccer_005ed_00000_0_2026-04-12_00-28-05/checkpoint_000530/checkpoint-530)

## 3. 官方 500 局复核

### 3.1 shaping-v1

- [checkpoint-400](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353/TeamVsBaselineShapingPPOTrainer_Soccer_639ea_00000_0_2026-04-12_06-14-28/checkpoint_000400/checkpoint-400)
  - baseline: `366/500 = 0.732`
- [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353/TeamVsBaselineShapingPPOTrainer_Soccer_639ea_00000_0_2026-04-12_06-14-28/checkpoint_000430/checkpoint-430)
  - baseline: `348/500 = 0.696`
- [checkpoint-530](../../ray_results/PPO_team_vs_baseline_shaping_restore_512x512_20260412_002744/TeamVsBaselineShapingPPOTrainer_Soccer_005ed_00000_0_2026-04-12_00-28-05/checkpoint_000530/checkpoint-530)
  - baseline: `351/500 = 0.702`

### 3.2 旧链条对照

- [checkpoint-160](../../ray_results/PPO_train_h100_cpu24_20260408_165721/PPO_Soccer_955eb_00000_0_2026-04-08_16-57-41/checkpoint_000160/checkpoint-160)
  - baseline: `363/500 = 0.726`
  - random: `497/500 = 0.994`
- [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225)
  - baseline: `382/500 = 0.764`
  - random: `496/500 = 0.992`
- [checkpoint-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30)
  - baseline: `393/500 = 0.786`
  - random: `497/500 = 0.994`

### 3.3 当前结论

- shaping-v1 的 `scratch-400` 已略高于旧 [checkpoint-160](../../ray_results/PPO_train_h100_cpu24_20260408_165721/PPO_Soccer_955eb_00000_0_2026-04-08_16-57-41/checkpoint_000160/checkpoint-160)
  - `0.732 > 0.726`
- 但仍明显低于旧 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225)
  - `0.732 < 0.764`
- 也低于旧 role 最优 [checkpoint-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30)
  - `0.732 < 0.786`
- `restore-530` 没有证明自己优于 `scratch-400`
  - `0.702 < 0.732`

因此：

- `restore + shaping v1` 暂不值得继续主推
- `scratch + shaping v1` 有信号，但仍不足以超过旧主线最优

## 4. 失败样本采集

### 4.1 工具链

- [evaluate_matches.py](../../cs8803drl/evaluation/evaluate_matches.py)
- [failure_cases.py](../../cs8803drl/evaluation/failure_cases.py)
- [capture_failure_cases.py](../../scripts/eval/capture_failure_cases.py)

### 4.2 500 局失败样本目录

- checkpoint-400 baseline failures:
  - [checkpoint400_baseline_500](artifacts/failure-cases/checkpoint400_baseline_500)
- checkpoint-160 baseline failures:
  - [checkpoint160_baseline_500](artifacts/failure-cases/checkpoint160_baseline_500)

### 4.3 聚合观察

#### checkpoint-400 (`129` 场失败)

- primary labels:
  - `late_defensive_collapse`: `53`
  - `low_possession`: `38`
  - `poor_conversion`: `18`
  - `unclear_loss`: `16`
  - `opponent_forward_progress`: `4`
- mean metrics:
  - `mean_ball_x = 1.57`
  - `tail_mean_ball_x = 1.76`
  - `team0_possession_ratio = 0.478`
  - `team0_progress_toward_goal = 5.42`
  - `team1_progress_toward_goal = 4.61`

#### checkpoint-160 (`135` 场失败)

- primary labels:
  - `late_defensive_collapse`: `64`
  - `low_possession`: `32`
  - `unclear_loss`: `18`
  - `poor_conversion`: `13`
  - `opponent_forward_progress`: `4`
  - `territory_loss`: `4`
- mean metrics:
  - `mean_ball_x = 0.53`
  - `tail_mean_ball_x = 0.91`
  - `team0_possession_ratio = 0.462`
  - `team0_progress_toward_goal = 4.38`
  - `team1_progress_toward_goal = 4.76`

### 4.4 失败画像结论

- `checkpoint-400` 相比 `checkpoint-160` 的进步，主要不是“更会终结”，而是：
  - 少一些 `late_defensive_collapse`
  - 球整体更靠前
  - 我方平均控球与推进略好
- 当前 shaping-v1 仍未解决的核心失败桶是：
  - `late_defensive_collapse`
  - `low_possession`
  - `poor_conversion`
- 特别要注意：
  - 在部分 `low_possession / poor_conversion` 失败局里，正向 shaping 仍然给出了误导性正反馈
  - 因此不适合直接把全部 shaping 统一放大

## 5. Horizon 假设复核

### 5.1 当前实际配置

- `gamma = 0.99`
- `lambda = 0.95`

### 5.2 训练期 episode 长度

#### scratch_continue

- `episode_len_mean`:
  - mean `34.8`
  - median `34.8`
  - min `32.9`
  - max `37.1`

#### restore

- `episode_len_mean`:
  - mean `32.7`
  - median `32.6`
  - min `30.7`
  - max `35.8`

### 5.3 失败局长度

#### checkpoint-400 failures

- overall:
  - median `31`
  - p75 `52`
  - max `143`
- `late_defensive_collapse`:
  - median `34`
  - p75 `52`

#### checkpoint-160 failures

- overall:
  - median `31`
  - p75 `52`
  - max `139`
- `late_defensive_collapse`:
  - median `32`
  - p75 `53`

### 5.4 结论

我们撤回“当前主问题是 horizon 太短”的推断。

在当前 episode 长度下：

- `gamma = 0.99`
- `lambda = 0.95`

对终局稀疏信号的回传是足够覆盖大多数 episode 的。  
因此，`late_defensive_collapse` 更可能是：

- 稀少防守状态下缺乏足够明确的导向信号
- 而不是单纯的 credit assignment 距离过长

## 6. v2 shaping 预声明

### 6.1 不动的变量

以下全部保持不变：

- `gamma`
- `lambda`
- `ball_progress_scale`
- `possession_bonus`
- `possession_dist`
- 模型结构
- `single_player` 语义
- possession 定义本身

### 6.2 v1 vs v2

#### v1 batch（显式锁定当前 shaping）

脚本：

- [soccerstwos_h100_cpu32_team_vs_baseline_shaping_v1_scratch_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_vs_baseline_shaping_v1_scratch_512x512.batch)

显式配置：

- `BASELINE_PROB = 1.0`
- `FCNET_HIDDENS = 512,512`
- `USE_REWARD_SHAPING = 1`
- `SHAPING_TIME_PENALTY = 0.001`
- `SHAPING_BALL_PROGRESS = 0.01`
- `SHAPING_OPP_PROGRESS_PENALTY = 0.0`
- `SHAPING_POSSESSION_BONUS = 0.002`
- `SHAPING_POSSESSION_DIST = 1.25`

#### v2 batch（只改 A + 负向 C）

脚本：

- [soccerstwos_h100_cpu32_team_vs_baseline_shaping_v2_deepzone_scratch_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_vs_baseline_shaping_v2_deepzone_scratch_512x512.batch)

仅覆盖以下值：

- `SHAPING_OPP_PROGRESS_PENALTY = 0.01`
- `SHAPING_DEEP_ZONE_OUTER_THRESHOLD = -8`
- `SHAPING_DEEP_ZONE_OUTER_PENALTY = 0.003`
- `SHAPING_DEEP_ZONE_INNER_THRESHOLD = -12`
- `SHAPING_DEEP_ZONE_INNER_PENALTY = 0.003`

### 6.3 v2 的假设

#### 主假设

增加“我方危险区惩罚 + 对手推进第二层惩罚”后：

- 可以减少 `late_defensive_collapse`
- 不需要改动 `gamma / lambda`
- 也不需要整体放大正向 shaping

#### 风险假设

如果 deep-zone 负向 shaping 过强，策略可能被推向被动化：

- 中场游荡
- 只会清球
- 缺乏主动压上

因此深位惩罚必须保持克制，单局累计不能接近一次进球的稀疏奖励量级。

## 7. v2 实验判据

### 7.1 主判据

在 `iter >= 300` 后选点，用官方 `500` 局 baseline 复核：

- 若 v2 比 v1 提升 `>= 0.05`
- 则视为主判据通过

### 7.2 机制判据

在 `500` 局失败样本中：

- `late_defensive_collapse`
  - 从当前约 `45%` 量级
  - 降到 `<= 30%`

若达到，则视为机制判据通过。

### 7.3 烟测判据

`10` iteration smoke 时：

- `entropy` 不下跌超过 `50%`
- `episode_reward_mean >= 1.0`
- 失败局 `team0_shaping_reward_sum ∈ [-0.5, +0.3]`

### 7.4 胜负规则

- 主判据 + 机制判据都通过：v2 胜
- 只过主判据：部分胜，进入下一轮
- 两项都不过：放弃当前 `A + 负向 C` 方案

## 8. 执行约束

- seed 数量：每边至少 `2`
- 推荐同时跑 `4` 个 job（v1/v2 各两个 seed）
- 不再使用单 seed 结果拍板
- 必须做严格 A/B，不允许把其他变量混进去

## 9. v1 / v2 单 seed A/B 结果

### 9.1 训练运行

#### v1

- run: [PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902)
- training curve: [training_loss_curve.png](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/training_loss_curve.png)
- 训练内 best eval checkpoint:
  - [checkpoint-410](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000410/checkpoint-410)
  - baseline `38/50 = 0.76`

#### v2

- run: [PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755)
- training curve: [training_loss_curve.png](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/training_loss_curve.png)
- 训练内 best eval checkpoint:
  - [checkpoint-460](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000460/checkpoint-460)
  - baseline `41/50 = 0.82`

### 9.2 Top-5% + ties 复核规则

对每条 run：

- 仅按 training-internal `baseline 50-ep` 排序
- 取前 `5%`
- 若第 `5%` cutoff 分数有并列，则同分全部复核
- 官方复核只看 `baseline 500`

在本次 `49` 个 baseline checkpoints 的设置下，前 `5%` 基数为 `3`，但由于 cutoff 并列，两条线最终都复核了 `6` 个 checkpoint。

### 9.3 官方 500 局 baseline 复核

#### v1 top-5% + ties

- [checkpoint-410](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000410/checkpoint-410): `334/500 = 0.668`
- [checkpoint-180](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000180/checkpoint-180): `292/500 = 0.584`
- [checkpoint-340](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000340/checkpoint-340): `350/500 = 0.700`
- [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000430/checkpoint-430): `373/500 = 0.746`
- [checkpoint-440](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000440/checkpoint-440): `354/500 = 0.708`
- [checkpoint-480](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000480/checkpoint-480): `358/500 = 0.716`

聚合：

- mean `0.687`
- median `0.704`
- max `0.746`
- min `0.584`

#### v2 top-5% + ties

- [checkpoint-460](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000460/checkpoint-460): `350/500 = 0.700`
- [checkpoint-350](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000350/checkpoint-350): `358/500 = 0.716`
- [checkpoint-370](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000370/checkpoint-370): `344/500 = 0.688`
- [checkpoint-390](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000390/checkpoint-390): `357/500 = 0.714`
- [checkpoint-440](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000440/checkpoint-440): `364/500 = 0.728`
- [checkpoint-490](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000490/checkpoint-490): `354/500 = 0.708`

聚合：

- mean `0.709`
- median `0.711`
- max `0.728`
- min `0.688`

### 9.4 v1 / v2 结果结论

- 若只看“单个最佳 checkpoint”，本轮 `v1 > v2`
  - `v1 best = 0.746 @ checkpoint-430`
  - `v2 best = 0.728 @ checkpoint-440`
- 若看 `top-5% + ties` 的整体分布，`v2` 更稳
  - `v2` 的 mean / median 更高，方差明显更小
- 因此，`v2` 的 deep-zone / negative-C shaping 更像是在：
  - 提高下限
  - 降低波动
  - 但没有把最好成绩推得更高

按本 snapshot 预注册的主判据：

- `v2` 未能实现相对 `v1` 的 `>= 0.05` 提升
- 因此主判据未通过

## 10. 修正版失败样本分析（v1@430 vs v2@440）

### 10.1 采样目录

- v1 rerun:
  - [v1_checkpoint430_baseline_500_rerun](artifacts/failure-cases/v1_checkpoint430_baseline_500_rerun)
- v2 rerun:
  - [v2_checkpoint440_baseline_500_rerun](artifacts/failure-cases/v2_checkpoint440_baseline_500_rerun)

这次 rerun 特别修正了一个重要问题：

- `v2` 的 failure capture 显式传入了
  - `opponent_progress_penalty_scale = 0.01`
  - `deep_zone_outer_threshold = -8`
  - `deep_zone_outer_penalty = 0.003`
  - `deep_zone_inner_threshold = -12`
  - `deep_zone_inner_penalty = 0.003`

因此本节中的 `v2 shaping_reward_sum` 与 `deep_zone_*` 标记是可信的。

### 10.2 胜负时间结构

#### v1 @ checkpoint-430

- team0 wins: `343`
- team1 wins: `157`
- win rate: `0.686`
- step stats:
  - all: mean `48.8`, median `37.5`, p75 `64`
  - team0 win: mean `52.3`, median `43`, p75 `67`
  - team1 win: mean `41.2`, median `29`, p75 `50`

#### v2 @ checkpoint-440

- team0 wins: `366`
- team1 wins: `134`
- win rate: `0.732`
- step stats:
  - all: mean `50.1`, median `40.5`, p75 `64`
  - team0 win: mean `52.5`, median `43`, p75 `66`
  - team1 win: mean `43.5`, median `32`, p75 `55`

结论：

- 两条线都呈现“赢得慢、输得快”
- `v2` 相比 `v1`，几乎没有改变赢局时间结构
- `v2` 的主要变化是：输局被拖得更久一些
- 这进一步支持：
  - `v2` 主要是在修“别太早崩”
  - 不是在让赢局更高效

### 10.3 失败类型聚合

#### v1 @ checkpoint-430 (`157` 场失败)

- `late_defensive_collapse`: `69`
- `low_possession`: `44`
- `poor_conversion`: `22`
- `unclear_loss`: `16`
- `territory_loss`: `6`

#### v2 @ checkpoint-440 (`134` 场失败)

- `late_defensive_collapse`: `65`
- `low_possession`: `30`
- `poor_conversion`: `16`
- `unclear_loss`: `15`
- `territory_loss`: `5`
- `opponent_forward_progress`: `3`

结论：

- `v2` 确实减少了：
  - `low_possession`
  - `poor_conversion`
- 但 `late_defensive_collapse` 仍然是最大失败桶
- 这说明 `v2` 的 A + 负向 C 并不是无效，而是：
  - 对“球权拿不到 / 前场转化差”有帮助
  - 但没有真正消灭“后段防线崩掉”这个主问题

### 10.4 Deep-zone 触发与 shaping 信号

在 [v2_checkpoint440_baseline_500_rerun](artifacts/failure-cases/v2_checkpoint440_baseline_500_rerun) 的保存轨迹中，deep-zone 标记被大量触发：

- `deep_zone_team0_outer`: `1134`
- `deep_zone_team0_inner`: `633`

同时，在 `late_defensive_collapse` 失败桶内：

- `team0_shaping_reward_sum`
  - v1: `-0.368`
  - v2: `-0.712`

这说明：

- deep-zone 与 opponent-progress 的负 shaping 确实在工作
- `v2` 并不是“参数写了但训练/诊断没真正打到”

### 10.5 修正版失败分析结论

本轮更可信的结论是：

- `v2` 不是错误方向
- 它确实减少了一部分：
  - `low_possession`
  - `poor_conversion`
- 它也确实让输局整体拖得更久
- 但它依然没能把 `late_defensive_collapse` 从最大失败桶的位置上打下来

因此：

- `v2` 更像是“提稳”的 shaping
- 不是“提上限”的 shaping

## 11. 最终结论

本 snapshot 的最终结论是：

1. `top 5% + ties + baseline 500` 是比单个 `50-ep` best checkpoint 更可信的选模规则，应保留。
2. `v2` deep-zone / negative-C shaping 没有赢下本轮 A/B：
   - 没有超过 `v1` 的最好点
   - 也没有超过旧 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225) 或 [checkpoint-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30)
3. 但 `v2` 也不是无效：
   - 它减少了部分 `low_possession / poor_conversion`
   - 它让训练和失败分布更稳
4. 下一步如果继续改 shaping，重点不该再是“单纯增加防守罚分”，而要更针对：
   - 为什么坏局只是被拖久，而没有被扳回来
   - 为什么前场优势还不能稳定转成更高的 baseline 胜率
