# SNAPSHOT-011: Team-vs-Baseline Shaping v3 Progress-Gated Ablation

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 已完成（首轮训练）

## 1. 背景

在 [SNAPSHOT-010](snapshot-010-shaping-v2-deep-zone-ablation.md) 中，我们完成了：

- shaping-v1 与 v2 (`deep-zone + negative-C`) 的单 seed A/B
- `top 5% + ties -> baseline 500` 的正式复核
- 修正版 failure capture 分析

得到的主要结论是：

1. `v2` 没有超过 `v1` 的最佳 checkpoint，上限未提高。
2. `v2` 提高了稳定性和下限，减少了一部分：
   - `low_possession`
   - `poor_conversion`
3. `v2` 仍未解决：
   - `late_defensive_collapse` 依然是最大失败桶
4. `v2` 更像是在“把坏局拖久”，而不是“把坏局变成赢局”。

在复查 shaping 代码后，发现当前 `ball_progress_scale` 的给分逻辑本身可能过宽：

- 只要 `dx > 0`，team0 就拿正向 progress reward
- 只要 `dx < 0`，team1 就拿正向 progress reward

这个逻辑默认不要求我方真实 possession，因此可能会把一些“不够真实的前场推进”也奖励进去。

## 2. v3 假设

### 主假设

如果保留 v2 的防守负 shaping：

- `opponent_progress_penalty_scale = 0.01`
- `deep_zone_outer_threshold = -8`
- `deep_zone_outer_penalty = 0.003`
- `deep_zone_inner_threshold = -12`
- `deep_zone_inner_penalty = 0.003`

并把正向 `ball_progress` 改成 **possession-gated**：

- 只有在确认 possession 时，才发放正向 progress reward

那么：

1. 可以减少“正向 shaping 奖励了并不真正有利于赢球的局面”这一类误导；
2. 有机会把 `v2` 已经获得的稳定性，转化成更高的 baseline 胜率上限；
3. 在 failure analysis 中，`poor_conversion / low_possession` 相关的正向 shaping 假阳性应当进一步减少。

### 风险假设

若 progress gating 过严，可能会导致：

- 过多正向 shaping 被截断
- 训练前期 reward 上升更慢
- 策略变得过保守，前场推进意愿下降

因此本轮不再叠加新的时间惩罚或额外对手控球惩罚，只做最小变更。

## 3. 实验定义

### 3.1 不动的变量

以下变量全部保持与 v2 一致：

- `gamma = 0.99`
- `lambda = 0.95`
- `ball_progress_scale = 0.01`
- `opponent_progress_penalty_scale = 0.01`
- `possession_bonus = 0.002`
- `possession_dist = 1.25`
- `deep_zone_outer_threshold = -8`
- `deep_zone_outer_penalty = 0.003`
- `deep_zone_inner_threshold = -12`
- `deep_zone_inner_penalty = 0.003`
- 模型结构：`512x512`
- `single_player = False`
- `team_vs_baseline`
- `scratch`

### 3.2 唯一新变量

- `progress_requires_possession = True`

含义：

- 正向 `ball_progress` 只有在“最近球员距离球足够近、可认定为 possession”时才给分；
- 负向 `ball_progress`、`opponent_progress_penalty`、deep-zone penalty 仍按现有规则工作。

## 4. 训练入口与脚本

### 4.1 训练入口

- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)

### 4.2 v3 batch

- [soccerstwos_h100_cpu32_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512.batch)

该脚本在 v2 基础上只新增：

- `SHAPING_PROGRESS_REQUIRES_POSSESSION=1`

其余 shaping 参数全部与 v2 保持一致。

## 5. 运行结果

### 5.1 实际 run

- [PPO_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512_20260413_014417](../../ray_results/PPO_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512_20260413_014417)

训练收尾 summary：

- `best_reward_mean = 1.8638 @ iteration 500`
- `best_checkpoint = checkpoint-400`
- `best_eval_checkpoint = checkpoint-370`
- `best_eval_baseline = 35/50 = 0.700`
- `best_eval_random = 50/50 = 1.000`

相关产物：

- [training_loss_curve.png](../../ray_results/PPO_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512_20260413_014417/training_loss_curve.png)
- [checkpoint_eval.csv](../../ray_results/PPO_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512_20260413_014417/checkpoint_eval.csv)
- [best_checkpoint_by_eval.txt](../../ray_results/PPO_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512_20260413_014417/best_checkpoint_by_eval.txt)

### 5.2 训练过程概览

`episode_reward_mean` 的代表点：

- iteration `1`: `-1.619`
- iteration `100`: `1.230`
- iteration `200`: `1.750`
- iteration `300`: `1.828`
- iteration `370`: `1.858`
- iteration `400`: `1.876`
- iteration `500`: `1.864`

训练本体没有数值爆炸，也没有明显的 loss 崩坏；问题不是训不起来，而是训练内 baseline 表现始终不够高。

### 5.3 训练内 top checkpoints

按 `baseline 50` 排序，前几名为：

- iteration `370`: `0.700`
- iteration `460`: `0.700`
- iteration `400`: `0.680`
- iteration `350`: `0.660`
- iteration `290`: `0.640`
- iteration `380`: `0.640`

这说明：

- `v3` 没有形成比 `v1/v2` 更高的高位平台
- 最好点也只停在 `0.70`

## 6. 与 v1 / v2 的对比结论

### 6.1 对比对象

- `v1 best @ 500`: [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000430/checkpoint-430) = `0.746`
- `v2 best @ 500`: [checkpoint-440](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000440/checkpoint-440) = `0.728`

### 6.2 当前判断

虽然 `v3` 还没做正式 `top 5% + ties -> baseline 500` 复核，但从训练内结果看已经足够说明：

1. `v3` 没有显示出比 `v2` 更强的候选窗口；
2. `progress_requires_possession = True` 这刀没有把稳定性转成更高上限；
3. 这条线更像是把正向 shaping 截得过严，导致进攻端信号被削弱。

因此，`v3` 不适合作为后续 shaping 继续迭代的基座。

## 7. 结论

`v3` 的主假设没有得到支持。

更具体地说：

- “把正向 progress reward 改成 possession-gated” 并没有修复当前平台问题；
- 它没有超过 `v2`，也没有接近 `v1` 的最好水平；
- 说明当前瓶颈并不在于“正向 progress 给分过宽”这一点上。

所以当前最合理的收束是：

- `v3` 到此为止，不继续加算力做正式大样本复核；
- 后续如果进入 `v4`，应当回到 `v2` 作为底座，而不是站在 `v3` 上继续加码。

## 8. 下一步

下一步 `v4` 的方向不再是“继续修正 progress 口径”，而是：

- 保留 `v2` 的防守型 shaping
- 针对“baseline 更容易快速拿下我们”这个短板，尝试：
  - `fast-loss penalty`
  - `defensive survival bonus`

目标不是单纯奖励拖时间，而是：

- 避免被 baseline 速胜
- 让坏局更容易进入 baseline 自身也会漂的长局窗口
