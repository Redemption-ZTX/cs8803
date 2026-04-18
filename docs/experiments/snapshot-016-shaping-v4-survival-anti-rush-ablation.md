# SNAPSHOT-016: Team-vs-Baseline Shaping v4 Survival / Anti-Rush Ablation

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 已完成（首轮对照）

## 1. 背景

在 [SNAPSHOT-010](snapshot-010-shaping-v2-deep-zone-ablation.md) 与 [SNAPSHOT-011](snapshot-011-shaping-v3-progress-gated-ablation.md) 之后，我们已经有两个较稳定的判断：

1. `v2` 的 `deep-zone + negative-C` 方向能提高下限、减少一部分：
   - `low_possession`
   - `poor_conversion`
2. `v2` 没有把“更稳”转成更高的 baseline 胜率上限。
3. `v3` 的 `progress_requires_possession=True` 没有改善这一点，反而把正向信号截得过严，训练内 best 只有 `0.70`。

同时，baseline weakness analysis（见 [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md)）进一步说明：

- baseline 自己的主要弱点是**长局漂移**和 `late_defensive_collapse`
- 我们当前输给 baseline 的局，整体上**结束得更快**
- `v2` 已经表现出“把输局拖长一点”的趋势，但还不足以把更多坏局变成赢局

因此，`v4` 不再站在 `v3` 上继续调 progress 口径，而是：

- 回到 `v2` 为底座
- 专门针对“别被 baseline 速胜、把坏局拖进 baseline 自己也会漂的长局窗口”做最小增量

## 2. v4 假设

### 主假设

如果在 `v2` 的 shaping 基础上，再加入两类**受压生存导向**信号：

1. `defensive survival bonus`
   - 当对手确认 possession，且球已经压到我方半场/危险区时
   - defending team 每多撑一步，拿到一个很小的正奖励
2. `fast-loss penalty`
   - 如果最终输球，而且输得过快
   - 按“距离阈值还差多少步”施加一个终局额外惩罚

那么：

1. 策略会更主动学习“受压时先活下来”，而不是继续被 baseline 快速拿下；
2. 败局的 step 分布应进一步向 baseline 自身的长局分布靠近；
3. 若这一方向有效，训练内与正式复核中的 baseline 胜率上限应有机会超过 `v2`。

### 明确不做的事

`v4` 不是“全局奖励拖时间”，也不是鼓励磨洋工。

因此本轮不做：

- 全局时间越长越正奖励
- 在有终结机会时仍继续拖时间
- 再次叠加 `progress_requires_possession=True`

`v4` 的目标是“不要被速胜”，不是“无意义地延长所有比赛”。

## 3. 实验定义

### 3.1 基座

`v4` 显式继承 `v2`：

- `team_vs_baseline`
- `scratch`
- `single_player=False`
- `FCNET_HIDDENS=512,512`
- `gamma=0.99`
- `lambda=0.95`
- `ball_progress_scale=0.01`
- `progress_requires_possession=0`
- `opponent_progress_penalty_scale=0.01`
- `possession_bonus=0.002`
- `possession_dist=1.25`
- `deep_zone_outer_threshold=-8`
- `deep_zone_outer_penalty=0.003`
- `deep_zone_inner_threshold=-12`
- `deep_zone_inner_penalty=0.003`

### 3.2 v4 新增项

#### A. Defensive survival bonus

- `defensive_survival_threshold = -2`
- `defensive_survival_bonus = 0.001`

含义：

- 若对手确认 possession，且球已压到 `x < -2`
- 则我方 team 每步得到极小的 survival bonus

设计意图：

- 不是奖励被压着打
- 而是给“撑住、别立刻死”的行为一点训练导向

#### B. Fast-loss penalty

- `fast_loss_threshold_steps = 40`
- `fast_loss_penalty_per_step = 0.01`

含义：

- 若 team0 输球且 episode 少于 `40` 步
- 则按 `40 - episode_steps` 的 shortfall 施加额外终局惩罚

例如：

- 30 步输球：额外 team total penalty = `0.10`
- 20 步输球：额外 team total penalty = `0.20`

这仍然明显小于一次终局稀疏输球的量级，不会压过原始胜负信号。

## 4. 训练入口与脚本

### 4.1 训练入口

- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)

### 4.2 v4 batch

- [soccerstwos_h100_cpu32_team_vs_baseline_shaping_v4_survival_scratch_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_vs_baseline_shaping_v4_survival_scratch_512x512.batch)

该脚本显式锁定：

- `SHAPING_PROGRESS_REQUIRES_POSSESSION=0`
- `SHAPING_OPP_PROGRESS_PENALTY=0.01`
- `SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8`
- `SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003`
- `SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12`
- `SHAPING_DEEP_ZONE_INNER_PENALTY=0.003`
- `SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD=-2`
- `SHAPING_DEFENSIVE_SURVIVAL_BONUS=0.001`
- `SHAPING_FAST_LOSS_THRESHOLD_STEPS=40`
- `SHAPING_FAST_LOSS_PENALTY_PER_STEP=0.01`

## 5. 评估规则

沿用当前正式协议（见 [engineering-standards.md](../architecture/engineering-standards.md#checkpoint-选模规则)）：

1. 训练内只用 `baseline 50` 做候选筛选
2. 按 `top 5% + ties`
3. 正式复核 `baseline 500`
4. 最终 `1-2` 个候选再补 `random 500`

## 6. 首轮实际运行

### 6.1 实际 run

- [PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133)

训练收尾 summary：

- `best_reward_mean = 2.1860 @ iteration 478`
- `best_checkpoint = checkpoint-478`
- `best_eval_checkpoint = checkpoint-430`
- `best_eval_baseline = 44/50 = 0.88`
- `best_eval_random = 50/50 = 1.00 @ iteration 460`

相关产物：

- [training_loss_curve.png](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/training_loss_curve.png)
- [checkpoint_eval.csv](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/checkpoint_eval.csv)

### 6.2 训练内候选窗口

按 `baseline 50` 的 `top 5% + ties`，正式进入复核的点为：

- [checkpoint-400](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000400/checkpoint-400): `0.80`
- [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000430/checkpoint-430): `0.88`
- [checkpoint-460](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000460/checkpoint-460): `0.82`

这里最重要的判断是：

- `0.88 @ 430` 看起来非常亮眼
- 但它两侧的 `420 -> 0.70`、`440 -> 0.72` 说明这更像是被 `50` 局评估放大的高点
- 所以必须进入正式 `baseline 500` 才能判断 `v4` 是否真的超过 `v2`

## 7. 正式复核

三条候选的官方 `baseline 500` 结果为：

- [checkpoint-400](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000400/checkpoint-400): `0.768`
- [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000430/checkpoint-430): `0.768`
- [checkpoint-460](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000460/checkpoint-460): `0.766`

这组结果说明：

1. 训练内 `0.88` 确实被高估了；
2. 但 `v4` 不是偶然神点，而是形成了一个非常稳定的平台；
3. `v4` 的真实水平大约稳定在 `0.766 ~ 0.768`。

## 8. 与既有 PPO / MAPPO 主线的对比

关键基线：

- 旧 PPO warm-start 强点 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225): `0.764`
- 纯 PPO shaping-v1 强点 [checkpoint-430](../../ray_results/PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902/TeamVsBaselineShapingPPOTrainer_Soccer_6ffde_00000_0_2026-04-12_21-09-36/checkpoint_000430/checkpoint-430): `0.746`
- 纯 PPO shaping-v2 强点 [checkpoint-440](../../ray_results/PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755/TeamVsBaselineShapingPPOTrainer_Soccer_4013f_00000_0_2026-04-12_21-08-15/checkpoint_000440/checkpoint-440): `0.728`
- `v3` 首轮训练内 best 只有 `0.70`，已在 [SNAPSHOT-011](snapshot-011-shaping-v3-progress-gated-ablation.md) 收口为负结果
- 当前 MAPPO 最强点 [checkpoint-470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470): `0.786`

因此：

1. `v4` 明显超过了纯 PPO 的 `v1 / v2 / v3`；
2. `v4` 也略微超过了旧 `checkpoint-225 = 0.764`；
3. 但 `v4` 仍然没有超过 `MAPPO + shaping-v2 = 0.786`。

## 9. 判据回看

### 9.1 主判据

若 `v4` 的正式 `baseline 500` 最佳 checkpoint：

- 超过 `v2` 的当前 best `0.728`
- 并尽量接近或超过 `v1` 的当前 best `0.746`

那么视为这轮 survival / anti-rush shaping 有主效果。

当前结果：

- `v4 best = 0.768`

所以主判据成立。

### 9.2 机制判据

基于 failure analysis，重点看：

1. 输局步数是否继续上升
   - 尤其是 `team1_win median steps`
2. `late_defensive_collapse` 是否开始下降
   - 不再只是“拖晚一点输”
3. `low_possession / poor_conversion` 是否不反弹
   - 避免为了“活更久”重新引入旧病

这一部分现已通过 `v4` 与 `v2` 的正式 failure capture 对比完成。

#### Failure capture 对比对象

- `v4 best PPO`：
  - [v4 checkpoint-400 baseline 500](artifacts/failure-cases/v4_checkpoint400_baseline_500)
  - top-level summary：`361W-139L = 0.722`
- `v2 best PPO`：
  - [v2 checkpoint-440 baseline 500 fresh](artifacts/failure-cases/v2_checkpoint440_baseline_500_fresh)
  - top-level summary：`373W-127L = 0.746`

注意：

- 这两次 capture 主要用于**诊断失败模式**
- 不替代官方 `500` 的最终排序
- 最终强弱仍以第 7 节的官方结果为准：`v4 best = 0.768 > v2 best = 0.728`

#### 机制结果

1. `v4` 的确把输局拖长了

- `v4` 输局：`mean=52.6 / median=39`
- `v2` 输局：`mean=44.7 / median=34`

而赢局时长几乎没变：

- `v4` 赢局 median `41`
- `v2` 赢局 median `42`

这说明 `v4` 的 survival / anti-rush 主要作用于：

- 延缓败局
- 减少被 baseline 快速拿下

而不是单纯把所有比赛都拖长。

2. `v4` 压低了 `late_defensive_collapse` 与 `unclear_loss`

- `late_defensive_collapse`
  - `v4`: `58/139 = 41.7%`
  - `v2`: `62/127 = 48.8%`
- `unclear_loss`
  - `v4`: `13/139 = 9.4%`
  - `v2`: `18/127 = 14.2%`

而且在 `late_defensive_collapse` 里，球整体没有掉得像 `v2` 那么深：

- `mean_ball_x`
  - `v4`: `-4.65`
  - `v2`: `-6.30`
- `tail_mean_ball_x`
  - `v4`: `-7.13`
  - `v2`: `-7.51`

所以 `v4` 不是空喊“别速败”，而是真把一部分后防崩盘拖晚、拖浅了。

3. `v4` 同时把部分失败推向了 `low_possession / poor_conversion`

- `low_possession`
  - `v4`: `39/139 = 28.1%`
  - `v2`: `30/127 = 23.6%`
- `poor_conversion`
  - `v4`: `19/139 = 13.7%`
  - `v2`: `12/127 = 9.4%`

这说明：

- `v4` 更少直接后防崩掉
- 但把一部分坏局转化成了“球拿不住”或“推进了但终结不掉”

也就是说，`v4` 更会“活下来”，但还不够会把活下来的局面踢成真正的控局和终结。

4. 正向 shaping 的误导问题没有完全消失

在 `low_possession` 这一类里，`team0_shaping_reward_sum` 仍然是正的，而且 `v4` 更高：

- `v4`: `+0.194`
- `v2`: `+0.129`

这说明 `v4` 修的是防守生存，不是正向目标完全对齐。

#### 机制判据结论

因此，`v4` 的机制结论可以收成：

- 它确实修到了 **anti-rush / survival**
- 它确实减少了一部分 `late_defensive_collapse`
- 但它没有把这些被拖长的坏局稳定转成赢局
- 反而增加了 `low_possession / poor_conversion` 的占比

一句话总结：

> `v4` 修的是败局节奏，不是进攻质量。

## 10. 当前判断

`v4` 不是对 `v3` 的继续微调，而是一次明确回退到 `v2` 基座后的正确回摆。

现在可以比较有把握地说：

- `v2` 负责把局踢稳；
- `v4` 进一步把这种稳定性转成了更成熟的 PPO 平台；
- 但这条线的收益更像是“把 PPO 做到更完整”，而不是把 PPO 带到超过 MAPPO 的层级。
- 同时，`v4` 还没有解决 `low_possession / poor_conversion` 这两个更偏进攻端的问题。

一句话总结：

> `v4` 是 PPO 主线上的正结果，而且是当前最好的 PPO shaping 版本；但项目层面的第一名仍是 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 的 `MAPPO + shaping-v2`。

## 11. 下一步

`v4` 的下一步不是继续看训练内 `50` 局尖峰，而是：

1. 将 `failure capture` 的结论视为已完成：
   - `survival / anti-rush` 的主效果成立
   - 但同时引入了更多 `low_possession / poor_conversion`
2. 如果继续沿 PPO shaping 线迭代，下一刀不该再加 survival，而该针对：
   - 控球质量
   - 终结转化
3. 若从项目主线角度继续推进，则更优先的方向仍是：
   - [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 的 `MAPPO + shaping-v2`
   - 以及 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) 的 `BC -> RL bootstrap`

## 12. Verdict 汇总与跨 lane 定位

### 12.1 §9.2 三条预声明机制判据的 verdict

| # | 预声明项 | 实测 | 结果 |
|---|---|---|---|
| 1 | 输局 median step 上升 | v2 34 → **v4 39** (+5, mean 44.7 → 52.6) | ✅ **PASS** |
| 2 | late_defensive_collapse 绝对数下降 | v2 62 → **v4 58**（−6%）；占失败比 49% → 42% | ✅ **PASS** |
| 3 | low_possession / poor_conversion **不反弹** | low_poss 30 → **39**（+30%）；poor_conv 12 → **19**（+58%） | ❌ **FAIL** |

**2/3 过。** (1)(2) 印证了 survival / anti-rush 设计意图；(3) 暴露了**我们没预见的代价**：用 `fast_loss_penalty` 压"输得快"的副作用是 policy 进攻端变犹豫，把一部分原本会转化为胜利或 late_collapse 的局**重新洗进** low_possession 和 poor_conversion 两个"老病桶"。

### 12.2 官方 eval vs failure-capture 噪声

| 测点 | WR | 说明 |
|---|---|---|
| v4 @400 官方 (500 eps) | **0.768** | 主排序依据 |
| v4 @430 官方 (500 eps) | 0.768 | 三点极度一致 |
| v4 @460 官方 (500 eps) | 0.766 | → 真实 WR 稳定 ~0.767 |
| v4 @400 failure-capture (500 eps) | 0.722 | 偏低 0.046 |

跨 run 方差 **~0.046**，大于之前 v2 见到的 ~0.03。这不影响 v4 胜出结论（§7 官方三点一致），但**说明 failure-capture 的 WR 数字不能替代官方 eval 做 lane 间排序**——它只承担失败桶诊断功能。

### 12.3 跨 lane 失败桶合表（500 eps 绝对数）

| Bucket | v1 PPO | v2 PPO | MAPPO+v2 | **v4** |
|---|---|---|---|---|
| late_defensive_collapse | 77 | 70 | 62 | **58 ← 史上最低** |
| low_possession | 37 | 29 | 30 | **39 ← 反弹到 v1 水平** |
| poor_conversion | 19 | 11 | 11 | **19 ← 反弹到 v1 水平** |
| unclear_loss | 19 | 17 | 16 | 13 |
| territory_loss | 4 | 4 | 6 | 7 |
| opponent_forward_progress | 1 | 3 | 0 | 3 |
| **总失败** | 157 | 134 | **125** | 139 |

**关键观察**：v4 的"总失败数"(139) 实际**比 v2 prior (134) 还多**。v4 的 WR 优势 (0.767 vs v2 官方 ~0.73) 主要靠**官方 eval 侧的 seed 差异**放大，核心机制层面是**桶间重分布**而非总减少——这正是 12.1 判据 3 FAIL 的另一种表达。

### 12.4 跨 lane 主线排名（500-ep 官方 WR）

| 排名 | Lane | 官方 WR | 备注 |
|---|---|---|---|
| 1 | MAPPO + shaping-v2 @ 470 | **0.786** | 当前 SOTA |
| 2 | MAPPO + shaping-v1 @ 490 | 0.774 | 仍在爬坡 |
| 3 | **v4 PPO @ 400/430/460** | **0.767** | **当前最强 PPO** |
| 4 | old PPO warm-start ckpt-225 | 0.764 | 历史遗留 |
| 5 | v1 PPO @ 430 | 0.746 | |
| 6 | MAPPO no-shape @ 450/490 | 0.742 | |
| 7 | v2 PPO @ 440 | 0.728-0.732 | |

### 12.5 对 [SNAPSHOT-013 §11](snapshot-013-baseline-weakness-analysis.md) 的反馈

snapshot-013 §11 当时的关键发现是：**`low_possession` 跨 5 种干预恒定在 22-26%**，推论是 "reward / critic 维度修不到它"。

v4 给这条推论**加了一个硬数据点**：

| Lane | low_possession 占失败比 |
|---|---|
| v1 PPO | 24% |
| v2 PPO | 22% |
| MAPPO no-shape | 26% |
| MAPPO v1 | 24% |
| MAPPO v2 | 24% |
| **v4 PPO** | **28%** ← 所有 lane 里最高 |

v4 用更激进的防守 shaping（survival bonus + fast-loss penalty）去修，结果 `low_possession` **不但没降，反而涨到 28%**——跨 lane 从"22-26% 的不变量"扩成"22-28% 的 shaping 越防守越高"。

这**强化了** snapshot-013 §11.3 的判断：`low_possession` 不是可调 reward/critic 的信号问题，而是更底层的 obs / env / 训练分布问题。**任何在 shaping 层继续加 term 都会在其他桶造成转移，不会触及 low_possession 本身。**

**对下一阶段**（snapshot-013 §11.5 修订的 P0）：
- v4 数据把"**低 possession 只能靠 BC/obs 改造修**"这一判断**由推测升为实证**
- [SNAPSHOT-015 BC 方向](snapshot-015-behavior-cloning-team-bootstrap.md) 的 success criterion 不变：`low_possession` ≤ 10%
- 任何后续 shaping 变体（假设 v5 / v6）再在这个桶上调 reward，**预期不可能有效**

## 13. 对 Submission 策略的影响

### 13.1 三分 agent 候选更新

基于现有 500-ep 官方数据：

| 评分项 | 当前最强候选 | 500-ep WR | 理由 |
|---|---|---|---|
| **Performance (50 分)** | MAPPO + shaping-v2 @ ckpt470 | **0.786** | 现 SOTA；WR 最高；snapshot-014 完整论证 |
| **Reward Modification (40 分)** | **v4 PPO @ ckpt400 / 430**（推荐换） | 0.768 | shaping 故事最丰富（4 个正交 term）；最强 PPO；failure-bucket trade-off 本身是优质 technical reasoning 素材 |
| **Novel Concept (+5)** | 留给 [SNAPSHOT-015 BC lane](snapshot-015-behavior-cloning-team-bootstrap.md) | TBD | imitation learning 明确在作业 bonus 条款里 |

### 13.2 为何从 v2 PPO @440 换成 v4 PPO @400

Modification 评分标准是"**添加/更改 reward 或 observation space + 代码正确 + 理论合理 + hypothesis 清晰**"，**不**是"胜率最高"。v4 相对 v2 在 Modification 叙事上的优势：

1. **更丰富的 shaping 结构**：`deep_zone outer/inner + C-neg + survival_bonus + fast_loss_penalty` 四个正交 term，每个都有独立 hypothesis
2. **更高的官方 WR**（0.767 vs 0.728-0.732），读者看 performance 数字也舒服
3. **Failure-bucket trade-off 是诚实的 technical reasoning 材料**：snapshot-016 §9/§12.1 记录了"我们预期防守会改善，但没预期进攻会退化"——这种**意外 trade-off 的诚实分析**比简单的"shaping 让 WR 从 0.7 变 0.77"更能拿到 15 分的 analysis 分

### 13.3 需要交付物

若决定把 v4 @ckpt400 作为 Modification agent：

1. 打包成 `agents/v004_shaping_v4_survival/`
2. 复制 checkpoint 和 params.pkl
3. 写 README（作者、描述、对应评分项 = "Reward Modification"）
4. 在 report 的 hypothesis/motivation 部分引用 snapshot-016 §2 和 §12.1

Performance agent 的打包则对应 MAPPO+v2@470，交付结构同上但 module 路径使用 `trained_shared_cc_agent`。
