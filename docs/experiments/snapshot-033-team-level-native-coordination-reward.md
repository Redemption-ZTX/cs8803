# SNAPSHOT-033: Team-Level Native Coordination Reward Shaping

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 已完成首轮结果（033-A）

## 0. 方法论位置

team-level native 系列的第三条——**环境/reward 层面**。

| snapshot | 维度 | 改动位置 |
|---|---|---|
| [031](snapshot-031-team-level-native-dual-encoder-attention.md) | 网络架构 | forward pass 结构 |
| [032](snapshot-032-team-level-native-coordination-aux-loss.md) | 训练目标 | loss function |
| **033（本）** | 环境/reward | shaping 信号 |

和 031/032 正交。三者都可以叠加，但首轮独立测。

## 0.5 首轮可执行口径

首轮 runnable 版本收敛为一个最小增量 lane：

1. **base 从 `028A@1220` 改成 `028A@1060`**
   - `1220` 是 official 峰值，但 `500ep` capture 从 `0.844` 回落到 `0.796`
   - `1060 = 0.810 official / 0.806 capture` 更稳，更适合作为 reward 增量实验的基座
2. **首轮只启动 `033-A`**
   - spacing + coverage 的 team-level PBRS
   - 先回答“真正的 team-level relation reward 是否能站住”
3. **`033-A-control / 033-B / 033-C` 延后**
   - `033-A-control` 作为第二轮归因增强项
   - `033-B` pass event 和 `033-C` 组合都先等 A 的首轮 verdict

## 1. 核心假设

当前所有 reward shaping（v2 / v4 / PBRS / field-role binding）都是 **per-agent shaping**——每个 agent 独立计算自己的 reward，team-level wrapper 在最后求和。这类 shaping 的本质是"每个 agent 独立优化 sub-objective"。

假设：**真正的团队级 shaping**（关系型 reward，必须同时应用到两个 agent 上才有意义）可以让 policy 直接被 reward 告知"整体应该长什么样"。

例子：

| 团队级 shaping | 关系信号 | per-agent 版本做不到 |
|---|---|---|
| **Spacing** | 两人距离 | ✓（per-agent 只能看到自己相对球的位置）|
| **Coverage** | 前-后分工 | ✓ |
| **Pass event** | 球从 A → B within team | ✓ |
| **Formation** | 两人相对 ball 的形态 | ✓ |

这些 shaping 项不能拆成 per-agent reward——它们衡量的是**两个 agent 的相对关系**。team-level architecture 天然契合：一个 policy 看两人 obs、输出两人 action、共同收到 team reward。

## 2. 三条候选 lane

| lane | shaping 设计 | 特点 |
|---|---|---|
| **033-A** (Spacing + Coverage, PBRS 形式) | 位置关系 potential function | 理论安全（PBRS 保最优策略）|
| **033-B** (Pass event) | 离散事件 bonus | 直接奖励有意义的传球 |
| **033-C** (组合 A+B) | 两者叠加 | 最终完整版 |

**首轮先跑 033-A**（PBRS 形式最安全）。`033-A-control`、`033-B` 和 `033-C` 都等 A 出结果再决定。

## 3. 路径 A — Spacing + Coverage（PBRS 形式）

### 3.1 Shaping 公式

定义团队级 potential function：

```
Φ(s) = α_spacing × spacing_potential(agent_0, agent_1, ball)
     + α_coverage × coverage_potential(agent_0, agent_1, ball)
```

其中：

```
spacing_potential(a0, a1, ball) =
  let d_01 = |pos_a0 - pos_a1|
  let d_0b = |pos_a0 - pos_ball|, d_1b = |pos_a1 - pos_ball|
  let near_ball_one = min(d_0b, d_1b) < 3.0   # 至少一人靠近球
  if near_ball_one and 2.0 ≤ d_01 ≤ 6.0:
    return 1.0                                # 好分布
  else:
    return 0.0

coverage_potential(a0, a1, ball) =
  let x_a0, x_a1, x_ball = x坐标s
  let front = max(x_a0, x_a1), back = min(x_a0, x_a1)
  if front > x_ball and back < x_ball:
    return 1.0                                # 球夹在两人之间
  else:
    return 0.0
```

每步 shaping reward：

```
r_team_shaping = γ × Φ(s') - Φ(s)
```

按 PBRS 理论（Ng, Harada, Russell 1999），这个 shaping 不改变最优策略，只加速学习。

### 3.2 为什么这个设计抗 gaming

1. **Potential 形式**：PBRS 保证不改变最优策略。agent 不能通过"永远维持高 potential" 来刷分——每步 reward 是 **potential 变化量**，稳态时为 0
2. **Gate by ball proximity**：`spacing_potential` 有 `near_ball_one < 3.0` 的门控——如果两人都远离球，potential 为 0，不给 reward。防止"站场角刷 spacing"
3. **coverage 也绑 ball x 坐标**：球两边要各有一人，绑定 ball 位置——防止 agent 无视球跑位

### 3.3 参数

| 参数 | 值 | 理由 |
|---|---|---|
| `α_spacing` | 0.01 | 和 v2 `ball_progress_scale` 同量级 |
| `α_coverage` | 0.005 | coverage 触发条件更严，降 weight |
| `γ` | 0.99 | 和训练 gamma 一致（PBRS 要求）|
| `near_ball_threshold` | 3.0 | 经验值，1-iter smoke 后再调 |
| `spacing_good_range` | [2.0, 6.0] | 场地宽度约 30，合理 spacing |

最大单步累积 shaping ≈ `0.01 × 1.0 + 0.005 × 1.0 = 0.015`，远低于 sparse goal reward 的 ±3。

### 3.4 是否保留 v2 shaping

**首轮**：保留 v2 shaping + 叠加新的 team-level PBRS。原因：
- 让 033-A 的 baseline 是 028A（v2 shaping）
- 新 shaping 作为**增量**看是否有效
- 如果完全替换 v2，混淆"新 shaping 带来收益" vs "删除 v2 带来收益"

**后续**：如果 033-A 成立，可以考虑 033-D = team-level shaping only（删掉 v2）看能否独立成立。

## 4. 路径 B — Pass Event Reward

### 4.1 Shaping 公式

```
if possessing_team flipped from agent_0 to agent_1 within team0 (或反之):
    r_team_shaping += 0.05
```

需要在 `RewardShapingWrapper` 里跟踪 `prev_possessing_agent`，和 C-event shaping（snapshot-026 C）的 tackle 检测逻辑类似。

### 4.2 风险和防护

**主要 gaming 风险**：pass ping-pong

两个 agent 在后场来回传球刷 reward。两个防护：

1. **必须是推进性 pass**：只有 `ball_x > prev_pass_ball_x` 才算（球在向对方门方向传）
2. **Cooldown**：同一次 pass 在 20 步内不能再次触发

### 4.3 参数

| 参数 | 值 |
|---|---|
| `pass_reward` | 0.05 |
| `pass_forward_threshold` | `ball_dx > 0` AND `new_ball_x > prev_ball_x` |
| `pass_cooldown_steps` | 20 |

## 5. 路径 C — 组合（conditional）

如果 033-A 和 033-B 都单独有正结果，再试组合。首轮不做。

## 6. 预声明判据

### 6.1 主判据（033-A）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 500 | ≥ 0.85（接近/突破 028A official 峰值）| team-level shaping 带来真实收益 |
| H2H vs 028A@1060 | ≥ 0.55 | 对等对抗下超越更稳的 base |
| **failure capture `low_possession`** | ≤ 22% | 首次真正打破 per-agent 的 22-28% 不变量 |

### 6.2 机制判据

| 项 | 期望 |
|---|---|
| episode 内 avg spacing | 稳定在 `[2.0, 6.0]` 以内（说明 shaping 生效）|
| `late_defensive_collapse` | 不恶化（≤ 50%，和 028A 持平或更低）|
| `reward_mean` curve | 平稳上升，不出现 shaping exploit 导致的"reward 暴涨但 WR 停滞" |

### 6.3 失败判据（Gaming / Over-fit 防护）

| 条件 | 解读 |
|---|---|
| internal-official gap > 0.12 | gaming（policy 刷 shaping 但 real WR 不涨）|
| reward_mean 持续升但 WR 停滞 100 iter | classic shaping over-optimization，立即止损 |
| official 500 ≥ 0.85 但 H2H vs 028A@1060 < 0.50 | baseline-specific（new shaping 让 policy 更擅长打 baseline 但对同级弱）|
| failure capture 中 `poor_conversion` > 15% | "控住球但打不进"——可能是 spacing/coverage 让 agent 过度保守 |

### 6.4 Gaming 防护具体措施

1. **系数必须小**：每个 term 最大单步贡献 ≤ 0.015，总和 ≤ 0.02（snapshot-032 同样原则）
2. **Potential-based 形式**（033-A）：理论保证不改变最优策略
3. **Gate by ball proximity**：所有位置类 shaping 必须绑定 ball 位置
4. **Visual sanity check**：best ckpt 用 `soccer_twos.watch` 看 5 局，确认没有"站场角"、"无脑互传"等退化行为
5. **对照 lane**：`033-A-control` 保留为第二轮；首轮先用 H2H 与 failure capture 读机制
6. **H2H 硬门槛**：所有 SOTA 声明必须以 H2H 判定
7. **Random 对手也要测**：如果 vs random 开始掉到 < 0.98，说明基本能力受损，立即止损

## 7. 执行矩阵

| lane | 改动 | 预算 | 优先级 |
|---|---|---|---|
| **033-A (spacing + coverage PBRS)** | 团队级位置 shaping | ~6h | **首轮主线** |
| 033-A-control | 同配置但 `team_shaping_weight=0` | ~6h | 第二轮对照 |
| 033-B (pass event) | pass reward | ~6h | 条件启动 |
| 033-C (A+B combined) | 组合 | ~6h | 条件启动 |

## 8. 工程依赖

### 8.1 需要新增

- `soccer_info.py` 新增：
  - `compute_team_spacing_potential()`
  - `compute_team_coverage_potential()`
  - `compute_pass_event_reward()`（033-B）
- `RewardShapingWrapper` 在 `team_vs_policy` 模式下：
  - 对团队级 shaping term，在求和两人 reward 之后**再加团队级 shaping**（不是 per-agent 加完再求和）
  - prev_Φ 状态跟踪
- batch 新增 env vars：`SHAPING_TEAM_SPACING_SCALE` / `SHAPING_TEAM_COVERAGE_SCALE` / `SHAPING_PASS_REWARD`

### 8.2 可以复用

- 026 C-event shaping（`compute_event_shaping`）的 possession tracking 逻辑——033-B 可直接套
- 028A batch 作为模板

### 8.3 需要确认

- `team_vs_policy` wrapper 能否在 step 之间保留 `prev_Φ` 状态（PBRS 必需）
- baseline 的 agent 是 team-level 还是 per-agent？（影响 info 结构）

## 9. 和其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [026 C-event](snapshot-026-reward-liberation-ablation.md) | per-agent 版 event shaping，033-B 的 per-agent 模板 |
| [028A](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | base |
| [031](snapshot-031-team-level-native-dual-encoder-attention.md) | 正交 native 方向 |
| [032](snapshot-032-team-level-native-coordination-aux-loss.md) | 正交 native 方向 |

## 10. 不做的事

- 不做**非 potential-based 位置 shaping**（如直接给 "两人距离 > 2m 就 +0.01"）——这类非 PBRS 容易改变最优策略
- 不做**基于绝对位置的 shaping**（如"谁到达 x=5 给 reward"）——这是 spurious inductive bias
- 不做**需要 baseline 行为预测的 shaping**（太复杂且依赖 baseline）
- 首轮不删 v2 shaping（见 §3.4）

## 11. 执行清单

1. 实现 `soccer_info.py` 的三个新函数
2. 修改 `RewardShapingWrapper` 支持团队级 PBRS state
3. 新增 batch 环境变量 & batch 脚本
4. 1-iter smoke：
   - 训练 log 中能看到 `team_spacing_phi = 0.xx` 等字段
   - PBRS delta 的量级合理（约 0.01/step）
5. 起 033-A
6. 训练中 watch 视觉 sanity（5 局）
7. 300 iter 后做 official 500 + failure capture + H2H vs `028A@1060` / `029B`
8. 按 §6 判据 verdict

## 12. 相关

- [SNAPSHOT-026 C-warm](snapshot-026-reward-liberation-ablation.md)
- [SNAPSHOT-028: team-level BC base](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-031: dual encoder](snapshot-031-team-level-native-dual-encoder-attention.md)
- [SNAPSHOT-032: aux loss](snapshot-032-team-level-native-coordination-aux-loss.md)

## 13. 首轮实现与结果（033-A）

### 13.1 实验口径

- lane: `033-A = 028A@1060 + spacing/coverage PBRS`
- batch: [soccerstwos_h100_cpu32_team_level_033A_team_coord_pbrs_on_028A1060_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_033A_team_coord_pbrs_on_028A1060_512x512.batch)
- run: [033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015)

训练摘要：

- best reward mean: `+2.2091 @ iteration 105`
- final checkpoint: [checkpoint-300](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000300/checkpoint-300)
- best internal eval checkpoint: [checkpoint-240](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000240/checkpoint-240) `= 0.900 (45W-5L-0T)`

### 13.2 Internal `baseline 50` 读法

`checkpoint_eval.csv` 在 `10 -> 290` 上完整，无需回填。internal 高点分布如下：

| checkpoint | internal baseline | 备注 |
|---|---:|---|
| [240](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000240/checkpoint-240) | 0.900 | internal 绝对峰值 |
| [260](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000260/checkpoint-260) | 0.880 | 后段高点 |
| [60](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000060/checkpoint-60) | 0.860 | 早期高点 |
| [130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130) | 0.860 | 中早期高点 |
| [280](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000280/checkpoint-280) | 0.860 | 后段高点 |

只看 internal，会自然把主候选押到 `240/260` 一带；但 official 结果显示 `033-A` 的 late-window internal 高点被明显高估，因此这条线不能仅按 internal 峰值收口。

### 13.3 Official `baseline 500`

按 `top 5% + ties + ±2 window` 复核，得到的 official `baseline 500` 如下：

| checkpoint | official baseline 500 |
|---|---:|
| [40](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000040/checkpoint-40) | 0.824 |
| [50](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000050/checkpoint-50) | 0.816 |
| [60](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000060/checkpoint-60) | 0.766 |
| [70](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000070/checkpoint-70) | 0.772 |
| [80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80) | 0.826 |
| [110](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000110/checkpoint-110) | 0.770 |
| [120](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000120/checkpoint-120) | 0.804 |
| [130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130) | 0.826 |
| [140](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000140/checkpoint-140) | 0.778 |
| [150](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000150/checkpoint-150) | 0.810 |
| [220](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000220/checkpoint-220) | 0.814 |
| [230](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000230/checkpoint-230) | 0.816 |
| [240](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000240/checkpoint-240) | 0.818 |
| [250](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000250/checkpoint-250) | 0.794 |
| [260](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000260/checkpoint-260) | 0.802 |
| [270](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000270/checkpoint-270) | 0.798 |
| [280](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000280/checkpoint-280) | 0.782 |
| [290](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000290/checkpoint-290) | 0.798 |

首轮 official 最高点并列为：

- [checkpoint-80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80) `= 0.826`
- [checkpoint-130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130) `= 0.826`

相对地，internal 峰值 [checkpoint-240](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000240/checkpoint-240) `0.900` 在 official 只剩 `0.818`。这一点支持把 `033-A` 读成“早/中早期 narrow window 有效”，而不是“越训越强的 late-window shaping 线”。

### 13.4 Failure Capture

对两处 official 并列最高点补了 `500ep` failure capture：

| checkpoint | official | capture | gap |
|---|---:|---:|---:|
| [80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80) | 0.826 | 0.786 | -0.040 |
| [130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130) | 0.826 | 0.778 | -0.048 |

启发式 failure bucket（由保存的 episode JSON 文件名统计）：

| bucket | `80` | `130` |
|---|---:|---:|
| `late_defensive_collapse` | 55/107 | 43/111 |
| `low_possession` | 31/107 | 33/111 |
| `poor_conversion` | 7/107 | 14/111 |
| `unclear_loss` | 7/107 | 13/111 |
| `territory_loss` | 5/107 | 6/111 |
| `opponent_forward_progress` | 2/107 | 2/111 |

这组 capture 略偏向 [checkpoint-80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80)：

- official 同分下，`80` 的 capture 更高（`0.786 > 0.778`）
- `80` 的 `poor_conversion / unclear_loss` 更低

但 `130` 的 `late_defensive_collapse` 略低，因此两者并不是完全一边倒。

### 13.5 Head-to-Head vs `028A@1060`

为了避免只看 `vs baseline`，补了两条对 [028A@1060](snapshot-028-team-level-bc-to-ppo-bootstrap.md) 的 direct H2H：

| matchup | result |
|---|---:|
| [033A@130 vs 028A@1060](../../docs/experiments/artifacts/official-evals/headtohead/033A_130_vs_028A_1060.log) | `259W-241L = 0.518` |
| [033A@80 vs 028A@1060](../../docs/experiments/artifacts/official-evals/headtohead/033A_080_vs_028A_1060.log) | `241W-259L = 0.482` |

这说明 `033-A` 当前不是“整个高窗都稳定超过 `028A`”的线，而更像：

- [checkpoint-80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80)：在 `vs baseline` 上更稳
- [checkpoint-130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130)：在 direct H2H 上更好

### 13.6 首轮分析（克制版）

当前证据支持以下较保守的读法：

1. `033-A` 不是负结果。
   无论看 official 最高点（`0.826`）还是和 `028A@1060` 的 direct H2H（`130` 这点 `0.518`），都说明 team-level coordination PBRS 至少给出了真实信号。

2. 这条信号目前更像 **narrow positive signal**，而不是稳定升级。
   internal late-window 高点（尤其 `240/260`）在 official 上被明显高估；真正站得住的点更靠前。

3. 当前不宜武断写成“`033-A` 已经稳定超过 `028A`”。
   `130` 能小胜 `028A@1060`，但 `80` 又会小负；因此更稳的表述应是：
   - `033-A` **在部分窗口上** 对 `028A` 呈现边际增益
   - 但增益目前仍偏窄窗、且不同判据给出的主候选并不完全一致

4. 如果必须保留两个代表点：
   - **baseline-oriented candidate**: [checkpoint-80](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000080/checkpoint-80)
   - **H2H-oriented candidate**: [checkpoint-130](../../ray_results/033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015/TeamVsBaselineShapingPPOTrainer_Soccer_0beb6_00000_0_2026-04-18_05-50-35/checkpoint_000130/checkpoint-130)

首轮因此先收口为：

> `033-A` 为 team-level coordination reward 的首轮正结果，证据表明它在部分窗口上可能略优于 `028A@1060`，但当前提升仍偏窄窗且对不同判据较敏感，尚不宜写成稳定升级或强冠军挑战成功。

### 13.7 v2 失败桶交叉读（补充）

对 §13.4 的两个 failure capture 目录，用 [`failure_buckets_v2.classify_failure_v2`](../../cs8803drl/imitation/failure_buckets_v2.py) 离线重分（metric 在旧 JSON 里已有，无需重采）。和 [032 §13.2](snapshot-032-team-level-native-coordination-aux-loss.md#132-v2-失败桶分布failure-capture-500-ep) 的 032A/032Ac 数据并列：

| v2 Bucket | 033A @80 (L=107) | 033A @130 (L=111) | 032A @170 (L=87) | 032Ac @200 (L=86) |
|---|---:|---:|---:|---:|
| defensive_pin | 47.7% | **34.2%** | 47.1% | 41.9% |
| territorial_dominance | 49.5% | **36.0%** | 47.1% | 44.2% |
| wasted_possession | 42.1% | 42.3% | 46.0% | 41.9% |
| possession_stolen | 35.5% | 32.4% | 32.2% | 36.0% |
| progress_deficit | 26.2% | 22.5% | 23.0% | 20.9% |
| unclear_loss | 10.3% | 13.5% | 11.5% | 12.8% |

**观察**（不做强推断）：

1. **033A 内部的两个 checkpoint 分布差异比和 032 系列的差异更大**。`defensive_pin` / `territorial_dominance` 在 80 → 130 这 50 iter 内下降 ~13pp。这是 §13.3 "late-window internal 高点被高估" 的一个机制性呼应：late-window policy 花在自己半场的时间更少，但这没转换成 WR。
2. **`wasted_possession` 在 033A 两个窗口都稳定在 42%**，和 032 系列量级一致。说明这个维度在当前所有测试变体里都**没被明显动过**——这是多个 lane 共同的现象，但也**只是现象**，不证明该维度"不可改"。
3. **`possession_stolen` 量级**（32-35%）与 032 系列（32-36%）基本重叠，没看到 033A 的 PBRS 对"保球"有独立增益。

### 13.8 Episode length 观察（补充）

从 `summary.json`:

| Run | W mean | W median | L mean | L median | L/W (mean) |
|---|---:|---:|---:|---:|---:|
| 033A @80 | 49.5 | 39.0 | 37.0 | 27.0 | 0.75 |
| 033A @130 | 46.8 | 39.0 | **44.7** | **32.0** | **0.95** |
| 032A @170 | 46.4 | 37.0 | 40.5 | 30.0 | 0.87 |
| 032Ac @200 | 45.9 | 36.0 | 34.5 | 26.0 | 0.75 |

**观察**:

1. 033A **两个 checkpoint 的 L 长度差距很大**（37 → 44.7），远大于赢球长度变化（49.5 → 46.8）。和 §13.7 def_pin 下降合起来看：late-window policy 在中场拖得更久，但最后仍输掉。
2. 这个模式**只在 033A @130 上出现**；033A @80 的 L 长度（37）其实和 032Ac（34.5）很接近，不像 turtle。说明**把 "033A = turtle" 这种单一画像贴上去并不准确**——真要看 checkpoint。

### 13.9 与 032 的交叉读（谨慎版）

032 和 033 都改了 team-level coordination，分别从 **loss 层**和 **reward 层**。把它们放一起读时要注意：

**可以说的** —— 033A 和 032 系列**在 baseline 500 ep mean 附近都没跳出 028A@1060 的 ±2pp**:

| Run | 500ep mean | 500ep max | vs warmstart (0.806) |
|---|---:|---:|---:|
| 028A_1060 warmstart | — | 0.806 (base) | 0 |
| 033A (spacing+coverage PBRS) | 0.806 | 0.826 | +0.0 / +2.0 |
| 032A (aux=0.05) | 0.814 | 0.826 | +0.8 / +2.0 |
| 032Ac (aux=0.00) | 0.810 | 0.836 | +0.4 / +3.0 |

**不可以说的** —— 不能从这组数据推出:

- ❌ "coordination-focused interventions 整类路线无效"
- ❌ "team reward shaping 全都没用"
- ❌ "033A 的 0.0pp mean 改善证明 PBRS 错"

**原因**（为什么这些推论不成立）:

1. **033A 没有同期 control lane**。§0.5 明写 `033-A-control` 延后。如果没有"同架构/同训 budget 但 `team_shaping_weight=0`"的对照，当前 033A 的 mean 0.806 无法归因到"PBRS 无效"或"028A@1060 continuation 本身就是这样"
2. **033A 测的是 ONE POTENTIAL**（spacing + coverage）。§4 还有 033-B (pass event) 和 §5 的 033-C (组合) 没跑。Potential 空间很大，某一特定选择的 null 不证伪整类
3. **500ep SE ≈ 0.017**。在这个误差带内，+0.0 / +0.8 的 mean 差距不足以排序各变体，只能说"都在 028A warmstart ±2pp 带内"
4. **32A @170 vs 028A@1060 H2H 是 0.536 显著胜**，033A @130 vs 028A@1060 是 **0.518 边缘胜**（z=0.81，不显著但方向一致）。这个对比暗示 032A 的改动在 peer play 里信号更稳，但"PBRS 比 aux 弱"**不能从单 lane 推**——仍可能是 spacing/coverage 这一具体 potential 选择的问题，不是 PBRS 形式或 team reward shaping 路线的问题

### 13.10 本轮 033A 真正 update 了什么先验

在**不过度推广**的前提下，033A 让我们更可信地持有以下更新:

| 假设 | 更新方向 | 强度 |
|---|---|---|
| "50 ep peak 能预测 500 ep" | 再次否定 | 强（和 032A 独立验证，两次都回归 8-10pp） |
| "internal 高点即 official 高点" | 否定 | 强（§13.3 显式验证：240/260 从 0.900/0.880 回落到 0.818/0.802） |
| "spacing+coverage PBRS 是通往 9/10 的高确定性路径" | 弱化 | 中（当前窄窗证据不够强，但没被证伪） |
| "coordination-focused 干预是通往 9/10 的主线" | 略微弱化 | **弱**（两个独立 lane 各自正信号窄窗，但未交叉否证。需要 033-A-control / 033-B 定性归因） |
| "team reward shaping 整类路线" | **不 update** | — （样本不够） |

### 13.11 下一步（如果继续 033 线）

基于 §13.6 的两 candidate 结构和 §13.9 的归因缺口，优先级建议:

1. **033-A-control**（同训 budget，`team_shaping_weight=0`）—— 唯一能回答"033A 的窄窗增益是 PBRS 效应还是 028A continuation 效应"的实验
2. **033-B (pass event)** —— 独立的 reward 设计，不仅 potential 形式 也换 target（进攻过渡 vs 空间分布），能检验"是 potential 形式的问题还是 target 的问题"
3. 在上面两个都有结果之前，**避免**把 033A 单 lane 结果过度拟合到"PBRS/coordination 路线强/弱"的断言
