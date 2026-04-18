# SNAPSHOT-019: MAPPO+v2 Opponent-Pool Anchor-Heavy Rebalance

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 已完成首轮结果

## 1. 背景

[SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 用 60/15/15/10 (baseline / MAPPO+v2 anchor / MAPPO+v1 / MAPPO no-shape) 的 frozen pool 把 baseline-WR SOTA 从 MAPPO+v2 的 0.786 推到 0.812。该结果暗示：**降低 baseline 训练比例并不损失 baseline 胜率，反而可能略微提升**。

历史数据点：

| 训练对手分布 | vs baseline 500-ep WR |
|---|---|
| 100% baseline (MAPPO+v2) | 0.786 |
| 60% baseline + 40% peer pool (pool@290) | **0.812** (+0.026) |
| ? % baseline | ? |

这条趋势支持"**peer-pressure training 反向带高 baseline-WR**"假设：和等强 peer 对打逼出更细致的策略，这些策略在 baseline 上也用得上。但 60→0% 不是越低越好，存在 sweet spot。

本 snapshot 的目标是**验证从 60% 大幅降到 30% 是否仍维持或继续提升 baseline-WR**，作为 "low-baseline ratio" 假设的 anchor 实验。

## 2. 与 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的差别

唯一变量：**pool 配比**。其余全部保持。

| 维度 | snapshot-018 | snapshot-019 |
|---|---|---|
| baseline | 60% | **30%** ← 主要变化 |
| MAPPO+v2 anchor | 15% | **30%** ← 显著上调（anchor-heavy） |
| MAPPO+v1 | 15% | 20% |
| MAPPO no-shape | 10% | 20% |
| v4 PPO | — | — |
| Warm-start 源 | MAPPO+v2@470 | MAPPO+v2@470 (一致) |
| 训练脚本 | [train_ray_mappo_vs_opponent_pool.py](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py) | 同 (复用) |
| shaping | v2 | v2 (一致) |
| 模型 / γ / λ / batch | 一致 | 一致 |
| 训练量 | 300 iter | 300 iter |

**关键**：所有变量除了 pool 配比保持不变，确保差异完全归因于 baseline-ratio 大幅下调与 anchor 上调。

## 3. 假设

### 主假设

继续降 baseline 配比（60→30）不会损失 baseline-WR，且可能继续提升，因为：

1. baseline-specialized 行为已被 100% 与 60% 配比训练充分学到
2. peer pool 提供"等强对手互磨"信号，逼出更通用策略
3. 通用策略 transfers 回 baseline 评分

### 子假设（anchor-heavy 设计特有）

baseline gravity 减半时，**self-anchor 上调 2 倍** (15%→30%) 能稳住 policy 不漂离已学到的强策略，**双锚（baseline + self-anchor）合起来等价于 snapshot-018 的单锚 60% baseline 的稳定性**。

如果这个子假设成立，本 lane 等价于"用一半 baseline 信号 + 一半 self-anchor 信号换取 40% 真实 peer pressure 训练"。如果不成立（anchor 过高造成 self-overfitting，§5.3 失败 2），整个 anchor-heavy 方向需要降回到 anchor 20% 重试。

### 备择假设

降到 30% 跨过 sweet spot，policy 失去对 baseline 的 specialized 优势：

- 训练分布偏离评分目标
- baseline-WR 回退到 0.78-0.80 区间

### 这次实验 决定接下来怎么走

| 实测 vs baseline 500-ep | 解读 | 下一步 |
|---|---|---|
| ≥ 0.83 | "低 baseline 反向带高"成立且收益显著 | 020 跑同配比 + v4-shaping 5-member |
| 0.81-0.83 | 假设成立但收益小 | 020 仍跑，但调整预期 |
| 0.78-0.81 | sweet spot 在 30-60% 之间 | 反向试 50% baseline；放弃继续降 |
| < 0.78 | 30% 跨过 sweet spot，明确退化 | 回退到 50/20/15/15 配比；021 暂缓 |

## 4. 实验设计

### 4.1 Warm-start 源

- [MAPPO+v2 @ ckpt470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
- 与 snapshot-018 相同起点 → 干净对照"配比改动的纯效应"
- 不用 pool@290 起点的理由：避免"高起点被低 baseline 训练拉下"和"配比本身不行"无法分辨

### 4.2 Opponent pool 组成（anchor-heavy 重平衡）

| 成员 | 比例 | 角色 | 来源 |
|---|---|---|---|
| `ceia_baseline_agent` | **30%** | 评分对齐底仓（大幅下调） | frozen |
| MAPPO+v2 @ ckpt470 (anchor) | **30%** | 防漂 anchor（与 baseline 并列最大权重） | frozen |
| MAPPO+v1 @ ckpt490 | 20% | 跨 shaping diversity | frozen |
| MAPPO no-shape @ ckpt490 | 20% | 跨 lane diversity | frozen |

**关键设计理由**：

- **30% baseline + 30% anchor 双锚结构**——baseline 锚住"评分目标对齐"，anchor 锚住"已学到的强策略不漂"
- **anchor 30% 是 snapshot-018 的 2×**——baseline gravity 减半时，self-anchor gravity 必须显著上升，否则 policy 容易 drift 到 peer 风格 cluster
- **v1 / no-shape 各 20%（同等权重）**——本 lane**不**专门 target vs MAPPO+v1 的 0.488 head-to-head 弱点，那是 snapshot-020/021 的范畴；本 lane 只测"baseline-ratio 大幅降但 anchor 顶上"是否 work
- **不引入 v4 PPO 或新成员**——保持单变量对照，只测配比 rebalance 本身

### 4.3 训练配置

完全继承 snapshot-018，只改配比环境变量：

- `variation = multiagent_player`
- `multiagent = True`
- `FCNET_HIDDENS = 512,512`
- `custom_model = shared_cc_model`
- `gamma = 0.99, lambda = 0.95`
- shaping：v2 (`opp_progress_penalty=0.01, deep_zone_outer=-0.003, deep_zone_inner=-0.003`)
- rollout：`rollout_fragment_length=1000, train_batch_size=40000, num_sgd_iter=10`
- 300 iter, ~12M steps, ~5h walltime

新建 batch 脚本：`scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_v3_low_baseline_pool_512x512.batch`，在 snapshot-018 batch 基础上覆盖：

```
POOL_BASELINE_PROB=0.30
POOL_ANCHOR_PROB=0.30
POOL_V1_PROB=0.20
POOL_BS0_PROB=0.20
```

## 5. 预声明判据

### 5.1 主判据（验证 peer-pressure 假设）

**500-ep 官方 WR vs baseline ≥ 0.81**

- 等价于"30% baseline 配比至少不输 60% baseline 配比"
- 这个阈值和 snapshot-018 §5.1 一致 (≥ 0.81)
- 比 snapshot-018 通过线高一档的话需要 **≥ 0.83** 才能算"假设充分成立"

### 5.2 机制判据

**A — vs anchor 不出现 self-overfitting**：
- snapshot-018 pool@290 vs MAPPO+v2@470 anchor = 0.520
- 本 lane anchor 占比 30% (2× snapshot-018)，**风险**：policy 学到对自己镜像的 specialized 反应
- **预期 vs anchor 200-ep WR ∈ [0.45, 0.60]**——明显高于 0.60 = self-overfit；低于 0.45 = 训练失衡
- 这是 anchor-heavy 设计最大风险点的直接验证

**B — vs 其他 peer 不退化**：
- vs MAPPO+v1@490 ≥ 0.45（snapshot-018 是 0.488，本 lane v1 占比从 15→20，不应明显回退）
- vs MAPPO no-shape@490 ≥ 0.45（snapshot-018 是 0.532，本 lane no-shape 占比从 10→20，应持平或微升）

**C — 失败桶分布**（弱判据，仅作机制解读）：
- `low_possession` 占失败比仍预期 ≥ 22%（snapshot-013 §11 的跨 lane 不变量预测，本 lane 不修这个桶）
- `late_defensive_collapse` 应保持 ≤ 50%（snapshot-018 是 48%，不应明显回升）

### 5.3 失败情形

任一触发即视为本配比方向失败，**不再继续降 baseline ratio**：

1. **vs baseline 500-ep < 0.78** → 30% baseline 跨过 sweet spot，明确退化
2. **vs anchor 200-ep ≥ 0.65** → anchor-heavy 配比导致 self-overfitting，policy 学到的是对镜像的 specialized 反应而非通用强策略
3. **vs MAPPO+v1 200-ep < 0.40** → 训练分布失衡导致针对 v1 的策略反而恶化
4. **vs random 200-ep < 0.95** → policy 丢失基础能力（极不可能但要测）

## 6. 风险

### R1 — sweet spot 可能在 40-50% 而不是 30%

snapshot-018 60% → 0.812，但 60→30 是 -50% 跨度。可能 sweet spot 在 40-50% 区间，30% 已跨过。如果 §5.3 失败 1 触发，下一轮回退到 50% 而不是继续降。

### R2 — peer training 不 transfer 回 baseline

"和 peer 对打学到的细致策略反向带高 baseline" 是理论假设，未被严格证实。可能 peer-学到的细致策略只针对 peer 风格，不 transfer。如果实测 vs baseline 0.78-0.80 但 vs MAPPO+v1 ≥ 0.55，说明假设方向对但 transfer 不发生——这种情况下本 lane 仍有 ensemble 价值。

### R3 — anchor 30% 是本 lane 最大风险点

snapshot-018 anchor 是 15%；本 lane 升到 30%（**2 倍**）。anchor 太像自己 → policy 学到对自己镜像的 specialized 反应，反而对真正不同的 baseline 策略无用。

这是 anchor-heavy 设计的 **核心 bet**：30% baseline + 30% anchor 双锚假设是"baseline 锚 + 自我锚两个 stable 信号合起来够防漂"，但若 anchor 实际起反作用，整个 lane 失败。

缓解：§5.3 失败 2 直接预声明 vs anchor ≥ 0.65 = 触发，强制识别 self-overfitting；§5.2 A 把 anchor 的 WR 区间收紧到 [0.45, 0.60] 作为持续监控。

### R4 — 没有 v4 / 跨架构成员

本 anchor 实验**有意**不加 v4 PPO，把跨 shaping 变量留到 [SNAPSHOT-020](snapshot-020-mappo-v4-fair-ablation.md)。所以本 lane 即使过主判据，也不能独立得出"配比 + 跨 shaping 都重要"的结论；只能说"配比一项贡献正向"。

## 7. 不做的事（明确边界）

- **不加 v4 PPO 或任何非 MAPPO frozen 成员**（留给 snapshot-020）
- **不改 shaping**（v2 冻结）
- **不改架构** / γ / λ / model size / obs space
- **不开 rolling self snapshot**（snapshot-018 §6 R3 推迟到 Phase-2，本 lane 同样）
- **不做 BC pre-train 叠加**（那是 snapshot-017）
- **不延长训练到 > 300 iter**（diminishing returns 区间，延长不解决配比问题）

## 8. 与并行 lane 的关系

本 snapshot 的设计逻辑是**"anchor"实验**——它给出"低 baseline 配比是否有用"这个核心问题的纯净答案：

- [SNAPSHOT-020 (planned)](snapshot-020-mappo-v4-fair-ablation.md) 将训"MAPPO+v4-shaping"作为 v4 PPO 替代品，加入 5-成员 pool 跑同配比 (30/20/20/20/10 分配)
- [SNAPSHOT-023 (conditional)](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) 仅在 020 显示"跨 shaping 风格也有正贡献"时启动，做 FrozenTeamCheckpointPolicy 适配让真 v4 PPO 进 pool（**注**：原计划编号 021，2026-04-15 021/022 被重用于 obs/reward 假设检验后移至 023）
- 三条线的归因逻辑：
  - 019 ✓ 020 ✗ → 配比是关键，跨风格成员加进来反而扰乱
  - 019 ✗ 020 ✓ → 单纯降 baseline 不够，必须配跨风格成员
  - 019 ✓ 020 ✓ → 两者独立有效，可叠加；023 值得做
  - 019 ✗ 020 ✗ → 低 baseline 整体方向错，回 60% 配比

## 9. 相关

- [SNAPSHOT-013: baseline weakness analysis](snapshot-013-baseline-weakness-analysis.md)（low_possession 跨 lane 不变量诊断 + §11 修订）
- [SNAPSHOT-014: MAPPO 公平对照](snapshot-014-mappo-fair-ablation.md)（warm-start 源）
- [SNAPSHOT-016: v4 shaping ablation](snapshot-016-shaping-v4-survival-anti-rush-ablation.md)（v4-shaping 设计源）
- [SNAPSHOT-017: BC -> MAPPO bootstrap](snapshot-017-bc-to-mappo-bootstrap.md)（并行 / 互补路线）
- [SNAPSHOT-018: opponent pool first run](snapshot-018-mappo-v2-opponent-pool-finetune.md)（直接 parent，配比 60% baseline）

## 10. 下一步执行清单

1. 复制 `scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_v2_opponent_pool_512x512.batch` 为 `..._mappo_v3_low_baseline_pool_512x512.batch`
2. 修改新 batch 的 `POOL_*_PROB` 环境变量到 30/25/25/20
3. 修改 `RUN_NAME` 前缀（避免和 snapshot-018 run dir 冲突）
4. 1-iter smoke 确认配比生效（pool sampling log 应显示新比例）
5. 启动 300 iter 训练（~5h H100）
6. `top 5% + ties → baseline 500` 选模
7. 对 best ckpt 做 §5.2 head-to-head + failure capture
8. 按 §3 决策表确定接下来动作
9. verdict 落本文件 §11+ (append-only)

## 11. 首轮结果（原始 run + 续跑 run 合并口径）

### 11.1 运行目录

- 原始 run：
  - [PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221)
- 续跑 run：
  - [PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337)

### 11.2 原始 run 结果

- 原始 run 在 `iteration 147` 中断：
  - `done = False`
  - `timesteps_total = 5.88M`
  - `episode_reward_mean ≈ -0.984`
- 训练内 `baseline 50` 最强窗口出现在很早的前段：
  - [checkpoint-40](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000040/checkpoint-40) = `0.88`
  - [checkpoint-60](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000060/checkpoint-60) = `0.86`
  - [checkpoint-140](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000140/checkpoint-140) = `0.84`

### 11.3 续跑 run 结果

- 续跑 run 正常结束：
  - `iteration 300`
  - `done = True`
  - `timesteps_total = 12.0M`
  - `episode_reward_mean ≈ -1.003`
- 续跑内最好的训练内 eval 为：
  - [checkpoint-230](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000230/checkpoint-230) = `0.84`
- 其他相对较好的点：
  - [checkpoint-210](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000210/checkpoint-210) = `0.80`
  - [checkpoint-260](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000260/checkpoint-260) = `0.82`
  - [checkpoint-280](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000280/checkpoint-280) = `0.82`

### 11.4 合并解读

- `019` 不是坏线，但它的强点非常明显地前移。
- 最强内部窗口仍然停留在原始 run 的早段：
  - `40 = 0.88`
  - `60 = 0.86`
- 续跑虽然把训练完整跑到了 `300 iter`，但没有超过 scratch 前段高点。
- 而且这条线和 `020` 不同：续跑没有带来更好的 reward 或更高的 eval，高点反而回落到 `0.84` 附近。

### 11.5 官方 `baseline 500` 复核

正式复核结果如下：

- [checkpoint-40](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000040/checkpoint-40) = `0.788`
- [checkpoint-60](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000060/checkpoint-60) = `0.746`
- [checkpoint-140](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000140/checkpoint-140) = `0.788`
- [checkpoint-230](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000230/checkpoint-230) = `0.776`

这说明：

- `019` 的“早段强窗口”确实能兑现到正式 `500`，但没有超过 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的 `0.812`
- 最佳正式分数由原始 run 的 `checkpoint-40 / checkpoint-140` 并列给出
- 续跑点没有带来更强的正式结果

### 11.6 Failure Capture

四个代表点的 failure capture 如下：

- [checkpoint-40](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000040/checkpoint-40)
  - capture `win_rate = 0.806`
  - primary: `late_defensive_collapse = 49`, `low_possession = 27`, `poor_conversion = 11`, `unclear_loss = 8`
- [checkpoint-140](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000140/checkpoint-140)
  - capture `win_rate = 0.760`
  - primary: `late_defensive_collapse = 54`, `low_possession = 34`, `unclear_loss = 19`
- [checkpoint-230](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000230/checkpoint-230)
  - capture `win_rate = 0.782`
  - primary: `late_defensive_collapse = 53`, `low_possession = 33`, `poor_conversion = 8`, `unclear_loss = 9`

从失败结构看：

- `checkpoint-40` 是这条线里最有价值的点，不只官方分最高之一，failure capture 也最好
- `checkpoint-140` 明显体现了“继续训把早段优势磨掉”
- `checkpoint-230` 比 `140` 略收回来一点，但仍未回到 `40` 的水平

### 11.7 当前结论

`019` 更像一条“前段很猛、后续会回落”的 lane：

- aggressive anchor-heavy pool 可能确实能快速推高早期 baseline performance
- 但继续训练下去，没有把这个优势稳定成更高平台

当前最值得正式 `baseline 500` 复核的点为：

- 第一优先：
  - [checkpoint-40](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000040/checkpoint-40)
- 第二优先：
  - [checkpoint-140](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/MAPPOVsOpponentPoolTrainer_Soccer_b2478_00000_0_2026-04-15_03-42-47/checkpoint_000140/checkpoint-140)
- 补充参考：
  - [checkpoint-230](../../ray_results/PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/MAPPOVsOpponentPoolTrainer_Soccer_d94ad_00000_0_2026-04-15_15-54-01/checkpoint_000230/checkpoint-230)
