# SNAPSHOT-031: Team-Level Native Dual Encoder + Cross-Agent Attention

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册 / 首轮实现已就绪（031-A）

## 0. 方法论位置

[SNAPSHOT-030](snapshot-030-team-level-advanced-shaping-chain.md) 只做 per-agent advanced shaping trick 的 port。本 snapshot 开始 **team-level native 方法** 系列的第一条——**网络架构层面**让模型显式利用联合 obs 的结构。

team-level native 系列分三个独立 snapshot：

| snapshot | 维度 | 改动位置 |
|---|---|---|
| **031（本）** | 网络架构 | forward pass 结构 |
| [032](snapshot-032-team-level-native-coordination-aux-loss.md) | 训练目标 | loss function |
| [033](snapshot-033-team-level-native-coordination-reward.md) | 环境/reward | shaping 信号 |

三者正交，原则上可以叠加，但首轮独立测。

## 0.5 首轮可执行口径

首轮 runnable 版本先按最小工程风险落地：

1. **只启动 `031-A`**
   - 先跑 Siamese dual encoder scratch
   - `031-B` 的 cross-agent attention 和 `031-C` 的 independent encoder 继续保留为条件 lane
2. **首轮不做 `028A` warm-start**
   - 原预注册里把 `028A@1220` 当作过一个 team-level base 候选
   - 但当前更可信的 `028A` 主候选已经收口到 `1060`，而 flat MLP checkpoint 也不能直接 warm-start 到 Siamese 结构
   - 因此首轮改成 **scratch + 长预算**，先回答“native dual encoder 本身是否站得住”
3. **判据同步对齐更稳的 base**
   - H2H 基准由 `028A@1220` 改为 `028A@1060`
   - official 判据不再追逐 `1220` 的 late spike，而是先要求至少达到 `028A@1060` 的稳定平台

## 1. 核心假设

当前 028A 用 flat MLP 处理 672 维联合 obs：

```
concat(obs_agent_0, obs_agent_1)  →  Linear(672→512)  →  ReLU  →  Linear(512→512)  →  policy/value
```

flat MLP 对 672 维里的"哪 336 维来自 agent 0"这个结构是隐式的——网络必须从数据里学出这个对齐。假设：**把 obs 结构显式编码进网络架构能加速学习并提升上限**。

具体地：两个 agent 的 obs 在物理语义上**同构**（都是 egocentric 336 维：raycast + 自身 state + 球/门相对位置）。**Siamese dual encoder** 强制两个 agent 共享同一个特征提取函数，这会让模型学到"一个 agent 在情况 X 下应该编码为特征 Y"的通用规则，而不是分别学两套 slot-specific 规则。

在此之上，**cross-agent attention** 让策略在每步显式决定"当前情况下应该多关注队友的哪些信息"。

## 2. 三条候选 lane

| lane | 结构 | 复杂度 |
|---|---|---|
| **031-A**（主线）| Siamese dual encoder，无 attention | 中 |
| **031-B**（conditional on A 成立）| 031-A + cross-agent attention | 中高 |
| **031-C**（对照 / sanity）| Independent dual encoder（不共享权重）| 中 |

首轮先跑 **031-A**。如果 031-A 显著改善（H2H vs 028A ≥ 0.55），再考虑 031-B 叠加。031-C 只在 031-A 出现 weight-sharing collapse 问题时作为对照启用。

## 3. 路径 A — Siamese Dual Encoder

### 3.1 网络结构

```
obs_agent_0 (336 dim) ──► Encoder_shared ──► feat_0 (256 dim)
                                        │
                                        ▼
obs_agent_1 (336 dim) ──► Encoder_shared ──► feat_1 (256 dim)
                                        │
                                        ▼
                            concat(feat_0, feat_1)  →  Merge MLP → policy head (MultiDiscrete([3,3,3,3,3,3]))
                                                                → value head
```

- `Encoder_shared`: `Linear(336→256) → ReLU → Linear(256→256) → ReLU`
- **两个 encoder branch 的 weight 完全共享**（siamese，Python 层面是同一个 `nn.Module` 实例被调用两次）
- `Merge MLP`: `Linear(512→256) → ReLU → Linear(256→128) → ReLU`
- 总参数量和 `[512, 512]` flat MLP 相当（两者都是约 700K params）

### 3.2 为什么用 Siamese 而不是 Independent

agent 0 / agent 1 的 slot 语义不稳定——每局 spawn 位置可能互换。如果两个 encoder 独立权重：

- 模型会学到 slot-specific 特征，但遇到 spawn swap 时需要"重新理解" agent 0 的身份
- 等价于模型被迫学两份几乎一样的功能 → 样本效率降半

Siamese 共享权重强制学到 slot-invariant 的特征。Merge 层单独负责"关系建模"。这是 CV 里 siamese network 的标准动机。

### 3.3 工程依赖

RLlib 1.4.0 的 PPO 默认使用 `FullyConnectedNetwork`（flat MLP）。需要自定义 model：

- 继承 `TorchModelV2`
- `forward` 方法拆分 obs → siamese → merge
- 注册到 `ModelCatalog` 并在 `trainer_config["model"]["custom_model"]` 指定

代码复用：`cs8803drl/branches/shared_central_critic.py` 有过 custom model 先例，可以参考它的注册 pattern。

### 3.4 超参

初期用最小扰动：

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 无（scratch） | 架构不兼容，首轮不用 028A checkpoint |
| encoder hidden | 256×2 | 参数量匹配 |
| merge hidden | 256, 128 | 匹配 |
| lr / sgd_iter / clip | 1e-4 / 4 / 0.15 | 028A 同款 |

**Warm-start 路径特殊**：028A 的 weight 是 flat MLP 格式，不能直接 load 到 siamese encoder。两个选择：

- **A1**: scratch——接受更多训练预算（50M 步）
- **A2**: BC init——先用 028A BC dataset 重新训 BC（但 BC trainer 用新 siamese encoder 架构），再 PPO warm-start

**首轮 runnable 版本已明确走 A1**。A2 保留为二轮增强项。

## 4. 路径 B — Cross-Agent Attention（conditional）

### 4.1 结构（在 031-A 基础上加）

```
feat_0, feat_1 from siamese encoder

Q_0 = Linear(feat_0)
K_1, V_1 = Linear(feat_1), Linear(feat_1)
attn_0 = softmax(Q_0 · K_1 / √d) · V_1   # agent 0 attends to agent 1

(对称做 attn_1)

Merge(feat_0, attn_0, feat_1, attn_1) → policy/value
```

每个 agent 通过 attention 显式决定"当前情况下队友 obs 的哪些部分最相关"。

### 4.2 何时启动

**仅在 031-A 出正结果时启动 031-B**。如果 031-A 都不如 028A，加 attention 不会救它。

## 5. 预声明判据

### 5.1 031-A 主判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 500 | ≥ 0.81 | 至少不低于 `028A@1060` 的稳定水平 |
| H2H vs 028A@1060 | ≥ 0.52 | 显式证明架构改动带来对等对抗优势 |
| Training curve | reward_mean 到 +2.0 的收敛 iter 数 ≤ 028A | 样本效率不差于 flat MLP |

### 5.2 031-A 机制判据

- encoder feature cos_sim（随机 sampled obs batch）在 [0.3, 0.9] 之间（避免 siamese collapse）
- attention entropy（如果 031-B 启动）> `ln(2) ≈ 0.69`

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| official 500 < 0.80 | 架构 scratch 训太慢 → 需要 A2（BC init）或放弃 |
| official 500 ≥ 0.84 但 H2H vs 028A < 0.50 | 架构改动没带来真实技能提升，只是在 baseline 上"学了等价策略" |
| feature cos_sim > 0.95 | siamese encoder collapse，两 branch 输出几乎相同，等价于 flat MLP |

### 5.4 Gaming 防护

网络架构改动本身很难 gaming baseline WR——没有直接的 reward exploit 路径。主要风险是**训练质量退化**（siamese collapse / attention collapse）：

1. 每 50 iter 监控一次 encoder feature 的随机 batch cos_sim
2. 如果 > 0.9 持续 100 iter，增加 encoder hidden 宽度或使用 dropout

**不需要**对抗 gaming：031 的判据已经把"H2H ≥ 0.52" 作为硬门槛，架构改动的价值必须通过对等对抗证明。

## 6. 执行矩阵

| lane | 预算 | 优先级 |
|---|---|---|
| **031-A (siamese scratch)** | 50M steps / ~16h | **首轮主线** |
| 031-B (+ attention) | +50M steps / ~16h | 条件启动 |
| 031-C (independent encoder) | 50M steps / ~16h | sanity 用 |

## 7. 和其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [027](snapshot-027-team-level-ppo-coordination.md) | team-level scratch base |
| [028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | team-level BC base + 031 的对照 |
| [030](snapshot-030-team-level-advanced-shaping-chain.md) | port 实验；031 是 native 方法 |
| [032](snapshot-032-team-level-native-coordination-aux-loss.md) | 同属 native 系列，改训练目标 |
| [033](snapshot-033-team-level-native-coordination-reward.md) | 同属 native 系列，改 reward |

## 8. 不做的事

- 不做 LSTM / transformer 等时序结构（需要重新设计 env wrapping）
- 不做 graph neural network（agent 数量 = 2，图结构退化）
- 不做 centralized critic 在 team-level 上的变体（actor 已经看到全局 obs，CC 冗余）

## 9. 执行清单

1. 实现 `SiameseTeamModel`（自定义 `TorchModelV2`）并注册 `ModelCatalog`
2. 1-iter smoke：确认 forward pass shape 正确、loss 可以 backprop、两 branch 输出不相同
3. 起 031-A batch，50M steps scratch
4. 训练中监控 feature cos_sim（每 50 iter）
5. 按 §5 判据做 verdict
6. 决定 031-B / 031-C 是否启动

## 11. 首轮结果（031A，2026-04-18）

### 11.1 训练完成 Training Summary（一手存档）

031A scratch dual-encoder team-level，跑了 **1250 iter**（其他 lane 一般 200-300）。完整 print summary 直接归档:

```
Training Summary
  stop_reason:      Trial status: TERMINATED
  run_dir:          ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948
  best_reward_mean: +2.2259 @ iteration 1069
  best_checkpoint:  .../checkpoint_001140/checkpoint-1140
  final_checkpoint: Checkpoint(persistent, .../checkpoint_001250/checkpoint-1250)
  best_eval_checkpoint: .../checkpoint_001040/checkpoint-1040
  best_eval_baseline:  0.940 (47W-3L-0T) @ iteration 1040
  best_eval_random:    1.000 (50W-0L-0T)
  eval_results_csv:    .../checkpoint_eval.csv
  loss_curve_file:     .../training_loss_curve.png
Done training
```

`best_checkpoint_by_eval.txt`:

```
checkpoint_iteration=1040
baseline_win_rate=0.940
baseline_record=47W-3L-0T
random_win_rate=1.000
random_record=50W-0L-0T
```

### 11.2 50ep 内评分布

- n = 124 evals (iter 10..1240)
- mean **0.7745**, median 0.82, **max 0.940**
- evals ≥ 0.94: 5 (iter 580 / 800 / 1000 / 1040 / 1170)
- evals ≥ 0.92: 8
- evals ≥ 0.90: 13
- 后段 iter ≥ 800 mean **0.853**（远高于其他 team-level lane）

5 个 ckpt 都达 0.940 不是单点抽样运气；分布在整条曲线 late window。

### 11.3 1000ep eval（已完成）

| ckpt | 50ep | 1000ep | regression |
|---:|---:|---:|---:|
| 580 | 0.940 | 0.812 | -12.8pp |
| 770 | 0.920 | 0.840 | -8.0pp |
| 800 | 0.940 | 0.850 | -9.0pp |
| 930 | 0.920 | 0.841 | -7.9pp |
| 1000 | 0.940 | 0.852 | -8.8pp |
| **1040** | 0.940 | **0.867** ← top | -7.3pp |
| **1170** | 0.940 | **0.865** | -7.5pp |

**汇总**: mean **0.847**, max **0.867 @ ckpt 1040**

### 11.4 全 frontier 对照表（031A 是项目首个 1000ep max ≥ 0.86 的非临时点）

| Run | 1000ep mean | 1000ep max | vs 029B@190 (warmstart 0.846) |
|---|---:|---:|---:|
| **031A** (scratch dual-encoder) | **0.847** | **0.867** ← project top | **+2.1pp on max, mean ≈** |
| 036D (per-agent learned reward) | 0.843 | 0.860 | +1.4pp |
| 029B@190 (per-agent SOTA, warmstart) | — | 0.846 | base |
| 039 (refresh broken ≈ 036D-style) | 0.829 | 0.843 | -0.3pp |
| 036C (orig learned reward) | 0.824 | 0.833 | -1.3pp |
| 028A@1060 (team-level base) | — | 0.783 | -6.3pp |

**031A 是项目首个 1000ep max 上明确超过 029B@190 warmstart 的结果**。两个 ckpt (1040 / 1170) 都 ≥ 0.865 不是单点。

### 11.5 1000ep rerun（已完成，2026-04-18 ~21:30）

| ckpt | 1st 1000ep | 2nd 1000ep | 2000-game avg | regression |
|---:|---:|---:|---:|---:|
| 1040 | 0.867 | 0.853 | **0.860** | -1.4pp |
| 1170 | 0.865 | 0.850 | **0.858** | -1.5pp |

`0.867 max` 部分是单次抽样上限。真实 WR ≈ **0.860** ± 0.011 (SE)。

### 11.6 H2H 1000ep vs 029B@190 / 025b@080 / 028A@1060（**关键验证，已完成**）

| matchup | sample | 031A wins | opp wins | 031A rate | z | p (one-sided) |
|---|---:|---:|---:|---:|---:|---:|
| **031A@1040 vs 029B@190** | **1000** | **552** | 448 | **0.552** | **3.290** | **0.0005 `***`** |
| **031A@1040 vs 025b@080** | **1000** | **532** | 468 | **0.532** | **2.024** | **0.0215 `*`** |
| **031A@1040 vs 028A@1060** | **1000** | **568** | 432 | **0.568** | **4.301** | **<0.0001 `***`** |

侧别:
- vs 029B: blue 0.580 / orange 0.524（都 >0.5，无侧别运气）
- vs 025b: blue 0.538 / orange 0.526（都 >0.5，无侧别运气）
- vs 028A: blue 0.606 / orange 0.530（都 >0.5；blue 侧不对称 +7.6pp，但方向一致）

**031A 同时显著击败 frontier 三个候选**，是项目第一个对所有 frontier baseline 都 H2H 同向赢的 lane：
- vs 029B@190 (per-agent baseline-axis 头名): 0.552, p=0.0005, **`***`**
- vs 025b@080 (per-agent peer-axis 隐性头名): 0.532, p=0.022, `*`
- **vs 028A@1060 (team-level 同 reward 同 base 候选): 0.568, p<0.0001, `***`** ← 三连最强

#### 11.6.1 vs 028A@1060 的特殊意义（同 reward 同 base 直接架构对比）

`028A@1060` 是 031A **可比性最强**的对照：

| 维度 | 028A@1060 | 031A@1040 | 同/异 |
|---|---|---|---|
| reward | v2 shaping (ball_progress + opp_progress_penalty + possession + time_penalty + deep_zone) | **完全一样** | 同 |
| trainer | TeamVsBaselineShapingPPOTrainer | TeamVsBaselineShapingPPOTrainer | 同 |
| obs space | 672 维 team-level | 672 维 team-level | 同 |
| 网络架构 | flat MLP `[512,512]` (~700K params) | **Siamese encoder `[256,256]×2 share + merge [256,128]`** (~700K params) | **改了** |
| 训练起点 | BC@1060 warmstart from team-level BC | scratch | **改了** |
| 训练时长 | 1060 iter from BC | 1250 iter from scratch | 改了 |

`031A vs 028A = 0.568` 给的最干净结论：**架构改造（Siamese dual encoder）确实带来真实 H2H 优势**。+6.8pp 的 H2H 增益对应 baseline 1000ep `0.860 vs 0.783 = +7.7pp` 的差距——两个轴方向一致且数量级相当，这是架构改造正向的两条独立证据。

**caveat（保留）**: 031A 训练时长 1250 iter，028A 是 BC bootstrap + 1060 iter PPO；不能完全分离 "架构贡献" 和 "训练时长贡献"。要严格分离需要做 028A 长跑到 1250 iter 同步对照（见 [§11.10](#1110-下一步按-roi)）。但 031A 在 H2H 上对所有 frontier 都赢，至少证明这条路径**有效**。

和 [snapshot-036D §10.6.7](snapshot-036d-learned-reward-stability-fix.md) 的 036D@150 vs 029B = 0.507 (tied) 形成强对比 —— **031A 是 peer-axis 上首个真正突破 frontier 的 lane**。

### 11.7 Failure capture v2 bucket（已完成）

| Bucket | 031A@1040 | 036D@150 | 029B@190 | 030D@320 |
|---|---:|---:|---:|---:|
| defensive_pin | 47.1% | 54.5% | 46.3% | 46.7% |
| territorial_dominance | 50.6% | 55.7% | 45.6% | 47.8% |
| wasted_possession | 42.4% | 38.6% | 47.8% | 37.8% |
| possession_stolen | 32.9% | 38.6% | 32.4% | 36.7% |
| **progress_deficit** | **15.3%** ← lowest | 21.6% | 22.8% | 27.8% |
| unclear_loss | 12.9% | 10.2% | 11.0% | 11.1% |

L episode steps: mean 34.4, median 30, max 116（无 turtle，结构正常）。

**关键观察**:
- `progress_deficit` **15.3%** 是全 frontier 最低 —— 031A **最善于阻止对手推进**（dual-encoder 架构帮 policy 更好地解读对手位置/速度信息）
- `wasted_possession` 42.4% 比 029B 47.8% 低 -5.4pp（弱于 036D 的 -9.2pp，但仍有改善）
- 没有 reward gaming pathology

### 11.8 当前可信结论（更新版）

031A 是**项目新 SOTA**:

- ✓ baseline 1000ep avg **0.860** > 029B@190 0.846 (+1.4pp on max scale)
- ✓ peer H2H vs 029B@190 = **0.552** (n=1000, z=3.29, p=0.0005, **rock-solid**)
- ✓ peer H2H vs 025b@080 = **0.532** (n=1000, z=2.02, p=0.022, `*` 边缘显著但方向稳定)
- ✓ peer H2H vs 028A@1060 = **0.568** (n=1000, z=4.30, p<0.0001, **`***` 同 reward 同架构类型最干净对照**)
- ✓ failure structure: progress_deficit 项目最低，无 turtle / 无 gaming
- ✓ 训练 stability: 无 inf 问题（不是 learned-reward lane）
- ✓ scratch trained 1250 iter，不依赖任何 warmstart

**这是首个同时在 baseline 和 peer 两个判据上都击败 frontier 三个候选 (029B@190 + 025b@080 + 028A@1060) 的 lane**。

### 11.9 caveat（保留）

- 031A 是 **scratch dual-encoder + team-level**，和 029B（per-agent + warmstart）是**完全不同的方法栈**。"突破 029B" 不能简单解读为"team-level 架构本质更强" —— [SNAPSHOT-037 RETRACTED](snapshot-037-architecture-dominance-peer-play.md) 教训仍适用：单一 H2H 不能定整个架构类的优势
- 031A 和 029B 训练时长 / 算力差异大（031A 14h scratch vs 029B B-warm handoff），公平性需要细想
- 031A vs 025b@080 ✓ **已完成 (0.532, p=0.022)**；vs 028A@1060 ✓ **已完成 (0.568, p<0.0001)**；vs 017@2100 / 030D@320 H2H **未测**
- 031A 仅有 ckpt 1040 测了 H2H；ckpt 1170 (0.858 1000ep) 未测，可能更强或更弱
- **架构 vs 训练时长贡献分离**仍未做：031A 跑 1250 iter，028A 1060 iter 来自 BC bootstrap。要严格分离需要 028A 长跑 1250 iter 同步对照

### 11.10 下一步（按 ROI）

1. ~~**031A@1040 vs 025b@080 H2H**~~ ✓ 已完成 0.532, `*`
2. ~~**031A@1040 vs 028A@1060 H2H**~~ ✓ 已完成 0.568, `***` (架构 H2H 已坐实)
3. **触发 [SNAPSHOT-040 (Stage 2 shaping handoff on 031A base)](snapshot-040-team-level-stage2-on-031A.md)** — 031A 已确认是合格 base，040B (PBRS handoff) 是预期 ROI 最高的下一步
4. **触发 [§12 (031-B cross-attention)](#12-031-b-激活cross-attention2026-04-18-预注册)** — 预注册条件已满足
5. **031A@1170 vs 029B@190 H2H** — second peak verify
6. **031A@1040 vs 036D@150 H2H** — frontier 内部排序
7. （可选）**028A 长跑 1250 iter** — 严格分离架构 vs 训练时长贡献

### 11.4 注意

- 031A 是 **scratch 训练**（不是 028A warmstart），网络是 SiameseTeamModel
- 训练时间 ~14h，远长于其他 lane
- 不是 learned reward 路线，没有 inf 问题（team-level shaping trainer）

## 12. 031-B 激活（cross-attention，2026-04-18 预注册）

### 12.1 激活条件已满足

§5 预注册原则：**仅在 031-A 出正结果时启动 031-B**。031A 已坐实（[§11.8](#118-当前可信结论更新版)）：
- ✓ baseline 1000ep avg `0.860` > 029B `0.846`
- ✓ peer H2H vs 029B@190 = `0.552` (`***`)
- ✓ peer H2H vs 025b@080 = `0.532` (`*`)

满足激活条件。**031-B 进入预注册执行队列**。

### 12.2 设计修正（vs 原始 §4 预注册）

**实施时发现 §4 原始设计是 degenerate**: 把 encoder 输出当成单 token (`(batch, encoder_hidden)`)，那么 `Q_0 · K_1` 是单标量、softmax 永远 = 1.0，attention **没有实际选择**——退化成"用 V_1 替代 feat_1" 的纯线性投影。

**修正后的设计**: 把 encoder 输出 `(batch, 256)` reshape 成 **`(batch, n_tokens=4, head_dim=64)`** —— 每个 agent 有 4 个"特征位"。这样 `Q_0 · K_1^T` 是 `(4 × 4)` 矩阵，softmax 在 4 个 token 上**有真实选择**，attention 学到 "agent 0 的第 i 个特征位应该多关注 agent 1 的第 j 个特征位"。

```python
# Encoder（与 031A 完全一样）
feat0 = shared_encoder(obs0)   # (batch, 256)
feat1 = shared_encoder(obs1)   # (batch, 256)

# Reshape 成 token 序列
tokens0 = feat0.view(batch, n_tokens=4, head_dim=64)
tokens1 = feat1.view(batch, n_tokens=4, head_dim=64)

# Bidirectional cross-attention（Q/K/V 投影对两 agent 共享 → siamese 风格）
Q0, K0, V0 = q_proj(tokens0), k_proj(tokens0), v_proj(tokens0)
Q1, K1, V1 = q_proj(tokens1), k_proj(tokens1), v_proj(tokens1)

attn01_w = softmax(Q0 @ K1^T / √64, dim=-1)   # (batch, 4, 4) — agent 0 token i attends to agent 1 token j
attn10_w = softmax(Q1 @ K0^T / √64, dim=-1)
out01 = attn01_w @ V1                         # (batch, 4, 64)
out10 = attn10_w @ V0

# Residual concat (feat 直通 + attention output 都进 merge → graceful degradation)
merged = Merge_MLP(concat(feat0, out01.flatten(1), feat1, out10.flatten(1)))   # 1024 → 256 → 128
```

**Token 拆分的物理含义**: encoder 学出的 256-dim 特征向量天然不是单一概念，把它切成 4 个 64-dim "特征位" 让 attention 操作粒度更合理（类似 set transformer / vision transformer 的 patch token）。

**为什么 Q/K/V 投影也共享 (siamese)**: agent 身份对称（spawn 可能互换），attention 模块也应该对 agent index 不敏感。这同时把参数量降到 `64×64×3 = 12K`（vs 独立投影 24K）。

### 12.2.1 实现位置

- 模型类: [`SiameseCrossAttentionTeamTorchModel`](../../cs8803drl/branches/team_siamese.py) in `team_siamese.py`
- 注册名: `team_siamese_cross_attention_model`
- 训练脚本入口: [`train_ray_team_vs_baseline_shaping.py`](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)，通过 `TEAM_CROSS_ATTENTION=1` 切到该模型
- 批处理脚本: [`soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch`](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch)

### 12.3 核心假设

> **031A 的 progress_deficit 15.3% 是 frontier 最低（[§11.7](#117-failure-capture-v2-bucket已完成)），暗示 dual encoder 已经学到了一定的对手信息编码。Cross-attention 让"对手信息编码"显式化、查询化，预期能进一步降低 progress_deficit 并提升 baseline 1000ep。**

### 12.4 预注册超参

```bash
# 与 031A 完全对齐，只新增 attention 模块
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1                        # ← 新增 gate
TEAM_CROSS_ATTENTION_TOKENS=4                 # ← 新增 — encoder_hidden / TOKENS = head_dim
TEAM_CROSS_ATTENTION_DIM=64                   # ← 新增 — TOKENS * DIM 必须 == encoder_hidden
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048
MAX_ITERATIONS=1250                           # 同 031A budget，公平对比
```

约束: `TEAM_CROSS_ATTENTION_TOKENS * TEAM_CROSS_ATTENTION_DIM == TEAM_SIAMESE_ENCODER_HIDDENS 最后一层`。当前默认 `4 * 64 = 256` ✓。如不匹配，模型 init 会 raise `ValueError`。

**Warmstart 选项**：
- **B1（首选）**：scratch（与 031A 同口径，公平比较）
- **B2（备选）**：warm-start from 031A@1040 —— ⚠️ **不直接可行**：031B 的第一层 merge MLP 输入维度是 `1024` (含 attention output)，031A 是 `512`，shape mismatch 导致 merge MLP 第一层无法 load。strict=False 下只有 encoder 层能复用，merge/heads 都得重新训。**B2 价值有限**，建议直接 B1
  - 如果坚持 B2: 需要写一个 weights surgery script，从 031A checkpoint 抽出 encoder 部分单独 load 到 031B（绕过 strict=False 默认丢弃所有 mismatch keys）

### 12.4.1 参数量对比

| 模型 | 总参数 | 增量来源 |
|---|---:|---|
| 028A flat MLP `[512,512]` | ~660K | baseline |
| **031A** Siamese encoder `[256,256] + [256,128]` | **~320K** | siamese 共享 → 参数量减半 |
| **031B** = 031A + cross-attention | **~462K** | +12K Q/K/V proj + ~130K merge 第一层扩到 1024 |

**勘误**: §1 / §3.1 / §11.4 中提到 "031A 总参数量约 700K params" 与 [`team_siamese.py`](../../cs8803drl/branches/team_siamese.py) 实测 `~320K` 不符——之前写法把 flat MLP `[512,512]` 的参数量错记到了 siamese 上。031A 实际参数量 ~320K（siamese 共享让 encoder 只有一份 weight）。这反而进一步证明 031A 的 +7.7pp WR 增益**不是来自更大网络**，而是来自架构先验。

### 12.5 预注册判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| 1000ep avg | ≥ 0.86 | 至少不退化于 031A |
| **1000ep peak** | **≥ 0.875** | 显式 +1.5pp 增益（031A SE ≈ 0.011） |
| H2H vs 031A@1040 | ≥ 0.52 | attention 模块带来真实对抗优势 |
| Attention entropy | > ln(2) ≈ 0.69 | 不发生 attention collapse（softmax 退化为 one-hot） |
| **Failure: progress_deficit** | **≤ 12%** | 比 031A 的 15.3% 再低 ≥ 3pp，机制层面证明 attention 改善对手读取 |

### 12.6 失败判据

| 条件 | 解读 |
|---|---|
| 1000ep < 0.85 | attention 模块过拟合 / 训练不稳；可考虑 dropout / fewer heads |
| 1000ep ≥ 0.86 但 H2H vs 031A < 0.50 | attention 没带来真实技能，只是在 baseline 上学了等价策略 |
| Attention entropy < 0.3 | attention collapse，等价于固定权重 routing；需要熵正则 |

### 12.7 相对 040 (Stage 2 shaping handoff) 的优先级

| 路径 | 预期增益 | 时长 | 风险 |
|---|---|---|---|
| **031-B (cross-attention)** | +0~+2pp baseline，+0~+5pp H2H | ~14h scratch / ~3h warmstart | architecture risk (collapse) |
| **[040B (PBRS handoff on 031A)](snapshot-040-team-level-stage2-on-031A.md)** | +0~+2.5pp baseline (镜像 029B 的 +2.6pp) | ~3h | shaping shock risk |

**两者正交**：031-B 改架构、040 改 reward。如果 GPU 充足应**并行**跑；如果只有一条 GPU 时段，**优先 040B**（短时长 + 路径已被 029B 验证 +2.6pp）。

### 12.8 实现进度（2026-04-18）

- [x] 在 [`team_siamese.py`](../../cs8803drl/branches/team_siamese.py) 新增 `SiameseCrossAttentionTeamTorchModel` (+ `register_team_siamese_cross_attention_model`)
- [x] [`train_ray_team_vs_baseline_shaping.py`](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) 读 `TEAM_CROSS_ATTENTION` / `TEAM_CROSS_ATTENTION_TOKENS` / `TEAM_CROSS_ATTENTION_DIM` env vars，按 gate 选 `team_siamese_cross_attention_model` 或 `team_siamese_model`
- [x] 模型 smoke pass:
  - forward `(batch=8, 672) → logits (8, 18) + value (8,)`
  - attention_entropy 起点 ≈ ln(4) = 1.386（满分，random init 正常）
  - backprop OK，loss finite
  - shape mismatch (`tokens × dim != encoder_hidden`) raise `ValueError` ✓
- [x] env-var routing smoke pass: `TEAM_SIAMESE_ENCODER=1 + TEAM_CROSS_ATTENTION=1 → team_siamese_cross_attention_model`
- [x] Batch [`soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch`](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch) 就绪
- [ ] 启动 031B scratch run（B1）—— 待 GPU slot
- [ ] 训练完成后: top-5 ckpt × 1000ep eval + capture + H2H vs 031A@1040 / 029B@190

## 13. 031B 首轮结果 (2026-04-19，append-only)

### 13.1 训练完成

run dir: `ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/` (merged)

- best_reward_mean: +2.2312 @ iter 1228
- final_iteration: 1250 (50M steps)
- final_reward_mean: +2.2080
- best_eval_baseline (50ep internal): **0.960 (48W-2L) @ iter 590** — 但 50ep SE ±0.028，是单点抽样高峰，**1000ep 实测 0.856** (-0.104 drift，单点 spike artifact)
- best_eval_random (50ep internal): 1.000

合并自 1080-iter 主 trial + 170-iter resume。

### 13.2 数值健康度

`progress.csv` 1203 iter 检查：

| 检查 | 结果 |
|---|---|
| total_loss inf 行 | 0 |
| policy_loss inf 行 | 0 |
| kl inf 行 | 0 |
| max\|total_loss\| | 6.70 |
| max\|kl\| | 10.71 |

**clean，无 inf/nan**。stdout `grep -i "inf|nan"` 289 条匹配全是 `INFO` log 子串（严格 word-boundary = 0）。

对比：036C 16% inf / 036D 31.7% inf / 039 28.3% inf 来自 learned reward NN 输出爆炸。031B 用 v2 解析 shaping，无 reward NN，无 inf 源。

### 13.3 1000ep 官方 eval (top 5% + ties + ±1, 24 ckpts, 2026-04-19 ~07:23)

按 [pick_top_ckpts](../../scripts/eval/pick_top_ckpts.py) 选出 24 ckpt 全跑 1000ep on `atl1-1-03-013-19-0` (port shelf 51005, parallel -j 7, total elapsed 1028s)。

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 420 | 0.838 | 838-162 | — | — |
| 430 | 0.850 | 850-150 | 0.94 | -0.090 |
| 440 | 0.843 | 843-157 | — | — |
| 500 | 0.831 | 831-169 | — | — |
| 510 | 0.819 | 819-181 | 0.92 | -0.101 |
| 520 | 0.840 | 840-160 | — | — |
| 580 | 0.832 | 832-168 | — | — |
| **590** | **0.856** | 856-144 | **0.96** | **-0.104** |
| 600 | 0.840 | 840-160 | — | — |
| 680 | 0.821 | 821-179 | — | — |
| 690 | 0.847 | 847-153 | 0.92 | -0.073 |
| 700 | 0.859 | 859-141 | — | — |
| 740 | 0.836 | 836-164 | — | — |
| 750 | 0.849 | 849-151 | 0.92 | -0.071 |
| 760 | 0.842 | 842-158 | — | — |
| 860 | 0.850 | 850-150 | — | — |
| 870 | 0.865 | 865-135 | 0.94 | -0.075 |
| 880 | 0.840 | 840-160 | — | — |
| 1090 | 0.870 | 870-130 | — | — |
| 1100 | 0.863 | 863-137 | 0.94 | -0.077 |
| 1110 | 0.863 | 863-137 | — | — |
| **1220** | **🏆 0.882** | **882-118** | — | — |
| 1230 | 0.881 | 881-119 | 0.94 | -0.059 |
| 1240 | 0.872 | 872-128 | — | — |

### 13.4 关键发现

1. **新 SOTA: ckpt 1220 = 0.882** (882W-118L, n=1000, SE ±0.010, 95% CI **[0.862, 0.902]**)
2. 1220 / 1230 / 1240 三个 late-window ckpt 全 ≥ 0.872 — **真稳定 peak**，不是单点 spike
3. **031B vs 031A**: max +2.2pp (0.882 vs 0.860)，mean +0.5-1pp。**cross-attention 架构有真实增益**
4. 0.96 内部 spike @ 590 是 50ep sample noise（1000ep 实际 0.856），再次验证 [snapshot-027 §9](snapshot-027-team-level-ppo-coordination.md) 的「late-window > 早期 spike」doctrine
5. **首次 cross 0.88+** — 0.882 距 0.90 grading 仅 -1.8pp，95% CI 上界 0.902 已达门槛

### 13.5 对 0.86 ceiling 论的反驳

040A/B/C/D + 041B + 044 全 saturate 在 [0.852, 0.865] 是基于 **flat MLP team-level (031A) base**。**031B 把 ceiling 推到 0.88** 证明：

> **"0.86" 不是 PPO 上限，是 031A 架构上限**。换更强的架构（cross-attention）就能过。

### 13.6 Raw recap (verification)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_001220/checkpoint-1220 vs baseline: win_rate=0.882 (882W-118L-0T)
.../checkpoint_001230/checkpoint-1230 vs baseline: win_rate=0.881 (881W-119L-0T)
.../checkpoint_001240/checkpoint-1240 vs baseline: win_rate=0.872 (872W-128L-0T)
.../checkpoint_001090/checkpoint-1090 vs baseline: win_rate=0.870 (870W-130L-0T)
.../checkpoint_000870/checkpoint-870  vs baseline: win_rate=0.865 (865W-135L-0T)
.../checkpoint_001100/checkpoint-1100 vs baseline: win_rate=0.863 (863W-137L-0T)
.../checkpoint_001110/checkpoint-1110 vs baseline: win_rate=0.863 (863W-137L-0T)
.../checkpoint_000700/checkpoint-700  vs baseline: win_rate=0.859 (859W-141L-0T)
.../checkpoint_000590/checkpoint-590  vs baseline: win_rate=0.856 (856W-144L-0T)
.../checkpoint_000430/checkpoint-430  vs baseline: win_rate=0.850 (850W-150L-0T)
.../checkpoint_000860/checkpoint-860  vs baseline: win_rate=0.850 (850W-150L-0T)
.../checkpoint_000750/checkpoint-750  vs baseline: win_rate=0.849 (849W-151L-0T)
.../checkpoint_000690/checkpoint-690  vs baseline: win_rate=0.847 (847W-153L-0T)
.../checkpoint_000440/checkpoint-440  vs baseline: win_rate=0.843 (843W-157L-0T)
.../checkpoint_000760/checkpoint-760  vs baseline: win_rate=0.842 (842W-158L-0T)
.../checkpoint_000600/checkpoint-600  vs baseline: win_rate=0.840 (840W-160L-0T)
.../checkpoint_000520/checkpoint-520  vs baseline: win_rate=0.840 (840W-160L-0T)
.../checkpoint_000880/checkpoint-880  vs baseline: win_rate=0.840 (840W-160L-0T)
.../checkpoint_000420/checkpoint-420  vs baseline: win_rate=0.838 (838W-162L-0T)
.../checkpoint_000740/checkpoint-740  vs baseline: win_rate=0.836 (836W-164L-0T)
.../checkpoint_000580/checkpoint-580  vs baseline: win_rate=0.832 (832W-168L-0T)
.../checkpoint_000500/checkpoint-500  vs baseline: win_rate=0.831 (831W-169L-0T)
.../checkpoint_000680/checkpoint-680  vs baseline: win_rate=0.821 (821W-179L-0T)
.../checkpoint_000510/checkpoint-510  vs baseline: win_rate=0.819 (819W-181L-0T)
[suite-parallel] total_elapsed=1028.1s tasks=24 parallel=7
```

完整 log: [docs/experiments/artifacts/official-evals/031B_baseline1000.log](../../docs/experiments/artifacts/official-evals/031B_baseline1000.log)

### 13.7 Stage 2 — failure capture 1220 (n=500, 2026-04-19 ~07:42)

run on `atl1-1-03-015-2-0` port 54005，v2 shaping 同训练配置（deep_zone -8/-12, ball_progress 0.01, etc.）

**Summary**:

```json
{
  "team0_module": "cs8803drl.deployment.trained_team_ray_agent",
  "team1_module": "ceia_baseline_agent",
  "episodes": 500,
  "team0_wins": 438,
  "team1_wins": 62,
  "ties": 0,
  "team0_win_rate": 0.876,
  "step_stats": {
    "all":        {"count": 500, "mean": 43.24, "median": 36.0, "p75": 56.0, "min": 8, "max": 196},
    "team0_win":  {"count": 438, "mean": 44.88, "median": 38.0, "p75": 57.0, "min": 8, "max": 196},
    "team1_win":  {"count":  62, "mean": 31.65, "median": 24.0, "p75": 36.0, "min": 9, "max": 110}
  },
  "saved_episode_count": 62
}
```

#### v2 bucket 重新分类（multi-label）

| bucket | 031B@1220 (n=62) | 031A@1040 (n=85, ref) | Δ |
|---|---:|---:|---:|
| defensive_pin | 27 (44%) | 40 (47%) | -3pp |
| territorial_dominance | 27 (44%) | (-) | — |
| **wasted_possession** | **23 (37%)** | 36 (42%) | **-5pp** |
| possession_stolen | 17 (27%) | 28 (33%) | -6pp |
| progress_deficit | 16 (26%) | (-) | — |
| unclear_loss | 12 (19%) | (-) | — |
| **tail_ball_x median** | **+2.40** | +1.37 | +1.03 |
| poss median | 0.452 | 0.500 | -0.048 |
| **total losses (out of 500)** | **62 (12.4%)** | 85 (17%) | **-27% relative** |

**机制读法**:

- 失败结构 **跟 031A 几乎一样**（每个 bucket 占比变化 < 5pp）
- **31B 不是修了某个特定 failure mode，而是在所有方面都好一点**：失败总数 -27%，pattern 类型基本相同
- tail_ball_x 更正 (+2.40 vs +1.37) → 输的局也是球在对方半场，**wasted_possession 仍是 dominant pattern**
- 31B's improvement = **uniform improvement** (cross-attention 让两个 agent 全程更好协调)，**不是 specific bucket fix**

完整 log: [docs/experiments/artifacts/failure-cases/031B_checkpoint1220_baseline_500/](../../docs/experiments/artifacts/failure-cases/031B_checkpoint1220_baseline_500/)

### 13.8 Stage 3 — H2H matrix (4 matchups, n=500 each, 2026-04-19 ~07:42)

按 SOP 起 4 个 H2H：架构 axis (vs 031A) + 3 个 per-agent peer。

| matchup | record | 031B WR | side blue/orange | z | p | sig |
|---|---|---:|---|---:|---:|:---:|
| **vs 031A@1040** (架构 axis) | 258-242 | **0.516** | 0.500 / 0.532 | 0.715 | 0.237 | — (NOT sig) |
| **vs 029B@190** (per-agent v2 SOTA) | 292-208 | **0.584** | 0.604 / 0.564 | 3.755 | <0.001 | **\*\*\*** |
| **vs 025b@080** (per-agent BC champion) | 283-217 | **0.566** | 0.568 / 0.564 | 2.951 | 0.0016 | **\*\*** |
| **vs 036D@150** (per-agent + learned reward) | 287-213 | **0.574** | 0.568 / 0.580 | 3.308 | 0.00047 | **\*\*\*** |

#### Raw H2H Recap (verification)

vs 031A@1040:
```
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: cs8803drl.deployment.trained_team_ray_opponent_agent
team0_overall_record: 258W-242L-0T
team0_overall_win_rate: 0.516
team0_blue 125W-125L (0.500) / orange 133W-117L (0.532)
```

vs 029B@190:
```
team0_overall_record: 292W-208L-0T
team0_overall_win_rate: 0.584
team0_blue 151W-99L (0.604) / orange 141W-109L (0.564)
```

vs 025b@080:
```
team0_overall_record: 283W-217L-0T
team0_overall_win_rate: 0.566
team0_blue 142W-108L (0.568) / orange 141W-109L (0.564)
```

vs 036D@150:
```
team0_overall_record: 287W-213L-0T
team0_overall_win_rate: 0.574
team0_blue 142W-108L (0.568) / orange 145W-105L (0.580)
```

完整 log: [headtohead/](../../docs/experiments/artifacts/official-evals/headtohead/) (031B_1220_vs_*.log)

#### 关键解读

1. **vs 031A H2H 0.516 NOT significant** — 即使 baseline 上 +2.2pp，**直接对决几乎平手**（z=0.715, p=0.24）。这是关键 caveat。
2. **vs 3 个 per-agent frontier 全部显著** (0.566-0.584, p ≤ 0.0016) — 跨架构优势 robust 且 reproducible
3. **vs 036D (cross-reward) ≈ vs 029B (same v2 reward)**: 0.574 vs 0.584 几乎一致 → **31B 优势主来自架构 (cross-attention)**，不依赖特定 reward path
4. **架构 axis 边缘 vs peer axis 显著的反差** = 031B 在 baseline 上多学的，**部分是 baseline-specialization**（同架构 base 之间 transfer 弱，跨架构 transfer 强）
5. side gap vs 031A: blue 0.500 / orange 0.532 → 与 [§11.6.1](#1161-vs-028a1060-的特殊意义同-reward-同-base-直接架构对比) 观察的 spawn 不对称模式一致（[snapshot-022 §6.4](snapshot-022-role-differentiated-shaping.md) 系列）
6. vs 028A H2H (snapshot-031 §11.6.1) 031A vs 028A = 0.568 (***)。**031B vs 031A = 0.516 (NS)**。可见**架构升级 step 1 (flat → Siamese) 比 step 2 (Siamese → cross-attention) 在 peer-axis 上效果大得多**。

### 13.9 综合 Verdict (Stage 1 + 2 + 3)

**031B = 项目当前 SOTA, 但有 caveat**：

| 判据 | 状态 |
|---|---|
| baseline 1000ep peak | ✅ 0.882, 比 031A +2.2pp，95% CI [0.862, 0.902] |
| baseline 跨 ckpt 一致性 | ✅ 1220/1230/1240 三个 late-window 都 ≥ 0.872 |
| failure structure 改善 | ⚠️ uniform -27% loss，non-specific |
| H2H vs 031A (架构 axis) | ⚠️ 0.516 NOT sig — peer-axis 上**不明显**比 031A 强 |
| H2H vs per-agent frontier (029B/025b/036D) | ✅ 全部显著 (0.566-0.584) |
| 数值健康 | ✅ 0 inf in 1203 iter |

**真实位置**: 031B 是**baseline-axis 项目最强 lane**，在 peer-axis 上**与 031A 平、压过所有 per-agent**。**架构改进是真的，但不是颠覆性的（vs 031A 没拉开）**。

### 13.10 031B 后续发展路径地图

#### Grading 现实校准

`0.882` 距 `0.90` grading 只 -1.8pp，但 grading 是**离散** 9/10 in 10 ep。即使 true WR=0.88：

```
P(≥9 wins | n=10, p=0.88)
  = C(10,9)·0.88^9·0.12 + 0.88^10
  = 0.372 + 0.279 = 0.651
```

→ **65% 概率单 run 拿 9/10**。这是常被忽视的 ground truth——可能我们已经够了，关键是 **正确 submit + 多 seed 容错**，而不是死磕 +1.8pp。

#### 不再做的事（明确放弃）

- ❌ **更多 shaping on 031B base** — [snapshot-040](snapshot-040-team-level-stage2-on-031A.md)(4 lanes) 已 saturate 在 031A 上，同样会在 031B saturate
- ❌ **KL distill on 031B** — [snapshot-042](snapshot-042-cross-architecture-knowledge-transfer.md) 已证明 stable but no breakthrough
- ❌ **同架构更长训练** — 031B vs 031A H2H 平意味着架构步骤 2 边际效用低；多 1000 iter 大概仅 +0.5pp

#### TIER 1 — 不需新训练（最高 ROI）

| 路径 | 状态 | 工程 | 预期增益 |
|---|---|---|---|
| **[snapshot-034](snapshot-034-deploy-time-ensemble-agent.md) ensemble**（031B + 031A + 029B + 036D probability averaging）| **进行中**（user 已 launch）| 1 天 | +2-5pp (PETS 经验)；若到 0.90+ 就直接达标 |
| **031B vs Random 1000ep verify** | 未做 | 10 min | 确认 grading 第二条 (Random 9/10) 不会 fail；预期 ≥ 0.96 |
| **multi-checkpoint submit prep** | 未做 | 1h | 提交时三 ckpt (1220/1230/1240) 三 seed 跑，覆盖单 run 风险 |

理由：031B/031A wasted_possession-dominant vs 036D defensive_pin-dominant ([snapshot-049](snapshot-049-env-state-restore-investigation.md)/047 v2 桶证据) → 失败模式正交 → ensemble 是 PETS 经典适用场景。

#### TIER 2 — 架构步骤 3（赌博）

cross-attention 是步骤 2。下一步候选：

| 候选 | 描述 | 预算 |
|---|---|---|
| **multi-head attention** | 当前是 single head，改 4 head | ~1 天 eng + 14h GPU |
| **transformer block** | attention + FFN + residual + LayerNorm | ~2 天 eng + 14h GPU |
| **memory (LSTM/GRU)** | 跟踪队友意图历史，跨 step 信息保留 | ~3 天 eng + 14h GPU |

**但 diminishing returns 已显著**：
- step 1 (flat → Siamese): baseline +7.7pp, peer +5.6pp（[§11.6.1](#1161-vs-028a1060-的特殊意义同-reward-同-base-直接架构对比) 031A vs 028A）
- step 2 (Siamese → cross-attention): baseline +2.2pp, peer +1.6pp NS（13.4 + 13.8）
- 预期 step 3: baseline +0-1pp, peer 几乎无增益

**只在 TIER 1 不够 + GPU 富余时启动**。

#### TIER 3 — 用 031B 当 ingredient

| 路径 | 状态 | 工程 |
|---|---|---|
| **031B-flavored cross-train**（[snapshot-046](snapshot-046-cross-train-pair.md) 变种）: 训 P vs frozen 031B@1220 | adapter ready | 3h GPU + 1 天 eval |
| **031B as student in hybrid takeover**（[snapshot-048](snapshot-048-hybrid-eval-baseline-takeover.md)）| 等当前 hybrid (031A) 完成 | 复用现成 evaluator，2-4h GPU |

#### TIER 4 — Grading 侧策略（非训练）

| 路径 | 描述 | 工程 |
|---|---|---|
| **Multi-checkpoint submission** | 031B@1220 / 1230 / 1240 三 ckpt 都 ≥ 0.872，提交多 seed 覆盖单 run 风险 | 1h |
| **Failure-state heuristic override** | deploy 时检测 wasted_possession state（球离对方门 < X 距离 + 时间 < Y 步），强制 shoot action | 1 天，hacky 但有效 |

后者只在 ≥ 0.88 仍不达标时启用。

#### 推荐执行顺序

**今/明天**（user 已 launching ensemble，并行做这两件）:
1. ✅ TIER 1.1 ensemble（**进行中**）
2. TIER 1.2 Random verify（10 min）
3. TIER 4.1 multi-ckpt submit prep（1h）

**3-5 天**（视 ensemble 结果决定）:
- 若 ensemble ≥ 0.90 → 直接进 grading，project 收口
- 若 ensemble ∈ [0.88, 0.90] → TIER 4.1 multi-seed grading; 同时 TIER 3.1 cross-train + TIER 2 多头 attention 一条 lane 试探
- 若 ensemble < 0.88 → TIER 3.2 hybrid takeover with 031B; TIER 2 全力推

**只在 ≥ 0.88 仍不达标时**:
- TIER 4.2 heuristic override（最后手段）

### 13.11 vs Random 1000ep verify (2026-04-19, TIER 1.2 落地)

按 [§13.10 TIER 1](#tier-1--不需新训练最高-roi) 跑 grading 第二条 verify (vs Random 9/10) on `atl1-1-03-013-19-0` port 58005, total elapsed 148.8s。

| ckpt | Random 1000ep | NW-ML | 距 grading 0.90 |
|---:|---:|:---:|---:|
| **031B@1220** | **0.990** | 990-10 | **+9.0pp** |
| **031B@1230** | **0.994** | 994-6 | **+9.4pp** |
| **031B@1240** | **0.994** | 994-6 | **+9.4pp** |

**Grading 第二条 robust pass**。10ep submission 下 P(≥9 wins | p=0.99) ≈ `C(10,9)·0.99^9·0.01 + 0.99^10 = 0.996` → 几乎不可能 fail。

**结合 baseline (§13.3)**, 031B 满足 grading 两条:

| 判据 | 阈值 | 031B@1220 1000ep | single-run grading P(≥9/10) |
|---|---:|---:|---:|
| vs baseline 9/10 | 0.90 | **0.882** | **0.65** |
| vs Random 9/10 | 0.90 | **0.990** | **0.996** |

→ **031B@1220 单 submission 同时满足两条 grading 的概率** ≈ 0.65 × 0.996 ≈ **0.65**。
- 若多 seed (N=3) 提交取 best run: P 提升到 1 - (1-0.65)^3 = **0.957**
- multi-ckpt submission（1220 + 1230 + 1240 都 ≥ 0.872 baseline 且 ≥ 0.99 random）= 高 robust 提交配比

完整 log: [docs/experiments/artifacts/official-evals/031B_random1000.log](../../docs/experiments/artifacts/official-evals/031B_random1000.log)

## 10. 相关

- [SNAPSHOT-028: team-level BC base](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-030: port experiments](snapshot-030-team-level-advanced-shaping-chain.md)
- [SNAPSHOT-032: aux loss](snapshot-032-team-level-native-coordination-aux-loss.md)
- [SNAPSHOT-033: team reward shaping](snapshot-033-team-level-native-coordination-reward.md)
