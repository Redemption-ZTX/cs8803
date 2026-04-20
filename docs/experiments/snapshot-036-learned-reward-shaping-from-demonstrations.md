# SNAPSHOT-036: Learned Reward Shaping from Demonstrations

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 已完成首轮结果（数值异常待修）

## 0. 灵感来源

HW4 Section 1（**Maximum Entropy Inverse Reinforcement Learning**, Ziebart et al. 2008）：

> Feature-expectation matching requires the learner to visit features just as often as the expert does ... we pick the solution with maximum entropy so that we add no extra bias beyond the data we observed.

核心思想：**不要人工猜测 reward；从 expert 行为里反推 reward**。

## 1. 当前问题

我们所有 shaping 都是**人类猜的**：

| shaping | 设计假设 | 暴露的弱点 |
|---|---|---|
| v2 | "持续推进 + 控球 + 不让对手推进 = 赢" | `low_poss 22-28%` 不变量被锁死，无法降下来 |
| PBRS goal-proximity | "球离对方门越近越好" | baseline-specific exploit (026B@250 H2H 输) |
| Field-role binding | "前 striker 后 defender 分工" | 在 BC@2100 上 +0.0，被冲掉 |
| Opp pool | "见过 peer 的 policy 防 exploit 更强" | 029-C 仅 +0.006，模糊正结果 |

**我们已经在 0.85-0.87 区间撞墙**。每个 trick 都给 +0.005-0.026，但**没有一个突破 0.90 = 9/10**。

根本原因：**人类对 soccer 策略的理解是有限的**。我们写不出能让 policy 学到 0.90 的 shaping。需要换个 reward 来源。

## 2. 三条路径（按理论上限排序）

| 路径 | 学习目标 | 数据来源 | 理论上限 vs baseline |
|---|---|---|---|
| **A** (经典 MaxEnt IRL) | 模仿 baseline 行为 | baseline 轨迹 | **≈ baseline = 50/50** |
| **B** (判别器/GAIL-style) | 行为分布像 baseline | baseline + non-baseline 轨迹 | 同 A，~50/50 |
| **C** (Win/Loss contrastive) | "做我们赢时做的事 / 避免输时做的事" | **我们自己的 W 和 L 轨迹** | **"我们的最佳版本"** |

### 2.1 为什么 A/B 路径上限是 50/50

- 纯模仿 baseline → policy 退化为 baseline 的副本
- baseline vs baseline 就是 ~50/50（[snapshot-013 BvB 实测](snapshot-013-baseline-weakness-analysis.md)）
- imitation 没有"超 baseline 的信号"

### 2.2 为什么路径 C 上限更高

- 我们当前 policy（029B@190 = 0.868）已经**显著强于 baseline**
- 它的 W 轨迹包含 "比 baseline 强" 的 key state-action pattern
- 它的 L 轨迹包含 "被 baseline exploit 的弱点"
- contrast 这两个信号 → 学到 **超 baseline 的 self-improvement reward signal**
- 没有 inductive bias——不猜什么是好策略，从实际表现里抽

**理论上限是 "我们能稳定执行的最佳水平"**，不是 baseline。

## 3. 路径 C 的详细设计（首轮主线）

### 3.1 数据采集

**多策略多样性**：

| 策略 | official 500 | 角色 | 采集数 |
|---|---|---|---|
| 029B@190 | 0.868 | 当前最强 | 800 episodes |
| 025b@80 | 0.842 + H2H 强 | H2H champion | 400 |
| 017 BC@2100 | 0.842 | per-agent 老 SOTA | 400 |
| 028A@1220 | 0.844 | team-level 路径 | 400 |
| baseline self-play | — | reference anchor | 400 |
| **总计** | | | **2400 episodes** |

每个 episode 记录：
- 完整 trajectory：`(s_0, a_0, ..., s_T)` 全部 step
- 最终 outcome：W (我方进球) / L (对方进球) / T (平)
- 结束 step 编号 T

预估：每 episode ~50 step，2400 ep × 50 = **120k state-action pairs**——足够训一个小 MLP reward model。

### 3.2 标签设计

每个 (s, a) 拿到一个 reward label，按"距离终局的衰减"加权：

```python
def label_per_step(trajectory, outcome):
    T = len(trajectory)
    sign = +1.0 if outcome == 'W' else (-1.0 if outcome == 'L' else 0.0)
    gamma = 0.95
    labels = []
    for t in range(T):
        # closer to end = stronger label
        weight = gamma ** (T - 1 - t)
        labels.append(sign * weight)
    return labels  # in [-1.0, +1.0]
```

理由：
- step T-1（最后一步）拿到完整 ±1.0
- step 0 拿到 ±0.95^49 ≈ ±0.08（小信号，因为远离结果）
- 自然 credit assignment，不需要 manual hand-tuning

### 3.3 三类标签的归一化

为了让 reward model 学到 `R(L) < R(baseline) < R(W)` 的相对顺序：

```python
final_label[s, a] = sign * weight  +  baseline_anchor_offset
# baseline_anchor_offset = 0 (由数据分布隐式定义)
```

加 baseline 数据作为 0 anchor：
- W 数据 label > 0
- L 数据 label < 0
- baseline 数据 label = 0（让 reward model 知道 "baseline 水平是 0"）

### 3.4 Reward Model 架构

```
input:  state (336 dim)  +  action (one-hot 27 dim)  → 363 dim
hidden: 256 → 128
output: scalar R(s, a)
loss:   MSE(R(s, a), label[s, a])
```

简单 MLP，不需要 fancy 架构。训练量 ~5k steps，~20 min on H100。

可选：用 **state-only** reward（不带 action）`R(s)` —— 更接近 PBRS 形式，避免 action-specific gaming。

### 3.5 集成到 PPO 训练

把 learned reward 作为 shaping，叠加在原始 sparse reward 上：

```python
r_total = r_sparse_goal + λ_irl * R_learned(s_t, a_t)
```

- `λ_irl`: 起始 0.01，必要时调 (snapshot-035 同样 ramping 思想)
- 训练 base：029B@190 warm-start + IRL shaping fine-tune 200 iter

或者 **完全替换 v2 shaping**：直接用 R_learned 代替 v2 所有 term，做 clean apple-to-apple 对照。

### 3.6 失败桶分层精细化（首轮推荐版本）

§3.1-§3.5 描述的是 path C 的**最简形式**——把所有 L 当作"输了"统一标负。但当前我们已经有 6 类失败桶 (`late_def_collapse / territory_loss / low_poss / poor_conv / opp_fwd_progress / unclear_loss`)，**把所有 L 平均会丢失这层信号**。

精细化方案：**多 head reward model，每个 head 对应一个失败桶**。

#### 3.6.1 多 head 架构

```
input:  state (336 dim) [+ action one-hot 27 dim]
shared: Linear(...) → ReLU → Linear(...) → ReLU  →  shared_feat
heads:
  head_late_def(shared_feat) → scalar R_late_def
  head_low_poss(shared_feat) → scalar R_low_poss
  head_poor_conv(shared_feat) → scalar R_poor_conv
  head_opp_fwd(shared_feat) → scalar R_opp_fwd
  head_territory(shared_feat) → scalar R_territory
  (head_unclear 略——unclear 本身没有可学的模式)
```

**Shared encoder + 5 head**——比 5 个独立模型省参数，且 shared encoder 学到通用 state representation。

#### 3.6.2 每个 head 的训练目标

每个 head 是一个 **二分类器**：判断当前 state-action 是来自 W trajectory 还是来自该桶的 L trajectory。

| head | 正样本 | 负样本 |
|---|---|---|
| `late_def` | W 轨迹（任意）的 (s, a) | L 轨迹中 `primary_label = late_def_collapse` 或 multi-label 含 late_def 的 (s, a) |
| `low_poss` | 同上 W | L 轨迹中 multi-label 含 low_poss 的 (s, a) |
| `poor_conv` | 同上 W | L 轨迹中 multi-label 含 poor_conv 的 (s, a) |
| `opp_fwd` | 同上 W | L 轨迹中 multi-label 含 opp_fwd_progress 的 (s, a) |
| `territory` | 同上 W | L 轨迹中 multi-label 含 territory_loss 的 (s, a) |

**关键设计**：用 `labels` 字段（multi-label set），不是 `primary_label`——避免 §5.3 提到的"单 primary 优先级排序掩盖低优先桶信号"问题。

每个 head 用 BCE loss + 时序 credit 衰减（§3.2 同样的 `γ^(T-t)` 加权）。

#### 3.6.3 Reward 集成

```python
r_total = r_sparse_goal + λ_irl * sum_k(w_k * head_k(s, a))
```

- 起始用 **均匀权重** `w_k = 1.0 / 5`
- 进阶：`w_k` 按当前 policy 的失败结构反向加权——例如当前 policy 的 `low_poss` 比例 26%，`w_low_poss` 加大；`unclear_loss` 8% 已经很低，`w_unclear` 不需要管
- 这是 **基于失败结构的自适应 reward 加权**，闭环更新

#### 3.6.4 为什么这个版本更强

直接对应 snapshot-026 §16.5 的项目级问题：

> `low_possession 22-28%` 不变量是 v2 shaping 自己的局部最优带——只要训练 reward 是 v2 系列，`low_poss` 就会停在这个区间。

精细化版本通过 **专门的 `head_low_poss` reward 信号**，**显式把 policy 从 low_poss 状态拉开**——而不是像 v2 那样只通过"持球 bonus"间接影响。

类比：
- v2 shaping: "推进 + 控球 + ..." (人工设计的间接信号)
- Path C 简化版: "学 W vs L 整体差异" (单一对比信号)
- **Path C 精细化版: "学 W vs (L because of X) 的差异"，X ∈ {late_def, low_poss, poor_conv, ...}** (多个并行专门信号)

每个 head 直接对准一个失败模式，比单一 reward model 更难 gaming（policy 必须同时改善 5 个 head 才能在 reward 上得分）。

#### 3.6.5 工程量

相比简化版 (§3.4)：
- Reward model 参数量 ×1.5（多了 5 个 head，但 shared encoder 主体不变）
- 训练数据用法不变（同一份 W/L 数据，只是 label 拆成 5 类）
- 训练时间几乎不变（同一个 forward pass 算 5 个 head）
- PPO 集成时多算 4 次 head，每步推理多 ~1ms，可忽略

**首轮推荐直接跑精细化版本**，不再单独跑简化版——精细化版本至少不会差，潜在收益更高。

### 3.7 训练判据

**Reward model 自身验证**：
- Hold-out 20% 数据测 prediction accuracy
- W/L pair 上的 ranking accuracy（R(W) > R(L) 的比例）

**RL fine-tune 后的 final policy**：见 §4

## 4. 预声明判据

### 4.1 主判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **036C-warm vs baseline 500** | **≥ 0.88** | 突破 029B@190 的 0.868 |
| **036C-warm vs baseline 500** | **≥ 0.90** | **达到 9/10 门槛**（终极目标）|
| H2H vs 029B@190 | ≥ 0.50 | 不输 best single |
| H2H vs 025b@80 | ≥ 0.50 | 不输 H2H champion |

### 4.2 机制判据

| 项 | 期望 |
|---|---|
| reward model ranking accuracy on holdout | ≥ 70% |
| failure capture `low_possession` | ≤ 25% (持球能力改善)|
| failure capture `late_def_collapse` | ≤ 48%（防守不退步）|

### 4.3 失败判据

| 条件 | 解读 |
|---|---|
| 036C-warm vs baseline < 0.84 | learned reward 比 v2 更差 → reward model 学到了 spurious feature |
| ranking accuracy < 60% | reward model 没学到有效信号 → 数据不够或 label 设计差 |
| 036C-warm 内评 vs official gap > 0.10 | 标准 gaming 信号——policy 学到欺骗 reward model 的 exploit |
| H2H vs 029B < 0.45 | 即使 baseline WR 升，对等对抗弱 → 类 026B@250 baseline-specific exploit |

### 4.4 Gaming 防护

learned reward 的 gaming 风险**比手工 shaping 更微妙**——policy 可能学到一些"看起来像 W 但实际不赢"的 spurious feature。

防护：

1. **Reward model size 不能太大**：256→128 这种小 MLP 难以 over-memorize. 不用 transformer 等大模型
2. **State-only vs state+action 对照**：先跑 state-only `R(s)`——如果 work，更难 game（policy 不能通过特定 action 刷分）
3. **Periodic reward model retrain**：训练中每 50 iter 用最新 policy 的 trajectory 更新 reward model—— 不让 policy 长期 exploit 静态 reward
4. **H2H 是 ground truth**：vs baseline WR 升但 H2H 不升 → gaming
5. **Reward model 加 entropy regularization**：MaxEnt IRL 原理本身就要求 max entropy → 防止 reward 过尖锐
6. **Visual sanity check**：best ckpt 看 5 局，确认没有"看起来像 W 但实际是 L"的 weird 行为

## 5. 路径 A/B 作为对照（首轮可不跑）

按上限分析，A/B 不可能突破 50/50 vs baseline，但可以作为：
- **机制 control**：证明 path C 的收益来自"contrast signal"而不是"任何 learned reward 都比 v2 好"
- **fallback**：如果 path C 训练不稳定，先确认 reward model pipeline 本身能 work

只有在 path C 出问题时再启动 path A 或 B。

## 6. 执行矩阵

| lane | 内容 | 预算 |
|---|---|---|
| **036-Stage1** | 数据采集（5 个 policy 各打 baseline 400-800 局，加 baseline self-play）| ~6h GPU |
| **036-Stage2** | Reward model 训练 + holdout 验证 | ~30min GPU |
| **036C-warm** | 029B@190 + learned reward shaping 200-iter fine-tune | ~6h GPU |
| 036C-control | 同配置但用原 v2 shaping（= 029B@190 续训）| ~6h GPU |
| **036A-fallback** | 经典 MaxEnt IRL on baseline only | 视情况启动 |
| **036B-fallback** | GAIL-style discriminator | 视情况启动 |

总首轮预算：~12h GPU。比 030/031/032/033 任一条都重，但理论收益最高。

## 7. 工程依赖

### 7.1 已存在

- 5 个 best checkpoints
- Trajectory 采集基础设施（`scripts/eval/evaluate_official_suite.py` 已经能 dump 整局数据用于 failure capture）
- 028A 训练入口（warm-start + shaping override）

### 7.2 需要新增

- **Trajectory dumper**：扩展 evaluator，每局保存完整 (s, a, s', r, done, outcome) 序列到文件
- **Reward model trainer**：`cs8803drl/imitation/learned_reward_trainer.py`
- **PPO learned-reward integration**：让 `train_ray_mappo_vs_baseline.py` 或 team-level 入口能加载 reward model checkpoint，在 env wrapper 里调用做 shaping
- batch 脚本

### 7.3 复用空间

- 015 player-level BC dataset 收集脚本——逻辑高度相似，可改写
- 026 C-event shaping 的 `RewardShapingWrapper` hook 位置

## 8. 和其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [015 BC bootstrap](snapshot-015-behavior-cloning-team-bootstrap.md) | 类似数据采集流程 |
| [026 reward liberation](snapshot-026-reward-liberation-ablation.md) | 给我们看了 PBRS 的限制；036 是真正"不猜 reward" |
| [029B](snapshot-029-post-025b-sota-extension.md) | 提供 warm-start 源 |
| [030/031/032/033] | 这些都是人工设计 trick；036 用 data-driven reward |
| [034 ensemble](snapshot-034-deploy-time-ensemble-agent.md) | 正交方向；036 训出新 SOTA 后可加入 ensemble |

## 9. 不做的事

- **不做 deep IRL with neural reward + neural dynamics**（Wulfmeier 2015）—— 工程量过大，stage1 用浅 MLP 验证 path C 概念
- **不做 GAIL adversarial training**——离散 action + 简单环境用判别器不需要 GAN 训练 trick
- **不做 RLHF preference learning（Bradley-Terry pairwise）**——首轮 episode-level label 已经足够；preference 留作 stage 2 升级
- **不做 reward model 持续 online learning**——首轮训完 freeze，避免 reward chasing 不收敛

## 10. 执行清单

1. 实现 trajectory dumper，扩展 evaluator
2. 跑 Stage 1：5 policy × 400-800 ep + baseline self-play
3. 实现 reward model trainer
4. 跑 Stage 2：训 reward model + holdout accuracy 验证
5. **决策点**：reward model 在 holdout W/L pairs 上 ranking accuracy 是否 ≥ 70%
   - 是 → 继续 036C
   - 否 → 调数据 / label 设计 / 模型，不上 RL
6. 实现 PPO 集成 learned reward
7. 1-iter smoke：确认 env wrapper 能加载 reward model 并产出合理 shaping
8. 起 036C-warm + 036C-control 并行
9. official 500 + failure capture + H2H
10. 按 §4 判据 verdict

## 12. 执行进度（实时回填）

### 12.1 [2026-04-18] Trajectory dumper 工程落地 ✅

工程交付物：
- [cs8803drl/imitation/__init__.py](../../cs8803drl/imitation/__init__.py) — 新 imitation package
- [cs8803drl/imitation/trajectory_dumper.py](../../cs8803drl/imitation/trajectory_dumper.py) — `TrajectoryRecorder` + `save_trajectory`
- [scripts/eval/dump_trajectories.py](../../scripts/eval/dump_trajectories.py) — CLI dumper

自审修复（3 个 critical bug）：
1. **Action 改为 MultiDiscrete `(T, 3) int8` 存储**（原先用自定义 flat encoding，可能和 `gym_unity.ActionFlattener` 不一致）
2. **RewardShapingWrapper 显式 all-zero**（原先用默认值 = 应用了 v2 shaping，会污染 cumulative reward 字段）
3. **obs 加 `.copy()`**（防 env 内部 buffer 复用）

Smoke 验证（10 ep on 029B@190 + 5 ep on 028A@1220）：
- Action shape `(T, 3)` int8 min=0 max=2 ✓
- Reward 无 shaping 叠加（L 局 team0 cumul = -2.000 即纯 sparse goal）✓
- Multi-label 正确保留（如 L 局 `['late_defensive_collapse', 'territory_loss', 'poor_conversion']`）✓
- per-agent (017/025b/029B) + team-level (028A) 两种 deployment wrapper 都通过 ✓

### 12.2 [2026-04-18] Stage 1 Trajectory 采集启动 🚧

4 条并行后台采集（on `atl1-1-03-014-23-0`）：

| lane | policy | episodes | base_port | 状态 |
|---|---|---|---|---|
| A | 029B@190 (per-agent) | 800 | 56305 | running |
| B | 025b@80 (per-agent) | 400 | 56405 | running |
| C | 017 BC@2100 (per-agent) | 400 | 56505 | running |
| D | 028A@1220 (team-level) | 400 | 56605 | running |

存储：`docs/experiments/artifacts/trajectories/036_stage1/{029B_190,025b_080,017_2100,028A_1220}`

不包含 baseline self-play（§3.3 的 anchor 数据），留作第二轮补充。

### 12.3 [2026-04-18] Stage 2 reward trainer + Stage 3 PPO wrapper 实现 ✅

两个 module 落地（Stage 1 采集仍在后台跑）：

**Stage 2 — [cs8803drl/imitation/learned_reward_trainer.py](../../cs8803drl/imitation/learned_reward_trainer.py)**：
- `MultiHeadRewardModel`：shared encoder (256→256) + 5 binary-classifier heads (`late_def / low_poss / poor_conv / opp_fwd / territory`)
- Input：`obs (336) + action_one_hot (3×3=9)` = 345 dim (use_action=True) 或 `obs (336)` (use_action=False)
- 参数量 ~320k（state+action 版）
- 每 head 用 masked BCE loss：
  - W 样本：所有 head 都 target=1 (`mask=True, sign=+1`)
  - L 样本：只在 episode 的 multi-label set 里的 head 上 target=0 (`mask=True, sign=-1`)
  - 其余：`mask=False`，该 head 不贡献 loss
- 时序 credit: `γ^(T-1-t) × sign`
- CLI：`scripts/eval/train_reward_model.py`（通过 `python -m cs8803drl.imitation.learned_reward_trainer` 直接跑）

**Stage 3 — [cs8803drl/imitation/learned_reward_shaping.py](../../cs8803drl/imitation/learned_reward_shaping.py)**：
- `LearnedRewardShapingWrapper`：gym.Wrapper，在 `env.step()` 之后注入 per-agent 学习 reward
- Per-step shaping: `λ × mean_over_heads( tanh(head_k(obs_i, act_i)) )`
- Tanh 使每 head 输出 ∈ (-1, +1)，所以 per-step shaping ∈ [-λ, +λ]
- **Action 编码**：如果 env 有 `ActionFlattener`，用 `flattener.lookup_action(int)` 把 RLlib Discrete(27) 还原为 MultiDiscrete；否则用项目 canonical encoding（`_unflatten_discrete_to_multidiscrete`）。两种都保证训练和部署时 action 表示一致。
- 只 shape team0（reward model 是对 team0 W/L 训练的）
- 诊断：`info["_learned_reward_shaping"]` 填 per-agent shaping 值供 failure capture 可选追踪
- 单元测试：`_as_multidiscrete(0) = [0,0,0]`, `(13) = [1,1,1]`, `(26) = [2,2,2]` ✓

### 12.4 [实时] Stage 1 采集进度（~6 min elapsed）

| lane | target | saved | rate/min |
|---|---:|---:|---:|
| 029B_190 | 800 | 52 | 8.7 |
| 025b_080 | 400 | 27 | 4.5 |
| 017_2100 | 400 | 20 | 3.3 |
| 028A_1220 | 400 | 37 | 6.2 |
| **total** | **2000** | **136** | **22.7** |

按当前速率预估全部完成 **~80-120 min**（bottleneck 是 017_2100 的 3.3 eps/min）。慢于最初 30 min 估计，原因是 4 个并行 Python 进程 + 4 个独立 Ray cluster 在 32 核上竞争 CPU（每 process 峰值 9+ cores）。

### 12.5 [2026-04-18] PPO 训练入口集成 ✅

修改 [cs8803drl/core/utils.py](../../cs8803drl/core/utils.py) 的 `create_rllib_env`：在 `RewardShapingWrapper` 之后插入 conditional `LearnedRewardShapingWrapper`（按 env_config["learned_reward_shaping"] 触发，避免破坏不需要它的 lane）。

修改 [cs8803drl/training/train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)：读取 env vars
- `LEARNED_REWARD_MODEL_PATH`（reward_model.pt 路径，必填触发）
- `LEARNED_REWARD_SHAPING_WEIGHT`（默认 0.01）
- `LEARNED_REWARD_APPLY_TO_TEAM1`（默认 0，只 shape team0）

写好 batch [soccerstwos_h100_cpu32_mappo_036C_learned_reward_on_029B190_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_036C_learned_reward_on_029B190_512x512.batch)：
- warm-start 029B@190
- 保留 v2 shaping，叠加 learned reward shaping
- 默认 weight=0.01，与 v2 同量级
- 300 iter / 12M steps fine-tune

### 12.6 [2026-04-18] Stage 1 采集进度更新（~10 min elapsed）

| lane | target | saved | done | rate/min | ETA |
|---|---:|---:|---:|---:|---:|
| 029B_190 | 800 | 64 | 8.0% | 6.4 | ~115 min |
| 025b_080 | 400 | 39 | 9.75% | 3.9 | ~92 min |
| 017_2100 | 400 | 33 | 8.25% | 3.3 | ~111 min |
| 028A_1220 | 400 | 54 | 13.5% | 5.4 | ~64 min |

Bottleneck 是 029B_190 (~115 min total)。预估全部完成 **~06:00 + 115 min = 07:55 EDT**。

### 12.7 工程进度

- [x] Trajectory dumper 实现 + smoke
- [x] Reward model trainer 实现（multi-head, masked BCE, γ temporal credit）
- [x] PPO LearnedRewardShapingWrapper 实现
- [x] `create_rllib_env` 注入 wrapper hook
- [x] `train_ray_mappo_vs_baseline.py` 读 env vars 触发
- [x] 036C-warm batch 脚本（待 reward model 路径填入）
- [x] **Failure-bucket v2 redesign**（§12.9）+ trainer `--label-version` switch + dry-run 验证
- [x] Stage 1 采集（全 4 lane 完成，2000 ep，374 L）
- [x] Stage 2 reward model 训练（v2, best epoch 6, val_loss 0.1644）
- [x] Stage 3 sanity: W/L 排序 **AUC = 0.9772**（§12.10）
- [x] Stage 4: 036C-warm 首轮训练（warm-start 029B@190 + learned reward）

**Stage 0-4 首轮闭环已完成**。当前最重要的 follow-up 不再是“给 036C 更多对局机会”，而是先修 Stage 4 训练期出现的 `inf` 数值异常，再决定是否继续放大 peer-H2H。

### 12.8 [2026-04-18] Trainer dry-run on partial 029B data (68 ep, 6058 samples)

| head | W_pos | L_neg | imbalance ratio |
|---|---:|---:|---:|
| late_def | 5434 | 580 | 9.4× |
| low_poss | 5434 | **18** | **302×** ⚠️ |
| poor_conv | 5434 | 390 | 13.9× |
| opp_fwd | 5434 | 254 | 21.4× |
| territory | 5434 | 448 | 12.1× |

**潜在 gaming/退化风险**：`low_poss` head 在 029B 数据上极度不平衡（W : L = 302 : 1）。原因——029B 输的时候主要是 `late_def + territory + poor_conv`，几乎不是 `low_poss`（这本身印证了 §1 的"low_poss 22-28% 不变量"——029B 已经压住了这个 bucket）。

预期：等 5 policy 完整数据汇合后，028A / 025b 可能贡献更多 `low_poss` L 局，imbalance 会缓解。如果 Stage 1 全完后 `low_poss` head 仍 < 100 L_neg，需要在 trainer v2 加 class weighting。

### 12.9 [2026-04-18] Failure-bucket v2: data-driven redesign

**动机** — §12.8 的 dry-run 发现 `low_poss` head 在 029B 数据上 302× imbalance。初读以为是数据稀疏问题，但**扫完全部 313 个 ep 的 metric 分布后**发现这是 bucket 设计缺陷：

#### 关键数据发现

阈值 audit —— `failure_cases.classify_failure` 的阈值是按"归一化"数值感觉写的，但 soccer 场地是**原始 Unity 坐标 ~[-15, +15]**:

| v1 阈值 | 场地比例 | 后果 |
|---|---|---|
| `mean_ball_x < -0.15` | 仅 1% 场长偏左 | territory_loss 过拟合触发 |
| `tail_mean_ball_x < -0.45` | 仅 3% 场长 | late_def 几乎所有败局触发 (53%) |
| `team0_poss < 0.35` | — | 对强策略死信号（029B 只 7.1% L 触发） |

反直觉分布（313 ep） —— **球权与胜负负相关**:

```
W eps (n=264):  mean_ball_x mean=+0.337  team0_poss med=0.483
L eps (n=57):   mean_ball_x mean=-0.542  team0_poss med=0.588 (p75=0.889!)
```

强策略的典型败局是"**有球但进不去**"，不是"没球"。v1 的 `low_poss` 没抓到这个。

#### v2 Bucket 设计

保留"球权重要"的直觉（用户明确要求），但拆为"控球 + 浪费" vs "控球 + 被夺"两种失败模式：

| v2 Head | 定义 | 对标 v1 | 为什么 |
|---|---|---|---|
| `defensive_pin` | `tail_mean_ball_x < -3.0` | late_def | 阈值从"稍微偏左"升到"真被压在自己 1/5 场" |
| `territorial_dominance` | `mean_ball_x < -1.5` | territory_loss | 解耦控场 vs 末段压迫 |
| `wasted_possession` | `team0_poss > 0.55 AND L` | poor_conv | **强策略核心失败模式**（数据支撑） |
| `possession_stolen` | `team0_poss < 0.35 AND L` | low_poss | 保留弱策略信号，但不对 029B 死信号 |
| `progress_deficit` | `(t1_prog - t0_prog) > 3.0` | opp_fwd | 改用绝对差（v1 除法出现 23M 爆炸值） |

阈值常量见 [`failure_buckets_v2.py`](../../cs8803drl/imitation/failure_buckets_v2.py)（`thresholds_dict()`），checkpoint metadata 自动记录。

#### v2 在 313 ep 上的模拟分布（离线重标，不需重采）

| Bucket | 017 L=11 | 025b L=14 | 028A L=27 | 029B L=14 | aggregate L=66 |
|---|---:|---:|---:|---:|---:|
| wasted_possession | 27.3% | 42.9% | 51.9% | **64.3%** | 48.5% |
| defensive_pin | 54.5% | 35.7% | 40.7% | 57.1% | 45.5% |
| territorial_dominance | 45.5% | 35.7% | 40.7% | 64.3% | 45.5% |
| possession_stolen | 45.5% | 28.6% | 22.2% | 7.1% | 24.2% |
| progress_deficit | 27.3% | 0% | 18.5% | 21.4% | 16.7% |
| unclear_loss | 0% | 21.4% | 18.5% | 21.4% | 16.7% |

**关键对比**：029B 上的主力失败信号 `wasted_possession` **从 v1 的 7.1% (low_poss) 跳到 64.3%** —— 正是 fine-tune target 需要学的"有球不会进"问题。

#### 工程集成

- `cs8803drl/imitation/failure_buckets_v2.py` — 纯函数 classifier + 阈值常量
- `learned_reward_trainer.py` 新增 `--label-version {v1,v2}` CLI，**默认 v2**
- Trainer 用 `classify_failure_v2(meta["metrics"], outcome)` 在**训练时 re-derive 标签**（meta 存的仍是 v1，无破坏性）
- Checkpoint config 记录 `label_version` + `classifier_thresholds`
- 推理 wrapper [`learned_reward_shaping.py`](../../cs8803drl/imitation/learned_reward_shaping.py) 本就从 config 读 `head_names`，**v2 checkpoint 自动兼容**，无需改动

#### Dry-run 验证（80 ep, 20/dir, CPU, 1 epoch, ~4 min）

```
[cli] label_version=v2  thresholds={defensive_pin_tail_x: -3.0, ...}
[build] table size: 7272 samples  outcome: {team0_win: 68, team1_win: 12}

per-head sample count:
  defensive_pin          W_pos=6638  L_neg=302
  territorial_dominance  W_pos=6638  L_neg=302
  wasted_possession      W_pos=6638  L_neg=268
  possession_stolen      W_pos=6638  L_neg=110  ← 最少也比 v1 low_poss 多 6×
  progress_deficit       W_pos=6638  L_neg=256

[epoch 1/1] val_loss=0.2411 (mean/head)
  defensive_pin          val_loss=0.2569  acc=0.958
  territorial_dominance  val_loss=0.2627  acc=0.958
  wasted_possession      val_loss=0.2560  acc=0.949
  possession_stolen      val_loss=0.2110  acc=0.984
  progress_deficit       val_loss=0.2190  acc=0.967
```

**Caveat**:
1. Accuracy 被 W/L 类别严重倾斜膨胀（预测全 W 就 0.956），**Stage 2 全量训完后**需用 **W vs L 排序 AUC** 真实评估 head。
2. Val loss 0.21-0.26 稳定、无 NaN/爆炸，架构健康。
3. `possession_stolen` 是 v2 里最少的 head（L_neg=110），但比 v1 `low_poss` (L_neg=18) 多 6×。Stage 1 全部 ~250 L ep 到齐后应到 400+。

#### 决策：Stage 2 直接走 v2

不再 A/B 对比 v1 vs v2 —— 数据已证明 v2 每个 head 都有足够信号，对 SOTA target 的主力失败信号尤其关键。Stage 2 命令：

```bash
python -m cs8803drl.imitation.learned_reward_trainer \
  --traj-dir docs/experiments/artifacts/trajectories/036_stage1/029B_190 \
             docs/experiments/artifacts/trajectories/036_stage1/025b_080 \
             docs/experiments/artifacts/trajectories/036_stage1/017_2100 \
             docs/experiments/artifacts/trajectories/036_stage1/028A_1220 \
  --out-dir ray_results/reward_models/036_stage2 \
  --label-version v2 \
  --epochs 10 --batch-size 512
```

### 12.10 [2026-04-18 07:40 EDT] Stage 1 完成 + Stage 2 训完 + Stage 3 sanity 通过

#### Stage 1 汇总（全 4 lane 完成）

| Lane | Total | W | L | L fraction |
|---|---:|---:|---:|---:|
| 029B_190 | 800 | 671 | 129 | 16.1% |
| 025b_080 | 400 | 322 | 78 | 19.5% |
| 017_2100 | 400 | 311 | 89 | 22.3% |
| 028A_1220 | 400 | 322 | 78 | 19.5% |
| **total** | **2000** | **1626** | **374** | **18.7%** |

30 MB 压缩 .npz + .meta.json。耗时约 115 min（029B 是瓶颈）。

#### Stage 2 训练（v2 reward model）

命令：`learned_reward_trainer --label-version v2 --epochs 10 --batch-size 1024` on H200 GPU (srun overlap)。

| n_train | n_val | best epoch | best val_loss | 总耗时 |
|---:|---:|---:|---:|---:|
| 166,086 | 18,454 | **6** | **0.1644** | ~27s |

每 epoch 2.4-2.7 s（GPU vs 之前 CPU 估 2500s —— **1000× 加速**）。Epoch 6 后 val_loss 反弹（0.1644 → 0.1805），属正常 overfit，best checkpoint 自动保留。

Checkpoint: `ray_results/reward_models/036_stage2/reward_model.pt` (1.3 MB, 319k params)。

#### Stage 3 sanity: W/L 排序 AUC

`scripts/data/stage3_reward_model_sanity.py` — 对全 2000 ep 跑推理，per-episode score = mean over steps × mean over 5 heads × tanh。

```
=== Episode-level W/L ranking AUC (mean-over-heads tanh logit) ===
ALL          n_W=1626 n_L= 374  AUC=0.9772
017_2100     n_W= 311 n_L=  89  AUC=0.9807
025b_080     n_W= 322 n_L=  78  AUC=0.9847
028A_1220    n_W= 322 n_L=  78  AUC=0.9621
029B_190     n_W= 671 n_L= 129  AUC=0.9817

=== Score distribution ===
  team0_win: n=1626 mean=+0.9259 std=0.0809 range=[+0.5269, +0.9995]
  team1_win: n= 374 mean=+0.4607 std=0.2337 range=[-0.2910, +0.9886]

=== Per-head AUC ===
  defensive_pin          AUC=0.9761
  territorial_dominance  AUC=0.9767
  wasted_possession      AUC=0.9769
  possession_stolen      AUC=0.9756
  progress_deficit       AUC=0.9759
```

**读法**：
- ✅ Reward model **强烈区分** W vs L 轨迹 (AUC 0.977)，不是靠 class imbalance 糊过去
- ✅ 四个 policy 跨 AUC 0.962-0.985 高度一致，没有某个 lane 的数据主导
- ✅ W 均分 +0.93 vs L 均分 +0.46，**tanh 空间差距 0.47**，对 λ=0.01 的 shaping 权重，单步期望 shaping 是 ±0.005 量级 —— 和 v2 shaping coefficients 同阶，不会淹没 sparse ±3 goal reward

**需要警惕 / caveat**:
1. **5 个 head 的 AUC 几乎相同** (0.9756-0.9769) —— shared encoder 已经学到了"这是 W/L"的通用信号，5 个 head 没有真的学出 5 种独立的失败模式。好处是 5-head 等效于 ensemble 降方差；坏处是"head k 解释 k 号失败桶"的可解释性被打折。
2. AUC 是在**训练数据上**算的（只不过 val 是样本级随机划分而非 episode 级）。对 fine-tune 时 029B 会到达的 OOD 状态的泛化力**未测**。真实 PPO 训练是对这个的最终考核。
3. `team1_win` 下 score 有少数 ep 得到 +0.98 的高分 —— 说明模型在一些"输但看起来像赢"的 edge case 上可能给错 shaping 方向。不可避免；λ=0.01 小权重控制了影响面。

#### 决策

**绿灯 Stage 4**。036C-warm PPO batch 脚本 [`soccerstwos_h100_cpu32_mappo_036C_learned_reward_on_029B190_512x512.batch`](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_036C_learned_reward_on_029B190_512x512.batch) 的默认 `LEARNED_REWARD_MODEL_PATH` 就指向 `ray_results/reward_models/036_stage2/reward_model.pt`，可直接 sbatch。

### 12.11 [2026-04-18 07:45 EDT] Stage 4 首次启动失败 + 修复

**错误**:
```
ValueError: Key set for infos must be a subset of obs:
  dict_keys([0, 1, 2, 3, '_learned_reward_shaping']) vs dict_keys([0, 1, 2, 3])
```

**根因** — RLlib 多 agent 契约要求 `info.keys() ⊆ obs.keys()` (每个 key 必须是 agent_id)。`LearnedRewardShapingWrapper.step()` 原本在 info dict 顶层插了字符串 key `_learned_reward_shaping` 做诊断输出，RLlib 直接报 ValueError 拒绝 rollout。

对照 — `RewardShapingWrapper` 也加顶层 `_reward_shaping` key，但只在 `debug_info=True` 时（默认 False），所以生产训练不会触发。我的 wrapper 无条件加 → 炸。

**修复** ([learned_reward_shaping.py:185-195](../../cs8803drl/imitation/learned_reward_shaping.py#L185-L195))：改为把 per-agent shaping delta 挂到每个 agent 的子 info dict 里：

```python
# 改后：
if isinstance(info, dict):
    for aid, r_shape in shaping.items():
        agent_info = info.get(aid)
        if isinstance(agent_info, dict):
            agent_info["_learned_reward_shaping"] = float(r_shape)
            agent_info["_learned_reward_shaping_weight"] = self._shaping_weight
```

这样 info 的顶层 key 仍然是 `{0,1,2,3}`，诊断值通过 agent sub-dict 暴露，RLlib 契约不破。

**影响范围** — 没有消费者依赖旧的顶层 `info["_learned_reward_shaping"]`（grep 确认），snapshot §4.3 文档 (line 389) 提到的 failure capture hook 尚未消费这个字段，所以不存在破坏性修改。

**Stage 4 需重新启动** — 原失败目录 `PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_074352/` 只含 0 个 iteration，可以让下次运行覆盖 (RUN_NAME 有时间戳)。

### 12.12 [2026-04-18] Stage 4 首轮结果：baseline 轴很强，但数值异常必须先修

#### 首轮训练摘要

正式 run:

- run_dir: [`PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657`](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657)
- internal best eval checkpoint: [checkpoint-170](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000170/checkpoint-170) = `0.960 (48W-2L)`
- final checkpoint: [checkpoint-300](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000300/checkpoint-300)

但 internal `170=0.960` 在 official 下没有站住，因此首轮不按 internal 孤峰收口，而按 `official baseline 1000 + failure capture` 选主候选。

#### official baseline 1000

首轮 `official baseline 1000` 测了 `40 / 140 / 150 / 170 / 240 / 270 / 290`：

| checkpoint | WR | 读法 |
|---|---:|---|
| [40](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000040/checkpoint-40) | 0.828 | 早期真高窗 |
| [140](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000140/checkpoint-140) | 0.830 | 中段稳定高点 |
| [150](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000150/checkpoint-150) | 0.832 | baseline-oriented 主候选 |
| [170](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000170/checkpoint-170) | 0.815 | internal `0.960` 未兑现 |
| [240](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000240/checkpoint-240) | 0.819 | 后段中等平台 |
| [270](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000270/checkpoint-270) | **0.833** | official 最佳点 |
| [290](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000290/checkpoint-290) | 0.810 | 尾段普通平台 |

读法：

- `036C` 不是纯 spike。`40 / 140 / 150 / 270` 都是强窗口。
- 但 internal 最高点 [checkpoint-170](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000170/checkpoint-170) 明显被高估，因此首轮不再把它当主候选。

#### failure capture 500

两个最强窗口的 `failure capture 500`：

| checkpoint | official 1000 | capture 500 | 主要读法 |
|---|---:|---:|---|
| [150](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000150/checkpoint-150) | 0.832 | **0.844** | 当前更稳的 baseline-oriented 主候选 |
| [270](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000270/checkpoint-270) | **0.833** | 0.834 | official 最佳点，capture 也站住 |

失败桶对比：

| checkpoint | late_defensive_collapse | low_possession | poor_conversion | territory_loss | unclear_loss |
|---|---:|---:|---:|---:|---:|
| [150](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000150/checkpoint-150) | 40/78 | 19/78 | 11/78 | 1/78 | 7/78 |
| [270](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000270/checkpoint-270) | 33/83 | 21/83 | 11/83 | 4/83 | 14/83 |

补充观察：

- [150](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000150/checkpoint-150) 的败局更“短、清楚”，`team1_win` mean step `32.3`，`unclear_loss` 更低。
- [270](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000270/checkpoint-270) 更容易打成长局，`team1_win` mean step `44.0`，最大局长到 `492`，`unclear_loss` 更高。

因此当前更稳的读法是：

- **baseline-oriented 主候选：`036C@150`**
- **frontier 参考点：`036C@270`**

#### peer H2H（仅参考，不作为当前主判据）

首轮只补了一个关键 peer H2H：

- [036C@270 vs 029B@190](../../docs/experiments/artifacts/official-evals/headtohead/036C_270_vs_029B_190.log) = `211W-289L = 0.422`

这说明：

- `036C` 还没有在 peer 轴上追平 [029B@190](snapshot-029-post-025b-sota-extension.md)
- 但当前 snapshot 的主判据仍以 `vs baseline` 为主，H2H 这里仅作为参考，不据此否定 `036C` 的 baseline 正信号

#### 数值异常（当前真正 blocker）

首轮 `036C` 训练日志里出现了明显的 `inf`：

- [progress.csv](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/progress.csv)
  - `info/learner/shared_cc_policy/learner_stats/kl` 在 **48 / 300** 个 iteration 上为 `inf`
  - `info/learner/shared_cc_policy/learner_stats/total_loss` 在 **20 / 300** 个 iteration 上为 `inf`

这些 `inf` 分布在中后段多个 iteration（例如 `21-23, 64, 82, 105, 149, 167, 242, 299 ...`），说明当前 line 虽然已经给出很强的 baseline 结果，但**优先级应是先修数值稳定性，再决定是否扩大 peer-H2H 或继续 rerun**。

#### 当前 verdict（克制版）

- `036C` 首轮已经给出**真实的 baseline 正信号**，而且强度进入 frontier 讨论区。
- 在目前已有 `official baseline 1000` 中，它明显高于 `030A / 030D / 017 / 028A / 032 / 033`，但仍低于 `029B@190`。
- 当前更稳的主候选是 [checkpoint-150](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000150/checkpoint-150)，不是 internal 孤峰 [checkpoint-170](../../ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000170/checkpoint-170)。
- **下一步优先级**：先修 `kl / total_loss = inf` 的数值问题，再重跑 `036C`，而不是先继续给更多 peer H2H“机会”。

## 11. 相关

- [HW4 Section 1: MaxEnt IRL notebook](../references/DRL_HW4.ipynb)
- [SNAPSHOT-013: baseline-vs-baseline failure analysis](snapshot-013-baseline-weakness-analysis.md)（论证为什么纯模仿 baseline 上限是 50/50）
- [SNAPSHOT-015: BC bootstrap](snapshot-015-behavior-cloning-team-bootstrap.md)（类似 trajectory 采集）
- [SNAPSHOT-029: 029B@190 warm-start 源](snapshot-029-post-025b-sota-extension.md)
- [SNAPSHOT-034: deploy-time ensemble](snapshot-034-deploy-time-ensemble-agent.md)
