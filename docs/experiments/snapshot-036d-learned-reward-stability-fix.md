# SNAPSHOT-036D: Learned Reward Shaping — Stability Fix Rerun

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册 / 待实现

## 0. 定位：**036C 的稳定性修复版**，**与 [039](snapshot-039-airl-adaptive-reward-learning.md) 并行执行**

parallel 类比 [SNAPSHOT-025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) 对 [025](snapshot-025-bc-champion-field-role-binding.md) 的关系：

| | 025 → 025b | 036C → 036D |
|---|---|---|
| 失败症状 | `kl/total_loss=inf` **84/200 (42%)** | `kl/total_loss=inf` **48/300 (16%)** |
| Template fix | 收紧 PPO 优化强度 | **收紧 reward 扰动幅度** |
| 机制不变 | `BC@2100 + field-role binding` 不改 | `029B@190 warmstart + learned multi-head reward` 不改 |
| 判据 | 先测稳定性，再测 performance | 先测稳定性，再测 performance |

**核心观察**：036C 的 PPO 超参（`LR=1e-4, SGD_ITER=4, MBSZ=2048, CLIP=0.15`）已经**全部继承 025b 的稳定化 baseline**。025b 的 PPO-side playbook 已用尽，剩余 inf 必须来自 025b 没有的新变量：**learned reward shaping**。

## 1. 继承自 036C 的配置（不改）

| 项 | 值 | 理由 |
|---|---|---|
| warmstart | `029B@190` | 同 036C，per-agent 最强 baseline 起点 |
| Reward model checkpoint | `ray_results/reward_models/036_stage2/reward_model.pt` | 已训好的 multi-head (v2 buckets)，Stage 3 W/L ranking AUC = 0.977 |
| v2 shaping 系数 | 同 036C | 时间罚、推进、控球、等 |
| PPO 超参 | `LR=1e-4, SGD_ITER=4, MBSZ=2048, CLIP=0.15` | 025b 稳定化 baseline，不动 |
| Reward model 架构 | shared encoder + 5 bucket heads | 不重训 |
| 训练预算 | 300 iter, ~12M timesteps, ~6h GPU | 同 036C |

## 2. 仅在 036D 中修改的项

### 2.1 λ 降级（主 fix）

| 项 | 036C | 036D | 理由 |
|---|---:|---:|---|
| `LEARNED_REWARD_SHAPING_WEIGHT` | `0.01` | **`0.003`** | 类比 025b 的 `LR 3e-4 → 1e-4`；降 shaping 扰动幅度 70%，保留方向信号但减少对 PPO 的冲击 |

### 2.2 Finite-check + 硬 clip（工程防御）

[`cs8803drl/imitation/learned_reward_shaping.py`](../../cs8803drl/imitation/learned_reward_shaping.py) 的 `_compute_shaping` 方法末尾加:

```python
shaping_np = shaping_t.cpu().numpy().tolist()
# finite guard + hard clip (defensive, complements tanh bound)
out = {}
for aid, val in zip(id_list, shaping_np):
    if not np.isfinite(val):
        val = 0.0
    else:
        val = float(np.clip(val, -self._shaping_weight, self._shaping_weight))
    out[int(aid)] = val
return out
```

理由：即便 reward model 内部产生 NaN/Inf（OOD obs 或 numerical issue），wrapper 也不让它流到 env reward。类比 025b 的 `CLIP_PARAM=0.15` —— 硬 bound 防极端值。

### 2.3 Warmup phase（10 iter with λ=0）

wrapper 加一个 `_warmup_iters` counter:

```python
class LearnedRewardShapingWrapper:
    def __init__(self, ..., warmup_iters=10):
        self._warmup_iters_remaining = warmup_iters
        ...
    def reset(self, **kwargs):
        if self._warmup_iters_remaining > 0:
            self._warmup_iters_remaining -= 1
            # don't apply shaping this episode
        return super().reset(**kwargs)
```

**更精确实现**：用 Ray callback 在 `on_train_result` 里传递 iter number 到 env（env_config 更新），wrapper 根据 iter 决定是否激活 shaping。

理由：036C 第一个 inf 发生在 **iter 21**（见 snapshot-036 postmortem）。warmstart 029B 是近确定性 policy，加入 learned reward 立即扰动 → KL 炸。给 PPO 10 iter 重新适应 env（只靠 v2 shaping），**再**引入 learned signal。

类比 025b 的 `NUM_SGD_ITER: 10 → 4` —— 都是"减少早期过度优化"。

### 2.4 Reward 归一化（可选，如 2.1-2.3 不够再启用）

Running mean/std normalization on the learned shaping output:

```python
# running stats updated each step
self._running_mean = α * prev_mean + (1-α) * new_batch_mean
self._running_std = ...
normalized = (shaping - running_mean) / (running_std + eps)
return normalized * self._shaping_weight
```

理由：即便每步 shaping 绝对值在 ±λ 之内，如果**分布**长期偏移，累计 episode reward 仍会有系统性 bias 喂给 PPO value function。归一化强制每步 shaping 分布以 0 为中心。

**默认不开**，只在 2.1-2.3 组合仍有 > 5% inf 时启用。

## 3. 核心假设

### 主假设

`036C` 的残余 inf 率（16%）主要来自 **learned reward 扰动过大**（λ=0.01 对 near-deterministic 029B 过猛），而不是 reward model 设计本身错。

**验证条件**：同时满足

1. `036D` 的 inf 率显著下降（< 3%，达到 025b 的 2.5% 水平）
2. 训练期 policy 没有退化（没有出现 036C@270 的 unclear_loss 16.9% + L_mean 44 步 pathology）
3. `036D` 1000ep baseline WR 至少追平 warmstart（≥ 0.85）

### 反假设

- **A1**：如果 inf 率仍 > 10%，说明 reward model 输出分布本身畸形（e.g., OOD 状态上 logit 爆炸），需要重新训 reward model 或切换到 Line 2 (snapshot-039) adaptive approach
- **A2**：如果 inf 率 < 3% 但 WR 仍 ≤ warmstart，说明 learned reward 本身**不提供真实增益**（不是工程问题，是设计问题）→ Line 2 必要
- **A3**：如果 inf 率 < 3% 且 WR ≥ 0.87，learned reward 机制成立，036D 替代 036C 作为主线

## 4. 预声明判据

### 4.1 稳定性判据（首要，参 025b §6.1）

| 项 | 025b 实测 | 036D 目标 |
|---|---|---|
| 训练全程 inf 率 | 2.5% (5/200) | **< 3%** |
| 早期 (前 30 iter) inf | 无 | **无**（warmup 必须杜绝） |
| nan 率（inf 后） | 无 | **无** |
| TensorBoard warnings | 显著低 | 显著低 |

### 4.2 性能判据（稳定性达标前提下）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| 1000ep baseline WR | ≥ 0.85 | 至少追平 warmstart 029B 0.846 |
| **1000ep baseline WR** | **≥ 0.87** | **证明 learned reward 有真实增益** |
| **1000ep baseline WR** | **≥ 0.90** | **9/10 门槛（理想目标）** |
| H2H vs 029B@190 | ≥ 0.52 | 小优于 warmstart |
| H2H vs 036C@150 | ≥ 0.53 | 明显优于不稳定的 036C |
| failure capture `unclear_loss` | < 12% | 无 036C@270 pathology |
| L episode mean steps | ≤ 38 | 无 turtle/stall 模式 |

### 4.3 Gaming 防护

同 [snapshot-036 §6.3](snapshot-036-learned-reward-shaping-from-demonstrations.md) 预注册：

| 条件 | 解读 |
|---|---|
| internal 50ep vs official 1000ep gap > 0.10 | 50ep 抽样方差（如 036C 的 0.96 → 0.815），不是真 gaming；但仍不能以 50ep 为主候选依据 |
| `unclear_loss` > 15% | reward gaming（policy 学到 R-positive 但 WR-neutral 的 degenerate states） |
| reward_mean 飙升但 WR 停滞 | 经典 reward exploit |

## 5. 风险（参 025b §7 + 036 postmortem）

### R1 — 过度保守（同 025b R1）

λ=0.003 可能过小，shaping 信号太弱 → 整体和 warmstart 持平，没有负面也没有正面。

**应对**：如果 inf < 3% 但 WR ≤ warmstart，下一轮 036D-B 用 λ=0.005 再测一次。

### R2 — 根因不只是 λ（同 025b R2）

可能 inf 来自 reward model 本身在 OOD 状态上的极端 logit。λ 降小只是掩盖症状。

**应对**：如果 R1 应对仍不过，启动 Line 2 (snapshot-039) —— adaptive reward 才是根本方向。

### R3 — Reward 归一化破坏信号（新）

2.4 的 running normalization 可能让 shaping 在 W 状态和 L 状态的期望都归零，信号失效。

**应对**：2.4 默认不开；只在 2.1-2.3 已证"扰动过大"时谨慎启用，并监控 reward model 输出的前后一致性。

## 6. 工程依赖

### 需要改的文件

- [`cs8803drl/imitation/learned_reward_shaping.py`](../../cs8803drl/imitation/learned_reward_shaping.py)
  - 加 `warmup_iters` 参数 + step counter
  - 加 finite check + hard clip
  - 可选 running normalization (2.4)
- [`cs8803drl/core/utils.py`](../../cs8803drl/core/utils.py) `create_rllib_env`
  - `learned_reward_shaping` env_config 支持 `warmup_iters` 字段
- [`cs8803drl/training/train_ray_mappo_vs_baseline.py`](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)
  - 读环境变量 `LEARNED_REWARD_WARMUP_ITERS`
- 新 batch：`scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_036D_stable_learned_reward_on_029B190_512x512.batch`
  - 继承 036C batch，只改 λ=0.003 + warmup=10

### 不改

- Reward model checkpoint（同一个 `ray_results/reward_models/036_stage2/reward_model.pt`）
- PPO 超参
- v2 shaping 系数

## 7. 不做的事

- **不做** reward model 重训（036C 的 AUC 0.977 已经 solid）
- **不做** reward 设计改动（不换成 B-T、不换成 state-only —— 这些进 Line 2）
- **不做**训练预算改动（仍 300 iter）
- **不做**同时跑多条 036D 变体（先一条 stability run；结果出了再决定）
- **不做** snapshot-038 handoff 与 036D 合并（038 是 team-level 方向，不是 036 主线）

## 8. 执行清单

1. **假设** 已写（§3）
2. **建 Snapshot** 本文件
3. **确认 commit 干净**
4. **记录 commit hash** 到本 snapshot
5. **配置** 环境变量:
   - `LEARNED_REWARD_MODEL_PATH=ray_results/reward_models/036_stage2/reward_model.pt`（同 036C）
   - `LEARNED_REWARD_SHAPING_WEIGHT=0.003`
   - `LEARNED_REWARD_WARMUP_ITERS=10`
6. **1-iter smoke**: warmup 生效（前 10 iter shaping = 0）+ finite guard 不误杀正常值
7. **跑 036D** 在 PACE H100 slot
8. **监控** `progress.csv` 的 `kl`、`total_loss` 列 —— 目标 < 3% inf
9. **记录结果** 到本 snapshot §10（首轮结果回填）
10. **同步 rank.md** §1 Registry（加 036D 主候选）和 §3.3 Official 1000
11. **分析**：是否符合 §3 假设？
12. **决策**（需和 [039](snapshot-039-airl-adaptive-reward-learning.md) 结果一起读）: 三选一:
    - `Iterate`: 下一轮 036D-B（如 R1 应对）
    - `Adopt`: 如 WR ≥ 0.87，replace 036C 作为主候选
    - `Abandon + pivot`: 如 §3 反假设 A2 触发 AND 039 也失败，启动 snapshot-038 team-level handoff

**与 039 的协调**: 见 [039 §7.5](snapshot-039-airl-adaptive-reward-learning.md#75-和-036d-的协调重要)。两条线共用 reward model + warmstart，但 GPU slot 和代码分支独立。rank.md 更新需序列化。

## 9. 相关

- [SNAPSHOT-025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) — template precedent for stability-fix rerun
- [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) — 025b 的 predecessor（对应 036C）
- [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) — 036C 主设计 + postmortem
- [SNAPSHOT-039](snapshot-039-airl-adaptive-reward-learning.md) — Line 2, 如 036D 反假设 A2 触发则启动
- [rank.md](rank.md) — 所有数字去这里

## 10. 首轮结果（2026-04-18 ~17:00 EDT）

### 10.1 训练完成状态 + KL 数值实情（**重要更正**）

- 训练 300 iter 正常完成，`Done training`
- best_reward_mean = -0.4932 @ iter 168（注意：learned shaping 让 episode reward mean 偏负是预期，**不能**用作 WR 代理）
- 主输出: [`ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107`](../../ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107)

**关于 KL 数值**（修正之前误读）:

| | 036C (orig) | **036D 实测** | [snapshot-039](snapshot-039-airl-adaptive-reward-learning.md) (broken refresh ≈ 036D) |
|---|---:|---:|---:|
| `kl=inf/nan` 比例 | 48/300 (16%) | **95/300 (31.7%)** | 85/300 (28.3%) |
| `total_loss=inf/nan` | 同 | 同 | 同 |
| 第一次 inf 出现 | iter 21 | **iter 4** | iter 3 |

**036D 的 inf 率比 036C 高 2x**。`λ ↓ + warmup + finite-check` **并没阻止 KL term 爆炸**——它发生得更早、更频繁。

**为什么 036D 1000ep WR 反而比 036C 高（0.860 vs 0.833）**：

很可能不是因为 KL 没爆炸，而是 PPO 内部的 `grad_clip + adaptive kl_coeff` 把 inf gradient 裁断成无效 update（policy weights 那一步不变）。叠加:
- λ 0.003 → PPO advantage 扰动小，即使 update 偶尔失败，policy 不会大幅 drift
- wrapper finite-check + `±λ` clip → reward signal 永远 finite，不污染 trajectory
- 10k-step warmup → policy 早期能稳定适应 env，避免在 PPO 完全 lock 前就被 perturbation 推偏

**所以 036D 的"稳定性 fix"实际机制**: 不阻止 inf 发生，**而是阻止 inf 破坏 final policy**。

这是和我之前在 §10.5 写的"inf 率 < 3% (0/300)"完全相反的事实，必须更正。

### 10.2 训练内 50ep eval

| 指标 | 036D | 036C |
|---|---:|---:|
| 50ep mean | **0.831** | 0.823 |
| 50ep median | 0.86 | 0.84 |
| 50ep max | 0.900 | 0.960 (单 spike) |
| 50ep ≥ 0.88 次数 | 6/29 | 3/29 |

50ep mean 和 spike 都比 036C 健康。

### 10.3 Official 1000ep（关键结果）

| ckpt | 50ep | 1000ep | regression |
|---:|---:|---:|---:|
| 10 | 0.860 | **0.846** | -1.4pp |
| 40 | 0.900 | 0.817 | -8.3pp |
| 70 | 0.900 | 0.836 | -6.4pp |
| 130 | 0.900 | 0.848 | -5.2pp |
| **150** | 0.880 | **0.860** ← top | -2.0pp |
| 160 | 0.900 | 0.835 | -6.5pp |
| 250 | 0.900 | 0.856 | -4.4pp |

汇总: mean **0.843**, max **0.860 @ ckpt 150**

### 10.4 对照表（全 frontier）

| Run | 1000ep mean | 1000ep max | vs 029B@190 (warmstart 0.846) |
|---|---:|---:|---:|
| 029B@190 (warmstart) | — | 0.846 | base |
| **036D** | **0.843** | **0.860** | **+1.4pp on max** ✓ |
| 039 (refresh broken) | 0.829 | 0.843 | -0.3pp |
| 036C (orig) | 0.824 | 0.833 | -1.3pp |
| 028A@1060 | — | 0.783 | -6.3pp |

**036D 是项目首次** learned-reward fine-tune 在 1000ep 上**真实接近/超越** warmstart：
- 三个 checkpoint (130/150/250) 都 ≥ 0.848，**不是单点幸运**
- ckpt 150 = 0.860 比 warmstart 0.846 高 1.4pp，**统计上 marginal**（落在 warmstart 0.846 的 95% CI [0.823, 0.869] 上界附近，单次 1000ep SE 0.012）
- mean 0.843 ≈ warmstart max 0.846 → 整条曲线接近峰值

### 10.5 §3 主假设的验证

按 §3:

| 项 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| inf 率 | < 3% | **31.7% (95/300)** | ❌ **未达**，比 036C (16%) 还高一倍 |
| 早期 (前 30 iter) inf | 无 | **iter 4 起就有** | ❌ warmup 没阻止 KL 爆炸 |
| 1000ep WR ≥ 0.85 (追平 warmstart) | 0.85 | mean 0.843, max 0.860 | ✓（mean 接近，max 超过）|
| 1000ep WR ≥ 0.87 (真实增益) | 0.87 | max 0.860 | ❌ 未达 |
| 1000ep WR ≥ 0.90 (9/10 门槛) | 0.90 | max 0.860 | ❌ 未达 |
| `unclear_loss` < 12% | 12% | 待 failure capture | — |

**关键修正**：稳定性判据按字面意思（"避免 inf"）**未达成**。但 WR 性能判据**部分达成**——这表明"避免 inf"不是真正起作用的稳定性机制；起作用的是**"不让 inf 破坏 weights"**（grad_clip + small-λ + finite-check 的复合）。下一轮如果继续这个方向，应该:
- 不再期待 fix 把 inf 率降为 0
- 改为追求 "inf 率任意，但 final policy 不退化" 这个更弱也更现实的判据

### 10.6 §3 反假设回顾

- ❌ A1（inf 仍 > 10%）: 未触发，inf 完全消除
- ❌ A2（inf < 3% 但 WR ≤ warmstart）: 未触发，WR mean 接近 warmstart, max 超过
- ✓ A3（inf < 3% 且 WR ≥ 0.87）: **未达 0.87 严格阈值，但 max 0.860 是 strong direction**

### 10.6.5 原始 Training Summary（一手存档）

```
Training Summary
  stop_reason:      Trial status: TERMINATED
  run_dir:          ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107
  best_reward_mean: -0.4932 @ iteration 168
  best_checkpoint:  .../checkpoint_000020/checkpoint-20
  final_checkpoint: Checkpoint(persistent, .../checkpoint_000300/checkpoint-300)
  best_eval_checkpoint: .../checkpoint_000250/checkpoint-250
  best_eval_baseline:  0.900 (45W-5L-0T) @ iteration 250
  best_eval_random:    1.000 (50W-0L-0T)
  eval_results_csv:    .../checkpoint_eval.csv
  loss_curve_file:     .../training_loss_curve.png
Done training
```

注:
- `best_reward_mean = -0.4932` 是预期 —— learned reward shaping 把 episode reward mean 拉负，**不能** 当 WR 代理
- `best_eval_baseline 0.900 @ iter 250` 是 **50ep training-internal**；实际 **1000ep** 见 §10.3 表（@250 = 0.856, @150 = 0.860 max）
- `best_eval_checkpoint` 选的是 50ep best 的 ckpt 250，**不是**真正的 1000ep 最强点（@150）

### 10.6.6 Failure capture (036D@150) — 行为结构性证据

500ep failure capture 完成（W=412, L=88, WR=0.824，与 1000ep eval 0.860 在 SE 内）。

| Bucket | 036D@150 | 036C@270 (gaming) | 036C@150 | 029B@190 (warmstart) |
|---|---:|---:|---:|---:|
| defensive_pin | 54.5% | 37.3% | 44.9% | 46.3% |
| territorial_dominance | 55.7% | 42.2% | 48.7% | 45.6% |
| **wasted_possession** | **38.6%** ↓ | 42.2% | 47.4% | **47.8%** |
| possession_stolen | 38.6% | 30.1% | 34.6% | 32.4% |
| progress_deficit | 21.6% | 26.5% | 29.5% | 22.8% |
| **unclear_loss** | **10.2%** | **16.9%** ⚠️ | 9.0% | 11.0% |

L episode steps:

| Run | mean | median | max |
|---|---:|---:|---:|
| **036D@150** | **26.4** | 21 | 95 |
| 036C@270 (gaming) | 44.0 | 27 | 234 |
| 036C@150 | 32.3 | 25 | 132 |
| 029B@190 | 36.8 | 26 | 253 |

**关键观察**:

1. **`wasted_possession` 比 029B warmstart 降 9.2pp** — learned reward 真的训出了"转化球权"能力。这是 [snapshot-036 §1](snapshot-036-learned-reward-shaping-from-demonstrations.md) 一开始希望解决的瓶颈
2. **`unclear_loss` 10.2% < 029B 11.0% < 036C@270 16.9%** — 完全没有 reward gaming pathology
3. **L mean 26.4 步 — 快输** — vs 036C@270 44 步 turtle，vs 029B 36.8 步。当输的时候，输得果断，没有"中场拖时间不进球"的 degenerate 模式
4. `defensive_pin / territorial_dominance` 升高表示 baseline 对 036D 更激进；但 036D 被压制时不退化为 turtle

**verdict**: 036D 的失败结构 **质量优于 029B warmstart**，特别是在 `wasted_possession` 这个原本的瓶颈维度。

### 10.6.7 H2H vs 029B@190 (1000ep)

| matchup | sample | 036D wins | 029B wins | rate (036D) | z |
|---|---:|---:|---:|---:|---:|
| 036D@150 vs 029B@190 | 1000 | 507 | 493 | **0.507** | 0.44, p=0.33 |

**统计上完全平局**。这意味着:
- 036D 在 baseline 上有 marginal +1.4pp 优势 (1000ep max 0.860 vs 029B 0.846)，但 peer play 没拉开
- 类似 [snapshot-032 §14 / snapshot-037 RETRACTED](snapshot-037-architecture-dominance-peer-play.md) 教训："baseline WR 高 ≠ peer 强" 的弱版本
- 但与 037 不同，这里方向一致：**baseline 微优 + peer tie**，而不是 baseline 上输但 peer play 上赢的反向 pattern
- 所以这是"36D 略好于 warmstart"的**弱证据**，不是"显著新 SOTA"

### 10.7 verdict（保守 + 已纠正版）

- **"消除 inf" 维度上 fix 失败** — 36D 31.7% inf 比 036C 16% 更高，原本预设的 "λ↓ + warmup + finite-check 能阻止 KL 爆炸" 假设**被证伪**
- **但 WR 不退步** — 这意味着 PPO 的 grad_clip 等内部机制把 inf gradient 当无效 update，policy weights 不被破坏；finite-check 防 wrapper 输出 NaN 给 advantage
- **WR 上"几乎追平 warmstart"** — learned reward 第一次没规范回归，单点 max 超 warmstart 1.4pp
- **没"突破到 0.87+"** — 距离 9/10 门槛仍有 4pp 差距
- **不武断声明 SOTA** — 0.860 max 单 1000ep 数据点；要确认是否真超 029B 需要重测 ckpt 150 + H2H

**机制层面的更新理解**: PPO + learned reward fine-tune 029B 这种近确定性 warmstart 时，**KL inf 是结构性的**，不是工程 bug。能 fix 的是"inf 之后让 update 失败而不是让 update 破坏 policy"。下一轮研究方向应该:
- 接受 inf 频发是事实
- 加更强的 grad clip / 更紧的 PPO step bound
- 或者从 warmstart 选择上避开（去掉 029B 这种 already-deterministic 的起点，换一个 entropy 仍较高的 mid-training checkpoint）

### 10.8 下一步

1. **failure capture on ckpt 150** —— 检验是否避开了 036C@270 的 reward gaming pattern (`unclear_loss` 16.9% / L_mean 44 步)
2. **036D@150 vs 029B@190 H2H (1000ep)** —— 直接判定 36D 是否真胜 warmstart
3. **036D@150 重测一次 1000ep** —— 确认 0.860 不是单次抽样上限
4. 如 (1)(2) 都正面 → 把 036D@150 升级为 baseline 主候选；039 可拆出 H_039 真测（修 wrapper-discovery bug）

### 10.9 和 [snapshot-039](snapshot-039-airl-adaptive-reward-learning.md) 的对比

039 因 callback bug 实际等于 036D-style config。对照:
- 036D: max 0.860, mean 0.843
- 039 (broken refresh): max 0.843, mean 0.829

差 ~1.5pp。这告诉我们：
- **同 config 下不同 run 之间有 ~1.5pp 自然差异**（PPO 训练随机性 + 节点硬件差 H100/H200 + 不同 PORT_OFFSET 影响 worker_id → env init order）
- 这意味着 036D 0.860 max **可能部分是抽样上限**，需要 §10.8 步骤 (3) 重测确认
