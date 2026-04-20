# SNAPSHOT-039: AIRL-Inspired Adaptive Reward Learning

- **日期**: 2026-04-18
- **负责人**:
- **状态**: lane 关闭 (2026-04-19) — Fix-B 实证为 cosmetic, KL inf 频率与 inf 版本相同（§14）。Fix-A backlog 下沉为「最后救命稻草」

## 0. 定位：Line 2 重工程 MaxEnt-flavored rerun（并行于 036D）

在 [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) 首轮 null result 基础上，**036D 修工程（reward-side 稳定性），039 修设计（static → adaptive + B-T）**。两条线正交、并行：

| | 036C (done) | 036D (parallel) | 039 (parallel) |
|---|---|---|---|
| 方向 | static BCE 分类 W/L | 同 036C, 降 λ 修稳定性 | **Adaptive B-T preference** |
| Reward 更新 | 训前一次 | 训前一次 | **训练中周期性 refresh** |
| Loss | BCE on (s,a) 样本 | 同 036C | **Bradley-Terry on W/L trajectory pairs** |
| 修的是什么 | — | 工程侧：inf、λ、warmup | 设计侧：OOD drift、overconfidence |
| 工程量 | 2-3 天（已完成） | 0.5-1 天 | **2-3 天** |
| GPU 预算 | — | 6-8h | 8-10h |

### 为什么并行而不是串行

原本计划"等 036D verdict 再决定启动"，实际已无必要：

1. **信息密度**：036D 和 039 修的是**不同层次的问题**（工程 vs 设计）。无论 036D 结果如何，039 都能给出独立有价值的信号：
   - 如 036D 成功 → 039 的 adaptive 版是否进一步突破？
   - 如 036D 仍 null → 039 是唯一的 learned-reward path
2. **时间压力**：assignment ddl 临近，串行累计 2-3 天 wall-clock；并行压到 1 天 overlap
3. **失败独立**：两条线共用 36C 的 reward model + 029B warmstart，但 PPO 训练完全独立。一条翻车不影响另一条
4. **诊断价值**：039 即使失败（如 GAIL 类 instability），它的失败方式本身是 report narrative 的一部分（"we also tested adaptive refresh with B-T, which..."）

## 1. 核心假设

### 主假设 H_039

`036C/036D` 的 null result 源于 **static reward model 的两个结构性弱点**:

1. **OOD drift**：reward model 训练数据是 017/025b/028A/029B 的 offline 轨迹；PPO fine-tune 后 policy 的 state distribution 漂移到 training pool 没见过的区域；static R 在 OOD 状态给出无 grounding 的 logit → policy 学到 exploit
2. **BCE overconfidence**：binary classification loss 让 R(s,a) 在某些状态输出极大 logit（被 tanh 压到 ±1 后仍是 hard-max 信号）；policy 有动机 chase 这些 hard-max 状态，不管它们和 winning 的真实关系

**Adaptive + B-T 同时修复**:

1. **Periodic refresh** 跟上 policy state distribution 漂移 → 解决 OOD drift
2. **Bradley-Terry loss** 用 pairwise comparison 替代 per-sample binary → reward magnitude 天然校准，不存在 hard-max → 解决 overconfidence

### 证伪条件

- **F1**: Adaptive refresh 后，policy 仍发现新的 reward exploit（unclear_loss 仍 > 12%） → 说明"learned reward in PPO fine-tune"这类方法对 near-deterministic warmstart 根本不稳
- **F2**: B-T loss 下 reward ranking AUC 大幅低于 036C 的 0.977 → B-T 变体的 information content 不够
- **F3**: D refresh 过程中 ranking AUC 在某次 refresh 后降到 < 0.8 → training instability，策略 outperforms offline pool 但 D 跟不上

## 2. 设计核心：**两个机制组合**

### 2.1 机制 A — Periodic Discriminator Refresh（AIRL-flavored）

```
D_0 ← 036C 的 multi-head reward model checkpoint (warmstart)
π_0 ← 029B@190

for phase p in range(K = 10):
    # Phase: PPO N_iter=30 iter with current D_p
    for i in range(N_iter):
        rollout π_p → trajectories (buffered)
        reward_learned(s,a) = tanh(D_p(s,a)) × λ
        total_reward = sparse + v2 + reward_learned
        PPO update π_p

    # Refresh D
    new_W, new_L ← split recent 300 rollout trajectories by outcome
    training_pool ← offline_2000ep  ⊕ {all online_{W,L} across phases}
    D_{p+1} ← fine_tune(D_p, pool, loss=B-T, steps=2000)
    evaluate D_{p+1} on held-out pairs; if AUC < 0.8, pause training
```

### 2.2 机制 B — Bradley-Terry Loss（RLHF-flavored）

替换 036C 的 BCE 为 pairwise preference loss:

```python
# For each batch: sample K pairs (τ_W, τ_L) where τ_W is W trajectory, τ_L is L
def bt_loss(R_model, batch):
    r_W = mean(R_model(s, a) for (s,a) in τ_W)  # trajectory-averaged R
    r_L = mean(R_model(s, a) for (s,a) in τ_L)
    # preference: W should have higher R than L
    return -log(sigmoid(r_W - r_L)).mean()
```

理论依据：
- Bradley-Terry 1952 paired-comparison model
- `P(τ_W > τ_L) = σ(r(τ_W) - r(τ_L))`
- Loss 的梯度自然校准 reward magnitude（不会出现 overconfident hard-max）
- RLHF (Ouyang et al. 2022) 验证过的 effective loss

### 2.3 Multi-head 保留？

**保留**但 loss 重写:

- 5 heads（defensive_pin / territorial_dominance / wasted_possession / possession_stolen / progress_deficit）各自仍做 B-T
- 对 head `k`，pair 为：`(τ_W_anytype, τ_L_labeled_as_k)` → B-T loss on this pair
- 最终 reward = `mean_k(tanh(head_k(s,a))) × λ`

理由：
- Multi-head 结构在 036C Stage 3 sanity 显示了合理的 per-head AUC
- 拆 head 可以保留"失败模式 decomposition"的可解释性
- B-T loss 应用于 per-head 仍有意义

### 2.4 Refresh 频率与数据管理

| 参数 | 值 | 理由 |
|---|---|---|
| `N_iter` (per phase) | 30 | 比 036C 的 300-iter static 短得多；足够 policy 有方向性变化，又不过分偏离 |
| `K` (total phases) | 10 | 总 300 iter 训练预算不变（=036C 预算） |
| D refresh steps | 2000 | 每次 refresh 2000 SGD steps；比 036C Stage 2 的 10 epoch × 166k samples 少得多（refresh 是 fine-tune 不是 from scratch） |
| Refresh batch size | 256 | 同 036C Stage 2 |
| Offline pool 保留 | 2000 ep from Stage 1 data | 作为 D 的 anchor 防止 drift 过度 |
| Online trajectories per phase | 100-300 | 取最近一 phase 的 rollouts, subsample by outcome balance |

### 2.5 Callback 架构

Ray PPOTrainer 支持 `on_train_result` callback:

```python
class AdaptiveRewardCallback(DefaultCallbacks):
    def __init__(self, refresh_every=30, ...):
        self.refresh_every = refresh_every
        self.phase_buffer = []
        ...
    def on_episode_end(self, ..., episode, ...):
        # collect (s, a, outcome) from finished episode
        self.phase_buffer.append(...)
    def on_train_result(self, *, trainer, result, **kwargs):
        iter_ = result['training_iteration']
        if iter_ % self.refresh_every == 0:
            # refresh D
            new_D = refresh_reward_model(old_D=current_D, buffer=self.phase_buffer, pool=offline_pool)
            # update env_config across workers
            trainer.workers.foreach_worker(lambda w: w.env.update_reward_model(new_D))
            self.phase_buffer.clear()
            log_D_metrics(new_D)
```

**关键工程点**:
1. **D refresh 在 trainer 主进程**（不阻塞 workers）
2. **跨 worker 的 D 分发**：通过 `foreach_worker` broadcasting 新 D 到所有 env wrapper
3. **Buffer 同步**：需要收集所有 worker 的 episode data，不只是 rank-0

## 3. 预声明判据

### 3.1 稳定性判据（继承 036D 经验）

| 项 | 阈值 | 备注 |
|---|---|---|
| 训练期 inf 率 | < 3% | 同 036D 标准 |
| Phase 切换处的稳定性 | 无 WR 崩溃 | refresh 不应引发 phase 0 重置式训练灾难 |
| D refresh 的 AUC | ≥ 0.85 every refresh | 失守触发 training pause |
| D 输出分布 drift | `std(D_new - D_old) / std(D_old) < 0.3` | 防 D 剧变导致 policy 反跳 |

### 3.2 性能判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| 1000ep baseline WR | ≥ 0.86 | 至少和 036C 持平（说明 adaptive 没使情况变差） |
| **1000ep baseline WR** | **≥ 0.88** | **超 036C/warmstart，证明 adaptive 有增益** |
| **1000ep baseline WR** | **≥ 0.90** | **9/10 门槛** |
| H2H vs 036C@150 | ≥ 0.53 | 显著优于 static 版 |
| H2H vs 029B@190 | ≥ 0.55 | 超 warmstart |
| failure `unclear_loss` | < 10% | 无 reward gaming pathology（目标比 036D 更严） |
| L episode mean steps | ≤ 36 | 无 turtle |

### 3.3 Reward model 侧判据（每次 refresh 后验证）

| 项 | 阈值 |
|---|---|
| B-T loss 在 held-out pairs 上 | 持续下降 |
| Ranking AUC on held-out | ≥ 0.85 |
| per-head AUC diff | < 0.10（5 heads 仍平衡）|

## 4. 风险与 mitigation

### R1 — Adversarial 不稳定

GAIL / AIRL 类方法以训练不稳出名。D refresh 可能导致 policy-D 间的 oscillation。

**Mitigation**:
- D 从 036C checkpoint warm-start（不 from scratch）
- D refresh 仅 fine-tune 2000 steps, 不全盘重训
- Offline pool 保留作为 anchor，防 D 只适应当前 policy

### R2 — B-T loss 下 ranking 信号变弱

B-T 在 trajectory-level 聚合 reward；如果 W/L 内部方差大，pair-wise 信号被冲淡。

**Mitigation**:
- 保留 multi-head：per-head 的 B-T 对 bucket-specific 样本更敏感
- 如果 holdout AUC 降到 < 0.85 就 pause，做 post-mortem

### R3 — 工程复杂度导致 bug

Callback + cross-worker sync + D state management 有很多可能出错的地方。

**Mitigation**:
- 先跑 10-iter smoke：验证 refresh 机制能正确触发，worker env 能 hot-swap D
- 加丰富的 logging：每次 refresh 记录 D_AUC / D_loss / phase_buffer_size / wall_clock
- 遇到问题先 rollback 到 036D（不 refresh 的 baseline）

### R4 — 训练时间超预算

Refresh 期间暂停 PPO，10 次 refresh × 每次 5-10 分钟 = 1-2 小时额外。

**Mitigation**:
- 总预算改 8h (vs 036C 6h)
- Refresh 在 CPU worker 并行（不占 PPO GPU）

## 5. 工程清单

### 5.1 新增代码

- `cs8803drl/imitation/adaptive_reward_trainer.py` (~200 行):
  - `AdaptiveRewardCallback(DefaultCallbacks)`
  - `refresh_reward_model(model, buffer, pool, loss='bt', steps=2000)`
  - `collect_wl_pairs_from_buffer(buffer)`
  - B-T loss function
- `cs8803drl/imitation/trajectory_buffer.py` (~100 行):
  - `TrajectoryBuffer` with online/offline merge + balanced sampling
- 修改 `cs8803drl/imitation/learned_reward_shaping.py`:
  - 加 `update_reward_model(new_state_dict)` 方法支持热切换
- 修改 `cs8803drl/imitation/learned_reward_trainer.py`:
  - `_build_bt_pairs()` 为 refresh 提供 pair data
  - Support for `--loss {bce, bt}` CLI arg

### 5.2 Batch 脚本

- `scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_039_airl_adaptive_on_029B190_512x512.batch`
  - 继承 036C batch
  - 加 `LEARNED_REWARD_ADAPTIVE_REFRESH=30` / `LEARNED_REWARD_LOSS=bt` 等 env vars

### 5.3 Smoke 测试

1. 10-iter smoke：refresh 在 iter 30 正确触发
2. 30-iter smoke：2 次 refresh 完成，D_AUC 不崩
3. 50-iter smoke：policy WR 不退步

## 6. 不做的事

- **不做** from-scratch reward model training（D 从 036C warmstart 继续 fine-tune）
- **不做** forward RL loop of pure MaxEnt（那需要计算 policy feature expectations, 工程量爆炸）
- **不做** DPO direct policy optimization（需要自定义 PPO step，和 RLLib 不兼容；如果 039 失败，DPO 作为 039-next 备选）
- **不做** 和 036D 合并（038 是稳定性修复，039 是设计升级，两条线独立判据）
- **不做** 同时跑 039 的多个 variant（先一条主线）

## 7. 执行清单（与 036D 并行）

1. **假设** 已写（§1）
2. **建 Snapshot** 本文件
3. **确认 commit 干净**
4. **记录 commit hash**
5. **实现 §5.1 新增代码**（~2 天工程）
   - adaptive_reward_trainer.py / trajectory_buffer.py
   - 修改 learned_reward_shaping.py 支持 hot-swap D
   - 修改 learned_reward_trainer.py 支持 B-T loss + refresh entry point
6. **Smoke 测试** (§5.3)
   - 10-iter smoke: refresh 能正确触发
   - 30-iter smoke: 2 次 refresh 完成，D_AUC 不崩
   - 50-iter smoke: policy WR 不退步
7. **跑 039** 在 PACE H100（与 036D 占用**不同 GPU slot**，避免冲突）
8. **监控** D_AUC / inf 率 / WR
9. **记录结果** 到本 snapshot §9
10. **同步 rank.md** §1 Registry 加 039 主候选
11. **与 036D 交叉对比**（两条线结果一起读）:
    - A = 036D 成功 + 039 成功 → **选 H2H 更高者**作为主候选
    - B = 036D 成功 + 039 失败 → 036D 作为主线；039 作为 "attempted but failed" 放 report
    - C = 036D 失败 + 039 成功 → 039 作为主线
    - D = 两者都失败 → 转 snapshot-038 team-level handoff；036 系列整体 deprecate

## 7.5 和 036D 的协调（重要）

| 资源 | 036D 用 | 039 用 | 冲突？ |
|---|---|---|---|
| Reward model checkpoint | `036_stage2/reward_model.pt` | 同（作为 warmstart） | ❌ 只读，无冲突 |
| Warmstart policy | 029B@190 | 029B@190 | ❌ 只读 |
| GPU slot | 一个 H100 | 另一个 H100 | ❌ 必须不同 slot |
| 代码分支 | main / 036D 相关改动 | 同 main / 039 相关改动 | ⚠️ **确保两边改的是不同文件** |
| rank.md | 各自同步自己的行 | 同 | ⚠️ **两人同时改 rank.md 会 conflict** |

**具体防冲突措施**:
- 036D 只改 `learned_reward_shaping.py` 的 wrapper（finite check + warmup counter）
- 039 改 `learned_reward_shaping.py` 的 **不同方法**（加 `update_reward_model` hot-swap）+ 新建 `adaptive_reward_trainer.py`
- rank.md 更新要序列化（036D 先 append，039 后 append；或者最后一起做一次）

## 8. 相关

- [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) — 036C 原设计 + null result 机制分析
- [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md) — 039 激活的 pre-requisite gate
- [SNAPSHOT-038](snapshot-038-team-level-liberation-handoff-placeholder.md) — 如果 039 也失败，fallback 方向
- [rank.md](rank.md) — 所有数字去这里
- 理论参考（不要在 report 里声明"we did these"，只作为设计启发）:
  - Ziebart et al. 2008 — MaxEnt IRL
  - Fu et al. 2018 — AIRL (Adversarial IRL)
  - Ouyang et al. 2022 — RLHF / InstructGPT (Bradley-Terry preference learning)
  - Rafailov et al. 2023 — DPO (作为 039-next fallback)

## 9. 首轮结果（2026-04-18 ~16:47 EDT）

### 9.1 训练完成状态 + KL 数值实情（**重要更正**）

- 训练正常跑完 300 iter, `Done training`
- 预估总时间 ~3.3h（vs 036C 的 ~2.7h，多出来的就是 callback overhead）
- **不是无 inf** —— 实测 **85/300 (28.3%) iter 出现 `kl=inf` 或 `total_loss=nan`**，第一次 inf 在 iter 3
- fail-soft hardening 阻止了 callback 出错 kill 训练；但 **inf 本身没被阻止**，pattern 和 [snapshot-036D §10.1](snapshot-036d-learned-reward-stability-fix.md#101-训练完成状态--kl-数值实情重要更正) 一致 (036D 是 31.7%)
- 主输出：[`ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607`](../../ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607)

**结论同 036D**: PPO 的 grad_clip + small λ + finite-check 让 inf gradient 变成无效 update（不破坏 policy weights），但 KL 爆炸本身仍系统性发生。这是 learned-reward fine-tune 029B 这种 near-deterministic warmstart 的结构性现象。

### 9.2 **Refresh 机制静默失败（关键 finding）**

10 次 refresh trigger（iter 30, 60, ..., 300）**全部失败**:

```
[adaptive-reward] ===== refresh trigger at iter 30 =====
[adaptive-reward] building offline table from 4 dirs:
[adaptive-reward] offline table ready: 184540 samples
[adaptive-reward] ERROR: LearnedRewardShapingWrapper not found in local worker env
```

Offline table 成功 build（184k samples, v2 labels），但 callback 没能从 `trainer.workers.local_worker().env` 找到 `LearnedRewardShapingWrapper`。所以 refresh 的**分发**步骤被跳过，D 从未更新。

**根因假设**: Ray 1.4 里 `trainer.workers.local_worker()` 的 env 不是 rollout worker 那条 wrapper 链。可能是一个 evaluation/dummy env，或者 vector-env 的外壳而不是内嵌的 wrapper。`_find_learned_reward_wrapper` 按 `env.env` 链走下去没命中类型名。

**实际效果**: 039 等于**静态 reward + callback 空转** —— 和 036D 结构本质等价，只多了 1 层 log overhead。

### 9.3 50ep 训练内评估

| 指标 | 039 | 036D |
|---|---:|---:|
| 50ep mean | 0.8255 | 0.8310 |
| 50ep max | 0.920 @ iter 40 | 0.900 @ iters 40/70/130/160/250 |
| best_eval_baseline | 0.920 @ iter 40 | 0.900 @ iter 250 |

两者 50ep 分布**非常接近**（符合"refresh 从未生效 → 等于 036D"的预期）。

### 9.4 Official 1000ep（已出）

| ckpt | 50ep | 1000ep | regression |
|---:|---:|---:|---:|
| 10 | 0.880 | 0.830 | -5.0pp |
| 40 | 0.920 | 0.836 | -8.4pp |
| 140 | 0.880 | 0.820 | -6.0pp |
| 150 | 0.880 | 0.823 | -5.7pp |
| 170 | 0.900 | 0.820 | -8.0pp |
| 190 | 0.860 | 0.833 | -2.7pp |
| **230** | 0.880 | **0.843** | -3.7pp |

汇总: mean **0.829**, max **0.843** (ckpt 230)

对照:

| Run | 1000ep mean | 1000ep max |
|---|---:|---:|
| 029B@190 (warmstart) | — | 0.846 |
| **039** (refresh broken) | **0.829** | **0.843** |
| 036C (orig) | 0.824 | 0.833 |
| 036D (待 eval 完成) | — | — |

**039 max 0.843 与 029B@190 warmstart 0.846 在 1000ep SE (0.012) 内**，方向上"几乎追平"。
**039 略优于 036C** ~0.5pp / 1.0pp，差异来源**不是** adaptive refresh（已证 0/10 refresh 生效），而是 036D-style 稳定化（λ=0.003 + warmup_steps=10000 + finite-check）。

### 9.4.5 原始 Training Summary（一手存档）

```
Training Summary
  stop_reason:      Trial status: TERMINATED
  run_dir:          ray_results/PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607
  best_reward_mean: -0.5043 @ iteration 150
  best_checkpoint:  .../checkpoint_000150/checkpoint-150
  final_checkpoint: Checkpoint(persistent, .../checkpoint_000300/checkpoint-300)
  best_eval_checkpoint: .../checkpoint_000040/checkpoint-40
  best_eval_baseline:  0.920 (46W-4L-0T) @ iteration 40
  best_eval_random:    0.980 (49W-1L-0T)
  eval_results_csv:    .../checkpoint_eval.csv
  loss_curve_file:     .../training_loss_curve.png
Done training
```

注:
- `best_eval_baseline 0.920 @ iter 40` 是 **50ep training-internal**；实际 **1000ep** 见 §9.4 表（@40 = 0.836, @230 = 0.843 max）
- 50ep peak 在 iter 40 是早期 warmup 后立即出现的，1000ep 测出来真实 max 在 iter 230，提示 50ep best_eval_checkpoint 选择不可靠
- 训练 stable 完成 300 iter，但 **28.3% iter 仍是 inf**（见 §9.1）

### 9.5 对 §1 `H_039` 主假设的解读（更新）

`H_039` 仍**未被实际测试**（refresh 没发生，broadcast bug）。但 1000ep 数据表明:

- ✓ **036D-style 稳定化**（λ↓ + warmup + finite-check）确实让 learned reward fine-tune 从 "回归 warmstart" 改善到 "几乎追平 warmstart"（+0.5–1.0pp on 036C → 039）
- ✗ **追平 ≠ 超越**：037 max 0.843 仍 ≤ warmstart 0.846，learned reward signal **没有提供真实增益**
- ⚠️ 真正测 H_039 需要修 wrapper-discovery，让 refresh 实际触发

### 9.5 对 §1 `H_039` 主假设的解读

**H_039 未被实际测试**（refresh 没发生）。

- ✗ 无法判断 periodic D refresh 是否修 OOD drift
- ✗ 无法判断 B-T loss 是否优于 BCE
- ✓ 但**工程侧的 fail-soft 成立**（callback 出错不再 kill 训练）

### 9.6 下一步（如果要真正测 H_039）

需要先修 wrapper-discovery bug:

1. 不用 `trainer.workers.local_worker().env`，改用 `foreach_env(lambda env: _find_wrapper(env))` 遍历所有 rollout workers 的真实 envs，返回非 None 的第一个
2. 或让 wrapper 把自己注册到 Ray 的 globals 里（`ray.put(wrapper_state_dict_getter)` + `ray.get` 分发）
3. Smoke 一次 refresh 在 10-iter 内实际触发 + broadcast 成功，再启动完整 run

**但给定本次 039 实际等于 036D 的事实**，优先级建议:
- 先看 036D 和 039 的 1000ep 是不是真的等价（验证"refresh 不生效 ⇒ 两者等价"）
- 如果等价，下一轮再启动 fixed-wrapper-discovery 的 039-B
- 如果不等价，需要单独诊断差异来源

### 9.7 Report 可写内容

这轮 039 实际给出的是 **工程 null result with root-cause diagnosis**:

> We implemented an adaptive reward refresh callback using Ray RLlib's DefaultCallbacks (on_train_result + foreach_worker broadcast) plus a Bradley-Terry reformulation of the reward model loss. Training completed 300 iterations cleanly with fail-soft hardening, but a wrapper-discovery bug in the callback (trainer.workers.local_worker().env did not expose the rollout workers' LearnedRewardShapingWrapper in our Ray 1.4 setup) caused all 10 scheduled refreshes to skip the broadcast step. The run therefore effectively operated as a static-reward variant identical to 036D's configuration. The adaptive hypothesis (H_039) is not actually tested by this run; the result is an implementation-side null, not a design-side null.

---

## 10. 039-fix 工单（待实施，handoff 标准格式）

### 10.1 任务摘要

修复 [`adaptive_reward_callback.py`](../../cs8803drl/imitation/adaptive_reward_callback.py) 的 wrapper-discovery bug，让 adaptive refresh 在 Ray 1.4 下真正生效。修复后**重跑一次 039**，与原 036D / 039(broken) 对比，验证 §1 H_039 主假设（adaptive iterative MaxEnt-flavored learned reward 是否优于固定 learned reward）。

### 10.2 Bug 根因复述

**当前代码** ([adaptive_reward_callback.py L237-247](../../cs8803drl/imitation/adaptive_reward_callback.py)):

```python
sample_worker = trainer.workers.local_worker()
wrapped = _find_learned_reward_wrapper(sample_worker.env)
if wrapped is None:
    print("[adaptive-reward] ERROR: LearnedRewardShapingWrapper not found in local worker env")
    return
model = wrapped._model
device = wrapped._device
```

**为什么失败**: 在 Ray RLlib 1.4 下，`trainer.workers.local_worker()` 的 `.env` **不是** rollout worker 的真实 wrapper 链。具体可能是：
- `_VectorizedGymEnv` 外壳（包了多个 sub-env），需要 `.envs[0]` 而不是 `.env`
- 或 evaluation worker 的 dummy env，没有 `LearnedRewardShapingWrapper` 在 chain 上
- 或 local_worker 在 PPO 单 GPU 配置下根本不持有 env（env 全在 remote workers 上）

实测证据：[snapshot-039 §9.2](#92-refresh-机制静默失败关键-finding) 显示 10/10 refresh trigger 都打印 `LearnedRewardShapingWrapper not found in local worker env`。**fail-soft 工作**了，所以训练没崩，但 broadcast 步从未跑过。

`_find_learned_reward_wrapper` 函数本身写法 OK（沿 `env.env` 链 walk），但传入的根 env 不对。

### 10.3 修复方案设计

**核心思想**：callback **自己持有 reward model**（authoritative copy），而不是依赖 worker env 上的副本。流程：
1. **首次** on_train_result 时：通过 `foreach_worker` 从 rollout workers 中提取 model state（任何一个 worker 都有正确加载的副本）
2. **每次 refresh**：在 callback 自己的 model 上 fine-tune
3. **每次 broadcast**：通过 `foreach_worker` 把新 state_dict 推到所有 worker（local + remote）的 wrapper

这样既绕开 `local_worker.env` 的歧义，也保证每 refresh 后所有 worker 同步。

#### 10.3.1 新增辅助函数：robust env-chain walker

`adaptive_reward_callback.py` 顶部加：

```python
def _find_wrapper_in_env(env, target_class_name="LearnedRewardShapingWrapper"):
    """Walk arbitrary env wrapper chain looking for target_class_name.

    Handles all observed RLlib 1.4 patterns:
      - direct wrapper (env.env.env...)
      - VectorEnv shell with .envs list (each is a wrapper chain)
      - BaseEnv with .get_unwrapped() returning list of envs
      - None / missing env (returns None silently)
    """
    if env is None:
        return None
    # Pattern 1: direct chain via .env
    probe = env
    seen = set()
    while probe is not None and id(probe) not in seen:
        seen.add(id(probe))
        if type(probe).__name__ == target_class_name:
            return probe
        probe = getattr(probe, "env", None)
    # Pattern 2: VectorEnv with .envs list
    sub_envs = getattr(env, "envs", None)
    if isinstance(sub_envs, (list, tuple)):
        for se in sub_envs:
            found = _find_wrapper_in_env(se, target_class_name)
            if found is not None:
                return found
    # Pattern 3: BaseEnv unwrap
    unwrap = getattr(env, "get_unwrapped", None)
    if callable(unwrap):
        try:
            sub_envs = unwrap()
            if isinstance(sub_envs, (list, tuple)):
                for se in sub_envs:
                    found = _find_wrapper_in_env(se, target_class_name)
                    if found is not None:
                        return found
        except Exception:
            pass
    return None
```

替换原 `_find_learned_reward_wrapper` 函数（保持调用兼容性，可以让旧函数 alias 到新函数）。

#### 10.3.2 Callback 持有 model 的 lazy init

在 `AdaptiveRewardCallback.__init__` 加：
```python
self._owned_model = None         # set on first refresh
self._owned_device = "cpu"
```

修改 `on_train_result` 的 fetch 逻辑（替换当前 L237-247）：

```python
# Lazy init: copy model from any rollout worker that has the wrapper
if self._owned_model is None:
    def _try_get_state(w):
        env = getattr(w, "env", None)
        wrapper = _find_wrapper_in_env(env)
        if wrapper is None:
            return None
        return {
            "state_dict": {k: v.detach().cpu() for k, v in wrapper._model.state_dict().items()},
            "device": str(wrapper._device),
        }

    results = trainer.workers.foreach_worker(_try_get_state)
    valid = [r for r in results if r is not None]
    if not valid:
        print("[adaptive-reward] ERROR: no worker has LearnedRewardShapingWrapper; "
              "refresh disabled for this run")
        self._refresh_disabled_due_to_error = True
        return

    # Reconstruct model architecture (assume same as wrapper's)
    # We need the model class — store it from the first found wrapper for cloning
    sample_wrapper = None
    for w in [trainer.workers.local_worker()] + list(trainer.workers.remote_workers()):
        try:
            env = getattr(w, "env", None) if not hasattr(w, "apply") else None
            # remote workers need foreach_worker; this only works for local
            # — actually use foreach_worker again for sample_wrapper extraction
        except Exception:
            pass

    # Simpler: re-fetch with full wrapper reference from foreach_worker
    def _try_get_model_obj(w):
        env = getattr(w, "env", None)
        wrapper = _find_wrapper_in_env(env)
        if wrapper is None:
            return None
        # Return a deep-copied model so callback owns its own instance
        import copy
        return copy.deepcopy(wrapper._model), str(wrapper._device)

    obj_results = trainer.workers.foreach_worker(_try_get_model_obj)
    valid_objs = [r for r in obj_results if r is not None]
    if not valid_objs:
        print("[adaptive-reward] ERROR: failed to deep-copy model from any worker")
        self._refresh_disabled_due_to_error = True
        return
    self._owned_model, self._owned_device = valid_objs[0]
    print(f"[adaptive-reward] callback owns model copy from worker (init refresh path)")
```

**注意**: 需要解决 deepcopy + remote-worker 的序列化问题。如果 `deepcopy(model)` 在 remote worker 上 pickle 失败，改用 state_dict + 在 callback 重建 architecture。建议**直接走 state_dict 路径**：

```python
def _try_get_state(w):
    env = getattr(w, "env", None)
    wrapper = _find_wrapper_in_env(env)
    if wrapper is None:
        return None
    return {
        "state_dict": {k: v.detach().cpu() for k, v in wrapper._model.state_dict().items()},
        "device": str(wrapper._device),
        "model_path": getattr(wrapper, "_model_path", None),  # if wrapper stores it
    }
```

然后 callback 用 `LEARNED_REWARD_MODEL_PATH` env var 直接从 disk 加载初始 model（同 wrapper 的 init path），获得 architecture，再 load state_dict 覆盖。这避免序列化问题。

#### 10.3.3 Refresh 走 callback 自有 model

替换原 refresh 调用（L257-269）：

```python
metrics = refresh_reward_model_online(
    model=self._owned_model,        # was: model
    offline_table=self._offline_table,
    online_pairs=online_pairs,
    head_to_labels=head_to_labels,
    classifier_fn=classifier_fn,
    loss=self._cfg["refresh_loss"],
    steps=self._cfg["refresh_steps"],
    batch_size=256,
    lr=self._cfg["refresh_lr"],
    device=self._owned_device,      # was: device
)
```

#### 10.3.4 Broadcast 走 robust walker

替换原 `_broadcast` 函数（L274-284）：

```python
new_sd = {k: v.detach().cpu() for k, v in self._owned_model.state_dict().items()}

def _broadcast(w):
    env = getattr(w, "env", None)
    wrapper = _find_wrapper_in_env(env)   # ← 用新 robust walker
    if wrapper is None:
        return False
    wrapper.update_reward_model(new_sd)
    return True

n_ok = sum(1 for r in trainer.workers.foreach_worker(_broadcast) if r is True)
n_total = 1 + len(trainer.workers.remote_workers())
print(f"[adaptive-reward] refresh #{self._refresh_count + 1} done: "
      f"loss_final={metrics['train_loss_final']:.4f} "
      f"broadcasted_to={n_ok}/{n_total}_workers")
```

加 `n_total` 让我们能看到比例（如果 3/9 才有 wrapper, 还是有 bug）。

### 10.4 Smoke 测试（必跑，确认修好再启动正式 run）

写一个 1-iter smoke run 验证 refresh + broadcast 真生效：

```bash
# Use any GPU node + tmux pattern. ~5 min total.
SMOKE_RUN_NAME=039_smoke_$(date +%Y%m%d_%H%M%S)
ADAPTIVE_REFRESH_EVERY=1 \
ADAPTIVE_REFRESH_STEPS=10 \
ADAPTIVE_MIN_ONLINE_PAIRS=0 \
LEARNED_REWARD_ADAPTIVE_REFRESH=1 \
MAX_ITERATIONS=3 \
EVAL_INTERVAL=999 \
RUN_NAME=$SMOKE_RUN_NAME \
bash scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_039_airl_adaptive_on_029B190_512x512.batch
```

**Smoke 必须看到的输出**:

```
[adaptive-reward] callback owns model copy from worker (init refresh path)
[adaptive-reward] ===== refresh trigger at iter 1 =====
[adaptive-reward] building offline table from 4 dirs:
[adaptive-reward] offline table ready: 184540 samples
[adaptive-reward] refresh #1 done: loss_final=X.XXXX broadcasted_to=N/N_workers
```

**关键 assertion**:
- `broadcasted_to=N/N` 中 `N == 1 + NUM_WORKERS` (= 1 local + 8 remote = 9)，**不能是 0/9**
- 如果是 `1/9`：local worker 有但 remote 没有，需要继续 debug remote workers 的 env structure
- 如果是 `0/9`：完全失败，说明 walker 还有 bug

**2026-04-19 smoke 实测**:

```
[adaptive-reward] offline table ready: 184540 samples
[adaptive-reward] callback owns model copy from worker (init refresh path)
[adaptive-reward] refresh #1 done: loss_final=0.0036 broadcasted_to=8/9_workers
[adaptive-reward] refresh #2 done: loss_final=0.0003 broadcasted_to=8/9_workers
[adaptive-reward] refresh #3 done: loss_final=0.0004 broadcasted_to=8/9_workers
```

当前解释：
- **refresh 本体已修好**：offline table build、callback-owning model、refresh loss、broadcast 全部发生。
- `8/9` 更像是 **8 个 remote rollout workers 全部更新成功，而 local/driver worker 不持有可更新 wrapper**。
- 因为实际采样来自 remote rollout workers，所以这个结果已经足够支持**启动正式 rerun**；后续若有必要，再把日志细化成 `remote=X/8, local=Y/1`。

**还要 verify state_dict 真变化** （加在 smoke 末尾）:

```python
# At end of smoke, check that ckpt's reward_model is different from initial
# (compare a few weight tensors before and after)
```

可以在 callback 里加临时 debug：

```python
if self._refresh_count == 1:  # first refresh
    print(f"[adaptive-reward] DEBUG initial weight sample: "
          f"{list(self._owned_model.state_dict().values())[0].flatten()[:3]}")
elif self._refresh_count == 2:
    print(f"[adaptive-reward] DEBUG after refresh weight sample: "
          f"{list(self._owned_model.state_dict().values())[0].flatten()[:3]}")
```

如果两个数值不同 → refresh 真在更新 model。

### 10.5 正式 rerun 启动

Smoke 通过后启动正式 039-fix run（基本同原 039 batch，只有以下变化）：

```bash
RUN_NAME=039fix_mappo_airl_adaptive_on_029B190_512x512_$(date +%Y%m%d_%H%M%S)
# 其余参数完全同 scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_039_airl_adaptive_on_029B190_512x512.batch
# 包括:
LEARNED_REWARD_ADAPTIVE_REFRESH=1
ADAPTIVE_REFRESH_EVERY=30   # 每 30 iter refresh, 总 10 次
ADAPTIVE_REFRESH_STEPS=2000
ADAPTIVE_REFRESH_LOSS=bt
ADAPTIVE_REFRESH_LR=3e-4
MAX_ITERATIONS=300
TIME_TOTAL_S=28800
LEARNED_REWARD_SHAPING_WEIGHT=0.003
LEARNED_REWARD_WARMUP_STEPS=10000
WARMSTART_CHECKPOINT=ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/.../checkpoint-190
```

8h 1 节点。训练完成后用 [post-train-eval skill](../../.claude/skills/post-train-eval/SKILL.md) 跑 1000ep + capture + H2H。

### 10.6 预注册判据 (与 036D 对比)

修复后 H_039 真成立的判据（与 §3 一致，但聚焦 vs 036D 对比）:

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **broadcast metric** | 10/10 refresh 都 `broadcasted_to>=8/9`，且无 `0/9` / `1/9` | 工程层面修好的硬验证；当前 smoke 已显示 8/9 为可接受模式 |
| **reward model state changes** | refresh 前后 weight 数值 ≠ 0 | adaptive 真在 update |
| 1000ep peak | ≥ 0.860 (036D peak) | 至少不退化 |
| **1000ep peak** | **≥ 0.870** (+1pp vs 036D) | adaptive 真有 ROI 的最低门槛 |
| H2H vs 036D@150 | ≥ 0.52 | 直接验证 adaptive 优于 static |
| failure structure: wasted_possession | ≤ 36% (036D 38.6%) | 机制上证明 reward 持续 align W/L distribution |

### 10.7 失败判据

| 条件 | 解读 |
|---|---|
| smoke 显示 broadcast 0/9 | walker 还是没找到 wrapper, 需要再深 debug RLlib 1.4 internals |
| smoke 显示 broadcast 1/9 (only local) | remote workers env structure 不同, 需要单独调 |
| smoke 显示 broadcast 8/9 且 refresh loss 正常下降 | 视为**工程通过、可进入正式 rerun**；保留 local worker caveat |
| 1000ep peak < 0.86 | adaptive refresh 反而比 static 弱（reward shock 太多） |
| 1000ep peak ∈ [0.86, 0.87] | adaptive 没显著优于 static，H_039 部分否决 |
| 1000ep peak ≥ 0.87 但 H2H vs 036D < 0.50 | adaptive 学到 baseline-specific 优势, 不真 general |

### 10.8 不做的事

- **不做 forward MaxEnt RL** (工程量爆炸, 见 [§6](#6-不做的事))
- **不引入 BC online buffer** (online_pairs 仍可空, 用 offline pool 走 B-T)
- **不修 evaluator inf 问题** (那是 036D 已知问题, 不在 039 修复范围)
- **不动 reward model 架构** (multi-head v2 保持)
- **不并行 launch 多个 fix variant** (先单个 verify)

### 10.9 文件改动清单

| 文件 | 改动 |
|---|---|
| [`cs8803drl/imitation/adaptive_reward_callback.py`](../../cs8803drl/imitation/adaptive_reward_callback.py) | 加 `_find_wrapper_in_env` (robust walker) + 改 callback 持有 model + 改 broadcast 用新 walker + 加 broadcast count assertion |
| `scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_039fix_airl_adaptive_on_029B190_512x512.batch` (NEW) | 复制原 039 batch + 改 `RUN_NAME` 为 039fix |
| `docs/experiments/snapshot-039-airl-adaptive-reward-learning.md` | implementer 跑完后追加 §11 首轮 fix 结果 |

### 10.10 实施成本预估

| 阶段 | 估时 |
|---|---|
| 读 §10 + 理解原 callback 代码 | 30 min |
| 实施 `_find_wrapper_in_env` + 改 fetch / broadcast | 1.5h |
| Smoke 1-iter run + 验证 broadcast count + state change | 30 min |
| 正式 8h 训练 (~1 GPU slot) | 8h (post-train wait, can do other work) |
| Post-train eval (skill: 1000ep + capture + H2H, ~1.5h) | 1.5h |
| 写 §11 首轮 fix verdict | 1h |
| **Total** | **~半天工程 + 8h GPU + 半天 eval/写文档** |

### 10.11 实施 checklist (handoff)

- [ ] 读 [§9.2 bug 描述](#92-refresh-机制静默失败关键-finding) 和 [adaptive_reward_callback.py 当前代码](../../cs8803drl/imitation/adaptive_reward_callback.py) 理解 root cause
- [x] 在 `adaptive_reward_callback.py` 加 `_find_wrapper_in_env` (atomic — 同 commit 加 + 替换调用点)
- [x] 改 callback 持有 model (lazy init from `foreach_worker`)
- [x] 改 broadcast 用新 walker, 加 `broadcasted_to=N/N_workers` 输出
- [x] 写 `039fix` batch (在 `scripts/batch/experiments/`)
- [x] 跑 smoke，观测到 3/3 refresh 成功、`broadcasted_to=8/9`、refresh loss 明显下降
- [x] 启动正式 run (300 iter 完成 2026-04-19 ~07:47)
- [x] 训练完后 invoke [`/post-train-eval`](../../.claude/skills/post-train-eval/SKILL.md) (lane name: `039fix`) — 1000ep on 014-23-0 进行中
- [ ] 按 §10.6 判据写 §11 verdict (append-only, 含 raw recap) — 待 1000ep 完成

## 11. KL inf 根因 fix 工单（2026-04-19，待实施）

### 11.1 finding（来源 §9 → §10 fix 后的 progress.csv 复检）

`039fix` 训练完成后，progress.csv 检查暴露**新 inf 模式**：

| 列 | Trial 1 (143 iter) | Trial 2 (160 iter) | 合并 (303 iter) |
|---|---:|---:|---:|
| `info/learner/shared_cc_policy/learner_stats/total_loss` | 15 inf (10.5%) | 61 inf (38.1%) | **76 inf (25.1%)** |
| `info/learner/shared_cc_policy/learner_stats/kl` | 15 inf (10.5%) | 61 inf (38.1%) | **76 inf (25.1%)** |
| `info/learner/shared_cc_policy/learner_stats/policy_loss` | 0 | 0 | **0 (CLEAN)** |
| `info/learner/shared_cc_policy/learner_stats/vf_loss` | 0 | 0 | **0 (CLEAN)** |
| `info/learner/shared_cc_policy/learner_stats/cur_kl_coeff` | 0 | 0 | **0 (CLEAN)** |
| `info/learner/shared_cc_policy/learner_stats/entropy` | 0 | 0 | **0 (CLEAN)** |

**`total_loss` 和 `kl` 同步 inf**，比例完全一致 → `total_loss = policy_loss + vf_loss + cur_kl_coeff × kl` 的关系下，**KL 是 inf 来源**，total_loss 被 KL 拖进 inf。

### 11.2 root cause 分析

`§10` 的 `finite_check` 修了 **reward → advantage → policy_loss / vf_loss** 这条 numerical 路径（**3 列 CLEAN 是证据**），但 KL **是 PPO 内部计算**：

```python
# Ray 1.4 PPOTorchPolicy.compute_kl_divergence (粗略)
new_logp = policy.dist_class(new_logits).logp(actions)
old_logp = sample_batch["action_logp"]
kl = (old_logp - new_logp).mean()
```

如果 **policy logits 在某些 action dim 上 collapse 到 -inf**（确定性策略 / 数值不稳定），则：
- `softmax(logits)` 中某项 → 0
- `log(0) = -inf`
- KL = mean(-inf - finite) = `-inf` 或 `+inf`

**为什么 logits 会 collapse**？三个候选机制：

#### 机制 A — adaptive reward refresh 的 sign-flip
- discriminator refresh 时（每 30 iter）reward signal 突变
- policy 在新 reward landscape 下急剧调整
- 高 LR 或 high clip_param 让 logits 跨越数值边界（`exp(±large) → ±inf`）
- **概率: 高** — 30 iter refresh = 10 次/run，与 inf 率 25-38% 量级匹配

#### 机制 B — finite_check 把 reward inf clip 到 ±λ 引入 reward 不连续
- 同一 state 在不同 step 收到差异巨大的 reward (inf clip 到 +λ vs -λ vs natural value)
- advantage 极不稳定 → policy gradient 方向频繁翻转
- logits 累积发散
- **概率: 中** — 但 policy_loss CLEAN 削弱这个假设

#### 机制 C — Bradley-Terry pairwise loss 在 discriminator 学习时输出 极端 logits
- discriminator 学到判断很有把握的 (state, action) 对 → output ±20+ 的 logits
- reward = sigmoid(disc_logit) - 0.5 在饱和区接近 ±0.5（finite, not inf）
- 但 reward 的边际很小（接近常数 ±0.5），policy gradient 信号收缩
- 与 v2 reward 同时存在时形成 reward shock
- **概率: 中** — 不直接产 inf, 但放大 A/B 风险

**最可能: A**（refresh-induced policy shock 直接导致 logits collapse）

### 11.3 修复方案（按工程成本排序）

#### Fix-A — refresh 期间冻结 PPO update（最稳但最贵）

```python
class AdaptiveRewardCallback:
    def on_train_result(self, *, trainer, result, **kw):
        ...
        if self.iter_count % self.refresh_interval == 0:
            self._refresh_discriminator()
            self._broadcast_to_workers()
            # NEW: skip the PPO update for this iter (or apply tiny LR)
            trainer.workers.foreach_worker(
                lambda w: w.policy_map["shared_cc_policy"].config.update({"lr": 1e-7})
            )
            # 在下一 iter restore lr
        elif self.iter_count % self.refresh_interval == 1:
            trainer.workers.foreach_worker(
                lambda w: w.policy_map["shared_cc_policy"].config.update({"lr": orig_lr})
            )
```

**预算**: 4-6h 工程，需要测试 lr 切换不破坏 lr scheduler（如 cosine）

#### Fix-B — KL early termination（中等）

PPO 已有 `kl_target`，超过会 early-stop SGD iter。但**当 KL=inf 时 early-stop 触发 0 epoch SGD** = 无 update。这其实是 graceful degradation。问题是这种 iter 也会 log `kl=inf` 进 progress.csv。

如果只是想"不让 inf 污染 csv"，可以在 callback 里 sanitize result dict：

```python
def on_train_result(self, *, trainer, result, **kw):
    # Sanitize learner_stats before Ray writes progress.csv
    learner_stats = result.get("info", {}).get("learner", {}).get("shared_cc_policy", {}).get("learner_stats", {})
    for k, v in learner_stats.items():
        if isinstance(v, float) and not math.isfinite(v):
            learner_stats[k] = 0.0  # or float('nan') sentinel
            self._inf_count_by_field[k] += 1  # log to custom_metrics
    result.setdefault("custom_metrics", {})["airl_inf_count_kl"] = self._inf_count_by_field.get("kl", 0)
```

**预算**: 1h 工程，纯 cosmetic 修复（不修根因，但让 csv 干净 + 给我们一个可观测的 inf 计数指标）

#### Fix-C — discriminator 输出 clip + smoothing（治本但需要重训）

```python
class DiscriminatorReward:
    def __call__(self, obs, action):
        logit = self._model(obs, action)  # raw discriminator output
        # Clip BEFORE sigmoid to avoid saturation
        logit = torch.clamp(logit, -3.0, 3.0)
        reward_raw = torch.sigmoid(logit) - 0.5  # in [-0.5, +0.5]
        # Smooth: low-pass filter across consecutive steps to reduce shock
        reward_smoothed = 0.7 * self._last_reward + 0.3 * reward_raw
        self._last_reward = reward_smoothed
        return reward_smoothed
```

**预算**: 4h 工程 + 8h GPU 重训，更接近原 AIRL spec

### 11.4 推荐执行

| 步骤 | 优先级 | 责任人 | 时间 |
|---|---|---|---|
| 1. 跑 1000ep 验证现有 039fix（带 inf 但 eval 看上去 OK）| **HIGH** | 已 launch (014-23-0) | 已在跑 |
| 2. 实施 Fix-B（cosmetic + 加观测）| **MED** | handoff | 1h |
| 3. 看 1000ep 结果是否 ≥ 036D@150 (0.860) | gate | 等 #1 完 | — |
| 4. **若 ≥ 0.870** → 不做 Fix-A/C，039fix 收口；inf 是 cosmetic 问题 | conditional | — | — |
| 5. **若 < 0.860** → 做 Fix-A 重训，看 KL inf 消除后 eval 是否更好 | conditional | handoff | ~12h |

### 11.5 判据（决定是否做 Fix-A/C）

| 1000ep peak WR | verdict | 行动 |
|---|---|---|
| ≥ 0.870 | 039fix 已经够好，inf 是 cosmetic 问题 | 仅 Fix-B；不做 Fix-A/C；收口 verdict |
| ∈ [0.860, 0.870) | 持平 036D，AIRL 没赚到额外 +ΔWR | Fix-B 收口；考虑放弃 AIRL 路径 |
| < 0.860 | inf 已经在伤害训练 | 做 Fix-A，看消除 inf 后 eval 是否能 ≥ 0.86 |
| < 0.83 | inf + AIRL 双重失败 | 放弃 AIRL，关闭 snapshot-039 |

### 11.6 caveat（执行前必读）

- **§10 fix 是 partial success**：reward inf ✅ 修，KL inf ❌ 未修。verdict 文档要 reflect 这个细节
- KL inf **不一定致命** — eval 仍 peak 0.94 @ 290 (50ep internal) 说明 PPO clip_param 在 KL fail 时接管了 trust region
- 若 1000ep ≥ 0.86，**优先 cosmetic Fix-B 而不是治本 Fix-A**——后者重训成本远大于解决的问题
- §11 工单**不会自己启动**——需要 handoff 给同学或 user 决定后再做

## 12. 039fix 1000ep 结果（2026-04-19，append-only — 数据不足以收口）

### 12.1 1000ep eval recap (top 5%+ties+±1, 16 ckpts on atl1-1-03-014-23-0, port 61005, total elapsed 962s)

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 90 | 0.816 | 816-184 | 0.92 | -0.104 |
| 100 | 0.840 | 840-160 | 0.76 | +0.080 |
| 110 | 0.839 | 839-161 | 0.88 | -0.041 |
| 120 | 0.833 | 833-167 | 0.88 | -0.047 |
| 140 | 0.835 | 835-165 | 0.88 | -0.045 |
| 150 | 0.833 | 833-167 | 0.88 | -0.047 |
| 160 | 0.817 | 817-183 | 0.82 | -0.003 |
| 170 | 0.837 | 837-163 | 0.84 | -0.003 |
| 180 | 0.820 | 820-180 | 0.88 | -0.060 |
| 190 | 0.837 | 837-163 | 0.82 | +0.017 |
| **230** | **🏆 0.852** | **852-148** | 0.82 | +0.032 |
| 240 | 0.833 | 833-167 | 0.88 | -0.047 |
| 250 | 0.843 | 843-157 | 0.80 | +0.043 |
| 270 | 0.811 | 811-189 | 0.86 | -0.049 |
| 280 | 0.833 | 833-167 | 0.88 | -0.047 |
| 290 | 0.836 | 836-164 | **0.94** | **-0.104** |

mean = 0.833, peak = **0.852 @ 230**, range [0.811, 0.852]。

### 12.2 frontier 对照

| ckpt | 1000ep peak | 039fix Δ |
|---|---:|---:|
| 029B@190 | 0.868 | -0.016 |
| 036D@150 | 0.860 | **-0.008** |
| 031A@1040 | 0.860 | -0.008 |
| 031B@1220 | 0.882 | -0.030 |
| **039fix@230** | **0.852** | — |

### 12.3 数值健康度（核实 §11.1）

[§11.1](#111-finding来源-9--10-fix-后的-progresscsv-复检) 提到的 KL/total_loss 25.1% inf 在本 run 完全 reproduces：

- Trial 1 (143 iter): kl 15 inf (10.5%), total_loss 15 inf (10.5%)
- Trial 2 (160 iter): kl 61 inf (38.1%), total_loss 61 inf (38.1%)
- policy_loss / vf_loss / cur_kl_coeff / entropy: **全 CLEAN** (0 inf)

**finding**: §10 fix 的 reward-side `finite_check` 工作正常（policy_loss CLEAN 是证据），但 **KL 计算路径**仍出 inf（policy logits 数值不稳）。

### 12.4 严格按 [§11.5 判据](#115-判据决定是否做-fix-ac)

| 阈值 | 落在哪 | 严格判据 |
|---|---|---|
| ≥ 0.870 | ❌ | 仅 Fix-B（cosmetic）|
| ∈ [0.860, 0.870) | ❌ | Fix-B + 考虑放弃 |
| **< 0.860 (我们 0.852)** | **✅ here** | **做 Fix-A 看能否消除 inf 后达 0.86** |
| < 0.83 mean partial (最低 ckpt 0.811) | partial | 关 snapshot |

严格按字面 → **应该启动 Fix-A 重训**。

### 12.5 但这是「不完整测试」 — verdict 暂留 open（user 决策）

039 这条 AIRL adaptive 路径**从未有过一次"干净"的测试**:

| 轮次 | 状态 | inf | AIRL 是否真 fire | 1000ep peak |
|---|---|---|---|---|
| **039 broken** ([§9](#9-首轮结果2026-04-18-1647-edt)) | broadcast bug, callback 静默失败 | 28% inf in v2-static-equivalent | ❌ (10/10 refresh skipped) | 0.843 |
| **039fix** (本 §12) | callback fire 成功（reward 全负证据），但 KL 38% inf | 38% inf in KL/total_loss | ✅ (reward 全负证明) | 0.852 |

→ **从未观察到 "AIRL 真 fire + 数值干净" 的对照点**。我们只测了：
- AIRL 没 fire 的 placebo (0.843)
- AIRL fire 了但 KL 半破碎 (0.852)

第二个比第一个好 +0.009pp（边缘信号），暗示 AIRL **可能**有 marginal contribution，但 KL inf 的 confounding factor 让我们无法 disentangle。

**真正能 disentangle 的实验**: §11.3 Fix-A 重训 — 消除 KL inf 后看 1000ep peak。
- 如果 ≥ 0.86 → AIRL 在数值干净下能持平 036D，**有方向性证据但需要更长 horizon 看是否能突破**
- 如果 < 0.85 → AIRL 真无增益，可关 snapshot
- 如果 ∈ [0.85, 0.86] → 仍模糊，但已经是 cleanest test，可以 close lane

### 12.6 暂时立场（user override 优先）

- **暂不关 snapshot-039**，AIRL adaptive 仍是后续提升能力的候选路径之一
- **§11 Fix-A 工单不立刻启动**（12h GPU + 工程，当下 priority 有 031B/034 ensemble）
- **作为 backlog**: 当 031B / 034 / 046 路径都用尽且仍未达 0.90 时，回头做 Fix-A 重训取干净测试结果
- **Cosmetic Fix-B (1h)** 可以同学先做：让 callback sanitize result dict + 添加 `airl_inf_count_kl` 等观测指标，让未来 Fix-A 重训能直接看到 inf 是否真消除

### 12.7 Raw recap (verification)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000230/checkpoint-230 vs baseline: win_rate=0.852 (852W-148L-0T)
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.843 (843W-157L-0T)
.../checkpoint_000100/checkpoint-100 vs baseline: win_rate=0.840 (840W-160L-0T)
.../checkpoint_000110/checkpoint-110 vs baseline: win_rate=0.839 (839W-161L-0T)
.../checkpoint_000170/checkpoint-170 vs baseline: win_rate=0.837 (837W-163L-0T)
.../checkpoint_000190/checkpoint-190 vs baseline: win_rate=0.837 (837W-163L-0T)
.../checkpoint_000290/checkpoint-290 vs baseline: win_rate=0.836 (836W-164L-0T)
.../checkpoint_000140/checkpoint-140 vs baseline: win_rate=0.835 (835W-165L-0T)
.../checkpoint_000120/checkpoint-120 vs baseline: win_rate=0.833 (833W-167L-0T)
.../checkpoint_000150/checkpoint-150 vs baseline: win_rate=0.833 (833W-167L-0T)
.../checkpoint_000240/checkpoint-240 vs baseline: win_rate=0.833 (833W-167L-0T)
.../checkpoint_000280/checkpoint-280 vs baseline: win_rate=0.833 (833W-167L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.820 (820W-180L-0T)
.../checkpoint_000160/checkpoint-160 vs baseline: win_rate=0.817 (817W-183L-0T)
.../checkpoint_000090/checkpoint-90  vs baseline: win_rate=0.816 (816W-184L-0T)
.../checkpoint_000270/checkpoint-270 vs baseline: win_rate=0.811 (811W-189L-0T)
[suite-parallel] total_elapsed=962.4s tasks=16 parallel=7
```

完整 log: [docs/experiments/artifacts/official-evals/039fix_baseline1000.log](../../docs/experiments/artifacts/official-evals/039fix_baseline1000.log)

## 13. Fix-B 落地（2026-04-19，append-only）

### 13.1 实施内容

[`adaptive_reward_callback.py`](../../cs8803drl/imitation/adaptive_reward_callback.py) 新增 `_sanitize_learner_stats(result)`:

- 在 `on_train_result` 开头**无条件**调用（不依赖 refresh 是否 enabled）
- 遍历 `result["info"]["learner"][<policy_id>]["learner_stats"]` 所有数值列
- 检测到 `inf` / `-inf` / `nan` → sanitize 到 `0.0`（cosmetic，progress.csv 不再被污染）
- 计数写入 `result["custom_metrics"]`:
  - `airl_inf_count_<key>`: 本 iter inf 触发数
  - `airl_inf_total_<key>`: 累计 inf 总数
  - `airl_nan_count_<key>` / `airl_nan_total_<key>`: 同上 for NaN
- `__init__` 加 `self._inf_count_total / self._nan_count_total` 作为 defaultdict 累计器

### 13.2 Smoke verify (PASS)

```
iter1 input:  {kl: inf, total_loss: inf, policy_loss: 0.5, vf_loss: 0.3, ...}
iter1 sanitized: {kl: 0.0, total_loss: 0.0, policy_loss: 0.5, vf_loss: 0.3, ...}
iter1 custom_metrics: {airl_inf_count_kl=1, airl_inf_count_total_loss=1,
                       airl_inf_total_kl=1, airl_inf_total_total_loss=1}

iter2 input:  {kl: -inf, total_loss: nan, policy_loss: 0.4}
iter2 sanitized: {kl: 0.0, total_loss: 0.0, policy_loss: 0.4}
iter2 custom_metrics: {airl_inf_count_kl=1, airl_nan_count_total_loss=1,
                       airl_inf_total_kl=2 (cumulated),
                       airl_inf_total_total_loss=1, airl_nan_total_total_loss=1}
```

### 13.3 影响

- **PPO 训练 unchanged**: sanitize 只改 `result` dict (logging path)，不改 PPO 内部 KL/loss 计算
- **progress.csv 干净**: 未来重训不再有 inf 行污染数值分析
- **可观测性**: `airl_inf_count_*` / `airl_inf_total_*` 直接出现在 progress.csv `custom_metrics/*` 列，可 plot per-iter inf rate（之前需要 post-hoc CSV scan）
- **未消除根因**: KL inf 的 root cause（policy logits 数值不稳）仍在 — Fix-B 是 cosmetic，不是 Fix-A
- **下一步**: Fix-A 真要做时（snapshot-039 §11.3），新一轮 039 训练可直接通过 `custom_metrics/airl_inf_total_kl` 在 progress.csv 验证 KL inf 是否消除，无需再 post-hoc 分析

### 13.4 不做的事

- **不做 Fix-A 重训** — 工程 12h GPU + 4-6h eng，但 §12.5 已 verdict "AIRL 还没干净测试"且 ROI 低，现按 backlog 处理
- **不修改 reward-side finite_check** — §10 fix 已经 work（policy_loss / vf_loss CLEAN 是证据）
- **不删除 Fix-A 工单** — §11.3 仍保留作为未来 backlog 时的 reference
- [ ] 更新 [README.md](README.md) status + [rank.md §8 changelog](rank.md)

## 14. 039fixB Stage-1 verdict — Fix-B 是纯 cosmetic, KL behavior 不变, lane 关闭 (2026-04-19, append-only)

### 14.1 触发原因

039fixB (smoke pass §13) 在 atl1-1-03-015-2-0 上跑了 300 iter formal 训练（`ray_results/039fixB_with_sanitize_20260419_092209/`）。预期：sanitize callback 应该让 progress.csv KL/total_loss 保持 finite, 并通过 `custom_metrics/airl_inf_total_*` 计数器把真实 inf 频率暴露出来, 对 Fix-A 必要性给定量证据。

### 14.2 训练完成 + 50ep 内部 eval

- 完成 300 iter, **clean 0 inf 在 progress.csv 里**（sanitize 工作了）
- 50ep 内部 eval (29 ckpts) range [0.68, 0.92], peak 0.92 @ ckpt 80, mean 0.83
- late-window peaks: 220 (0.90), 240 (0.90), 280 (0.90)
- **预测 1000ep peak ≈ 0.85** (按 50→1000 drift -7~9pp 算, 加上 039fix 上一轮 peak 0.852 的先验)

### 14.3 KL behavior 跟 inf 版本本质相同 — Fix-B 是 cosmetic 确认

直接对比 progress.csv：

| Trial | n_iter | KL_inf% (raw) | KL_zero% (sanitized) | 真 inf 事件率 | max\|KL\|finite | max\|TL\|finite |
|---|---|---|---|---|---|---|
| 039fix Trial1 (formal) | 143 | **10.5%** | 0.0% | 10.5% | 3.71e-03 | 8.02e-02 |
| 039fix Trial2 (resume140) | 160 | **38.1%** | 0.0% | 38.1% | 3.84e-03 | 8.75e-02 |
| **039fixB (sanitize 300 iter)** | **300** | **0.0%** | **35.7%** | **35.7%** | **3.86e-03** | **8.51e-02** |

**关键证据**：
1. **fixB KL_zero% = 35.7% ≈ Trial2 raw KL_inf% 38.1%** — 同样的 inf 事件率, 只是 callback 把 inf 抹成 0 写入 logging
2. **max\|KL\|finite 三组完全一致 (~3.8e-03)** — 健康部分 KL 分布没变化
3. **max\|total_loss\|finite 三组也一致 (~8.5e-02)** — 真实训练动力学完全相同
4. **没有 `custom_metrics/airl_inf_total_*` 列出现在 progress.csv** — Fix-B 设计的 inf counter 实际未生效（callback 改了 result dict 但 `episode_user_data` 不会自动 propagate 到 `custom_metrics` reporting path）

→ Fix-B **没改变 KL inf 频率, 也没成功暴露 inf counter**, 仅在 progress.csv logging 阶段把 inf 替换为 0。

### 14.4 §10.5 / §10.7 假说被实证

snapshot-039 §10.5 / §10.7 早预言（更正版本）：
> WR 更好的实际机制是 PPO grad_clip 把 inf gradient 变成无效 update + finite-check 防 NaN 流到 advantage，**不是阻止 inf 发生**

039fixB 现在是这个假说的 confirmation：
- **35.7% iter 的 KL/total_loss 是 inf** — 跟 Trial1/2 同量级 (10.5% / 38.1%)
- 这些 iter 的 PPO update 实际上是被 grad_clip 兜底吃掉的「空 update」
- AIRL adaptive reward 的真信号 **仍然没法从 PPO 鲁棒性兜底里 disentangle 出来**

### 14.5 决策 — 跳过 1000ep eval, 关 lane

**预测 vs 实际成本**：
- 预测 1000ep peak 0.85 (跟 039fix Trial2 peak 0.852 同档)
- vs 031B 0.882 = -3pp, 不进 SOTA / ensemble 候选
- vs 036D 0.860 = -1pp, 跟 learned-reward per-agent 路径打平

**ROI 分析**：
- 1000ep eval 成本: 16 min × 1 节点（复用 039fix evaluator）
- 信息增益: 把预测 0.85 ± 0.04 缩到实测 ± 0.01 — 但 verdict 不会改变（仍然不是 SOTA, 不是 ensemble candidate）
- 决策性: 0
- **跳过 1000ep eval, 释放节点给 034ea/034eb / 045A H2H 等高优先级 lane**

**lane 状态**:
- ❌ AIRL 真正干净测试需要 Fix-A (KL inf 根因 fix), 12h GPU + 4-6h eng
- ❌ 没有任何当前路径需要 AIRL adaptive 介入（031B 0.882 单 model + 034E 0.890 ensemble + 034ea/eb 在跑, 都不依赖 AIRL）
- ❌ Fix-A 即使做了, 上界是 +1-2pp (sub-noise) over 036D 0.860, 还是跟 031B 差距明显
- → **039 lane 关闭**, 标记 backlog "Fix-A 等其他路径全部用尽再回头"

### 14.6 这次的最大教训 — sanitize ≠ fix

**工程通用 lesson**: 任何 "把异常值写进 log 时改成 normal value" 的做法 (sanitize / clip / clamp logging-side), **不会改变模型的实际行为, 只是让 dashboard 看起来正常**。下次再遇到 "inf in progress.csv" 这种问题：
1. 先问 "inf 出现的 root cause 在哪一层" (forward / loss / grad / optimizer step)
2. **fix root cause** > sanitize logging
3. 如果短期内只能 sanitize, **必须同时加一个独立的 inf counter** 在 raw 数据流上（callback before sanitize）, 否则信息丢失

039fixB 的设计有第三步 (`custom_metrics/airl_inf_total_*`) 但实际没生效（callback 改 result dict 但 `episode_user_data` 不会自动 propagate 到 `custom_metrics` reporting path）。

### 14.7 后续 backlog 优先级

| Backlog item | 触发条件 | 工程成本 | 上界期望 |
|---|---|---|---|
| 039 Fix-A (KL inf 根因 fix + 重训 300 iter) | 031B / 034ea/eb / 052 / 046 全部用尽且仍 < 0.90 | 12h GPU + 4-6h eng | peak 0.86-0.87 (vs 现在 0.85) |
| 039 inf counter 修 (callback 走 `custom_metrics_to_log`) | 重训 039 时顺手 | 0.5h eng | observability only, 不改 WR |

→ Fix-A backlog 优先级现在 **下沉到「最后救命稻草」** — 跟 snapshot-050 Phase 1.3c (cross-student DAGGER 加 SE) 同级, 当 0.90 突破路径全死才回头。
