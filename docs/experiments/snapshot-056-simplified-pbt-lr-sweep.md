## SNAPSHOT-056: Simplified PBT — 4-lane LR sweep on 031B-arch (Tier S2 MVP)

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); 等 4 batch scripts + 节点 ready 后并行 launch

## 0. 背景与定位

### 0.1 PBT (Population-Based Training) 全貌

Jaderberg et al. 2017 PBT 的核心组件：
1. **N parallel agents**, each with different HP
2. **Periodic exploit**: bad-performing agents copy good agents' weights + HP
3. **Periodic explore**: copied agents mutate HP (perturb ±20% or random resample)
4. **Continuous training**: no restart between exploit/explore

在 RLlib v1.4 上, Ray Tune 提供 `PopulationBasedTraining` scheduler 实现这一切, 但需要:
- 自定义 `perturbation_interval` (典型 5-10 个 trial.report)
- 配置 `hyperparam_mutations` dict
- `quantile_fraction` 控制 exploit
- `resample_probability` 控制 explore
- Ray Tune trial pause/resume, 涉及 `Trainable.save()` / `restore()` 兼容性

**当前 RLlib v1.4.0 + Ray Tune PBT 的已知问题**:
- pause/resume on Soccer-Twos: Unity worker 重启可能 hang
- pause 时 saved state 与 hyperparam 不一致
- PBT 与自定义 `TeamVsBaselineShapingPPOTrainer` 集成需大量适配

完整 PBT 工程量: 1-2 周 (调通 pause/resume + 多 trial port 调度 + perturbation 工程)。

### 0.2 本 snapshot 的 MVP 简化

**只保留 PBT 的"populate"维度**, 砍掉 exploit/explore/mutation:
- 4 个 parallel scratch run, 各自不同 HP
- 训完后选 best, **不在 trial 间交换 weights**
- 等同于"systematic LR grid sweep"，不是真 PBT

为什么称作 "PBT-MVP" 而不是 "grid sweep": 选 4 个 HP 是为了在 PPO 的关键 HP 维度 (LR) 上 cover ±2× 范围, 这种 systematic 探索是 PBT 的第一步也是最便宜的一步。如果 grid 中有 lane 显著好, 我们获得了 "新 SOTA HP 方向" 的信号; 然后再决定是否投资真 PBT 工程。

## 1. 核心假设

### H_056

> 031B 的 LR=1e-4 是 ad-hoc 选择, **未经 systematic sweep**。在 ±2× 范围内可能存在更好的点。**至少 1 lane 1000ep peak ≥ 0.886** (= 031B 0.882 + 0.4pp marginal SE)。

### 子假设

- **H_056-a**: PPO LR 对 plateau location 敏感 (literature: Schulman 2017 PPO paper 的 LR sweep 在 Atari 上跨 4× 范围影响显著)
- **H_056-b**: 031B 的 1e-4 可能 over-tuned 到 028A→031A→031B 渐进过程 (handed-down inheritance), 没有针对 final 031B 配置 re-tune
- **H_056-c**: 4 个 LR 中至少 1 个偏离 1e-4 ≥ 50% 的会有显著差异 (positive 或 negative)

## 2. 设计

### 2.1 Grid 选择

**4 个 LR, 其他 HP 全部 = 031B**:

| Lane | LR | 倍数 vs 031B | 假设 |
|---|---|---|---|
| **056A** | 3e-5 | 0.3× | very low — 收敛慢但稳定, 可能 underfit |
| **056B** | 7e-5 | 0.7× | low — 接近 031B 但更保守, 可能更稳 |
| **056C** | 1.5e-4 | 1.5× | moderate high — 接近 031B 但更激进 |
| **056D** | 3e-4 | 3× | high — 可能 unstable, 但若收敛会快 |

**控制点**: 031B 自己 LR=1e-4 是已知 0.882, 不再 re-run (避免浪费 node)。

**为什么不 sweep CLIP / GAE_λ / entropy 等其他 HP**: 一次 sweep 1 个轴最 clean。如果 LR sweep 全失败, 转 CLIP sweep (snapshot-056B 续)。

### 2.2 训练超参

```bash
# 共同 (跟 031B 一字不变)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048
GAMMA=0.99 GAE_LAMBDA=0.95 ENTROPY_COEFF=0.0

# v2 shaping (031B 同款)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (跟 031B 一字不变, 12h scratch)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10

# Per-lane 独立
# 056A: LR=3e-5
# 056B: LR=7e-5
# 056C: LR=1.5e-4
# 056D: LR=3e-4
```

### 2.3 端口 / 节点协调

4 个 lane 必须并行, 端口/节点不重叠:

| Lane | PORT_SEED | BASE_PORT | EVAL_BASE_PORT |
|---|---|---|---|
| 056A | 41 | 57555 | 55905 |
| 056B | 42 | 57605 | 55955 |
| 056C | 43 | 57655 | 56005 |
| 056D | 44 | 57705 | 56055 |

(避开 054=31, 053A=23, 051D=51, 031B-noshape=13)

需要**4 个 free node**, 各 ≥ 12h remaining。

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: 至少 1 lane 1000ep peak ≥ 0.886 | +0.4pp vs 031B (>SE) | LR sweep 找到 better point, 转 fine sweep |
| §3.2 主: 至少 1 lane peak ≥ 0.890 | +0.8pp | LR 是 031B 提升的主因 (而非架构) |
| §3.3 突破: 至少 1 lane peak ≥ 0.900 | grading 门槛 | declare project success |
| §3.4 持平: 4 lane all peak ∈ [0.875, 0.886) | sub-marginal | LR 不敏感, 031B 1e-4 已经接近最优 |
| §3.5 退化: 4 lane all peak < 0.870 | LR 全偏离 | 031B 1e-4 是 narrow optimum, 偏离即崩 |

## 4. 简化点 + 风险 + 降级预期 + 预案 (**用户要求 mandatory**)

### 4.1 简化 S2.A — 无 mutation / exploit

| 简化项 | 完整 PBT | 当前 MVP | 节省工程 |
|---|---|---|---|
| Bad trial 复制 good trial weights | 每 5 iter exploit (quantile 25%) | 不复制, 4 lane 独立训练 | ~3 天 |
| HP perturbation | mutate ±20% | 不 mutate | ~2 天 |
| Bad trial early kill | quantile pruning | 不 kill, 全跑 1250 iter | ~1 天 |

**风险**:
- 即使 LR=3e-4 早期发散, 仍占用 12h GPU
- 即使 LR=7e-5 在 iter 200 就显著领先, 其他 lane 不会"复制"它的 weights / HP
- ceiling = 4 个固定 LR 中最好的, 无法发现 "LR=8e-5 + 中段降到 5e-5" 这种动态最优

**降级预期**: 真 PBT 文献上比 grid sweep 增益 +1~3pp。本 MVP 顶多达到 grid sweep ceiling, 预期比真 PBT 低 1~2pp。

**预案 (3 层)**:
- L1 (轻度): 训练中段 (iter 400) 检查所有 lane 50ep WR, 如果某 lane 显著最差 (<-3pp from best), 手动 kill 它, 用空节点开新 LR
- L2 (中度): 在每 lane 内部添加 LR cosine decay (LR start → LR/4 over 1000 iter), 模拟 mutation 的"HP 渐进 decay"
- L3 (重度): 真 PBT 工程开始 (Ray Tune integration), 是项目级别决定, 不在本 snapshot scope

### 4.2 简化 S2.B — 单轴 LR sweep (不联合 CLIP/GAE_λ/entropy)

| 简化项 | 完整 grid | 当前 MVP |
|---|---|---|
| HP 维度 | 4D grid (LR × CLIP × GAE × entropy) = 4^4 = 256 cells | 1D LR × 4 levels = 4 cells |

**风险**:
- LR 不是 PPO 唯一关键 HP, CLIP_PARAM (trust region) 同样 critical
- 真 sweet spot 可能在 (LR=2e-4, CLIP=0.1) 联合点上, 不在 LR axis 上单独可见

**降级预期**: 单轴 sweep 错过联合最优, 预期 -0.3~-0.8pp vs 4D grid。

**预案**:
- L1: 如果 LR sweep 发现 best lane, 在 best lane 周围开 CLIP sweep 4 lane (新 snapshot 056-CLIP)
- L2: 如果 best lane 与 031B 接近 (Δ < 0.5pp), 跳过 CLIP, 直接转 entropy / GAE sweep

### 4.3 简化 S2.C — 4 lane 而非 8/16

| 简化项 | 完整 PBT | 当前 MVP |
|---|---|---|
| Population size | 8-16 | 4 |

**风险**:
- 4 个 LR sample 在 [3e-5, 3e-4] 范围内 spacing 较大 (log scale 0.3, 0.7, 1.5, 3.0)
- 真 best 可能在 LR=4e-5 或 LR=1e-4 (我们没测) — undersampling

**降级预期**: -0.3pp vs 8-LR sweep。

**预案**:
- L1: 如果 best lane 在 grid 边缘 (056A 或 056D), 立即在外缘开 4 个新 LR (e.g., 1e-5, 5e-4, ...)
- L2: 如果 best lane 在中段 (056B/C), 在它周围开 fine-grid 4 lane (e.g., LR ∈ {6e-5, 8e-5, 1e-4, 1.2e-4})

### 4.4 简化 S2.D — 同一 seed (无 stochastic exploration variance)

| 简化项 | 完整 | 当前 MVP |
|---|---|---|
| Seed 配置 | 不同 seed × 同 HP × multiple repeats | 同 seed × 4 不同 HP × 1 run |

**风险**:
- PPO 训练对 seed 敏感 (Henderson 2018), 4 lane 中的 win/loss 可能受 seed luck 主导, 而非 HP
- 单 lane 0.886 vs 0.882 的差可能 ≥ seed variance 区间

**降级预期**: ±0.5pp seed-induced variance, 可能让 verdict 变 inconclusive。

**预案**:
- L1: 如果 best lane 与 031B 差距 < 0.5pp, 不宣称 "LR sweep 找到更好点"
- L2: 任何 lane peak ≥ 0.886 必须 rerun 1 次 (不同 seed) 验证
- L3: 多 seed × 多 HP 是真 PBT 才能 afford 的, MVP 接受 seed variance

### 4.5 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | 4 LR × 1250 iter | base 12h GPU × 4 |
| 1 | Best lane ≥ 0.886 但 single seed | rerun best lane 不同 seed | +12h GPU |
| 2 | Step 1 验证后, best 在 grid 边缘 | extend grid 外缘 4 lane | +12h × 4 |
| 3 | Step 2 后 best ≥ 0.890 | sweep CLIP/GAE/entropy 4 lane | +12h × 4 |
| 4 | Step 3 后 best ≥ 0.895 | 投资真 PBT 工程 | +1-2 周 |
| 5 | 任一 step 全部退化 | declare LR sweep 路径 dead | — |

## 5. 不做的事

- 不 launch 真 Ray Tune PBT (工程 1-2 周, 不在本 MVP scope)
- 不 mutate / exploit 任何 lane
- 不修改训练脚本 (只改 env vars + new launch script)
- 不混入架构 / reward 改动
- 不与 054 / 055 / 053A / 051D / 031B-noshape 抢节点

## 6. 执行清单

- [ ] 1. 写 4 个 launch scripts: `scripts/eval/_launch_056{A,B,C,D}_lr_sweep_scratch.sh` (~1h)
- [ ] 2. 找 4 个 free node (≥ 12h 各)
- [ ] 3. Pre-launch health check 各节点
- [ ] 4. 并行 launch 4 lane
- [ ] 5. 实时 monitor: 每 200 iter check 50ep WR trajectory
- [ ] 6. 训完 invoke `/post-train-eval` lane name `056A/B/C/D` × 4
- [ ] 7. Stage 2 capture peak ckpts (最多 2 个)
- [ ] 8. Stage 3 H2H: only best lane, vs 031B@1220 (sanity)
- [ ] 9. Verdict append §7 (含 4 lane 比较表)
- [ ] 10. 根据 verdict 决定走 §4.5 step 1 / 2 / 3 / 5

## 7. Verdict (待 4 lane 训练完成后 append)

_Pending_

## 8. 后续发展线 (基于 verdict 路径图)

### Outcome A — 至少 1 lane 突破 (peak ≥ 0.886)
- 走 §4.5 step 1 rerun verify, 排除 seed luck
- 若 verify 通过, step 2 extend grid 或 step 3 multi-axis sweep

### Outcome B — 全部持平 (4 lane 都 [0.875, 0.886))
- LR 在 ±2× 范围内不敏感, **031B 1e-4 是 narrow flat optimum**
- 转 CLIP / GAE / entropy 单轴 sweep (新 snapshot 056-X)

### Outcome C — 全部退化 (4 lane 都 < 0.870)
- LR 偏离 1e-4 即崩, **031B 1e-4 是 narrow sharp optimum**
- 这本身是有价值的科学结论 (031B HP 已经是 PPO sweet spot)
- LR sweep 路径关闭, 转其他 HP 轴 (CLIP / GAE)

### Outcome D — 4 lane variance 大但没结论 (Δ ≤ 0.5pp 都在 SE 内)
- §4.4 步骤 (rerun multi-seed) 是必要的, 单 seed 实验不可信
- 决定: 是否投资多 seed × multi HP 的 mini-PBT

## 9. 相关

- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B baseline, control
- [team_siamese.py](../../cs8803drl/branches/team_siamese.py) — 031B model
- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) — 训练入口

### 理论支撑

- **Jaderberg et al. 2017** "Population Based Training of Neural Networks" — PBT 完整方法, NeurIPS 2017
- **Henderson et al. 2018** "Deep Reinforcement Learning that Matters" — DRL seed sensitivity 文献
- **Schulman et al. 2017** "Proximal Policy Optimization Algorithms" — PPO 原始 LR sweep 在 Atari 上 4× 范围影响显著

### 与"完整 PBT"的区别 (单独节, framing 用)

| 维度 | 完整 PBT | 本 MVP 056 |
|---|---|---|
| Population size | 8-16 | 4 |
| HP perturbation | ±20% mutate every N iter | 无 mutate, 固定 HP |
| Exploit (bad copy good) | quantile 25% copy quantile 75% | 无 |
| Early kill bad trials | yes | 无 |
| HP space | typically 5-D (LR/clip/entropy/GAE/value-clip) | 1D (LR only) |
| 工程量 | 1-2 周 (Ray Tune integration + pause/resume) | 1-2h (4 batch + monitor) |
| 预期 ceiling vs 当前 SOTA | +1-3pp | +0.5-1pp |
