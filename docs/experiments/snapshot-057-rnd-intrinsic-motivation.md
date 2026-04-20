## SNAPSHOT-057: RND (Random Network Distillation) — Intrinsic Motivation 探索奖励 (Tier A3)

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); pending free node + S1/S2 顺序

## 0. 背景与定位

### 0.1 当前 reward axis 已尝试

| 路径 | 代表 lane | 1000ep peak | 评注 |
|---|---|---|---|
| Sparse only (no shaping) | 031B-noshape (in-flight) | TBD | ablation, 测 v2 价值 |
| v2 dense shaping | 031B@1220 | 0.882 | current SOTA single-model scratch |
| v2 + outcome PBRS | 053A (just done) | TBD | combo path |
| Learned reward (sparse + neural reward) | 045A, 051A/B/D | 0.867~0.888 | combo with shaping |
| AIRL adversarial | 039 | 0.836 | weak |

**没试过**: **intrinsic motivation 探索奖励** — RND / ICM / curiosity。

### 0.2 为什么 RND 在 Soccer-Twos 上理论上有价值

Soccer-Twos 的核心难点:
- **球场覆盖不均**: agent 容易陷入 "守门员" 模式 (ball_x = -10 to 0, 自己半场), 不去尝试进攻新区域
- **稀疏 goal reward**: 1 episode 平均 1-2 进球, 大部分 step 没 reward signal
- **探索瓶颈**: PPO + entropy=0.0 在 trained agent 上 entropy 收敛到 ~0.5, 行为同质化

RND 提供 dense exploration signal:
- novel state → 高 intrinsic reward → policy 更主动探索 unfamiliar regions
- 已访问 state → 低 intrinsic → 不再 distract from extrinsic
- 自动 anneal: 不需要手动 schedule

## 1. 核心假设

### H_057

> 加 RND intrinsic reward (β=0.01) on top of 031B + v2 shaping, **1000ep peak ≥ 0.886** (+0.4pp vs 031B 0.882)。

### 子假设

- **H_057-a**: 031B 当前 entropy ~0.5 处于"中等保守"区, 加 intrinsic 让 entropy 维持高一点 (~0.6), explore 更多新 state
- **H_057-b**: novel state coverage 提升 → 训练 distribution 多样化 → 测试时面对 baseline 的 novel 行为更鲁棒
- **H_057-c**: RND 比 ICM (Inverse Curiosity Module) 更稳定, 因为 RND 不需要 inverse dynamics model 的训练 (Burda 2019 实证)

## 2. 设计

### 2.1 RND 架构

```
RND Module:
  target_network: obs (672) → MLP(256, 256) → embed (64)   [FROZEN, random init]
  predictor_network: obs (672) → MLP(256, 256) → embed (64)  [TRAINABLE]

Per env step (in rollout worker):
  embed_target = target_network(obs).detach()
  embed_pred   = predictor_network(obs)
  intrinsic_reward = ||embed_target - embed_pred||² / 64   # MSE per dim
  total_reward = env_reward + β * intrinsic_reward

Per training update:
  rnd_loss = MSE(embed_pred, embed_target.detach())
  total_loss = ppo_loss + λ_rnd * rnd_loss   # λ_rnd = 0.5
```

### 2.2 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Target net 架构 | 2-hidden MLP (256, 256) | 跟 student encoder 一致, 容量匹配 |
| Predictor 架构 | 同 target | symmetry (Burda 2019 标准) |
| Intrinsic 缩放 β | 0.01 | 初步保守; v2 shaping 平均 reward 量级 ~0.01, β 等量级 |
| RND loss weight λ | 0.5 | 减半防 RND 训练干扰 PPO |
| 是否 normalize intrinsic | 是 (running stat) | Burda 2019 强调; 防 reward scale 漂移 |
| 是否 normalize obs for RND | 是 (running mean/std) | Burda 2019 强调 |
| RND 生命周期 | 全程 (1250 iter), 不衰减 β | 保持探索压力, 让 PPO clip 自己平衡 |

### 2.3 工程实现

新增 module: `cs8803drl/branches/rnd_module.py`

```python
class RNDModule(nn.Module):
    def __init__(self, obs_dim=672, hidden=256, embed_dim=64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False
        # Running normalization for obs + intrinsic reward
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("intrinsic_running_std", torch.ones(1))

    def compute_intrinsic(self, obs):
        # Normalize obs
        obs_n = ((obs - self.obs_mean) / (self.obs_std + 1e-6)).clamp(-5, 5)
        with torch.no_grad():
            t = self.target(obs_n)
        p = self.predictor(obs_n)
        intrinsic = ((t - p) ** 2).mean(dim=-1)  # (B,)
        # Normalize by running std
        intrinsic_n = intrinsic / (self.intrinsic_running_std + 1e-6)
        return intrinsic_n, ((t.detach() - p) ** 2).mean()  # second is rnd_loss
```

Hook in env wrapper or training callback:
- Either modify `RewardShapingWrapper` to add intrinsic per step
- Or add RND-aware callback that injects intrinsic into batch before PPO update

Cleanest approach: callback. RND module on learner GPU, batch-process all obs after rollout.

### 2.4 训练超参

```bash
# 同 031B
TEAM_SIAMESE_ENCODER=1 TEAM_CROSS_ATTENTION=1 ...
USE_REWARD_SHAPING=1 SHAPING_BALL_PROGRESS=0.01 ...
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4 ...

# RND 新增
RND_ENABLED=1
RND_HIDDEN_DIM=256
RND_EMBED_DIM=64
RND_INTRINSIC_BETA=0.01
RND_LOSS_WEIGHT=0.5
RND_OBS_NORMALIZE=1
RND_INTRINSIC_NORMALIZE=1
```

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.886 | +0.4pp vs 031B | RND 有真增益 |
| §3.2 主: peak ≥ 0.890 | +0.8pp | exploration bonus 显著有效 |
| §3.3 突破: peak ≥ 0.900 | grading 门槛 | declare success |
| §3.4 持平: peak ∈ [0.875, 0.886) | RND noise | curiosity bonus 不重要 |
| §3.5 退化: peak < 0.870 | RND 干扰 | β 太大或 RND 训练干扰 PPO |

## 4. 简化点 + 风险 + 降级 + 预案

### 4.1 简化 A3.A — 单 β 值

| 简化项 | 完整 | 当前 |
|---|---|---|
| β tuning | 4-β grid (0.001, 0.005, 0.01, 0.05) | 单 β=0.01 |

**风险**: β 选错; β=0.01 可能太大/太小。
**降级预期**: -0.5pp vs sweep。
**预案**: peak < 0.880 → 立即开 4-β sweep on free nodes。

### 4.2 简化 A3.B — 不衰减 β

| 简化项 | 完整 | 当前 |
|---|---|---|
| β schedule | β decay over training (Burda 2019 fixed; some 后续 papers anneal) | 全程 fixed β=0.01 |

**风险**: 后期 RND signal 应该 ≈0 (predictor 已 converge), 不衰减可能引入 noise。
**降级预期**: -0.3pp。
**预案**: 加 cosine decay β=0.01 → 0.001 over training。

### 4.3 简化 A3.C — Target net 不重新随机化

| 简化项 | 完整 | 当前 |
|---|---|---|
| Target re-randomization | RIDE (Raileanu 2020) 周期性 reset target | 全程 fixed target |

**风险**: predictor 一旦 converge, intrinsic 完全消失 → 后期等同于 vanilla PPO。
**降级预期**: 中后期 RND 失效, 等价 031B。
**预案**: 加 target reset 每 200 iter (复杂)。

### 4.4 简化 A3.D — 用 raw obs (无 frame stack)

Burda 2019 在 Atari 上用 4-frame stack。Soccer-Twos 已经在 obs 里 encode 了 velocity, 所以不 stack。
**风险**: temporal info 不足, RND 区分 state 能力下降。
**预案**: 后续可加 obs[t-1]; obs[t-2] concat 作为 RND input。

### 4.5 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | β=0.01 fixed | base 12h GPU |
| 1 | peak < 0.880 | 4-β sweep | +12h × 4 |
| 2 | sweep 全失败 | 加 cosine β decay | +12h |
| 3 | step 2 失败 | RIDE-style target reset | +12h |
| 4 | step 3 失败 | 加 frame stack | +12h |
| 5 | step 4 失败 | declare RND 路径 dead, 转 ICM 或其他 | — |

## 5. 不做的事

- 不在 implementation 完成 + smoke pass 之前 launch
- 不混入架构改动 (用 031B-arch)
- 不与其他 lane 抢节点 (新 PORT_SEED)
- 不 sweep RND 多 hyperparam (仅 β=0.01 first pass)
- 不和 Curriculum (snapshot-058) 同时开 (避免分散精力)

## 6. 执行清单

- [ ] 1. 实现 `cs8803drl/branches/rnd_module.py` (~3h)
- [ ] 2. 添加 callback or wrapper hook 把 intrinsic 注入 reward (~3h)
- [ ] 3. Smoke test (RND init, forward, intrinsic compute, normalize)
- [ ] 4. 写 launch script
- [ ] 5. 找 free node, launch
- [ ] 6. Verdict 后 follow §4.5

## 7. Verdict

_Pending_

## 8. 后续路径

### Outcome A — 突破 (peak ≥ 0.886)
- β sweep 找最优
- 与 PBT (056) 联合: best-LR + RND combo

### Outcome B — 持平
- 走 §4.5 step 1-4 降级序列
- 如全失败, RND 路径关闭, 试 **ICM** 或 **NGU (Never Give Up)**

### Outcome C — 退化
- β=0.01 太大破坏 PPO
- 试更小 β (0.001), 若仍退化, **intrinsic motivation 路径整体关闭**

## 9. 相关

### 理论支撑
- **Burda et al. 2019** "Exploration by Random Network Distillation" (ICLR) — RND 原始 paper, Atari hard exploration tasks
- **Pathak et al. 2017** "Curiosity-Driven Exploration by Self-Supervised Prediction" — ICM 对照
- **Raileanu & Rocktäschel 2020** "RIDE: Rewarding Impact-Driven Exploration" — target reset 改进

### 代码
- [team_siamese.py](../../cs8803drl/branches/team_siamese.py) — 待集成的 student arch
- 需新增: `cs8803drl/branches/rnd_module.py`, RND-aware callback
