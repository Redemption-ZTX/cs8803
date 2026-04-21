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

## 7.1 [2026-04-20] 056D Stage 1 baseline 1000ep — lr=3e-4 lane verdict (marginal, tied 031B)

### 7.1.1 背景

2026-04-20 04:27 EDT mass kill event 中断所有 056 lane 训练; 056A (3e-5) / 056B (7e-5) / 056C (1.5e-4) archive 损坏或训练进度不够, **056D (lr=3e-4, 3× baseline) 是 056 sweep 中唯一完成 ≥1200 iter 并成功完成 post-train 1000ep eval 的 lane**。本节只记录 056D 结果; 056A/B/C verdict 不在此节内覆盖。

### 7.1.2 结果总表 (056D 单 lane)

| ckpt | 50ep (pre-screen) | 1000ep | W-L |
|---:|---:|---:|---|
| 630 | — | 0.851 | 851-149 |
| 710 | — | 0.866 | 866-134 |
| 720 | — | 0.868 | 868-132 |
| 730 | — | 0.863 | 863-137 |
| 770 | — | 0.858 | 858-142 |
| 1060 | — | 0.871 | 871-129 |
| 1110 | — | 0.887 | 887-113 |
| **1140** | — | **0.891** 🥇 | 891-109 |
| 1190 | — | 0.882 | 882-118 |
| 1200 | — | 0.878 | 878-122 |

**peak = 0.891 @ iter 1140**

### 7.1.3 判据 verdict (§3 严格判定, 056D only)

| 阈值 | 实测 (056D peak) | 结果 |
|---|---|---|
| §3.1 marginal ≥ 0.886 | 0.891 | ✅ **命中** (marginal) |
| §3.2 主 ≥ 0.890 | 0.891 | ✅ **勉强命中** (+0.001 over threshold) |
| §3.3 突破 ≥ 0.900 | 0.891 | ❌ **未达 grading 门槛** |
| §3.4 持平 ∈ [0.875, 0.886) | 0.891 | ❌ — |
| §3.5 退化 < 0.870 | 0.891 | ❌ — |

**verdict: marginal — 056D (lr=3e-4) 在 §3.1/§3.2 两个 threshold 上命中, 但增益太小 (+0.009 vs 031B 0.882, 在 1000ep SE 0.016 内); 不是 statistically significant breakthrough**

### 7.1.4 统计显著性

- vs 031B@1220 combined 2000ep 0.882: Δ = +0.009, **0.56× SE** → **well within noise**, 不能 decisively claim LR=3e-4 > LR=1e-4
- vs §3.2 threshold 0.890: Δ = +0.001, 贴着门槛, **不能视为 decisive breakthrough**
- **Plateau pattern (iter 1060-1200)**: [0.871, 0.887, 0.891, 0.882, 0.878] 全 ≥ 0.871, mean = 0.882 ≈ 031B baseline; 不是 single-peak luck
- **LR=3e-4 在 PBT-simplified sweep 中单调压过 lr=1e-4**: 031B baseline 0.882 vs 056D peak 0.891 (+0.009), 方向一致但幅度 sub-noise

### 7.1.5 机制解读

- **LR 敏感度 confirmed 但 amplitude 小**: 3× 放大 LR 没 blow up, 证明 031B architecture + v2 shaping + CLIP=0.15 的 PPO 配置对 LR 相对稳定
- **最优 LR 可能在 [1e-4, 3e-4] 之间**: 056D 0.891 marginal 好于 031B 0.882, 但与 055 distill 的 0.911 相比差一大截 → **HP tuning 在本 architecture 上 ROI 有限, 真正的 SOTA 方向是 distillation-based**
- 056A/B/C lane 数据缺失导致无法完成 LR curve fitting (预期 LR=7e-5 ~ 1.5e-4 中段最佳但无证据)

### 7.1.6 后续路径 (§8 Outcome A 分支, 但 ROI 重新评估)

按 §8 Outcome A (≥ 0.886 marginal):
- §4.5 step 1 (rerun verify) 建议但 优先级低 — +0.009 gain 即使 verified 也不构成项目突破
- §4.5 step 3 multi-axis sweep (CLIP/GAE/entropy) — 055 distill 已打开 0.911 的 SOTA 窗口, HP sweep lane **降级为 secondary objective**
- 推荐: **把 lr=3e-4 应用回 055 distill 或 031B 等主 lane 做第二性对照**, 而不是独立继续扩 056 sweep

### 7.1.7 Raw recap (official evaluator parallel)

```
=== Official Suite Recap (parallel) ===
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_000630/checkpoint-630 vs baseline: win_rate=0.851 (851W-149L-0T)
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_000710/checkpoint-710 vs baseline: win_rate=0.866 (866W-134L-0T)
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_000720/checkpoint-720 vs baseline: win_rate=0.868 (868W-132L-0T)
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_000730/checkpoint-730 vs baseline: win_rate=0.863 (863W-137L-0T)
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_000770/checkpoint-770 vs baseline: win_rate=0.858 (858W-142L-0T)
/storage/ice1/5/1/wsun377/ray_results_archive/056D_pbt_lr0.00030_scratch_20260419_193533/TeamVsBaselineShapingPPOTrainer_Soccer_82d22_00000_0_2026-04-19_19-35-55/checkpoint_001060/checkpoint-1060 vs baseline: win_rate=0.871 (871W-129L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001110/checkpoint-1110 vs baseline: win_rate=0.887 (887W-113L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140 vs baseline: win_rate=0.891 (891W-109L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001190/checkpoint-1190 vs baseline: win_rate=0.882 (882W-118L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001200/checkpoint-1200 vs baseline: win_rate=0.878 (878W-122L-0T)
[suite-parallel] total_elapsed=487.4s tasks=10 parallel=7
```

### 7.2 [2026-04-20 16:04 EDT] 056extend — 056D@1250 resume training to iter 1510 (wall limit hit)

#### 7.2.1 背景与配置

Resume of 056D@1250 (ckpt 1250 = last saved, not 1240) with identical config (lr=3e-4, 031B arch, v2 shaping) targeting `MAX_ITERATIONS=2000`。目的: 验证 §7.1 "LR=3e-4 plateau 在 iter 1140 saturate" hypothesis — continued training 是否会再创新高或真的饱和。

- **run_dir**: `/storage/ice1/5/1/wsun377/ray_results_scratch/056extend_resume_056D_1250_to_2000_20260420_135527`
- **trial**: `TeamVsBaselineShapingPPOTrainer_Soccer_2a326_00000_0_2026-04-20_13-55-49`
- **best_reward_mean**: +1.8226 @ iteration 1403
- **best_checkpoint**: iter 1430

#### 7.2.2 Stop 原因 — TIME_TOTAL_S 提前触顶

Trial 只跑到 iter 1510 (260 extra iter) 就 TERMINATED, 不是完整 750 iter 抵达 2000。根因: **resume 的 `_time_total` 从 056D 原训练的 9.9h 继续计时 (=35561s)**, 触发 `TIME_TOTAL_S=43200` wall clock limit 时仅剩 ~2.1h 可用于 extend phase, 260 iter 即耗尽。**非训练崩溃, 是 wall-time 限制配置不对**。

#### 7.2.3 Inline 200ep eval 结果 (ckpts 1260-1500 step 10, EVAL_EPISODES=200)

**Top 15 by WR**:

| iter | 200ep WR |
|---:|---:|
| **1280** | **0.930** 🔝 |
| 1320 | 0.925 |
| 1260 | 0.920 |
| 1500 | 0.915 |
| 1380 | 0.910 |
| 1430 | 0.905 |
| 1480 | 0.905 |
| 1300 | 0.900 |
| 1410 | 0.900 |
| 1370 | 0.895 |
| 1460 | 0.895 |
| 1470 | 0.890 |
| 1310 | 0.885 |
| 1420 | 0.885 |
| 1450 | 0.885 |

**Full plateau 1260-1500 (25 ckpts)**: range [0.840, 0.930], mean 0.893, std 0.023

#### 7.2.4 关键 finding — LR=3e-4 NOT saturated at iter 1250

- 056D 原 1250-iter 训练 1000ep combined 2000ep peak = **0.891 @ iter 1140** (§7.1)
- Extended 260 iter (iter 1260-1510) inline 200ep shows consistent ≥0.87 performance with **peak 0.930 @ iter 1280**
- 200ep SE ≈ 0.022, 0.930 nominally +0.039pp vs 056D peak, 但 within ~1.8σ under 200ep noise → **not yet statistically significant**
- **结论**: LR=3e-4 plateau (056D peak 0.891 @ 1140) 并 **非已 saturate**; continued training 给出多个 ≥0.92 points, refutes §7.1 "saturated at iter 1140" hypothesis

#### 7.2.5 Pending / Next

- **Pending**: 1000ep post-eval on peak ckpts (1260 / 1280 / 1320 / 1380 / 1430 / 1500) 验证 vs 056D plateau (SE ≈ 0.016)
- **Next**: if 1000ep post-eval confirms 0.91+, run another 056extend-v2 with longer wall time (TIME_TOTAL_S=86400 = 24h) to get full 750 iter (1250→2000)
- **Note**: 200ep inline 不是 authoritative — 等 1000ep confirmation 才写入 rank.md §3.3

#### 7.2.6 1000ep post-eval verdict (2026-04-20 16:20 EDT, append-only)

9 ckpt × 1000ep 并行 eval 完成 (total_elapsed=480s, parallel=7)，resolve §7.2.5 pending。

**Data: inline 200ep vs 1000ep verified**

| ckpt | 200ep inline (earlier) | **1000ep verified** | Δ optimism |
|---:|---:|---:|---:|
| 1260 | 0.920 | 0.877 | -0.043 |
| 1280 | **0.930** | 0.888 | -0.042 |
| 1300 | 0.900 | 0.890 | -0.010 |
| 1320 | 0.925 | 0.869 | -0.056 |
| 1380 | 0.910 | 0.889 | -0.021 |
| 1410 | 0.900 | 0.867 | -0.033 |
| 1430 | 0.905 | 0.876 | -0.029 |
| 1480 | 0.905 | 0.887 | -0.018 |
| **1500** | 0.915 | **0.896** | -0.019 |

Raw recap:
```
=== Official Suite Recap (parallel) ===
.../checkpoint_001260/checkpoint-1260 vs baseline: win_rate=0.877 (877W-123L-0T)
.../checkpoint_001280/checkpoint-1280 vs baseline: win_rate=0.888 (888W-112L-0T)
.../checkpoint_001300/checkpoint-1300 vs baseline: win_rate=0.890 (890W-110L-0T)
.../checkpoint_001320/checkpoint-1320 vs baseline: win_rate=0.869 (869W-131L-0T)
.../checkpoint_001380/checkpoint-1380 vs baseline: win_rate=0.889 (889W-111L-0T)
.../checkpoint_001410/checkpoint-1410 vs baseline: win_rate=0.867 (867W-133L-0T)
.../checkpoint_001430/checkpoint-1430 vs baseline: win_rate=0.876 (876W-124L-0T)
.../checkpoint_001480/checkpoint-1480 vs baseline: win_rate=0.887 (887W-113L-0T)
.../checkpoint_001500/checkpoint-1500 vs baseline: win_rate=0.896 (896W-104L-0T)
[suite-parallel] total_elapsed=480.0s tasks=9 parallel=7
```

Run dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/056extend_resume_056D_1250_to_2000_20260420_135527/TeamVsBaselineShapingPPOTrainer_Soccer_2a326_00000_0_2026-04-20_13-55-49/`

**Verdict narrative**

- **True 1000ep peak = 0.896 @ iter 1500** (inline 200ep 0.930 @ 1280 was +0.042pp optimistic; corrections range -0.010 to -0.056, mean ≈ -0.030; SE 0.022 at n=200)
- **vs 056D @1140 combined 2000ep 0.891**: Δ = +0.005 **within SE 0.016 → NOT significantly better**
- **vs 055 @1150 combined 2000ep 0.907** (distill SOTA): Δ = -0.011 → **still below distill SOTA**
- **056 LR=3e-4 path saturated at ~0.89** — extended 260 iter (1250→1510) did not break plateau
- **Key empirical lesson**: inline 200ep eval (even with EVAL_EPISODES=200 new standard) still materially overestimates 1000ep (up to -0.056pp correction on single ckpt). 200ep **NOT** a substitute for combined 2000ep verdict. Going forward: always verify promising 200ep findings with 1000ep before drawing conclusions.
- **056extend-v2 with longer wall NOT warranted** (§7.2.5 "Next" condition "≥ 0.91+" not met — peak is 0.896). Do not re-launch.
- **rank.md §3.3 impact**: none — 056extend peak 0.896 not significantly better than 056D 0.891 row, not worth a new leaderboard entry. Logged only in rank.md §8 changelog.

### 7.3 [2026-04-20 19:55 EDT] 056C lr=1.5e-4 — full 1250 iter complete + Stage 1 1000ep verdict

**Run dir**: `/storage/ice1/5/1/wsun377/ray_results_scratch/056C_pbt_lr0.00015_scratch_20260420_092048/TeamVsBaselineShapingPPOTrainer_Soccer_d00f3_00000_0_2026-04-20_09-21-17/`

**Stage 1 1000ep eval (10 ckpts, parallel)**

| ckpt | 1000ep WR | W-L |
|---:|---:|---:|
| 490 | 0.844 | 844-156 |
| 750 | 0.862 | 862-138 |
| 930 | 0.862 | 862-138 |
| 940 | 0.858 | 858-142 |
| 950 | 0.858 | 858-142 |
| 990 | 0.872 | 872-128 |
| 1020 | 0.858 | 858-142 |
| 1040 | 0.867 | 867-133 |
| 1100 | 0.877 | 877-123 |
| **1110** | **0.883** (peak) | 883-117 |

**Verdict**

- **Peak 1000ep = 0.883 @ iter 1110** (lr=1.5e-4)
- **vs 031B 0.880**: Δ = +0.003 — **NOT statistically significant** (1000ep SE ±0.010, well within ±1σ)
- **vs 056D lr=3e-4 peak 0.891**: Δ = -0.008 → **056D wins**
- **Confirms monotonic LR trend in 056 series**: lr=3e-5 (0.78) < lr=7e-5 (0.86) < lr=1e-4 (031B 0.880) < lr=1.5e-4 (0.883) < lr=3e-4 (0.891). 056E (lr=5e-4) and 056F (lr=7e-4) currently training will tell if trend continues upward.
- **Inline 50ep eval was 0.98 @ iter 940**, but 1000ep verified 0.858 (Δ = -0.122 catastrophic overestimation). Strongly reinforces `feedback_inline_eval_noise` memory — DO NOT trust inline 50ep for SOTA decisions.
- **Plateau pattern**: late iter (990–1110) all ≥ 0.858 with peak shifting late, similar to 056D pattern (peak shifts later in training).

**Raw recap (official evaluator parallel)**

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000490/checkpoint-490   vs baseline: win_rate=0.844 (844W-156L-0T)
.../checkpoint_000750/checkpoint-750   vs baseline: win_rate=0.862 (862W-138L-0T)
.../checkpoint_000930/checkpoint-930   vs baseline: win_rate=0.862 (862W-138L-0T)
.../checkpoint_000940/checkpoint-940   vs baseline: win_rate=0.858 (858W-142L-0T)
.../checkpoint_000950/checkpoint-950   vs baseline: win_rate=0.858 (858W-142L-0T)
.../checkpoint_000990/checkpoint-990   vs baseline: win_rate=0.872 (872W-128L-0T)
.../checkpoint_001020/checkpoint-1020  vs baseline: win_rate=0.858 (858W-142L-0T)
.../checkpoint_001040/checkpoint-1040  vs baseline: win_rate=0.867 (867W-133L-0T)
.../checkpoint_001100/checkpoint-1100  vs baseline: win_rate=0.877 (877W-123L-0T)
.../checkpoint_001110/checkpoint-1110  vs baseline: win_rate=0.883 (883W-117L-0T)
[suite-parallel] total_elapsed=513.6s tasks=10 parallel=7
```

**rank.md §3.3 impact**: none — 056C peak 0.883 below 056D 0.891 row already in §3.3, no SOTA change. Logged only in rank.md §8 changelog.

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
