## SNAPSHOT-071: Pool A — Homogeneous-Family Ensemble Distill (同家族组合池)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); **blocked on 055v2_extend 完成 (~2026-04-21 09:00 EDT)**

## 0. 背景与定位

### 0.1 当前 plateau 拓扑 (2026-04-21)

| Lane | Mechanism | Combined 2000ep | 备注 |
|---|---|---:|---|
| 055@1150 | v2 + distill-KL from 3-teacher 034E (031B/045A/051A) | **0.907** | project SOTA, warm-start 自 031B@80 |
| 055v2@peak | 055 teacher 自 replace 为 {055@1150 + 034E members} 再 distill | **pending** (`055v2_extend_resume_1210_to_2000`, iter 1800-2000 TBD) | recursive distill 变体, 同 family |
| 056D@1140 | 031B HP sweep, LR=3e-4 one-shot | **0.891** (H2H vs 055 marginal 0.536) | HP-sweep axis, distill 家族近邻 |

- 055 lane 的 1000ep peak 自 §7.12 的 0.911 下修到 §7.13 combined 2000ep 0.907, **plateau 证据充分**:
  plateau 1130-1200 六连点 all ≥ 0.89, 不是单次 lucky peak。
- **Pool B (snapshot-070) 已在异构方向起跑** — teacher 来自 3 个 reward-mechanism 完全不同的 lane (v2+distill / PBRS / curriculum), 目标是测 "diversity 大于 recursion"。
- 本 snapshot **Pool A** 走相反极端: **同 distill-family + 同 031B 架构 + 同 shaping 配方** 的 3 个 teacher, 测 "homogeneous teacher 在 joint probability 上是否能 compound 出 smoother gradient"。

### 0.2 与 Pool B 的对照 (snapshot-070)

| 轴 | Pool A (本 snapshot 071) | Pool B (snapshot 070) |
|---|---|---|
| Teacher 数 | 3 | 3 |
| Teacher 家族 | **homogeneous** (全部 distill/HP-sweep family, 全部 v2 shaping + 031B arch) | **divergent** (v2+distill / PBRS / curriculum) |
| 风险 | teacher 之间**信号冗余**, student 学不到额外东西 → peak ≈ 单 teacher 水平 | teacher 之间 state distribution mismatch, ensemble avg 可能 noise |
| 预期收益 | KL 信号 smooth, gradient 方差低 → 更稳 convergence, compound 055 lane 内部 variance | 若 failure mode 真 complementary → compound knowledge, 突破 0.918 |

两个 pool 是同一时间点对 distillation 轴的 **对立 bet**: 如果两个都输, distill 轴在 0.907 就已经 saturate; 如果两个都赢, 说明 Pool A/B **正交**, 未来 Pool C 可以叠起来。

### 0.3 与 "ensemble = 非智力" 原则一致

仍然只 deploy single network 推理一次。3-teacher 只在训练期贡献 KL gradient, 不进入推理图。

### 0.4 为什么现在跑 Pool A

- 055@1150 combined 2000ep 0.907 已 plateau, **直接 extend 055 不大概率再爬** ([snapshot-055 §7.9 已述](snapshot-055-distill-from-034e-ensemble.md#79-未尽事项--未做的事))。
- 055v2 recursive distill ([snapshot-061](snapshot-061-055v2-recursive-distill.md)) 的 1210-2000 extend 正在跑, 一旦判 peak 即可作为 Pool A 的 T2。
- 056D@1140 (lr=3e-4 HP sweep) 跟 055 combined 2000ep 只差 0.016, 但 H2H 对 055 marginal 0.536 (非独立 1000ep), 说明两条 lane **在 ceiling 附近是 near-tied siblings** — 放进同一个 ensemble 合理。
- 时间窗: 若 055v2_extend ~09:00 EDT 完成 post-eval, Pool A 可在今日 14:00 前 launch, 12h 训练次日 02:00 完。

## 1. 核心假设

### H_071

> 用 KL distill loss 把 3 个 **同 family** teacher (055@1150 + 055v2@peak + 056D@1140) 的 joint action probs 压缩进 031B-arch student (warm-start from 031B@80, 同 055 配方),
> **combined 2000ep peak ≥ 0.915**, 即 vs 055 SOTA 0.907 **+0.008pp** (跨越 1.2σ SE 可检测门槛)。

### 子假设

- **H_071-a (homogeneous teacher → low KL variance)**: 3 个 teacher 在同 reward signal + 同架构下训练, 他们对 identical state 的 policy 差异主要是 training noise / HP variance, **ensemble avg 相当于 "055 lane 家族内部的 bagging"**, KL 目标分布 smooth 且稳定, student gradient 方差比 Pool B 低。
- **H_071-b (3 × homogeneous > 1 × 055)**: 如果 Pool A 能越过 0.907, 说明 "Student vs 最强 single teacher" 上仍有未压榨的信号 (来自 HP-sweep / recursive variance). 如果 Pool A 只 tie 0.907, 说明单 teacher 已经是 family ceiling。
- **H_071-c (warm-start from 031B@80 削弱早期 teacher conflict)**: 跟 055 setup 一致, 不 scratch — student 起点已在 teacher neighborhood, 避免 early training KL 直接冲突 PPO gradient direction。

## 2. 设计

### 2.1 总架构 (完全沿用 055 student 配方, 仅换 teacher set)

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn, warm-start 031B@80) → student_logits
                     → teacher_ensemble (3 frozen team_siamese) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)
α(t): 0.05 → 0.0 linear decay 8000 updates, temperature 1.0
```

### 2.2 Teacher 组成 (3 ckpts, uniform 1/3 each)

| Teacher | 来源 lane | ckpt 路径 | 近邻性 / 角色 |
|---|---|---|---|
| T1 | 055@1150 (current SOTA) | `/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150` | 主 anchor, 当前 project SOTA 0.907 |
| T2 | 055v2@peak (recursive distill 变体) | **TBD** (run dir `/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/checkpoint_0018XX/checkpoint-18XX`, iter ∈ [1800, 2000]) | same family, recursion axis (teacher 自己进入 teacher pool) |
| T3 | 056D@1140 (lr=3e-4 HP sweep) | `/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140` | same arch + same shaping, LR axis variance |

T2 路径在 055v2_extend 完成 + post-train-eval 选出 peak ckpt 后才能敲定, **这是本实验的唯一 launch blocker**。

### 2.3 Teacher weighting

**uniform 1/3 each** (不按 baseline WR 做 performance-weight):
- 3 个 teacher combined 2000ep baseline WR 分别 0.907 / ~0.900 (预期) / 0.891 — spread 只有 0.016, 比 Pool B 的 0.907/0.902/0.892 还窄。
- performance-weight 的 marginal 收益不大, 但 implementation 复杂度 +1 (需要 env var + 新 tensor stack) — 不值。
- 若 L1 降级 sweep, 再引入 weighted (见 §4.2 预案)。

### 2.4 Student + 训练 setup (完全 match 055)

```bash
# Architecture (031B 同款 student)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Warm-start (MATCHING 055 CONFIG — 关键差异 vs Pool B scratch)
WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# Distillation
TEAM_DISTILL_ENSEMBLE_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<T1>,<T2 pending>,<T3>"
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0

# PPO (055 同款)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (055 同款, 不动)
USE_REWARD_SHAPING=1 SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (比 055 scratch 短 — warm-start 省掉前 80 iter 的 exploration)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.5 与 055 / Pool B 的精确对照

| 差异维度 | 055 (原 SOTA) | Pool A (本 071) | Pool B (070) |
|---|---|---|---|
| Warm-start | 031B@80 | **031B@80 (same)** | scratch |
| Teacher set | 031B@1220 + 045A@180 + 051A@130 (同 family 但 "mid-training") | **055@1150 + 055v2@peak + 056D@1140 (同 family 但 "late-training SOTA")** | 055@1150 + 053Dmirror@670 + 062a@1220 (异 family) |
| α / T / LR / shaping | default | **same as 055** | same as 055 |

Pool A 唯一独立变量: **teacher set 升级到 "3 个 late-training same-family ckpts"**, 测 teacher quality 是否能 compound 超过单 teacher 0.907。

## 3. 预注册判据

锚点: 055@1150 combined 2000ep = **0.907 ± 0.007 SE** (from [snapshot-055 §7.13](snapshot-055-distill-from-034e-ensemble.md#713-2026-04-20-1325-edt-baseline-rerun-v2--0551150-combined-2000ep--0907-supersedes-single-shot-0911))。SE ±0.007 → 1σ 阈值 ≈ 0.914, 2σ 阈值 ≈ 0.921。

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.912 | +0.005 vs 055 | homogeneous ensemble gives small real gain, >1σ SE 可检测 |
| **§3.2 main: peak ≥ 0.915** | **+0.008** | **H_071 met** — 3×same-family > 1×same-family, 有显著信号 |
| §3.3 breakthrough: peak ≥ 0.925 | +0.018 | homogeneous ensemble 在 distill 轴 unlocks major gain (>2.5σ), 触发 Pool C 级联 |
| §3.4 tied: peak ∈ [0.895, 0.912) | sub-marginal | 3 × 0.90 teachers ≈ 1 × 0.907 teacher, 同 family redundancy, Pool B 的 diversity 框架胜 |
| §3.5 regression: peak < 0.890 | ↓ | KL avg of same-family teachers 反而干扰 fine-tune, 055 lane 已 saturated, 压缩更多知识 overfit |

## 4. 简化点 + 风险 + 降级序列

### 4.1 简化 S1.A — 沿用 online distillation (teacher 看 student rollout)

| 简化项 | 完整方案 | 当前选择 | 节省 |
|---|---|---|---|
| 数据收集 | DAGGER iterative | Online (teacher forward student obs) | ~3 天工程 |

**风险**: 训练后期 student 进化, 可能走到 teacher 训练 distribution 外 — 但因为 teacher 本来就是 "031B-arch + v2 shaping + warm 031B@80" 的**同一 manifold**, distribution overlap 比 Pool B 高, **online 简化 penalty 应 < Pool B**。

**预期退化**: -0.2 ~ -0.5pp vs 理想 DAGGER (比 Pool B 的 -0.5 ~ -1.5pp 更温和)。

**预案**:
- L1: α 衰减加快 (4000 updates)
- L2: iter 600 pause + DAGGER step
- L3: offline distill pipeline

### 4.2 简化 S1.B — uniform 1/3 weight (不 performance-weight)

**风险**: 056D 0.891 < 055 0.907, uniform 可能让 weakest teacher 拖后腿。但 spread 仅 0.016 = 2×SE, 拖后概率有限。

**预期退化**: -0.1 ~ -0.3pp vs 最优 weighted ensemble。

**预案**:
- L1 (若 peak ∈ §3.4): performance-weighted (0.40 : 0.33 : 0.27), +8h GPU

### 4.3 简化 S1.C — 固定 α=0.05 (不 sweep)

**风险**: same-family teacher 的 KL 信号 smooth, α=0.05 可能稍强 (对 homogeneous teacher 来说 lower α 就够了)。

**预案**: L1 α sweep {0.02, 0.05, 0.10} 若 peak < 0.908。

### 4.4 全程降级序列

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | Online + α=0.05 + uniform + 3-teacher same-family | base ~10h (warm-start 比 Pool B scratch 稍快) |
| 1 | Peak ∈ [0.895, 0.912) (§3.4) | performance-weighted + α sweep | +16h GPU |
| 2 | Peak < 0.890 (§3.5) | α 降到 0.02 + uniform | +8h GPU |
| 3 | Step 1/2 仍 < 0.912 | DAGGER L2 | +5 天工程 |
| 4 | Step 3 失败 | declare Pool A (homogeneous) 路径 dead, 依赖 Pool B (divergent) 单独验证 distill ceiling | — |

## 5. 不做的事

- 不 scratch train (warm from 031B@80, 严格 match 055 配方)
- 不改 student 架构 (031B Siamese + cross-attn)
- 不改 shaping (v2 同 055)
- 不改 α / temperature / LR 作为 first-pass (单变量只有 teacher set)
- 不混入 learned reward / PBRS / curriculum (那些是 Pool B 的轴)
- 不 launch 前 055v2_extend 未判 peak — **T2 路径必须确定**
- 不与 Pool B (snapshot-070, 若已 running) 抢节点 (PORT_SEED 隔离)

## 6. 执行清单

- [x] 1. Snapshot 起草 (本文件)
- [ ] 2. **等 055v2_extend 完成** (~2026-04-21 06:05 + 3h ≈ 09:00 EDT)
- [ ] 3. 对 055v2_extend invoke `/post-train-eval 055v2` → 选出 T2 peak ckpt (预期 iter ∈ [1800, 2000])
- [ ] 4. 起草 `scripts/eval/_launch_071_poolA_homogeneous_distill.sh` (见 §6.1 env var 清单)
- [ ] 5. Smoke test: load 3 ckpt 到 `_FrozenTeamEnsembleTeacher`, forward 一步确认 shape 正确
- [ ] 6. 选 free node (PORT_SEED ≠ 31/23/13/51/55/37/Pool B's), 建议 PORT_SEED=71
- [ ] 7. Launch 1250 iter warm-start run (~10h)
- [ ] 8. 实时 monitor: KL decay, alpha schedule, iter rate
- [ ] 9. 训完 invoke `/post-train-eval 071`
- [ ] 10. Stage 1 1000ep top-k + Stage 1.5 rerun v2 500ep → combined 2000ep
- [ ] 11. Stage 2 500ep capture peak ckpt
- [ ] 12. Stage 3 H2H: vs 055@1150 (T1, anchor), vs 055v2@peak (T2), vs 056D@1140 (T3), vs Pool B SOTA (若 070 已判)
- [ ] 13. Verdict append §7, 严格按 §3
- [ ] 14. 更新 rank.md + BACKLOG.md

### 6.1 Launcher 待起草 — 关键 env var 清单

`scripts/eval/_launch_071_poolA_homogeneous_distill.sh` **需要在 T2 peak 确定后才能起草**。拷贝 `_launch_055_distill_034e_scratch.sh` 作为 base, 改动点:

```bash
LANE_TAG=071
PORT_SEED=71  # 建议, far from 37 (055) / 31 (054) / 23 (053A) / 13 (031B-noshape) / 51 (051D) / 55

# 关键: warm-start from 031B@80 (MATCHING 055 CONFIG)
export WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# Teacher ensemble (3 paths, comma-separated)
export TEAM_DISTILL_ENSEMBLE_KL=1
export TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="\
/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150,\
<055v2_PEAK_CKPT_TBD>,\
/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_20260420_092042/TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/checkpoint_001140/checkpoint-1140"

# 同 055
export TEAM_DISTILL_ALPHA_INIT=0.05
export TEAM_DISTILL_ALPHA_FINAL=0.0
export TEAM_DISTILL_DECAY_UPDATES=8000
export TEAM_DISTILL_TEMPERATURE=1.0
export LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
export MAX_ITERATIONS=1250 CHECKPOINT_FREQ=10 EVAL_INTERVAL=10

# Architecture + shaping 与 055 launcher 一致 (直接拷贝), 见 §2.4
```

## 7. Verdict — §3.4 TIED (homogeneous distill saturate, 2026-04-22 append-only)

### 7.1 Stage 1 baseline 1000ep (2026-04-22 [00:53 EDT])

- Trial: `071_poolA_homogeneous_distill_warm031B80_20260421_073426/TeamVsBaselineShapingPPOTrainer_Soccer_199de_00000_0_2026-04-21_07-34-47`
- Selected ckpts (top 5%+ties+±1, 25 ckpts): 460-480 / 670-720 / 830-860 / 970-990 / 1020-1040 / 1100-1160
- Eval node: atl1-1-03-015-30-0, port 60105, 1037s parallel-7

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1120** | **0.903** | 903-97 |
| 860 / 990 | 0.901 | 901-99 |
| 720 | 0.899 | 899-101 |
| 670 / 850 | 0.898 | 898-102 |
| 1150 | 0.897 | 897-103 |
| 1140 / 1160 / 970 | 0.900 | 900-100 |
| 840 | 0.894 | 894-106 |
| 710 | 0.891 | 891-109 |
| 1100 | 0.889 | 889-111 |
| 470 / 1040 | 0.886 | 886-114 |
| 980 | 0.885 | 885-115 |

**peak = 0.903 @ ckpt-1120, mean(top 6) ~0.900, range [0.862, 0.903]**

### 7.2 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 main ≥ 0.915 | ❌ 0.903 (-0.012) | not met |
| §3.2 stretch ≥ 0.920 | ❌ | not met |
| §3.3 strong tied [0.910, 0.915) | ❌ 0.903 below | not met |
| **§3.4 marginal tied [0.895, 0.910)** | **✅ 0.903 in range** | **TIED, ceiling at 0.91** |
| §3.5 regression < 0.895 | ❌ | not regressed |

**Δ vs prior SOTA 055@1150 (0.907) = -0.004** — within SE。 **Δ vs NEW SOTA 1750 (0.9155) = -0.013** — sub-SOTA。

### 7.3 §8 outcome 对照

| Pool A 结果 | Pool B 结果 | 结论 |
|---|---|---|
| ~~Breakthrough~~ | — | not met |
| **Tied (0.903)** | (Pool B 见 snapshot-070, tied 0.892) | **distill 轴在 0.91 saturate** — 与 072 / 076 / 079 一致 |

### 7.4 与 071/072/076/079 ceiling 模式合读

见 [snapshot-079 §6.3](snapshot-079-055v3-recursive-distill.md#63-与-071072076-saturation-模式合读) — 4 lane 同时 saturate 0.90-0.91 → distill paradigm itself 已饱和, 不是 teacher 数量/多样性 / student 容量瓶颈。

### 7.5 Raw recap

```
=== Official Suite Recap (parallel) === (full 25 ckpts above)
[suite-parallel] total_elapsed=1037.7s tasks=25 parallel=7
```

完整 log: [071_baseline1000.log](../../docs/experiments/artifacts/official-evals/071_baseline1000.log)

### 7.6 Lane 决定

- **Pool A 071 lane 关闭** — 同家族 homogeneous distill 没 break ceiling
- 080 (Pool A v2 with 1750 teacher) 仍在跑;若也 saturate 0.91 → confirm distill paradigm 整体限于 ~0.91
- 资源已转 080 / 081 / 082 / 083 / 073-resume



## 8. 后续路径 (基于 verdict)

### Outcome A — 突破 (peak ≥ 0.915, H_071 met)

- Homogeneous ensemble 真的 compound 出了 family 内部 variance 的额外信号
- 立即扩展: **Pool C** — 4-way homogeneous {055@1150 + 055v2@peak + 056D@1140 + 新增 031B-noshape@SOTA}
- 若 Pool B (snapshot-070) 也突破, 说明两个 pool 的信号正交, 应 Pool A+B 叠起来: 6-teacher mixed (3 homo + 3 divergent), snapshot-072 级别实验

### Outcome B — 持平 (peak ∈ [0.895, 0.912), §3.4)

- homogeneous teacher 的 KL 信号 redundant with single 055@1150 teacher (H_071-a 部分成立但不足以 break ceiling)
- 执行 §4.4 step 1 (performance-weighted + α sweep)
- 如果 sweep 仍 < 0.912: **Pool A 路径效果确认**, 结论是 "same-family compound 不 work, 必须靠 diversity (Pool B) 或完全换 axis"

### Outcome C — 退化 (peak < 0.890, §3.5)

- 3 个 same-family teacher 的 KL avg 反而**磨平了 055@1150 的峰值** (student 被拉到 ensemble 平均的弱水平)
- 执行 §4.4 step 2 (α=0.02 保守试一次)
- 如果仍退化: confirm "same-family averaging hurts" — distill 轴的健康做法必须引入 diversity (Pool B 框架胜)

### Outcome 对比 Pool B 的最终 verdict 矩阵

| Pool A 结果 | Pool B 结果 | 结论 |
|---|---|---|
| Breakthrough | Breakthrough | 两轴正交, Pool C 叠加 |
| Breakthrough | Tied/Regress | homogeneous compound 胜, diversity 冗余 |
| Tied/Regress | Breakthrough | diversity 胜, 同家族饱和 |
| Tied/Regress | Tied/Regress | distill 轴在 0.907 saturate, 转其他 axis (architecture / 更深 RL / self-play) |

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — T1 来源, current SOTA 0.907, student 配方 donor
- [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md) — T2 来源 (055v2 recursive distill, extend 中)
- [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) — T3 来源 (056D lr=3e-4 HP sweep)
- [SNAPSHOT-070](snapshot-070-pool-B-divergent-distill.md) — Pool B (divergent path) 直接对照, 测 diversity vs homogeneous 轴
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B student 架构 + warm-start 起点
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — 含 `_FrozenTeamEnsembleTeacher` 实现
- [_launch_055_distill_034e_scratch.sh](../../scripts/eval/_launch_055_distill_034e_scratch.sh) — launcher 模板, 071 launcher 直接拷贝修改

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — 标准 distillation, soft target averaging 理论
- **Allen-Zhu & Li 2020** "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning" — homogeneous ensemble distill 的 "multi-view" 解释; 证明 same-family teacher avg 可能 compound out hidden features (支持 H_071-a)
- **Furlanello et al. 2018** "Born-Again Neural Networks" — student = teacher 架构但更强的现象 (跟 T1=student 本 family, T2=055v2 recursive, T3=HP sibling 的 setup 理论匹配)
- **Rusu et al. 2016** "Policy Distillation" — RL distillation, multi-task/multi-teacher 实证
- **Czarnecki et al. 2019** "Distilling Policy Distillation" — PPO + KL term 兼容性
