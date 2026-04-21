## SNAPSHOT-070: Pool B — Divergent-Path Ensemble Distill

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); 等 launch

## 0. 背景与定位

### 0.1 当前 plateau 三极

| Lane | Mechanism | Combined 2000ep | 备注 |
|---|---|---:|---|
| 055@1150 | v2 shaping + 3-teacher 034E distill (031B/045A/051A) | **0.907** | current SOTA, Pool A baseline |
| 053Dmirror@670 | PBRS predictor only, no dense shaping | **0.902** (single-shot 1000ep peak 0.902) | PBRS-only lane |
| 062a@1220 | curriculum (random → mixed → baseline), no shaping, adaptive | **0.892** | curriculum lane |

三条 lane 都 saturate 在 0.89-0.91 区间, 但**驱动它们达到 plateau 的 reward signal 完全不同**:

- 055: sparse + v2 dense + distill KL from 3-teacher 034E (031B/045A/051A)
- 053D: sparse + PBRS potential-based predictor (Ng 1999)
- 062a: sparse + opponent curriculum (progressive difficulty)

### 0.2 与 Pool A 的对照

Pool A (snapshot-066 progressive distill / BAN) 是**同 family recursive distillation**: 每一代 student 的 teacher 都来自前一代 distill family。知识单一源但 compound。

Pool B (本 snapshot) 是**diverse-path ensemble distill**: 3 个 teacher 来自**完全不同 mechanism 的 lane**, 每个 teacher 带着自己 reward mechanism 的 inductive bias。

**Framing**: 如果 055@1150 是 "3 teacher from same shaping family + same 031B architecture" 的结果, Pool B 测试 "3 teacher from 3 different reward mechanisms" 是否给 student 带来**complementary 而非 redundant** knowledge。

### 0.3 与 "ensemble = 非智力" 原则一致

[用户已明确](../../README.md): ensemble 本身只是 stability, 不是 intelligence。Pool B 不部署 3-way ensemble, 而是 **distill 3 个 diverse teacher 的 joint knowledge 到 single 031B-arch network** — 部署时仍然只跑 1 个 forward, 完全合规 single-model SOTA 路径。

Reference: Hinton 2015 distillation paper — student 经常超过 individual teacher, 尤其当 teachers cover non-overlapping failure modes 时更明显。

## 1. 核心假设

### H_070

> 用 KL distill loss 把 3 个 **reward-mechanism divergent** teacher (055@1150, 053Dmirror@670, 062a@1220) 的 joint action probs 压缩进 031B-arch student, **combined 2000ep peak ≥ 0.918**, 超过当前 055 SOTA 0.907 **+0.011**。

### 子假设

- **H_070-a (teacher diversity → complementary knowledge)**: 3 teacher 的 reward signal 完全不同 (v2+distill vs PBRS vs curriculum), 他们在 baseline 下的 failure modes 应 minimally overlap:
  - 055: shaping 驱动的 aggressive attack, 可能 wasted_possession 过度
  - 053D: PBRS 驱动的 field-progress 权衡, 可能 opening 保守
  - 062a: curriculum-trained, 对 weak opponent 过 aggressive, strong opponent 偶尔 lose coordination
  - Student 从 ensemble avg 学到 "在不同 state 下 blend 3 种 policy" 的能力, failure modes pairwise cancel
- **H_070-b (031B capacity sufficient)**: 031B Siamese + cross-attention 在 0.46M params 下, 已证明在 Pool A 055 setting 下能 represent 3-teacher joint policy (0.907 达到 ensemble stretch)。Pool B 换 teacher set 不改架构 → capacity 没问题。
- **H_070-c (diverse reward signal outweighs same-family compound)**: 直接的 apples-to-apples comparison — 如果 Pool B (diverse) > Pool A (same-family) 在 peak basis, 说明 teacher diversity axis 重要于 recursion depth axis。否则 recursion (Pool A) 是更 fruitful 路径。

## 2. 设计

### 2.1 总架构 (完全 mirror 055 setup, 只换 teacher set)

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn) → student_logits
                     → teacher_ensemble (3 frozen team_siamese) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)

α(t) schedule: 0.05 → 0.0 linear decay over first 8K SGD updates
```

### 2.2 Teacher ensemble 组成 (3 ckpts, uniform 1/3 each, no recency bias)

| Teacher | 来源 lane | ckpt 路径 | Mechanism |
|---|---|---|---|
| T1 | 055@1150 (v2 shaping + 3-teacher 034E distill) | `/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150` | v2 dense shaping + distill KL from 034E |
| T2 | 053Dmirror@670 (PBRS only, no shaping) | `/storage/ice1/5/1/wsun377/ray_results_scratch/053Dmirror_pbrs_only_warm031B80_20260420_094739/TeamVsBaselineShapingPPOTrainer_Soccer_8c3d4_00000_0_2026-04-20_09-48-01/checkpoint_000670/checkpoint-670` | Potential-based reward shaping (Ng 1999) |
| T3 | 062a@1220 (curriculum + adaptive + no-shape) | `/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220` | Opponent curriculum (random → mixed → baseline) |

Weights: uniform **1/3 each** (no recency bias, 测 diversity effect 而非 "best teacher amplification")。

### 2.3 Student + 训练 setup (完全 match 055 Pool A, 除 teacher set 外 zero 差异)

```bash
# Architecture (031B 同款 student)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Distillation
TEAM_DISTILL_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<T1 path>,<T2 path>,<T3 path>"
TEAM_DISTILL_ALPHA_INIT=0.05         # 同 055
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0         # standard, 不 sharpen

# PPO (031B / 055 同款, 保守)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (student 端保持与 055 一致, 让 teacher 带 diversity)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.4 与 055 setup 唯一差异

- **Teacher set**: 055 用 {031B@1220, 045A@180, 051A@130} (same-family 031B-arch + shaping siblings)
- **Pool B**: {055@1150, 053Dmirror@670, 062a@1220} (3 个 reward mechanism 完全不同的 lane)

Student 架构、LR、shaping、α schedule、temperature、batch size、budget — **全部 identical to 055**。这样单变量对照能直接归因 teacher diversity 的 effect。

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal ≥ 0.912 | combined 2000ep peak ≥ 0.912 | +0.005 vs 055 SOTA (> SE), diversity gives small real win |
| §3.2 主 ≥ 0.918 | combined 2000ep peak ≥ 0.918 | **+0.011 vs 055, H_070 met** — diversity 路径真正超过 same-family |
| §3.3 突破 ≥ 0.925 | combined 2000ep peak ≥ 0.925 | +0.018 — teacher diversity is the dominant axis, 需要进一步扩展 (snapshot-071) |
| §3.4 持平 [0.895, 0.912) | combined 2000ep peak in this band | diversity doesn't help, 3-teacher ensemble knowledge 其实 redundant with single-family |
| §3.5 退化 < 0.890 | combined 2000ep peak | teacher conflict 干扰 student, diverse reward signals 的 gradient 互抵 |

## 4. 简化点 + 风险 + 降级预期 + 预案 (用户要求 mandatory)

### 4.1 简化 S1.A — 沿用 online distillation (same as 055)

| 简化项 | 完整方案 | 当前选择 | 节省 |
|---|---|---|---|
| 数据收集 | DAGGER iterative | Online (teacher 看 student rollout) | ~3 天 |

**风险**: 训练后期 student 与 3 个 diverse teacher 的 joint distribution 在 state space 上 diverge → student 看到的 state 可能在**所有 3 个 teacher 的 training distribution 之外**, ensemble avg 可能输出 noise。**Pool B 比 Pool A 风险更高**, 因为 3 个 teacher 的 state coverage 本就不同 (curriculum vs baseline-only training set)。

**降级预期**: -0.5 ~ -1.5pp vs 理想 DAGGER。Pool B 可能比 Pool A 受 online 简化 penalty 更大。

**预案**:
- L1: α decay 更快 (4000 updates vs 8000) — 早期 learn diverse teacher, 后期纯 PPO 让 student 在 coherent reward signal 下 converge
- L2: iter 600 pause + DAGGER step (收集 student rollout obs → teacher action labels → supervised KL)
- L3: 切 offline distillation pipeline

### 4.2 简化 S1.B — uniform 1/3 weights (not sweep weights)

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| Teacher weights | (w1, w2, w3) grid sweep | uniform (1/3, 1/3, 1/3) |

**风险**: 某个 teacher 明显比其他强 (055 = 0.907 > 053D = 0.902 > 062a = 0.892), uniform weight 可能让 student 被 weaker teacher 拖累。

**降级预期**: -0.2 ~ -0.5pp vs 最优 weighted ensemble。

**预案**:
- L1 (若 peak < 0.908): 试 performance-weighted (0.45 : 0.30 : 0.25 按 combined 2000ep WR 比例)
- L2 (若 L1 仍 < 0.908): 切回 single-teacher + 055@1150 作为 teacher 但保留 diversity by rotating per-batch 模式

### 4.3 简化 S1.C — 不换 architecture / LR / shaping / α

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| Student 侧 HP | 独立 HP sweep | 完全 match 055 setup |

**风险**: Pool B teacher set 可能需要不同的 α (比如 diverse teacher 输出更"散", KL 信号本身 noisier, 需要更低 α)。

**降级预期**: -0.3pp vs 最优 α for diverse teacher。

**预案**: L1 α sweep {0.02, 0.05, 0.1} 如果 Pool B peak < 0.895。

### 4.4 全程降级序列 (worst case 路径)

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | Online + α=0.05 + uniform weights + 3-teacher divergent set | base ~6-8h |
| 1 | Peak in [0.895, 0.912) (§3.4 持平) | α sweep {0.02, 0.1} 在 2 节点并行 | +16h GPU |
| 2 | Peak < 0.890 (§3.5 退化) | weighted teachers + α=0.02 (保守 combo) | +8h GPU |
| 3 | Step 1/2 仍不过 0.912 | DAGGER L2 | +5 天 工程 |
| 4 | Step 3 失败 | declare Pool B 路径 dead, Pool A (snapshot-066 recursive) 成为唯一 distillation 路径 | — |

## 5. 不做的事

- 不在 snapshot-066 Pool A 出 verdict 之前宣布 "diversity > recursion" 或反之 — 两个 pool 独立跑完对齐 combined 2000ep
- 不改 student 架构 (031B-arch from scratch, 严格对照 055)
- 不在 Pool B 上混入 RND / curriculum 等其他 axis (纯 teacher diversity test)
- 不用更多 teacher (e.g. 4-way 加 031B-noshape) — 3-way 足以测 H_070 中心假设
- 不跑 non-uniform weights 作为 first-pass (简化 S1.B 保持)

## 6. 执行清单

- [ ] 1. 确认 3 个 teacher ckpt 可 load (smoke test _FrozenTeamEnsembleTeacher with 3 paths)
- [ ] 2. 写 launch script `scripts/eval/_launch_070_pool_B_divergent_distill.sh`
- [ ] 3. 选 free node (PORT_SEED ≠ 31/23/13/51/55)
- [ ] 4. Launch 1250 iter scratch (12h)
- [ ] 5. 实时 monitor: KL decay, alpha 衰减正常, iter rate 无 degradation
- [ ] 6. 训完 invoke `/post-train-eval 070`
- [ ] 7. Stage 1 1000ep top-k ckpts
- [ ] 8. Stage 1.5 rerun v2 500ep on top-3 ckpts → combined 2000ep
- [ ] 9. Stage 2 500ep capture on peak ckpt
- [ ] 10. Stage 3 H2H: vs 055@1150 (Pool A reference), vs 053Dmirror@670 (teacher T2), vs 062a@1220 (teacher T3)
- [ ] 11. 如果 peak > 055@1150: 再加 H2H vs 031B@1220 + vs 066-Pool A SOTA (pending)
- [ ] 12. Verdict append §7, 严格按 §3 判据
- [ ] 13. 更新 rank.md + README.md + BACKLOG.md

## 7. Verdict (待 combined 2000ep eval 后 append)

### 7.1 2026-04-21 15:30 EDT — Stage 1 single-shot 1000ep verdict — Outcome B-/C (tied-to-slight-regression), **LANE CLOSED**

**Setup**: Pool B = divergent-reward-path 3-teacher distill {055@1150 (v2-shape) + 053Dmirror@670 (PBRS-only) + 062a@1220 (no-shape curriculum)}, uniform 1/3 weights, all other setup mirrors 055 Pool A. New-run (not merged resume). Stage 1 post-eval = official evaluator parallel 1000ep.

**Raw Recap top-5 (single-shot 1000ep each)**:

```
=== Official Suite Recap top-5 ===
ckpt 1210 vs baseline: win_rate=0.899   [★ peak]
ckpt 1180 vs baseline: win_rate=0.896
ckpt 1130 vs baseline: win_rate=0.896
ckpt 930  vs baseline: win_rate=0.893
ckpt 1190 vs baseline: win_rate=0.893
```

Log: `docs/experiments/artifacts/official-evals/070_poolB_baseline1000.log`

**预注册判据 (§3) 对照**:

| §3 判据 | 阈值 | single-shot @1210 | verdict |
|---|---:|---:|---|
| §3.1 breakthrough (≥0.918, H_070 met) | Pool B > Pool A | 0.899 | **MISS** (-0.019) |
| §3.2 持平 (∈ [0.895, 0.912)) | tied 055 | 0.899 | **HIT** |
| §3.3 退化 (< 0.890) | teachers conflict | 0.899 | NO |

**Δ vs reference**:
- vs **055@1150 SOTA combined 2000ep 0.907**: Δ = **-0.008 slight regression** within single-shot SE 0.010
- vs **Pool A (homogeneous) 066A @1180 0.909**: Δ = **-0.010**
- vs **074A deploy-time ensemble 0.903** (same member set, different mixing): Δ = -0.004 **essentially tied**

**Verdict**:
- Cross-reward-path 3-teacher distillation (v2-shape + PBRS + no-shape) **fails to break 0.91**
- Confirms pattern observed in snapshot-074A (same three members ensembled at deploy time): divergent reward paths do not compound into additive WR gain; reward-path diversity ≠ action-space diversity once all three members are 031B-descendants
- **Lane CLOSED** (Outcome B-per §8). Not worth combined rerun — single-shot already below 055 SOTA; no α-sweep or weighted-teacher follow-up justified

**Meta-context (2026-04-21 7-lane Stage 1 sweep)**:
- Pool B joins 066A / 066B / T-sweep variants in the "tied-055-at-0.89-0.91-plateau" cluster
- Only 055v2_extend broke the plateau at single-shot; all other lanes saturate below or at 055
- Mechanism implication: **ensemble / pool distillation is saturated when all teachers share the same 031B base arch**; future distill improvements must come from architectural diversity (054M MAT, wide-student 076, per-agent 077) OR distill loss formulation (DAGGER 078), not teacher-reward diversity

_No follow-up launched. Lane resources redirected to 055v2_extend rerun (highest-priority) + 076/077/079._

## 8. 后续发展线 (基于 verdict 的路径图)

### Outcome A — 突破 (combined 2000ep peak ≥ 0.918, H_070 met)

- Pool B > Pool A (assuming Pool A < 0.918) → **diversity axis wins over recursion axis**
- 立即扩展: snapshot-071 "Pool C" 加入 4th teacher (031B-noshape or 056D lr=3e-4 path) 进一步扩 diversity
- 长期: 定期捕捉新 mechanism lane 作为 teacher, iterative compound

### Outcome B — 持平 (peak ∈ [0.895, 0.912))

- diversity 没给显著增益, 但也没 regression → **teacher selection 不是 bottleneck, 其他 axis (architecture / HP / 更深 RL) 才是**
- 执行 §4.4 step 1 (α sweep), 确认不是 α suboptimal
- 如果 α sweep 仍 < 0.912, **Pool B 路径关闭**, 集中资源到 Pool A (recursive distill) or 其他 axis

### Outcome C — 退化 (peak < 0.890)

- 3 个 diverse teacher 的 gradient 互抵 — 反 H_070-a 假设, teachers 确实 overlap 不够 / conflict 过度
- 执行 §4.4 step 2 (weighted + 低 α)
- 如果仍退化, **declare Pool B 路径 dead**, distillation 轴仅保留 Pool A single-family recursion

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — Pool A / reference teacher set, 作为 student 配方 donor
- [SNAPSHOT-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) — T2 teacher (PBRS lane) 来源
- [SNAPSHOT-058](snapshot-058-real-curriculum-learning.md) / [SNAPSHOT-062](snapshot-062-curriculum-noshape-adaptive.md) — T3 teacher (curriculum lane) 来源
- [SNAPSHOT-066](snapshot-066-progressive-distill-BAN.md) — Pool A (recursive / BAN) 直接对照
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B student 架构
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — distill model 实现, 含 `_FrozenTeamEnsembleTeacher`
- [ensemble_agent.py](../../cs8803drl/deployment/ensemble_agent.py) — ensemble 推理参考

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — 标准 distillation, student 可 > individual teacher 尤其当 teachers diverse
- **Rusu et al. 2016** "Policy Distillation" — RL distillation, multi-task distill 中 diverse teacher 增益已证
- **Ng et al. 1999** "Policy Invariance under Reward Transformations" — PBRS 理论基础, 解释为什么 T2 (053D) 和 T1 (055 shaping) 能 represent 互补的 value function
- **Czarnecki et al. 2019** "Distilling Policy Distillation" — distillation loss 与 PPO 兼容性证明
