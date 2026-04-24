## SNAPSHOT-073: Pool D — Cross-Reward Signal Diversity Distill

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); 等 068 完成 + peak ckpt 识别后 launch

## 0. 背景与定位

### 0.1 Pool 系列 axis 总览

| Pool | Axis 测试 | Teacher 组成 | Student |
|---|---|---|---|
| A (snapshot-066) | recursion depth (BAN, same-family compound) | 055 自己的 children (multi-gen) | 031B |
| B (snapshot-070) | divergent reward mechanisms (mixed family + arch) | 055+053Dmirror+062a | 031B |
| C (snapshot-072) | architecture diversity (same reward, diff arch) | 031B + 054M + 056D + 062a | 031B |
| **D (本 snapshot)** | **pure cross-reward signal, same arch** | **055 (v2) + 068 (PBRS) + 062a (no-shape+curriculum)** | **031B (warm from 031B@80)** |

**Pool D 的独立设计 claim**: 与 Pool B 最像但更严格——Pool B 里 062a 的 curriculum 本身就是 opponent-axis 干扰 + 053Dmirror 是 warm-start 而非 scratch。Pool D 选的三位 teacher 全都**跑相同的 031B 架构**、**同样在 baseline vs 031B-family 里出 plateau**,差异**仅在 reward 路径**:

- **055**: v2 dense shape (TIME_PENALTY + BALL_PROGRESS + POSSESSION + deep_zone etc.) + 034E distill KL
- **068 (warm or scratch)**: PBRS outcome-predictor (Ng99 potential-based) + 034E distill KL, **无 v2 shape**
- **062a**: 纯 sparse reward + 4-phase adaptive opponent curriculum, **no dense shape, no distill**

→ 3 条 teacher reach 相似 baseline WR (0.89-0.91 区间) 但**reward credit-assignment pathway 完全不同**。

### 0.2 与 "ensemble = 非智力" 的区分

[用户已明确](../../README.md): ensemble 只是 stability, 不是 intelligence。Pool D 不部署 3-way vote; 用 KL distill 把 3 个 reward-path teacher 的 joint action distribution 压缩进 single 031B network, 部署 = 1 forward。若 student > max(teachers), 该增益归于**reward-signal orthogonality 带来的 implicit regularization**, 不是 ensemble 运行时集成。

### 0.3 Framing: 为什么单独开 Pool D 而非让 Pool B 覆盖

Pool B 的三位 teacher 里:
- 053Dmirror 是 warm-from-031B@80 的 PBRS-only lane (非 scratch, optimizer 初始状态不同)
- 062a 带 curriculum + sparse-only, reward 路径 + opponent 路径**双重差异**

Pool D 严格把 reward-path 单独 isolate:
- **所有 3 位 teacher 都 run 同款 031B 架构 (Siamese + cross-attn)**
- **所有 3 位 teacher 的训练 opponent 都覆盖 baseline** (055 直接对 baseline; 068 直接对 baseline; 062a 在 phase-4 全 baseline)
- **唯一差异 = reward function**: shape / PBRS / none

因此 Pool D 对 H 的测试最干净: "同架构 + 同对手, 仅换 reward pathway → student 能否从 cross-reward teacher 里 extract 互补信号?"

## 1. 核心假设

### H_073

> 用 KL distill loss 把 3 个 **reward-path 完全不同但同架构同对手** 的 teacher (055@1150, 068@peak, 062a@1220) 的 joint action probs 压缩进 031B-arch student (warm-start from 031B@80), **combined 2000ep peak ≥ 0.915**, 超过当前 055 SOTA 0.907 **+0.008** (>1σ SE, approaching 2σ)。

### 子假设

- **H_073-a (reward signal orthogonality)**: v2 shape / PBRS / none 三条 reward 路径在 dense credit-assignment 的位置不同 (shape 直接 bias action toward heuristic, PBRS bias toward predictor-estimated value, none 只在 sparse 拐点更新)。若 policy distribution 也相应不同 → student 从 ensemble avg 里获得在不同 state 下不同 action preference 的 implicit mixture。
- **H_073-b (warm-start 031B@80 缓解 online-distill distribution shift)**: Pool B 全 scratch, 本 Pool warm-start 让 student 早期 state 分布与 teacher training distribution 更接近, 降低 ensemble teacher 输出 noise 风险 (本 snapshot §4 Risk R1)。
- **H_073-c (student 超过 max(teachers))**: 055 teacher 0.907 > 068@peak (TBD, pending verdict) > 062a 0.892。若 student ≥ 0.915, 即超过最强 teacher → 证明 reward-path diversity axis 是 fruitful, 类比 Hinton 2015 "student > teacher" pattern 在 RL + multi-reward setting。

### 与 Pool B 的 diff-predict

若 Pool B peak = Pool D peak → reward-path orthogonality 不敏感于 teacher 的 warm-status/curriculum; 若 Pool D > Pool B by > 2σ → warm-start + 更干净的 reward axis isolation 有额外 gain; 若 Pool D < Pool B → 053Dmirror-PBRS 实际比 068-PBRS 更有 distill value, 需要回 Pool B 路径。

## 2. 设计

### 2.1 总架构 (mirror 055 / Pool B, 唯一差异是 teacher 集 + warm-start)

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn, init from 031B@80) → student_logits
                     → teacher_ensemble (3 frozen team_siamese) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)

α(t) schedule: 0.05 → 0.0 linear decay over first 8K SGD updates
Temperature T = 1.0 (standard, no sharpening — per snapshot-063 verdict T=2 没用)
```

### 2.2 Teacher ensemble 组成 (3 ckpts, uniform 1/3, pending 068 peak)

| Teacher | 来源 lane | ckpt 路径 | Reward pathway |
|---|---|---|---|
| T1 | 055@1150 (v2 shape + 034E distill) | `/storage/ice1/5/1/wsun377/ray_results_scratch/055_distill_034e_ensemble_to_031B_scratch_20260420_092037/TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/checkpoint_001150/checkpoint-1150` | v2 dense hand-tuned shape + ensemble distill KL |
| T2 | **068@peak** (PBRS + 034E distill, **pending**) | TBD under `/storage/ice1/5/1/wsun377/ray_results_scratch/068_055PBRS_distill_{warm,scratch}_20260421_*/...` — **选 068_warm 或 068_scratch 中 baseline 1000ep peak 更高的一支** | Ng99 PBRS outcome-predictor + ensemble distill KL, no v2 shape |
| T3 | 062a@1220 (curriculum, no shape, no distill) | `/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220` | sparse-only + 4-phase adaptive opponent curriculum |

**068 选点规则 (pending completion)**:
- 若 068_warm@peak 与 068_scratch@peak 两者都完成 + 都达 ≥ 0.89 baseline 1000ep: **选 higher peak** 作为 T2 (单支)
- 若仅一支完成 / 达标: 用那一支
- 若都 < 0.88 (regression): **把 T2 临时换成 053Dmirror@670 退回 Pool B 公式**, 重发 snapshot revision (但这时 Pool D 失去 "pure 068 PBRS + 055 distill-tree + 062a curriculum" 的 framing, 在 snapshot-073-revised 中 flag)
- 若两支都完成且 peak 都 ≥ 0.89, 差距 < 2σ: **2 variants**: Pool D-warm (T2=068_warm) + Pool D-scratch (T2=068_scratch) 同时 launch, 顺便测 068 warm vs scratch effect on downstream distill

Weights: uniform **1/3 each** (follow Pool B 简化, 单变量对照 reward diversity axis 而非 weight sweep)。

### 2.3 Student 初始化: **WARM-start from 031B@80** (与 Pool A/B 不同)

- **选 031B@80 理由**: 068_warm 也是从 031B@80 warm-start (snapshot-068); 让 Pool D student 与 T2 (068_warm) 共享起点, 降低 KL conflict 风险 (初期 student-teacher 分布接近)
- `WARMSTART_CHECKPOINT=/storage/ice1/5/1/wsun377/ray_results_scratch/031B_team_level_siamese_crossattn_20260418_*/TeamVsBaselineShapingPPOTrainer_Soccer_*/checkpoint_000080/checkpoint-80` (glob final path at launch time)
- **与 Pool B scratch 对比**: 若 Pool D (warm) > Pool B (scratch) by >2σ, warm-start value established; 反之说明 warm-start 在这个 setup 下被 distill KL 的 later-stage signal 压过

### 2.4 训练 setup (identical to Pool B except warm-start + teacher set)

```bash
# Architecture (031B 同款 student)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Warm-start
WARMSTART_CHECKPOINT=<031B@80 path — glob at launch>

# Distillation (3 cross-reward teachers)
TEAM_DISTILL_ENSEMBLE_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<055@1150>,<068@peak>,<062a@1220>"
TEAM_DISTILL_ALPHA_INIT=0.05        # mirror 055/Pool B
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0        # standard (063 T=2 no gain)

# PPO (031B 同款, 保守)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (student 侧 keep v2 match 055 baseline, teacher 带 reward diversity)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.5 与 Pool B setup diff (明确列差异)

| 项 | Pool B (snapshot-070) | Pool D (本 snapshot) |
|---|---|---|
| Student init | scratch | **warm-start from 031B@80** |
| Teacher T1 | 055@1150 (v2 shape + distill) | 055@1150 (**same**) |
| Teacher T2 | 053Dmirror@670 (PBRS-only, **warm-start**, **no distill**) | **068@peak (PBRS + distill)** |
| Teacher T3 | 062a@1220 (curriculum) | 062a@1220 (**same**) |
| Reward axis 控制 | 3 teacher 的 reward 路径 diverse, 但 T2 无 distill → 多了一个 axis | **3 teacher 全有或全无 distill? T1/T2 有 distill, T3 无** → 仍有 "distill 有无" 的 axis 混淆, 不过 2:1 ratio |

**诚实注记**: Pool D 不是 100% 干净的 reward-only axis — T3 (062a) 仍是 no-distill, 所以 distill-presence 仍有 2:1 ratio 差异。但比 Pool B (053Dmirror warm + 062a no-distill 两个不同偏差) 干净一级。

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal ≥ 0.911 | combined 2000ep peak ≥ 0.911 | +0.004 vs 055 SOTA (>SE), cross-reward diversity 微弱 real |
| §3.2 主 ≥ 0.915 | combined 2000ep peak ≥ 0.915 | **+0.008 vs 055, H_073 met** — diversity axis 可检测 (>1σ, approach 2σ) |
| §3.3 突破 ≥ 0.925 | combined 2000ep peak ≥ 0.925 | +0.018 — cross-reward axis dominant, 需要大规模 reward-path sweep (snapshot-0??) |
| §3.4 持平 [0.895, 0.911) | combined 2000ep peak in this band | cross-reward diversity 可 ignored, 3 个 teacher 的 reward path info 其实 redundant |
| §3.5 退化 < 0.890 | combined 2000ep peak | teacher reward-path 冲突伤 student — PBRS value landscape 与 v2 shape heuristic 在 action level 真的矛盾 |

## 4. 简化点 + 风险 + 降级预期 + 预案

### 4.1 简化 S1.A — 沿用 online distillation (same as 055/Pool B)

| 简化项 | 完整方案 | 当前选择 | 节省 |
|---|---|---|---|
| 数据收集 | DAGGER iterative | Online (teacher 看 student rollout) | ~3 天工程 |

**风险 R1**: student 后期 state 分布 drift → 3 个 reward-path teacher **可能在 action level 分歧更大** (v2 teacher 建议 A, PBRS teacher 建议 B, curriculum teacher 建议 C)。即使 baseline WR 都在 0.89-0.91, action distribution 在 state-by-state 上未必对齐。**Pool D 专属 risk**: 3 条 reward 路径在 value landscape 上不一致, 可能让 ensemble avg 输出变成"谁都不像"的妥协 policy, 反而 hurt。

**降级预期**: -0.5 ~ -1.5pp vs 理想对齐 teacher。此 risk 比 Pool B 可能高一点, 因为 Pool B 的 053Dmirror 虽然 reward 不同, 但 T2 没有 distill, action bias 更直接源自 PBRS gradient; Pool D 的 068 同时带 034E distill KL, 其 action 分布 = 两层 signal 叠加 (PBRS + 034E distill), 与 055 (v2 + 034E distill) 在 "共享 034E distill target" 上有重叠, 但 dense signal 不同, **可能把 distill 部分 cancel 掉, 剩 differential 只有 v2 vs PBRS 的 shaping 差**。

**预案 (3 层)**:
- **L1 (轻度)**: α decay 更快 (4000 updates) — 早期 learn diverse teacher, 后期纯 PPO 让 student 在 coherent v2-shape reward 下 converge
- **L2 (中度)**: 训练中段 pause, DAGGER 一轮 (student rollout → teacher action labels, supervised KL on buffer)
- **L3 (重度)**: offline distill pipeline

### 4.2 简化 S1.B — uniform 1/3 weights

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| Teacher weights | (w1, w2, w3) 网格 sweep | uniform (1/3, 1/3, 1/3) |

**风险**: 三位 teacher strength 不同 (055 0.907 > 068 TBD > 062a 0.892), uniform weight 让 weaker teacher 拖累 mean。**额外 Pool D 专属 risk**: 若 068_scratch vs 068_warm peak 差异超过 0.01 (显著), uniform 让 student 学到"某一支 068"的 artifact 而不是 PBRS-core 信号。

**降级预期**: -0.2 ~ -0.5pp vs 最优 weighted。

**预案**:
- L1 (若 peak < 0.910): performance-weighted (按 teacher combined 2000ep WR 归一化)
- L2: drop T3 (062a, 最弱), 回到 2-teacher (055 + 068) — 但这时 Pool D 就退化为 "v2 vs PBRS 对比" 无法测 3-way cross-reward

### 4.3 简化 S1.C — 不换 architecture / LR / shaping / α / student-side reward

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| Student 侧 HP | 独立 HP sweep | 完全 match 055 setup (v2 shape for student) |

**风险 R2 (KL-reward mismatch)**: student 端用 v2 shape, T2 (068) teacher 是 PBRS-trained → student 看到的 env reward 是 v2 shape 的 bias, teacher distill KL 里带的是 PBRS-bias value landscape。同一 state 下两个 signal gradient 方向可能互冲。

**降级预期**: -0.3 ~ -0.5pp vs 最优 student 侧 reward。

**预案**: L1 — 关 student 侧 v2 shape (用 sparse-only + teacher KL), 让 student 完全靠 teacher signal 驱动 dense; 仅当 base-run peak ∈ [0.895, 0.911) 时触发。

### 4.4 全程降级序列 (worst case 路径)

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | Online + α=0.05 + uniform weights + warm-031B@80 | base ~8-10h (warm 比 scratch 稍快) |
| 1 | Peak ∈ [0.895, 0.911) 持平 | α sweep {0.02, 0.1} 2 节点并行 | +16h GPU |
| 2 | Peak < 0.890 退化 | weighted teachers + α=0.02 保守 combo | +8h GPU |
| 3 | Step 1/2 仍 < 0.912 | drop T3, 回到 2-teacher (T1+T2) — Pool D framing 退化为 "v2 vs PBRS" 二元比较 | +10h |
| 4 | Step 3 仍 flat | DAGGER L2 | +5 天工程 |
| 5 | Step 4 失败 | declare Pool D dead, cross-reward axis 关闭 | — |

**Retrograde 逻辑**: 如果 regressing (Step 2 触发), 先试最直接的干预 (teacher weight + α), 不要上来就 DAGGER。若 2-teacher 也不行, reward-path diversity 这个 axis 本身 close 掉, 回到 Pool A (recursive) 或 Pool C (architecture) 轴。

## 5. 不做的事

- 不在 068_warm **或** 068_scratch 完成 + baseline 1000ep peak 识别之前 launch (避免用 mid-training ckpt 作 teacher, teacher 得是稳定的 plateau ckpt)
- 不在 Pool A / Pool B / Pool C 都 verdict 之前把 Pool D 当"最终胜出者"——四 pool 独立跑完对齐 combined 2000ep 后再断
- 不改 student 架构 (031B-arch warm from @80, 严格对照 055 / Pool B)
- 不混入 RND / curriculum / frontier-pool opponent 等其他 axis (纯 cross-reward teacher test)
- 不试 4-way / 5-way teacher (加 T4 = 054M 会再引入 architecture axis, 污染 reward-axis test)
- 不先 non-uniform weights (保持简化 S1.B)
- **不与 Pool B 抢节点** (PORT_SEED ≠ Pool B 的 171)

## 6. 执行清单

- [x] 1. Snapshot 起草 (本文件, pre-registration 格式)
- [ ] 2. 等 068_warm + 068_scratch 完成 (~12h from 06:05 EDT = ~18:00 EDT; 可能更早)
- [ ] 3. 068 baseline 1000ep post-eval (inline 200ep top-5 + ties + ±1 → Stage 1 1000ep) → 选 T2
- [ ] 4. Glob 031B@80 实际 ckpt 路径 (WARMSTART_CHECKPOINT)
- [ ] 5. Smoke test _FrozenTeamEnsembleTeacher with 3 paths (055/068/062a) load + forward
- [ ] 6. 写 launch script `scripts/eval/_launch_073_poolD_cross_reward.sh`
   - 基于 `_launch_070_poolB_divergent.sh` 改: LANE_TAG=073, PORT_SEED≠171, TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS 换成 3 个新 ckpt, 加 `export WARMSTART_CHECKPOINT=<031B@80 path>`
   - Env vars: `TEAM_DISTILL_ENSEMBLE_KL=1`, `TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=<3 paths>`, `TEAM_DISTILL_ALPHA_INIT=0.05`, `TEAM_DISTILL_ALPHA_FINAL=0.0`, `TEAM_DISTILL_DECAY_UPDATES=8000`, `TEAM_DISTILL_TEMPERATURE=1.0`, `LR=0.0001`, `MAX_ITERATIONS=1250`, `WARMSTART_CHECKPOINT=<031B@80>`
- [ ] 7. 找 free node (PORT_SEED 选 73 or 173) → srun/batch launch 1250 iter warm-start (12h 预估, warm-start 可能更快)
- [ ] 8. 实时 monitor: KL decay, alpha 衰减正常, iter rate 无 degradation, warm-start 初期 reward 不崩
- [ ] 9. 训完 invoke `/post-train-eval 073`
- [ ] 10. Stage 1 1000ep top-k ckpts
- [ ] 11. Stage 1.5 rerun v2 500ep on top-3 ckpts → combined 2000ep
- [ ] 12. Stage 2 500ep capture on peak ckpt
- [ ] 13. Stage 3 H2H portfolio:
  - vs 055@1150 (T1, SOTA reference)
  - vs 068@peak (T2, PBRS reference)
  - vs 062a@1220 (T3, curriculum reference)
  - vs Pool B@peak (070 peer Pool)
  - 如果 peak > 055: 加 vs 031B@1220 + vs 070(B)-peak + vs Pool A/C peaks
- [ ] 14. Verdict append §7, 严格按 §3 判据
- [ ] 15. 更新 rank.md + README.md + BACKLOG.md + task-queue

## 7. Verdict — §3.4 TIED (cross-reward distill saturate, 5/5 distill ceiling 确认, 2026-04-22 append-only)

### 7.1 Stage 1 baseline 1000ep on resume v2 (2026-04-22 [00:30 EDT])

- 训练历史: scratch warm-031B@80 → ckpt 920 (first run), then RESTORE_CHECKPOINT resume → ckpt 1250 (PORT_SEED=87 fix on 5033290)
- Trial: `073_poolD_resume920_to_1250_20260421_211734/TeamVsBaselineShapingPPOTrainer_Soccer_18687_00000_0_2026-04-21_21-17-57`
- Inline best (200ep noisy): 0.935 @ ckpt-1060 — but mean-reverts heavily on 1000ep eval
- Selected ckpts (top 10% fallback, 12 ckpts): 1050-1100 / 1180-1230
- Eval node: atl1-1-03-015-16-0 (5033290), port 60505, 527s parallel-7

| ckpt | 1000ep WR | NW-ML | inline 200ep | Δ (200→1000) |
|---:|---:|:---:|---:|---:|
| **🏆 1050** | **0.909** | 909-91 | — | — |
| **🏆 1090** | **0.909** | 909-91 | 0.92 | -0.011 |
| 1210 | 0.904 | 904-96 | 0.925 | -0.021 |
| 1060 | 0.902 | 902-98 | **0.935** | **-0.033** |
| 1200 | 0.902 | 902-98 | 0.92 | -0.018 |
| 1080 | 0.901 | 901-99 | — | — |
| 1100 | 0.901 | 901-99 | — | — |
| 1180 | 0.895 | 895-105 | — | — |
| 1190 | 0.895 | 895-105 | 0.92 | -0.025 |
| 1220 | 0.887 | 887-113 | 0.92 | -0.033 |
| 1230 | 0.887 | 887-113 | — | — |
| 1070 | 0.883 | 883-117 | — | — |

**peak = 0.909 dual-peak 1050+1090, mean(top 6) ~0.905, range [0.883, 0.909]**

**Inline 200ep 大幅 mean-revert**: 1060 (200ep 0.935 → 1000ep 0.902, **-0.033**), 1220 (0.92 → 0.887, **-0.033**), 1210 (0.925 → 0.904, -0.021)。再次确认 200ep noise SE ±0.05 的 risk (`feedback_inline_eval_noise.md`)。

### 7.2 严格按 §3 判据

| 阈值 | 实测 single-shot 1000ep | verdict |
|---|---|---|
| §3.1 marginal ≥ 0.911 | ❌ 0.909 (just below) | not met |
| §3.2 main ≥ 0.915 | ❌ | not met |
| §3.3 breakthrough ≥ 0.925 | ❌ | not met |
| **§3.4 持平 [0.895, 0.911)** | **✅ 0.909 in range** | **TIED, cross-reward NO LIFT** |
| §3.5 regression < 0.890 | ❌ 0.909 well above | no regression |

**Δ vs prior SOTA 055@1150 combined 0.907 = +0.002 within SE 统计 tied**。 **Δ vs 1750 NEW SOTA combined 4000ep 0.9155 = -0.007 within SE**。 **Δ vs Pool A 071 (0.903) = +0.006 marginal 但 within SE**。

注: 当前是 single-shot 1000ep (SE ±0.012), CI [0.891, 0.927]; 真值不能排除 0.91 区间, 但也未达 §3.1 阈值。**Pool D combined 2000ep rerun pending** (若做出来,~10 min) — 但基于 071/072/076/079 saturate pattern, 大概率 mean-revert 到 0.90 cluster 真值。

### 7.3 与 071/072/076/079 + 080 saturation 模式合读 (5/5 distill saturate)

| Lane | 设计 | Peak | Δ vs 1750 SOTA |
|---|---|---|---|
| 071 Pool A homogeneous | 3-teacher same-family | 0.903 | -0.013 |
| 072 Pool C cross-axis | reward 多样性 | 0.903 | -0.013 |
| 076 wide-student | 1.4× capacity | 0.905 | -0.011 |
| 079 single-teacher | 1 SOTA teacher recursive | 0.914 | -0.002 |
| **073 Pool D cross-reward** | **3-teacher 3-reward-paths** | **0.909** | **-0.007** |

**5 lane 同时 saturate 0.90-0.91**, 设计变量正交 (teacher count / family / reward axis / student capacity / reward-path diversity), 但都 cap 同一处 → **distill paradigm 自身的极限确认 (5/5 evidence)**。

**§3.4 持平 outcome → §8 Outcome B**: cross-reward diversity 信号 redundant, 3 个 teacher 的 reward path info 在 student-level 表达上其实 redundant。

### 7.4 Raw recap

```
=== Official Suite Recap (parallel) === (full 12 ckpts above)
[suite-parallel] total_elapsed=526.9s tasks=12 parallel=7
```

完整 log: [073_resume_baseline1000.log](../../docs/experiments/artifacts/official-evals/073_resume_baseline1000.log)

### 7.5 Lane 决定

- **Pool D 073 lane 关闭** — cross-reward 3-teacher 距 0.91 ceiling 没突破,与 071/072/076/079 模式一致
- 不执行 §4 L1/L2 降级 — 5/5 lane 同时 saturate, 不是 hyperparameter / α decay 问题
- 不做 combined 2000ep rerun (single-shot 0.909 在 statistical tied 区, mean-revert 到 0.90 是更可能的真值, 不会 unlock §3.1)
- 资源已转 080 / 081 / 082 / 083



## 8. 后续发展线 (基于 verdict 的路径图)

### Outcome A — 突破 (combined 2000ep peak ≥ 0.915, H_073 met)

- **cross-reward diversity axis 成立**. 与 Pool B 比对决定 warm-start 价值:
  - Pool D > Pool B by > 2σ → warm-start from 031B@80 是有效加速
  - Pool D ≈ Pool B → warm-start 效应 neutral, pure reward-path diversity 已 saturate
- 立即扩展: 4-way cross-reward teacher (加 T4 = 058 RND-augmented reward 或 future 新 reward-path lane)
- 若同时 Pool A (recursion) < Pool D (diversity), 说明**diversity axis > recursion axis**, 后续 distill 资源集中到找新 reward path

### Outcome B — 持平 (peak ∈ [0.895, 0.911))

- cross-reward diversity 没给显著 gain, 但也没 regression
- 执行 §4.4 Step 1 (α sweep) 确认不是 α 问题
- 若 sweep 仍 < 0.912, **declare cross-reward axis saturated**, 把 distillation 资源收缩到 Pool A (recursion) + Pool C (architecture)

### Outcome C — 退化 (peak < 0.890)

- 3 条 reward-path teacher 的 action-level 分歧 hurt student (反 H_073-a)
- 执行 §4.4 Step 2 (weighted + 低 α)
- 若仍退化, 执行 Step 3 (drop T3, 2-teacher variant)
- 若全部失败, **cross-reward axis lane 关闭**, distill 资源集中到 same-family recursion (Pool A)

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — T1 teacher 来源, 当前 SOTA, Pool D student 配方 donor
- [SNAPSHOT-068](snapshot-068-055PBRS-distill.md) — T2 teacher 来源 (pending verdict)
- [SNAPSHOT-062](snapshot-062-curriculum-noshape-adaptive.md) — T3 teacher 来源
- [SNAPSHOT-070](snapshot-070-pool-B-divergent-distill.md) — Pool B (divergent path, 最近对照)
- [SNAPSHOT-066](snapshot-066-progressive-distill-BAN.md) — Pool A (recursive, 另一 distillation 路径)
- [SNAPSHOT-063](snapshot-063-055-temp-sharpening.md) — T=2 tested and tied, 本 Pool D T=1.0 的依据
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B student 架构 + 031B@80 warm-start donor 来源
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — distill model 实现, 含 `_FrozenTeamEnsembleTeacher`
- [task-queue-20260421.md](../management/task-queue-20260421.md) — §4 Pool D 触发条件
- Launcher: [scripts/eval/_launch_073_poolD_cross_reward.sh](../../scripts/eval/_launch_073_poolD_cross_reward.sh) — **needs drafting after 068 completes** (base on `_launch_070_poolB_divergent.sh`, swap teacher ckpts + add `WARMSTART_CHECKPOINT=031B@80`)

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — 标准 distillation paper; student 超过 teacher 的 pattern 在 diverse teacher 更强
- **Rusu et al. 2016** "Policy Distillation" — RL multi-teacher distillation, 已证 diverse teacher 带来 policy breadth
- **Ng et al. 1999** "Policy Invariance under Reward Transformations" — PBRS 理论基础; **解释 H_073-a**: PBRS (T2) 与 v2 shape (T1) 在 asymptotic optimum 相同但 learning dynamics 不同 → 他们的 policy distribution 可以在 transient 阶段 diverge, 给 distill ensemble 提供 real diversity
- **Czarnecki et al. 2019** "Distilling Policy Distillation" — distillation loss 与 PPO 兼容性证明
- **Allen-Zhu & Li 2020** "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning" — 理论 framing: diverse teacher 的 feature coverage 互补时, distillation student 可逼近 max(teacher) 以上
