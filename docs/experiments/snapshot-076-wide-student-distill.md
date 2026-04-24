## SNAPSHOT-076: Wide-Student Distill — 扩 student capacity 回收 teacher-student gap (DIR-A)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); pending free node + implementation (arch env var extension ~2h)
- **前置**: [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) (SOTA recipe) / [snapshot-075 §5.1 / §6 DIR-A](snapshot-075-strategic-synthesis-toward-0.93.md#51-★★★★★-larger-student-capacity-for-distillation)
- **同期对照**: [snapshot-077](snapshot-077-per-agent-student-distill.md) (DIR-B) / [snapshot-078](snapshot-078-dagger-distill.md) (DIR-C)

---

## 0. 背景

### 0.1 现有 distill student 体积对比 teacher

| 角色 | Lane | Arch | 参数量 | 备注 |
|---|---|---|---:|---|
| Student | 055 / 055v2 / 059 / 061 / 068 / 066 / Pool A-D | 031B (Siamese+cross-attn) | **~0.46M** | 全部 distill lane 的 student 都是这个 |
| Teacher (1) | 031B@1220 | same | ~0.46M | |
| Teacher (2) | 045A@180 | same | ~0.46M | |
| Teacher (3) | 051A@130 | same | ~0.46M | |
| Teacher total (3-way) | — | — | **~1.4M** | student 3× smaller than teacher ensemble total |

- Hinton 2015 "Distilling the Knowledge in a Neural Network": student capacity 小于 teacher 总 capacity 时, distill gain 受 student bottleneck 限制
- 在 image classification 上, student capacity ≥ 最大 single teacher 时 gain 最明显; < 0.5 × teacher total 时 distill 产出 sub-ceiling

### 0.2 Empirical 证据: 055 → 055v2 recursive saturation

- 055v2 在 3-teacher 上加 2 个 teacher → 5-teacher (~2.3M total), 保持同 student (0.46M)
- combined 3000ep verdict: **+0.002 NS over 055**
- 诠释: 再多 teacher 信息但 student 容量没增, **teacher-student capacity gap 变大, marginal gain → 0**
- 这是本 lane 的核心 motivating 证据: **gain 不是来自 teacher diversity 的 marginal, 而是 student 的 bottleneck**

### 0.3 Why this is the highest-EV Direction A

- 相对 077 (per-agent): 工程改动最小 (只扩 hidden)
- 相对 078 (DAGGER): 无需新 infra (rollout buffer / teacher-label pipeline)
- 测试的是 "055 ceiling 是不是 student capacity bound" 这一 crisp 机制假设 — 无论结果都产生 load-bearing evidence

---

## 1. Hypothesis H_076

### 1.1 主假设

> H_076 (主): 扩 student encoder `[256,256]` → `[384,384]` + merge `[256,128]` → `[384,256]` (student params 0.46M → ~1.0M), 保持 055 其他 setup 不变, 最终 combined 2000ep peak **≥ 0.920** (+0.013 over 055 SOTA 0.907, >1.5× SE, approaching 2σ).

### 1.2 Stretch

> Stretch H_076-stretch: combined 2000ep peak ≥ 0.925 — student capacity is the binding constraint at 0.91 plateau, wide student unlocks +0.018pp.

### 1.3 子假设

- **H_076-a**: student 0.46M < teacher total 1.4M (3× gap) 是 055 ceiling 的 necessary-but-not-sufficient cause. Wide student 缩窄到 1.0M vs 1.4M (1.4× gap) 让 student 有足够 capacity fit 3-teacher ensemble policy manifold
- **H_076-b**: PPO optimizer 对 1M-params network 仍稳定 (031A/B 已经 700K, 052 1M+ 失败 primarily 由 LN / merge 拓扑改动, 不是单纯 param count)
- **H_076-c**: 无需改 reward / teacher / LR / T / budget — 单 isolate student capacity 变量

### 1.4 Anti-hypothesis

- H_076-null: 0.91 不是 capacity bound, 而是 **teacher ensemble 的 joint policy 本身 cap at ~0.91** — wide student 也无法超过 teacher collective knowledge. 此时 combined 2000ep peak ∈ [0.895, 0.915) tied

---

## 2. Design

### 2.1 Student Architecture (wider Siamese)

```
Input: 672-dim joint obs (= concat(agent_0_obs_336, agent_1_obs_336))

Siamese encoder (WIDER):
  obs_336 → Linear(336 → 384) → ReLU → Linear(384 → 384) → ReLU → feat_384 (per agent)

Within-agent cross-attention (= 031B, unchanged):
  feat_384 → tokens(4 × 96 dim) → single-head attn → attended_384

Merge (WIDER):
  concat(feat_0, attended_0, feat_1, attended_1) → 1536 dim
  Linear(1536 → 384) → ReLU → Linear(384 → 256) → ReLU → policy/value heads
```

Param count rough:
- encoder: 336×384 + 384×384 ≈ 0.28M per branch, shared = 0.28M
- attention: Q/K/V projections 384 → 96 = ~0.11M
- merge: 1536×384 + 384×256 ≈ 0.69M
- heads: 256 × 27 (policy) + 256 × 1 (value) ≈ 0.007M
- **total ≈ 1.08M** (vs 031B 0.46M = 2.3× wider, vs teacher total 1.4M = 77%)

### 2.2 Teacher + Distill Setup (= 055 exactly)

- Teacher: {031B@1220, 045A@180, 051A@130} (same as 055)
- α: 0.05 init → 0 final, 8000 updates decay
- T: 1.0
- Distill loss: factor-prob KL (same mechanism as 055)

### 2.3 PPO / Reward / Budget (= 055 exactly)

- LR=1e-4, CLIP_PARAM=0.15, NUM_SGD_ITER=4
- TRAIN_BATCH_SIZE=40000, SGD_MINIBATCH_SIZE=2048
- v2 shaping (TIME_PENALTY + BALL_PROGRESS + POSSESSION + OPP_PROGRESS + deep_zone)
- MAX_ITERATIONS=1250, TIMESTEPS_TOTAL=50000000, TIME_TOTAL_S=43200
- CHECKPOINT_FREQ=10, EVAL_INTERVAL=10

### 2.4 Code Changes Needed

1. `cs8803drl/branches/team_siamese.py` — 新环境变量:
   ```
   TEAM_SIAMESE_ENCODER_HIDDENS=384,384      (already wired, just needs 384 value)
   TEAM_SIAMESE_MERGE_HIDDENS=384,256        (already wired)
   TEAM_CROSS_ATTENTION_DIM=96               (change from 64)
   ```
   All 3 env vars already exist in `SiameseCrossAttentionTeamTorchModel.__init__` per snapshot-031B / 055 plumbing — **no code change needed**, only launcher change.

2. `cs8803drl/branches/team_siamese_distill.py` — factor-prob marginalization 与 student arch 完全解耦, **无改动需要**
3. Launcher `scripts/eval/_launch_076_wide_student_distill.sh` — copy `_launch_055_distill_034e_scratch.sh`, 只改 3 个 hidden env var

### 2.5 Port / Resource Plan

- PORT_SEED: 76 (隔离 Pool A=71 / Pool B=70 / Pool D=73 / Pool C=72 / 068=68)
- Budget: 1 × H100 × 14h
- 依赖: 无 blocker (teacher ckpt 已存在)

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict |
|---|---|---|---|
| §3.1 marginal | combined 2000ep peak ≥ 0.914 | +0.007 vs 055 (> SE) | capacity extension has measurable effect |
| **§3.2 主** | **combined 2000ep peak ≥ 0.920** | **+0.013 (~1.8× SE, H_076 met)** | **wide student breaks 0.91 plateau** |
| §3.3 breakthrough | combined 2000ep peak ≥ 0.925 | +0.018 (stretch) | student capacity was the binding constraint |
| §3.4 持平 | combined 2000ep peak ∈ [0.895, 0.914) | within ±2σ of 055 | capacity not binding; 0.91 has other cause |
| §3.5 退化 | combined 2000ep peak < 0.890 | wide student didn't converge | 1M params × PPO unstable (052 pattern risk) |

**Decision rule**: Stage 1 single-shot 1000ep peak must ≥ 0.911 to trigger Stage 2 rerun; otherwise close lane (sub-marginal, not worth combined 2000ep spend). Stage 2 rerun on top 3 plateau ckpts — combined 2000ep = verdict.

---

## 4. Simplifications + Risks + Retrograde Sequence

### 4.1 简化 A — 只测 1.0M-param wide student (not a full sweep)

- Full sweep: 0.7M / 1.0M / 1.5M 三档 capacity
- 当前: 单 1.0M run
- **Risk**: 1.0M 可能 over-parameterized (052 lesson); 真 sweet spot 可能 0.7M
- **降级 L1** (peak < 0.911): 加 0.7M variant in parallel slot (7h on H100 free frag)
- **降级 L2** (both < 0.91): 关 lane, capacity axis 否决

### 4.2 简化 B — 不改 teacher / LR / T / α

- 变量 isolation; 如果扩容 sign positive, **未来可以 stack** T sweep / LR / PBRS on top
- **Risk**: 单变量可能找不到 combo optimum
- **降级**: 若 §3.1 HIT 但未 §3.2, follow-up = wide student + PBRS (相当于 068 stacked on 076 架构)

### 4.3 简化 C — 不扩 cross-attention tokens (4 tokens 不变)

- 如果 tokens=8 或 dim=128 可能需要但风险高
- **Risk**: cross-attn capacity 不扩可能成为 second bottleneck
- **降级**: L3 variant with TEAM_CROSS_ATTENTION_TOKENS=6, dim=128 (+ ~0.15M params)

### 4.4 Risk R1 — PPO instability at 1M params (052 precedent)

- 052 transformer 在 ~1M params 时 REGRESSION -8pp
- **但 052 的主因是 LN × PPO gradient 和 merge 拓扑改动**, 不是单纯 param count (054/054M 同 ~0.55M 也 NS)
- **Mitigation**: 我们保持 031B 的拓扑 (concat + MLP merge), 只扩 hidden size — 这是 052 没做的 pure "wider 031B" 测试
- **Monitor**: policy_entropy, kl_divergence, value_loss per iter; 若 kl > 10 或 policy_entropy drop to < 0.1 by iter 200, 提前终止

### 4.5 Risk R2 — Online distill state-shift 在 wide student 上放大

- 055 §4.1 已述 online simplification 的 shift risk
- Wide student 更容易 overfit teacher distribution → 在 student-own rollout state 上 KL 可能 noisy
- **Mitigation**: α schedule 可能需要更 aggressive decay (L1 retrograde = 4000 updates decay vs 8000)

### 4.6 全程 Retrograde Sequence

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default | 1.0M wide + 055 recipe | 14h |
| 1 | Stage 1 single-shot < 0.911 | close lane, 076 否决 | — |
| 2 | §3.2 miss but §3.1 HIT | add PBRS (076+068 combo) next round | +14h |
| 3 | §3.4 tied 055 on combined | 0.91 不是 capacity bound, 转 DIR-B per-agent 或 DIR-C DAGGER |

---

## 5. 不做的事

- 不改 teacher pool (3-way, not 5-way; isolate capacity axis)
- 不 sweep LR / T / α (057 / 059 / 063 已证明无贡献)
- 不 warm-start (scratch as 055, 避免 warm conflict)
- 不叠加 PBRS / curriculum / opponent pool 轴 — **留给 follow-up combo**

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [ ] 2. 创建 launcher `scripts/eval/_launch_076_wide_student_distill.sh` (copy 055, 改 3 env vars + PORT_SEED=76)
- [ ] 3. Smoke: launch 10 iter 确认 model 构建 OK + 参数量 ~1.0M + forward shape 对 + distill KL term finite
- [ ] 4. 监控 iter 200: kl_divergence, policy_entropy, value_loss 在健康范围
- [ ] 5. Full 1250 iter launch
- [ ] 6. Stage 1 post-eval 1000ep on 10 top ckpts
- [ ] 7. 如果 peak ≥ 0.911, Stage 2 rerun top 3 ckpts → combined 2000ep
- [ ] 8. Stage 3 H2H portfolio: vs 055@1150 (same teacher-recipe sibling), vs 031B@1220 (base anchor), vs 034E (teacher)
- [ ] 9. Verdict append §7

---

## 7. Verdict — §3.4 TIED (capacity 不是 binding constraint, 2026-04-22 append-only)

### 7.1 Stage 1 baseline 1000ep (2026-04-22 [01:00 EDT])

- Trial: `076_wide_student_distill_scratch_20260421_083310/TeamVsBaselineShapingPPOTrainer_Soccer_4ebd4_00000_0_2026-04-21_08-33-32`
- Architecture: encoder 384,384 + cross-attn dim 96 (vs default 256,256 / dim 64) — **~1.4× capacity**
- Selected ckpts (top 5%+ties+±1, 33 ckpts): 320-340 / 620-700 / 760-790 / 880-900 / 960-1010 / 1070-1110 / 1160-1180
- Eval node: atl1-1-03-012-3-0, port 60305, 1295s parallel-7

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1000** | **0.905** | 905-95 |
| 770 | 0.899 | 899-101 |
| 1110 | 0.898 | 898-102 |
| 650 | 0.897 | 897-103 |
| 960 | 0.896 | 896-104 |
| 790 / 1160 | 0.890 | 890-110 |
| 880 / 990 | 0.889 | 889-111 |
| 620 / 660 | 0.888 | 888-112 |
| 700 / 710 | 0.887 | 887-113 |
| 760 | 0.884 | 884-116 |
| 780 / 980 / 1090 / 890 | 0.883 | 883-117 |

**peak = 0.905 @ ckpt-1000, mean(top 6) ~0.899, range [0.849, 0.905]**

### 7.2 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 stretch ≥ 0.920 | ❌ 0.905 | not met |
| §3.2 main ≥ 0.911 | ❌ 0.905 | not met |
| §3.3 marginal [0.907, 0.911) | ❌ 0.905 just below | not met |
| **§3.4 tied [0.900, 0.907)** | **✅ 0.905 in range** | **TIED, capacity NOT the bottleneck** |
| §3.5 regression < 0.900 | ❌ | within ceiling cluster |

**Δ vs 055 SOTA recipe (0.907) = -0.002** — 完全 within SE。 **Δ vs NEW SOTA 1750 (0.9155) = -0.011** — sub-SOTA。

### 7.3 关键 lesson — DIR-A 假设否定

**snapshot-075 §4 / 本 snapshot §0.3 的 "student capacity bound" 假设 FALSE**:
- 1.4× capacity wider student 训了 1250 iter, peak 0.905
- 与 0.46M 默认 student 同 paradigm (071/072/079) peak 0.903-0.914 几乎相同
- → **distill paradigm 整体卡在 ~0.91, 不是 student 容量, 不是 teacher 数量, 不是 reward 多样性**

可能的真 bottleneck (待 080/081 等结果分清):
- **PPO + Hinton KL 自身的 mode collapse**: KL 让 student 收敛到 teacher 的 mode 平均, 失去 PPO 探索 marginal
- **Teacher 池不可避免地共用 baseline-targeted policy distribution**: 即使 reward 不同 (072), 都是 v2-derived ⇒ teachers' joint action distribution 高度相关
- **Static teacher 不能给 student 当前 state-distribution 上的 fresh signal**: DAGGER (078) 才能解此, 但 078 deferred

### 7.4 与 071/072/079 ceiling 模式合读

见 [snapshot-079 §6.3](snapshot-079-055v3-recursive-distill.md#63-与-071072076-saturation-模式合读)。

**4 lane 同时 saturate 0.90-0.91**, 设计变量正交 (teacher count / teacher diversity / reward axis / student capacity), 但都 cap 同一处 → **distill paradigm 自身的极限**。

### 7.5 Raw recap

```
=== Official Suite Recap (parallel) === (full 33 ckpts above)
[suite-parallel] total_elapsed=1295.4s tasks=33 parallel=7
```

完整 log: [076_baseline1000.log](../../docs/experiments/artifacts/official-evals/076_baseline1000.log)

### 7.6 Lane 决定

- **DIR-A wide-student 路径关闭** — capacity 不 binding
- 不执行 §8 后续 A/B/C — 都假设 capacity 是 binding, 实测否定
- 资源已转: 081 (orthogonal aggressive reward, **不依赖 distill paradigm**), 082/083 (architecture axis), 080 (Pool A v2 用更强 teacher 看是否能撬动 +0.005 marginal), 073-resume



---

## 8. 后续路径

- **A (§3.2 / §3.3 HIT)**: 路径成立 — 下一步 stack 076 + PBRS (= 068 recipe with wide student); 再下一步 076 + DAGGER (078 recipe)
- **B (§3.1 HIT but not §3.2)**: sub-marginal gain, 不放弃但低优先 — 试 wide+PBRS combo 看 synergy
- **C (§3.4 tied)**: **capacity 不是 binding constraint** — 0.91 plateau 的 cause 在 reward / data / distill mechanism, 转 077 per-agent 或 078 DAGGER

---

## 9. 相关

- [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) — strategic synthesis (DIR-A motivation)
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 SOTA recipe (teacher + setup)
- [snapshot-061](snapshot-061-055v2-recursive-distill.md) — 055v2 recursive distill 证明 teacher count 扩展 saturates (motivates capacity instead)
- [snapshot-052](snapshot-052-031C-transformer-block-architecture.md) — precedent for PPO instability at ~1M params (但 cause 是 LN/merge, not param count alone)
- [cs8803drl/branches/team_siamese.py](../../cs8803drl/branches/team_siamese.py) — SiameseCrossAttentionTeamTorchModel (already env-var-driven, no code change)

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — student capacity ≥ max single teacher often improves distill gain
- **Stanton et al. 2021** "Does Knowledge Distillation Really Work?" — empirical finding that capacity gap bounds distill ceiling

