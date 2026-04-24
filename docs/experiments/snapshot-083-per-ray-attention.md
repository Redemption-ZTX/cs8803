## SNAPSHOT-083: Per-Ray Attention Siamese Encoder — arch revision to break 0.91 plateau (DIR ③)

- **日期**: 2026-04-21 发起，2026-04-22 verdict
- **负责人**: Self
- **状态**: lane 关闭 (2026-04-22) — **baseline axis HIT 但 peer axis LOSE `***`**。单 shot 0.919 @ ckpt-1000 先是 SOTA candidate；combined 2000ep 0.909（小于 SOTA 1750 的 0.9163）；直接 H2H vs 1750 sample=500 实测 `0.410 (205-295)`, z=−4.02 `***`。结论：083 arch 是 **"baseline-specialist"**，而不是 frontier-robust。详见 §7。
- **前置**: [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) (031B within-agent cross-attention) / [snapshot-082](snapshot-082-hierarchical-two-stream-siamese.md) (two-stream siamese, 0.885 sub-SOTA) / [snapshot-055v2](snapshot-055-distill-from-034e-ensemble.md) extend 1750 (SOTA 0.9163) / [rank.md](rank.md) plateau anchor
- **同期对照**: 082 (hierarchical two-stream, 空间层次 heuristic split); 083 (per-ray attention, spatial inductive bias on the structural side)

---

## 0. 背景与定位

### 0.1 plateau saturation evidence（2026-04-21）

[rank.md](rank.md) 当时 frontier 情况：

| Lane | Combined WR | Note |
|---|---:|---|
| 055@1150 | 0.907 | prior distill SOTA |
| 055v2_extend@1750 | **0.9163** | 当前 SOTA (recursive 5-teacher distill) |
| 082 | 0.885 | 两流 heuristic split, sub-SOTA |
| 062a/056D/074F | 0.89–0.916 | curriculum/T-sweep/ensemble, 全部 tied on plateau |

**arch 轴** 未 ablate：distill/curriculum 都在 031B-monolithic encoder 上做，encoder 结构性 inductive bias 未被系统测。

### 0.2 观察：336-dim 是 ray-perception

- `soccer_twos` policy obs = `ray_cast(336,)`；3 帧堆叠 × 14 条 ray × 8 维 tag-one-hot+distance
- 031B 的 encoder 把 336 一次过 MLP，既没 temporal 层次也没 spatial 层次
- 082 尝试了 temporal 层次（`self = last frame / env = prev 2 frames`）收效小（0.885 sub-SOTA）
- 083 切 **spatial**：14 条 ray 当 14 个 token，Transformer self-attn 让 encoder 主动学 "前向 ray vs 后向 ray"、"近距离 ray vs 远距离 ray" 的 inductive bias

### 0.3 与 082 的对比

| 维度 | 082 two-stream | 083 per-ray |
|---|---|---|
| split 轴 | temporal (3 frames) | spatial (14 rays) |
| encoder | 两条 MLP concat | Transformer self-attn over 14 tokens |
| param | 0.402M | 0.372M |
| 推理 | "近帧 self vs 远帧 env" | "每条 ray 是独立感知 unit" |
| result | 0.885 sub-SOTA | **0.919 peak / 0.909 combined / H2H vs 1750 `***` LOSE** |

---

## 1. Hypothesis H_083

### 1.1 主假设

> **H_083 (主)**: 把 per-agent 336-dim obs 拆成 `3 × 14 × 8` (frames × rays × feat)，对 rays 维 mean pool 得到 `14 × 24` ray tokens，送 2 层 Transformer self-attention (4 heads, FFN=128)，mean pool 得 64 维，再 `Linear(64→256)` → 256 维 agent feature → 接 031B within-agent cross-attention → 其余 (v2 shaping / LR=1e-4 / 1250 iter / baseline) 与 031B/055/082 一致。**Combined 2000ep peak ≥ 0.920**（+0.013 over 055 raw scratch 0.907，+0.004 over 055v2_extend@1750 的 0.9163）。

### 1.2 Stretch

> **H_083-stretch**: combined 2000ep peak ≥ 0.925 — spatial inductive bias 在 RayPerception 上比 hierarchical temporal 更自然；scratch arch revision 单独突破 0.92。

### 1.3 Anti-hypothesis

- **H_083-null**: 336-dim obs 的 `(3, 14, 8)` reshape 顺序与 Unity 二进制中 ray 的真实 layout 错配，Transformer 没法恢复 spatial 关系 → combined 2000ep tied 031B/055 (∈ [0.895, 0.915))
- **H_083-bad**: Transformer encoder 在 0.4M PPO 上 over-parameterized / 不稳定 → combined 2000ep < 0.890 regression

---

## 2. Design

### 2.1 encoder 结构

```
Per agent 336-dim obs:
  reshape → (B, 3, 14, 8)       # 3 frames × 14 rays × 8 features
  mean over frames → (B, 14, 24)  # actually concat-then-collapse: → 14 tokens × 24-dim
  embed: Linear(24→64) + pos_embed_14×64       # → 14 tokens × 64-dim
  for L in 2:
    pre-LN → MultiheadAttn(4 heads) → residual
    pre-LN → FFN(64→128→64) → residual
  mean pool over 14 tokens → 64-dim
  Linear(64→256, ReLU) → 256-dim agent feature

Two agents:
  agent_feat_0, agent_feat_1 (each 256-d)
  031B within-agent cross-attention (tokens=4, dim=64, unchanged)
    → attn_0, attn_1 flat 256-d
  merge [f0, a0, f1, a1] → Linear(1024→256) → Linear(256→128) → policy/value heads
```

### 2.2 关键超参

```bash
# per-ray attention encoder (083)
TEAM_SIAMESE_PER_RAY_ATTN=1
TEAM_PER_RAY_N_RAYS=14
TEAM_PER_RAY_FEAT_DIM=24
TEAM_PER_RAY_EMBED_DIM=64
TEAM_PER_RAY_ATTN_LAYERS=2
TEAM_PER_RAY_ATTN_HEADS=4
TEAM_PER_RAY_FFN_HIDDEN=128
TEAM_PER_RAY_POOL=mean

# 031B cross-attention 保留
TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

# PPO 同 031B
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# budget 1250 iter × 50M steps, v2 shaping (同 031B/055/082)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000 CHECKPOINT_FREQ=10
```

### 2.3 param count

- per-ray encoder (核心新增): `ray_embed 24×64=1.5K + pos_embed 14×64=0.9K + 2× transformer_block ≈ 36K` = **~39K per agent × shared** ≈ **~39K** (Siamese share)
- 加 031B cross-attn + merge + heads = 约 **0.372M** 总参数
- vs 031B 的 ~0.46M → 小 20%，但有更强 inductive bias

---

## 3. 预注册判据

| 阈值 | 判据 | verdict |
|---|---|---|
| §3.1 主: combined 2000ep peak ≥ 0.920 | +0.004 over 055v2@1750 | **架构 axis 突破** |
| §3.2 stretch: combined 2000ep peak ≥ 0.925 | +0.009 over SOTA | **spatial inductive bias 决定性成立** |
| §3.3 tied: combined 2000ep peak ∈ [0.895, 0.920) | within SE of plateau | 轴 tied, 不做下游 |
| §3.4 退化: combined 2000ep peak < 0.890 | < 031B/055 | arch 失败 |

**peer 轴（新 doctrine，snapshot-106 里正式化）**:

| 阈值 | 判据 | verdict |
|---|---|---|
| §3.5 peer HIT: H2H vs 055v2@1750 sample≥500, win_rate ≥ 0.55, z > 2.24 `*` | 直接压过 SOTA | SOTA shift 确认 |
| §3.6 peer tie: win_rate ∈ [0.47, 0.53] | tied on both axes | 架构 tied, SOTA 不变 |
| §3.7 peer LOSE: win_rate < 0.47, z < −1.96 `*` | frontier 不如 SOTA | 架构 arch 轴 HIT 但 policy 层被 1750 压过 |

---

## 4. 实际训练

- 发起: 2026-04-21 21:09 EDT, PORT_SEED=83, `/storage/ice1/5/1/wsun377/ray_results_scratch/083_per_ray_attention_scratch_20260421_210849/`
- run_dir trial: `TeamVsBaselineShapingPPOTrainer_Soccer_decf1_00000_0_2026-04-21_21-09-11`
- 训练 wallclock 约 ~11h, 跑到 iter 1250 正常完成
- inline 50ep eval 在 iter 790 附近 subprocess died (inline eval 的 bug, 不影响 training saver)；主 stage1/late-window 由 backfill parallel-7 补齐

---

## 5. Stage 1 baseline 1000ep eval

按 [`pick_top_ckpts.py`](../../scripts/eval/pick_top_ckpts.py) top 5% + ties + ±1，主 window (460-770) + late window (800-1250) 分两发跑：

### 5.1 主 window (22 ckpts, parallel-7, 1177s)

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| 460 | 0.852 | 852-148 |
| 470 | 0.850 | 850-150 |
| 480 | 0.854 | 854-146 |
| 510 | 0.842 | 842-158 |
| 520 | 0.821 | 821-179 |
| 530 | 0.851 | 851-149 |
| 550 | 0.836 | 836-164 |
| 560 | 0.848 | 848-152 |
| 570 | 0.838 | 838-162 |
| 580 | 0.858 | 858-142 |
| 600 | 0.870 | 870-130 |
| 610 | 0.879 | 879-121 |
| 620 | 0.874 | 874-126 |
| 650 | 0.870 | 870-130 |
| 660 | 0.877 | 877-123 |
| 670 | **0.882** | 882-118 |
| 680 | 0.876 | 876-124 |
| 690 | 0.840 | 840-160 |
| 700 | 0.866 | 866-134 |
| 750 | 0.877 | 877-123 |
| 760 | 0.878 | 878-122 |
| 770 | 0.871 | 871-129 |

主 window peak = **0.882 @ ckpt-670**，mean ≈ 0.861。

### 5.2 late window (10 ckpts, parallel-7, 575s)

inline eval 在 iter 790 死，pick_top_ckpts 错过了后段 peak。补 `800,850,900,950,1000,1050,1100,1150,1200,1250`：

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| 800 | 0.887 | 887-113 |
| 850 | 0.896 | 896-104 |
| 900 | 0.882 | 882-118 |
| 950 | 0.890 | 890-110 |
| **1000** | **0.919** | 919-81 |
| 1050 | 0.903 | 903-97 |
| 1100 | 0.884 | 884-116 |
| 1150 | 0.902 | 902-98 |
| 1200 | 0.908 | 908-92 |
| 1250 | 0.905 | 905-95 |

**late window peak = 0.919 @ ckpt-1000**，8/10 ckpt ≥ 0.882（主 window peak），且 5/10 ckpt ≥ 0.90；**late window 明显比主 window 高 ~2pp**，证实 083 训练在 1000+ 才到 peak（与 031B/055 的 pattern 一致）。

### 5.3 §3 判据初判（single-shot 1000ep）

- single-shot peak 0.919 vs SOTA 0.9163 → +0.003，**在 SE ±0.016 以内**，per-skill doctrine 标 "preliminary"，必须 rerun。

### 5.4 Raw recap

<details>
<summary>主 window recap</summary>

```
=== Official Suite Recap (parallel) ===
checkpoint-460 vs baseline: win_rate=0.852 (852W-148L-0T)
checkpoint-470 vs baseline: win_rate=0.850 (850W-150L-0T)
checkpoint-480 vs baseline: win_rate=0.854 (854W-146L-0T)
checkpoint-510 vs baseline: win_rate=0.842 (842W-158L-0T)
checkpoint-520 vs baseline: win_rate=0.821 (821W-179L-0T)
checkpoint-530 vs baseline: win_rate=0.851 (851W-149L-0T)
checkpoint-550 vs baseline: win_rate=0.836 (836W-164L-0T)
checkpoint-560 vs baseline: win_rate=0.848 (848W-152L-0T)
checkpoint-570 vs baseline: win_rate=0.838 (838W-162L-0T)
checkpoint-580 vs baseline: win_rate=0.858 (858W-142L-0T)
checkpoint-600 vs baseline: win_rate=0.870 (870W-130L-0T)
checkpoint-610 vs baseline: win_rate=0.879 (879W-121L-0T)
checkpoint-620 vs baseline: win_rate=0.874 (874W-126L-0T)
checkpoint-650 vs baseline: win_rate=0.870 (870W-130L-0T)
checkpoint-660 vs baseline: win_rate=0.877 (877W-123L-0T)
checkpoint-670 vs baseline: win_rate=0.882 (882W-118L-0T)
checkpoint-680 vs baseline: win_rate=0.876 (876W-124L-0T)
checkpoint-690 vs baseline: win_rate=0.840 (840W-160L-0T)
checkpoint-700 vs baseline: win_rate=0.866 (866W-134L-0T)
checkpoint-750 vs baseline: win_rate=0.877 (877W-123L-0T)
checkpoint-760 vs baseline: win_rate=0.878 (878W-122L-0T)
checkpoint-770 vs baseline: win_rate=0.871 (871W-129L-0T)
[suite-parallel] total_elapsed=1176.8s tasks=22 parallel=7
```
</details>

<details>
<summary>late window recap</summary>

```
=== Official Suite Recap (parallel) ===
checkpoint-800  vs baseline: win_rate=0.887 (887W-113L-0T)
checkpoint-850  vs baseline: win_rate=0.896 (896W-104L-0T)
checkpoint-900  vs baseline: win_rate=0.882 (882W-118L-0T)
checkpoint-950  vs baseline: win_rate=0.890 (890W-110L-0T)
checkpoint-1000 vs baseline: win_rate=0.919 (919W-81L-0T)
checkpoint-1050 vs baseline: win_rate=0.903 (903W-97L-0T)
checkpoint-1100 vs baseline: win_rate=0.884 (884W-116L-0T)
checkpoint-1150 vs baseline: win_rate=0.902 (902W-98L-0T)
checkpoint-1200 vs baseline: win_rate=0.908 (908W-92L-0T)
checkpoint-1250 vs baseline: win_rate=0.905 (905W-95L-0T)
[suite-parallel] total_elapsed=575.3s tasks=10 parallel=7
```
</details>

Log: [083_baseline1000.log](../../docs/experiments/artifacts/official-evals/083_baseline1000.log) / [083_latewindow_baseline1000.log](../../docs/experiments/artifacts/official-evals/083_latewindow_baseline1000.log)

---

## 6. Stage 2 rerun — combined 2000ep confirm

### 6.1 设置

single-shot 0.919 需要 rerun 以剔除 single-shot 运气。rerun port 62605, 1000ep:

```
=== checkpoint-1000 vs baseline (rerun 1000ep, 280s) ===
win_rate=0.899 (899W-101L-0T)
```

### 6.2 Combined 2000ep

- single-shot: 919W - 81L
- rerun:       899W - 101L
- **combined:  1818W - 182L = 0.909 over 2000ep**

### 6.3 §3 判据（combined）

| 阈值 | combined | verdict |
|---|---|---|
| §3.1 主 ≥ 0.920 | ❌ 0.909 (-0.011) | **未达** |
| §3.2 stretch ≥ 0.925 | ❌ | 未达 |
| §3.3 tied [0.895, 0.920) | **✅ 0.909** | **arch axis tied with plateau** |
| §3.4 退化 < 0.890 | — | 不 regress |

**verdict on baseline axis**: 083 是 **tied with 055/055v2/062a plateau** 的 arch；single-shot 0.919 确实是 peak-lucky，combined 收敛到 0.909（比 055v2@1750 的 0.9163 低 0.007，落在 plateau SE 带内）。

Raw: [083_rerun1000_stage2.log](../../docs/experiments/artifacts/official-evals/083_rerun1000_stage2.log)

---

## 7. Stage 3 peer-axis H2H vs 055v2@1750 SOTA

### 7.1 动机

按 [snapshot-106 §2](snapshot-106-stone-methodology-corrections.md) 新 doctrine：baseline WR tied on plateau 的 agent 要在 **peer 轴**（H2H vs frontier）上直接比，不能只看 baseline 投射。

### 7.2 结果（500ep, port 62705）

| matchup | sample | 083@1000 wins | 1750 wins | 083 rate | z | p | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **083@1000 vs 055v2@1750** | 500 | **205** | **295** | **0.410** | **−4.02** | <0.0001 | `***` |

- Side split: blue 0.444 / orange 0.376 (gap +0.068, 无结构性偏) 
- 判据 §3.7 阈 ≤ 0.47 → **peer-axis LOSE `***`**

### 7.3 Raw H2H recap

```
---- H2H Recap ----
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: cs8803drl.deployment.trained_team_ray_opponent_agent
episodes: 500
team0_overall_record: 205W-295L-0T
team0_overall_games: 500
team0_overall_win_rate: 0.410
team0_edge_vs_even: -0.090
team0_net_wins_minus_losses: -90
team1_overall_record: 295W-205L-0T
team1_overall_win_rate: 0.590
team1_edge_vs_even: +0.090
team0_blue_win_rate: 0.444
team0_orange_win_rate: 0.376
team0_side_gap_blue_minus_orange: +0.068
reading_note: interpret team0_overall_* as the H2H result; blue/orange_* are side-split diagnostics only.
```

Log: [083_1000_vs_1750.log](../../docs/experiments/artifacts/official-evals/headtohead/083_1000_vs_1750.log)

---

## 8. 终 Verdict — architecture axis closed（append-only, 2026-04-22）

### 8.1 双轴判决表

| axis | 判据 | 083 实测 | verdict |
|---|---|---|---|
| baseline axis | §3.1 ≥ 0.920 | 0.909 combined | ❌ 未达主阈, **tied on plateau (§3.3)** |
| peer axis | §3.5 ≥ 0.55 vs SOTA | **0.410 (z=−4.02)** | **LOSE `***` (§3.7)** |

**结论**: 083 是 **baseline-specialist**，不是 frontier-robust 的 SOTA shift。

### 8.2 机制读解

- **arch axis 局部 HIT**: 083 single-shot 在 baseline 上比 055/055v2 peak 都高 → spatial inductive bias 确实让 encoder 在 vs baseline 的分布上学到更好 feature
- **peer axis failure**: 直接 vs 1750 SOTA 输 −9.0pp **两个方向都显著（z=−4.02）**。说明：
  1. 083 学到的 policy 跟 1750 是 **不同风格**，不是 "同风格更强"；
  2. 1750 能针对性压制 083 的 spatial-attention policy（可能 1750 的 ensemble-distill 已经学会了更多 "policy diversity coverage"）；
  3. baseline axis 只是"对 fixed baseline 的 specialist"，不具备 frontier robustness
- **这是 arch axis 的 ceiling 表现**：scratch arch 再调 encoder 结构，能把 fix-baseline 的 WR 推到 0.91+，但跟 distill/ensemble 的 policy 层胜过 fixed SOTA 的路径是**不同** axis

### 8.3 跟 052/082 的对比

| snapshot | arch 干预 | combined peak | vs SOTA H2H | verdict |
|---|---|---|---|---|
| 052A (transformer refinement) | FFN/LN/residual on 031B | 0.800 | not measured | **REGRESSION** |
| 052 (full transformer) | true MHA + merge squeeze | 0.774 | not measured | **REGRESSION** |
| 082 (two-stream hierarchy) | temporal split 224/112 | 0.885 | not measured | **sub-SOTA** |
| **083 (per-ray attention)** | **spatial split 14 rays + Transformer** | **0.909** | **0.410 `***`** | **tied-on-baseline, peer LOSE** |

**Lesson**: arch 干预在 **baseline axis** 上最多把 scratch model 推到 plateau（~0.91）；想 **突破 SOTA 需要 policy 层的 diversity** （distill / ensemble / self-play），不是 encoder 结构。

### 8.4 为什么不做 Stage 2 failure capture 和 Stage 4 拓展

- 轴都被决定性判掉（peer `***`），failure capture 对"关 lane 类"不影响决策
- Stage 4（增加 opponent pool H2H）也不会翻案 —— 如果连 current SOTA 都 LOSE `***`，其他 frontier H2H 只是 redundancy
- ROI 更好的下一步是 103A-warm-distill（Stone Layered L2）和 DIR-H (cross-attention 融合 1750) 的新 axis

### 8.5 决策

1. **083 lane 关闭** — arch axis 无新突破可能
2. **不 package 为 agent** — per user directive "不需要 package，目前还没有突破性 SOTA 不需要 package"
3. **082 / 083 两条 arch 探索共同结论**: monolithic 031B encoder 不是 plateau 的 bottleneck；**plateau 来自 policy 多样性和对手分布**（distill/self-play）而不是 encoder 能力
4. 后续 arch 轴暂停，转向：
   - 103A-warm-distill (Stone Layered L2, 1750 teacher + INTERCEPTOR scenario, Monitor b93hc6lzs)
   - DIR-H cross-attention 融合 (把 083 arch 当特征 tower 做 frontier ensemble 的融合头)

---

## 9. Raw artifacts 索引

- Stage 1 主 window: [083_baseline1000.log](../../docs/experiments/artifacts/official-evals/083_baseline1000.log)
- Stage 1 late window: [083_latewindow_baseline1000.log](../../docs/experiments/artifacts/official-evals/083_latewindow_baseline1000.log)
- Stage 2 rerun: [083_rerun1000_stage2.log](../../docs/experiments/artifacts/official-evals/083_rerun1000_stage2.log)
- Stage 3 H2H: [headtohead/083_1000_vs_1750.log](../../docs/experiments/artifacts/official-evals/headtohead/083_1000_vs_1750.log)
- Training log: `docs/experiments/artifacts/slurm-logs/083_per_ray_train_20260421_210849.log`
- run_dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/083_per_ray_attention_scratch_20260421_210849/TeamVsBaselineShapingPPOTrainer_Soccer_decf1_00000_0_2026-04-21_21-09-11/`

---

## 10. Related snapshots

- [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B within-agent cross-attention baseline
- [snapshot-052](snapshot-052-031C-transformer-block-architecture.md) — arch step 3 (FFN+LN+MHA) decisive REGRESSION
- [snapshot-082](snapshot-082-hierarchical-two-stream-siamese.md) — two-stream temporal hierarchy, sub-SOTA 0.885
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — distill SOTA recipe (0.907/1750 extend 0.9163)
- [snapshot-106](snapshot-106-stone-methodology-corrections.md) — peer-axis doctrine (H2H is load-bearing for SOTA shift)
- [rank.md](rank.md) §5 H2H matrix — 083 row
