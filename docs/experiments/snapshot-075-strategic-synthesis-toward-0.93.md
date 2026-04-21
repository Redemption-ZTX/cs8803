## SNAPSHOT-075: Strategic Synthesis — 从 0.907 推向 0.93 Baseline WR

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: Synthesis / 路径梳理 (meta-snapshot, 不对应具体 lane; 作为 076/077/078 预注册的前置文档)
- **范围**: 综合 snapshot-028 ~ snapshot-074E (涵盖 team-level bootstrap / 架构 axis / reward axis / distillation axis / ensemble / curriculum) 以及 rank.md §3.3 / §5.3 的所有 combined 2000ep + H2H 数据。

---

## 0. Purpose + Scope

项目当前 frontier 单模型 SOTA = `055@1150` combined 2000ep **0.907**，distill from 034E ensemble。多条独立路径在 0.89-0.91 区间 saturate。本 snapshot 系统梳理过去 47 个 snapshot 的 arch / reward / distill / ensemble / curriculum 五条轴，回答:

1. **哪些改动真的产生 ≥+2pp 的 jump**, 哪些只是 "path alternative" 但没突破
2. **0.91 plateau 是什么性质的 ceiling** — arch / reward / distill / sample 各有没有未挖掘的余地
3. **下一步哪 3 条 lane 最可能在 ROI 意义下把 SOTA 推到 ≥ 0.92-0.93**

判据基线: **combined 2000ep (n=2000, SE ≈ 0.007)**。目标 0.93 = +0.023 over 0.907 ≈ 3.3 × SE — 需要 "decisive beat SOTA" 级别的真实机制增益，不是 noise。

---

## 1. Historical Growth Map

### 1.1 Architecture Axis (team-level 672-dim joint obs)

| Step | 架构 | 代表 lane | Combined 2000ep peak | Δ vs prev | 证据类型 |
|---|---|---|---:|---:|---|
| 0 | Flat MLP `[512,512]` | 028A@1060 | 0.783 (1000ep single) | base | warm BC + v2 |
| **1** | **Siamese dual encoder `[256,256]×2 + merge`** | **031A@1040** | **0.860 (2000-game avg)** | **+7.7pp** | scratch, H2H vs 028A = **0.568 `***`** |
| **2** | **+ within-agent token cross-attention (4 tokens × 64 dim)** | **031B@1220** | **0.880** | **+2.0pp** | scratch, H2H vs 031A = 0.516 NS; vs 029B = 0.584 `***` |
| 3a | + FFN + LN + residual (`052A`) | 052A | ~0.800 (1000ep) | **−8pp** 💀 | REGRESSION, LN 破坏 PPO gradient |
| 3b | + true MHA + 512-dim merge (`052`) | 052 | ~0.774 (1000ep) | −10.6pp vs 031B | REGRESSION |
| 3c | + cross-AGENT attention, V zero-init (`054`) | 054@1100/1230 | 0.880 (1000ep double peak) | 0pp (tied) | cross-arch attention 没贡献 |
| 3d | + cross-agent + pre-LN + FFN (`054M` MAT-med) | 054M@1230 | 0.889 (1000ep single-shot) | +0.009 NS | 进一步验证架构 axis saturated |

**结论**: 架构 axis 在 031B 0.880 触 ceiling。step 1 (flat → Siamese) 是项目**最大的单次跳跃**; step 2 (+ cross-attention) 是第二大。step 3 在 4 个子变体（transformer block / MAT-min / MAT-med / true MHA）全部 sub-noise 或 regression。**架构 axis 没有未挖掘的 +2pp**。

### 1.2 Reward / Shaping Axis

| 路径 | 代表 | Combined 2000ep | Δ vs sparse-only | 评注 |
|---|---|---:|---:|---|
| Sparse only (v2 ablation) | 031Bnoshape@1030 | **0.875** | base | v2 shaping 对 peak 几乎不贡献 (NS) |
| v2 dense hand-tuned shape | 031B@1220 | 0.880 | +0.5pp NS | 主要加速 convergence 不提升 peak |
| v2 + field-role binding | 025b@080 | 0.804 (1000ep) | — | early per-agent 路线 |
| v2 + PBRS goal-prox | 030D@320 | 0.816 (1000ep) | — | baseline-specific exploit |
| v2 + learned reward (W/L head) | 036D@150 | 0.860 (1000ep) | +0.5pp on per-agent | per-agent only, not transfer to Siamese |
| learned reward only (no v2) | 051D@740 | 0.889 (2000ep) | +1.4pp | 但 031A 基础上 H1 假设没更强 |
| v2 + outcome-PBRS (Ng99) combo | 053A@190 | 0.891 (2500ep) | +1.6pp | ≈ 034E ensemble 水平 |
| **Outcome-PBRS only (no v2)** | **053Dmirror@670** | **0.902 (1000ep single)** | **+2.7pp vs 031B-noshape** | warm from weak 031B@80, **纯 potential-based signal** |
| v2 + curriculum + adaptive (no shape) | 062a@1220 | 0.892 | +1.7pp vs 031B-noshape | 0.911 single-shot 下修到 0.892 combined |

**结论**: Reward axis 对 peak WR 的贡献 **集中在 "calibrated outcome predictor" 路线**, 而非手工 shaping。outcome-PBRS (053D) + distill 叠加 (068 pending) 是 reward axis 目前唯一真正 actionable 的增益源。

### 1.3 Distillation Axis

| Gen | Lane | Teacher pool | Combined 2000ep | Δ vs prev | 评注 |
|---|---|---|---:|---:|---|
| G0 | 031B@1220 (no distill) | — | 0.880 | base | |
| G0.5 | 034E ensemble (deploy-time avg) | {031B + 036D + 029B} | 0.890 (ensemble) | +0.010 | non-intelligence stability lift |
| **G1** | **055@1000/1150** | **034E 3-teacher {031B + 045A + 051A}** | **0.902 → 0.907** | **+0.017-0.027 over 031B base** 🥇 | **Student > teacher H2H 0.590 `***`** |
| G2 (LR变体) | 059 (055 + LR=3e-4) | = G1 | 0.898 (1000ep) | -0.009 | LR axis 在 distill 上 saturated |
| G2 (recursive) | 055v2@1000 | 5-teacher {G1 + 031B + 045A + 051A + 056D} | 0.909 (3000ep) | +0.002 NS | recursive 没有 decisive beat G1 |
| G2 (T sweep) | 055temp@1030 (T=2) | = G1 teacher, T=2.0 | 0.904 (1000ep) | -0.003 NS | T=2 对 Multi-Discrete RL 不贡献 |
| Pool B (diverse) | 070 {055 + 053Dmirror + 062a} | cross-mechanism | pending | — | _Expected plateau_ — 成员都已在 0.89-0.91 |
| Pool A (hom) | 071 {055 + 055v2 + 056D} | same family | pending | — | _Expected ≈ 055_ |
| Pool C (max-div) | 072 {055v2 + 056D + 054M + 062a} | 4-axis | pending | — | _054M cross-arch KL conflict risk_ |
| Pool D (cross-reward) | 073 {055 + 068 + 062a} | pure reward-path | pending 068 | — | PBRS + distill 理论上最干净 |
| 066A/B progressive | BAN stages | 055 self + weighted | pending | — | _Expected ≈ 055_ (self-distill saturation) |
| 068 PBRS + distill | 055 recipe + PBRS | 3-teacher G1 | pending | — | **最 promising 未验证 lane** |

**结论**: distill G0 → G1 是项目第二大单次跳跃 (+0.027)。G1 → G2 (recursive / LR / T / pool A/B/C) 各种 follow-up 全部在 ±1σ 内 tied。**G1 到 G2 的 ceiling 很可能在 0.91**, 除非改变 student arch 或 reward signal (068)。

### 1.4 Ensemble Axis (deploy-time)

| Lane | 组成 | Combined / 1000ep | Δ vs best single | Status |
|---|---|---:|---:|---|
| 034C | {029B + 025b + 017} per-agent | 0.843 (1000ep) | −0.005 vs 029B | non-lift |
| 034E | {031B + 036D + 029B} mixed | 0.890 | −0.017 vs 055 | stability but 055 distill > 034E by +9pp H2H |
| 074A | {055 + 053Dmirror + 062a} team-level-only | 0.903 | −0.004 vs 055 | tied (anti-lift) |
| 074B | {055 + 054M + 062a} arch-diverse | 0.877 | **−0.030 regression** | MAT cross-arch conflict |
| 074C | {055 + 056D + 053Dmirror} H2H-orth | 0.902 | −0.005 | tied |
| 074D | {055 + 055v2 + 053A_cont} failure-bucket | 0.900 | −0.007 | tied |
| 074E | 074A + predictor rerank | pending | — | **novel signal combining** |

**结论**: naive prob-averaging ensemble **已在 4 个独立 member-choice 方案下 replicate 了 "null lift" pattern**。team-level-only ensemble 没有正 lift (+0.003 peak, -0.030 worst)。Deploy-time prob-averaging 作为机制本身已耗尽; 唯一未测的 novel signal 方向 = predictor-based rerank (074E)。

### 1.5 Curriculum / Opponent Distribution Axis

| Lane | 设计 | Combined 2000ep | Δ vs 031B-noshape | 评注 |
|---|---|---:|---:|---|
| 058 simplified | fixed boundary + v2 shape | 0.842 | −0.033 | sub-SOTA 确认 |
| 062a | 0/200/500/1000 + adaptive gate + no-shape | 0.892 | +0.017 | recovery from 058 failure |
| 062c | 0/300/700/1100 + adaptive + no-shape | 0.886 | +0.011 | slower 比 baseline schedule 稍差 |
| 046 cross-train vs 031A | frozen 031A 100% | pending | — | — |
| 043 frontier pool | baseline 40% + frontier 60% | 0.900 (043A') / 0.893 (043B') | +0.018-0.025 | 但 peer H2H 仍输 055 |

**结论**: curriculum / opponent-distribution axis 大约能达到 **0.89-0.90 tier**, 但 peer H2H 显示仍然 lose to 055 decisively (058 vs 055 = 0.288 `***` negative). curriculum 不是 0.93 的 feasible path alone.

---

## 2. Big-Jump Anatomy — 每次 ≥+2pp 的真 jump

### 2.1 Jump 1: 028A (flat MLP) → 031A (Siamese) — **+7.7pp**

- **机制**: 把 672-dim joint obs 从 concat-MLP 改为 Siamese shared encoder (slot-invariant feature)
- **证据**: vs 028A@1060 H2H = 0.568 `***` (n=1000), 同 reward 同 base (team-level v2)
- **Caveat**: 031A 跑 1250 iter vs 028A BC+1060 iter; 架构 vs 训练时长未完全 disentangle
- **复刻难度**: 已用尽 — 架构 step 2 (+2.0pp) 已最后的架构 gain

### 2.2 Jump 2: 031A → 031B (within-agent cross-attention) — **+2.0pp (1000ep single)**

- **机制**: Siamese 基础上加 within-agent token cross-attention, 把 encoder 256-dim feature 拆成 4 tokens × 64 dim attend
- **证据**: 031B combined 2000ep 0.880 vs 031A 2000-game avg 0.860 = +2.0pp; H2H 031B vs 031A = 0.516 NS (direct H2H 不拉开, 暗示大部分 gain 来自 baseline-specialization 而非 general skill)
- **Caveat**: 架构 step 3+ 全失败, cross-attention 机制就是 architecture ceiling

### 2.3 Jump 3: 031B → 055 distill from 034E — **+2.7pp**

- **机制**: 用 {031B + 045A + 051A} 3-way ensemble avg probs 作为 KL target distill 进 031B-arch student, 同时保留 v2 shaping + PPO env reward
- **证据**: 055@1000 combined 2000ep 0.902 vs 031B 0.880 = +2.2pp; 055@1150 combined 0.907 = +2.7pp (both rerun-verified); H2H **student > teacher** by +9pp (vs 034E = 0.590 `***`, Hinton 2015 pattern)
- **Decisive peer beat**: vs 031B = 0.638 `***`, vs 029B = 0.696 `***`, vs 028A = 0.750 `***`, vs 025b = 0.702 `***`
- **机制 insight**: ensemble 的 "非智力 stability" 被 KL distill 转成 single-model "真智力" — 这条 jump 是项目最 load-bearing 的一次，建议基于它继续走 distill 家族

### 2.4 Jump 4 (候选): 031B → 053Dmirror (PBRS-only) — **+2.2pp on 1000ep**

- **机制**: 丢掉 v2 hand-tuned shaping, 用 calibrated outcome predictor 的 ΔV(s) 作为 PBRS bonus (λ=0.01, Ng99 potential-based)
- **证据**: 053Dmirror@670 1000ep 0.902 (single-shot, combined 2000ep 未做) vs 031B 0.880 = +2.2pp
- **Caveat 重大**: combined 2000ep 未 verified; single-shot 1000ep 在本项目历史上 5 次 overestimate by 0.005-0.038pp (058/062a/062c/055v2 correction pattern)
- **真实 peak 估** ≈ 0.89-0.90 (与 053A@190 combined 0.891 一致, 两者同 reward-path family)

### 2.5 Non-Jump (规模 <2pp) — 所有其他 "improvement"

- Role binding (+0.000), PBRS goal-prox (+0.002), opp-pool 029C (+0.006), learned reward per-agent (+0.014), v2 ablation → with-v2 (+0.005 NS), curriculum on 031B (+0.010), LR sweep 056D (+0.011 NS), MAT-arch (+0.009 NS), temperature sweep (-0.003 NS), recursive distill 055v2 (+0.002 NS)

**Key insight**: 项目真正的 jump 大小分布 highly skewed — **2 次 ≥+2pp (031A 架构; 055 distill)**, 其余 40+ lane 全 <+2pp。说明 0.93 目标需要**再找一个 ≥+2.3pp 机制**, 不是继续堆 <1pp 的 axis。

---

## 3. Dead Ends / Tied Paths — 已关闭的 lane

| Lane | 机制 | 结果 | Lesson |
|---|---|---|---|
| 052 transformer block | LN + FFN + residual on 031B | -8~-11pp REGRESSION | LayerNorm × PPO gradient conflict; transformer 对 RL 不是 free lunch |
| 054 MAT-min (V zero-init) | cross-agent attention residual | 0pp tied 031B | agent-to-agent 信号没 emergent value, 架构 step 3 机制本身死 |
| 054M MAT-medium | + LN + FFN on 054 | +0.009 NS | 重试失败, 架构 axis 彻底封 |
| 063 T sweep (T=2.0) | softer distill target | -0.003 NS | Hinton T=2-4 不适用 Multi-Discrete RL; T-sweep 整个路径否决 (不测 T=1.5/2.5/3.0) |
| 059 LR=3e-4 on distill | = 055 + 056D winner LR | -0.009 NS | LR axis on top of distill saturated; HP-sweep axis 不叠加 |
| 074B arch-diverse ensemble | {055 + 054M + 062a} | -0.030 REGRESSION | deploy-time prob-avg 要求 arch 同质 |
| 074A/C/D team-level ensemble | 3 个不同 Siamese blood | tied (±0.005) | prob-avg team-level-only mechanism 用尽 |
| 055v2 recursive distill | 5-teacher incl. 055 self | +0.002 NS | self-distill saturation; teacher pool 3→5 无 compound |
| 066 BAN progressive distill | 055 self-distill stage 2 | pending but expected ≈ 055 | Furlanello 2018 pattern 在 RL 上可能不成立 |
| 058 simplified curriculum | fixed boundary + v2 shape | -0.038 sub-SOTA | simplification 伤 curriculum path; 但 062a 上调后 +0.050 recovery |
| 057 RND intrinsic | curiosity reward | snapshot-only, not run | 低优先, explore 不是当前 bottleneck |
| 046 cross-train vs 031A | scratch vs frozen 031A 100% | snapshot-only | snapshot-043 diversity + snapshot-055 distill 更有效率 |
| 039 AIRL adaptive reward | discriminator reward | 0.843 (1000ep) | baseline-mimicking path 理论上限 = baseline |

---

## 4. Past Simplifications That Cost Us

| Simplification | 出处 | 是否 cost | 如何 |
|---|---|---|---|
| 054 drop FFN (只加 cross-agent attn + zero-init V) | snapshot-054 §2.2 | **没 cost** | 054M 补做后仍 0.889 NS, 所以 drop FFN 是正确简化 |
| 058 drop adaptive gate (simplified 第一版) | snapshot-058 §4.1 | **cost -0.050pp** | 062a 加回 adaptive gate 后 +0.050pp recovery |
| 055 drop DAGGER iterative distill (Online only) | snapshot-055 §4.1 | **暂无 cost**, 055 已 0.907 | 但 Pool A/B/C/D 未来可能受 online distribution-shift 限制 |
| 055 drop α / T sweep | snapshot-055 §4.2 | **没 cost** | T sweep (063) 独立验证 T=1 是正确 |
| 055 drop 4-way ensemble teacher (只用 3-way) | snapshot-055 §4.3 | **微小 cost** | 055v2 (5-teacher) 只+0.002 NS, 说明 teacher count 3→5 不重要 |
| 057 RND 从未跑 | snapshot-057 | unknown | intrinsic motivation 理论上可解 "late-game policy diversity" bottleneck, 但未验证 |
| 046 cross-train never executed | snapshot-046 | unknown | 不能确认 "PPO ceiling is local optimum" 假设, 056D lr=3e-4 winner 只用了 HP axis |
| Student arch 固化为 031B-arch | 055 / 061 / 068 / Pool A-D | **possibly cost** | **所有 distill 路径 student 都是 031B-Siamese-cross-attn; 没测过 "更大 capacity student" 或 "per-agent student" 的 distill** — 这是 §5.1 / §5.4 的核心未挖掘 axis |

**Critical gap (用户角度)**: 所有已 execute 的 distill lane student = 031B-arch (0.46M params). 如果 0.91 ceiling 来自 student capacity, 扩大 student 是最直接 test。Hinton 2015 显示 student capacity 与 teacher ensemble complexity 的比例影响 final gain。

---

## 5. Unexplored Axes Worth Trying

按 "expected +pp × feasibility" 降序:

### 5.1 (★★★★★) **Larger Student Capacity for Distillation**

- **现状**: 所有 distill lane (055, 055v2, 059, 068, 066, Pool A-D) student 都是 **031B-arch 0.46M params**. 3-way ensemble teacher 总 capacity ~1.4M, 比 student 大 3×
- **Hypothesis**: student capacity 不足以 represent 3-teacher joint policy → 0.91 ceiling 部分来自 student bottleneck
- **Design**: 扩 encoder `[256,256]` → `[384,384]` 或 `[512,512]`, merge `[256,128]` → `[384,256]`, 参数量 0.46M → ~1.2M (仍 < 3-teacher total, Hinton 2015 范围内)
- **Risk**: 参数膨胀 × PPO 可能触 052 过拟合; PPO 对 hidden size 更 sensitive 于 image classification
- **Expected**: +0.005-0.020pp, **P(≥0.92) ≈ 35%, P(≥0.93) ≈ 10%**
- **Cost**: 1 lane × 14h

### 5.2 (★★★★☆) **PBRS-reward + Distill Combo — snapshot-068 已启动**

- 已在 Pool 目录里; 最 promising 原因: 055 + 053Dmirror 两条 +2.2pp+ jump 从未叠加测过 (唯一组合)
- Expected: +0.005-0.015, **P(≥0.92) ≈ 40%, P(≥0.93) ≈ 15%** (如果 068 single-shot 已过 0.91, combined 验证后可能 ≥ 0.92)
- **Note**: 已在 queue 中, 不作为 076/077/078 proposal material (per user constraint)

### 5.3 (★★★★☆) **Per-agent Student 吸收 team-level Teacher Distill**

- **现状**: 056D / 055 / 031B 全部 team-level Siamese 架构; 029B / 036D per-agent 架构被 034E 作为 teacher, 但 student 从未是 per-agent
- **Hypothesis**: per-agent student 的 336-dim obs 独立观察 + Siamese-shared 天然 robust, distillation + per-agent capacity 可能 escape team-level plateau
- **Design**: 在 MAPPO / per-agent shared-policy 上 distill 034E ensemble 的 per-agent marginal probs
- **Risk**: 工程改动 — 需要新的 per-agent distill marginal projection (已有 team 方向, 未有 per-agent 方向)
- **Expected**: +0.000-0.015, P(≥0.92) ≈ 25%, P(≥0.93) ≈ 8% — reward-side 不改, 单靠 student arch 切换 gain 有限

### 5.4 (★★★★☆) **Student Architecture Orthogonal to Teacher — Dilated Conv / Transformer-small**

- **现状**: 全部 student = Siamese + cross-attn (031B). 054M MAT 作为 teacher 或 student 全部 KL-tied (074B) 或 NS (054M self)
- **Hypothesis**: orthogonal arch student 学 teacher action distribution 可能打破 representation redundancy
- **Design**: 试 dilated 1D conv on obs (2 heads: self / teammate), 或 small transformer 2-layer + 128 dim (比 052 轻量)
- **Risk**: 高 — 052 already -10pp; 小 transformer 可能也 regression
- **Expected**: P(≥0.92) ≈ 15%, P(≥0.93) ≈ 5% — 高风险 探索

### 5.5 (★★★☆☆) **DAGGER Iterative Distillation**

- **现状**: 所有 distill 用 online (teacher 看 student rollout). 055 §4.1 明示是 simplification
- **Hypothesis**: student-teacher state distribution shift 限制 0.91 ceiling. DAGGER step 让 teacher 在 student 真 rollout state 上 label
- **Design**: 分 3 stage, stage boundary pauses ~2000 rollout episodes collect + supervised KL on stored buffer
- **Risk**: 工程 ~5 天, 边际 ROI 未知
- **Expected**: P(≥0.92) ≈ 20%, P(≥0.93) ≈ 8%

### 5.6 (★★★☆☆) **Predictor-Enhanced Train-Time Rerank (not just deploy)**

- **现状**: 074E 只 deploy-time rerank. Train-time 用 predictor 作为 action preference rerank 没 snapshot
- **Design**: 每 rollout step 用 predictor V(s) + V(s) × action 的 gradient 生成 action advantage bonus, 加到 PPO objective
- **Risk**: 设计未落地, 需要 2 天 engineering; PPO 和 predictor signal 混合的 variance 未知
- **Expected**: P(≥0.92) ≈ 15%, P(≥0.93) ≈ 5%

### 5.7 (★★☆☆☆) **Auxiliary State-Value Prediction Loss**

- **现状**: 项目没试过 aux loss 在 distill 上 (032 aux loss 是早期 per-agent)
- **Design**: student 除 policy / value head 外, 多加一个 winning-probability head, 用 episode outcome supervise
- **Expected**: P(≥0.92) ≈ 10%, P(≥0.93) ≈ 3%

### 5.8 (★★☆☆☆) **Long-Training + Annealing LR 065/066 extend**

- **现状**: 055 1000-1200 iter plateau [0.89, 0.91]. 056extend resume 1510 iter. 未 try 2500+ iter 看是否真 converge
- **Design**: resume 055 or 068 at 1250, extend to 2500 iter with LR anneal 1e-4 → 3e-5
- **Expected**: P(≥0.92) ≈ 15%, P(≥0.93) ≈ 5%

---

## 6. Proposed 3 Main Directions (076 / 077 / 078)

从 §5 挑选 top 3 按 expected-value × feasibility:

### DIR-A (snapshot-076) — **扩容 student 的 ensemble distill (Wide Student)**

- **Core**: 055 recipe 完全保留, 只把 student 架构从 031B-arch (0.46M) 扩到 "wide-031B" (~1.0-1.2M)
- **Hypothesis**: student capacity 不足是 0.91 ceiling 的必要原因之一; 扩 3× 可回收 +0.005-0.020 teacher-student capacity 差距
- **Threshold**: combined 2000ep ≥ 0.920 (§3.2)
- **P(≥0.92) ≈ 35%, P(≥0.93) ≈ 10%**
- **Why #1**: 最小 design 改动, teacher infra 完全复用; 如果失败 direct 证明 "0.91 不是 capacity-bound"; 如果成功可以 stack PBRS (068) 和 recursive (061)

### DIR-B (snapshot-077) — **Per-agent Student Distill (Architecture-Orthogonal Student)**

- **Core**: 用 per-agent MAPPO/shared-policy student 吸收 034E ensemble teacher 的 per-agent marginal probs
- **Hypothesis**: team-level Siamese 已 saturate; per-agent 的 336-dim egocentric obs + independent policy rollout 提供 orthogonal representation bias
- **Threshold**: combined 2000ep ≥ 0.915 (§3.2), stretch ≥ 0.925
- **P(≥0.92) ≈ 25%, P(≥0.93) ≈ 8%**
- **Why #2**: 项目里 per-agent SOTA 029B 0.846 被 team-level 0.907 碾压, 但**原因是 reward+arch 混合**; 这个 lane isolate arch 变量。中等风险, 需新 per-agent distill marginal projection (~1-2 天工程)
- **Stop condition**: 如果 per-agent student 600 iter 仍未超 029B@190 0.846, 早关

### DIR-C (snapshot-078) — **DAGGER-style Two-Stage Distill (State-Distribution Matching)**

- **Core**: online distill 两阶段 — phase 1 iter 0-600 online (= 055), phase 2 iter 600-1250 DAGGER (teacher labels student rollout states)
- **Hypothesis**: 055 ceiling 部分来自 student-teacher state distribution shift; DAGGER 把 teacher move 到 student 真 state 分布, 提供更 reliable KL target
- **Threshold**: combined 2000ep ≥ 0.918
- **P(≥0.92) ≈ 20%, P(≥0.93) ≈ 8%**
- **Why #3**: 理论上 RL distillation 最应该做 DAGGER (Ross 2011, Sun 2019), 055 §4.1 L2 本来就是 planned follow-up。**工程 3-5 天**, risk medium。Stop: stage 2 第一次 buffer supervision 结果若 tied stage-1 peak, 终止

---

## 7. Appendix — Cross-References

- Architecture axis: [snapshot-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) / [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) / [snapshot-052](snapshot-052-031C-transformer-block-architecture.md) / [snapshot-054](snapshot-054-mat-min-cross-agent-attention.md) / [snapshot-060](snapshot-060-054M-mat-medium.md)
- Reward axis: [snapshot-021](snapshot-021-actor-teammate-obs-expansion.md) (v2 origin) / [snapshot-030](snapshot-030-team-level-advanced-shaping-chain.md) / [snapshot-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) / [snapshot-036D](snapshot-036d-learned-reward-stability-fix.md) / [snapshot-045](snapshot-045-architecture-x-learned-reward-combo.md) / [snapshot-051](snapshot-051-learned-reward-from-strong-vs-strong-failures.md) / [snapshot-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) / [snapshot-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) / [snapshot-068](snapshot-068-055PBRS-distill.md)
- Distillation: [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) / [snapshot-059](snapshot-059-055lr3e4-combo.md) / [snapshot-061](snapshot-061-055v2-recursive-distill.md) / [snapshot-063](snapshot-063-055-temp-sharpening.md) / [snapshot-066](snapshot-066-progressive-distill-BAN.md) / [snapshot-070](snapshot-070-pool-B-divergent-distill.md) / [snapshot-071](snapshot-071-pool-A-homogeneous-distill.md) / [snapshot-072](snapshot-072-pool-C-newcomer-frontier.md) / [snapshot-073](snapshot-073-pool-D-cross-reward.md)
- Ensemble: [snapshot-034](snapshot-034-deploy-time-ensemble-agent.md) / [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) / [snapshot-074B](snapshot-074B-arch-diversity.md) / [snapshot-074C](snapshot-074C-h2h-orthogonal.md) / [snapshot-074D](snapshot-074D-failure-bucket-orthogonal.md) / [snapshot-074E](snapshot-074E-predictor-rerank.md)
- Curriculum / Opponent: [snapshot-043](snapshot-043-frontier-selfplay-pool.md) / [snapshot-046](snapshot-046-cross-train-pair.md) / [snapshot-058](snapshot-058-real-curriculum-learning.md) / [snapshot-062](snapshot-062-curriculum-noshape-adaptive.md)
- Canonical rank table: [rank.md §3.3](rank.md#33-official-baseline-1000frontier--active-points-only) + [§5.3](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的)

---

## 8. Honest Assessment of 0.93 Goal

- **0.91 plateau 的本质**: 5 条独立 path (distill / recursive / curriculum / LR sweep / PBRS) 已 converge 到 0.89-0.91 区间; 不是 single-lane stuck, 而是 **当前 arch+reward+data 组合的真 ceiling**
- **0.93 需要**: +0.023 over 055 SOTA 0.907 = 3.3 × SE (2000ep); 即使 +0.020 gain 也要 3 sample 独立 replicate 才 decisive
- **最可信的加法剂量**: 一个 "机制级" 改动 (非 HP / non-axis-combo) + 一个 "信号级" 改动 (reward 或 data 路径) — 这是项目历史上 jump 1 (架构) + jump 3 (distill) 的复合 pattern
- **如果 DIR-A/B/C 全失败的诚实 readback**: 0.907 可能就是这 arch+env+ reward 组合的真 ceiling; 突破需要换 env wrapper (frame stack / obs augmentation / LSTM) 或调 baseline env (impossible per assignment); 此时建议 declare 0.907 为 final submission, 专注 report + presentation 质量

---

## 9. 2026-04-21 18:30 EDT post-hoc UPDATE — 055v2_extend@1750 = NEW SOTA 0.916, 0.91 ceiling partially invalidated (append-only)

**VERDICT**: §2 "0.91 ceiling" claim is **partially invalidated by 2026-04-21 18:30 finding**. A path past the ceiling exists — it is NOT env/arch-bound as §8 honest readback feared. See [snapshot-061 §7.4](snapshot-061-055v2-recursive-distill.md#74-2026-04-21-1830-edt--055v2_extend1750--new-project-sota-combined-3000ep-09163-append-only).

**The finding**:
- `055v2_extend@1750` combined 3000ep = **0.9163** (CI [0.906, 0.926], SE ≈ 0.005)
- Δ vs 055@1150 SOTA 0.907 = **+0.009** (z ≈ 1.23, p ≈ 0.11 one-sided — approaching marginal significance but not yet p<0.05)
- Combined 4000ep rerun pending for final p<0.05 confirmation (jobid 5032909, ETA ~10 min from 18:32)

**What changed relative to §2-§3 assumptions**:
- §2 "5 条 path converge 到 0.89-0.91 真 ceiling" is correct for those 5 paths BUT **055v2_extend opened a 6th path that is above 0.91**
- §3 DIR-A (wide-student) and DIR-C (DAGGER) are still open experiments; the 055v2_extend finding **adds a 4th productive direction not in the original §4-§6 plan**: **"extend recursive distill past teacher's own MAX_ITERATIONS horizon"**
- The key distinction: 1750 is **+50 iter past the original MAX_ITERATIONS=1216 of 055v2 teacher training window**, and **~160 iter past the TRUE teacher peak at 055v2@1000**. The gain appeared in the **late-extend** window, not in the original training horizon.

**New direction (DIR-D, informal, not yet pre-registered as snapshot-080)**:
- **Extend past teacher's own peak** — apply to every in-progress distill lane (076 wide-student, 077 per-agent, 079 055v3) by letting training run +500-800 iter past what teacher's training cut off at
- **Hypothesis**: student in recursive/Gen-2+ distill setups has different convergence dynamics than teacher-training setup; the "natural" training horizon of teacher is NOT the right horizon for student
- **Evidence level**: n=1 validated lane (055v2_extend). Need ≥ 2 independent lanes (076 / 077 / 079) to generalize. This re-opens DIR-A (076 wide-student) + DIR-C (078 DAGGER) with the explicit addition of "extend past teacher horizon" to each.

**Distinctions that matter (precise numbers)**:
- 1750 is **+50 iter past original MAX_ITER=1216** (the MAX_ITERATIONS of 055v2's own training)
- 1750 is **~160 iter past the TRUE teacher peak 055v2@1000** (which is the peak, not the MAX_ITER)
- 1830 (the §7.3 putative single-shot peak 0.923) collapsed to combined 0.910 upon rerun — biggest mean reversion of 8 peaks (-0.013pp); was single-shot noise
- 6 of 8 §7.3 peaks all mean-reverted to 055 plateau (-0.003 to -0.013pp). Only 1750 survived.

**Implications for §2 "0.91 ceiling"**:
- The ceiling was **real for the 5 paths as exercised, but not for extend-past-teacher-horizon recursive distill**
- 0.93 goal updated probability:
  - With DIR-A/B/C alone (original §6 plan): P(≥0.92) ≈ 25-35%, P(≥0.93) ≈ 5-10%
  - With DIR-D ("extend past teacher horizon") stacked on DIR-A/B/C: P(≥0.92) ≈ 45-55%, P(≥0.93) ≈ 15-20% (speculative — n=1 validation so far)
- **0.907 is NO LONGER the final submission candidate** pending combined 4000ep confirmation → upgrade to **055v2_extend@1750 combined 3000ep 0.9163** (or combined 4000ep if rerun completes)

**Downstream action items**:
- 076/077/079 should include "extend past MAX_ITER" as standard Stage 2 follow-up after Stage 1 baseline eval
- 074 ensemble family (all closed) could be reopened with 055v2_extend@1750 swapped in for 055@1150 (zero-training cost, worth one retry)
- Pool A used 055v2@1000 as teacher — consider upgrading to 055v2_extend@1750 (Pool A rerun)

