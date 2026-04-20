# SNAPSHOT-050: Cross-Student DAGGER Probe (Hybrid Takeover with Stronger-Teacher 替换 baseline)

- **日期**: 2026-04-19
- **负责人**:
- **状态**: Phase 1.1 + 1.3 (100ep + 500ep) 完成；Phase 2 暂缓（信号 borderline，等 045 capture / 051 1000ep verdict 重排优先级）

## 0. 为什么做

[SNAPSHOT-048 verdict](snapshot-048-hybrid-eval-baseline-takeover.md#74-strategic-结论) 明确区分：

> **"DAGGER-from-baseline 路径死"，但 DAGGER 框架本身没死**。
> - 已死: ceia_baseline_agent 作为 teacher（baseline self-play WR ≈ 0.50, student 0.86 已 outgrown）
> - 仍存活: 031B@1220 (0.882) / 034 ensemble / agent2 作为 teacher

snapshot-050 是 048 的延伸——把 baseline 替换成**更强的 teacher (frontier checkpoint)**，看 takeover 的 Δ 能否变正。这是真正回答「DAGGER 框架本身能不能 work」的实验。

复用 048 的 plumbing:
- 同一个 `evaluate_hybrid.py`（已加 `--takeover-{module,checkpoint}` flags 让 takeover policy 可参数化）
- 同 trigger 设计（α/β window-based，[snapshot-048 §2.2](snapshot-048-hybrid-eval-baseline-takeover.md#22-trigger-函数)）
- 同 100ep / 1000ep eval 框架

## 1. 核心假设

### H1（teacher quality → takeover Δ 单调）

> takeover Δ vs student-solo 与 teacher WR 强度正相关。048 baseline (0.50) 给 -7 到 -14pp；越强 teacher 应该给越接近 0 或正的 Δ。

### H2（不同 failure mode 的 teacher 互补性）

> 在 student 的失败 state 上，**failure mode 跟 student 不同**的 teacher 比 **failure mode 相同**的 teacher 更可能给正 Δ。
> - 031A wasted_possession-主导 + 036D defensive_pin-主导 = 互补
> - 031A vs 031B (同为 wasted_possession-主导) = 不互补

### H3（031B as universal teacher）

> 031B 是项目当前最强 (1000ep 0.882)，应该比 031A 在所有 state 上更强 (gain +2.2pp)。但 [snapshot-031 §13.8](snapshot-031-team-level-native-dual-encoder-attention.md#138-stage-3--h2h-matrix-4-matchups-n500-each-2026-04-19-0742) H2H vs 031A = 0.516 NOT sig 暗示 peer-axis 上 031B 没明显更强。所以 031B as teacher 给 031A 的 Δ 大概率也 sub-noise。

## 2. Phase 1 结果（2026-04-19, 036D → 031A）

### 2.1 设置

- Student: 031A@1040 (team-level Siamese, 1000ep 0.860)
- Teacher: 036D@150 (per-agent + learned reward, 1000ep 0.860)
- Trigger: α (window mean ball_x < -3.0) / β (window mean < -5.0)
- Episodes: 100 each (Phase 1 = sanity probe; 1000ep 仅在 Δ ≥ +0.02 时启动)
- Node: atl1-1-03-010-30-0, ports 63401/63421/63441

### 2.2 数据

| condition | trigger | swap_pct | WR (100ep) | side blue/orange | Δ vs C0 |
|---|---|---:|---:|:---:|---:|
| **C0_031A_solo** | none | 0% | **0.83** | 0.90 / 0.76 | ref |
| C1_031A_α_teacher036D | α (window<-3) | 24.7% | **0.84** | 0.82 / 0.86 | **+1pp** |
| C2_031A_β_teacher036D | β (window<-5) | 19.3% | **0.84** | 0.86 / 0.82 | **+1pp** |

### 2.3 verdict (Phase 1.1)

按 [048 §3.1 判据](snapshot-048-hybrid-eval-baseline-takeover.md#31-主判据-dagger-上限):

| Δ | Phase 1.1 | 解读 |
|---|---|---|
| ≥ +0.03 | ❌ | 不达启动 DAGGER 阈值 |
| (+0.01, +0.03) | ⚠️ 边缘 | 加 sample 缩窄 SE |
| (-0.01, +0.01) | **✅ 正中** | **基本无影响** |
| ≤ -0.01 | ❌ | 路径死 |

**Phase 1.1 = 036D as teacher 对 031A NEUTRAL** (+1pp sub-noise)。比 baseline takeover 的 -7~-14pp 改善 +8~+15pp（teacher quality 量级 jump），但**不构成 cross-student DAGGER 的有效证据**。

### 2.4 关键观察

1. **036D 不再是 distractor** — 比 baseline 改善 +8~+15pp 量级。证实 H1（teacher quality 与 takeover Δ 单调）部分方向
2. **swap_pct 24.7% / 19.3% 比 048 同 trigger 高** (048 alpha=17.2% / beta=10.4% on 031A) — 暗示 036D 接管后球**反而更深陷我方半场** (036D defensive style → 持续 fire trigger)。这是 H2 互补性的一个反向 signal: 036D 的 defensive style 在 031A 的 trigger states 上**没改善 ball position**
3. **side gap 反转**: solo blue 0.90 / orange 0.76 (-14pp gap) → α/β 都把 orange 拉起来 (0.82-0.86) blue 略降。**036D 在 orange side 帮了 031A**，blue 略压低 — 不知道为什么 (可能 spawn-asymmetric 训练 artifact)

## 3. Phase 1.2 + 1.3 (进行中, 2026-04-19)

### 3.1 Phase 1.2: 031B → 031A (架构内 stronger teacher)

- Student: 031A@1040 (0.860)
- Teacher: 031B@1220 (**0.882**, 项目当前 SOTA)
- 节点: atl1-1-03-011-23-0, ports 64401/64421/64441
- tmux: `phase1_2_031B_to_031A`

**预测**: H3 — 031B vs 031A peer H2H = 0.516 NS 暗示 peer-axis 不显著强 → takeover Δ 大概率也 sub-noise (类似 Phase 1.1)。但 baseline-axis +2.2pp 可能 transfer 一部分到 trigger states → 弱方向性 +Δ 可能。

### 3.2 Phase 1.3: 031A → 036D (failure mode 互补的反向 pair)

- Student: 036D@150 (0.860, defensive_pin 主导)
- Teacher: 031A@1040 (0.860, wasted_possession 主导)
- 节点: atl1-1-03-015-30-0, ports 65401/65421/65441
- tmux: `phase1_3_031A_to_036D`

**预测**: H2 (互补) 是这条 lane 的核心。036D 的 defensive_pin 失败 state，031A 不擅长那种 state（031A 也大量 defensive_pin loss），但 031A 的 attacker style 也许能 break out of pin。**这是理论上最 promising 的一对**。

## 4. 后续计划（Phase 2+）

### Phase 2 启动条件

任一 Phase 1.x Δ ≥ +0.03 → 启动 Phase 2 (真正的 DAGGER training)

Phase 2 设计 (待 Phase 1 verdict):
- 取 Δ 最大的 pair (teacher, student)
- 训练 student PPO, 在 trigger state 加 BC loss = MSE(student_logits, teacher_logits) 或 KL(student || teacher)
- 控制实验: 同设置但 BC loss 关掉
- 1000ep eval, 看 student 是否 internalize teacher 在 trigger state 的优势

如果**所有 Phase 1.x Δ < +0.02** → cross-student DAGGER 整条路径关闭。

### Phase 1.4 (条件): ensemble teacher

待 [034 ensemble](snapshot-034-deploy-time-ensemble-agent.md) 完成且 ≥ 0.88 → 跑 ensemble as teacher → 各 student。Ensemble 是最强 teacher 候选。

## 5. 不做的事

- **不在 Phase 1 跑 1000ep** — 100ep SE ±0.05 已足够 distinguish "Δ +1pp NEUTRAL" vs "Δ +5pp 启动 Phase 2"
- **不写 DAGGER training 代码**直到 Phase 1 给出 +Δ 信号
- **不复用 049 编号** — 049 已是 env state restore investigation; baseline-DAGGER training 也已 in 048 否决；本 snapshot 用 050

## 6. 相关

- [SNAPSHOT-048](snapshot-048-hybrid-eval-baseline-takeover.md) — DAGGER-from-baseline verdict; 本 snapshot 是 §7.4 strategic 结论的延伸
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) — ensemble teacher 候选
- [SNAPSHOT-031 §13.8](snapshot-031-team-level-native-dual-encoder-attention.md#138-stage-3--h2h-matrix-4-matchups-n500-each-2026-04-19-0742) — 031B vs 031A H2H = 0.516 NS evidence
- [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md) — baseline self-play WR ≈ 0.50 数据来源
- v2 桶: 031A wasted_possession 主导 (42%) vs 036D defensive_pin 主导 (55%) — 失败模式正交证据
- [evaluate_hybrid.py](../../cs8803drl/evaluation/evaluate_hybrid.py) — 已加 `--takeover-module/--takeover-checkpoint` 参数化 takeover teacher

## 7. Phase 1.3b verdict — 500ep rerun on free node (2026-04-19, append-only)

### 7.1 触发原因

Phase 1.3 100ep 报 Δ +10/+12pp 看似强信号，但 C0_036D_solo = 0.74 是 1.5σ 低 outlier (036D 真 1000ep = 0.860, 100ep SE ±0.05)。Phase 1.3b 把 3 condition 重跑 500ep 缩窄 SE，验证真信号大小。

### 7.2 设置

- Student / Teacher / Trigger 同 Phase 1.3
- Episodes: **500 (each)** — SE ±0.017 (vs 100ep ±0.05)
- Node: atl1-1-03-011-23-0 (Phase 1.3 同节点 rerun)
- artifact: `docs/experiments/artifacts/cross-student-hybrid/P13b_*.json`

### 7.3 数据

| condition | trigger | swap_pct (overall/per-ep) | WR (500ep) | side blue/orange | Δ vs C0 |
|---|---|---:|---:|:---:|---:|
| **C0_036D_solo** | none | 0% / 0% | **0.830** | 0.876 / 0.784 | ref |
| C1_036D_α_teacher031A | α (window<-3) | **29.7% / 23.3%** | **0.862** | 0.872 / 0.852 | **+0.032 (1.9σ)** |
| C2_036D_β_teacher031A | β (window<-5) | 15.7% / 12.9% | **0.846** | 0.852 / 0.840 | +0.016 (0.94σ) |

### 7.4 verdict — borderline（双 baseline 解读分裂）

**实验内对照** (用本实验 C0=0.830 当 reference):
- Δ_C1 = **+0.032** ≈ **正好压在 §4 Phase 2 启动阈值 +0.03**, 1.9σ 边缘显著
- Δ_C2 = +0.016, 1σ sub-noise

**跨实验对照** (用 036D 真 1000ep 0.860 当 reference):
- Δ_C1 = +0.002 sub-noise
- Δ_C2 = -0.014 轻微退化
- 因 C0=0.830 比 036D 平台 -3pp, 是 n=500 的 1.5σ low draw（仍在 95% CI 内）

→ **真信号区间 [+0.002, +0.032]，C1 仅在 C0 漂低的运气下卡 +0.03 trigger**。不构成稳健 cross-student DAGGER 启动证据。

vs Phase 1.3 100ep (C0=0.74 outlier 时报 Δ +10/+12pp): 信号缩水 4-5×, 与 outlier 假设吻合。

### 7.5 §4 Phase 2 启动决策 — **暂缓不启动**

按 §4 阈值, C1 alpha trigger 严格 hit +0.03，机械上够 trigger Phase 2。但：

1. **信号统计弱** (1.9σ, 边缘)，且对 baseline 选择敏感（用 036D 1000ep 算 sub-noise）
2. **Phase 2 工程成本高** (≥6h GPU + 一条新 lane + DAGGER training infra)
3. **机会成本**: 同期还有 045 capture/H2H、051A/B 1000ep、052A 031C-min、046E 三条更高 ROI 的 in-flight lane
4. **可逆性**: Phase 1.3b 已经持久化, Phase 2 启动可以延后到 045/051 verdict 出来后再排

**决策**: snapshot-050 lane 暂缓 Phase 2 启动, 标记 borderline 不关闭。如果 051C-annealed (reward curriculum) 路径死, **回头看 Phase 2 vs 034 ensemble 哪个 ROI 更高**。

### 7.6 H1 / H2 / H3 占位 — 见 §7.8 Phase 1.2 + 1.3b 综合读图

### 7.7 Phase 1.2 数据（100ep, 已完成）— H3 + H2 综合读图

Phase 1.2 (031B → 031A) 100ep 已跑：

| condition | trigger | swap_pct (overall/per-ep) | WR (100ep) | Δ vs C0 (=0.870) |
|---|---|---:|---:|---:|
| C0_031A_solo | none | 0% / 0% | **0.87** | ref |
| C1_031A_α_teacher031B | α (window<-3) | 18.2% / 13.0% | **0.87** | **0.00** |
| C2_031A_β_teacher031B | β (window<-5) | 16.8% / 15.0% | **0.79** | **-0.08** |

**100ep SE ±0.05, Δ -0.08 = 1.6σ 边缘**, 不是 robust negative 但方向清楚: 越激进 takeover (β) 越差。

### 7.8 三组 phase1 综合读图 — H1 / H2 / H3 update

| 假设 | Phase 1.1 (036D→031A, 100ep) | Phase 1.2 (031B→031A, 100ep) | Phase 1.3b (031A→036D, 500ep) |
|---|---|---|---|
| **H1 teacher quality → Δ** | NEUTRAL +1pp (036D=031A 同 baseline) | NEUTRAL/negative α=0, **β=-8pp** (031B 0.882>031A 0.860 但 takeover 没赢) | borderline +3.2pp α |
| **H2 互补 failure mode** | NEUTRAL — 036D defensive 接管 031A wasted_possession 没明显 fix | N/A (031B 跟 031A 同为 wasted_possession 主导) | **borderline +** — 031A attacker 接管 036D defensive_pin α 略 +3pp 与 H2 方向一致 |
| **H3 031B universal teacher** | N/A | **❌ 弱化** — 031B 即使是项目 SOTA, 当 teacher 接管 031A 仍 NEUTRAL/negative | N/A |

**综合方向**:
- H3 弱化（031B baseline +2.2pp 没 transfer 到 takeover），与 [snapshot-031 §13.8](snapshot-031-team-level-native-dual-encoder-attention.md#138-stage-3--h2h-matrix-4-matchups-n500-each-2026-04-19-0742) 「031B vs 031A peer H2H 0.516 NS」一致
- H2 仅 Phase 1.3b 给出 borderline + 方向 (+3.2pp α)，但单实验弱
- **H1 teacher quality 不是单调** — 即使 teacher (031B 0.882) > student (031A 0.860), takeover 也可能 NEUTRAL 或 negative

**Phase 2 启动决策保持**: 三个 phase1 lane 都没给 ≥+0.03 robust signal，**Phase 2 不启动**。snapshot-050 lane 暂缓，标记 borderline 不关闭。

如果 045 capture / 051 1000ep 两条 in-flight 都 verdict 否决 → 重排 GPU 时再考虑 Phase 1.3c 1000ep 把 +3.2pp α 信号缩到 ±0.012 SE 验证；但当前优先级低。
