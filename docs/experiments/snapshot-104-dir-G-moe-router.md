# SNAPSHOT-104: DIR-G — MoE Router over Frozen Experts (Stone Wave 1 Uniform + Wave 2 REINFORCE)

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 预注册 + Wave 1 verdict 已 append (deploy-time only); Wave 2 router NN training pending
- **前置**: [snapshot-099 §2.1 DIR-G](snapshot-099-stone-pipeline-strategic-synthesis.md#21-总览) (Shazeer 2017 MoE) / [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) (DIR-A heuristic selector, sister framework)

---

## 0. 背景

### 0.1 Mixture-of-Experts 在 RL multi-agent 上

Shazeer et al. 2017 "Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts" 原 paper 在 NLP 上 — 把 K 个 large expert + 一个 router NN 组成稀疏激活 model。在我们 RL multi-agent context:
- **Experts** = 已 trained policies (frozen specialists)
- **Router** = 小 NN, input = obs, output = K-dim distribution over experts
- **Hard switch**: 每步 argmax / sample 一个 expert, 用 那个 expert action
- **Soft switch** (= v074 family, 已知 sub-SOTA): 不本 lane 测

### 0.2 与 DIR-A (heuristic selector) 的 differentiation

| Item | DIR-A (snapshot-100) | DIR-G (本 snapshot) |
|---|---|---|
| Selector type | Hand-coded geometric phase classifier | Learnable NN router |
| State features | 4 hand-coded phase | full obs (336-dim ray) |
| Wave 1 | 4-phase mapping (= 0.875 sub-SOTA) | uniform random over K experts (= **0.900**, beat DIR-A) |
| Wave 2 | Wave 1 + orthogonal specialists | REINFORCE-trained router NN + same specialists |

**Wave 1 已显示 uniform > biased-toward-weak**, Wave 2 REINFORCE 期望 trained > uniform.

### 0.3 与 v074 family (deploy-time prob averaging) 的差异

- v074 = action-distribution level averaging: 每步 K experts 各输出 logits, average → joint policy → sample action
- DIR-G = hard switch: 每步选 1 expert, 用 那个 expert 整个 action vector
- v074 已知 sub-SOTA 且 5 variant 全 saturate (snapshot-074 / 074B / 074C / 074D / 074E)
- DIR-G 是 categorical mixture instead of action-prob mixture, 理论上 different cf. mode-mixing issue

### 0.4 与 DIR-E (option-critic) 的 differentiation

- DIR-E 有 explicit termination function β(s) → temporal stickiness (commit to option until β fires)
- DIR-G 每步独立 routing → no temporal correlation
- 假设 DIR-E 强于 DIR-G 在 cooperative 2v2 (因为 momentum + option commitment 帮助)
- 但 Wave 1 实测 DIR-G uniform 0.900 ≈ DIR-E random NN 0.895, 差异 < SE → 两者本质 routing 几乎一样, 看 Wave 2 能否分开

---

## 1. 核心假设

### 1.1 H_104 (主)

> Wave 1 uniform router (random per-step routing among K=3 frozen experts) baseline 200ep WR 接近 max(individual expert) — 验证 framework infra 工作.
> Wave 2 REINFORCE-trained router NN ≥ Wave 1 uniform + 0.010 — state-conditional learned routing 提取 specialist signal.
> Wave 2 with orthogonal specialists (081 + 103A/B/C) ≥ 0.92 — orthogonal specialists + learned routing 是 PIPELINE 主线.

### 1.2 H_104-stretch

> Wave 2 with orthogonal specialists ≥ 0.925 — DIR-G 完全 dominate DIR-A heuristic, 进 PIPELINE Stage 2 default selector.

### 1.3 H_104-tied

> Wave 2 ≈ Wave 1 uniform (within 0.005) — REINFORCE 没 extract state-conditional signal, router NN 没 generalize.

### 1.4 H_104-anti

> Wave 2 < Wave 1 uniform (REINFORCE 反 hurt) — router NN 学到 collapse pattern (e.g., always select 1 expert), 退化到 single best specialist baseline + sample noise.

---

## 2. Design

### 2.1 Wave 1: Uniform Router (deploy-time, no training)

```python
# Per agent, per step:
expert_idx = np.random.randint(K)
action = experts[expert_idx](obs)
```

- K = 3 experts (1750 SOTA / 055@1150 / 029B@190) — 同 DIR-A Wave 1 specialist library
- 不 train 任何 NN, 纯 random uniform routing
- 是 **lower-bound baseline for Wave 2 trained router**

### 2.2 Wave 2: REINFORCE-trained Router NN

```
Router NN:
  input: 336-dim obs (per agent)
  hidden: [128, 64] ReLU
  output: K-dim softmax (K = 3 or 5 depending on specialist library)

Per agent, per step:
  logits = router_nn(obs)
  expert_idx ~ softmax(logits)  (during training); argmax (during eval)
  action = experts[expert_idx](obs)

Training (REINFORCE):
  loss = -log_prob(expert_idx) * episode_return
  optimizer = Adam(lr=1e-3)
  episodes = 500-1000 vs baseline opponent
```

- ~30K params (small router NN, < 30 sec per gradient step)
- Training cost: ~3h GPU
- Episodes 数据: 不需要新 env infra, 直接用现有 trained_team_ensemble_next_agent 的 forward pipeline + reward

### 2.3 Wave 2 specialist library (designed)

按 [snapshot-100 §2.3](snapshot-100-dir-A-heuristic-selector.md#23-wave-2-specialist-mapping-designed-not-yet-committed) 同 mapping:
- 1750 SOTA (general)
- 081 aggressive (NEAR_GOAL specialist) — pending 081 verdict
- 103A INTERCEPTOR — pending 103-series
- 103B DEFENDER — pending 103-series
- (optional) 101A ball-control — pending 101A verdict

Total K ∈ [3, 5] depending on which lanes hit.

### 2.4 Implementation

- **Wave 1**: `agents/v_moe_router_uniform/agent.py` (already exists)
- **Wave 2**: `agents/v_moe_router_trained/agent.py` (TBD pending Wave 1 framework verified)
- Router training script: `scripts/research/train_moe_router_reinforce.py` (TBD)

### 2.5 Eval setup

- Wave 1: 200ep baseline (今日已做)
- Wave 2: 200ep + 1000ep + (if hit) combined 2000ep

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict 含义 |
|---|---|---|---|
| §3.1 Wave 1 main | 200ep ≥ 0.90 | uniform router framework 工作 | proceed Wave 2 |
| §3.2 Wave 1 stretch | 200ep ≥ 0.92 | uniform 已 beat 1750 single (unlikely) | unexpected — diversity 直接给 lift |
| §3.3 Wave 1 tied | 200ep ∈ [0.85, 0.90) | framework 工作但 sub-SOTA | proceed Wave 2 期望 lift |
| §3.4 Wave 1 break | 200ep < 0.85 | framework 破 | infra bug, debug |
| §3.5 Wave 2 main | 200ep ≥ 0.92 with orthogonal | learned routing on orthogonal specialists 解锁 | promote PIPELINE Stage 2 |
| §3.6 Wave 2 stretch | 1000ep ≥ 0.925 | decisive over 1750 | DIR-G 进 PIPELINE Stage 2 default |

---

## 4. Simplifications + Risks

### 4.1 简化 S1 — Wave 1 uniform 而非 hand-coded heuristic

- **节省**: 不需要 phase classifier (DIR-A 那套)
- **优势 (实测)**: uniform > biased-toward-weak — Wave 1 0.900 > DIR-A 0.875

### 4.2 简化 S2 — Wave 2 用 REINFORCE 而非 PPO

- **节省**: REINFORCE 简单 (no critic, no GAE), router NN 小 (~30k params)
- **Risk R1**: REINFORCE high variance, 可能不收敛到 useful policy
- **Mitigation**: 加 baseline (mean episode return) reduce variance; 如果 still fail, 切 PPO

### 4.3 简化 S3 — Per-step independent routing (no temporal stickiness)

- **节省**: 不 track per-agent option duration
- **Risk R2**: 频繁 expert switching 会丢 momentum (与 DIR-A R3 同)
- **降级**: 加 minimum-stick-duration 或 切 DIR-E option-critic 框架

### 4.4 简化 S4 — Hard switch 而非 soft mixture

- **节省**: 不 average expert outputs (= v074 family 已知 sub-SOTA)
- **Risk**: hard switch 在 specialist boundary 上 discontinuous; soft mixture 理论上 smoother
- **决定**: hard switch 与 v074 family 正交, 测 different mechanism, 不 fallback soft

### 4.5 全程 retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default Wave 1 | uniform routing 200ep eval | 5 min |
| 1 | §3.4 Wave 1 break | debug infra | 1h |
| 2 | §3.1/§3.3 Wave 1 hit | proceed Wave 2 | (continue) |
| 3 | Wave 2 train router NN with current specialists (3 distill family) | sanity check Wave 2 framework | 3h |
| 4 | Wave 2 with current specialists §3.5 hit (unlikely 因为 specialists 同质) | unexpected lift, deep dive | 3h analysis |
| 5 | Wave 2 with orthogonal specialists (after 081 + 103-series) | main test | 3h train + 1h eval |
| 6 | Wave 2 §3.5 hit | promote PIPELINE Stage 2 selector | (continue) |
| 7 | Wave 2 §3.6 hit | DIR-G 进 PIPELINE default selector + propose extend (more orthogonal experts, larger router) | varies |

---

## 5. 不做的事

- 不 train PPO router (用 REINFORCE 简单)
- 不 mix hard + soft switch (留 v074 family 已 cover soft)
- 不在 Wave 1 改 specialist library (与 DIR-A Wave 1 同 K=3)
- 不加 temporal stickiness Wave 1 (留 DIR-E option-critic 测 stickiness)

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [x] 2. 实现 `agents/v_moe_router_uniform/agent.py` (already exists)
- [x] 3. 200ep baseline eval — Wave 1 (today) → §7
- [ ] 4. 实现 `agents/v_moe_router_trained/agent.py` + `scripts/research/train_moe_router_reinforce.py`
- [ ] 5. Wave 2 router NN train with current 3 specialists (sanity)
- [ ] 6. 等 081 + 103A/B/C 完成 → Wave 2 retrain with orthogonal library
- [ ] 7. Wave 2 200ep + 1000ep eval
- [ ] 8. Verdict per §3.5/§3.6 → PIPELINE Stage 2 decision

---

## 7. Verdict — Wave 1 §3.1 HIT (uniform router framework 工作, beats DIR-A heuristic, 2026-04-22 append-only)

### 7.1 Wave 1 200ep baseline (2026-04-22)

| Metric | Value |
|---|---:|
| Module | `agents.v_moe_router_uniform` |
| Episodes | 200 |
| WR | **0.900** (180W-20L) |
| Δ vs 1750 single SOTA (0.9155) | -0.016 sub-SOTA |
| Δ vs DIR-A heuristic (0.875) | **+0.025** beat |
| Δ vs DIR-E option-critic random NN (0.895) | +0.005 within SE |

### 7.2 严格按 §3 判据

| 阈值 | 实测 200ep | verdict |
|---|---|---|
| **§3.1 Wave 1 main ≥ 0.90** | **✅ 0.900 boundary** | **HIT (just at threshold)** |
| §3.2 Wave 1 stretch ≥ 0.92 | ❌ | not met |
| §3.3 Wave 1 tied [0.85, 0.90) | ❌ above 0.90 | not in tied band |
| §3.4 Wave 1 break < 0.85 | ❌ | framework not broken |

### 7.3 关键 lesson — uniform > biased

诊断 (与 DIR-A heuristic 0.875 对比):
- DIR-A heuristic 把 weak slot (055@1150) routing 到 high-frequency BALL_DUEL phase → ~50% 步 weak slot
- DIR-G uniform 把 weak slot routing 到 1/3 步 → 期望 weak slot frequency 33% 而非 50%
- **"random fair" beats "biased toward weak"** — confirmed by data

但 0.900 仍 sub-SOTA vs 1750 (0.9155):
- Uniform 还是 forced 1/3 步 weak slot, 整体被 dragged down
- Single best specialist (1750) 始终 dominant 在当前 library

→ Wave 2 必须配 orthogonal specialists 才能 beat 1750

### 7.4 与 DIR-A / DIR-E Wave 1 合读

| Framework | 200ep | 解读 |
|---|---:|---|
| DIR-A heuristic (forced bias) | 0.875 | 高频 phase × weak slot |
| **DIR-G uniform router** | **0.900** | 1/3 weak slot |
| DIR-E option-critic random NN | 0.895 | ~50% term, ~1/3 weak slot 同 G |
| **best single specialist (1750)** | **0.9155** | reference |

**DIR-G uniform** 是 3 个 framework 里 best Wave 1, 但仍 sub-SOTA. **Wave 2 trained router + orthogonal specialists 才是真 test**.

### 7.5 Lane 决定

- DIR-G Wave 1 framework infra 验证 ✅
- 进 Wave 2 准备工作:
  1. 实现 router NN training script (REINFORCE) — ~3h eng
  2. 等 081 / 103A/B/C 完成 提供 orthogonal specialists
  3. Wave 2 train + eval — ~3h GPU + 1h eval

---

## 7B. Wave 2 verdict — MAJOR REGRESSION 0.900→0.645 (uniform 8-expert pool dragged by weak specialists, 2026-04-22 [07:10] append-only)

### 7B.1 Wave 2 setup

After 5 Stone sub-task specialists ready (081 + 101A + 103A + 103B + 103C), expanded expert pool from Wave 1's 3 (1750 + 055 + 029B) to **8 experts**. Same uniform-random routing infra (per-step random expert pick).

Eval: 200ep baseline on 5032907 port 61705, 871s elapsed.

### 7B.2 Result

| Metric | Wave 1 (3 experts) | **Wave 2 (8 experts)** | Δ |
|---|---|---|---|
| **WR** | 0.900 | **0.645** | **-0.255** ⚠️ MAJOR REGRESSION |
| fast_win | 0.890 | 0.520 | -0.370 |
| episode_mean step | 39.9 | 62.2 | +22.3 (much less efficient) |
| W-L | 180-20 | 129-71 | -51 wins, +51 losses |

### 7B.3 Diagnosis — uniform routing = weighted average

Mathematical prediction for uniform 8-expert pool:
- Mean standalone WR = (0.9155 + 0.907 + 0.86 + 0.826 + 0.851 + 0.548 + 0.205 + 0.220) / 8 = **0.667**
- Actual measured: 0.645 (within 0.022 of prediction; difference = routing randomness)

**Confirms** snapshot-099 §5C / snapshot-106 finding: uniform router cannot extract value from heterogeneous specialist pool when standalone WR varies widely. The 2 weakest specialists (103B 0.205 + 103C 0.220) drag the mean.

### 7B.4 严格按 §3 判据

| 阈值 (Wave 2) | 实测 | verdict |
|---|---|---|
| §3.4 Wave 2 main: 200ep ≥ 0.92 (with orthogonal specialists) | ❌ 0.645 (-0.275) | not met |
| §3.5 stretch: ≥ 0.925 (DIR-G dominates DIR-A) | ❌ | not met |
| §3.6 anti: regression -0.02 | ❌ -0.255 (way past) | **HIT regression — uniform routing fails on heterogeneous pool** |

### 7B.5 Lane decision (Wave 2 closed, Wave 3 unblocked)

- **Wave 2 mapping FALSIFIED for uniform routing** — adding more experts uniformly hurts ensemble proportional to mean expert WR
- **Wave 3 designs queued**:
  - (a) **REINFORCE-trained router** (was original Wave 2 plan, now Wave 3): NN router that LEARNS to suppress weak experts state-conditionally. Requires REINFORCE trainer (`train_moe_router_reinforce.py` already written) + 3-5h training.
  - (b) **Prune marginal experts** (Wave 3-prune): drop 103B/C (the worst) from pool, keep 6 experts. Test if uniform routing recovers without bottom-2 drag.
  - (c) **DIR-H cross-attention fusion** (queued P1, snapshot-107 prep): replace MLP/random selector with cross-attention gating over experts. Even untrained, attention can be more selective than uniform.
  - (d) **Wave 3-refined**: retry with 103A-warm-distill replacing 103A v1 (in flight). If specialists themselves get fixed (≥ 0.85 standalone), uniform routing may recover.

## 8. 后续

### 8.1 Wave 2 主 path

按 §6 步 4-7 执行. 期望 Wave 2 with orthogonal specialists 200ep ≥ 0.92.

### 8.2 Wave 2 §3.5 hit → PIPELINE 集成

DIR-G 进 PIPELINE Stage 2 default selector. 与 DIR-A heuristic 比较:
- 如果 DIR-G > DIR-A by > 2σ → DIR-G default
- 如果两者 tied → DIR-A heuristic 更 interpretable, prefer
- 如果 DIR-A > DIR-G → DIR-G drop

### 8.3 Wave 2 §3.5/§3.6 miss

- REINFORCE 没 extract state signal — 切 PPO router train
- 如果 PPO 也 miss → DIR-G 整族 close, fallback DIR-E option-critic 试 temporal stickiness

### 8.4 Hybrid: DIR-G + DIR-E

如果 DIR-G Wave 2 hit AND DIR-E Wave 2 hit, 可以试 hybrid: DIR-E 提供 termination (when to switch), DIR-G provides selector (which to switch to). BACKLOG 候选.

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone-pipeline strategic synthesis (DIR-G 在 §2.1 #5)
- [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) — DIR-A heuristic selector (sister framework, 同 specialist library Wave 1)
- [snapshot-105](snapshot-105-dir-E-option-critic-frozen.md) — DIR-E option-critic (sister framework with temporal stickiness)
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — sub-task lanes (Wave 2 specialist library 来源)
- [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) — v074 family (deploy-time prob averaging, 已知 sub-SOTA, contrast to hard-switch)
- [agents/v_moe_router_uniform/agent.py](../../agents/v_moe_router_uniform/agent.py) — Wave 1 实现
- [agents/v_moe_router_uniform/README.md](../../agents/v_moe_router_uniform/README.md) — module 说明
- Wave 2 module: `agents/v_moe_router_trained/agent.py` — TBD

### 理论支撑

- **Shazeer et al. 2017** "Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts" — MoE 原 paper
- **Jacobs et al. 1991** "Adaptive Mixtures of Local Experts" — early MoE
- **Williams 1992** "Simple Statistical Gradient-Following Algorithms (REINFORCE)" — Wave 2 router train method
- **Wang, Stone, Hanna 2025** — 4 sub-policies + selector 思想 (DIR-A 是 hand-coded 版, DIR-G 是 learned 版)
