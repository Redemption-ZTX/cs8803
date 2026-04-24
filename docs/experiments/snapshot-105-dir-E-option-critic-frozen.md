# SNAPSHOT-105: DIR-E — Option-Critic with Frozen Intra-Option Policies (Stone Wave 1 Random NN + Wave 2 REINFORCE)

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 预注册 + Wave 1 verdict 已 append (deploy-time only with random-init NN); Wave 2 termination/selector NN training pending
- **前置**: [snapshot-099 §2.1 DIR-E](snapshot-099-stone-pipeline-strategic-synthesis.md#21-总览) (Bacon 2017 Option-Critic) / [snapshot-104](snapshot-104-dir-G-moe-router.md) (DIR-G MoE router, sister framework without temporal stickiness)

---

## 0. 背景

### 0.1 Bacon Harb Precup 2017 — Option-Critic

"The Option-Critic Architecture" 在 Sutton/Precup/Singh 1999 options framework 上端到端 differentiable training. 核心三组件:
- **Intra-option policies** $\pi_k(a|s)$: 选定 option k 时的 action policy
- **Termination function** $\beta_k(s) \in [0, 1]$: 当前 option 是否结束
- **Policy over options** $\pi_\Omega(k|s)$: option 选择 (= selector)

执行流程:
```
if terminated (β fired) or first step:
    k = π_Ω(s)        # select new option
a = π_k(a|s)          # use intra-option policy
```

### 0.2 在本项目上的 frozen-options 适配

完整 Option-Critic 端到端 training 三个 head — 工程量大, 我们做 **frozen intra-option** variant:

- **Intra-option policies** = 我们的 K 个 packaged agents (frozen, no training)
- **Only train**: termination β + selector π_Ω (~30k params, < 1h training)

这是合理简化, 因为我们已经有 trained specialists, 只缺 "什么时候用什么". Bacon 2017 paper 也 explicitly 提 "frozen intra-option" 作为简化变体。

### 0.3 与 DIR-G (MoE router) 的 differentiation

| Item | DIR-G (snapshot-104) | DIR-E (本 snapshot) |
|---|---|---|
| Mechanism | Per-step independent routing | Option commitment + termination |
| Temporal | Memoryless | Sticky (stay until β fires) |
| Parameters | Router NN (~30k) | Termination NN + Selector NN (~30k each) |
| Wave 1 | Uniform random | Random-init NN (β/selector both untrained) |
| Wave 1 result | 0.900 | 0.895 (≈ DIR-G within SE) |

Wave 1 两者 tied, 看 Wave 2 trained 是否能 differentiate.

### 0.4 Why temporal stickiness might help

- 在 SoccerTwos cooperative 2v2 里, **specialist policies 内部有 momentum** — 如 DEFENDER 在 own half 站位, 频繁 switch 会让 agent 站到一半就被 router 改成 ATTACKER 跑去对方半场, **action sequence incoherent**
- Option commitment 保证 agent stick with 选定 specialist 直到 specialist 自己 (via β) 觉得 phase 该结束
- 类比 hierarchical RL 里的 "macro-action" — 一个 option = N 步 sub-action sequence, 不 mid-way 切换

---

## 1. 核心假设

### 1.1 H_105 (主)

> Wave 1 random-init NN baseline 200ep WR 接近 DIR-G uniform router (~0.90) — sanity check OC framework infra 工作.
> Wave 2 REINFORCE-trained termination + selector NN ≥ Wave 1 random + 0.010 — temporal stickiness + state-conditional selector 提取 specialist signal.
> Wave 2 with orthogonal specialists (081 + 103A/B/C) ≥ 0.92 — beat 1750 single, 进 PIPELINE Stage 2 selector 候选.

### 1.2 H_105-stretch

> Wave 2 ≥ DIR-G Wave 2 + 0.005 — temporal stickiness + 更精细 routing 比纯 hard switch 好, OC 框架 dominate MoE.

### 1.3 H_105-anti

> Wave 2 ≤ DIR-G Wave 2 — temporal stickiness 不需要 (cooperative 2v2 step-level decision 已足够), OC 额外 termination NN 增加 variance 反而 hurt.

### 1.4 H_105-collapse

> Wave 2 termination β collapse 到 0 (从不 terminate) 或 1 (每步 terminate, = degenerate to DIR-G). 此情况触发 entropy regularization on β.

---

## 2. Design

### 2.1 Wave 1: Random-init NN baseline

```python
# Random-init at module load:
termination_nn = MLP([336, 64, K])  # logits, sigmoid → β_k(s) ∈ [0, 1]
selector_nn = MLP([336, 64, K])     # logits, softmax → π_Ω(k|s)

# Per agent, per step:
if first_step or option_terminated:
    k_logits = selector_nn(obs)
    current_option = sample(softmax(k_logits))

action = experts[current_option](obs)

beta_logits = termination_nn(obs)
option_terminated = bernoulli(sigmoid(beta_logits[current_option])) > 0.5
```

- ~50k params total (termination + selector NNs)
- Random init expects β ≈ 0.5 ⇒ ~50% step termination probability — close to DIR-G uniform behavior, BUT with potential 50% temporal-correlation on average

### 2.2 Wave 2: REINFORCE-trained NN

```
Total loss = -log_prob(option_k) * episode_return
           + -log_prob(termination_decision) * advantage_at_termination
           + entropy_reg(β) (to prevent collapse to 0 or 1)
           + entropy_reg(selector) (to prevent collapse to single option)

Optimizer: Adam(lr=1e-3)
Episodes: 500-1000 vs baseline opponent
Training cost: ~3h GPU
```

**Critical**: Wave 2 必须加 entropy reg on β to prevent collapse — 详见 Bacon 2017 §4 explicitly 提这点。

### 2.3 Wave 2 specialist library (designed)

同 [snapshot-104 §2.3](snapshot-104-dir-G-moe-router.md#23-wave-2-specialist-library-designed):
- 1750 SOTA / 081 aggressive / 103A INTERCEPTOR / 103B DEFENDER
- (optional) 101A ball-control

K ∈ [3, 5] depending on which lanes hit.

### 2.4 Implementation

- **Wave 1**: `agents/v_option_critic_random/agent.py` (already exists)
- **Wave 2**: `agents/v_option_critic_trained/agent.py` (TBD)
- Training script: `scripts/research/train_option_critic_reinforce.py` (TBD, 与 DIR-G 同 framework 共用 ~70% code)

### 2.5 Eval setup

- Wave 1: 200ep baseline (今日已做)
- Wave 2: 200ep + 1000ep + (if hit) combined 2000ep
- Diagnostic: log per-agent option duration histogram (检查是否 collapse 到 0 或 always-on)

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict 含义 |
|---|---|---|---|
| §3.1 Wave 1 main | 200ep ≥ 0.88 | random NN 框架 infra 工作 | proceed Wave 2 |
| §3.2 Wave 1 collapse check | β ratio ∈ [0.30, 0.70] | random NN 接近 50% termination | sanity, no init bias |
| §3.3 Wave 2 main | 200ep ≥ 0.92 | trained NN unlock state-conditional + temporal stickiness | promote PIPELINE Stage 2 候选 |
| §3.4 Wave 2 stretch | 1000ep ≥ 0.925 | decisive over 1750 | DIR-E 进 PIPELINE Stage 2 default selector |
| §3.5 Wave 2 vs DIR-G compare | DIR-E - DIR-G ≥ +0.005 | temporal stickiness 真有 lift | DIR-E preferred over DIR-G |
| §3.6 Wave 2 collapse | β collapse to 0 or 1 | entropy reg insufficient | strengthen entropy reg + retrain |

---

## 4. Simplifications + Risks

### 4.1 简化 S1 — Frozen intra-option (= experts), only train β + selector

- **节省**: 不 train K 个 large policy (节省 ~K × 12h GPU)
- **基础**: Bacon 2017 explicitly 允许 frozen-option 简化变体
- **Risk**: 完整 Option-Critic 训练 intra-option 也, frozen 限制 specialist 适应 router; 但我们 specialist 已 well-trained, 不需要再调

### 4.2 简化 S2 — Wave 2 用 REINFORCE 而非 Bacon 2017 PPG

- **节省**: REINFORCE 简单 (无 critic head), 与 DIR-G 共用大部分 code
- **Risk R1**: REINFORCE high variance 在 termination NN 上更明显 (β 是 binary decision)
- **Mitigation**: 加 baseline (mean episode return) 分离 termination loss / selector loss; 如果 still fail 切 PPO

### 4.3 简化 S3 — 单 head NN (不 share encoder between β + selector)

- **节省**: 简单实现, 避免 termination + selector 互相干扰
- **Risk R2**: 重复 encode 336-dim obs (~30k params duplicate), training 慢一点
- **不 mitigate** — params 量小, 训练时间 cost 可接受

### 4.4 Risk R3 — β collapse

Bacon 2017 §4 specifically warns: 没 entropy reg 时 β 会 collapse 到 0 (从不 terminate, degenerate to "use single option entire episode") 或 1 (每步 terminate, degenerate to DIR-G).

- **Mitigation**: 加 `entropy_coeff_beta = 0.01` to encourage β ≈ 0.5 baseline + state variation
- **Monitor**: per-episode β histogram, 如果 std(β) < 0.05 表示 collapse

### 4.5 全程 retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default Wave 1 random NN | 200ep eval | 5 min |
| 1 | §3.1 Wave 1 hit | proceed Wave 2 | (continue) |
| 2 | Wave 2 train with current 3 specialists | sanity check Wave 2 framework | 3h |
| 3 | Wave 2 §3.6 collapse | strengthen entropy reg, retrain | +3h |
| 4 | Wave 2 with orthogonal specialists (after 081 + 103-series) | main test | 3h train + 1h eval |
| 5 | Wave 2 §3.3 hit | promote PIPELINE Stage 2 候选 | (continue) |
| 6 | Wave 2 §3.4/§3.5 hit | DIR-E 进 PIPELINE default selector + propose extend (3-head termination per agent type) | varies |

---

## 5. 不做的事

- 不 train intra-option policies (frozen by design)
- 不 train PPG / PPO (REINFORCE 简单)
- 不 share encoder across β + selector (单独 NN per head)
- 不 mix DIR-E + DIR-G (留 BACKLOG)
- Wave 1 不调任何超参 (random init = sanity check baseline)

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [x] 2. 实现 `agents/v_option_critic_random/agent.py` (already exists)
- [x] 3. 200ep baseline eval — Wave 1 random NN (today) → §7
- [ ] 4. 实现 `agents/v_option_critic_trained/agent.py` + `scripts/research/train_option_critic_reinforce.py`
- [ ] 5. Wave 2 train with current 3 specialists (sanity)
- [ ] 6. Diagnostic: β collapse check
- [ ] 7. 等 081 + 103A/B/C 完成 → Wave 2 retrain with orthogonal library
- [ ] 8. Wave 2 200ep + 1000ep eval
- [ ] 9. Verdict per §3.3/§3.4/§3.5 → PIPELINE Stage 2 decision

---

## 7. Verdict — Wave 1 §3.1 HIT (random NN framework infra 工作, ≈ DIR-G within SE, 2026-04-22 append-only)

### 7.1 Wave 1 200ep baseline (2026-04-22)

| Metric | Value |
|---|---:|
| Module | `agents.v_option_critic_random` |
| Episodes | 200 |
| WR | **0.895** (179W-21L) |
| Δ vs 1750 single SOTA (0.9155) | -0.020 sub-SOTA |
| Δ vs DIR-A heuristic (0.875) | +0.020 beat |
| Δ vs DIR-G uniform router (0.900) | -0.005 within SE (tied) |

### 7.2 严格按 §3 判据

| 阈值 | 实测 200ep | verdict |
|---|---|---|
| **§3.1 Wave 1 main ≥ 0.88** | **✅ 0.895** | **HIT** |
| §3.2 Wave 1 collapse check | β termination ratio observed ~50% | sanity passed (random NN 平均 50%) |
| §3.3-§3.6 Wave 2 thresholds | _Pending Wave 2_ | — |

### 7.3 关键 lesson — random NN ≈ uniform router (within SE)

诊断 (与 DIR-G uniform 0.900 对比):
- DIR-E random-init NN 在 random init 下 β ≈ 0.5, 几乎每步随机决定是否 terminate
- 当前 option 也 random init, 等同于 uniform random (期望)
- 这就是为什么 DIR-E Wave 1 (0.895) ≈ DIR-G Wave 1 (0.900) — 两者 mechanism 在 random init 下几乎 equivalent

**Wave 1 主功能 = sanity check that OC framework infra 工作, 不期望 beat DIR-G**.

### 7.4 与 DIR-A / DIR-G Wave 1 合读

| Framework | 200ep | 解读 |
|---|---:|---|
| DIR-A heuristic (forced bias) | 0.875 | 高频 phase × weak slot |
| DIR-G uniform router | 0.900 | 1/3 weak slot |
| **DIR-E option-critic random NN** | **0.895** | ~50% term, ~1/3 weak slot 同 DIR-G |
| **best single specialist (1750)** | **0.9155** | reference |

**3 个 framework Wave 1 都 sub-SOTA, distill family specialists 是 root cause**. Wave 2 + orthogonal specialists 是真 test.

### 7.5 Lane 决定

- DIR-E Wave 1 framework infra 验证 ✅
- 进 Wave 2 准备工作 (与 DIR-G 共用大部分):
  1. 实现 OC training script (REINFORCE + entropy reg on β) — ~3h eng
  2. 等 081 / 103A/B/C 完成 提供 orthogonal specialists
  3. Wave 2 train + eval — ~3h GPU + 1h eval
  4. β collapse diagnostic 必做

---

## 7B. Wave 2 verdict — MAJOR REGRESSION 0.895→0.655 (random NN ≈ uniform on 8-expert pool, 2026-04-22 [07:10] append-only)

### 7B.1 Wave 2 setup

Same expert pool expansion as DIR-G (3 → 8 experts including 5 Stone sub-task specialists). DIR-E specifics: random-init OptionCriticHead with sigmoid termination (~50% per step) + softmax selector over 8 options.

Eval: 200ep baseline on 5033292 port 61805.

### 7B.2 Result

| Metric | Wave 1 (3 experts) | **Wave 2 (8 experts)** | Δ |
|---|---|---|---|
| **WR** | 0.895 | **0.655** | **-0.240** ⚠️ MAJOR REGRESSION |
| Δ vs DIR-G Wave 2 (0.645) | +0.005 (within SE) | +0.010 (within SE) | tied |
| Δ vs DIR-A heuristic Wave 2 (0.765) | — | -0.110 | option-critic worse |

### 7B.3 Diagnosis — temporal stickiness 不补救

Random NN with sigmoid termination (~50%) ≈ uniform router with 2-step option averaging ≈ DIR-G uniform routing. Wave 2 confirms: temporal stickiness cannot recover from weak expert pool — what matters is expert quality + selector intelligence, not stickiness.

### 7B.4 严格按 §3 判据

| 阈值 (Wave 2) | 实测 | verdict |
|---|---|---|
| §3.5 Wave 2 main 200ep ≥ 0.92 | ❌ 0.655 | not met |
| §3.6 Wave 2 ≥ DIR-G Wave 2 | ❌ 0.655 vs 0.645 = +0.010 (within SE) | tied DIR-G, no temporal advantage |
| §3.7 Wave 2 anti regression | ❌ -0.240 | **HIT regression** |

### 7B.5 Lane decision

- **Wave 2 mapping FALSIFIED** — random NN selector + uniform-equivalent routing both fail on heterogeneous pool
- **Wave 3 designs queued**:
  - REINFORCE-trained termination β + selector — same as DIR-G Wave 3 plan (snapshot-104 §7B.5)
  - Hybrid DIR-E + DIR-G (queued in BACKLOG): trained β decides "when to switch" + trained selector decides "which to switch to"
  - DIR-H cross-attention fusion (queue P1) — orthogonal selector arch, may handle heterogeneous pool better
  - Wave 3-refined with 103A-warm-distill (in flight) — if specialists themselves get fixed

---

## 8. 后续

### 8.1 Wave 2 主 path

按 §6 步 4-9 执行. 期望 Wave 2 with orthogonal specialists 200ep ≥ 0.92.

### 8.2 Wave 2 vs DIR-G 比较 (§3.5)

并行训 DIR-E + DIR-G Wave 2, 同 specialist library + 同 episodes:
- 如果 DIR-E - DIR-G ≥ +0.005 → temporal stickiness 真有 lift, DIR-E preferred
- 如果两者 tied → DIR-G 更简单 (只 1 NN), prefer DIR-G
- 如果 DIR-G > DIR-E → temporal stickiness 不需要在 cooperative 2v2 step-level, DIR-E drop

### 8.3 Wave 2 §3.6 collapse handling

如果 β collapse to 0 (从不 terminate):
- agent 一直用第 1 步选定 option 整 episode → degenerate to "random single specialist"
- 加强 entropy reg β coefficient 0.01 → 0.05 retrain

如果 β collapse to 1 (每步 terminate):
- agent 每步 re-select option → degenerate to DIR-G uniform
- 同样加强 entropy reg, 或缩小 termination NN learning rate

### 8.4 PIPELINE 集成

- 如果 §3.3 hit, DIR-E 进 PIPELINE Stage 2 selector 候选
- 如果 §3.4/§3.5 hit, DIR-E 是 default selector
- 如果两者全 miss, DIR-E 退出 PIPELINE selector pool

### 8.5 Hybrid: DIR-E + DIR-G (BACKLOG)

如果两者 Wave 2 都 hit, 试 hybrid:
- DIR-E β termination 决定 "when to switch"
- DIR-G router 决定 "which to switch to"
- 理论上 best-of-both-worlds, 但工程量大, BACKLOG.

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone-pipeline strategic synthesis (DIR-E 在 §2.1 #4)
- [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) — DIR-A heuristic selector (sister framework, hand-coded)
- [snapshot-104](snapshot-104-dir-G-moe-router.md) — DIR-G MoE router (sister framework without temporal stickiness)
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — sub-task lanes (Wave 2 specialist library 来源)
- [agents/v_option_critic_random/agent.py](../../agents/v_option_critic_random/agent.py) — Wave 1 实现
- [agents/v_option_critic_random/README.md](../../agents/v_option_critic_random/README.md) — module 说明
- Wave 2 module: `agents/v_option_critic_trained/agent.py` — TBD

### 理论支撑

- **Bacon, Harb, Precup 2017** "The Option-Critic Architecture" — 原 paper, frozen intra-option 简化 explicit 提
- **Sutton, Precup, Singh 1999** "Between MDPs and Semi-MDPs" — options framework 基础
- **Stolle & Precup 2002** "Learning Options in Reinforcement Learning" — 早期 options discovery
- **Frans et al. 2018** "Meta Learning Shared Hierarchies" — 现代 hierarchical RL with shared sub-policies
- **Wang, Stone, Hanna 2025** ICRA — 4 sub-policies + selector 思想 (DIR-E 是 learned selector + termination 版)
