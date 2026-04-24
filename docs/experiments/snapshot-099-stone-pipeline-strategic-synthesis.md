# SNAPSHOT-099: Stone-Pipeline Strategic Synthesis — 5/5 Distill Saturation 后的范式 pivot

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: Synthesis / 路径梳理 (meta-snapshot, 不对应具体 lane; 作为 100-105 + PIPELINE 预注册的前置文档)
- **范围**: 综合 [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) + 071/072/073/076/079 distill saturation 证据 + Peter Stone lab 4 篇文献 (Stone Veloso 2000 / Stone Sutton Kuhlmann 2005 / Stone PhD 1998 TPOT-RL / Wang Stone Hanna ICRA 2025) → 6-direction work plan (DIR-A/B/E/F/G + PIPELINE)。

---

## 0. 背景 — 为什么必须 pivot

### 0.1 Distill 范式 5/5 saturation (今日 2026-04-22 confirmed)

snapshot-075 §6 当时给的 3 条 path (DIR-A wide-student / DIR-B per-agent / DIR-C DAGGER) 已经 partially executed:

| Lane | 设计变量 | 1000ep peak | Δ vs 1750 SOTA 0.9155 |
|---|---|---:|---:|
| 071 Pool A homogeneous | 3-teacher 同家族 | 0.903 | -0.013 |
| 072 Pool C cross-axis | reward 多样性 | 0.903 | -0.013 |
| 073 Pool D cross-reward | 3-teacher 3 reward path | 0.909 | -0.007 |
| 076 wide-student | 1.4× capacity | 0.905 | -0.011 |
| 079 single-teacher recursive | 1 SOTA teacher | 0.914 | -0.002 |

5 lane **正交设计变量** (teacher count / family / reward axis / student capacity / reward-path diversity) 同时 cap 在 0.90-0.91。snapshot-076 §7.3 / snapshot-079 §6.3 已写进 verdict: **distill paradigm 自身的极限**, 不是 hyperparameter / α decay / teacher pool 选择问题。

### 0.2 还有 1 个未测但低预期 lane

- 077 per-agent student distill — engineering complete 但单测期望 P(≥0.92) ≈ 25%, 即使 hit 也只是 +0.005 marginal, **不能依赖**
- 078 DAGGER — 工程 3-5 天, 在 5/5 saturation 证据下 ROI ≈ 5-15%
- 080 Pool A v2 1750-teacher swap — 仍是 distill paradigm 内, 大概率落入同 0.90-0.91 cluster

### 0.3 0.9155 SOTA + 0.93 目标的 gap

- Best 单模型 = `055v2_extend@1750` combined 4000ep **0.9155**
- 假设要求: 9/10 vs Random + 9/10 vs Baseline = **WR ≥ 0.90** (assignment 硬要求)
- 项目长期 stretch goal: **0.93** (3.3 × SE over 0.9155)
- distill paradigm 现在的 effective ceiling ≈ 0.91 → +0.020 to 0.93 **不在 distill 内可达**

### 0.4 Pivot 决策

继续在 distill 上 throw resources = **expected additional gain < 1pp, P(≥0.92) < 20%**。
必须找 **fundamentally different 的 mechanism**, 这就是为什么转 Stone-line。

---

## 1. Stone Lab 4 篇文献 — 各自贡献

| # | Paper | 年份 | 核心 idea | 对本项目的 actionable 贡献 |
|---|---|---|---|---|
| 1 | Stone & Veloso "Layered Learning in Multiagent Systems" | 2000 | Task decomposition: 把复杂多智能体 task 拆成 hierarchical sub-skills, 每层独立训练再 compose | **DIR-B** layered training pipeline (Phase 1 ball-control, Phase 2 passing, Phase 3 team strategy) |
| 2 | Stone, Sutton, Kuhlmann "RL for RoboCup-Soccer Keepaway" | 2005 | 用 sub-task (Keepaway) 作 benchmark + SMDP options 做 macro-action | (DIR-C 候选, **dropped** vs PPO+GAE 已超越) |
| 3 | Stone PhD thesis 1998 / TPOT-RL (Stone Veloso 1999 RoboCup-98) | 1998-99 | Opaque-transition credit assignment; 报告 +12.7pp goal-share lift over hand-coded baseline | (DIR-D 候选, **dropped → BACKLOG as DIR-D-QMIX**) |
| 4 | Wang, Stone, Hanna ICRA 2025 (arXiv 2412.09417) "RL Within the Classical Robotics Stack" | 2025 | 4 sub-policies (own state/action spaces) + heuristic selector; 拿了 2024 RoboCup SPL Challenge Shield Division | **DIR-A** heuristic state-conditional selector (Wave 1 deploy-time, no train); 是 Stone-line 最 recent + 最 actionable |

### 1.1 为什么 paper 4 (Wang/Stone/Hanna 2025) 是 anchor

- **Empirical 已 win**: 2024 RoboCup Challenge Shield 实赛胜利, 不是 simulation only
- **Recipe 干净**: 不需要新 RL, 只需要 (1) 训出多个 specialist + (2) 构造 selector
- **Wave 1 = deploy-time only**: zero training cost, 立刻验证 framework 是否 viable
- **Wave 2-3 path 清晰**: 如果 Wave 1 framework 看起来有 lift, 后续可加 trained selector / specialized state spaces / action constraints

### 1.2 为什么 paper 1 (Stone Veloso 2000 Layered) 进 DIR-B

- 这是 **Stone's bedrock idea**: 多智能体 task 不需要 single end-to-end policy, 可以拆 layer 训
- 我们当前所有 lane 都 end-to-end PPO from scratch 或 warm-start — **没人测过 hierarchical training schedule**
- DIR-B Phase 1 = "ball control specialist vs RANDOM only" 是最 natural 的第一层

### 1.3 为什么 paper 2 (Keepaway SMDP) 不直接进 plan (DIR-C dropped)

- SMDP options 在 2005 Q-learning + linear function approx 时代是 SOTA
- 现代 PPO + GAE 已经隐式做 multi-step credit assignment
- option discovery 算法 (option-critic Bacon 2017) 在 DIR-E 复用更好
- **如果 DIR-E HIT**, 我们不需要回头补 DIR-C

### 1.4 为什么 paper 3 (TPOT-RL) 不直接进 plan (DIR-D dropped → BACKLOG)

- TPOT-RL 主 contribution = opaque-transition credit assignment 在 RoboCup 1998 pass/dribble decision 上
- modern alternative = **QMIX (Rashid 2018)** monotonic value-function factorization, theoretically superior
- 与 DIR-F VDN (Sunehag 2017) overlap: DIR-F 已经测 monotonic critic decomposition
- 留 BACKLOG = **DIR-D-QMIX**, 仅当 DIR-F VDN 出 positive lift 才回头 (recovery condition 详见 BACKLOG)

---

## 2. 6-Direction Work Plan

### 2.1 总览

| Dir | 名字 | Mechanism | Train cost | Eng cost | ETA | Source paper |
|---|---|---|---|---|---|---|
| **DIR-A** | Heuristic state-conditional selector | Wave 1: 4-phase classifier from 336-dim ray obs → route to 1 of K experts | **0** (deploy-time only) | ~200 LOC | < 1 day | Wang/Stone/Hanna 2025 |
| **DIR-B** | Layered training pipeline | Phase 1 ball-control specialist (vs RANDOM only) → Phase 2 passing → Phase 3 team | 4h (Phase 1) + 8h (Phase 2) + 12h (Phase 3) | ~150 LOC env wrapper | 3-5 days | Stone Veloso 2000 |
| **DIR-E** | Option-Critic over frozen experts | Bacon 2017 framework with frozen intra-option policies + small NN for term + selector | Wave 1: 0; Wave 2: ~3h | ~400 LOC | 2 days | Bacon Harb Precup 2017 (option-critic) |
| **DIR-F** | VDN decomposed critic | Joint V = V_0(s_0) + V_1(s_1) + bias; centralized actor + decomposed critic | 12h scratch | ~250 LOC (new model) | 1-2 days | Sunehag et al. 2017 (VDN) |
| **DIR-G** | Learned MoE Router | Wave 1: uniform random; Wave 2: REINFORCE-trained NN router (hard switch per step) | Wave 1: 0; Wave 2: ~3h | ~300 LOC | 2 days | Shazeer 2017 / generic MoE |
| **PIPELINE** | Final integration | 5-stage compose of A+B+E+F+G + sub-task specialists | varies | varies | 1 week | (own synthesis) |

### 2.2 单 direction 的 expected payoff (precise terms)

按 EV = P(positive lift) × magnitude:

| Dir | P(beat 0.9155 SOTA) | 期望 magnitude (if hit) | EV (pp) |
|---|---:|---|---:|
| DIR-A Wave 1 | 25% | +0.005 to +0.020 | +0.003 |
| DIR-A Wave 2 (with orthogonal specialists) | 40% | +0.010 to +0.030 | +0.008 |
| DIR-B Phase 1 alone | 15% | +0.0 to +0.010 | +0.0008 |
| DIR-B 整 pipeline | 35% | +0.010 to +0.030 | +0.007 |
| DIR-E Wave 2 (trained NN) | 30% | +0.005 to +0.020 | +0.004 |
| DIR-F VDN | 35% | +0.005 to +0.025 | +0.005 |
| DIR-G Wave 2 (REINFORCE router) | 35% | +0.005 to +0.020 | +0.005 |
| **PIPELINE 合成** | **45%** | **+0.010 to +0.040** | **+0.011** |

**Best EV = PIPELINE**, 但前提是 DIR-A/B/E/F/G 至少其中 2 条出 positive 才能 compose 出可信合成。**所以策略 = 先单跑 5 direction 各自 verdict, 然后用 verdict 决定 PIPELINE 配置**。

---

## 3. DIR-D-QMIX — BACKLOG 与 recovery condition

### 3.1 为什么从 7-direction 砍到 6

原 plan 包括 DIR-D = TPOT-RL aux head (Stone Veloso 1998):
- aux head 学 V(s, opaque) 为 PPO 提供额外 credit assignment signal
- 升级版 = QMIX monotonic mixing network

砍掉 DIR-D 的两个原因:

1. **与 DIR-F VDN 严重 overlap**: VDN = Σ V_i; QMIX = monotonic NN mixer of V_i — 同一系列, 设计变量只是 mixer 是 sum 还是 NN. 如果 DIR-F 单跑 negative, QMIX 大概率也 negative; 如果 DIR-F positive, QMIX 才有意义升级
2. **TPOT-RL 1998 empirical lift +12.7pp 在 1998 hand-coded baseline 上, 现代 PPO+GAE 已经隐式覆盖**: TPOT-RL 的 opaque-transition signal 在 multi-step return 上的核心思想 = GAE λ-return, 已是 PPO 默认

### 3.2 Recovery condition

仅当**两条**条件都成立时把 DIR-D-QMIX 从 BACKLOG 提回 active:

- **DIR-F VDN positive lift confirmed** (combined 2000ep ≥ 0.918)
- 用户/grader 觉得 QMIX 的 monotonic 严格性 比 VDN 的 sum 更值得 sell

工程估计: ~600 LOC (mixer NN + training loop change), ~6h GPU。详见 [BACKLOG.md § Stone-pipeline backlog](BACKLOG.md#stone-pipeline-backlog-deferred-from-6-direction-work-plan-2026-04-22)。

---

## 4. Rolling-launch sequencing decision

### 4.1 决策原则

- **Deploy-time-only 先**: DIR-A / DIR-G / DIR-E Wave 1 全部 zero-train, 几小时内可 compose + 200ep 验证
- **Train-time 中后**: DIR-B / DIR-F 各 4-12h GPU
- **Sub-task lanes 平行**: 103A/B/C 共 ~12h GPU 可 3 节点并行
- **PIPELINE 最后**: 必须等 DIR-A Wave 2 + sub-task 完成才能合成

### 4.2 序列 (今日开始 → ~5 天)

| Day | 动作 | Direction | 备注 |
|---|---|---|---|
| 0 (今晚) | DIR-A v_selector_phase4 + DIR-G v_moe_router_uniform + DIR-E v_option_critic_random Wave 1 200ep eval | A/G/E Wave 1 deploy-only | 已完成, 见 §5 |
| 1 | DIR-B 101A_layered_p1_ballcontrol launch | B Phase 1 | jobid 5033289 in flight |
| 1 | DIR-F 102A_vdn_scratch launch | F | jobid 5032914 in flight |
| 2 | 103A/B/C sub-task lanes launch (after sub-task design approval) | DIR-A Wave 2 prep | 12h parallel |
| 2 | DIR-G + DIR-E Wave 2 router NN training (REINFORCE) | A/G/E Wave 2 | ~3h each, 可与 sub-task 并行 |
| 3 | 081 aggressive complete (单独 lane, 也是 DIR-A Wave 2 specialist) | sub-task input | ETA 3h from start |
| 4 | DIR-B Phase 2 (passing on top of Phase 1 specialist) | B continued | depends on 101A verdict |
| 5 | PIPELINE 5-stage integration | PIPELINE | 见 §6 |

### 4.3 资源 budget (cumulative)

- 5 days × ~3 nodes (when possible) = ~360 GPU-hours
- 远低于 distill paradigm 的"再 throw 5 lane × 12h"= 60h 的 sunk cost
- ROI 估算: EV ≈ +0.011pp PIPELINE × ~80% chance to ship ≈ +0.009pp expected gain — comparable to the entire wave-1 distill saturation cost but pointing in a new direction

---

## 5. Wave 1 framework 已出结果 (今日, append-only)

3 个 deploy-time-only framework 已在 200ep baseline 测过:

| Framework | Module | 200ep baseline WR | 备注 |
|---|---|---:|---|
| DIR-A heuristic selector | `agents.v_selector_phase4` | **0.875** (175W-25L) | Hand-coded geometric phase classifier; Wave 1 mapping NEAR/MID/POS=1750-SOTA, BALL_DUEL=055@1150. 启发式 forced 大量步进 weak slot (BALL_DUEL phase due to ball-near-agent dominance). Sub-SOTA |
| DIR-G uniform router | `agents.v_moe_router_uniform` | **0.900** (180W-20L) | 3 expert (1750/055/029B) per-step uniform routing. Beats DIR-A Wave 1 because uniform > biased toward weak |
| DIR-E option-critic random-NN | `agents.v_option_critic_random` | **0.895** (179W-21L) | Bacon 2017 OC with frozen intra-option + random-init term/selector NN. ~50% term rate observed |

**3 个 framework 全部 sub-SOTA vs 1750 single (0.9155)**:

诊断 — **specialists too similar (all distill family)**:
- 1750 = 055v2_extend (distill from 055v2 teacher)
- 055@1150 = same family, earlier checkpoint
- 029B = per-agent SOTA, but trained on similar v2 reward
- 任何 router (heuristic / uniform / random NN) 不能从 3 个 high-overlap specialists 提取互补信号

**Wave 2 trigger condition**:
- 081 aggressive (orthogonal reward family, 期望 baseline WR [0.75, 0.88]) ready
- 103A/B/C sub-task specialists ready (interceptor / defender / dribble)
- 然后用 **真正正交** 的 specialist 重测 DIR-A Wave 2

详见 [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) / [snapshot-104](snapshot-104-dir-G-moe-router.md) / [snapshot-105](snapshot-105-dir-E-option-critic-frozen.md)。

---

## 5B. Wave 2 launched (2026-04-22 [06:55] append-only)

All 5 sub-task specialists ready (081 + 101A + 103A + 103B + 103C; see snapshots 081/101/103). Wave 2 of all 3 frameworks launched in parallel:

| Framework | Wave 2 specialist pool / mapping | Eval node | Pending |
|---|---|---|---|
| **DIR-A heuristic** Wave 2 | NEAR-GOAL → 081 (0.826) / BALL_DUEL → 103A (0.548) / POSITIONING → 1750 / MID-FIELD → 1750 | 5032911 port 61605 | 200ep baseline, ~7min |
| **DIR-G uniform router** Wave 2 | 8-expert pool: 3 generalists (1750/055/029B) + 5 specialists (081/101A/103A/B/C) | 5032907 port 61705 | 200ep, ~10min (8 experts slower) |
| **DIR-E option-critic random-NN** Wave 2 | same 8-expert pool, NN auto-adapts to 8 options | 5033292 port 61805 | 200ep, ~10min |

**Wave 2 design notes (2026-04-22)**:
- DIR-A heuristic: only 4 specialists used (NEAR/BALL_DUEL/POS/MID); skip marginal 103B (0.205) and 103C (0.220) from phase map (would force-route through weak slot); use 1750 SOTA as POSITIONING/MID-FIELD safe defaults.
- DIR-G/E: include all 8 experts in pool because routing is learned (uniform / random NN) — exposes the framework to all diversity, marginal experts get low weight if router/critic is sensible.
- "No packaging" per user 2026-04-22 — references ckpt paths directly (no `agents/v_*/` dirs created for the new specialists). Will re-evaluate packaging if Wave 2 hits PIPELINE V1 ≥ 0.92 threshold.

详见 [snapshot-100 §7B Wave 2](snapshot-100-dir-A-heuristic-selector.md) / [snapshot-104 §7B](snapshot-104-dir-G-moe-router.md) / [snapshot-105 §7B](snapshot-105-dir-E-option-critic-frozen.md) (results pending).

---

## 5D. Methodology corrections (2026-04-22 [07:30] append-only)

**See [snapshot-106](snapshot-106-stone-methodology-corrections.md) for detailed analysis.**

DIR-A/G/E Wave 2 三框架同 regress (-0.110/-0.255/-0.240) 触发 deep-dive 重读 Stone 4 paper。**3 个 root-cause methodology 错误识别**:

1. **Training-eval distribution mismatch** — 103 lanes 用 BASELINE_PROB=0.0 (vs random only training),但 eval vs baseline → specialist 没见过 baseline opponent type
2. **Stone Layered Learning bootstrap 跳过** — Layer N 没用 Layer N-1 frozen, specialists 学不到 team coordination
3. **Wrong metric** — 我们用 full-game WR,但 SPL 2024 用 task-success in controlled scenario (e.g., "9/10 1v2 NEAR-GOAL scoring"). Full-game WR conflates skill 与 ensemble quality

**Fixes queued (3 P0/P1)**:
- P0 **103A-refined**: warm-start 103A@500 + BASELINE_PROB=0.7 + frozen 1750 teammate + INTERCEPTOR aux reward — Stone-bootstrap fix
- P1 **Sub-task success metric harness**: SPL-style eval per specialist
- P2 **103B/C-refined** (after 103A-refined verify) + **Layer 2 pass-decision specialist** (true Layered Learning Phase 2)

Wave 3 (in flight, 14 evals: 5 ablation + 6 scenario-replay + 3 control) 出来后 verdict 写到 100/103/104/105。Methodology corrections **不依赖** Wave 3 outcome — 即使 Wave 3 全 negative,specialists v1 verdict 不否定 paradigm。

---

## 5C. Other lanes status (2026-04-22 [06:55] append-only)

| Lane | Status | Verdict snapshot |
|---|---|---|
| 080 Pool A v2 (1750-teacher distill) | CLOSED 0.906 §3.4 (6/6 distill saturation) | [snapshot-080](snapshot-080-pool-A-v2-with-1750-teacher.md) |
| 081 aggressive | CLOSED 0.826 §3.2 (NEAR-GOAL specialist) | [snapshot-081](snapshot-081-aggressive-offense-reward.md) |
| 101A layered Phase 1 | CLOSED 0.851 (Layer 1 specialist) | [snapshot-101](snapshot-101-dir-B-layered-phase1.md) |
| 102A DIR-F VDN | TRAINING (ckpt ~400/1250, 32%) | [snapshot-102](snapshot-102-dir-F-vdn.md) pending |
| 103A INTERCEPTOR | CLOSED 0.548 still-climbing (P1 resume queued) | [snapshot-103 §7A](snapshot-103-dir-A-wave3-sub-task-specialists.md) |
| 103B DEFENDER | CLOSED 0.205 plateau marginal | [snapshot-103 §7C](snapshot-103-dir-A-wave3-sub-task-specialists.md) |
| 103C DRIBBLE | CLOSED 0.220 plateau marginal (P1 v2 queued) | [snapshot-103 §7B](snapshot-103-dir-A-wave3-sub-task-specialists.md) |
| 082 two-stream (architecture) | TRAINING (ckpt ~1060/1250, 85%, inline peak 0.905) | [snapshot-082](snapshot-082-hierarchical-two-stream-siamese.md) pending |
| 083 per-ray attention (architecture) | TRAINING (ckpt ~1020/1250, 82%) | [snapshot-083](snapshot-083-per-ray-attention.md) pending |
| **DIR-A/G/E Wave 2** | EVAL RUNNING (200ep) | results pending ~7-10min |

---

## 6. PIPELINE 5-stage integration sketch

PIPELINE 是 DIR-A/B/E/F/G + 103-series sub-task specialists 的最终合成。仅当至少 2 条 directions 出 positive 才执行:

### Stage 1 — Specialist library

汇总所有 trained specialists 成 library:
- 1750 SOTA (general)
- 055@1150 (general, alt family)
- 029B@190 (per-agent)
- 081 aggressive (orthogonal reward) — pending
- 101A ball-control (DIR-B Phase 1) — in flight
- 103A INTERCEPTOR (BALL_DUEL specialist) — pending
- 103B DEFENDER (POSITIONING specialist) — pending
- 103C MID-FIELD-DRIBBLE (optional) — pending

### Stage 2 — Selector network choice

按 §5 Wave 2 verdict 选 best-EV selector:
- 如果 DIR-A Wave 2 hit: heuristic + orthogonal specialists
- 如果 DIR-G Wave 2 hit (REINFORCE-trained router): learned router
- 如果 DIR-E Wave 2 hit: option-critic with temporal stickiness
- 否则 fallback uniform router

### Stage 3 — Critic plug-in

- 如果 DIR-F VDN hit: 用 VDN-trained policy 替换 1 个或多个 specialists
- 否则 keep monolithic critic policies

### Stage 4 — Layered training compose

- 如果 DIR-B 整 pipeline hit: Phase 3 trained policy 作为 NEW general specialist 加 library
- 否则 Phase 1 ball-control specialist 作为 BALL_DUEL 替代 (替代 055@1150 weak slot)

### Stage 5 — DAGGER bootstrap (Phase 6 in BACKLOG)

- 4 个 sub-behaviors 互相 query: e.g., MID-FIELD policy 在 NEAR-GOAL state 上失败时, 用 NEAR-GOAL specialist 给 label 修正
- 这是 Stone Layered Learning 的 bootstrapping idea apply 到 sub-behavior 边界
- ~4h GPU + ~500 LOC (DAGGER buffer + label generation)
- BACKLOG 已记录, recovery condition = PIPELINE Stage 1-4 已 produce 可 compose 的 library

### 6.1 PIPELINE expected gain

- 如果 §5 Wave 1 framework 全 sub-SOTA, **真正 unlock 在 Wave 2 + sub-task specialist 上**
- **必要条件**: orthogonal specialists 真存在 (081 aggressive + 103A/B/C 至少 2 条)
- **充分条件**: selector 能识别 specialist 适用 phase (DIR-A heuristic 或 DIR-G/E learned)
- 如果两个条件都成立: **P(beat 0.9155) ≈ 45%, expected magnitude +0.010 to +0.040**

---

## 7. Verdict — 框架预注册, 单 direction verdict 见各自 snapshot

本 snapshot 是 strategic synthesis, 不持有自己的 verdict。各 direction 的 verdict 在:

| Snapshot | Direction | Status |
|---|---|---|
| [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) | DIR-A heuristic selector | Wave 1 verdict 已 append (0.875) |
| [snapshot-101](snapshot-101-dir-B-layered-phase1.md) | DIR-B layered Phase 1 | _Pending — 训练中 jobid 5033289_ |
| [snapshot-102](snapshot-102-dir-F-vdn.md) | DIR-F VDN | _Pending — 训练中 jobid 5032914_ |
| [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) | sub-task 103A/B/C | _Pending — 未 launch_ |
| [snapshot-104](snapshot-104-dir-G-moe-router.md) | DIR-G MoE router | Wave 1 verdict 已 append (0.900) |
| [snapshot-105](snapshot-105-dir-E-option-critic-frozen.md) | DIR-E option-critic | Wave 1 verdict 已 append (0.895) |

PIPELINE 合成 snapshot **未 pre-registered, 等 §5 + sub-task verdict 后单独写**。

---

## 8. 后续

### 8.1 短期 (24-48h)

1. 等 101A (DIR-B Phase 1) 训练完成 → §3 verdict, decide Phase 2
2. 等 102A (DIR-F VDN) 训练完成 → §3 verdict, decide DIR-D-QMIX recovery
3. 写 103A/B/C launchers + env extension (sample_expert_scenario 4-phase) → 12h GPU 并行
4. 等 081 aggressive 完成 → DIR-A Wave 2 phase mapping commit

### 8.2 中期 (3-5 days)

5. DIR-G + DIR-E Wave 2 router NN training (REINFORCE) — ~3h each
6. DIR-B Phase 2 training (passing) on top of 101A specialist — ~8h
7. 重测 DIR-A Wave 2 with orthogonal specialists (1750 + 081 + 103A/B/C) → 200ep + 1000ep
8. 如果至少 2 条 directions 出 positive lift → PIPELINE Stage 1-4 integration → 1000ep + combined 2000ep verdict

### 8.3 长期 (>1 week)

9. 仅当 PIPELINE Stage 1-4 出 ≥ 0.918 combined 2000ep 时, 触发 Stage 5 DAGGER bootstrap (BACKLOG Phase 6)
10. 仅当 DIR-F VDN positive 时, 触发 DIR-D-QMIX (BACKLOG)

### 8.4 失败回退路径

- 如果 §5 Wave 2 全部 still sub-SOTA AND PIPELINE compose 也 sub-SOTA → **declare distill+selector 都饱和**, 0.9155 是 final submission, 转 report quality (snapshot-075 §8 honest readback 路径)
- 如果 DIR-F VDN regression → 关 DIR-D-QMIX recovery 通道, monotonic factorization 整族否决

---

## 9. 相关

### 内部 cross-references

- [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) — pre-pivot strategic synthesis (distill paradigm 当时还有 path forward 假设)
- [snapshot-076](snapshot-076-wide-student-distill.md) — DIR-A wide-student verdict (capacity not binding)
- [snapshot-079](snapshot-079-055v3-recursive-distill.md) — single-teacher recursive verdict (saturation 5/5 evidence)
- [snapshot-073](snapshot-073-pool-D-cross-reward.md) — cross-reward path verdict (saturation evidence)
- [snapshot-081](snapshot-081-aggressive-offense-reward.md) — orthogonal reward family scratch (DIR-A Wave 2 specialist input)
- [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) — DIR-A Wave 1 + Wave 2 plan
- [snapshot-101](snapshot-101-dir-B-layered-phase1.md) — DIR-B layered Phase 1 pre-reg
- [snapshot-102](snapshot-102-dir-F-vdn.md) — DIR-F VDN pre-reg
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — sub-task 103A/B/C pre-reg
- [snapshot-104](snapshot-104-dir-G-moe-router.md) — DIR-G MoE router pre-reg + Wave 1
- [snapshot-105](snapshot-105-dir-E-option-critic-frozen.md) — DIR-E option-critic pre-reg + Wave 1
- [BACKLOG.md § Stone-pipeline backlog](BACKLOG.md#stone-pipeline-backlog-deferred-from-6-direction-work-plan-2026-04-22) — DIR-D-QMIX + Wave 3 sub-behavior policies + PIPELINE Phase 6 DAGGER

### 理论支撑

- **Stone & Veloso 2000** "Layered Learning in Multiagent Systems" — DIR-B 基础, hierarchical sub-skill decomposition
- **Stone, Sutton, Kuhlmann 2005** "RL for RoboCup-Soccer Keepaway" — sub-task benchmark + SMDP options (DIR-C dropped)
- **Stone PhD 1998 / Stone & Veloso 1999** "Team-Partitioned Opaque-Transition RL (TPOT-RL)" — opaque credit assignment (DIR-D dropped → BACKLOG as DIR-D-QMIX)
- **Wang, Stone, Hanna 2025** ICRA "RL Within the Classical Robotics Stack: A Case Study in Robot Soccer" arXiv:2412.09417 — DIR-A 基础, 4 sub-policies + heuristic selector (2024 RoboCup SPL Challenge Shield Division winner)
- **Bacon, Harb, Precup 2017** "The Option-Critic Architecture" — DIR-E 基础
- **Sunehag et al. 2017** "Value-Decomposition Networks for Cooperative Multi-Agent Learning" — DIR-F 基础
- **Rashid et al. 2018** "QMIX: Monotonic Value Function Factorisation for Deep MARL" — DIR-D-QMIX 基础 (BACKLOG only)
- **Shazeer et al. 2017** "Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts" — DIR-G 基础

### Code refs

- [cs8803drl/branches/expert_coordination.py](../../cs8803drl/branches/expert_coordination.py) — `sample_expert_scenario` (现有 ATTACK/DEFENSE; 103-series 需要扩到 4 phases)
- [cs8803drl/branches/team_siamese_vdn.py](../../cs8803drl/branches/team_siamese_vdn.py) — DIR-F VDN model (new file, smoke PASS)
- [agents/v_selector_phase4/agent.py](../../agents/v_selector_phase4/agent.py) — DIR-A Wave 1 module
- [agents/v_moe_router_uniform/agent.py](../../agents/v_moe_router_uniform/agent.py) — DIR-G Wave 1 module
- [agents/v_option_critic_random/agent.py](../../agents/v_option_critic_random/agent.py) — DIR-E Wave 1 module
- [scripts/eval/_launch_101A_layered_p1_ballcontrol.sh](../../scripts/eval/_launch_101A_layered_p1_ballcontrol.sh) — DIR-B Phase 1 launcher
- [scripts/eval/_launch_102A_vdn_scratch.sh](../../scripts/eval/_launch_102A_vdn_scratch.sh) — DIR-F VDN launcher
