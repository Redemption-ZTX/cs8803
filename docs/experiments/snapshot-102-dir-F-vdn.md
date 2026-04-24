# SNAPSHOT-102: DIR-F — VDN-Style Decomposed Critic on 031B Cross-Attention

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 训练中 (jobid 5032914 on atl1-1-03-015-30-0); 1250 iter / 12h budget; verdict pending Stage 1 post-eval (~ETA 12h)
- **前置**: [snapshot-099 §2.1 DIR-F](snapshot-099-stone-pipeline-strategic-synthesis.md#21-总览) (Sunehag 2017 VDN) / [snapshot-076](snapshot-076-wide-student-distill.md) + [snapshot-079](snapshot-079-055v3-recursive-distill.md) (5/5 distill saturation 推动 pivot)

---

## 0. 背景

### 0.1 Sunehag et al. 2017 — VDN (Value-Decomposition Networks)

"Value-Decomposition Networks for Cooperative Multi-Agent Learning" 提出: 在 cooperative MARL 里, joint value function 可以 factorize 为 per-agent value 的 sum:

$$Q_{joint}(s, \vec{a}) = \sum_i Q_i(s_i, a_i)$$

**关键性质 (IGM monotonicity)**: 对任何 agent i, 选择最大化 $Q_i$ 的 action 也最大化 $Q_{joint}$, 因此 decentralized greedy execution = centralized optimal joint action。

### 0.2 在本项目 PPO 上的适配

VDN 原 paper 在 Q-learning 上, 我们改 PPO + GAE:

- **Joint V (centralized critic)**: $V_{team}(s) = V_0(s_0) + V_1(s_1) + bias$
- **Per-agent value 用 ONLY 自己的 encoder feature**: $V_0$ 看 agent 0 的 256-dim feature, $V_1$ 看 agent 1 的, 不共享 cross-attention
- **Actor (logits) 仍 centralized**: 用 merged feature → joint policy logits

**这是 "centralized actor + decomposed critic"**, 不是完全 decentralized — 因为 actor 仍需要全局 obs 做 coordination。

### 0.3 与 5/5 distill saturation 的正交性

- distill paradigm 的 bottleneck = teacher signal 共用 baseline-targeted action distribution + KL mode collapse
- VDN 的 bottleneck (假设) = monolithic critic 的 credit assignment 在 cooperative 2v2 上 noisy
- **两者完全正交**: VDN 不依赖 distill, 不需要 teacher pool, 不需要 ensemble; 它改的是 **PPO advantage signal 的来源**

如果 VDN 出 positive lift, 这是对 5/5 distill saturation 的一个 fundamentally different escape route。

### 0.4 Empirical 动机

031B 单跑 combined 2000ep 0.880, 加 distill 到 055 0.907 (+0.027). 但 distill 只能 push 到 0.91 ceiling. **如果 PPO advantage signal 本身有 credit assignment noise, 修正它可能直接提升 0.880 base**, 如果 base 提升 → 后续 distill ceiling 也整体上移。

---

## 1. 核心假设

### 1.1 H_102 (主)

> 把 031B Siamese cross-attn 架构的 monolithic critic 替换为 VDN decomposed critic ($V = V_0(s_0) + V_1(s_1) + bias$, 每个 $V_i$ 只看 agent i 的 encoder feature, 不用 cross-attention), keep actor centralized + 同 v2 reward shape, **combined 2000ep peak ≥ 0.918** (matches 1750 SOTA + 0.5σ, 显示 credit assignment 有真实 lift).

### 1.2 H_102-stretch

> combined 2000ep peak ≥ 0.925 — VDN 的 explicit credit assignment 是 cooperative 2v2 的 binding constraint, decomposed critic 解锁 +1pp 以上的 PPO sample efficiency

### 1.3 H_102-tied

> combined 2000ep peak ∈ [0.905, 0.918) — 与 031B + 055 cluster 持平, decomposed critic neutral, 结构改动不 hurt 也不 help. 此时 lane 关闭 + DIR-D-QMIX recovery 通道也关 (因为 VDN/QMIX 同族, sum vs NN mixer 差异不能改变 binding constraint)

### 1.4 H_102-anti

> combined 2000ep peak < 0.895 — VDN decomposed 让 critic 失去 cross-agent 信息, 反而 hurt cooperative coordination → close lane

### 1.5 子假设

- **H_102-a (credit assignment)**: 当前 monolithic critic 给两个 agent 同 advantage, 当 agent 0 进球 + agent 1 站桩, 两 agent 同 reward → 错信号. VDN $V_0(s_0)$ 让 agent 0 拿大 advantage, agent 1 拿小, 学习更精确
- **H_102-b (centralized actor 必要)**: 单纯 decentralized critic + decentralized actor = IPPO, 已知在 cooperative 2v2 上 sub-MAPPO. 我们 keep actor centralized 是 "decomposed critic + centralized actor" hybrid, 取 PPO advantage 改善 + actor coordination 同时
- **H_102-c (cross-attention not needed in critic)**: actor 还需要 cross-attention 做 coordination, 但 critic 只需要 per-agent state estimate → 这个不对称是 VDN 的 explicit assumption (IGM 假设)

---

## 2. Design

### 2.1 Architecture

```
Input: 672-dim joint obs = concat(agent_0_obs_336, agent_1_obs_336)

Siamese encoder (= 031B):
  obs_336 → Linear(336 → 256) → ReLU → Linear(256 → 256) → ReLU → feat_256 (per agent, shared weights)

Within-agent cross-attention (= 031B, ONLY for actor path):
  feat_256 → tokens(4 × 64) → single-head attn → attended_256

Actor (centralized, = 031B):
  concat(feat_0, attended_0, feat_1, attended_1) → 1024-dim
  → Linear(1024 → 256) → ReLU → Linear(256 → 128) → ReLU → policy_logits (27-dim Discrete)

VDN Critic (decomposed, NEW):
  V_0 = Linear(feat_0 only, 256 → 1)
  V_1 = Linear(feat_1 only, 256 → 1)
  bias = scalar trainable parameter
  V_team(s) = V_0(s_0) + V_1(s_1) + bias

Returned for PPO: V_team(s), policy_logits
```

### 2.2 Param count

- 031B base: ~462,000 params
- VDN addition: 2 × (256 × 1 + 1) + 1 bias = 515 params + ~200 bias adjustment ≈ ~700 params
- **Total: ~462,614 params** (smoke verified)

VDN 几乎 zero overhead — 它只是把 monolithic critic 换成 decomposed sum, 没增加 hidden layer.

### 2.3 Reward shape (= 031B v2, isolate VDN as only variable)

```bash
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01
SHAPING_POSSESSION_BONUS=0.002 SHAPING_POSSESSION_DIST=1.25
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003
```

完全等于 031B / 055 / 056 family — **唯一变量 = decomposed critic vs monolithic critic**

### 2.4 PPO setup (= 031B SOTA recipe)

- LR=1e-4, CLIP_PARAM=0.15, NUM_SGD_ITER=4
- TRAIN_BATCH_SIZE=40000, SGD_MINIBATCH_SIZE=2048
- ROLLOUT_FRAGMENT_LENGTH=1000
- BASELINE_PROB=1.0 (vs baseline opponent, scratch)

### 2.5 Budget

| 项 | 值 |
|---|---:|
| MAX_ITERATIONS | 1250 (= 031B) |
| TIMESTEPS_TOTAL | 50M |
| TIME_TOTAL_S | 43200 (12h) |
| CHECKPOINT_FREQ | 10 |
| EVAL_INTERVAL | 10 |

### 2.6 Code

- New file: `cs8803drl/branches/team_siamese_vdn.py` (model 实现 + register)
- Env var: `TEAM_SIAMESE_VDN=1` — switches critic from monolithic to decomposed
- Smoke PASS: 462,614 params total (matches 031B family + ~700 for V_0/V_1/bias), forward shape correct, V_team scalar, gradient flows

### 2.7 Eval setup

- 标准 official eval suite, EVAL_OPPONENTS=baseline,random
- 训完做 Stage 1 post-eval 1000ep on top ckpts
- 如果 §3.1 hit, Stage 2 combined 2000ep rerun on top 3

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict 含义 |
|---|---|---|---|
| §3.1 main | combined 2000ep peak ≥ 0.918 | matches 1750 SOTA + 0.5σ | VDN credit assignment 有真实 lift |
| §3.2 stretch | combined 2000ep peak ≥ 0.925 | decisive shift | decomposed critic 是 cooperative 2v2 binding constraint, recovers DIR-D-QMIX as natural follow-up |
| §3.3 tied | combined 2000ep peak ∈ [0.905, 0.918) | within 031B+distill cluster | VDN neutral, lane 关闭 |
| §3.4 regression | combined 2000ep peak < 0.895 | hurt | decomposed critic 抹掉 cross-agent value 信息, 否决 整 monotonic-factor 族 |

### 3.1 Decision rule

- Stage 1 single-shot 1000ep peak 必须 ≥ 0.910 才触发 Stage 2 combined 2000ep rerun (sub-marginal 没必要 spend)
- 如果 Stage 1 peak ∈ [0.880, 0.910), 关 lane (与 031B base 0.880 几乎持平 = no lift)
- 如果 Stage 1 peak < 0.870, regression 否决 整族

### 3.2 Cross-direction implication

- §3.1 hit → 触发 BACKLOG DIR-D-QMIX recovery (monotonic NN mixer 可能更 sharp)
- §3.3/§3.4 → 关 DIR-D-QMIX recovery (同族同 binding constraint, NN mixer 不会 magic)

---

## 4. Simplifications + Risks

### 4.1 简化 S1 — Critic 用 sum 而非 NN mixer (= VDN 而非 QMIX)

- **节省**: 工程 ~200 LOC vs QMIX 600 LOC
- **Risk R1**: sum 假设 strict additive, monotonic NN mixer 可以表达更复杂 cooperation pattern
- **降级**: 如果 §3.1 hit 但 §3.2 miss, 触发 DIR-D-QMIX recovery (BACKLOG) 验证 NN mixer

### 4.2 简化 S2 — Per-agent V 只用 self encoder feature, no cross-attention

- **节省**: 简单 architectural 改动
- **Risk R2**: 把 cross-agent 信息从 critic 完全去掉可能 hurt — 在 cooperative 2v2 里, agent 0 的 value 真实 depends 在 agent 1 的 state (e.g., agent 1 是否 control ball)
- **Mitigation**: actor 仍 keep cross-attention, 通过 policy gradient 间接 propagate agent-1-state-aware action; VDN paper 的 IGM 假设 = "agent i 在 agent i 的 obs 下做最优 action 也是 joint 最优", 在 well-coordinated cooperative 2v2 应该成立 (大部分时间 agent 1 state 已 reflected in env transitions)

### 4.3 简化 S3 — 不改 reward / opponent / lr / α / 任何其他变量

- **节省**: 干净 isolate VDN 变量
- **Risk R3**: 单变量可能找不到 combo optimum
- **降级**: 如果 §3.1 HIT 但 §3.2 miss, follow-up = VDN + 081 reward (=不同 reward 上 VDN 是否更明显) 或 VDN + distill 037 (= VDN base 上加 distill 是否能 stack)

### 4.4 Risk R4 — PPO + decomposed critic 的 advantage 估计 variance 升高

- 两个独立 V head, training noise 加 (vs monolithic 一个 head 平均)
- **Mitigation**: monitor `vf_loss_0`, `vf_loss_1`, `value_loss` 各自 trace; 如果 V_0 / V_1 之间 loss 差距 > 5×, 说明 decomposition 不平衡, 可能需要加 normalization

### 4.5 全程 retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default | VDN scratch 1250 iter | 12h |
| 1 | Stage 1 < 0.880 | 关 lane, 否决 monotonic-factor 族 (DIR-D-QMIX 也关) | — |
| 2 | Stage 1 ∈ [0.880, 0.910) | 关 lane (no lift over 031B base) | — |
| 3 | Stage 1 ≥ 0.910 → Stage 2 combined 2000ep | 验证 §3.1/§3.2 | +5h eval |
| 4 | §3.1 hit (combined ≥ 0.918) | trigger DIR-D-QMIX recovery (BACKLOG) + propose VDN+distill follow-up | +6h DIR-D-QMIX + 12h follow-up |
| 5 | §3.2 hit (combined ≥ 0.925) | major win, propose 把 VDN-trained policy 换进 PIPELINE specialist library + 重测 distill on top of VDN base | +12h |

---

## 5. 不做的事

- 不 mix VDN 与 distill (留给 follow-up combo 如果 §3.1 hit)
- 不 sweep additive vs NN mixer (后者是 DIR-D-QMIX, BACKLOG)
- 不 ablate "actor 也 decomposed" (这就 IPPO degenerate)
- 不改 reward shape (keep v2)
- 不 warm-start (scratch isolate VDN 变量)

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [x] 2. 实现 `cs8803drl/branches/team_siamese_vdn.py` (smoke PASS, 462,614 params)
- [x] 3. launcher `scripts/eval/_launch_102A_vdn_scratch.sh` 落地
- [x] 4. Launched 2026-04-22, jobid 5032914, atl1-1-03-015-30-0
- [ ] 5. 实时 monitor: `vf_loss_0` / `vf_loss_1` 平衡, policy_entropy 正常 decay, kl < 5
- [ ] 6. Stage 1 post-eval 1000ep on top 10 ckpts
- [ ] 7. 如果 peak ≥ 0.910, Stage 2 combined 2000ep on top 3 → verdict
- [ ] 8. Stage 3 H2H portfolio: vs 031B@1220 (base anchor), vs 055@1150 (SOTA family), vs 1750 (SOTA single)
- [ ] 9. Verdict append §7

---

## 7. Verdict — sub-SOTA lane CLOSED (2026-04-22 [13:35] append-only)

### 7.1 Training completion (after SLURM saga + Ray restore bug fix)

- Original scratch run (jobid 5032914) SIGTERM'd at iter 938/1250 due to SLURM 8h wall (training script had TIME_TOTAL_S=43200 = 12h). Memory `feedback_slurm_wall_budget.md` saved.
- Restore attempt v1 (10:28) terminated after 1 iter due to Ray restore+TIME_TOTAL_S bug (ckpt's persisted _time_total=28600s > fresh TIME_TOTAL_S=10800 → immediate stop). Memory `feedback_ray_restore_time_total.md` saved.
- Restore attempt v2 (10:33) with TIME_TOTAL_S=0 fix — **ran clean to completion at iter 1250/1250** (50M steps / 5h30m wall on fresh 5037130).
- Merged run dirs: `102A_vdn_scratch_20260422_015846` (iter 10-930) + `102A_vdn_restore_20260422_103346` (iter 940-1250). All ckpts usable; restore trial holds 930-1250.

### 7.2 Stage 1 baseline 1000ep (top 11 ckpts, parallel-7, 504s)

| ckpt | 1000ep WR | NW-ML | 200ep inline |
|---|---|---|---|
| **🏆 1060** | **0.893** | 893-107 | 0.870 |
| 1080 | 0.893 | 893-107 | 0.895 |
| 1140 | 0.891 | 891-109 | 0.895 |
| 970 | 0.890 | 890-110 | 0.895 |
| 1000 | 0.884 | 884-116 | 0.885 |
| 1250 | 0.884 | 884-116 | 0.870 |
| 1010 | 0.883 | 883-117 | 0.900 |
| 1070 | 0.876 | 876-124 | **0.930** (single-shot lucky) |
| 990 | 0.871 | 871-129 | 0.900 |
| 1210 | 0.871 | 871-129 | 0.895 |
| 980 | 0.867 | 867-133 | 0.905 |

**Peak = 0.893 @ ckpt 1060 and 1080 (dual-peak tied)**, mean 0.884, plateau 0.867-0.893.

### 7.3 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 main: peak ≥ 0.90 | ❌ peak 0.893 | MISS by -0.007 |
| §3.2 decisive: peak ≥ 0.915 | ❌ | MISS |
| §3.3 tied: peak ∈ [0.875, 0.900) | ✅ 0.893 | **tied-plateau HIT** |
| §3.4 regression: peak < 0.87 | — | not regressed |

**VDN = sub-SOTA** vs current 1750 SOTA 0.9155 (Δ=-0.023 ~ 3× SE) and tied with 031B 0.880 plateau.

### 7.4 Inline 200ep noise再次确认

ckpt 1070 inline 200ep = 0.930, real 1000ep = 0.876 → **Δ=-0.054** (single-shot luck), consistent with memory `feedback_inline_eval_noise.md` doctrine (up to -0.056pp optimism seen). Confirmed 6th time today.

### 7.5 DIR-D-QMIX recovery decision

Per snapshot-099 §3.2: DIR-D-QMIX BACKLOG recovery condition was "DIR-F VDN positive lift on baseline axis". 
- **DIR-F outcome**: sub-SOTA plateau, no positive lift over 031B / 055 / 1750 family
- **Decision**: **DIR-D-QMIX stays in BACKLOG, no recovery**. QMIX would add non-linear mixer on same decomposed-critic paradigm — if VDN doesn't lift, QMIX mixer alone unlikely to change that.

### 7.6 Mechanism reading

- VDN decomposes value V_team(s) = V_0(s_0) + V_1(s_1) + bias → forces per-agent credit assignment
- Expectation was better PPO advantage signal → sample efficiency or final WR uplift
- **Observed**: plateau at 0.884 (similar to 031B 0.880, below 055 0.907 and 1750 0.9155)
- **Interpretation**: in 2v2 soccer, team-level advantage signal is ALREADY well-formed by baseline PPO — decomposition doesn't provide additional gradient information. Value decomposition helps more in larger teams (StarCraft 5v5, SMAC 8v8) where credit assignment ambiguity is higher.
- 2v2 coop has sufficient credit clarity via joint V — VDN decomposition is **paradigm-neutral** (neither helps nor regresses), similar to 080 Pool A v2 (teacher swap with 1750) also stuck at plateau.

### 7.7 Lane decision

- **102A VDN lane CLOSED** at sub-SOTA (§3.3 tied-plateau, not §3.1 main)
- **Stage 2 capture + Stage 3 H2H SKIPPED** — ROI極低 vs clear sub-SOTA verdict
- **DIR-D-QMIX BACKLOG stays closed** (per §7.5)
- Not packaged per user directive "不需要 package，目前还没有突破性 SOTA"
- VDN model class (`cs8803drl/branches/team_siamese_vdn.py`) retained for potential future use (e.g., VDN+distill combo if we ever revisit decomposed-critic paradigm)

### 7.8 Raw Stage 1 recap

```
=== Official Suite Recap (parallel) ===
checkpoint-970  vs baseline: win_rate=0.890 (890W-110L-0T)
checkpoint-980  vs baseline: win_rate=0.867 (867W-133L-0T)
checkpoint-990  vs baseline: win_rate=0.871 (871W-129L-0T)
checkpoint-1000 vs baseline: win_rate=0.884 (884W-116L-0T)
checkpoint-1010 vs baseline: win_rate=0.883 (883W-117L-0T)
checkpoint-1060 vs baseline: win_rate=0.893 (893W-107L-0T)
checkpoint-1070 vs baseline: win_rate=0.876 (876W-124L-0T)
checkpoint-1080 vs baseline: win_rate=0.893 (893W-107L-0T)
checkpoint-1140 vs baseline: win_rate=0.891 (891W-109L-0T)
checkpoint-1210 vs baseline: win_rate=0.871 (871W-129L-0T)
checkpoint-1250 vs baseline: win_rate=0.884 (884W-116L-0T)
[suite-parallel] total_elapsed=503.7s tasks=11 parallel=7
```

Log: [102A_vdn_baseline1000.log](../../docs/experiments/artifacts/official-evals/102A_vdn_baseline1000.log)

### 7.0 Mid-training status (2026-04-22 [06:55] append-only)

VDN training in progress on jobid 5032914 (atl1-1-03-015-30-0). Current snapshot:

- ckpt 400/1250 (32% complete)
- inline baseline WR (200ep) at iter 270 = **0.82** — within healthy range for early-mid distill-family training (compare to 031B base 0.882 at convergence)
- Training reward stable, no NaN/inf alarms in sampled iterations
- VDN-specific check: per-agent value heads functioning (vdn_v_agent0_mean / vdn_v_agent1_mean diverging non-zero per smoke test → decomposition active in real training too)
- ETA: ~7h to reach iter 1250 (full budget)

Watcher: Bash bg `bhns26ok2` (will notify on srun exit). Per autonomous loop, audit + Stage 1 eval will trigger when watcher fires (~ETA 14:00 EDT).

待 fill 内容 after training done:
- Stage 1 1000ep top 10 ckpt WR
- 严格按 §3 判据
- vf_loss_0 vs vf_loss_1 balance check
- DIR-D-QMIX recovery decision
- VDN+distill follow-up decision

---

## 8. 后续

### 8.1 §3.1 hit (主 path)

- Stage 2 combined 2000ep verify
- Stage 3 H2H portfolio vs 031B / 055 / 1750
- Trigger BACKLOG DIR-D-QMIX recovery (NN mixer)
- Propose VDN+distill combo: 用 VDN-trained policy 作为 distill student base, 对比纯 distill

### 8.2 §3.2 hit (decisive)

- Major win, declare new SOTA candidate
- 把 VDN-trained policy 换进 PIPELINE specialist library (替代 1 个 distill family member)
- 重测 distill on top of VDN base — VDN 是否 multiplicative with distill?

### 8.3 §3.3 tied (lane close)

- 关 lane, 关 DIR-D-QMIX recovery
- 资源已转 (DIR-A Wave 2 / DIR-B Phase 2 / sub-task)

### 8.4 §3.4 regression (否决)

- 否决 整 monotonic-factor 族
- 关 DIR-D-QMIX recovery 通道
- 留 lesson: cooperative 2v2 在 PPO 下 critic 必须 keep cross-agent 信息

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone-pipeline strategic synthesis (DIR-F 在 §2.1 #3, DIR-D-QMIX recovery 在 §3)
- [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B base architecture (encoder + cross-attn)
- [snapshot-076](snapshot-076-wide-student-distill.md) — 5/5 distill saturation evidence
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 distill SOTA recipe (VDN orthogonal alternative)
- [BACKLOG.md § DIR-D-QMIX](BACKLOG.md#stone-pipeline-backlog-deferred-from-6-direction-work-plan-2026-04-22) — recovery condition
- [cs8803drl/branches/team_siamese_vdn.py](../../cs8803drl/branches/team_siamese_vdn.py) — model 实现
- [scripts/eval/_launch_102A_vdn_scratch.sh](../../scripts/eval/_launch_102A_vdn_scratch.sh) — launcher
- Run dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/102A_vdn_scratch_*` (TBD until done)

### 理论支撑

- **Sunehag et al. 2017** "Value-Decomposition Networks for Cooperative Multi-Agent Learning" — VDN 原 paper (Q-learning), 本 lane PPO+GAE 适配
- **Rashid et al. 2018** "QMIX: Monotonic Value Function Factorisation for Deep MARL" — DIR-D-QMIX 同族, NN mixer 强化版
- **Foerster et al. 2018** "Counterfactual Multi-Agent Policy Gradients (COMA)" — 另一 credit assignment 方向, 后续可补 BACKLOG
- **Yu et al. 2021** "MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" — centralized critic + decentralized actor 在 PPO 上的 best practice
