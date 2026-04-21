## SNAPSHOT-078: DAGGER-Style Two-Stage Distill — 解决 online-distill state-shift (DIR-C)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); blocked on DAGGER infrastructure 工程 (~3-5 天)
- **前置**: [snapshot-055 §4.1 L2](snapshot-055-distill-from-034e-ensemble.md#41-简化-s1a--online-distillation-single-rollout-而非-dagger) (originally planned as retrograde L2) / [snapshot-075 §5.5 / §6 DIR-C](snapshot-075-strategic-synthesis-toward-0.93.md#55-★★★☆☆-dagger-iterative-distillation)
- **同期对照**: [snapshot-076](snapshot-076-wide-student-distill.md) (DIR-A) / [snapshot-077](snapshot-077-per-agent-student-distill.md) (DIR-B)

---

## 0. 背景

### 0.1 Online distill 的理论 bug

所有 current distill lane (055, 055v2, 059, 061, 068, 066, Pool A-D) 都用 **online distillation**:

```
Per rollout step (student self-play vs baseline):
  obs = env_step(student.action)     # student-generated state distribution
  teacher.forward(obs) → teacher_probs  # teacher evaluated on student's rollout state
  KL(student, teacher) → loss
```

**Issue**: teacher 是在 **teacher 自己 rollout distribution** 上训练的 (teacher = 031B/045A/051A 的 self-play trajectories). 当 student 在 iter 500+ 显著偏离 teacher 分布时 (e.g., student 学 ensemble avg 后 策略 mode 不同于 any single teacher), teacher 被 evaluated 在 **out-of-distribution** state:

- Teacher policy 可能 unreliable (学过 → reliable, 未学过 → random-like)
- KL signal 变 noisy or mis-leading
- Student 向 noise-ish teacher 靠拢 → 0.91 plateau

### 0.2 DAGGER (Dataset Aggregation, Ross et al. 2011)

Standard DAGGER loop:
1. 初始 policy π_0 (可以 scratch 或 warm)
2. 用 π_0 rollout → 收集 states
3. 在这些 states 上 query expert/teacher → 收集 (state, expert_action) labels
4. 合并 buffer, supervised train π_1
5. Iterate π_1 → π_2 ...

**Key**: teacher 被 evaluated on **student's rollout state distribution**, 但 teacher 输出的是 **teacher's own optimal action**; student 是在 student-distribution + teacher-label supervised 学。

### 0.3 Why RL distill + DAGGER 合适

- Ross 2011 证明 DAGGER 的 regret bound 比 behavior cloning 好一个 factor (从 O(T²) → O(T))
- RL-DAGGER (Sun et al. 2019, "AggreVaTeD") 把 DAGGER + advantage weighting 推广到 RL setting, 在 model-based RL 和 IRL 上 empirically +2-5% 对 online distill
- 在 soccer 2v2 上 DAGGER 从未测试过

### 0.4 Why this is DIR-C (lowest P 但最 mechanism-解释)

- 076 (wide student) 和 077 (per-agent) 是 **capacity / arch axis** 测试
- 078 是 **state-distribution matching axis** — 直接 probe "online simplification 对 0.91 ceiling 的贡献"
- 如果 078 HIT, 证明 state-shift 是主因素, 为 follow-up combo (076+078, 077+078) 铺路
- 如果 078 tied, 证明 online 简化不是 binding constraint, 保守 path 不走 DAGGER 省 3-5 天工程

---

## 1. Hypothesis H_078

### 1.1 主假设

> H_078 (主): 两阶段 DAGGER-style distill (iter 0-600 online = 055, iter 600-1250 DAGGER-refined KL) 改善 teacher signal quality 在 student 分布上, **combined 2000ep peak ≥ 0.918** (+0.011 over 055 SOTA, > 1.5× SE).

### 1.2 Stretch

> Stretch H_078-stretch: combined 2000ep peak ≥ 0.925 — online state-shift 是 0.91 plateau 的主因, DAGGER 机制完全解决.

### 1.3 子假设

- **H_078-a**: 055 的 0.91 ceiling **有 ~1-2pp 来自 online state-shift** (teacher 在 student-rollout OOD state 上 unreliable, noise-injected KL)
- **H_078-b**: DAGGER 的 supervised KL step (on stored student-rollout states with teacher-labels) 提供 **higher-quality gradient** 比 online, 因为:
  1. Teacher 在这些 states 上 output 被 cache, 不被 student evolution 影响
  2. 多次 epoch 在相同 state-label pair 上 supervise, sample efficiency 提升 3-5×
  3. KL variance 降低 (固定 target vs moving target)
- **H_078-c**: Phase 1 (online) 保证 student 学会 基本 policy shape; Phase 2 (DAGGER) 精修 — 与 PPO 主训练 compatible (non-destructive to value function)

### 1.4 Anti-hypothesis

- H_078-null: online state-shift 不是 0.91 主因; DAGGER 只 marginal +0.002-0.005, engineering cost 不值
- H_078-negative: DAGGER buffer 的 supervised KL 与 PPO on-policy 冲突, 训练不收敛 (PPO 假设 on-policy, DAGGER 是 off-policy supervised; 需要 careful weighting)

---

## 2. Design

### 2.1 Two-Phase Schedule

| Phase | iter range | KL signal source | PPO objective |
|---|---|---|---|
| Phase 1 (Online warmup) | 0 — 600 | On-policy rollout state, teacher forward directly = 055 | standard PPO |
| **Phase 2 (DAGGER refinement)** | **600 — 1250** | **Stored buffer of (state, teacher_probs) pairs from student rollouts iter 400-600** | **PPO + supervised KL on buffer** |

**Phase transition** at iter 600:
1. Pause training at iter 600 (saved ckpt as warm-start)
2. DAGGER step: roll student for 2000 episodes vs baseline, store (obs_t, teacher(obs_t)_probs) for every 5th state (sample ~50K pairs)
3. Resume training: PPO + **online KL (α=0.025) + DAGGER buffer KL (α=0.025)** 同时施加; total α=0.05 保持
4. Every 200 iter 可选 refresh buffer (re-rollout + re-label, Ross 2011 aggregation)

### 2.2 Buffer Structure

```python
dagger_buffer = {
    'states': Tensor (N, 672),       # joint obs from student rollouts iter 400-600
    'teacher_probs': Tensor (N, 729), # 034E 3-teacher ensemble avg probs (frozen at collection time)
}
```

- N ≈ 50,000 pairs (~2000 ep × 25 sampled states per ep)
- Storage: ~0.6GB (float32), fits in GPU memory

### 2.3 DAGGER KL Loss

Per PPO minibatch:
- 50% samples from current online rollout (teacher forward live)
- 50% samples from DAGGER buffer (teacher probs cached)

```python
# Online KL
kl_online = KL(student(obs_rollout), teacher(obs_rollout))
# DAGGER KL
kl_dagger = KL(student(obs_buffer), cached_teacher_probs_buffer)

total_loss = ppo_loss + α/2 * kl_online + α/2 * kl_dagger
```

### 2.4 Teacher + Training Setup

- Teacher: {031B@1220, 045A@180, 051A@130} (034E, same as 055)
- α schedule: 0.05 init → 0 final, 8000 updates decay (same as 055)
- T: 1.0
- LR: 1e-4
- Student arch: 031B-base (0.46M, **same as 055** — isolate DAGGER axis alone)
- v2 shaping, 1250 total iter, budget 14h

### 2.5 Code Changes (~3-5 天)

1. **Buffer management** (`cs8803drl/branches/dagger_buffer.py`, new file, ~150 行):
   - `DaggerBuffer` class: append, sample, save/load
   - `collect_dagger_buffer(student_ckpt, teacher_ensemble, n_episodes=2000, state_subsample=5)` helper
2. **Modify `team_siamese_distill.py`** (CAREFUL file, ADR needed):
   - Add `dagger_buffer_path` custom config
   - Modify KL loss computation: mix 50/50 online + buffer per minibatch
3. **Training script modifications** (CAREFUL file, ADR):
   - Env var: `DAGGER_BUFFER_PATH`, `DAGGER_PHASE_START_ITER=600`
   - At phase transition, pause training, run `collect_dagger_buffer`, resume with buffer
4. **Phase-transition script** (`scripts/eval/_dagger_phase_transition.sh`):
   - Takes Phase 1 final ckpt, runs 2000 ep rollout + teacher forward, saves buffer
5. **Launcher** `scripts/eval/_launch_078_dagger_distill.sh`
6. Smoke: Phase 1 10 iter + buffer collection 100 ep + Phase 2 10 iter, verify KL finite + actions valid

### 2.6 Port / Resource

- PORT_SEED: 78
- Budget: 1 × H100 × 14h for training + 1 × H100 × 0.5h for buffer collection at transition = ~15h total
- 工程 3-5 天 (engineering + smoke); 可与 076/077 并行 dev

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict |
|---|---|---|---|
| §3.1 marginal | combined 2000ep peak ≥ 0.911 | +0.004 (= 055 single-shot noise level) | DAGGER has measurable effect |
| **§3.2 主** | **combined 2000ep peak ≥ 0.918** | **+0.011 (H_078 met)** | **state-distribution matching meaningfully helps distill** |
| §3.3 breakthrough | combined 2000ep peak ≥ 0.925 | +0.018 (stretch) | online state-shift is dominant 0.91 cause |
| §3.4 持平 | combined 2000ep peak ∈ [0.895, 0.911) | within ±2σ of 055 | DAGGER doesn't meaningfully improve vs online |
| §3.5 退化 | combined 2000ep peak < 0.890 | sub-031B | PPO × DAGGER buffer KL off-policy conflict |

**Phase 2 early stop**: 若 Phase 2 first 100 iter (600-700) inline 200ep eval 比 Phase 1 终点 (ckpt_600) 明显退化 (> -0.05pp), 立即关 lane (H_078-negative 验证)

---

## 4. Simplifications + Risks + Retrograde

### 4.1 简化 A — 2 phase only (not iterative DAGGER)

- Full DAGGER: 4+ iterations of collect → supervised → rollout
- 当前: 1-shot buffer collection at iter 600, 1-refresh optional at iter 1000
- **Risk**: 1-shot buffer 过时快, buffer distribution shift 仍存在
- **降级**: L1 add buffer refresh at iter 1000 (+0.5h overhead)

### 4.2 简化 B — 50/50 online/buffer mix (not sweep)

- Could sweep {25/75, 50/50, 75/25, 100/0}
- 当前: 50/50
- **Risk**: 50/50 可能 suboptimal
- **降级**: L2 sweep on best variant

### 4.3 简化 C — buffer 不更新 (static cached)

- Ross 2011 allows teacher re-query on buffer; 我们只 query 一次 at collection time
- **Risk**: teacher_probs cache 过时 (teacher 是 frozen 所以实际 没 drift, OK)
- **Mitigation**: 因为 teacher frozen, cached probs reliable; 只有 student distribution 会漂移, 所以 refresh 逻辑 可选 L1

### 4.4 Risk R1 — PPO × off-policy DAGGER buffer 冲突

- PPO clip 假设 on-policy; buffer 是 off-policy (来自 iter 400-600 student) 
- **Mitigation**: 
  1. Buffer KL 不参与 PPO advantage 计算 (纯作为 auxiliary loss)
  2. PPO clip 只对 online rollout advantage 生效
  3. Buffer KL gradient 与 PPO loss gradient 独立
- **Monitor**: 若 Phase 2 开始 100 iter kl_divergence 从 1.5 突然升到 > 5, 关 lane

### 4.5 Risk R2 — Buffer 50K 过小 / 过大

- 过小: 覆盖率不够, student generalize 失败
- 过大: storage + minibatch sampling 开销
- **Mitigation**: 50K 是 200 iter × 10 episodes × 25 states/ep, 覆盖率估 reasonable; 可 L1 加到 100K

### 4.6 Risk R3 — 工程 3-5 天 up-front; DIR-A/B 可能先 HIT

- **Mitigation**: 并行 — DIR-A (076) 2h launch after smoke, result in 1 day; DIR-B (077) 2 天 dev + 14h; DIR-C (078) 3-5 天 dev + 14h
- 如果 DIR-A HIT §3.2, 可以 DIR-C 暂停 (combo 078+076 后续再 revisit)

### 4.7 全程 Retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default | Phase 1 online 600 iter + Phase 2 DAGGER 650 iter | 15h |
| 1 | Phase 2 退化 (inline < Phase 1 ckpt_600 by >0.05) | 关 lane, H_078-negative 成立 | 5h saved |
| 2 | §3.2 miss but §3.1 HIT | add buffer refresh at 1000 (L1) | +1h |
| 3 | §3.1 miss | online/buffer ratio sweep (25/75 + 75/25) | +28h |
| 4 | full lane all miss | DAGGER 路径否决, 0.91 plateau 不是 state-shift bound | — |

---

## 5. 不做的事

- 不 iterative DAGGER (N=4+, full Ross 2011) — 工程 ≥1 周
- 不改 teacher / LR / T / α (= 055, isolate DAGGER axis)
- 不与 076 / 077 合并同 lane (先独立测 DAGGER axis, 组合 follow-up)
- 不 warm-start phase 1 (scratch as 055)

---

## 6. Execution Checklist

- [ ] 1. snapshot 起草 (本文件)
- [ ] 2. 实现 `cs8803drl/branches/dagger_buffer.py` (DaggerBuffer + collect_dagger_buffer helper)
- [ ] 3. 扩展 `team_siamese_distill.py` 支持 mixed online + buffer KL (需要 ADR 记录 CAREFUL file 改动)
- [ ] 4. 扩展 train script 支持 DAGGER_PHASE_START_ITER + buffer load (需要 ADR)
- [ ] 5. 写 phase transition script
- [ ] 6. Smoke: Phase 1 10 iter, collect 100 ep buffer, Phase 2 10 iter, verify KL 组合 finite + policy entropy 不炸
- [ ] 7. Launcher `_launch_078_dagger_distill.sh`, PORT_SEED=78
- [ ] 8. Full 1250 iter launch (Phase 1 600 iter + transition + Phase 2 650 iter)
- [ ] 9. Phase 2 first 100 iter monitoring: 若退化 > -0.05 vs Phase 1 ckpt_600, 关 lane
- [ ] 10. Stage 1 post-eval 1000ep on top 10 ckpts (选 iter 700-1250 范围, Phase 2 主要对象)
- [ ] 11. Stage 2 rerun top 3 → combined 2000ep
- [ ] 12. Stage 3 H2H: vs 055@1150 (same recipe, no DAGGER — 直接 measure DAGGER effect), vs 031B@1220 (base), vs 034E (teacher)
- [ ] 13. Verdict append §7

---

## 7. Verdict

_Pending_

---

## 8. 后续路径

- **A (§3.2 / §3.3 HIT)**: DAGGER validated — 下一步 stack 078 + 076 (DAGGER on wide student), 078 + 068 (DAGGER + PBRS distill)
- **B (§3.1 HIT)**: marginal gain, follow-up = iterative DAGGER (N=3+ iterations) 测试 ROI 是否爬坡
- **C (§3.4 tied)**: online state-shift 不是 0.91 binding cause; 放弃 DAGGER follow-ups, focus on 076/077 axes
- **D (§3.5 退化)**: PPO × buffer KL conflict 验证, engineering approach 需重新设计

---

## 9. 相关

- [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) — strategic synthesis (DIR-C motivation)
- [snapshot-055 §4.1 L2](snapshot-055-distill-from-034e-ensemble.md) — originally planned as retrograde L2 (已过 10 天未执行)
- [cs8803drl/branches/team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — online distill baseline (需扩展)

### 理论支撑

- **Ross et al. 2011** "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAGGER) — original, O(T) regret bound vs behavior cloning O(T²)
- **Sun et al. 2019** "AggreVaTeD: Aggregated Value-based Imitation Learning" — DAGGER + advantage weighting in RL, empirically +2-5% on MuJoCo tasks
- **Ho & Ermon 2016** "Generative Adversarial Imitation Learning" — alternative approach to state-distribution matching; DAGGER is more directly applicable given we have direct teacher query access

