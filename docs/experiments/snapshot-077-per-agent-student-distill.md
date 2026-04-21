## SNAPSHOT-077: Per-Agent Student Distill — 用 per-agent 架构吸收 team-level teacher ensemble (DIR-B)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); blocked on per-agent distill marginal projection 工程 (~2 天)
- **前置**: [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) / [snapshot-075 §5.3 / §6 DIR-B](snapshot-075-strategic-synthesis-toward-0.93.md#53-★★★★☆-per-agent-student-吸收-team-level-teacher-distill)
- **同期对照**: [snapshot-076](snapshot-076-wide-student-distill.md) (DIR-A) / [snapshot-078](snapshot-078-dagger-distill.md) (DIR-C)

---

## 0. 背景

### 0.1 当前 distill 架构单一性

| Lane | Student arch | Teacher arch |
|---|---|---|
| 055 / 055v2 / 059 / 061 / 068 / 066 / Pool A-D | team-level Siamese+cross-attn (031B) | mixed (031B team-level + 045A team-level + 051A team-level + 029B/036D per-agent) |
| 034E ensemble (deploy) | N/A | {031B team + 036D per-agent + 029B per-agent} |
| 074A-E (deploy ensemble) | N/A | team-level-only |

**所有 execution 过的 distill student 都是 team-level**。034E 的成员 **包含 per-agent teacher** (036D, 029B), 但当 ensemble 作为 teacher 给 055 student 时, student 硬是 team-level — per-agent teacher 的信息流经了 team-level marginal projection + KL-to-team-student, 可能 **lose per-agent-specific inductive bias**。

### 0.2 Empirical gap

- 031B (team-level) 0.880 combined vs 029B (per-agent) 0.846 combined — **+3.4pp team-level 优势**
- 但这个优势是 **arch + reward + warmstart** 混合: 031B scratch 1250 iter + Siamese, 029B per-agent + warmstart + v2
- **Pure arch 对比无数据**: 如果 per-agent student 用 **同 teacher + 同 budget + 同 reward**, 它能不能突破 0.88 ceiling?

### 0.3 Why per-agent student on ensemble teacher might work

- Per-agent 架构 obs 336-dim, team-level 672-dim — **信号 entropy 低 2×**, student 学 teacher policy 的 effective capacity 相对更高
- Per-agent 的 slot-symmetric rollout (每个 episode 2 个 independent 轨迹 contributions) **sample efficiency 2× team-level** (team-level 每 episode 1 个 joint trajectory)
- Team-level teacher ensemble 的 per-agent marginal probs 天然 fit per-agent student 架构
- Hinton 2015 未测 "different-architecture student-teacher", 但 cross-arch distill 在 vision (ResNet → MobileNet) 上被反复 verified 有效

### 0.4 Why this is DIR-B (not DIR-A)

- 相对 076 (wide team student): 需要新工程 (per-agent distill marginal projection: 27-dim per-agent vs 729-dim joint), 已 snapshot-055 §2.3 述及但未 implement
- 相对 078 (DAGGER): 工程成本相近; 但 DIR-B 直接 probe "arch bottleneck" 的 load-bearing 机制问题
- Expected gain 保守 — reward / teacher 不变, 单 arch 切换 possibly only +0.005-0.015; 真 **+0.02 需要 per-agent arch + 其他 axis 叠加**, 本 lane 只是 single-axis test

---

## 1. Hypothesis H_077

### 1.1 主假设

> H_077 (主): per-agent shared-policy student (336-dim egocentric obs, shared Siamese encoder) 吸收 034E 3-teacher ensemble 的 per-agent marginal probs, **combined 2000ep peak ≥ 0.915** (+0.008 over 055 SOTA, > 1× SE).

### 1.2 Stretch

> Stretch H_077-stretch: combined 2000ep peak ≥ 0.925 — per-agent student 避免 team-level 的 joint-obs redundancy, 真正利用 slot-symmetric 2× sample efficiency.

### 1.3 子假设

- **H_077-a**: 034E 3-way ensemble teacher 的 per-agent marginal probs **保留足够策略信息** 供 per-agent student 学习 (per-agent marginal = 3 dims × 3 classes per agent, factor-prob space 已证明 in 055 eq. (2) 稳定)
- **H_077-b**: Per-agent shared-policy student 的 slot-symmetric rollout (2 独立轨迹 per episode) 提供 2× sample efficiency, 1250 iter 足够收敛
- **H_077-c**: Per-agent student 架构上 orthogonal to team-level, 不继承 031B arch bottleneck (如有); 034E 教师的 team-level 成员 (031B, 045A) 的策略被 marginal 投影后, per-agent student 学到的**是 team coordination 的 per-agent local approximation, 可能不是 suboptimal degrade** (理论上 joint policy 的 best-response marginal ≈ per-agent optimal)

### 1.4 Anti-hypothesis

- H_077-null: per-agent student 即使在 distill 加持下, **arch bottleneck** 让 peak cap at ~0.87-0.88 (比 team-level 低 2-3pp), 与 036D 级别 tied, 证明 per-agent 是 inherent inferior arch

---

## 2. Design

### 2.1 Student Architecture — Per-agent Shared Policy

```
Input: 336-dim obs (per agent)

Siamese Per-agent Encoder:
  obs_336 → Linear(336 → 256) → ReLU → Linear(256 → 256) → ReLU → feat_256

Shared-policy head (shared for both agents 0 and 1):
  feat_256 → Linear(256 → 128) → ReLU → policy/value heads
  policy head: Linear(128 → 27)
  value head: Linear(128 → 1)
```

Param count: ~0.35M (与 029B ~0.4M 可比)

**Policy mapping function**: both agents 0 and 1 call same shared policy (shared-policy MAPPO pattern from snapshot-004 / starter). 每 rollout step 独立对每个 agent 调用 `student(obs_agent_k) → action_k`.

### 2.2 Teacher Ensemble Setup (per-agent marginal projection)

```python
# Teacher ensemble: 034E 3-way (031B_team + 045A_team + 051A_team, all team-level)
teacher_ensemble_forward(joint_obs):
    joint_probs_per_teacher = [t.forward(joint_obs) for t in 3 teachers]  # (B, 729) each
    avg_joint_probs = stack(joint_probs_per_teacher).mean(dim=0)  # (B, 729)
    # NEW per-agent marginal projection:
    per_agent_probs_agent0 = marginal_over_agent1(avg_joint_probs)  # (B, 27)
    per_agent_probs_agent1 = marginal_over_agent0(avg_joint_probs)  # (B, 27)
    return per_agent_probs_agent0, per_agent_probs_agent1
```

Marginal projection:
- Factor joint action a = (f0a0, f0a1, f0a2, f1a0, f1a1, f1a2), 3 factors per agent × 3 classes
- Per-agent marginal = sum other agent's factors out
- Implementation: per-agent 27-dim = factor-concat (f0a0_probs, f0a1_probs, f0a2_probs) × (f1a0_probs, f1a1_probs, f1a2_probs)

### 2.3 Distill Loss

Per-agent student student_probs_k = student(obs_k) → (B, 27).

Loss = PPO_loss(student) + α(t) · [KL(student_probs_0 || teacher_margin_0) + KL(student_probs_1 || teacher_margin_1)] / 2

α schedule: 0.05 init → 0 final, 8000 updates decay (= 055)

### 2.4 Training Setup (= 055 where possible)

- LR=1e-4, CLIP_PARAM=0.15, NUM_SGD_ITER=4
- TRAIN_BATCH_SIZE=40000, SGD_MINIBATCH_SIZE=2048
- v2 shaping: same
- MAX_ITERATIONS=1250 (per-agent 2× sample efficiency, 理论上 600 iter 就足, 但保守 1250 match 055)
- Opponent: 100% baseline (= 055)

### 2.5 Code Changes Needed (~2 天)

1. **New file**: `cs8803drl/branches/per_agent_distill.py`
   - `PerAgentSharedPolicyDistillTorchModel(TorchModelV2)` — per-agent Siamese shared-policy student arch
   - `_FrozenTeamEnsemblePerAgentTeacher` — wraps 3 team teachers, does joint_probs → per-agent marginal projection on forward
   - Marginal projection math (factor-probs factorization): ~80 行
2. **train_ray_team_vs_baseline_shaping.py** (CAREFUL file, document in ADR):
   - Branch: 若 `PER_AGENT_STUDENT_DISTILL=1` wire PerAgentSharedPolicyDistill model
   - Per-agent rollout (already supported in MAPPO lane, copy from 029B setup)
3. **New launcher**: `scripts/eval/_launch_077_per_agent_distill.sh`
4. Smoke: 10 iter, verify forward shapes, marginal projection matches teacher joint argmax

### 2.6 Port / Resource

- PORT_SEED: 77 (隔离 076=76 / Pool 70-73 / 068=68)
- Budget: 1 × H100 × 14h
- 依赖: 无 external blocker; 工程 2 天 + smoke

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict |
|---|---|---|---|
| §3.1 marginal | combined 2000ep peak ≥ 0.911 | +0.004 vs 055 (> SE/2) | per-agent distill has measurable effect |
| **§3.2 主** | **combined 2000ep peak ≥ 0.915** | **+0.008 (H_077 met)** | **per-agent arch path validated as SOTA-tier** |
| §3.3 breakthrough | combined 2000ep peak ≥ 0.925 | +0.018 (stretch) | per-agent arch is strictly better for distill |
| §3.4 持平 | combined 2000ep peak ∈ [0.885, 0.911) | within ±3σ of 055 tied, but cross-arch so likely noisy | per-agent distill works but no unique upside |
| §3.5 退化 | combined 2000ep peak < 0.880 | sub-031B tied | **per-agent arch bottleneck hypothesis validated** |

**Early stop rule**: 若 iter 600 inline eval 200ep WR 仍 < 0.60 mean, 关 lane — per-agent student 在 distill 加持下 convergence 若比 team-level 慢 2× 以上则失败

---

## 4. Simplifications + Risks + Retrograde

### 4.1 简化 A — Per-agent shared policy (not independent 2-policy)

- 替代: independent 2 policy, 每 agent 自己的权重
- 当前: shared policy, 2 agents 共用同一 network
- **Risk**: shared 可能丢 slot 特异性 (但 snapshot-004 已证明 shared 在 per-agent 架构上 sample efficiency 更高)
- **降级**: L1 尝试 independent per-agent (+1 day 工程 + 2× params)

### 4.2 简化 B — 034E 3-teacher pool (not 5-teacher)

- 与 055 对齐 isolation; teacher pool 改动已 in 055v2 (+0.002 NS)
- **Risk**: 同 055 — teacher 选择没 sweep
- **降级**: 若 076 wide-student + 3-teacher HIT §3.2, 不在 per-agent 上 test 5-teacher

### 4.3 简化 C — v2 shaping, 不用 PBRS

- 与 055 对齐; 068 测试 PBRS × team-level distill
- **降级**: 若 077 HIT §3.1/§3.2, 组合 lane (per-agent + PBRS) = 079 follow-up

### 4.4 Risk R1 — Per-agent marginal projection 信息损失

- joint policy's per-agent marginal **不等于** per-agent optimal response
- 理论: 如果 joint policy 有 strong correlation between agents (joint_prob[(a0=x, a1=y)] != marginal[a0=x] × marginal[a1=y]), marginal projection 丢失 coordination 信息
- **Mitigation**: 055 用 joint probs KL, 079 可以 follow-up 用 "joint KL through teacher-conditioned student state"
- **Monitor**: teacher joint entropy vs sum of per-agent marginal entropies — 若 gap > 0.3, coordination 被丢了

### 4.5 Risk R2 — Per-agent student 收敛比 team-level 慢 (sample efficiency inverse)

- 虽然 H_077-b claim 2× sample efficiency, 实际 per-agent MAPPO 在 sparse reward 上历史表现 **slower convergence than team-level** (参 snapshot-027 team-level scratch 比 snapshot-014 MAPPO scratch 早 ~200 iter 收敛至 0.82)
- **Mitigation**: 如果 iter 600 不达 0.60, 提前关 lane

### 4.6 Risk R3 — 工程 2 天成本 up-front

- 076 (wide student) 几乎 zero 工程, ROI 更高
- **Mitigation**: 并行 — 76 先 launch (no code 改动); 077 工程进行中; 078 工程进行中; 3 个 lane 串行 verdict

### 4.7 全程 Retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default | per-agent shared student + 034E 3-teacher | 14h train + 2 天 dev |
| 1 | iter 600 inline < 0.60 | 早终止, lane closed | saves 10h |
| 2 | Stage 1 single-shot < 0.886 | close, per-agent arch 否决 | — |
| 3 | §3.4 tied team-level 0.89 | per-agent not strictly better, 但 arch-orthogonal 资产成立; follow-up = per-agent teacher in Pool B | +14h next round |

---

## 5. 不做的事

- 不用 independent 2-policy per-agent (shared is simpler baseline)
- 不用 per-agent teachers only (per-agent 036D + 029B) — 原 034E 混合 teacher 是测试 team→per-agent cross-arch 最干净的
- 不 warm-start (scratch as 055)
- 不混合 PBRS (留给 follow-up)

---

## 6. Execution Checklist

- [ ] 1. snapshot 起草 (本文件)
- [ ] 2. 实现 `cs8803drl/branches/per_agent_distill.py` (PerAgentSharedPolicyDistillTorchModel + _FrozenTeamEnsemblePerAgentTeacher)
- [ ] 3. 扩展 `train_ray_team_vs_baseline_shaping.py` — PER_AGENT_STUDENT_DISTILL branch wiring
- [ ] 4. Smoke test: 10 iter, 检查 student policy_probs shape (B, 27), teacher marginal shape (B, 27), KL finite, action correct Discrete marginalization
- [ ] 5. Launcher `scripts/eval/_launch_077_per_agent_distill.sh`, PORT_SEED=77
- [ ] 6. Full 1250 iter launch
- [ ] 7. iter 600 checkpoint: 若 inline 200ep mean < 0.60, 关 lane
- [ ] 8. Stage 1 post-eval 1000ep on 10 top ckpts
- [ ] 9. 如果 peak ≥ 0.906, Stage 2 rerun → combined 2000ep
- [ ] 10. Stage 3 H2H: vs 055@1150 (team-level SOTA sibling, different arch), vs 029B@190 (per-agent baseline-axis 头名), vs 036D@150 (per-agent learned-reward reference)
- [ ] 11. Verdict append §7

---

## 7. Verdict

_Pending_

---

## 8. 后续路径

- **A (§3.2 / §3.3 HIT)**: per-agent distill path validated — 下一步 per-agent + wide (076 capacity extension), 再下一步 per-agent + PBRS (077+068), 最后 per-agent + DAGGER (077+078)
- **B (§3.1 HIT)**: marginal gain, follow-up = per-agent in Pool B 5-teacher (arch-diverse ensemble 的更 principled 替代, 参 snapshot-072 Pool C 的 arch-diversity 尝试)
- **C (§3.4 tied)**: arch-orthogonal 的 0.907 sibling asset — 可以作为未来 ensemble 成员 (per-agent + team-level-distill 双 blood, 替代 074B arch-diverse 的 failed 尝试)
- **D (§3.5 退化)**: per-agent arch bottleneck 假设验证, distill 路径不走 per-agent, 回主 DIR-A

---

## 9. 相关

- [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) — strategic synthesis (DIR-B motivation)
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 SOTA recipe (teacher setup; 055 §2.3 已述 per-agent marginal projection idea 但未 implement)
- [snapshot-004](snapshot-004-role-ppo-and-shared-policy-ablation.md) — shared-policy per-agent arch baseline
- [snapshot-029](snapshot-029-post-025b-sota-extension.md) — 029B per-agent SOTA reference
- [snapshot-036D](snapshot-036d-learned-reward-stability-fix.md) — 036D per-agent learned reward reference (peer arch axis teacher)
- [cs8803drl/branches/team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — reference for distill wiring (需要 per-agent 镜像实现)

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — cross-architecture distillation in vision successful
- **Rusu et al. 2016** "Policy Distillation" — RL distill cross-arch (DQN → smaller net) empirically effective
- **Czarnecki et al. 2019** "Distilling Policy Distillation" — joint policy marginalization in multi-agent RL 在理论上 preserves best-response under mild correlation assumptions

