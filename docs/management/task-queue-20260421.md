# Task Queue — Ideas Tracking (2026-04-21)

Tracks every experiment direction proposed + its status. When a node becomes free, pick highest-priority pending task from §3.

---

## 1. Previously-proposed directions status

### 1.1 ✅ COMPLETED / RUNNING

| ID | Direction | Status | Where |
|---|---|---|---|
| A | 055 + 053D (replace v2 with PBRS) | 🏃 RUNNING (2 variants) | 068_warm (5028782) + 068_scratch (5028915) |
| B | 055v2 extend to 2000 | 🏃 RUNNING iter 1330+/2000 | 5028776 |
| C | 054M extend to 1750 | 🏃 RUNNING iter 1370+/1750 | 5028757 |
| D | T sweep 完整 (T=1.5/2.0/3.0/4.0) | 🏃 RUNNING (all 4 + 055temp anchor) | various |
| E | 056E/F LR sweep upward (5e-4/7e-4) | 🏃 RUNNING (E only, F stuck) | 5028918 |
| F | Failure capture + H2H for 5 plateau lanes | 🏃 4 cap running, 4 H2H DONE | — |
| G | 066A/B progressive distill | 🏃 066A running; 066B stuck | 5028754 / 5028777 |

### 1.2 ⏸ NOT YET LAUNCHED (pending queue)

| ID | Direction | Priority | Why |
|---|---|---|---|
| **H** | **055v2 + 043 (frontier pool opponent)** | **P1** | User just approved |
| I | 新 ensemble + 新 distill (055+055v2+056D teacher) | P2 | Pending 055v2 extend verdict |
| J | 058-L3 (frontier-pool curriculum) | P3 | BACKLOG |
| K | 055v3 recursive (next iteration) | P4 | Needs 055v2 extend result first |
| L | Logit-level distill | P5 | Speculative |
| M | Multi-model combo selection (see §4) | P2 | Needs Tier 1 data in |

### 1.3 ❌ PAUSED / CLOSED

- 056-PBT-full (user paused, 60h ROI poor)
- 058-L4 PAIRED (too complex)
- 054L/054T (architecture saturated)

## 2. Salvage status (cluster failure 00:58) — 04:52 verified

| Lane | Progress | Job | Status |
|---|---|---|---|
| 055v2_extend | 1410/2000 (70%) | 5028776 | ✅ running |
| 054M_extend | 1440/1750 (82%) | 5028757 | ✅ running |
| 055temp_resume | 1220/1250 (98%) | 5028774 | 🔥 nearly done |
| 063_T30_resume | 790/1250 (63%) | 5028751 | ✅ running |
| 063_T15_resume | 810/1250 (65%) | 5028756 | ✅ running |
| 066A_resume | 960/1250 (77%) | 5028754 | ✅ running |
| 056E_resume | 1030/1250 (82%) | 5028918 | ✅ running |
| 066B_resume | 790/1250 (63%) | 5028917 | ✅ running (revived 04:37) |
| 068_warm | 50/800 (6%) | 5028782 | ✅ running |
| 068_scratch | 40/1250 (3%) | 5028915 | ✅ running |
| 069 055v2+043 | 20/1250 (2%) | 5028750 | ✅ running |
| 070 Pool B | 10/1250 (1%) | 5028916 | ✅ running |
| 066B-old | 0 ckpts after 1h40m | ~~5028777~~ | ❌ killed 04:51 (Ray init hang) |
| 056F_stuck | @970, 1h10m no progress | ~~5028752~~ | ❌ killed 04:57 (Ray init hang) |
| 063_T40_stuck | @370, 1h10m no progress | ~~5028773~~ | ❌ killed 04:57 (Ray init hang) |

**Retries on fresh nodes (launched 04:55-04:57)**:
| Lane | Job | Status |
|---|---|---|
| 056F_retry | 5028749 | ✅ training process live 04:56 |
| 063_T40_retry | 5028753 | ✅ training process live 04:57 |

**PD**: 5028919 / 5028920 awaiting R transition (reserved for Pool A post-eval / post-055v2_extend).

**Ray init hang hypothesis**: possibly node-specific. Stuck processes alive but not producing ckpts. Retry on fresh node may work.

## 3. Next-node dispatch queue (priority order, updated 06:45)

### Blocker 语义重读(user clarification 2026-04-21 06:45)

> **Blocker ≠ 暂停 lane**。意思是"blocker task 的**那次具体运行**的 ckpt 暂不考虑",但 lane 本身仍继续跑,用当前已知的最优 ckpt 作 fallback。

应用到各 pool 的实际状态:
- **Pool A**: 055v2@1000 pre-extend 0.909 combined verified → **可立即启动**(不必等 extend)
- **Pool C**: 054M@1230 pre-extend 0.889 + 055v2@1000 已存在 → **可启动**,extend 是 bonus
- **Pool D**: 跟 Pool B 成员差异来自 068 PBRS;没 068 就退化成 Pool B clone → **真 blocker**,必须等 068
- **074A-E**: 无 blocker,全部 eval-ready

When node frees up, launch next (user directive 04:45: 066B > 056F > 063_T40 for stuck):

1. ✅ **055v2+043** (069) — launched on 5028750
2. ✅ **Pool B divergent distill** (070) — launched on 5028916
3. ✅ **066B revive** — launched on 5028917
4. ✅ **056F retry** — launched on 5028749 (held-srun mode)
5. ✅ **063_T40 retry** — launched on 5028753 (held-srun mode)
6. **Pool A** (after 055v2_extend peak identified, ~2h)
7. **Pool C** (after 055v2 + 054M done)
8. **074 (新 034-next deploy-time ensemble)** 🆕 **P2** — user-requested 2026-04-21 06:05
   - **Motivation**: 034E-frontier (031B+045A+051A) 已被 055 distill 吸收为 teacher (SOTA 0.907)。当前 single-model 主力已在 0.90 plateau,zero-training ensemble 依旧有 overconfidence-reduction 的结构收益(PETS 同构)。034 新一代应用**2026-04-20 以来的新 SOTA 资产**做 deploy-time vote/average,不再限于 031B family
   - **Config 草案**: 3-4 member team-level ensemble,候选成员 **{055@1150, 055v2@peak, 056D@1140, 062a@1220, 053Dmirror@670}** 选 3-4 个(架构相同/reward 不同 → orthogonal failure modes)
   - **Zero training**(只写 wrapper + eval),所以**不占训练节点**,任意空闲节点就能跑 1000ep eval
   - **Dependencies**: 需 055v2@peak (等 055v2_extend 完成,~3h) 来最终决定成员组合
   - **Design**: 待起草 snapshot-074-deploy-time-ensemble-next.md + deploy wrapper (参考 034E-frontier 代码)
9. **Pool D** (after 068 done, ~12h)
10. **077 DIR-B per-agent student distill** 🆕 **READY-TO-LAUNCH** (2026-04-21 engineering complete)
    - **Engineering**: `cs8803drl/branches/per_agent_distill.py` (PerAgentSharedPolicyDistillTorchModel + _FrozenTeamEnsemblePerAgentTeacher) + `cs8803drl/training/train_ray_mappo_vs_baseline.py` `PER_AGENT_STUDENT_DISTILL=1` branch. Smoke test PASSED (teacher loads from 031B@1220 / 045A@180 / 051A@130; distill KL finite; student 27-dim logits marginalize to 3x3 factor probs).
    - **Launcher**: `scripts/eval/_launch_077_per_agent_distill.sh` (PORT_SEED=77, 1250 iter, ~12h)
    - **Hypothesis**: per-agent slot-symmetric student 吸收 034E teacher ensemble; combined 2000ep peak ≥ 0.915 (H_077 主 §3.2)。工程 blocker 已解除,5029745 (held-srun R) 可立即起 lane。
    - **Design**: docs/experiments/snapshot-077-per-agent-student-distill.md (pre-registered thresholds). Note: uses slot-0 convention (own_obs→slot 0) for teacher marginal projection, which is a minimally-invasive simplification vs full joint-KL-marginal.
11. **058-L3 frontier pool curriculum** (backlog P3)
12. **055v3 recursive distill** (backlog P4)
13. **055temp extend → 1750/2000** (低 ROI,延后,见 snapshot-063 §7.4)

## 4. Multi-model combo pools — QUEUED (user approved 2026-04-21 ~04:30)

Selection criteria applied:
1. WR ≥ 0.88 combined (candidates meet threshold)
2. Architectural / reward / optimizer diversity preferred (teachers should capture different modes)
3. H2H distinct where possible
4. Same student arch (031B) for KL compat

### Pool A — snapshot-071 drafted 06:20 (homogeneous distill, 3-teacher)
Teachers: 055@1150 + 055v2@peak + 056D@1140, uniform 1/3 weight, H≥0.915
- ★ Blocker: 055v2_extend peak identification (~3h)
- Ready trigger: 055v2_extend >1900 iter + peak ckpt confirmed via Stage 1 post-eval
- Launcher pending: `scripts/eval/_launch_071_poolA_homogeneous_distill.sh`

### Pool B — launched as snapshot-070 on 5028916 (divergent, 3-teacher)
Teachers: 055@1150 + 053Dmirror@670 (PBRS) + 062a@1220 (curriculum no-shape)
- ★ Already running (~18h ETA)

### Pool C — snapshot-072 drafted 06:20 (cross-axis 4-teacher)
Teachers: 055v2@peak + 056D@1140 + 054M@peak + 062a@1220, weighted 0.3/0.25/0.2/0.25
- ★ Blocker: 055v2 + 054M peaks
- ★ KL-conflict risk明示 (054M cross-agent-attn vs 3 Siamese majority) — weighted scheme damage control

### Pool D — snapshot-073 drafted 06:20 (cross-reward, 3-teacher)
Teachers: 055@1150 (v2 shape) + 068_warm@peak 或 068_scratch@peak (PBRS) + 062a@1220 (no-shape)
- ★ Blocker: 068 (~12h remaining)

### 074 — snapshot-074 drafted 06:20 (ZERO-TRAINING deploy-time ensemble)
**074A 3-way**: 055@1150 + 053Dmirror@670 + 062a@1220 (orthogonal distill + PBRS + curriculum)
**074B 4-way** (conditional): +056D@1140 if 074A HIT §3.2 (≥0.914) 但未到 §3.1 (≥0.920)
- ★ **UNBLOCKED — 所有 ckpts 已存在, 可立即启动**
- Zero training (30 LOC wrapper reuse ProbabilityAveragingMixedEnsembleAgent), ~1h total wall (smoke + 1000ep + random + H2H vs 055)
- ~200MB GPU / 1 CPU, 可跟 training lane 并行,或单占一节点
- **CAUTION**: snapshot-034 §11.4 team-level-only ensemble 曾 -0.1pp anti-lift; main H 可能 miss, alt H (bucket rounding) 是 fallback

### 074B/C/D/E — 🆕 2026-04-21 (drafted by subagent) — more ensemble variants

All share the same zero-training / ~1h wall-clock profile as 074A.
Wrappers already implemented in `cs8803drl/deployment/trained_team_ensemble_next_agent.py`:
`TeamEnsembleNextAgent` (equal or weighted mean-of-probs) and `OutcomePredictorRerankEnsembleAgent` (074E).

| Variant | Members | Angle | Priority | Blockers |
|---|---|---|:-:|---|
| **074A** | 055@1150 + 053Dmirror@670 + 062a@1220 | distill + PBRS + curriculum (blood diversity) | P1 | none |
| **074B** | 055@1150 + **054M@1230** + 062a@1220 | arch diversity (MAT cross-AGENT attn + 2 Siamese) | P2 | MAT register check; 054M@extend peak unverified |
| **074C** | 055@1150 + 056D@1140 + 053Dmirror@670 | H2H-least-correlated (055 vs 056D = 0.536 NOT sig) | P2 | `053Dmirror vs *` H2H 未测 — 选 accept |
| **074D** | 055@1150 + **055v2@1000** + 053Acont@430 | failure-bucket orthogonal (late_def/poor_conv/unclear) | P3 | 055v2 与 055 family 近，可能 decision-correlated |
| **074E** | 074A 成员 + **calibrated v3 outcome predictor** top-K re-rank | predictor-value-head novelty | P2 | predictor load smoke test first |

All 5 launchers are at `scripts/eval/_launch_074{A,B,C,D,E}_ensemble_eval.sh`
(base ports 65005/65105/65205/65305/65405, `-j 1`, 1000ep baseline).
Corresponding snapshots: `snapshot-074-...` (074A), `snapshot-074B-arch-diversity.md`,
`snapshot-074C-h2h-orthogonal.md`, `snapshot-074D-failure-bucket-orthogonal.md`,
`snapshot-074E-predictor-rerank.md`.

**Launch order recommendation (subagent)**:
1. **074A first** — baseline comparison anchor for everything else.
2. **074E second** (if 074A ≥ §3.2) — predictor novelty is the highest-info variant.
3. **074B third** — MAT arch diversity (after MAT register smoke passes).
4. **074C fourth** — if 074A tied, to diagnose H2H-informed selection hypothesis.
5. **074D fifth** — lowest-priority because bucket labels are hindsight signal.

### Launch priority (updated 06:20 + subagent 2026-04-21)
1. **074A first** — UNBLOCKED + zero training + highest info-per-hour
2. **074E** — predictor novelty, novel research contribution
3. **074B** — arch diversity axis
4. **Pool A** — 055v2@peak 后 (~3h)
5. **074C / 074D** — pending decision from A/E verdicts
6. **Pool C** — 055v2+054M peaks 后 (~3h overlap with A)
7. **Pool D** — 068 后 (~12h)

Pool B already running 070 → no separate launch needed.

---

## 5. Protocol for post-training handling (user directive 2026-04-21 04:40)

**Per-lane processing flow** (each lane processed independently, serial):
1. **Wait for lane completion** (.done flag / last ckpt stable)
2. **Merge** (if resume lane): `scripts/tools/merge_training_runs.py` to combine pre-crash + post-resume run dirs → unified view
3. **Global consideration**: look at MERGED view as one training curve, with awareness that **resume-node-transition introduces bias** (KL spike, advantage norm reset, optimizer momentum lost at boundary iter). DO NOT judge based on isolated pre-resume or post-resume segment. Resume's early post-resume iters should be read as "restart warmup" not "continued dynamics".
4. **Post-eval via skill**: Stage 1 (1000ep pick_top on MERGED ckpts) → Stage 2 (rerun + capture) → Stage 3 (H2H per snapshot recommendation)
5. **Doc update BEFORE next action**: update snapshot §7 verdict + rank.md §3.3/§5/§8 + this queue doc. Re-read docs before next lane's processing.

**Parallelism rules**:
- Training: opportunistic (launch queue item when node free and no higher priority pending)
- Post-eval: **serial** (one at a time, not batched)
- Doc update: serial (subagent OK)

**Stuck task priority** (confirmed):
- 066B (mid ROI, 62% partial, 4h to finish) — attempt revive when possible
- 056F (low-mid, 77% partial, 2.5h) — revive only if no better use
- 063_T40 (low, 30% partial, 7.5h) — bottom priority; reasonable to skip

**Judgment principles**:
- DO NOT conclude a lane's verdict from single-shot single-sample data
- Resume boundary effect is REAL — early post-resume ckpts may be worse than peak due to state reset, not "regression"
- Combined 2000ep+ minimum for SOTA claims
- Cross-lane comparison only after all candidate lanes have been properly measured

---
Last update: 2026-04-21 04:52 EDT (verified node→lane mapping, killed 066B-old stuck, identified 2 idle + 2 stuck nodes)

**2026-04-21 later update**: 077 DIR-B engineering complete (§3 #10). Per-agent distill ready-to-launch, smoke test passed.

---

## 6. 2026-04-21 07:30+ EDT status delta

### 6.1 Training lanes — RUNNING (launched since §2 snapshot)

| Lane | Job | First ckpt | Status |
|---|---|---|---|
| **076 wide-student** (DIR-A, arch capacity distill) | **5028920** | **@ 08:39** | ✅ RUNNING |
| **077 per-agent distill** (DIR-B, cross-arch student) | **5029745** | just launched | ✅ RUNNING |
| **079 055v3 recursive distill** (BAN round 2, self-distill follow-up) | **5028757** | first ckpt earlier | ✅ RUNNING |

### 6.2 074 family — ALL 5 VARIANTS CLOSED (5/5 sweep done)

All variants evaluated; family meta-conclusion in [snapshot-074 §7.6](../experiments/snapshot-074-034-next-deploy-time-ensemble.md#76-2026-04-21-0730-edt--074-family-全-5-variants-完成--meta-conclusion-汇总-append-only):
- **074A** {055+053Dmirror+062a} = 0.903 §3.3 tied
- **074B** {055+054M+062a} = 0.877 §3.4 REGRESS (arch-diversity hurts)
- **074C** {055+056D+053Dmirror} = 0.902 §3.3 tied (H2H-orthogonal null lift)
- **074D** {055+055v2+053Acont} = 0.900 §3.3 tied (bucket-orthogonal null lift)
- **074E** 074A + v3 predictor rerank = 0.893 §3.4 marginal regression

**Deploy-time prob-avg ensemble paradigm CLOSED for this project.** 4/5 tied at arithmetic-mean plateau (~0.90), 1/5 regression (cross-arch), 1/5 drag (predictor rerank). Resources 流回 training-side (Pool A/B/C/D + 076/077/079 above).

### 6.3 Post-eval verdicts completed (2026-04-21 07:30 EDT)

| Lane | Verdict | Snapshot section |
|---|---|---|
| **054M_extend** | Stage 1+2+3 DONE: peak 0.904 single / 0.897 combined 1500ep; H2H vs 055 = 0.464 tied (z=-1.61, p=0.054); **架构 axis 重新打开 catch up to 055**, 不 beat | [snapshot-060 §7.2](../experiments/snapshot-060-054M-mat-medium.md#72-2026-04-21-0730-edt--054m_extend-stage-123-verdict-append-only--架构-axis-重新打开但只是-catch-up-to-055) |
| **056F retry** | Stage 1 DONE: peak 0.868 @ 1250 terminal; **sub-SOTA confirmed, lane CLOSED**; LR upward sweep past 3e-4 actively harmful | [snapshot-065 §7.1](../experiments/snapshot-065-056EF-lr-sweep-upward.md#71-2026-04-21-0730-edt--056f-retry-stage-1-baseline-1000ep-verdict-append-only--sub-sota-confirmed-lane-closed) |
| **074C/D/E** | All three verdicts written (see §6.2 above) | snapshot-074C/D/E §7 |

### 6.4 Revised priority queue (post 074-family close, 07:30 EDT)

With 074 family closed and 054M extend verdict in, the next-node dispatch queue tightens:
1. **Pool A** (3-teacher homogeneous distill) — unblocked once 055v2 peak ID stable (in-flight on 5028776)
2. **Pool C** (4-teacher cross-axis, NOW includes 054M@1460 as a tied-SOTA teacher candidate) — still waits for 055v2/054M peaks
3. **Pool D** (cross-reward) — waits for 068
4. **076/077/079** — already launched per §6.1
5. Lower priority: **055v3 speculative, 058-L3, logit-level distill**

**Grading submission primary still 055@1150** (combined 2000ep 0.907). All ensemble directions exhausted.

---

## 7. 2026-04-21 14:55 EDT status delta

### 7.1 Verdicts just completed + documented

| Lane | Verdict | Snapshot §7 link |
|---|---|---|
| **063_T40** (T=4.0) | 4-sample combined 3000ep = **0.9123 tied 055** (SE 0.005); first 3 samples mean 0.919 misled → 4th sample 0.897 corrected; H2H vs 055 = 0.508 NOT sig。**T sweep REMAINS CLOSED**, Pattern C (flat) 强确认 | [snapshot-063 §7.7](../experiments/snapshot-063-055-temp-sharpening.md#77-t40-063_t40-4-sample-combined-3000ep-correction--tied-055-t-sweep-remains-closed-2026-04-21-1455-edt-append-only) / [snapshot-067 §7](../experiments/snapshot-067-temperature-sweep-full.md#7-verdict) |
| **068_warm** | single-shot 0.892 @ iter 780, below 053Dmirror 0.902 (-0.010) and 055 0.907; **warm+distill+PBRS 不 synergize** | [snapshot-068 §7.1.1](../experiments/snapshot-068-055PBRS-distill.md#711-068_warm-stage-1-post-eval-completed-0940-edt) |
| **068_scratch** | single-shot 0.905 dual peak (1140+1180), **§3.4 HIT tied 055 (Outcome C)**; scratch beats warm by +0.013; PBRS ≡ v2 for distill (shape-choice free) | [snapshot-068 §7.1.2](../experiments/snapshot-068-055PBRS-distill.md#712-068_scratch-stage-1-post-eval-completed-1430-edt) |
| **069** (055v2 + 043 frontier pool) | single-shot 0.908 @ terminal 1250, **§3.4 HIT tied 055 +0.001**; frontier-opponent hypothesis partially validated (no regression) but no breakthrough | [snapshot-069 §7](../experiments/snapshot-069-055v2-plus-043-frontier-pool.md#7-verdict) |

### 7.2 7 parallel post-evals LAUNCHED ~14:55 EDT (ETA ~17:40 EDT)

Post-Stage-1 batch on 7 idle nodes (results pending; will be documented upon completion):

| Lane | What | Expected verdict direction |
|---|---|---|
| **055v2_extend** | Stage 1 1000ep on extended ckpts (1210 → 2000) | tied 055v2 0.909 or marginal shift |
| **063_T30** | Stage 1 1000ep (T=3.0 in T sweep) | likely tied (Pattern C confirmation) |
| **063_T15** | Stage 1 1000ep (T=1.5 in T sweep) | likely tied (Pattern C confirmation) |
| **070 Pool B** | Stage 1 1000ep (divergent 3-teacher {055+053Dmirror+062a}) | §3.2 HIT if cross-reward-diversity lifts; else tied |
| **066A** | Stage 1 1000ep (progressive distill BAN round 1 teacher) | §3.2 HIT or tied 055 |
| **066B** | Stage 1 1000ep (progressive distill BAN round 2 teacher) | §3.2 HIT or tied 055 |
| **056E** | Stage 1 1000ep (lr=5e-4 PBT variant) | likely §3.4 tied or mild regression vs 056D 0.891 |

### 7.3 Training lanes status (at 14:55 EDT)

| Lane | Job | Progress | Status |
|---|---|---|---|
| **071 Pool A** (homogeneous 3-teacher distill warm 031B@80) | — | **94%** | 🏃 RUNNING |
| **072 Pool C v2** (cross-axis 4-teacher distill warm 031B@80) | — | **92%** | 🏃 RUNNING |
| **076 wide-student** (DIR-A arch capacity distill) | 5028920 | **85%** | 🏃 RUNNING |
| **079 055v3 recursive distill** (warm 031B@80) | 5028757 | **85%** | 🏃 RUNNING |
| **077 per-agent distill** (DIR-B cross-arch student) | 5029745 | **73%** | 🏃 RUNNING |
| **073 Pool D** (cross-reward warm 031B@80) | — | **52%** | 🏃 RUNNING |
| **T40 Stage 2 capture** (post-stage-1 data collection on 063_T40@1060 peak) | 5028753 | just launched | 🏃 RUNNING (held-srun repurposed) |

### 7.4 Revised submission status

- **Primary grading candidate**: 055@1150 (combined 2000ep 0.907) — unchanged
- **Diversity candidates** (tied SOTA independent recipes): 055v2@1000 (combined 3000ep 0.909), 054M@1460 (combined 1500ep 0.897), 068_scratch@1140 (single-shot 0.905, PBRS recipe), 069@1250 (single-shot 0.908, frontier-pool recipe), 063_T40@1060 (combined 3000ep 0.912, T=4.0 recipe)
- All `within SE of 055` — project SOTA ceiling ~0.91 robust across {v2 shape, PBRS, T=1/2/4, recursive distill, frontier opponent, MAT arch}
- No single axis (shape / T / opponent / arch) produced actionable uplift above 055 — confirms ceiling is distillation-structural (teacher pool + student capacity + KD temp) not reward/opponent/arch-induced

## 8. 2026-04-21 18:30 EDT UPDATE — 🏆🏆 NEW PROJECT SOTA: `055v2_extend@1750` = 0.9163 combined 3000ep (append-only)

- **NEW SOTA**: `055v2_extend@1750` combined 3000ep 0.9163 (Stage 1 1000ep 0.917 + rerun 2000ep 0.916). Δ vs 055@1150 SOTA 0.907 = **+0.009** (z≈1.23, p≈0.11 one-sided — approaching marginal significance but **not yet p<0.05**). Combined 4000ep rerun launched 18:32 (jobid 5032909, port 52505, ETA ~10 min) → will tighten SE to ~0.004 and likely push p<0.05 if win_rate stays ≥0.915.
- **8-peak mean-reversion test**: only iter 1750 survived. 7 others (1250/1270/1710/1830/1850/1860/1960) all reverted -0.003 to -0.013pp. Logs: `055v2_extend_baseline1000.log` + `055v2e_{peak}_rerun2000.log` series + `055v2e_1750_rerun1000v3.log` (pending).
- **Primary grading candidate upgrade recommendation**: switch from 055@1150 (combined 2000ep 0.907) → **055v2_extend@1750** (combined 3000ep 0.9163) pending combined 4000ep confirmation.
- **Teacher-reference implications** for all future Pool A/B/C/D and ensemble experiments:
  - **Pool A (snapshot-071)** currently uses 055v2@1000 (combined 3000ep 0.909) as the recursive-distill teacher. Consider rerun with 055v2_extend@1750 as upgraded teacher (Δ +0.007 over current Pool A teacher).
  - **074 ensemble family (all closed)** — members could be swapped 055@1150 → 055v2_extend@1750 for one zero-training retry. Low probability (anti-lift pattern reproduced 5x) but swap is free.
  - **079 055v3 recursive distill** (currently 85% training) — Stage 2 should include "extend past MAX_ITER horizon" as standard follow-up, per newly-validated iteration-budget principle.
  - **076 wide-student + 077 per-agent + 078 DAGGER** — apply same "extend past teacher's own training horizon" rule after their Stage 1 baseline eval.
- **New informal direction DIR-D** (not yet a full snapshot): **"extend recursive distill past teacher's MAX_ITERATIONS horizon"**. n=1 validated (055v2_extend). Need ≥ 2 independent lanes to confirm; 076/077/079 extend retries are the fastest tests.
- See [snapshot-061 §7.4](../experiments/snapshot-061-055v2-recursive-distill.md#74-2026-04-21-1830-edt--055v2_extend1750--new-project-sota-combined-3000ep-09163-append-only) + [rank.md §3.3 top row](../experiments/rank.md#33-official-baseline-1000frontier--active-points-only) + [snapshot-075 §9](../experiments/snapshot-075-strategic-synthesis-toward-0.93.md#9-2026-04-21-1830-edt-post-hoc-update--055v2_extend1750--new-sota-0916-0907-ceiling-invalidated).
