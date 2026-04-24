# SNAPSHOT-109: Peter Stone pipeline completeness audit

- 日期: 2026-04-23
- 状态: paper-vs-implementation gap audit (meta-snapshot, no own lane)
- 前置: [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) (6-DIR strategic synthesis), [snapshot-100](snapshot-100-dir-A-heuristic-selector.md)/[101](snapshot-101-dir-B-layered-phase1.md)/[103](snapshot-103-dir-A-wave3-sub-task-specialists.md)/[107](snapshot-107-stone-layered-p2-passdecision.md)/[108](snapshot-108-stone-layered-l2-improvement-paths.md) (DIR verdicts)
- WebFetch source: arXiv 2412.09417v1 HTML (Wang/Stone/Hanna 2025) — sub-policy table + selector rules extracted; Stone-Veloso 2000 PDF unreachable (404 / binary), used snapshot-099/107 transcribed summary

---

## 1. Per-paper audit

### Paper 1 — Stone & Veloso 2000 "Layered Learning"

| Aspect | Paper | Our impl | Gap | Effort | Critical? |
|---|---|---|---|---|---|
| **Layer 1** ball interception | trained as individual sub-skill, target ~95% success on isolated interception | 101A@460 = 0.851 baseline WR (full 2v2 vs random, ball-control reward, scenario=full game not isolated) | We measure "full-game baseline WR" not "isolated interception success rate". 0.851 ≠ ~95% on the SPL Layer-1 metric | 0 LOC if we accept the gap; ~80 LOC + 4h GPU to add a controlled "interception success" eval harness | **N** — close enough as a competent warm source per `feedback_stone_layered_warm_source.md` |
| **Layer 2** pass decision (CMUnited DT-tree on top of L1) | trained on top of frozen L1 individual skill; output = pass-target choice | 104A_passdecision (snapshot-107) launched 2026-04-22, jobid 5037129, ~3h ETA — warm-from-101A + KL anchor + pass_subtask scenario + new event_pass reward | (a) we do **not freeze L1 as teammate** (R4 in 107) — both agents trained simultaneously with shared policy = "Layered Lite" not Stone original "Layered Full". (b) we use shared-policy PPO continuous output, not Stone's discrete "decision tree over pass-targets" | 0 (in flight); +200 LOC + 8h GPU for true Layered Full (per-agent + frozen teammate via `multiagent_team` lane) | **Y** for "Stone faithfulness" sell, **N** for capability — Lite path already validated by 103A-wd v2 |
| **Layer 3** team strategy / positional play | composed on top of L1+L2 frozen | NOT designed. snapshot-099 §6 Stage 4 placeholder; snapshot-107 §8.1 conditional on §3.1 HIT | Whole layer missing. Concrete: warm from 104A + new "team_coord_subtask" scenario (formation, off-ball positioning) + frozen L1+L2 teammate | ~250 LOC env + 12h GPU + new snapshot | **Y** if we want full 3-layer story; **N** for SOTA — diminishing returns vs encoder-saturation ceiling per snapshot-108 §2 |
| **Composition at deploy** | hierarchical: top-layer policy is the run-time policy, lower layers baked into action space / behaviour cloning | Our 103A-wd v2 + 104A do single-policy deploy (not hierarchical) | We never compose by HIERARCHY — only by SELECTOR (DIR-A heuristic, all closed) or single-policy deploy | covered by PIPELINE Stage 2/4 below | partially handled |

### Paper 2 — Stone, Sutton, Kuhlmann 2005 "Keepaway SMDP"

| Aspect | Paper | Our impl | Gap | Effort | Critical? |
|---|---|---|---|---|---|
| Sub-task benchmark (Keepaway 3v2) | controlled benchmark for evaluating MARL | We use full 2v2 game WR as proxy | scenario_reset (interceptor / pass / dribble) is closest analogue but does not isolate to a 3v2 keepaway-style task | ~100 LOC scenario init + 4h GPU eval-only | **N** |
| SMDP options + Sarsa(λ) | macro-action options with linear function approx | DIR-E option-critic frozen experts (snapshot-105) closed regression 0.655; PPO+GAE already does multi-step credit assignment | Drop was justified per snapshot-099 §1.3. Modern PPO subsumes SMDP-options on standard MDP | n/a | **N** |

**Drop verdict re-confirmed**: paper 2 has no actionable extraction beyond the keepaway-style sub-task framing, which we already cover via scenario_reset in 103-series.

### Paper 3 — Stone & Veloso 1999 "TPOT-RL"

| Aspect | Paper | Our impl | Gap | Effort | Critical? |
|---|---|---|---|---|---|
| Opaque-transition credit | partition state-action space; assign credit through long opaque transitions | None implemented | DIR-D dropped → BACKLOG as DIR-D-QMIX. Recovery condition (DIR-F VDN positive lift) not met (snapshot-099 §3.2; DIR-F closed sub-SOTA 0.893) | ~600 LOC QMIX mixer + 6h GPU | **N** — recovery gate failed |

**Drop verdict re-confirmed**. Worth noting we never implemented the actual TPOT-RL value partitioning (only the upgrade path QMIX which is also closed). If we wanted to be faithful we would need the opaque-transition credit assignment in PPO advantage estimation — large eng cost, very speculative lift, GAE already covers most of it.

### Paper 4 — Wang, Stone, Hanna ICRA 2025 (arXiv 2412.09417)

WebFetch result (HTML version) revealed the **exact selector design and per-policy state/action spaces** that snapshot-099/100 had handwaved:

**Their 4 sub-policies (per Table I of the paper)**:

| Policy | Action space | Observation space | Train scenario |
|---|---|---|---|
| Mid-field | ΔΘ kick angle (1-D) | ball, can-kick, goal center, goalposts, field sides, last 3 ball positions | 1v0 AbstractSim |
| Ball Duel | ΔX, ΔY, ΔΘ (egocentric vel) | ball, can-kick, closest teammate, goalposts, field, last 3 ball positions | 2v0 AbstractSim |
| Near-goal | ΔX, ΔY, ΔΘ | ball, opponent goalposts, last 3 ball positions | 1v0 high-fidelity SimRobot |
| Positioning | ΔX, ΔY, ΔΘ, **Stand** | ball, strategy position, all defenders, goalposts, field, last 3 ball positions | abstract sim |

**Their selector** (mutually exclusive condition-based switching, no FSM):
- Positioning: teammate estimated closer to ball AND upright
- Near-goal: agent near ball AND inside opponent goal box
- Ball Duel: opponent within 0.5m of ball
- Mid-field: default

**Comparison to our implementation**:

| Aspect | Their paper | Our v_selector_phase4 | Gap | Effort | Critical? |
|---|---|---|---|---|---|
| 4-phase selector | mutually exclusive geometric rules | mutually exclusive geometric rules from 336-d ray (BALL_TAG=0) | shape-equivalent ✅ | done | — |
| Per-policy **own state space** | each policy has different obs (e.g., Near-goal sees only ball + opp goalposts) | **all our specialists use full 336-d ray obs** | We never trained a specialist with masked obs | ~120 LOC per masked-obs encoder + 4h GPU per specialist (4 specialists = 16h GPU) | **Y** — likely a real source of their lift (specialization through input bottleneck) |
| Per-policy **own action space** | each policy has different action shape (kick-angle vs egocentric-vel vs Stand) | **all our specialists output Discrete(3×3×3) MultiDiscrete** | We never trained a specialist with constrained action space | ~80 LOC per action mask + 4h GPU each | **Y** — same reason |
| Train-time **scenario isolation** | 1v0 / 2v0 / 1v0-high-fid (controlled simulators) | scenario_reset in 103A/B/C (full 4-agent game with placement init); 101A vs random only | We do init-perturbation but agents always coexist with full team. Closer to "warm-start + scenario placement" than "trained in isolated 1v0/2v0" | ~150 LOC env wrapper for true 1v0/2v0 + 4h GPU each | **Y** for the BALL_DUEL specialist (their best lift came from 2v0 isolation) |
| Selector type | hand-coded, mutually exclusive | hand-coded, mutually exclusive (Wave 1) | shape-equivalent ✅ | — | — |
| **No DAGGER / no BC** | RL only (PPO via SB3) | RL only (PPO via RLlib) | shape-equivalent ✅ | — | — |
| **No learned selector** | hand-coded, no NN router | DIR-G MoE (closed regression), DIR-E OC (closed regression) | We tried learned selectors; their paper does NOT — alignment with paper says STOP trying learned selectors | n/a | re-validates DIR-G/E drop |

**Misalignment summary for paper 4**: our DIR-A heuristic selector is structurally faithful, but our **specialists are NOT** — every single 103-series specialist (103A/B/C, 081, 101A) was trained on the **full 336-d obs and full MultiDiscrete action space**. Wang et al. attribute their gains substantially to per-policy obs/action restriction, which we have never tested.

---

## 2. PIPELINE 5-stage gap analysis

### Stage 1 — Specialist library

**Done**: 1750 SOTA, 055@1150, 029B@190, 081 NEAR-GOAL, 101A Layer-1 ball-control, 103A INTERCEPTOR (and v1/resume/wd v1+v2), 103B DEFENDER (marginal), 103C DRIBBLE (marginal).

**Missing**:
- **Goalkeeper / near-own-goal defender** specialist — Stone CMUnited had explicit GK; we have none. ~200 LOC scenario + 4h GPU.
- **Set-piece / kickoff** specialist (ball restart from center) — never built. Likely low EV (kickoff is ≤2% of game state) → **defer**.
- **Positioning specialist with obs bottleneck** (paper 4 Positioning has its own obs space) — never trained. ~120 LOC + 4h GPU.
- **Pass-decision specialist** (104A) — IN FLIGHT (snapshot-107).

Coverage: ~70% of paper-4 specialist roles. Goalkeeper is the only structurally-missing role with concrete game-state coverage value.

### Stage 2 — Selector network choice

**Tested**: DIR-A heuristic (Wave 1-4 closed, best variant tied 1750), DIR-G MoE (closed regression), DIR-E option-critic (closed regression), DIR-H cross-attn (closed sub-SOTA W3 untrained, regression W2 trained).

**Missing**:
- **Re-test heuristic with refined specialists** — every selector was tested on v1 specialists which Wave 3 §7C proved 0/5 net-positive. We have not re-tested the heuristic with 103A-wd v2 (0.92 single-shot, 0.9157 combined — strictly better than pre-fix specialists).
- **Paper-4-faithful selector** would only need refined specialists, no new selector design.

Stage 2 is **not selector-bottlenecked**; it is **specialist-bottlenecked**.

### Stage 3 — Critic plug-in

**Status**: never explicitly tried.

What it would mean in our codebase: a **centralized critic** V(s_team) that scores joint state and provides advantage to whichever sub-policy was selected at step t — instead of each sub-policy carrying its own critic baked in at warm-from-1750 time. Implementation = new RLlib `CentralizedCriticModel` over team-obs concat (~400 LOC, 6h GPU to retrain). Closest existing analogue: DIR-F VDN's joint V = V_0+V_1, which closed sub-SOTA 0.893. Critic plug-in for selector ensemble has no precedent verdict in our project, but the VDN result is moderate evidence against.

### Stage 4 — Layered training compose

**In progress (Lite)**: 104A pass-decision warming from 101A. This is **Layered Lite** (snapshot-107 §R4) — both agents share the warm policy, no frozen teammate.

**Missing for "Layered Full"**:
- Per-agent policy split (current `team_vs_baseline_shaping` is shared-policy)
- Frozen-101A teammate while training new agent — requires the `multiagent_team` lane (which exists in `train_ray_base_ma_teams.py`) plus a frozen-policy injection mechanism
- Phase 3 "team coordination" scenario + reward — undesigned

Effort to true Layered Full: ~300 LOC (frozen-policy injection in MA lane) + 12h GPU. Phase 3 design: ~150 LOC + 12h GPU + new snapshot.

### Stage 5 — DAGGER bootstrap

**Status**: never started; BACKLOG Phase 6.

What it actually means in our context: **online expert query**. While the student policy plays, when it enters a state where the heuristic selector says "phase X" but the student's chosen specialist is "phase Y" (or where the student's action diverges from expert's action by some KL threshold), we record (state, expert_action) labels and add them to a supervised loss alongside PPO. Expert candidates: 103A-wd v2 @400 (current best, 0.9157) or 1750 SOTA. Implementation = ~500 LOC (DAGGER buffer + label generation in env worker + BC loss head) + ~6h GPU.

**Note on alignment**: paper 4 (Wang/Stone/Hanna) does **not** use DAGGER. Stone-Veloso 2000 also does not — they use frozen-layer composition. So Stage 5 is a *cross-paper synthesis* (DAGGER from Ross 2011), not a Stone-paper extraction. Calling it "Stone Stage 5" is mildly inaccurate — should be relabeled "DAGGER refinement (orthogonal to Stone)".

---

## 3. Critical synthesis — minimum work to PIPELINE V1

Ranked by EV/cost (highest first):

1. **Re-run DIR-A heuristic ablation A-F with refined specialists** (103A-wd v2 @400 in BALL_DUEL slot, 081-narrow in NEAR-GOAL, 101A in POSITIONING, 1750 default). Wave 3 §7C verdict was based on v1 specialists, all sub-frontier. Cost: 0 LOC + 6 × 1000ep evals = ~6h on free node. EV: +0.005 to +0.015 if any single specialist now nets positive. **No new training**.

2. **Wait for 104A (snapshot-107) verdict**, ~3h ETA. Then if §3.1 HIT (≥0.92), add to specialist library and re-run #1. If §3.5 regression, abandon Layered Phase 2 path and route P1 (continue 103A-wd to 900 iter, snapshot-108 §4) instead.

3. **Per-policy obs/action space restriction** (paper-4-faithful retrain of 103A-INTERCEPTOR and 103B-DEFENDER with masked obs). Highest-EV "missing-from-paper" item. Cost: ~250 LOC + 8h GPU. EV: +0.005 to +0.020 if specialization-via-bottleneck reproduces Wang et al.

4. **Layered Full (per-agent, frozen-101A teammate)** — only if 104A Lite hits. Cost: ~300 LOC + 12h. EV: +0.000 to +0.010 (marginal beyond Lite).

5. **DAGGER refinement** with 103A-wd v2 as expert. Cost: ~500 LOC + 6h. EV: +0.000 to +0.010, but **not Stone-faithful** — sell as Ross 2011 cross-paper. Defer until #1-#3 verdict.

6. **Phase 3 team-coordination layer + Goalkeeper specialist** — only justified if PIPELINE V1 ≥ 0.92 closes ≥ +0.010 over 1750 真值, otherwise diminishing returns vs encoder saturation (snapshot-108 §2).

**Minimum to claim "PIPELINE V1 full Stone pipeline ran"**: items #1 + #2 verdict + write a snapshot-110 PIPELINE-V1-results doc. Estimated 12h elapsed (mostly waiting for 104A and parallel evals). Cost: ~50 LOC of selector preset wiring + 0 new training.

**Minimum to claim "we replicated Wang/Stone/Hanna 2025 faithfully"**: add item #3. ~250 LOC + 8h GPU.

---

## 4. Misalignments to reconsider

1. **Wang/Stone/Hanna per-policy obs/action restriction** (paper §III, Table I) — we never replicated this and it is plausibly the source of their reported gains. **All 5 sub-task lanes** (103A/B/C, 081, 101A) used full 336-d obs + full MultiDiscrete action. This is the single largest misalignment.
2. **Stone Layered "frozen teammate"** — we did Layered Lite (shared-policy warm + KL anchor), not Layered Full (per-agent + frozen Layer N-1 as teammate). Acknowledged in snapshot-107 §R4 but not yet executed.
3. **Stone-Veloso 2000 Layer 1 success metric** — paper measures isolated sub-task success rate (~95% in CMUnited); we measure full-game baseline WR (0.851). Per `feedback_stone_layered_warm_source.md` 0.851 was deemed competent enough to warm L2, and 103A-wd v2 confirmed this empirically — but the metric mismatch means we cannot directly cite "Layer 1 met Stone's threshold".
4. **DIR-G/E learned selector vs paper 4 hand-coded** — paper 4 explicitly does NOT learn the selector. Our DIR-G/E negative results (snapshots 104, 105) are *consistent* with paper 4's design choice. We should retire learned-selector exploration unless we have a Stone-orthogonal motivation.
5. **PIPELINE Stage 5 labelled "DAGGER" as Stone-pipeline** — DAGGER is Ross-Gordon-Bagnell 2011, not Stone. Re-label in snapshot-099 §6 to avoid attribution drift.

---

## 5. Next-launch recommendation

**LAUNCH 1 (today, zero training cost)**: Re-run DIR-A heuristic ablation A-F with refined specialists. Edit `agents/v_selector_phase4/agent.py` to add a new `ablation_v2_103Awd` preset that swaps the BALL_DUEL slot to 103A-warm-distill v2 ckpt-400. Run 6 × 1000ep evals on free node (~6h parallel-3). Expected verdict in snapshot-100 §7G (new section, append-only). Decision rule: any preset Δ vs 1750 baseline > +0.005 → trigger PIPELINE V1 launch with that specialist mix.

**LAUNCH 2 (after 104A verdict)**: Conditional on snapshot-107 §3.1 HIT (104A ≥ 0.92), launch combined PIPELINE V1 with {1750 default, 081 NEAR-narrow, 104A BALL_DUEL/MID, 103A-wd v2 fallback}. Verdict goes in new snapshot-110.

**Do NOT launch**: any new training-time specialist before LAUNCH 1 verdict. Specialist library is not the binding constraint — selector-vs-refined-specialist re-evaluation is. The paper-4-faithful obs/action-restriction retrain (#3 above) is justified only after LAUNCH 1 + 2 close out.

**Bottom line**: we are roughly **65%** of the way to a "full Stone pipeline" — Stages 1+2 covered (with caveats), Stage 4 in progress (Lite), Stages 3+5 untouched but lower marginal EV. The single highest-EV outstanding item is **specialist-refresh re-test of the DIR-A selector**, costing only eval time.
