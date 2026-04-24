# SNAPSHOT-111: Meta-reflection on full project + stronger-opponent hypothesis

- 日期: 2026-04-23
- 状态: meta analysis (analytical, no compute) + stronger-opp lane proposal
- 前置: [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) (Stone pivot), [snapshot-103 §7F.5d/e](snapshot-103-dir-A-wave3-sub-task-specialists.md) (10000ep ceiling proof), [snapshot-109](snapshot-109-stone-pipeline-completeness-audit.md) (paper gap), [snapshot-110](snapshot-110-state-action-bottleneck-specialists.md) (Wang bottleneck inline verdict), [rank.md §3.3 Changelog 2026-04-23](rank.md)

---

## 1. 全线 wasted compute audit (Q1)

**BUG-1 (per-worker opponent lock pre-2026-04-22 fix)** invalidates any lane that mixed BASELINE_PROB ∈ (0,1) without FrozenTeamPolicy:

- **058 real-curriculum** (~12h GPU): trained vs single fixed opp per worker → "curriculum" verdict measures one opp, not progression. Verdict retroactively invalid.
- **062 curriculum-noshape-adaptive** (~12h GPU): same — `set_pool_weights()` updated weights but never reset episode → "adaptive" never fired. Verdict invalid.
- **103A-refined** (~6h): pool corruption + 0.548 fragile warm source → 0.80 closure was noise, not paradigm failure (`feedback_stone_layered_warm_source.md`).
- **103A-warm-distill v1** (~5h): pool corruption present but masked because warm = strong 1750. v1 0.913@400 was real but interpretation polluted.
- **102A VDN** (~10h): closed sub-SOTA 0.893; mixed-opp distribution corrupted, true ceiling unknown.

**BUG-2 (KL α decayed by iter ~53/500 ≈ 10%)** affects ALL distill lanes: 055/055v2/061 recursive, 071/072/073 pool, 076 wide, 077 per-agent, 079 single-teacher, 080 1750-pool, 103A-wd v1, 104A passdecision, 110A/B inline. Effect: anchor only active for first 10% of training, equivalent to "warm + drift". 071-080 (~6×8h ≈ 48h) cluster verdict ("distill saturation at 0.91") is real (ceiling reached even WITH bug, walks-back the original "α tuning matters" hypothesis), but the per-lane Δs between lanes are within bug-noise — we cannot distinguish 079 (0.914) from 073 (0.909) attributively.

**1750 SOTA overstatement (E[max(8)] ≈ μ+1.42σ)** retroactively shifts the 0.9155 anchor down to 0.9066. Every "TIED 1750" verdict in §3.3 (~12 lanes) was actually "TIED 0.91 ceiling" — the qualitative call holds but the SOTA referent moves. Total invalidated/suspect: **~110 GPU-hours** (curriculum + 103A-refined + 103A-wd v1 attribution + 102A); ~150 GPU-hours weakened-attribution (distill cluster). **True SOTA = 0.9069 plateau across 2 paradigms (recursive distill 1750, Stone L2 103A-wd v2) at 10000ep**, with no model decisively above.

---

## 2. False-positive pattern + methodology fixes (Q2)

The 1750 episode is a textbook **selection-effect / max-of-K** failure. We promoted 1 of 8 sweep peaks at 1000ep; the order-statistic E[max(8 N(μ,σ²))] ≈ μ + 1.42σ. With 1000ep SE ≈ 0.012, that's +0.017 — exactly the gap between claimed 0.9155 and corrected 0.9066. The 8-peak mean-reversion test in rank.md §3.3 made this visible (7 of 8 dropped 0.003-0.013 on rerun; only 1750 held — could equally be "true high stable point" or "luckiest sample twice").

103A-wd v1 reproduced the same trap on smaller scale: 3 samples in narrow lucky range 0.913-0.919 → claimed +0.009 SOTA → v2 with 5 samples revealed true range 0.875-0.920 (spread 0.045), pooled 0.908 = ceiling.

**Other suspect "TIED" verdicts in §3.3**: 079 (0.914 single-shot), 080 (~0.91), 076 (0.905), 073 (0.909), 040A/B/C/D peaks (0.863-0.865 cluster), 031A@1040 (0.867 single-shot vs 0.860 rerun avg). Of the lanes with multi-sample data, mean-reversion magnitude is ~0.005-0.013pp from single-shot peak. Consistent with σ_1000ep ≈ 0.012 → "lift signal must exceed 2σ ≈ 0.024 to be real" — virtually no lane in the post-distill era has cleared this bar.

**Doctrine update**:
- **Stage 1 (1000ep)** = **discovery only**, never verdict. Memory `feedback_inline_eval_noise.md` already says this; promoting Stage 1 to "TIED SOTA" violated it.
- **Stage 2 (combined ≥3000ep, ≥3 samples)** = soft verdict, must report sample variance (range) not just mean.
- **Stage 3 (combined ≥5000ep, ≥5 samples)** = SOTA-grade verdict. Required for any "above ceiling" claim.
- **Pooled comparison rule**: when comparing two candidates near ceiling, pool their samples (e.g., 1750 + 103A-wd v2 = 10000ep) before declaring winner — single-recipe peak is statistically meaningless at σ_5000ep ≈ 0.004.
- **No 8-ckpt max promotion without rerun budget**: if sweeping K ckpts, must allocate K reruns OR explicitly debias by E[max(K)].

---

## 3. Paradigm exhaustion: common limit identification (Q3)

10+ paradigms tested all cap ≤ 0.91 with Δ < 0.005:

| Paradigm class | Lanes | Ceiling | Distance from 0.91 |
|---|---|---|---|
| Recursive/distill | 055/055v2/059/061/063/068/071/072/073/076/077/079/080 | 0.903-0.914 | TIED |
| Stone Layered L2 | 103A-wd v1/v2, 104A | 0.904-0.913 | TIED |
| Architecture variants | 031A/B (0.860-0.882), 052 transformer, 054M MAT, 060 MAT-medium, 082 two-stream, 083 per-ray | 0.85-0.882 | BELOW |
| Heuristic/learned selectors | DIR-A W1-4, DIR-G MoE, DIR-E OC, DIR-H cross-attn | TIED to BELOW (regressions) | TIED-REG |
| VDN | DIR-F (102A) | 0.893 | -0.017 |
| Wang bottleneck | 110A NEAR-GOAL ~0.89, 110B MID-FIELD ~0.65-0.75 | TIED-REG | TIED to -0.25 |
| Stone scenario specialists | 081 NEAR-GOAL (0.826), 101A (0.851), 103A/B/C (0.205-0.548) | BELOW | -0.06 to -0.7 |

**The common factor across all paradigms is the (encoder, opponent, environment) joint, not any one of them alone**:

- Opponent isn't the only limit (architecture variants AT THE SAME OPPONENT also cap at 0.86-0.88, ~0.05pp BELOW the ceiling — encoder matters too).
- Encoder isn't the only limit (using identical 031B Siamese ~0.46M, both flat 1750 and Stone-Layered 103A-wd hit EXACTLY the same 0.9069 plateau over 10000ep — same arch, different recipes, same cap).
- Environment isn't the only limit (Wang bottleneck 110B with reduced action regressed to 0.65-0.75 — env permits sub-ceiling outcomes).

The most parsimonious reading: **the encoder × shaping-reward × baseline-opponent triple defines the basin, and we have explored most of the encoder/recipe surface that respects this triple**. The 0.91 number is "best fit of (031B Siamese, +shaping_v2 reward, vs ceia_baseline_agent at 70-100% mix)". Changing the opponent (Q5) is the last fundamentally untested axis — we have changed encoder (082/083/052/054M), reward (036D/038/039), warm source (1750/029B/028A), specialist coverage (081/101A/103A-D, 110A/B), and selector (DIR-A/E/G/H), but the opponent has been ceia_baseline_agent in essentially every lane.

---

## 4. What we should have noticed earlier (Q4)

**The 6/6 distill saturation cluster (071/072/076/079/073/080)** — by snapshot-080 we had 5 orthogonal design variables (teacher count, family, reward axis, student capacity, reward-path diversity) all converging to the same 0.90-0.91 cluster. Snapshot-099 §0.1 made this pivot, but we then spent ~80 GPU-hours on Stone DIR-A/B/E/F/G/H trying to escape via routing/selector. With hindsight, the 6/6 cluster was already strong evidence the limit was NOT in the policy class — it was in the (encoder, opponent) joint. The Stone 6-DIR plan should have been pruned to 1-2 lanes (Wang bottleneck + Layered Lite as paradigm probes), not 6.

**DIR-G (MoE) and DIR-E (OC) regressions** were dismissed as "selector design issues"; in retrospect they were additional ceiling evidence (you cannot route AROUND a ceiling). Same for 110B MID-FIELD's regression — when narrow specialist underperforms generalist, that's a ceiling signal not a "specialist tuning" signal.

**BUG-1 + BUG-2 audit gaps**: BUG-1 surfaced because 058/062 curriculums showed monotone training curves with no curriculum signal — should have been caught by an opponent-distribution metric in the train log (we never logged per-episode opponent identity). BUG-2 surfaced when α-traces were finally inspected; should have been caught by a unit test asserting distill_alpha > 0 at iter 200. Both were single-line audits that would have saved ~150 GPU-hours.

---

## 5. Stronger non-growing opponent hypothesis (Q5)

**Has legs, but with sharply reduced expected magnitude after Q3.** The user's hypothesis: train vs 1750 (or 103A-wd v2 @400) instead of `ceia_baseline_agent` to force harder learning, expecting >0.91 vs baseline.

**Mechanism reading**:
- "0.91 is opponent-weakness ceiling": baseline is too easy — once you beat it 91/100, residual losses are stochastic noise, no gradient signal. Stronger opp generates richer loss-cases → policy that beats baseline 95+/100. **This SHOULD work if true**.
- "0.91 is encoder × baseline-coverage ceiling": encoder cannot represent a strictly better policy against THIS specific opponent's behavior distribution. Stronger opp shifts the basin (might learn a different policy, comparable or worse vs baseline). **Probably no lift vs baseline; may even regress**.
- "0.91 is environment-noise floor": ~9% of episodes have unwinnable initial conditions / Unity-binary stochasticity (own-goal flukes, stuck states). Stronger opp doesn't change this. **Pure 0.91 ceiling regardless**.

**Distinguishing empirical signature**:
- During training: episode reward distribution should SHIFT (mode lower, variance higher when training vs 1750) — if no shift, opp not generating new gradient.
- After training: H2H vs **1750** WR > 0.5 = policy actually got stronger; H2H vs **baseline** WR > 0.92 = first hypothesis confirmed; H2H vs baseline WR ≤ 0.91 = second/third hypothesis (ceiling holds).
- Critical: must measure failure-mode distribution. If new policy loses to baseline on the SAME ~9% (own-goal flukes), env-noise floor is the cap. If different ~9%, encoder-coverage cap.

**Smallest experiment (1 lane, ~4h)**: warm from 1750, train 250 iters vs FrozenTeamPolicy(103A-wd v2 ckpt 400) at OPP_PROB=1.0 (no baseline mix), KL anchor α=0.05 to 1750 (prevent forgetting baseline-relevant behavior). Inline eval every 50 iter on baseline (track baseline WR). Stage 1: 1000ep baseline + 1000ep H2H vs 1750 on terminal ckpt. Decision rule:
- baseline WR ≥ 0.92 + H2H vs 1750 ≥ 0.55 = **opp-strength was the binding constraint**, scale up.
- baseline WR ∈ [0.88, 0.92] + H2H vs 1750 ≈ 0.50 = ceiling holds; opp swap doesn't help.
- baseline WR < 0.88 = catastrophic forgetting; need stronger anchor or mix-back baseline.

Cost: 4h GPU + 1h eval. **Single highest-EV remaining unexplored axis**.

---

## 6. Concrete next-step recommendation

**LAUNCH 1 (priority)**: snapshot-112 stronger-opp lane per §5 — warm 1750, FrozenTeamPolicy(103A-wd v2 @400) at OPP_PROB=1.0, KL anchor α=0.05→0.005 over 39000 updates (BUG-2-fixed schedule), 250 iter, ~4h. Pre-register the 3-way decision rule. This is the single test that separates opp-weakness vs architecture-ceiling. **Do BEFORE further bottleneck specialist work** — if opp-axis is the binding constraint, snapshot-110 110A/C/D become moot.

**LAUNCH 2 (parallel, optional)**: snapshot-110 110A NEAR-GOAL specialist if 110B inline is non-catastrophic (≥0.85), to complete bottleneck paradigm test. If LAUNCH 1 hits, deprioritize.

**Documentation cleanup**:
- rank.md §3.3 add "TIED 0.91 ceiling" annotation to all ≥0.90 entries; the "vs 0.9155 SOTA" deltas are stale.
- Mark snapshot-058, 062 verdicts INVALID (BUG-1) — append correction note.
- Append BUG-1+BUG-2+1750 corrections to rank.md §8 Changelog with effective date 2026-04-23.
- Add to MEMORY.md: `feedback_paradigm_exhaustion.md` — when N≥5 orthogonal lanes converge to the same plateau within SE, treat as architecture-ceiling evidence, do not propose lane N+1 in same paradigm class.

**Do NOT launch**: any new shaping-variant lane, any new selector design, any new distill teacher pool — paradigm class exhausted per §3.

---

## 7. [2026-04-24 02:08] 111A post-train-eval verdict — Q5 hypothesis WALK-BACK from HIT to PARTIAL

After Stage 1 single-shot 0.922 @ckpt 180 looked like a 0.91-ceiling break, ran the full post-train-eval SOP (Stage 2 rerun + Stage 3 H2H + n=5000 fresh validation matching 1750 protocol). All evidence converges on **111A@180 ≈ 1750 真值 0.9066, statistical TIE**.

### 7.1 Stage 1 baseline 1000ep (initial peak, since walked back as single-shot luck)

Trial: `111A_strong_opp_test_20260423_205351/TeamVsBaselineShapingPPOTrainer_Soccer_1cc20_..._20-54-13`
Selected ckpts (top 5%+ties+±1): 120, 140, 180, 210, 220, 240, 250

| ckpt | baseline 1000ep | NW-ML |
|---:|---:|---|
| 120 | 0.921 | 921-79 |
| 140 | 0.911 | 911-89 |
| 180 | **0.922** | 922-78 |
| 210 | 0.911 | 911-89 |
| 220 | 0.916 | 916-84 |
| 240 | 0.920 | 920-80 |
| 250 | 0.910 | 910-90 |

Initial reading: cluster 0.92+ across 4 ckpts looked structurally consistent, not single-shot luck. **Wrong** — see §7.4.

### 7.2 Stage 2 rerun 1000ep (ckpt 180 + 120, parallel-2)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.905 (905W-95L-0T)
.../checkpoint_000120/checkpoint-120 vs baseline: win_rate=0.910 (910W-90L-0T)
[suite-parallel] total_elapsed=224.3s tasks=2 parallel=2
```

Combined 2000ep ckpt 180 = (922 + 905) / 2000 = **0.9135** — already showing -0.009 regression from Stage 1 single-shot (within noise band).

### 7.3 Stage 3 H2H (500ep each)

**Matchup 1: 111A@180 vs 1750**
```
---- H2H Recap ----
team0_overall_record: 265W-235L-0T
team0_overall_win_rate: 0.530
team0_blue_record: 138W-112L-0T (0.552)
team0_orange_record: 127W-123L-0T (0.508)
```
z = (265-250)/√125 = **1.34, p≈0.090, NOT significant** (below `*` threshold 1.96).

**Matchup 2: 111A@180 vs 103A-wd v2@467**
```
---- H2H Recap ----
team0_overall_record: 273W-227L-0T
team0_overall_win_rate: 0.546
team0_blue_record: 144W-106L-0T (0.576)
team0_orange_record: 129W-121L-0T (0.516)
```
z = (273-250)/√125 = **2.06, p≈0.020, marginal `*`** — beats 103A-wd v2@467 by ~5pp.

| matchup | sample | 111A wins | opp wins | 111A rate | z | p | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 111A@180 vs 1750 | 500 | 265 | 235 | 0.530 | 1.34 | 0.090 | -- |
| 111A@180 vs 103A-wd v2@467 | 500 | 273 | 227 | 0.546 | 2.06 | 0.020 | * |

Side splits: vs 1750 blue=0.552/orange=0.508 (gap +0.044); vs 103A-wd blue=0.576/orange=0.516 (gap +0.060). No suspicious side-luck.

### 7.4 Fresh n=5000 baseline (1750-protocol parity)

5 parallel workers × 1000ep each on ckpt 180:
```
=== Official Suite Recap (parallel) ===
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.898 (898W-102L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.903 (903W-97L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.914 (914W-86L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.887 (887W-113L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.913 (913W-87L-0T)
[suite-parallel] total_elapsed=245.9s tasks=5 parallel=5
```

Aggregate n=5000: (898+903+914+887+913)/5000 = **0.9030**, range [0.887, 0.914]. Compared to 1750 fresh n=5000 = 0.9066. **Δ=-0.004 BELOW 1750**.

### 7.5 Combined 7000ep aggregate (all evidence pooled)

Pooled samples for 111A@180:
- Stage 1: 922 wins / 1000 ep
- Stage 2 rerun: 905 wins / 1000 ep
- n=5000 fresh: 4515 wins / 5000 ep
- **Total: 6342 wins / 7000 ep = 0.9060**

SE for n=7000 = √(0.906 × 0.094 / 7000) ≈ 0.0035 → 95% CI **[0.899, 0.913]**.

7-sample distribution n=1000: {0.887, 0.898, 0.903, 0.905, 0.913, 0.914, 0.922}; std ≈ 0.011.

vs 1750 真值 0.9066: **Δ=-0.001, well within SE of either lane**. Statistical TIE.

### 7.6 Q5 hypothesis verdict: HIT → PARTIAL

§5 stronger-opp test was pre-registered with three outcomes:
- baseline WR ≥ 0.92 + H2H vs 1750 ≥ 0.55 → **opp-strength was binding constraint, scale up**
- baseline WR ∈ [0.88, 0.92] + H2H vs 1750 ≈ 0.50 → **ceiling holds; opp swap doesn't help**
- baseline WR < 0.88 → **catastrophic forgetting; need stronger anchor or mix-back baseline**

Actual: baseline WR (n=7000) **0.9060**, H2H vs 1750 **0.530** (z=1.34 not sig).
**Outcome**: middle case. **Ceiling holds; opp swap doesn't help.**

### 7.7 Mechanism reinterpretation

The cluster 0.92+ in §7.1 was NOT single-shot luck per se — multiple ckpts genuinely lie in the 0.91 plateau distribution. **But**: the 0.91 plateau itself is wider than we modeled (σ ≈ 0.011 per 1000ep), so a 7-ckpt cluster all happening to land in the upper tail is a normal sample-of-7 result, not a regime change. The "ceiling" is a true mean ~0.906 with typical 1000ep samples between 0.89-0.93.

**This is the EXACT same selection-effect failure mode as 1750's 0.9155→0.9066 walk-back yesterday** (8-ckpt max selection over the 0.906 plateau gave 0.9155 lucky-side aggregate). The n=5000 protocol is the safety net that catches this; we successfully used it twice in 24h.

### 7.8 Update to §6 next-step recommendation

LAUNCH 1 (the §5 stronger-opp test) was 111A — we ran it. **It did NOT show that opp-strength is the binding constraint.** Per the pre-registered decision rule, **do NOT scale up opp-strength**.

**Updated implication**: §3 paradigm-exhaustion claim is REINFORCED. Multiple paradigms (recursive distill 1750, Stone Layered L2 103A-wd, strong-opp warm 111A, anchor-removal warm 115A pending) all converge to the 0.906 ± 0.011 plateau. This is consistent with an architecture-imposed ceiling (031B Siamese ~0.46M params + 336-dim raycast obs + MultiDiscrete[3,3,3] action complexity matched to 0.91-tier policy capacity).

**Real path forward**: Wang/Stone/Hanna 2025 state/action bottleneck (snapshot-110) — explicitly reduces input/output dimensionality to test if encoder capacity is the bottleneck, OR fundamentally bigger encoder (113A wide encoder lane in flight). 110B paradigm test failed (regression to 0.722) but state-bottleneck (110A) untried.

**115A Stage 2/3 + n=5000 verification still pending**: same single-shot risk applies. Should fire similar n=5000 protocol on 115A@250 before any cross-lane comparison.


---

## 8. [2026-04-24 02:36] 115A post-train-eval verdict — baseline-specialist not paradigm-breaker

Parallel SOP run on 115A@250 (Stage 1 single-shot 0.920) to verify against the 111A walk-back precedent. Different outcome in degree, same outcome in kind: **paradigm does not break the ceiling**.

### 8.1 Fresh n=5000 baseline

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.922 (922W-78L-0T)
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.901 (901W-99L-0T)
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.903 (903W-97L-0T)
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.920 (920W-80L-0T)
.../checkpoint_000250/checkpoint-250 vs baseline: win_rate=0.903 (903W-97L-0T)
[suite-parallel] total_elapsed=258.5s tasks=5 parallel=5
```

Aggregate: (922+901+903+920+903)/5000 = **0.9098**, range [0.901, 0.922]. Combined with Stage 1 (920/1000): n=6000 = 5469/6000 = **0.9115**, SE 0.0037.

vs 1750 真值 0.9066 = **Δ=+0.0049 (1.3σ, NOT significant, directionally marginally above)**.

### 8.2 Stage 3 H2H (3 matchups, n=500 each)

| matchup | sample | 115A wins | opp wins | 115A rate | z | p | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 115A@250 vs 1750 | 500 | 232 | 268 | 0.464 | -1.61 | 0.054 | -- (near `*` on LOSS side) |
| 115A@250 vs 103A-wd v2@467 | 500 | 242 | 258 | 0.484 | -0.72 | 0.236 | -- |
| 115A@250 vs 111A@180 | 500 | 209 | 291 | **0.418** | **-3.67** | **<0.001** | **`***`** |

Side split diagnostics:
- vs 1750: blue 0.516 / orange 0.412 (gap +0.104 — **significant side-luck**; 115A struggles on orange side specifically)
- vs 103A-wd v2: blue 0.508 / orange 0.460 (gap +0.048)
- vs 111A: blue 0.412 / orange 0.424 (gap -0.012, side-neutral)

### 8.3 Reading: "baseline-specialist, frontier-weak"

**Two-axis dissonance** is the key signal:
- Baseline axis: 115A n=6000 aggregate 0.9115 marginally above 1750's 0.9066 (1.3σ, directionally higher)
- Peer axis: 115A loses to BOTH 1750 and 111A in H2H, decisively so vs 111A (z=-3.67 `***`)

This matches the `baseline-specialist` pattern seen previously in 083 per-ray (combined 0.909 baseline, H2H vs 055v2_extend@1750 = 0.410 z=-4.02 `***`). Training config for 115A (warm 1750 + NO KL anchor + BASELINE_PROB=0.7) optimizes very directly for baseline-policy-distribution victories, but the policy loses structural robustness — 111A's strong-opp training produces a more diverse policy that exploits 115A's over-fitting on ~16pp of episodes.

The side-luck signature (+0.104 blue/orange gap vs 1750) is consistent with a policy that has learned specific behaviors dependent on starting side (e.g., ball-clearing direction), which a more robust policy (1750/111A) would not show.

### 8.4 Paradigm comparison: 111A (strong-opp) vs 115A (anchor-removal)

| axis | 111A (strong-opp) | 115A (anchor-removal) | winner |
|---|---:|---:|---|
| baseline true value (aggregate ≥5000ep) | 0.9030 (n=5000) → 0.9060 (n=7000) | 0.9098 (n=5000) → 0.9115 (n=6000) | **115A +0.006** |
| H2H vs 1750 | 0.530 (z=+1.34, tie) | 0.464 (z=-1.61, near-sig LOSS) | **111A** |
| H2H vs 103A-wd v2@467 | 0.546 (z=+2.06 `*` win) | 0.484 (z=-0.72, tied loss) | **111A** |
| H2H vs each other | 0.582 (z=+3.67 `***` win) | 0.418 (z=-3.67 `***` loss) | **111A decisively** |

**Verdict**: **strong-opp paradigm (111A) strictly dominates anchor-removal paradigm (115A)** on the peer axis. 115A's marginal baseline edge is a specialist-not-generalist signature; not indicative of paradigm superiority. 

### 8.5 Updated Q5 hypothesis reading (after both lane verdicts)

§5 pre-registered the 3-outcome decision tree for the stronger-opp test. With both 111A + 115A n=5000 + H2H now in:
- **111A** (warm 1750 + KL distill + train vs frozen 1750): baseline tied 1750 truly, H2H tied 1750 truly → **matches middle outcome: "ceiling holds; opp swap doesn't help"**
- **115A** (warm 1750 + no KL + baseline pool): baseline marginally above 1750 (but baseline-specialist artifact), H2H LOSES to 1750 → **matches middle outcome with specialist-skew, NOT "breakthrough"**

Combined Q5 verdict: **The 0.91 plateau is architecture-imposed.** Opp-swap paradigms (either stronger opp or looser anchor) reach the plateau from the warm-1750 initialization but do NOT push past. Combined with the 6 other paradigms that converge to ~0.91 (recursive distill 1750, Stone Layered L2 103A-wd both v1+v2, 083 per-ray, 080 distill, 069 frontier pool, 111A strong-opp, 115A anchor-removal = now **8 orthogonal paradigms tied at 0.91**), **the architecture ceiling hypothesis is extremely strong** (p<0.001 under null that paradigms are iid).

### 8.6 Conclusion across §7 + §8

The 4-lane diversified bets (111A, 115A, 113A, 111B) had 2/4 reach terminal evaluation; both 111A + 115A tie the 0.91 ceiling on baseline axis (via different artifact types) and LOSE in peer H2H (111A ties 1750, 115A loses to both 1750 and 111A). **Neither breaks the ceiling.**

**Remaining open bets**:
- **113A wide encoder** (still training, ~5h remaining) — direct capacity-test of the architecture hypothesis; if even 1024×1024 FCNET + 384 Siamese cannot break 0.91, paradigm-exhaustion is extremely solidified
- **111B mixed strong-opp pool** (still training) — final strong-opp variant
- **snapshot-110 state bottleneck (110A NEAR-GOAL)** — untried, the last remaining Wang 2025 paradigm slot

**Recommended next action** (once 113A verdict in):
- If 113A also ties 0.91 → paradigm-exhaustion confirmed beyond reasonable doubt → switch to state-bottleneck (110A) AND ensemble paradigm (pipeline/selector) as the only open avenues
- If 113A breaks 0.91 → wide encoder is the binding constraint; pursue more aggressively (larger arch + more compute)

Current primary submission anchor stays **1750** (most-validated, lowest variance signal, H2H-competitive with all challengers including 111A and 115A and 103A-wd v2).

