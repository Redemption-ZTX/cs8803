# Pipeline: from zero to `055v2_extend@1750`

This document explains how the submitted agent in this package grew from the
initial project baseline into `055v2_extend@1750`.  It is intentionally written
as a decision history: what problem each step saw, what choice was made, what
changed in the model or training recipe, and what the measured outcome meant.

The short version is:

```text
scratch PPO / reward shaping
  -> BC bootstrap
  -> per-agent stabilized PPO
  -> 029B per-agent SOTA
  -> 031A team-level Siamese dual encoder
  -> 031B cross-agent attention
  -> 034E deploy-time ensemble
  -> 055 single-network distillation
  -> 055v2 recursive distillation
  -> 055v2_extend@1750 final submitted checkpoint
```

The honest current reading is also important: early notes treated
`055v2_extend@1750` as a `0.915-0.916` model.  Later large-sample validation
corrected that.  Its true per-episode win rate against the course baseline is
about `0.9066-0.907`, not `0.916`.  It remains the submitted anchor because it
is the most validated, low-variance, self-contained, H2H-competitive checkpoint,
and it clears the course autograder protocol.

## 1. Evaluation doctrine

The project repeatedly learned that small evaluation samples are dangerous.
The final pipeline should be read with this doctrine in mind.

| Metric | Meaning | Use |
|---|---|---|
| `baseline 50` | inline training eval | discovery only; too noisy for verdicts |
| `official 500` | official evaluator over 500 episodes | useful screening |
| `official 1000` | official evaluator over 1000 episodes | normal checkpoint comparison |
| combined `2000-5000+` | multiple independent samples | needed for SOTA claims |
| H2H | direct policy-vs-policy match | catches baseline specialists |
| failure capture | saved losing episodes + bucket analysis | explains why a model wins or loses |

One of the biggest late-stage corrections was the `1750` selection-effect
walk-back.  `1750` was promoted from an 8-checkpoint sweep.  For a noisy
1000-episode evaluator, choosing the best of 8 candidates creates an expected
positive bias of roughly `mu + 1.42 sigma`.  With `sigma_1000ep ~= 0.012`, that
is around `+0.017`, almost exactly the difference between the early `0.9155`
claim and the later `0.9066` fresh validation.

Methodological consequence:

- Stage 1 `1000ep` results are discovery, not verdicts.
- Near the `0.90-0.91` plateau, a true SOTA claim needs at least `3000ep`, and
  preferably `5000ep`.
- H2H remains necessary because some models score well against baseline but
  lose to frontier peers.

## 2. The milestone ladder

| Stage | Representative | Core decision | Result / meaning |
|---|---|---|---|
| Early shaping | v1/v2 shaping | Add low-magnitude dense rewards without drowning sparse goals | PPO became trainable but stayed below SOTA |
| BC bootstrap | `017@2100` | Use `ceia_baseline_agent` as teacher, then PPO fine-tune | early champion around `0.842` |
| Stable role binding | `025b@080` | Keep field-role mechanism, tighten PPO updates | `0.842`, stronger H2H than `017` |
| Per-agent extension | `029B@190` | PBRS warm point -> v2 shaping handoff | `0.868`, per-agent SOTA |
| Team-level architecture | `031A@1040` | Replace flat joint MLP with Siamese dual encoder | true WR about `0.860`, H2H beats old frontier |
| Cross-agent attention | `031B@1220` | Add 4-token bidirectional cross-attention | `0.882`, strongest scratch single model |
| Deploy-time ensemble | `034E` | Average frontier policies at inference | `0.890` 1000ep, strong H2H, but not final intelligence |
| Distillation Gen 1 | `055@1000/1150` | Compress 3-teacher ensemble into one 031B student | `0.902` then `0.907`, new single-model SOTA |
| Recursive distill | `055v2@1000` | Add `055@1150` and LR-diverse `056D` to teacher pool | `0.909`, tied plateau |
| Extended horizon | `055v2_extend@1750` | Continue recursive distill to 2000 iters, select stable checkpoint | true mean about `0.907`; final anchor |

## 3. Stage 0: basic PPO and reward shaping

The project began with the normal Soccer-Twos problem: two agents, each with a
336-dimensional ray-cast observation, must act cooperatively in a 2v2 soccer
environment.  The sparse goal reward is too delayed to make scratch PPO easy.

The first meaningful engineering task was not to invent a large model.  It was
to make the training and evaluation loop honest:

- align the runtime reward shaping with the single-player environment wrapper;
- make sure the official evaluator really measured baseline win rate;
- stop trusting training reward or 50-episode spikes as final evidence;
- save failures so the team could classify why episodes were lost.

The v2 shaping recipe became the stable reward backbone:

| Term | Scale / threshold | Purpose |
|---|---:|---|
| time penalty | `0.001` | prefer finishing instead of stalling |
| ball progress | `0.01` | reward moving ball toward opponent goal |
| opponent progress penalty | `0.01` | punish allowing opponent advance |
| possession bonus | `0.002` | encourage controlled ball contact |
| deep-zone outer penalty | threshold `-8`, penalty `0.003` | avoid being trapped in own half |
| deep-zone inner penalty | threshold `-12`, penalty `0.003` | stronger anti-collapse signal |

The design principle was conservative: shaping should guide PPO, not replace the
goal reward.  This got the project from "can learn something" into usable
policies, but it did not by itself approach the `0.90` target.

## 4. Stage 1: imitation learning and the first real baseline

The next decision was to use `ceia_baseline_agent` as a teacher.  This was
available, queryable, and exactly aligned with the grading opponent.  The first
BC data pipeline collected baseline self-play trajectories and trained a
behavior cloning policy.

The pure BC policy learned baseline-like behavior but did not beat baseline:

- pure BC vs baseline: about `0.554`;
- pure BC vs random: about `0.974`.

The important conclusion was not "BC is enough."  It was:

> BC gives a teacher-aligned starting point; PPO can spend its budget improving
> soccer behavior instead of rediscovering basic movement from random actions.

That led to BC -> MAPPO/PPO fine-tuning, represented by `017@2100`, which became
an early champion around `0.842`.

A negative result from this phase shaped later work: across many interventions,
the `low_possession` failure bucket stayed in a narrow `22-28%` band.  That
meant BC improved overall execution but did not remove the structural
coordination bottleneck.

## 5. Stage 2: `025b`, role binding, and PPO stabilization

After `017`, the project tried field-role binding.  The idea was to give the two
players slightly different incentives, such as striker-like and defender-like
behavior based on spawn depth.

`025` showed that the mechanism did not immediately destroy the champion
checkpoint, but it also produced frequent numerical instability:

- high-frequency `kl` / `total_loss = inf`;
- TensorBoard warnings about NaN/Inf;
- training still sometimes looked usable, which made the bug easy to misread.

The `025b` decision was deliberately narrow:

> Do not change the role-binding mechanism.  Only reduce PPO update violence.

The changes were:

| Hyperparameter | `025` | `025b` | Reason |
|---|---:|---:|---|
| LR | `3e-4` | `1e-4` | reduce warm-start shock |
| SGD iterations | `10` | `4` | avoid over-optimizing each batch |
| minibatch size | `1024` | `2048` | reduce noise |
| PPO clip | `0.20` | `0.15` | constrain policy step |

Outcome:

- bad iterations dropped from `84/200` to `5/200`;
- official baseline 500 peak remained `0.842`;
- H2H showed `025b` beat `017`.

The interpretation was subtle.  `025b` did not solve the failure structure, but
it improved stability and peer robustness.  It became a real champion candidate,
not merely a baseline specialist.

## 6. Stage 3: `029B`, the last strong per-agent step

`029` was designed as a three-lane test of what was left in the per-agent
architecture:

| Lane | Hypothesis | Outcome |
|---|---|---|
| `029A` | stronger `025b` base can release PBRS value | rejected |
| `029B` | PBRS-learned behavior can be handed back to v2 shaping | partial hit |
| `029C` | opponent pool can improve peer robustness | weak positive |

The winner was `029B@190`:

- official baseline 500: `0.868`;
- H2H vs `025b@80`: `0.492`, statistical tie;
- H2H vs `017@2100`: `0.512`;
- H2H vs PBRS specialist `026B@250`: `0.562`.

This switched the per-agent SOTA chain to:

```text
017@2100  ->  025b@080  ->  029B@190
0.842         0.842         0.868
```

The mechanism was not "PBRS plus v2 gives clean additive gain."  Failure analysis
showed that the original PBRS low-possession benefit was partly washed out by
v2 fine-tuning.  The better reading is:

> `029B` found a better v2 local optimum because it started from a different
> PBRS-trained phase point.

This was the practical end of the per-agent path.  It raised the ceiling, but
still remained short of `0.90`.

## 7. Stage 4: `031A`, the architecture pivot to team-level control

The project then changed the modeling assumption.  Earlier team-level models
used a flat MLP over concatenated observations:

```text
concat(obs_agent_0, obs_agent_1)  ->  MLP  ->  joint action
672 dim                              512,512
```

The issue is that the model must infer from data that the first 336 dimensions
and the second 336 dimensions are two copies of the same kind of object: a
player's egocentric observation.  It also has to survive teammate slot swaps.

`031A` introduced a Siamese dual encoder:

```text
obs0 336 -> shared encoder -> feat0
obs1 336 -> shared encoder -> feat1
concat(feat0, feat1) -> merge MLP -> joint action
```

Decision logic:

- the two agent observations are physically and semantically homologous;
- sharing the encoder forces slot-invariant feature learning;
- the merge layer can focus on relationships and coordination;
- the model should be more sample-efficient than a flat MLP.

Results:

- first 1000ep peak around `0.867`;
- rerun average around `0.860`;
- H2H vs `029B@190`: `0.552`, significant;
- H2H vs `025b@080`: `0.532`;
- H2H vs comparable team-level `028A@1060`: `0.568`.

This was the first model to beat the old frontier on both baseline and peer axes.
The project learned:

> Explicit team-level structure matters more than another per-agent reward tweak.

## 8. Stage 5: `031B`, cross-agent attention

Once `031A` was validated, `031B` added cross-agent attention.  The first design
idea was degenerate: if the encoder output is treated as one token, attention
collapses because softmax over one item is always `1.0`.

The corrected design split the 256-dimensional feature into four 64-dimensional
tokens:

```text
feat0 -> 4 tokens x 64
feat1 -> 4 tokens x 64

agent0 tokens attend to agent1 tokens
agent1 tokens attend to agent0 tokens

concat(feat0, attn0, feat1, attn1) -> merge -> policy/value
```

This gives each player a low-cost channel for asking, "which part of my
teammate's encoded state matters right now?"

Results for `031B@1220`:

- official baseline 1000: `0.882`;
- H2H vs `029B@190`: `0.584`;
- H2H vs `025b@080`: `0.566`;
- H2H vs `036D@150`: `0.574`;
- H2H vs `031A@1040`: `0.516`, not significant.

Interpretation:

- compared to per-agent frontier, `031B` is decisively stronger;
- compared to `031A`, the attention gain is real on baseline but modest in H2H;
- `031B` became the scratch single-model architecture root for the final agent.

## 9. Stage 6: `034E`, deploy-time ensemble as a teacher-discovery step

At `031B ~= 0.882`, the project still needed a path past `0.90`.  The first
ensemble idea was inspired by PETS-style model ensembles: different policies
should make different mistakes, and probability averaging might reduce failure
variance.

The first per-agent triad failed:

```text
029B + 025b + 017 -> 0.806 baseline 500
3 x 029B control  -> 0.850 baseline 500
```

This told us the ensemble framework could run, but the chosen members'
probabilities interfered with each other.

The successful frontier ensemble shifted to stronger, more complementary assets.
The most important result was `034E-frontier`:

```text
031B@1220 + 036D@150 + 029B@190
```

Results:

- baseline 500: `0.904`;
- baseline 1000: `0.890`;
- H2H vs `031B@1220`: `0.596`;
- H2H vs `029B@190`: `0.590`;
- H2H vs previous `034C`: `0.544`.

But `034E` was not the final answer.  It was a deploy-time ensemble, meaning it
needed multiple model forwards and averaged policies at runtime.  The project
framing treated that as stability optimization, not single-network intelligence.

Its real value was as evidence:

> Frontier policies have partly complementary failure modes.  If their behavior
> can be compressed into one model, the result may be a true single-policy gain.

## 10. Stage 7: `055`, distilling ensemble behavior into one policy

`055` converted the ensemble idea into a single trained network.  The student
used the `031B` architecture.  The teacher ensemble used:

```text
031B@1220 + 045A@180 + 051A@130
```

The loss was:

```text
PPO loss + alpha(t) * KL(student || teacher_avg_probs)
```

Main settings:

| Item | Value |
|---|---:|
| student arch | 031B Siamese + cross-attention |
| KL alpha | `0.05 -> 0.0` |
| decay | `8000` updates |
| temperature | `1.0` |
| LR | `1e-4` |
| PPO clip | `0.15` |
| train batch | `40000` |
| minibatch | `2048` |
| SGD iterations | `4` |
| reward | v2 shaping |

The critical point is that the teacher average was only a target distribution.
The student still optimized environment reward.  This allowed it to exceed a
literal runtime average.

First strong result: `055@1000`.

- original 1000ep: `0.904`;
- rerun 1000ep: `0.900`;
- combined 2000ep: `0.902`;
- H2H vs `034E` teacher: `0.590`;
- H2H vs `031B` base: `0.638`;
- H2H vs `043A` peer: `0.622`.

This was the first clean single-model result over `0.90`.  The most important
mechanistic conclusion:

> A fixed ensemble averages experts unconditionally.  The distilled student can
> condition on observation and learn when to behave like which teacher.

Later sweep result: `055@1150`.

- combined 2000ep: `0.907`;
- H2H vs `031B`: `0.620`;
- H2H vs `029B`: `0.696`;
- H2H vs `025b`: `0.702`;
- H2H vs `056D`: `0.536`, marginal / not significant.

`055` is the largest real conceptual jump in the project after `031B`.

## 11. Stage 8: `055v2`, recursive distillation

After `055`, the next hypothesis was generational:

> If `055` successfully compressed the first teacher ensemble, then adding
> `055@1150` back into the teacher pool might let a new student learn a cleaner
> version of the compressed knowledge.

`055v2` used a five-teacher pool:

```text
055@1150
031B@1220
045A@180
051A@130
056D@1140
```

Changes from `055`:

- teacher pool expanded from 3 to 5;
- `055@1150` became a recursive teacher;
- `056D@1140` added LR-diverse behavior;
- LR changed to `3e-4`;
- architecture, KL schedule, temperature, PPO clip, and v2 reward stayed the same.

Initial results looked like a possible breakthrough:

- Stage 1: `0.906`;
- Stage 2 rerun: `0.922`;
- combined 2000ep: `0.914`.

But the third sample was `0.898`, so the corrected combined result was:

```text
0.906 / 0.922 / 0.898 -> combined 3000ep = 0.909
```

Final interpretation:

- recursive distillation is SOTA-tier;
- adding `055` and `056D` does not hurt;
- teacher pool expansion alone does not decisively beat `055`;
- the project has likely entered a `0.907-0.909` plateau.

## 12. Stage 9: `055v2_extend`, why training was extended

The next idea was not a new architecture.  It was an iteration-budget question.
The `055v2` student had a more complex teacher signal, so perhaps the standard
`~1250` iteration horizon was too short.

The lane was resumed and extended to 2000 iterations.  The first 1000ep sweep
over late checkpoints showed:

| Checkpoint | Single-shot 1000ep |
|---:|---:|
| 1830 | `0.923` |
| 1860 | `0.921` |
| 1750 | `0.917` |
| 1850 | `0.914` |
| 1250 | `0.913` |
| 1270 | `0.912` |
| 1960 | `0.911` |
| 1710 | `0.911` |

This looked like a real late-window plateau above the `055` ceiling.  But
because the project had already learned how noisy peak selection can be, all
eight top peaks were rerun.

The 8-peak rerun:

| Peak | Stage 1 | Rerun 2000ep | Combined | Reading |
|---:|---:|---:|---:|---|
| 1250 | `0.913` | `0.908` | `0.910` | reverted |
| 1270 | `0.912` | `0.905` | `0.908` | reverted |
| 1710 | `0.911` | `0.902` | `0.905` | reverted |
| 1750 | `0.917` | `0.916` | `0.9163` | survived |
| 1830 | `0.923` | `0.903` | `0.910` | reverted |
| 1850 | `0.914` | `0.896` | `0.902` | reverted |
| 1860 | `0.921` | `0.901` | `0.908` | reverted |
| 1960 | `0.911` | `0.904` | `0.906` | reverted |

At that point, `1750` looked special because it was the only peak that did not
mean-revert in the 2000ep rerun.  It was promoted as the new project SOTA.

The original mechanism hypothesis was:

- training past the teacher horizon lets the student further internalize the
  multi-teacher soft targets;
- `1750` might be a stable late optimum rather than a lucky checkpoint;
- extending recursive distillation was therefore useful.

This was directionally useful, but the exact `0.916` number later proved too
optimistic.

## 13. Final correction: what `1750` really means now

Later meta-analysis and fresh validation corrected the claim.

Fresh n=5000 for `055v2_extend@1750`:

```text
0.923 / 0.906 / 0.903 / 0.889 / 0.912
mean = 0.9066
```

The correction:

- early `0.915-0.916` was a selection-effect high estimate;
- the true mean is about `0.9066-0.907`;
- this is still frontier-level and consistent with `055@1150` and `055v2@1000`;
- no later model clearly beats it on both baseline and H2H.

The current SOTA-level plateau is best understood as:

```text
true frontier mean ~= 0.906-0.909
typical 1000ep samples can range roughly 0.89-0.93
```

This is why a single `0.92` evaluation is not a breakthrough by itself.

## 14. Why the submitted agent is still `055v2_extend@1750`

Even after the correction, `1750` remains the primary submission anchor because:

- it is one of the most validated checkpoints in the project;
- it is robust in H2H against later challengers;
- it has lower variance than many single-shot alternatives;
- it is directly packaged in this folder as a self-contained inference agent;
- it cleared the course autograder protocol:
  - `10/10` vs Baseline;
  - `10/10` vs Random;
  - `10/10` vs TA competitive agent.

The submitted code in `agent.py` reconstructs the `031B`-style
Siamese-cross-attention policy and loads:

```text
checkpoint_001750/checkpoint-1750
```

No runtime ensemble is needed.  The final package is a single network.

## 15. What actually drove the gains

The biggest gains did not come from one hyperparameter.  They came from three
structural decisions.

### 15.1 BC bootstrap

BC changed the starting point.  Instead of learning movement and basic baseline
behavior from scratch, PPO began near a reasonable policy and improved from
there.

Main gain:

```text
scratch / early MAPPO ~0.78
BC fine-tuned 017     ~0.842
```

### 15.2 Team-level Siamese + cross-attention

The architecture changed the problem from two loosely coupled per-agent
decisions into one coordinated team action.  Siamese encoders made teammate slots
share semantics; attention gave each agent a learned coordination channel.

Main gain:

```text
029B per-agent       0.868
031B team x-attn     0.882
```

The gain looks modest on baseline, but H2H showed a much stronger qualitative
shift against prior per-agent policies.

### 15.3 Distillation

Distillation converted ensemble diversity into a single conditional policy.
This was the largest late-stage jump:

```text
031B scratch ceiling       ~0.880-0.882
055 distilled student      ~0.902-0.907
055v2 / 1750 true plateau  ~0.907-0.909
```

The student beat its teacher ensemble in H2H because it was not forced to average
all teachers at every state.  It could learn state-dependent behavior.

## 16. Why the pipeline plateaued

Many later ideas reached the same frontier but did not break it:

- more teacher-pool variants;
- temperature sweeps;
- wider students;
- per-ray attention;
- VDN-style critic decomposition;
- strong-opponent warm-starts;
- baseline-specialist anchor removal;
- Stone/layered specialists.

The repeated pattern was:

- baseline axis sometimes gives `0.91-0.92` single-shot;
- larger samples pull the mean back to `~0.906-0.909`;
- H2H often reveals that a baseline-specialist is weaker than the anchor.

The best current explanation is that the project is near a ceiling created by
the combination of:

- 336-dimensional ray-cast observation;
- factored `MultiDiscrete([3,3,3])` actions per agent;
- the `031B`-family encoder/coordination architecture;
- the `ceia_baseline_agent` opponent distribution;
- v2 shaping basin.

Further progress would likely require changing the observation/action bottleneck
or policy abstraction, not simply adding another teacher or another reward tweak.

## 17. Final lineage summary

The clean lineage of the submitted model is:

```text
v2 shaping
  -> BC bootstrap
  -> 017@2100
  -> 025b@080
  -> 029B@190
  -> team-level architecture exploration
  -> 031A@1040 Siamese dual encoder
  -> 031B@1220 Siamese + cross-attention
  -> 034E teacher ensemble
  -> 055@1000 / 055@1150 distillation
  -> 055v2@1000 recursive distillation
  -> 055v2_extend@1750 final packaged checkpoint
```

The conceptual lineage is even shorter:

```text
learn the game
  -> imitate the baseline
  -> stabilize PPO
  -> switch from per-agent to team-level coordination
  -> add cross-agent attention
  -> use ensembles to discover complementary behavior
  -> distill that behavior into one network
  -> validate and package the most stable late checkpoint
```

## 18. Source map

Primary documents used to reconstruct this pipeline:

- `../docs/experiments/rank.md`
- `../docs/experiments/snapshot-010-shaping-v2-deep-zone-ablation.md`
- `../docs/experiments/snapshot-012-imitation-learning-bc-bootstrap.md`
- `../docs/experiments/snapshot-017-bc-to-mappo-bootstrap.md`
- `../docs/experiments/snapshot-025b-bc-champion-field-role-binding-stability-tune.md`
- `../docs/experiments/snapshot-029-post-025b-sota-extension.md`
- `../docs/experiments/snapshot-031-team-level-native-dual-encoder-attention.md`
- `../docs/experiments/snapshot-034-deploy-time-ensemble-agent.md`
- `../docs/experiments/snapshot-055-distill-from-034e-ensemble.md`
- `../docs/experiments/snapshot-061-055v2-recursive-distill.md`
- `../docs/experiments/snapshot-111-meta-reflection-and-stronger-opp.md`
- `../report/main.tex`

