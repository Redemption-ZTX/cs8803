---
name: post-train-eval
description: Post-training evaluation SOP for a Soccer-Twos lane. Three stages: 1000ep baseline → 500ep failure capture (1-2 ckpts) → 500ep H2H vs snapshot-recommended opponents. After all 3 stages complete, may self-extend (rerun / extra H2H / 2nd capture) but always asks user first. Invoke when user says `/post-train-eval <lane>`, "对 <lane> 做 post-train eval", or "跑 <lane> 的训后评估".
---

# Post-training eval SOP (`post-train-eval`)

Three-stage standard procedure for post-training evaluation of a Soccer-Twos lane. Each stage has decision points where you ask the user. Always include raw script output when writing back to docs (for future verification).

## Inputs

User specifies one of:
- **lane name** (e.g., `040A`, `031B`) → resolve to run dir under `ray_results/`
- **run_dir path** explicit

Optional:
- **node / jobid**: if omitted, run `bash scripts/eval/list_my_gpu_nodes.sh` and ask user which node to use
- **port_shelf** anchor (default: pick from free range, see `engineering-standards.md § Batch / 节点 / 端口 经验集`)

## Lane-type → module mapping

| lane variant | team0_module | team0 env var | opponent module | opponent env var |
|---|---|---|---|---|
| **per-agent** (MAPPO, shared_cc) | `cs8803drl.deployment.trained_shared_cc_agent` | `TRAINED_RAY_CHECKPOINT` (or `TRAINED_SHARED_CC_CHECKPOINT`) | `cs8803drl.deployment.trained_shared_cc_opponent_agent` | `TRAINED_SHARED_CC_OPPONENT_CHECKPOINT` |
| **team-level** (joint policy) | `cs8803drl.deployment.trained_team_ray_agent` | `TRAINED_RAY_CHECKPOINT` | `cs8803drl.deployment.trained_team_ray_opponent_agent` | `TRAINED_TEAM_OPPONENT_CHECKPOINT` |

Identify lane variant from training script: `train_ray_mappo_vs_baseline.py` → per-agent, `train_ray_team_vs_baseline_shaping.py` → team-level.

---

## Stage 1: baseline 1000ep eval

### 1.1 Resolve trial dir

A run dir may have multiple trials (e.g. retried after port collision). Iterate `ray_results/<lane_run>/TeamVsBaseline*/` or `MAPPOVsBaseline*/`. Pick the trial with **most checkpoint dirs**. Tie-break by latest mtime.

If two trials are tied on both criteria, **ask user which one**.

### 1.2 Auto-discover ckpts

**HARD RULE (non-negotiable, do not deviate)**: ckpt selection for 1000ep eval = **`top 5% + ties + ±1 window`**. This is the project's standard checkpoint-selection doctrine ([engineering-standards.md § Checkpoint 选模规则](../../docs/architecture/engineering-standards.md#checkpoint-选模规则)).

- "top 5%" = `ceil(N_ckpts × 0.05)` highest-WR ckpts (always ≥ 1)
- "ties" = include ANY ckpt tied with the cutoff WR
- "±1 window" = include ckpts at adjacent iter steps (e.g., if iter 50 selected, also include 40 and 60), in iteration sequence (not WR proximity)

**Fallback rule (when N_selected < 5)**: short training runs (e.g. 200 iter / 20 ckpts) make top 5% = 1 ckpt, even with ties + ±1 may yield only 3 ckpts — statistically too thin for confident verdict. **If `pick_top_ckpts.py` returns < 5 ckpts, re-run with `--top-pct 10`** (top 10% + ties + ±1). If still < 5, accept that and don't widen further.

**Run** (default uses `win_rate`; specialist lanes via `--metric`):

```bash
# standard (per-agent / team-level grading lane)
python scripts/eval/pick_top_ckpts.py <run_dir>

# specialist 044A spear
python scripts/eval/pick_top_ckpts.py <run_dir> --metric fast_win_rate

# specialist 044B shield
python scripts/eval/pick_top_ckpts.py <run_dir> --metric non_loss_rate
```

**DO NOT widen the rule** because the data "looks variable" or you have a hunch about extra ckpts (e.g., to "top 10%" / wider neighbor window). The 5% rule is **precisely designed for noisy 50ep data** — widening defeats its purpose. If the rule selects N ckpts, test those N, no more.

**DO NOT cap with `--max`** unless GPU time is critically scarce AND user explicitly approves a smaller subset.

**If you think the rule misses important ckpts**: that's gut talking, not data. The doctrine exists to override gut. Trust the rule.

If `>10 ckpts` selected, mention to user as FYI but still proceed unless they object.

### 1.3 Construct + launch eval tmux

Template (copy verbatim, fill placeholders):

```bash
tmux new -d -s eval_<short>_b1000 "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/parallel-logs/<short>_baseline1000
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module <team0_module> \
  --opponents baseline \
  -n 1000 \
  -j 7 \
  --base-port <port_shelf>005 \
  --save-logs-dir docs/experiments/artifacts/official-evals/parallel-logs/<short>_baseline1000 \
  --checkpoint <full path to checkpoint-N> \
  ... [one --checkpoint flag per ckpt iter from §1.2] \
  2>&1 | tee docs/experiments/artifacts/official-evals/<short>_baseline1000.log
read
'"
```

- `<short>`: lane label (e.g., `040A`, `031B`, `044A`)
- `<port_shelf>`: pick free anchor from 50000-63000 range (avoid currently-running lanes; see `list_my_gpu_nodes.sh`)
- ckpt paths: full absolute paths from §1.1 trial dir + §1.2 iter list

Launch via `srun --jobid=<JOBID> --overlap bash <wrapper>` if not on the node.

### 1.4 Wait + parse

- Each ckpt × 1000ep ≈ 4-5 min; with `-j 7` parallel, 7 ckpts ≈ 5-7 min, 14 ckpts ≈ 10-15 min
- Monitor: `tail -f <log>`
- After completion, the recap section starts with `=== Official Suite Recap ===` containing one line per ckpt with `... vs baseline: win_rate=0.XXX (NW-ML-0T)`

### 1.5 Document Stage 1

Append to:
- **rank.md** §3.3 "official baseline 1000" table — only if any ckpt is in frontier-relevant range (within ±0.02 of current SOTA), or if it's a previously-untested lane
- **snapshot-NNN.md** §results section with **the FULL `=== Official Suite Recap ===` block in a code fence**, plus a markdown table summarizing peak ckpt + WR for quick reading

Format example:
```markdown
### N.M Stage 1 baseline 1000ep [2026-MM-DD]

Trial: `TeamVsBaseline...c0729_..._00-08-42`
Selected ckpts (top 5% + ties + ±1): 130, 140, 150, 170, 180, 190

| ckpt | baseline 1000ep | NW-ML |
|---:|---:|---|
| 130 | 0.840 | 840-160 |
| ... | ... | ... |

**Raw recap** (for verification):
\`\`\`
=== Official Suite Recap ===
.../checkpoint_000130/checkpoint-130 vs baseline: win_rate=0.840 (840W-160L-0T)
...
\`\`\`
```

---

## Stage 2: failure capture (500ep, 1-2 ckpts)

### 2.1 Pick ckpts

After Stage 1 reveals top performers, pick **1-2 ckpts**:
- The top 1 by 1000ep WR
- (Optional) A structurally interesting peer: e.g., second-best with different reward dynamics, or a ckpt with notably different episode-length pattern

If more than 2 candidates seem worth capturing, **ask user**. Do not capture all of them automatically.

### 2.2 Construct + launch capture tmux

**IMPORTANT**: shaping flags must match the lane's training config. Look at the lane's batch script (`scripts/batch/experiments/soccerstwos_*<lane>*.batch`) for the `SHAPING_*` env vars and translate to `--*` CLI flags. Common mappings:

| env var | CLI flag |
|---|---|
| `SHAPING_TIME_PENALTY` | `--time-penalty` |
| `SHAPING_BALL_PROGRESS` | `--ball-progress-scale` |
| `SHAPING_GOAL_PROXIMITY_SCALE` | `--goal-proximity-scale` |
| `SHAPING_OPP_PROGRESS_PENALTY` | `--opponent-progress-penalty-scale` |
| `SHAPING_POSSESSION_DIST` | `--possession-dist` |
| `SHAPING_POSSESSION_BONUS` | `--possession-bonus` |
| `SHAPING_DEEP_ZONE_OUTER_THRESHOLD` | `--deep-zone-outer-threshold` |
| `SHAPING_DEEP_ZONE_OUTER_PENALTY` | `--deep-zone-outer-penalty` |
| `SHAPING_DEEP_ZONE_INNER_THRESHOLD` | `--deep-zone-inner-threshold` |
| `SHAPING_DEEP_ZONE_INNER_PENALTY` | `--deep-zone-inner-penalty` |
| `SHAPING_FAST_LOSS_THRESHOLD_STEPS` | `--fast-loss-threshold-steps` |
| `SHAPING_FAST_LOSS_PENALTY_PER_STEP` | `--fast-loss-penalty-per-step` |
| `SHAPING_EVENT_*_REWARD` | `--event-*-reward` |

If lane uses defaults (no `SHAPING_*` env in batch beyond `USE_REWARD_SHAPING=1`), use v2 default values shown in template below.

Template (copy verbatim, adjust shaping flags per lane):

```bash
tmux new -d -s fail_<short>_<ckpt> "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/failure-capture-logs
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/capture_failure_cases.py \
  --checkpoint <full path to checkpoint-N> \
  --team0-module <team0_module> \
  --opponent baseline \
  -n 500 \
  --max-steps 1500 \
  --base-port <port_shelf>005 \
  --save-dir /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/failure-cases/<short>_checkpoint<ckpt>_baseline_500 \
  --save-mode losses \
  --max-saved-episodes 500 \
  --trace-stride 10 \
  --trace-tail-steps 30 \
  --reward-shaping-debug \
  --time-penalty 0.001 \
  --ball-progress-scale 0.01 \
  --goal-proximity-scale 0.0 \
  --progress-requires-possession 0 \
  --opponent-progress-penalty-scale 0.01 \
  --possession-dist 1.25 \
  --possession-bonus 0.002 \
  --deep-zone-outer-threshold -8 \
  --deep-zone-outer-penalty 0.003 \
  --deep-zone-inner-threshold -12 \
  --deep-zone-inner-penalty 0.003 \
  --defensive-survival-threshold 0 \
  --defensive-survival-bonus 0 \
  --fast-loss-threshold-steps 0 \
  --fast-loss-penalty-per-step 0 \
  --event-shot-reward 0.0 \
  --event-tackle-reward 0.0 \
  --event-clearance-reward 0.0 \
  --event-cooldown-steps 10 \
  2>&1 | tee docs/experiments/artifacts/official-evals/failure-capture-logs/<short>_checkpoint<ckpt>.log
read
'"
```

### 2.3 Document Stage 2

Append to **snapshot-NNN.md** §failure section:

1. **The FULL `---- Summary ----` block** in a code fence (W/L/T counts, episode_steps stats)
2. v2 bucket breakdown table (parsed from save-dir's analysis or manually computed):

| Bucket | <short>@<ckpt> | comparator (e.g., 029B@190) | Δ |
|---|---:|---:|---:|
| defensive_pin | XX% | YY% | ±Z |
| territorial_dominance | ... | | |
| wasted_possession | | | |
| possession_stolen | | | |
| progress_deficit | | | |
| unclear_loss | | | |

3. Episode L stats (mean / median / max) — flag turtle if mean ≥ 50 or max ≥ 200
4. Brief mechanistic reading (1-3 bullets), conservative — e.g., "wasted_possession dropped -9.2pp vs 029B → ball-progress shaping working"

---

## Stage 3: H2H (500ep, snapshot-recommended opponents)

### 3.1 Pick opponents from snapshot analysis

**Do NOT use a hardcoded opponent list.** Read the lane's snapshot to determine what frontier comparisons are scientifically meaningful for this experiment's hypothesis. Examples:

- 040 lanes (Stage 2 PBRS handoff on 031A) → H2H vs **031A@1040 (base)** to test handoff gain + **029B@190** to compare to per-agent SOTA
- 041 lanes (Stage 2 PBRS on 036D) → vs **036D@150 (base)** + **029B@190**
- 044 specialists (spear/shield) → typically NOT for H2H since they're sparring; if requested, vs whatever main agent they sparred against
- 044 main league agent → vs every frontier in the pool to confirm not over-fit
- General lane (no snapshot guidance) → default to **029B@190 + 025b@080 + (031A@1040 if team-level / 036D@150 if per-agent)**

**Always ask user to confirm/modify the opponent list before launching**. Show your shortlist + rationale.

### 3.2 Frontier opponent ckpt registry

| Label | Lane | Path |
|---|---|---|
| 017@2100 | per-agent | `ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100` |
| 025b@080 | per-agent | `ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80` |
| 029B@190 | per-agent | `ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190` |
| 036C@270 | per-agent | `ray_results/PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657/MAPPOVsBaselineTrainer_Soccer_c0729_00000_0_2026-04-18_07-57-19/checkpoint_000270/checkpoint-270` |
| 036D@150 | per-agent | `ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150` |
| 028A@1060 | team-level | `ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060` |
| 030A@360 | team-level | `ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_e96f0_00000_0_2026-04-18_05-11-28/checkpoint_000360/checkpoint-360` |
| 030D@320 | team-level | `ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_3eb0c_00000_0_2026-04-18_05-11-37/checkpoint_000320/checkpoint-320` |
| 031A@1040 | team-level | `ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040` |

If a needed opponent isn't here, find it via `find ray_results/ -name "checkpoint-<N>" | grep <lane>`.

### 3.3 Construct + launch H2H tmux per matchup

Template (copy verbatim, fill placeholders + correct env vars per §"Lane-type mapping" table):

```bash
tmux new -d -s h2h_<short><ckpt>_<opp_short> "bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/headtohead
PYTHONPATH=\$PWD\${PYTHONPATH:+:\$PYTHONPATH} \
<m1_env_var>=<m1 ckpt full path> \
<m2_env_var>=<m2 ckpt full path> \
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python scripts/eval/evaluate_headtohead.py \
  -m1 <m1_module> \
  -m2 <m2_module> \
  -e 500 \
  -p <port_shelf>205 \
  2>&1 | tee /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/<short>_<ckpt>_vs_<opp_short>.log
read
'"
```

- Log filename convention: `<m1_short>_<m1_ckpt>_vs_<opp_short>.log` (e.g., `040B_180_vs_031A_1040.log`, `036C_270_vs_029B_190.log`)
- Run multiple matchups in parallel with different port shelves if multiple nodes available

### 3.4 Read H2H result CORRECTLY (HARD RULE)

**Per [rank.md §0.2](../../docs/experiments/rank.md)** — DO NOT misread direction.

After H2H completes, read the `---- H2H Recap ----` block at the end of the log. The relevant field is `team0_overall_win_rate`. The `team0_*` matches the `-m1` module argument.

```bash
# Verify direction:
grep -A 30 "H2H Recap" <log> | head -35
```

Confirm `team0_module:` line matches the `-m1` you passed. If not (e.g., reversed by some caller), the result direction must be flipped.

**Do NOT use blue/orange split as the headline result** (those are side diagnostics).

### 3.5 Compute z / p

For sample size n with H2H result of `wins/n`:
- `z = (wins - n/2) / sqrt(n/4)`
- `p` (one-sided) ≈ approximation from normal CDF
- Significance: `|z|>1.96` = `*`, `|z|>2.58` = `**`, `|z|>3.29` = `***`

Don't claim significance below `*`.

### 3.6 Document Stage 3

Append to:
- **rank.md** §5.1 H2H matrix cell + §5.3 detail table row (with `n / z / p`)
- **snapshot-NNN.md** §H2H section with **the FULL `---- H2H Recap ----` raw block in code fence**, plus markdown table:

```markdown
| matchup | sample | <short>@<ckpt> wins | opp wins | <short> rate | z | p | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| <short>@<ckpt> vs <opp> | 500 | XXX | YYY | 0.XXX | Z.ZZ | 0.XXX | * / ** / *** / -- |

Side split (无侧别运气 check): blue 0.XXX / orange 0.XXX
```

---

## Stage 4 (optional, scope-B extension)

After Stage 1-3 complete, you MAY initiate ONE additional follow-up — but **always ask user first** with rationale. Do not chain Stage 4 actions.

Trigger conditions:
- **Sampling rerun (500+500 → combined 1000)**: H2H showed marginal result (`|z|` in `[1.5, 2.0]`). Rerun same matchup with different `-p` port for n=1000 combined.
- **Additional H2H opponents**: Stage 3 results suggest an obvious comparison gap (e.g., "this lane now ties 029B but we never tested vs 031A"). Suggest 1-2 additional matchups.
- **2nd ckpt failure capture**: Stage 1 showed two roughly equal 1000ep peaks (within ±0.005). Capture the second one too for structural comparison.

Do NOT initiate beyond Stage 4. If the lane needs more, **recommend in summary and let user decide**.

---

## Conservative conclusion rules

- **SE thresholds**: 100ep ±0.05, 500ep ±0.022, 1000ep ±0.016. Don't write "X is significantly better than Y" unless their difference > 2× SE.
- **Single-shot caveat**: If peak ckpt has only one 1000ep run, mark as "preliminary" until a rerun confirms.
- **H2H z**: don't claim significance below `*` (`|z|>1.96`).
- **No ranking change without H2H**: a lane's place in §6 ranking changes only when supported by direct H2H, not by baseline-axis alone.
- **Always include raw script output**: paste the `---- Summary ----` (eval/capture) or `---- H2H Recap ----` (H2H) block verbatim, in a code fence.

## Document write rules (append-only)

When writing results to snapshot or rank.md:
1. **Append-only** — never delete or overwrite existing data points
2. **Date stamp** the addition (e.g., `### 11.6 [2026-04-19] H2H vs 029B@190`)
3. **Reference original log path** at end of section for traceability
4. **Update rank.md §8 changelog** with a one-line entry summarizing the addition
5. Include **raw script output block** (Stage 1: recap; Stage 2: summary; Stage 3: H2H Recap)

## Decision points (where to ask user)

1. Multiple trials with checkpoints in same run dir → ask which one
2. Auto-discovered ckpt list >10 → confirm running all
3. Free GPU node ambiguous → confirm node + jobid
4. Stage 2 ckpt selection: more than 1-2 obvious candidates → confirm
5. Stage 3 opponent list → **always confirm** before launching
6. Stage 4 extension → **always confirm** before initiating

## Things this skill does NOT do

- Submit SLURM jobs (assume node already allocated; use `scripts/eval/list_my_gpu_nodes.sh` to scout)
- Modify training scripts or batch files
- Make ranking judgments without direct H2H data
- Auto-rerun anything beyond Stage 4
- Process specialist (`044A` spear, `044B` shield) eval the same as standard lanes — for those, ckpt selection metric is `fast_win_rate` / `non_loss_rate`, ask user to confirm metric choice
