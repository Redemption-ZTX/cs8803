# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Behavior

- **Research the codebase before editing. Never change code you haven't read.**
- **Challenge the user's approach** — when the user proposes a technical solution, question its reasoning and offer alternatives if you see a better path. Respect the user's final decision, but don't be a yes-man. The goal is to catch bad ideas before they cost training time. If the user insists after hearing your concerns, execute their plan faithfully.
- Follow the requirement, not the assumed solution. If the user says "make the agent win 9/10 against baseline", the requirement is the win rate — the method is open to challenge.
- **The assignment document is SSOT** — [docs/references/Final Project Instructions Document.pdf](docs/references/Final%20Project%20Instructions%20Document.pdf) is the single source of truth for all requirements and grading criteria. The Markdown transcription ([.md](docs/references/Final%20Project%20Instructions%20Document.md)) is for convenience; if they disagree, follow the PDF.

## Project Overview

CS8803 DRL course project — multi-agent deep reinforcement learning on the Soccer-Twos 2v2 environment. Based on the [soccer-twos-starter](https://github.com/mdas64/soccer-twos-starter) kit with custom training scripts, reward shaping, and curriculum learning.

### Assignment Requirements (from SSOT)

- **Submission**: multiple trained agents, each as a zipped module folder with `AgentInterface.act()` + `README.md`
- **Modification (40 pts)**: alter Observation Space OR Reward Function (not both required), code must be syntactically correct and logical. Bonus +5 for novel concept (curriculum, imitation learning, etc.)
- **Performance (50 pts)**: win 9/10 vs Random Agent (25 pts) + win 9/10 vs Baseline Agent (25 pts). Bonus +5 vs competitive agent2 (not yet released)
- **Report (100 pts)**: 1-2 pages, must include: algorithm + library + theory (10), hyperparameter table (10), modification description (5), hypothesis/motivation (10), training curves per agent (10), comparison graph/table (10), labeled axes (5), performance comparison statement (15), technical reasoning (15), figures + references (10)

## Tech Stack & Constraints

- **Python 3.8** (strict) | **Ray RLlib 1.4.0** (strict) | Unity ML-Agents 0.27.0
- Pin versions: `protobuf==3.20.3`, `pydantic==1.10.13`
- Framework: PyTorch (default for all training scripts)
- Do NOT upgrade Ray or Python — upstream compatibility requirement
- Environment action space: `MultiDiscrete([3,3,3])` flattened to `Discrete` via `ActionFlattener` (note: assignment doc says "continuous" but actual env uses discrete)

## Setup

```bash
bash scripts/setup/setup.sh            # Full setup: conda + deps + baseline + verify
bash scripts/setup/setup.sh --verify   # Verify only
```

For manual steps, PACE cluster notes, and troubleshooting, see [docs/architecture/engineering-standards.md](docs/architecture/engineering-standards.md#环境搭建).

### Work Environment

- **PACE cluster**: primary training environment (GPU, long runs). All work under `$SCRATCH` — home has 15GB limit. Never run training on login node. Submit via SLURM only.
- **Local (Windows + GPU + CUDA)**: development, debugging, short training runs, evaluation, visualization. Unity binary is bundled with soccer_twos — no Unity installation needed.
- **Deploy guide**: [docs/management/deploy-and-verify.md](docs/management/deploy-and-verify.md)

## Common Commands

```bash
python examples/example_random_players.py                              # Watch random agent
python -m cs8803drl.training.train_ray_base_team_vs_random            # Starter-aligned scratch base lane: team_vs_random
python -m cs8803drl.training.train_ray_base_team_vs_baseline          # Baseline-targeted scratch base lane: team_vs_policy vs baseline
python -m cs8803drl.training.train_ray_base_ma_teams                  # Starter-aligned scratch base lane: shared-policy multiagent_team
python -m cs8803drl.training.train_ray_team_vs_random_shaping          # Main training script
python -m cs8803drl.training.train_ray_role_specialization             # Dual-policy role-specialized PPO
python -m cs8803drl.training.train_ray_shared_policy_role_token        # Shared-policy multi-agent PPO + role token
python -m cs8803drl.training.train_ray_selfplay                        # Self-play training
python -m cs8803drl.training.train_ray_curriculum                      # Curriculum training
python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent  # Visual eval
python -m soccer_twos.watch -m cs8803drl.deployment.trained_team_ray_agent     # Watch team-level base checkpoint
python -m soccer_twos.watch -m cs8803drl.deployment.trained_ma_team_agent      # Watch shared-policy multiagent_team base checkpoint
python -m cs8803drl.evaluation.eval_rllib_checkpoint_vs_baseline -c <checkpoint_path>  # Checkpoint eval
python -m cs8803drl.evaluation.evaluate_matches -m1 <agent_module> -m2 ceia_baseline_agent  # Match eval
python scripts/eval/evaluate_official_suite.py --team0-module cs8803drl.deployment.trained_ray_agent --opponents baseline -n 200 --checkpoint <checkpoint_path>  # Official evaluator
```

Hyperparameters are controlled via environment variables. Full env var reference in [docs/architecture/engineering-standards.md](docs/architecture/engineering-standards.md#环境变量速查).

## Architecture (summary)

- `cs8803drl/core/utils.py` — Central env factory (`create_rllib_env`), `RewardShapingWrapper`, baseline policy loader
- `cs8803drl/core/soccer_info.py` — Shared match-info parsing (`score`/`winner`/positions) and pure reward-shaping logic
- `cs8803drl/core/checkpoint_utils.py` — Canonical checkpoint parsing (unpickle, weight extraction, sanitization)
- `cs8803drl/training/train_ray_base_team_vs_random.py` — Starter-aligned scratch base-model lane (`team_vs_random`)
- `cs8803drl/training/train_ray_base_team_vs_baseline.py` — Baseline-targeted scratch base-model lane (`team_vs_policy` vs `baseline`)
- `cs8803drl/training/train_ray_base_ma_teams.py` — Starter-aligned scratch base-model lane (shared-policy `multiagent_team`)
- `cs8803drl/deployment/trained_team_ray_agent.py` — Team-level base checkpoint wrapper for starter-aligned `team_vs_random`
- `cs8803drl/deployment/trained_ma_team_agent.py` — Shared-policy multiagent-team checkpoint wrapper for starter-aligned `multiagent_team`
- `cs8803drl/training/train_ray_team_vs_random_shaping.py` — Main training: single-agent PPO, reward shaping, baseline eval
- `cs8803drl/training/train_ray_role_specialization.py` — Experimental branch: dual-policy role-specialized PPO
- `cs8803drl/training/train_ray_shared_policy_role_token.py` — Experimental branch: shared-policy multi-agent PPO with role tokens
- `cs8803drl/branches/role_specialization.py` / `cs8803drl/branches/shared_role_token.py` — policy mapping, warm-start, role-token utilities
- `cs8803drl/training/train_ray_selfplay.py` — 2v2 multi-agent with frozen baseline + self-play pool
- `cs8803drl/training/train_ray_curriculum.py` — Task progression from `curriculum.yaml`
- Agent modules inherit `soccer_twos.AgentInterface`, implement `act()`, submit as zipped directories

Full architecture: [docs/architecture/overview.md](docs/architecture/overview.md). Code audit: [docs/architecture/code-audit.md](docs/architecture/code-audit.md).

## Rules

### File Protection Levels

**FROZEN — do NOT modify under any circumstances:**
- `example_player_agent/`, `example_team_agent/`, `ceia_baseline_agent/` — upstream/baseline, grading depends on them
- `examples/` — archived upstream examples
- `requirements.txt` — shared dependency spec
- `docs/references/*.pdf`, `docs/references/upstream-README.md` — source of truth

**CAREFUL — modify only with good reason, document in ADR:**
- `cs8803drl/core/utils.py`, `cs8803drl/core/checkpoint_utils.py`, `cs8803drl/training/train_ray_selfplay.py`, `cs8803drl/training/train_ray_team_vs_random_shaping.py`, `cs8803drl/training/train_ray_curriculum.py`, `curriculum.yaml`, `sitecustomize.py`
- When modifying `cs8803drl/core/checkpoint_utils.py`, sync copies to `agents/_template/agent.py` and all `agents/vNNN_*/agent.py`

**FREE — create and modify freely:**
- `agents/` (all experiment agent versions), `docs/`, `scripts/`, `report/`, new `.py` files, project meta files

### Mandatory Practices

1. **Documentation is not optional** — every code merge needs `CHANGELOG.md`; every experiment needs a `snapshot-NNN.md` BEFORE running; every new file must be indexed
2. **Indexes must stay in sync** — when any file is added/removed/renamed, update ALL indexes: `docs/README.md` file tree, root `README.md` project structure, relevant sub-index (`experiments/README.md`, `adr/README.md`, `code-audit.md`), and any references in `CLAUDE.md`
   Directory placement and root-vs-subdirectory rules are defined in [docs/management/directory-governance.md](docs/management/directory-governance.md)
3. **Cross-references are mandatory** — no orphan docs; snapshots link to code-audit/ADR; ADRs link to code
4. **Snapshots are append-only** — never modify completed `code-audit-NNN.md` or `snapshot-NNN.md`; write a new one instead
5. **No silent experiments** — no training without recording; no editing script defaults for experiments (use env vars); no deleting ray_results without extracting metrics
6. **Agent modules must always be submission-ready** — each `agents/vNNN_*/` has `__init__.py` + `README.md` + works with `soccer_twos.watch`. Final submission: copy best versions to root, zip per assignment format

### Task-Specific Procedures

**Before committing code** → read and follow [docs/architecture/engineering-standards.md § Commit Procedure](docs/architecture/engineering-standards.md#commit-流程)

**Before running an experiment** → read and follow [docs/architecture/engineering-standards.md § Experiment Iteration](docs/architecture/engineering-standards.md#实验迭代)

**Before modifying a CAREFUL file** → read [docs/architecture/code-audit.md](docs/architecture/code-audit.md) for the latest audit of that module

## Git Policy

- `ray_results/` and `checkpoint*/` must NOT be committed (training outputs, too large)
- `soccerstwos-*.out` must NOT be committed (SLURM job logs)
- `ceia_baseline_agent/` IS committed — 148MB, baseline checkpoint needed for training and evaluation

## PACE Cluster Rules

These come directly from the [assignment document](docs/references/Final%20Project%20Instructions%20Document.md#pace-documentation):

- **All work under `$SCRATCH`** — home directory is 15GB, will lock you out if full. Clone repo, install conda, store ray_results all in scratch.
- **Never run training on login node** — submit via `sbatch scripts/batch/starter/soccerstwos_job.batch`
- **Do NOT contact PACE support** — course does not have direct student support from PACE
- **Start training early** — GPU queues spike near DDL, you may wait hours
- **Use Open OnDemand** for file transfer: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133
- **Connect via GT VPN first** → `ssh GT_USERNAME@login-ice.pace.gatech.edu`

### GPU Resource Request

```bash
srun --gres=gpu:H100:1 --mem=100G -t 08:00:00 --pty bash
```

- **SCRATCH path**: `/storage/ice1/5/1/wsun377`
- **Home quota**: 30GB (currently ~2.1GB used)
- **Conda env activation**: `module load anaconda3/2023.03 && source activate soccertwos`

### Cluster Status (2026-04-08 12:18 EDT)

| JOBID | Partition | Status | Elapsed | Time Limit | Node | GPU |
|-------|-----------|--------|---------|------------|------|-----|
| 4627063 | coe-gpu | Running | 00:10:26 | 8:00:00 | atl1-1-03-010-20-0 | H100:1 |
