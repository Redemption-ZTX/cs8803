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
bash scripts/setup.sh            # Full setup: conda + deps + baseline + verify
bash scripts/setup.sh --verify   # Verify only
```

For manual steps, PACE cluster notes, and troubleshooting, see [docs/architecture/engineering-standards.md](docs/architecture/engineering-standards.md#环境搭建).

### Work Environment

- **PACE cluster**: primary training environment (GPU, long runs). All work under `$SCRATCH` — home has 15GB limit. Never run training on login node. Submit via SLURM only.
- **Local (Windows + GPU + CUDA)**: development, debugging, short training runs, evaluation, visualization. Unity binary is bundled with soccer_twos — no Unity installation needed.
- **Deploy guide**: [docs/management/deploy-and-verify.md](docs/management/deploy-and-verify.md)

## Common Commands

```bash
python examples/example_random_players.py                              # Watch random agent
python train_ray_team_vs_random_shaping.py                             # Main training script
python train_ray_selfplay.py                                           # Self-play training
python train_ray_curriculum.py                                         # Curriculum training
python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent  # Visual eval
python eval_rllib_checkpoint_vs_baseline.py -c <checkpoint_path>       # Checkpoint eval
python evaluate_matches.py -m1 <agent_module> -m2 ceia_baseline_agent  # Match eval
```

Hyperparameters are controlled via environment variables. Full env var reference in [docs/architecture/engineering-standards.md](docs/architecture/engineering-standards.md#环境变量速查).

## Architecture (summary)

- `utils.py` — Central env factory (`create_rllib_env`), `RewardShapingWrapper`, baseline policy loader
- `checkpoint_utils.py` — Canonical checkpoint parsing (unpickle, weight extraction, sanitization). Agent modules contain copies for zip submission
- `train_ray_team_vs_random_shaping.py` — Main training: single-agent PPO, reward shaping, baseline eval
- `train_ray_selfplay.py` — 2v2 multi-agent with frozen baseline + self-play pool
- `train_ray_curriculum.py` — Task progression from `curriculum.yaml`
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
- `utils.py`, `checkpoint_utils.py`, `train_ray_selfplay.py`, `train_ray_team_vs_random_shaping.py`, `train_ray_curriculum.py`, `curriculum.yaml`, `sitecustomize.py`
- When modifying `checkpoint_utils.py`, sync copies to `agents/_template/agent.py` and all `agents/vNNN_*/agent.py`

**FREE — create and modify freely:**
- `agents/` (all experiment agent versions), `docs/`, `scripts/`, `report/`, new `.py` files, project meta files

### Mandatory Practices

1. **Documentation is not optional** — every code merge needs `CHANGELOG.md`; every experiment needs a `snapshot-NNN.md` BEFORE running; every new file must be indexed
2. **Indexes must stay in sync** — when any file is added/removed/renamed, update ALL indexes: `docs/README.md` file tree, root `README.md` project structure, relevant sub-index (`experiments/README.md`, `adr/README.md`, `code-audit.md`), and any references in `CLAUDE.md`
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
- **Never run training on login node** — submit via `sbatch scripts/soccerstwos_job.batch`
- **Do NOT contact PACE support** — course does not have direct student support from PACE
- **Start training early** — GPU queues spike near DDL, you may wait hours
- **Use Open OnDemand** for file transfer: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133
- **Connect via GT VPN first** → `ssh GT_USERNAME@login-ice.pace.gatech.edu`
