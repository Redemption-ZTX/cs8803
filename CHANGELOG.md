# Changelog

格式基于 [Keep a Changelog](https://keepachangelog.com/)。

<!-- 模板：

## [vX.Y.Z] - YYYY-MM-DD

### Added
### Changed
### Fixed
### Removed

-->

## [Unreleased]

### Added
- Handoff documentation and project governance files, including `CLAUDE.md`, the `docs/` hub, the agent template, and `scripts/setup.sh`.
- `checkpoint_utils.py` to centralize RLlib checkpoint parsing and restore helpers shared by training and evaluation code.
- Archived upstream example scripts under `examples/` so the root directory stays focused on active project entry points.

### Changed
- Rewrote the root `README.md` and `docs/README.md` to reflect the current CS8803 project structure and workflows.
- Updated training and evaluation scripts to reuse shared checkpoint-loading utilities instead of duplicating restore logic.
- Refreshed the PACE batch script and `.gitignore` to better support reproducible experiments and keep generated artifacts out of git.

### Removed
- Deleted tracked `ray_results/` artifacts and the root-level `ceia_baseline_agent.zip` archive from version control.
- Removed root-level legacy example scripts after archiving them under `examples/`.
