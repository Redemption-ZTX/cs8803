# Agent: v_029B_190 (per-agent SOTA)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Description**: Per-agent (336-dim) MAPPO with shared centralized critic,
  512×512 MLP. Trained as PBRS-handoff from 029A B-warm with v2 reward shaping.
  Each of the 2 agents acts on its own 336-dim observation. 1000ep baseline
  WR ≈ 0.86. Project's strongest per-agent model and ensemble member in
  034E / 074F-lineage experiments.
- **Usage**: `python -m soccer_twos.watch -m agents.v_029B_190`
