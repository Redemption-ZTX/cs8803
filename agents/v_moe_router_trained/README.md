# Agent: v_moe_router_trained (Stone DIR-G Wave 2, REINFORCE-trained router)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Description**: Wave 2 of DIR-G MoE Router. Same K-expert framework as
  `v_moe_router_uniform` (Wave 1) but the per-step expert selection is driven
  by a NN trained via REINFORCE (`scripts/research/train_moe_router_reinforce.py`)
  on episode returns vs ceia_baseline_agent. Loads trained router weights from
  `router_weights.pt` if present; falls back to fresh-init (≈ uniform) if
  weights file is missing.
- **Hypothesis**: state-conditional learned routing extracts more signal than
  uniform random (Wave 1 = 0.900). Expect ≥ 0.92 if router learns sensible
  geometry-aware selection AND specialist pool includes orthogonal members
  (081 aggressive + 103-series, auto-included if their packages exist).
- **Differs from**:
  - `v_moe_router_uniform`: trained NN router vs uniform random
  - `v_selector_phase4`: NN learned vs hand-coded geometric heuristic
  - `v074f`: state-conditional hard switch vs action-prob soft averaging
- **Usage**: `python -m soccer_twos.watch -m agents.v_moe_router_trained`
