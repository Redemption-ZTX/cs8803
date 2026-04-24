# Agent: v_moe_router_uniform (Stone DIR-G Wave 1, uniform-routing baseline)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Inspiration**: Mixture-of-Experts framework (Shazeer et al. 2017) applied
  to deploy-time agent selection. Differs from v074f (probability averaging
  at action level) and v_selector_phase4 (geometric heuristic routing) by
  using a **learnable per-step router function** to pick ONE expert per
  step (hard switch).
- **Description**: 3 frozen expert agents + a UniformRouter (no NN — Wave 1
  baseline). Each step, each agent independently samples an expert uniformly
  at random and uses that expert's argmax action. This Wave 1 isolates the
  value of "diverse member set with random routing" as the lower bound.
  Wave 2 will swap in a learned NN router trained via REINFORCE on episode
  outcomes.
- **Experts** (3 total, same as v_selector_phase4 for direct comparison):
  1. `1750_sota` — team-level 055v2_extend@1750 (project SOTA, combined 4000ep 0.9155)
  2. `055_1150` — team-level prior SOTA (combined 2000ep 0.907)
  3. `029B_190` — per-agent SOTA (1000ep ≈ 0.86)
- **Usage**: `python -m soccer_twos.watch -m agents.v_moe_router_uniform`
