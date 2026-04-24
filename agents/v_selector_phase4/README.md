# Agent: v_selector_phase4 (Stone DIR-A Wave 1)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Inspiration**: Wang/Stone/Hanna ICRA 2025, "Reinforcement Learning Within the
  Classical Robotics Stack" (arXiv 2412.09417). Decomposes behavior into 4
  sub-policies activated by heuristic geometry checks instead of a single
  monolithic policy. Won 2024 RoboCup SPL Challenge Shield Division.
- **Description**: 4-phase task-conditional selector. Each step, every agent
  independently classifies its current "game phase" from its 336-dim ray
  observation (no extra inference, pure numpy) and routes the forward call to
  the best specialist for that phase:
  - **NEAR-GOAL** (ball nearest < 0.18, centroid ahead) → 1750 SOTA
  - **BALL-DUEL** (ball nearest in [0.18, 0.40)) → 055@1150 (different family)
  - **POSITIONING** (teammate has better ball view) → 1750 SOTA (Wave 1 placeholder)
  - **MID-FIELD** (default) → 1750 SOTA
- **Differs from v074f**: 074F averages action probabilities across 3 members.
  This routes hard by state phase (no averaging dilution).
- **Wave 1 status**: placeholder phase mapping using the 3 single-model agents
  already packaged. Wave 2 will plug in 081 aggressive specialist for
  NEAR-GOAL/BALL-DUEL once that lane finishes training.
- **Usage**: `python -m soccer_twos.watch -m agents.v_selector_phase4`
