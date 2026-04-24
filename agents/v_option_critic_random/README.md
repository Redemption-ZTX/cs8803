# Agent: v_option_critic_random (Stone DIR-E Wave 1, random-init NN)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Inspiration**: Bacon, Harb, Precup 2017 ("The Option-Critic Architecture")
  applied with **frozen intra-option policies** = our 3 packaged experts. Only
  the high-level termination + selector NN is trainable.
- **Description**: 3 frozen expert agents + a small option-critic head NN
  (336 → 64 → 64 → K logits + 1 termination prob). At each step, head decides
  whether to terminate the currently-running option and pick a new one
  (Bacon's β-policy + π_Ω). Wave 1 uses a randomly-initialized head — so
  behavior is roughly "random selector with ~50% per-step termination". This
  isolates the **temporal-abstraction baseline** before learning.
- **Wave 2 plan**: train head via REINFORCE on episode returns; head learns
  WHEN to switch options + WHICH option to pick.
- **Differs from**:
  - `v_moe_router_uniform`: every step picks a new expert (no temporal stickiness)
  - `v_selector_phase4`: hand-coded geometric switching (not learned)
  - `v074f`: action-prob averaging (no hard switch)
- **Usage**: `python -m soccer_twos.watch -m agents.v_option_critic_random`
