# Agent: v_031B_1220 (architecture baseline)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Description**: Team-level (672-dim) PPO agent, foundational architecture
  step (Siamese dual encoder + 1-head cross-attention, 4 tokens × 64 dim).
  Trained from scratch with v2 reward shaping; no distillation, no warm-start.
  1000ep baseline WR = 0.882. Anchor architecture for 074F ensemble and
  warm-start base for many downstream lanes.
- **Usage**: `python -m soccer_twos.watch -m agents.v_031B_1220`
