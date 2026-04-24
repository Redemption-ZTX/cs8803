# Agent: v_055_1150 (prior project SOTA)

- **Authors**: Team CS8803DRL — wsun377 + collaborator (ve11ichor1223@gmail.com)
- **Description**: Team-level (672-dim joint obs) PPO agent. Architecture =
  Siamese dual encoder + 1-head cross-attention (4 tokens × 64 dim). Trained
  from scratch via Hinton-style distillation from a 3-teacher ensemble
  (031B@1220 + 036D@150 + 029B@190). v2 reward shaping. Combined 2000ep
  baseline WR = 0.907.
- **Usage**: `python -m soccer_twos.watch -m agents.v_055_1150`
