# CS8803DRL_AGENT

**Agent name**: 055v2_extend@1750

**Author(s)**: Team CS8803DRL — wsun377 (ve11ichor1223@gmail.com)

## Description

Submission package for the team's primary final-project agent:

- Team-level PPO policy for SoccerTwos
- Architecture: Siamese dual encoder + bidirectional cross-attention
- Training recipe: recursive distillation with v2 reward shaping
- Selected checkpoint: `055v2_extend@1750`

This package is self-contained for inference:

- no dependency on `cs8803drl.*`
- no dependency on `/storage/...`
- no dependency on teacher ensemble checkpoints at submission time

## Usage

```bash
python -m soccer_twos.watch -m CS8803DRL_AGENT
python -m soccer_twos.watch -m1 CS8803DRL_AGENT -m2 ceia_baseline_agent
python -m soccer_twos.watch -m1 CS8803DRL_AGENT -m2 example_team_agent
```
