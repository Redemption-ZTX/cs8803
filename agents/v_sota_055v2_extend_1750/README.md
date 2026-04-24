# v_sota_055v2_extend_1750 — Project SOTA Submission

**Agent name**: 055v2_extend@1750  
**Authors**: wsun377 (ve11ichor1223@gmail.com)  
**Architecture**: 031B Siamese dual encoder + 1-layer cross-attention (4 tokens × 64 dim)  

## Performance vs Baseline

- **Combined 4000ep baseline WR = 0.9155** (CI 95% [0.908, 0.924])
- 4 independent 1000ep samples: 0.917 / 0.916 (×2 from 2000ep batch) / 0.913 — stable peak
- vs 055@1150 prior SOTA (combined 2000ep 0.907): Δ +0.009, H2H 0.538 (z=1.70, p=0.045 sig)
- vs 031B baseline 0.880: Δ +0.036, sig

## Modification Description

**Reward**: v2 shaping (modified from sparse goal default):
- time_penalty 0.001 + ball_progress_scale 0.01 + opponent_progress_penalty_scale 0.01
- possession_bonus 0.002 (within possession_dist 1.25)
- deep_zone penalties (-0.003 for ball x ≤ -8, additional ≤ -12)

**Architecture novel concept**: Recursive distillation from teacher ensemble + scratch-extend:
- Generation 1: 055 distilled from 034E ensemble (031B + 045A + 051A) → +1.7pp over teacher
- Generation 2 (this agent): 055v2 = recursive distill from 5-teacher pool 
  {055@1150 + 031B@1220 + 045A@180 + 051A@130 + 056D@1140}
  trained scratch with LR=3e-4 to iter 1216, then extended to iter 2000
- Peak at iter 1750 verified stable across 3-4 independent eval samples

## Training Hyperparameters

- Algorithm: PPO (Ray RLlib 1.4.0)
- LR: 3e-4 (3× baseline 055)
- Clip param: 0.15
- Train batch: 40000 env steps, SGD minibatch 2048, 4 epochs
- Rollout fragment: 1000
- 8 workers × 5 envs each
- Distillation KL: α_init=0.05 → α_final=0.0, decay 8000 updates, T=1.0
- Total iters: 2000 (1216 scratch + 784 extend)

## Usage

```bash
# Watch in Unity (3D rendering)
python -m soccer_twos.watch -m agents.v_sota_055v2_extend_1750

# Eval vs baseline
python -m soccer_twos.watch -m agents.v_sota_055v2_extend_1750 -m2 ceia_baseline_agent
```
