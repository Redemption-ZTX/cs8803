# v_xattn_fusion_w3_anchored — DIR-H W3 untrained 1750-anchored fusion

Soft state-conditional fusion of K=8 experts via a lightweight cross-attention
mechanism, with a +3.0 logit bias on 1750 SOTA to provide a "safe routing"
default. Directly addresses the Wave 2 regression root cause (specialists
dominating their phase slots when forced hard-switched).

## Architecture

```
obs (336-d) ─ Linear(336→64, no-bias) ─ Q (64-d)
                                         │
                             ┌───────────┴───────────┐
                             │  Q @ Kᵀ / √64 + bias │   Keys: (8, 64) random init
                             └───────────┬───────────┘   Bias: [+3.0, 0,0,0,0,0,0,0]
                                         │
                                     softmax
                                         │
                                    weights (8,)
                                         │
    expert_i(obs) ────────────── weights[i] ────── Σ weights[i] * p_i(obs)
                                                       │
                                                   fused probs (27,)
                                                       │
                                              argmax/sample → action
```

At init (no training), the anchor bias ensures w[1750] ≈ 0.75 and each of the
7 specialists gets ≈ 0.035. Smoke test:
```
weights: [0.749, 0.036, 0.034, 0.035, 0.034, 0.036, 0.038, 0.037]  → sum=1.00
```

## Experts (8, matching v_moe_router_uniform Wave 2 pool)

| idx | Name | Role | Peak baseline WR |
|---|---|---|---|
| **0** | **1750_sota** | **ANCHOR — project SOTA** | **0.9155** |
| 1 | 055_1150 | prior distill SOTA | 0.907 |
| 2 | 029B_190 | per-agent SOTA | ~0.868 |
| 3 | 081_aggressive | NEAR-GOAL specialist | 0.826 |
| 4 | 101A_ballcontrol | Phase 1 specialist | 0.851 |
| 5 | 103A_interceptor | BALL_DUEL specialist | 0.548 |
| 6 | 103B_defender | POSITIONING specialist | 0.205 |
| 7 | 103C_dribble | MID-FIELD specialist | 0.220 |

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `XATTN_ANCHOR_BIAS` | 3.0 | Logit bias on 1750 (higher = stronger anchor) |
| `XATTN_D_KEY` | 64 | Query/key dim (fixed for W3 W1 untrained) |
| `XATTN_SEED` | 0 | Torch seed for Q/key random init |

## Expected performance (untrained W3)

- vs 1750 alone: marginal difference (~±0.005) — specialist perturbation is soft (~3.5% each)
- vs v074F weighted uniform ensemble (0.903): possibly higher because of 1750 bias
- vs v_selector_phase4 wave2 (0.765 regression): **should recover to ~SOTA-tier** since no forced specialist slots

## Training follow-up (not yet launched)

- **W2 fully trained**: REINFORCE on episode reward, all params (Q proj + keys + bias) trainable — see task-queue
- **W3 trained anchored**: same but keep anchor bias frozen or with regularization

## Usage

```bash
python -m soccer_twos.watch -m agents.v_xattn_fusion_w3_anchored
python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v_xattn_fusion_w3_anchored \
  --opponents baseline \
  -n 1000 -j 1 --base-port 62905 \
  --checkpoint /home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750
```

## Related

- [cs8803drl/branches/xattn_fusion_nn.py](../../cs8803drl/branches/xattn_fusion_nn.py) — `XAttnFusionNN` module
- [agents/v_moe_router_uniform/](../v_moe_router_uniform/) — DIR-G hard-routing baseline (1 expert per step, uniform sample)
- [agents/v_selector_phase4/](../v_selector_phase4/) — DIR-A heuristic hard-switch selector
- [snapshot-100](../../docs/experiments/snapshot-100-dir-A-heuristic-selector.md) — Wave 2 regression context
- [task-queue.md](../../docs/experiments/task-queue.md) — DIR-H item
