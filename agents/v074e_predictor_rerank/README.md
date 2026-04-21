# v074e_predictor_rerank

Deploy-time ensemble with outcome-predictor top-K re-rank.

Members (same as 074A):
- `055@1150`, `053Dmirror@670`, `062a@1220`.

Augmentation: the calibrated v3 outcome predictor (`P(team0_win | obs seq)`)
is queried on the running trajectory buffer. When the top-1 vs top-2
ensemble probability margin is < 0.10 (uncertainty regime), action
probabilities are re-weighted toward the top-K candidates using
`V(s)`-based tilting. When the ensemble is confident, falls through to
plain mean-of-probs.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OUTCOME_RERANK_ENABLE` | `1` | Set to `0` to disable predictor (mirrors 074A) |
| `OUTCOME_RERANK_TOPK` | `3` | Candidate action set size |
| `OUTCOME_RERANK_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `OUTCOME_RERANK_PREDICTOR_PATH` | `docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt` | Override weights |
| `OUTCOME_RERANK_BUFFER` | `80` | Trajectory window length |

Watch:
```bash
python -m soccer_twos.watch -m1 agents.v074e_predictor_rerank -m2 ceia_baseline_agent
```

See `docs/experiments/snapshot-074E-predictor-rerank.md`.
