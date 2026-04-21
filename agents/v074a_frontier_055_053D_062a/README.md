# v074a_frontier_055_053D_062a

Deploy-time probability-averaging ensemble (3 team-level members):

- `055@1150` — distillation SOTA anchor (combined 2000ep 0.907).
- `053Dmirror@670` — PBRS-only blood (single-shot 1000ep 0.902).
- `062a@1220` — curriculum + no-shape blood (combined 2000ep 0.892).

Uniform weights, greedy action selection (`ENSEMBLE_GREEDY=1`).

Watch:
```bash
python -m soccer_twos.watch -m1 agents.v074a_frontier_055_053D_062a -m2 ceia_baseline_agent
```

Eval:
```bash
bash scripts/eval/_launch_074A_ensemble_eval.sh
```

See `docs/experiments/snapshot-074-034-next-deploy-time-ensemble.md` for the
full design (pre-registered thresholds, risks, retrograde plan).
