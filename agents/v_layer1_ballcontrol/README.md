# v_layer1_ballcontrol — Stone Layered Phase 1 Specialist (101A@460)

Self-contained agent module for the Stone & Veloso 2000 Layered Learning Phase 1
ball-control specialist, packaged for PIPELINE V1 specialist library use and
zipped submission compatibility.

## Performance

Verified 2026-04-22 (see [snapshot-101 §7](../../docs/experiments/snapshot-101-dir-B-layered-phase1.md)):

| Metric | Result | Threshold | Status |
|---|---|---|---|
| baseline 1000ep WR | **0.851** @ ckpt 460 | §3.1 ≥0.85 | **HIT (just barely)** |
| random 200ep WR (inline) | 0.99 | §3.2 ≥0.95 | HIT |
| vs 029B per-agent SOTA | ~0.86 estimated tie | — | surprise transfer |
| vs 1750 project SOTA | sub-frontier expected | — | standalone sub-frontier |

## Training recipe (short)

- **Architecture**: 031B Siamese encoder + cross-attention (standard project backbone)
- **Budget**: 500 iter / 20M steps / ~4h H100
- **Opponent**: random only (BASELINE_PROB=0.0)
- **Reward**: v2 shaping with ball_progress=0.01 + possession_bonus=0.002; NO shot_reward, NO defensive_survival, NO deep_zone penalty
- **Scenario**: standard 2v2 team env (no scenario reset)

## Intended use

- **PIPELINE V1 specialist library** member — "Layer 1 ball-control" role in ensemble / selector routing
- **Warm-start source for Stone Layered Phase 2** (pass-decision specialist, deferred P2 in task-queue)
- **NOT a direct submission candidate** — standalone 0.851 < project SOTA 1750 (0.9155)

## Files

- [agent.py](agent.py) — `Agent` class inheriting `TeamRayAgent`, self-contained checkpoint path resolution
- [__init__.py](__init__.py) — exports `Agent`
- [params.pkl](params.pkl) — Ray PPO trial params (needed for policy reconstruction)
- [checkpoint_000460/](checkpoint_000460/) — binary checkpoint + tune_metadata

## Usage

```bash
# Watch self-play or vs baseline
python -m soccer_twos.watch -m agents.v_layer1_ballcontrol
python -m soccer_twos.watch -m agents.v_layer1_ballcontrol -m2 ceia_baseline_agent

# Official eval
python scripts/eval/evaluate_official_suite_parallel.py \
  --team0-module agents.v_layer1_ballcontrol \
  --opponents baseline,random \
  -n 1000 -j 7
```

## Package manifest

Zipped submission layout (self-contained, no `ray_results/` absolute refs):

```
v_layer1_ballcontrol/
├── agent.py
├── __init__.py
├── README.md
├── params.pkl
└── checkpoint_000460/
    ├── checkpoint-460
    └── checkpoint-460.tune_metadata
```

## Related

- [snapshot-101](../../docs/experiments/snapshot-101-dir-B-layered-phase1.md) — DIR-B Phase 1 pre-reg + verdict
- [snapshot-099 §8.2](../../docs/experiments/snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone 6-DIR work plan, Phase 2 trigger criteria
- [agents/v_sota_055v2_extend_1750/](../v_sota_055v2_extend_1750/) — same packaging pattern, project SOTA
