# v034ea — Frontier ensemble {031B, 045A, 051A}

3-way probability-averaging ensemble. Sub-variant of 034e family (replaces
036D + 029B with 045A + 051A).

## Members

| ckpt | indiv 1000ep WR | v2 fingerprint |
|---|---|---|
| 031B@1220 | 0.882 | progress_deficit + unclear_loss (offensive-balanced) |
| 045A@180 | 0.867 | wasted_possession 55.4% (high possession, poor conversion) |
| 051A@130 | 0.888 | ultra-defensive turtle (mean_ball_x -0.76, tail -1.20) |

Average member WR: 0.879 (vs 034e avg 0.863).

## Hypothesis

Three orthogonal failure modes (offensive / wasted-possession / ultra-defensive)
should give better ensemble lift than 034e's {balanced, defensive, defensive}.

## Source

[snapshot-051 §8.6](../../docs/experiments/snapshot-051-learned-reward-from-strong-vs-strong-failures.md#86-对-ensemble-034f-的-implication--051a-是新候选).
