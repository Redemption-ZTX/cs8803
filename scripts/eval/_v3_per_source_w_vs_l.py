#!/usr/bin/env python3
"""v3 W vs L per-source distinguishability.

Per source: compute Cohen's d and AUC for each metric on W vs L within that source.
Then aggregate (mean/median |d|) to find metrics that consistently discriminate.

Reuses v3_metrics_full_W_L.pkl from previous analysis.
"""
import pickle
import json
import os
from collections import defaultdict
import numpy as np


def cohen_d(w_vals, l_vals):
    if len(w_vals) < 2 or len(l_vals) < 2:
        return 0.0
    w = np.asarray(w_vals, dtype=float)
    l = np.asarray(l_vals, dtype=float)
    sw, sl = w.std(), l.std()
    pooled = np.sqrt(((len(w) - 1) * sw**2 + (len(l) - 1) * sl**2) / (len(w) + len(l) - 2))
    if pooled < 1e-10:
        return 0.0
    return float((w.mean() - l.mean()) / pooled)


def main():
    pkl_path = 'docs/experiments/artifacts/v3_dataset/v3_metrics_full_W_L.pkl'
    print(f'[v3-per-source] Loading {pkl_path}')
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    print(f'[v3-per-source] {len(all_data)} eps loaded')

    METRICS = ['mean_ball_x', 'tail_mean_ball_x', 'team0_possession_ratio', 'team0_progress_toward_goal',
               'team1_progress_toward_goal', 'threat_density', 'deep_threat_density', 'max_ball_x',
               'defensive_depth', 'min_ball_x', 'centroid_forward_bias', 'forward_velocity',
               'sustained_pressure_max', 'possession_x_correlation', 'late_only_attack',
               'shot_attempts_team0', 'shot_attempts_team1', 'forward_momentum_density',
               'possession_efficiency', 'shot_to_threat_ratio']

    # Group by source
    by_src = defaultdict(lambda: {'W': [], 'L': []})
    for e in all_data:
        s = e['_source']
        if e['_outcome'] == 'team0_win':
            by_src[s]['W'].append(e)
        else:
            by_src[s]['L'].append(e)

    sources = sorted(by_src.keys())
    print(f'[v3-per-source] {len(sources)} sources')

    # Per-source per-metric Cohen's d
    src_d = {}  # src -> {metric: d}
    for s in sources:
        w_eps = by_src[s]['W']
        l_eps = by_src[s]['L']
        nW, nL = len(w_eps), len(l_eps)
        if nW < 5 or nL < 5:
            continue  # skip if either side too few
        src_d[s] = {'_nW': nW, '_nL': nL}
        for m in METRICS:
            d = cohen_d([e[m] for e in w_eps], [e[m] for e in l_eps])
            src_d[s][m] = d

    # Per-metric stats across sources: mean |d|, median |d|, % sources with |d|>=0.5
    print(f'\n=== Per-metric W-vs-L distinguishability (averaged across {len(src_d)} sources) ===')
    print(f'{"metric":<30s}  {"mean|d|":>9s}  {"med|d|":>9s}  {">=0.5":>7s}  {">=0.8":>7s}  {"avg_d":>8s}')

    metric_stats = []
    for m in METRICS:
        ds = [src_d[s][m] for s in src_d]
        absds = [abs(d) for d in ds]
        n_med = sum(1 for d in absds if d >= 0.5)
        n_lge = sum(1 for d in absds if d >= 0.8)
        metric_stats.append({
            'metric': m,
            'mean_abs_d': float(np.mean(absds)),
            'median_abs_d': float(np.median(absds)),
            'pct_medium': n_med / len(absds),
            'pct_large': n_lge / len(absds),
            'avg_signed_d': float(np.mean(ds)),
        })

    metric_stats.sort(key=lambda x: -x['mean_abs_d'])
    for s in metric_stats:
        print(f'  {s["metric"]:<30s}  {s["mean_abs_d"]:>9.3f}  {s["median_abs_d"]:>9.3f}  '
              f'{s["pct_medium"]*100:>6.0f}%  {s["pct_large"]*100:>6.0f}%  {s["avg_signed_d"]:>+8.3f}')

    # Highlight: top-K candidate v3 reward heads
    print(f'\n=== Top 10 candidates for v3 reward model heads (by mean |d|) ===')
    for s in metric_stats[:10]:
        direction = "W>L" if s['avg_signed_d'] > 0 else "W<L"
        print(f'  {s["metric"]:<30s}  mean|d|={s["mean_abs_d"]:.3f}  dir={direction}  median sources>=0.5: {s["pct_medium"]*100:.0f}%')

    # Save
    out = {
        'n_sources_used': len(src_d),
        'sources': list(src_d.keys()),
        'metric_stats': metric_stats,
        'per_source_d': {s: {k: v for k, v in src_d[s].items()} for s in src_d},
    }
    out_path = 'docs/experiments/artifacts/v3_dataset/v3_per_source_W_vs_L_stats.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[v3-per-source] Saved {out_path}')


if __name__ == '__main__':
    main()
