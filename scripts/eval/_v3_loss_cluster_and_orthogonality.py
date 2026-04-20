#!/usr/bin/env python3
"""v3 Part 2: Cluster L episodes + check v3 metric orthogonality vs v2 buckets.

Two analyses on L-only subset (~6975 eps):
  A. K-means cluster (k=3..8) on z-normalized 20 metrics.
     Report silhouette + cluster centers (best k).
  B. v2 bucket multilabel computed per episode; check correlation between
     each v3 metric and each v2 bucket indicator.
     Identify v3 metrics ORTHOGONAL to all v2 buckets (max |r| < 0.30 across all 6 buckets).

Reuses v3_metrics_full_W_L.pkl.
"""
import pickle
import json
import os
import numpy as np
from collections import defaultdict, Counter
from cs8803drl.imitation.failure_buckets_v2 import classify_failure_v2

METRICS = ['mean_ball_x', 'tail_mean_ball_x', 'team0_possession_ratio', 'team0_progress_toward_goal',
           'team1_progress_toward_goal', 'threat_density', 'deep_threat_density', 'max_ball_x',
           'defensive_depth', 'min_ball_x', 'centroid_forward_bias', 'forward_velocity',
           'sustained_pressure_max', 'possession_x_correlation', 'late_only_attack',
           'shot_attempts_team0', 'shot_attempts_team1', 'forward_momentum_density',
           'possession_efficiency', 'shot_to_threat_ratio']

V2_BUCKETS = ['defensive_pin', 'territorial_dominance', 'wasted_possession',
              'possession_stolen', 'progress_deficit', 'unclear_loss']


def load_data():
    pkl_path = 'docs/experiments/artifacts/v3_dataset/v3_metrics_full_W_L.pkl'
    print(f'[load] {pkl_path}')
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    L = [e for e in all_data if e['_outcome'] == 'team1_win']
    print(f'[load] {len(all_data)} eps total, {len(L)} losses')
    return L


def part_a_kmeans(L_eps):
    """K-means on z-normalized metrics, k=3..8, report silhouette + best k."""
    print(f'\n=== Part A: K-means clustering on {len(L_eps)} L episodes ===')
    X = np.array([[e[m] for m in METRICS] for e in L_eps], dtype=float)
    # Z-normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1
    Xz = (X - mean) / std
    print(f'X shape: {Xz.shape}, normalized')

    # numpy-only minimal k-means (avoid sklearn dep)
    def kmeans_np(X, k, n_iter=100, n_init=10, seed=0):
        rng = np.random.RandomState(seed)
        best_labels, best_inertia, best_centers = None, np.inf, None
        for init in range(n_init):
            # k-means++ init: pick k centers spread out
            idx0 = rng.randint(len(X))
            centers = [X[idx0].copy()]
            for _ in range(k - 1):
                d2 = np.min(np.linalg.norm(X[:, None, :] - np.stack(centers)[None, :, :], axis=2)**2, axis=1)
                p = d2 / d2.sum()
                idx = rng.choice(len(X), p=p)
                centers.append(X[idx].copy())
            centers = np.stack(centers)
            for _ in range(n_iter):
                # assign
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = dists.argmin(axis=1)
                # update
                new_centers = np.zeros_like(centers)
                for c in range(k):
                    mask = labels == c
                    if mask.sum() > 0:
                        new_centers[c] = X[mask].mean(axis=0)
                    else:
                        new_centers[c] = centers[c]
                if np.allclose(new_centers, centers, atol=1e-6): break
                centers = new_centers
            inertia = sum(((X[labels == c] - centers[c])**2).sum() for c in range(k))
            if inertia < best_inertia:
                best_inertia, best_labels, best_centers = inertia, labels, centers
        return best_labels, best_centers, best_inertia

    def silhouette_np(X, labels, sample_n=2000):
        if len(X) > sample_n:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X), sample_n, replace=False)
            X = X[idx]; labels = labels[idx]
        # For each point: a = mean intra-cluster dist, b = min mean dist to any other cluster
        sils = []
        for i in range(len(X)):
            same = labels == labels[i]; same[i] = False
            other_clusters = set(labels) - {labels[i]}
            if same.sum() == 0 or len(other_clusters) == 0: continue
            a = np.mean(np.linalg.norm(X[same] - X[i], axis=1))
            b = min(np.mean(np.linalg.norm(X[labels == c] - X[i], axis=1)) for c in other_clusters)
            sils.append((b - a) / max(a, b))
        return float(np.mean(sils))

    print(f'\n{"k":>3s}  {"inertia":>12s}  {"silhouette":>11s}')
    results = []
    for k in range(3, 9):
        labels, centers_z, inertia = kmeans_np(Xz, k, n_iter=50, n_init=5)
        sil = silhouette_np(Xz, labels)
        results.append((k, float(inertia), sil, centers_z, labels))
        print(f'{k:>3d}  {inertia:>12.1f}  {sil:>11.3f}')

    # Pick best k by silhouette
    best = max(results, key=lambda x: x[2])
    bk, bi, bs, centers_z, bl = best
    print(f'\n[best k={bk}, silhouette={bs:.3f}]')

    # Cluster centers in original metric space
    centers = centers_z * std + mean  # back to original
    cluster_sizes = Counter(bl)
    print(f'\n=== Cluster centers (k={bk}) ===')
    print(f'{"metric":<30s}', end='')
    for c in range(bk):
        print(f'  c{c}({cluster_sizes[c]:4d})', end='')
    print()
    for i, m in enumerate(METRICS):
        print(f'  {m:<28s}', end='')
        for c in range(bk):
            print(f'  {centers[c, i]:>8.3f}', end='')
        print()
    return {'best_k': bk, 'silhouette': bs, 'cluster_sizes': dict(cluster_sizes),
            'centers': centers.tolist(), 'labels': bl.tolist()}


def part_b_orthogonality(L_eps):
    """For each L ep: compute v2 bucket multilabel.
    For each (v3 metric, v2 bucket) pair: compute correlation.
    Identify v3 metrics ORTHOGONAL to all v2 buckets."""
    print(f'\n=== Part B: v3 metric × v2 bucket orthogonality ===')
    # Compute v2 buckets per ep
    v2_indicators = []
    for e in L_eps:
        # v2 expects metrics dict
        v2 = classify_failure_v2(
            {k: e[k] for k in ('mean_ball_x', 'tail_mean_ball_x', 'team0_possession_ratio',
                                'team0_progress_toward_goal', 'team1_progress_toward_goal')},
            'team1_win'
        )
        ind = {b: 1 if b in v2['labels'] else 0 for b in V2_BUCKETS}
        v2_indicators.append(ind)

    # v2 bucket frequency
    print(f'v2 bucket frequencies:')
    for b in V2_BUCKETS:
        f = sum(i[b] for i in v2_indicators) / len(v2_indicators)
        print(f'  {b:<28s}  {f:.1%}')

    # Correlation matrix
    print(f'\n=== |Pearson(v3 metric, v2 bucket indicator)| ===')
    print(f'{"v3 metric":<30s}', end='')
    for b in V2_BUCKETS:
        print(f'  {b[:10]:>10s}', end='')
    print(f'  {"max|r|":>8s}')

    metric_max_r = []
    for m in METRICS:
        vec = np.array([e[m] for e in L_eps])
        max_r = 0
        max_b = None
        rs = []
        for b in V2_BUCKETS:
            ind = np.array([i[b] for i in v2_indicators], dtype=float)
            if vec.std() < 1e-10 or ind.std() < 1e-10:
                r = 0.0
            else:
                r = float(np.corrcoef(vec, ind)[0, 1])
            rs.append(r)
            if abs(r) > abs(max_r):
                max_r = r
                max_b = b
        marker = ' ✓ORTHO' if abs(max_r) < 0.30 else ''
        print(f'  {m:<28s}', end='')
        for r in rs:
            print(f'  {r:>+10.3f}', end='')
        print(f'  {max_r:>+8.3f}{marker}')
        metric_max_r.append({'metric': m, 'max_r': max_r, 'max_b': max_b, 'orthogonal': abs(max_r) < 0.30})

    # Summary: orthogonal metrics
    ortho = [r for r in metric_max_r if r['orthogonal']]
    print(f'\n=== {len(ortho)} v3 metrics ORTHOGONAL (|r| < 0.30) to all v2 buckets ===')
    for r in ortho:
        print(f'  {r["metric"]:<28s}  max|r|={abs(r["max_r"]):.3f} (vs {r["max_b"]})')

    return {'metric_max_r': metric_max_r, 'orthogonal_count': len(ortho)}


def main():
    L_eps = load_data()
    res_a = part_a_kmeans(L_eps)
    res_b = part_b_orthogonality(L_eps)
    out_path = 'docs/experiments/artifacts/v3_dataset/v3_cluster_orthogonality.json'
    with open(out_path, 'w') as f:
        json.dump({'kmeans': res_a, 'orthogonality': res_b}, f, indent=2, default=str)
    print(f'\n[save] {out_path}')


if __name__ == '__main__':
    main()
