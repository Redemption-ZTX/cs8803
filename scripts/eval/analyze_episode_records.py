#!/usr/bin/env python
"""Aggregate saved episode JSON records into label and metric summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _step_stats(values: List[int]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    idx_75 = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.75)))
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = float(ordered[mid])
    else:
        median = float((ordered[mid - 1] + ordered[mid]) / 2.0)
    return {
        "count": int(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": median,
        "p75": float(ordered[idx_75]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _load_records(episodes_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(episodes_dir.glob("episode_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    outcome_counter = Counter()
    primary_counter = Counter()
    label_counter = Counter()
    all_steps: List[int] = []
    steps_by_outcome: Dict[str, List[int]] = defaultdict(list)
    metrics_all: Dict[str, List[float]] = defaultdict(list)
    metrics_by_primary: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        outcome = str(record.get("outcome", "unknown"))
        classification = record.get("classification", {}) or {}
        primary = str(classification.get("primary_label", "unknown"))
        labels = [str(x) for x in classification.get("labels", [])]
        metrics = record.get("metrics", {}) or {}
        steps = int(record.get("steps", 0) or 0)

        outcome_counter[outcome] += 1
        primary_counter[primary] += 1
        label_counter.update(labels)

        all_steps.append(steps)
        steps_by_outcome[outcome].append(steps)

        for key, value in metrics.items():
            if value is None:
                continue
            try:
                val = float(value)
            except Exception:
                continue
            metrics_all[str(key)].append(val)
            metrics_by_primary[primary][str(key)].append(val)

    return {
        "episodes": int(len(records)),
        "outcomes": dict(outcome_counter),
        "primary_labels": dict(primary_counter),
        "labels": dict(label_counter),
        "steps_all": _step_stats(all_steps),
        "steps_by_outcome": {
            outcome: _step_stats(values) for outcome, values in steps_by_outcome.items()
        },
        "metrics_all": {key: _mean(values) for key, values in metrics_all.items()},
        "metrics_by_primary_label": {
            primary: {key: _mean(values) for key, values in metrics.items()}
            for primary, metrics in metrics_by_primary.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", action="append", required=True)
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: Dict[str, Any] = {}
    for raw_dir in args.episodes_dir:
        episodes_dir = Path(raw_dir).resolve()
        records = _load_records(episodes_dir)
        summary = _aggregate(records)
        results[str(episodes_dir)] = summary

        print(f"=== Episode Record Analysis: {episodes_dir} ===")
        print(f"episodes: {summary['episodes']}")
        print(f"outcomes: {summary['outcomes']}")
        print(f"primary_labels: {summary['primary_labels']}")
        print(f"labels: {summary['labels']}")
        if summary["steps_all"] is not None:
            steps = summary["steps_all"]
            print(
                "steps_all: "
                f"mean={steps['mean']:.1f} median={steps['median']:.1f} "
                f"p75={steps['p75']:.1f} min={steps['min']:.0f} max={steps['max']:.0f}"
            )
        print("")

    if args.save_json:
        out_path = Path(args.save_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
