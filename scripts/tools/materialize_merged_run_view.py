#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List


def _load_eval_rows(eval_csv: Path) -> List[Dict[str, str]]:
    if not eval_csv.exists():
        return []
    with eval_csv.open(newline="", encoding="utf-8", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def _union_fieldnames(rows: List[Dict[str, str]]) -> List[str]:
    seen = set()
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _safe_link_name(run_root: Path, child: Path) -> str:
    return f"{run_root.name}__{child.name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a merged run-root view from multiple split Ray experiment roots."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Source run roots. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output-run-dir",
        required=True,
        help="Destination merged-view root under ray_results/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate output root if it already exists.",
    )
    args = parser.parse_args()

    run_roots = [Path(path).resolve() for path in args.run_dir]
    for run_root in run_roots:
        if not run_root.exists():
            raise SystemExit(f"Run root does not exist: {run_root}")

    output_root = Path(args.output_run_dir).resolve()
    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"Output run root already exists: {output_root}. Use --force to recreate it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    merged_eval_rows: List[Dict[str, str]] = []
    source_lines = ["Merged run roots:"]

    for run_root in run_roots:
        source_lines.append(str(run_root))

        for child in sorted(run_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "checkpoint_eval_logs":
                continue
            link_name = _safe_link_name(run_root, child)
            link_path = output_root / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(child, link_path, target_is_directory=True)

        eval_csv = run_root / "checkpoint_eval.csv"
        rows = _load_eval_rows(eval_csv)
        for row in rows:
            row_copy = dict(row)
            row_copy["_source_eval_csv"] = str(eval_csv)
            row_copy["_source_run_root"] = str(run_root)
            merged_eval_rows.append(row_copy)

    merged_eval_rows.sort(
        key=lambda row: (
            int(float(row.get("checkpoint_iteration") or 0)),
            row.get("opponent", ""),
            row.get("checkpoint_dir", ""),
            row.get("_source_run_root", ""),
        )
    )

    fieldnames = _union_fieldnames(merged_eval_rows)
    if merged_eval_rows and fieldnames:
        with (output_root / "checkpoint_eval.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_eval_rows:
                writer.writerow(row)

    (output_root / "merged_view_sources.txt").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )
    print(f"Created merged run view: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
