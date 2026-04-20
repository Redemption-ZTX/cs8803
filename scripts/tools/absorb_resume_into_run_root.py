#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _safe_link_name(resume_root: Path, child: Path) -> str:
    return f"{resume_root.name}__{child.name}"


def _dedupe_eval_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        key = (
            row.get("checkpoint_dir", ""),
            row.get("checkpoint_file", ""),
            row.get("checkpoint_iteration", ""),
            row.get("opponent", ""),
            row.get("status", ""),
            row.get("log_path", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(
        key=lambda row: (
            int(float(row.get("checkpoint_iteration") or 0)),
            row.get("opponent", ""),
            row.get("checkpoint_dir", ""),
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Absorb split resume roots back into a canonical run root."
    )
    parser.add_argument(
        "--canonical-run-dir",
        required=True,
        help="Original canonical run root that should become the long-term single source of truth.",
    )
    parser.add_argument(
        "--resume-run-dir",
        action="append",
        required=True,
        help="Resume run roots to absorb back into the canonical run root. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink",),
        default="symlink",
        help="How to expose absorbed trial dirs inside the canonical root. Currently only symlink is supported.",
    )
    args = parser.parse_args()

    canonical_root = Path(args.canonical_run_dir).resolve()
    if not canonical_root.exists():
        raise SystemExit(f"Canonical run root does not exist: {canonical_root}")

    resume_roots = [Path(path).resolve() for path in args.resume_run_dir]
    for resume_root in resume_roots:
        if not resume_root.exists():
            raise SystemExit(f"Resume run root does not exist: {resume_root}")

    source_lines = ["Canonical run root:", str(canonical_root), "", "Absorbed resume roots:"]

    for resume_root in resume_roots:
        source_lines.append(str(resume_root))
        for child in sorted(resume_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "checkpoint_eval_logs":
                continue
            link_name = _safe_link_name(resume_root, child)
            link_path = canonical_root / link_name
            if link_path.exists() or link_path.is_symlink():
                continue
            os.symlink(child, link_path, target_is_directory=True)

    merged_eval_rows: List[Dict[str, str]] = []
    for eval_csv in [canonical_root / "checkpoint_eval.csv"] + [root / "checkpoint_eval.csv" for root in resume_roots]:
        for row in _load_eval_rows(eval_csv):
            merged_eval_rows.append(dict(row))
    merged_eval_rows = _dedupe_eval_rows(merged_eval_rows)

    backup_eval_csv = canonical_root / "checkpoint_eval.pre_absorb_backup.csv"
    canonical_eval_csv = canonical_root / "checkpoint_eval.csv"
    if canonical_eval_csv.exists() and not backup_eval_csv.exists():
        canonical_eval_csv.replace(backup_eval_csv)

    fieldnames = _union_fieldnames(merged_eval_rows)
    if merged_eval_rows and fieldnames:
        with canonical_eval_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_eval_rows:
                writer.writerow(row)

    (canonical_root / "absorbed_resume_sources.txt").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )

    summary_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/tools/print_merged_training_summary.py"),
        "--run-dir",
        str(canonical_root),
        "--output",
        "merged_training_summary.txt",
    ]
    subprocess.run(summary_cmd, cwd=str(REPO_ROOT), check=True)
    print(f"Absorbed {len(resume_roots)} resume roots into {canonical_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
