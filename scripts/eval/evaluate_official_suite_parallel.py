#!/usr/bin/env python
"""Parallel variant of evaluate_official_suite.

Accuracy-preserving: each worker is an isolated subprocess with its own
Unity base_port and Ray session temp dir, so numerical results are
bit-identical to the serial script. Only the scheduling changes.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from cs8803drl.core.checkpoint_utils import resolve_checkpoint_file  # noqa: E402


DEFAULT_H100_PYTHON = Path.home() / ".venvs" / "soccertwos_h100" / "bin" / "python"
PORT_STEP = 20
MAX_BASE_PORT = 65505


def _resolve_eval_checkpoint(path: str) -> str:
    candidate = Path(path).resolve()
    if candidate.is_dir():
        model_pt = candidate / "model.pt"
        metadata_json = candidate / "metadata.json"
        if model_pt.exists() and metadata_json.exists():
            return str(candidate)
    return resolve_checkpoint_file(str(candidate))


def _default_python() -> str:
    if DEFAULT_H100_PYTHON.exists():
        return str(DEFAULT_H100_PYTHON)
    return sys.executable


def _safe_base_port(base_port: int, offset: int = 0) -> int:
    base_port = int(base_port)
    if base_port < 1024 or base_port > MAX_BASE_PORT:
        raise ValueError(
            f"base_port must be in [1024, {MAX_BASE_PORT}] for official evaluator, got {base_port}"
        )
    span = ((MAX_BASE_PORT - base_port) // PORT_STEP) + 1
    return base_port + (int(offset) % span) * PORT_STEP


def _normalize_opponents(raw: str):
    mapping = {
        "baseline": "ceia_baseline_agent",
        "random": "example_player_agent.agent_random",
    }
    out = []
    for piece in (raw or "baseline").split(","):
        name = piece.strip()
        if not name:
            continue
        out.append((name if name in mapping else name, mapping.get(name, name)))
    if not out:
        raise ValueError("At least one opponent must be specified.")
    return out


def _extract_official_metrics(output_text: str, team0_module: str):
    lines = output_text.replace("\r", "\n").splitlines()
    policy_header = f"{team0_module}:"
    in_policy = False
    metrics = {}

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if not in_policy:
            if stripped == policy_header:
                in_policy = True
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2 and not stripped.startswith("policy_"):
            break

        match = re.match(r"policy_(wins|losses|draws|win_rate):\s+(.+)", stripped)
        if match:
            key = match.group(1)
            value = match.group(2)
            if key == "win_rate":
                metrics[key] = float(value)
            else:
                metrics[key] = int(float(value))

    required = {"wins", "losses", "draws", "win_rate"}
    if not required.issubset(metrics):
        raise RuntimeError(
            f"Could not parse official evaluator output for {team0_module}. Parsed={metrics}\n"
            f"--- output tail ---\n{os.linesep.join(lines[-80:])}"
        )
    return metrics


def _run_one(
    *,
    checkpoint: str,
    team0_module: str,
    opponent_label: str,
    opponent_module: str,
    episodes: int,
    base_port: int,
    python_bin: str,
    ray_tmp_dir: str,
):
    checkpoint_file = _resolve_eval_checkpoint(checkpoint)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["TRAINED_RAY_CHECKPOINT"] = checkpoint_file
    env["RAY_DISABLE_DASHBOARD"] = "1"
    env["RAY_USAGE_STATS_ENABLED"] = "0"
    # per-worker isolation: each subprocess gets its own Ray session dir.
    # Ray 1.4 auto-retries the redis port starting from 6379, so parallel
    # workers will naturally land on distinct ports.
    env["RAY_SESSION_TMPDIR_OVERRIDE"] = ray_tmp_dir
    os.makedirs(ray_tmp_dir, exist_ok=True)

    cmd = [
        python_bin,
        "-m",
        "soccer_twos.evaluate",
        "-m1",
        team0_module,
        "-m2",
        opponent_module,
        "-e",
        str(int(episodes)),
        "-p",
        str(int(base_port)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"Official evaluation failed for {checkpoint_file} vs {opponent_label} "
            f"with exit code {proc.returncode}.\n{output}"
        )

    metrics = _extract_official_metrics(output, team0_module)
    return checkpoint_file, metrics, output


def _run_one_task(task):
    t0 = time.time()
    checkpoint_file, metrics, output = _run_one(**task["kwargs"])
    elapsed = time.time() - t0
    return {
        "task_idx": task["task_idx"],
        "checkpoint_file": checkpoint_file,
        "opponent_label": task["kwargs"]["opponent_label"],
        "metrics": metrics,
        "output": output,
        "elapsed": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint dir or file. Repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--team0-module",
        default="cs8803drl.deployment.trained_ray_agent",
        help="Team0 agent module to evaluate.",
    )
    parser.add_argument(
        "--opponents",
        default="baseline",
        help="Comma-separated opponents. baseline,random or explicit module names.",
    )
    parser.add_argument("-n", "--episodes", type=int, default=100)
    parser.add_argument("--base-port", type=int, default=65105)
    parser.add_argument("--python-bin", default=_default_python())
    parser.add_argument(
        "--save-logs-dir",
        default="",
        help="Optional directory to store raw official evaluator logs.",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=1,
        help="Max concurrent evaluations. 1 = serial (default).",
    )
    parser.add_argument(
        "--ray-tmp-root",
        default="/tmp/ray_eval_parallel",
        help="Root dir for per-worker Ray session temp dirs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if int(args.episodes) < 2:
        raise SystemExit(
            "Official soccer_twos.evaluate requires at least 2 episodes because its summary "
            "expects both blue and orange side stats."
        )
    opponents = _normalize_opponents(args.opponents)
    save_logs_dir = Path(args.save_logs_dir).resolve() if args.save_logs_dir else None
    if save_logs_dir is not None:
        save_logs_dir.mkdir(parents=True, exist_ok=True)

    # Build task list. Each task is one (checkpoint, opponent) pair with its
    # pre-assigned Unity base_port and Ray session ports.
    tasks = []
    for ckpt_idx, checkpoint in enumerate(args.checkpoint):
        ckpt_file_display = _resolve_eval_checkpoint(checkpoint)
        for opp_idx, (opponent_label, opponent_module) in enumerate(opponents):
            task_idx = len(tasks)
            current_port = _safe_base_port(int(args.base_port), task_idx)
            ray_tmp_dir = os.path.join(args.ray_tmp_root, f"worker_{task_idx:03d}")
            tasks.append(
                {
                    "task_idx": task_idx,
                    "display_name": f"{Path(ckpt_file_display).name} vs {opponent_label}",
                    "kwargs": {
                        "checkpoint": checkpoint,
                        "team0_module": args.team0_module,
                        "opponent_label": opponent_label,
                        "opponent_module": opponent_module,
                        "episodes": int(args.episodes),
                        "base_port": current_port,
                        "python_bin": args.python_bin,
                        "ray_tmp_dir": ray_tmp_dir,
                    },
                }
            )

    parallel = max(1, int(args.parallel))
    parallel = min(parallel, len(tasks))
    print(
        f"[suite-parallel] {len(tasks)} tasks, parallel={parallel}, "
        f"episodes={args.episodes}, base_port={args.base_port}",
        flush=True,
    )
    for task in tasks:
        kw = task["kwargs"]
        print(
            f"  task#{task['task_idx']:03d} {task['display_name']} "
            f"unity_port={kw['base_port']} ray_tmp={kw['ray_tmp_dir']}",
            flush=True,
        )

    recap = [None] * len(tasks)
    t_start = time.time()

    if parallel == 1:
        # Serial path — behaves identically to evaluate_official_suite.py.
        for task in tasks:
            print(f"=== Official Eval: {task['display_name']} (task#{task['task_idx']}) ===", flush=True)
            result = _run_one_task(task)
            if save_logs_dir is not None:
                safe_name = (
                    f"{Path(result['checkpoint_file']).name}"
                    f"_vs_{result['opponent_label']}.log"
                )
                (save_logs_dir / safe_name).write_text(result["output"])
            recap[result["task_idx"]] = result
            print(
                f"win_rate={result['metrics']['win_rate']:.3f} "
                f"({result['metrics']['wins']}W-{result['metrics']['losses']}L-"
                f"{result['metrics']['draws']}T) "
                f"elapsed={result['elapsed']:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_run_one_task, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    print(
                        f"[suite-parallel] task#{task['task_idx']} FAILED "
                        f"({task['display_name']}): {exc}",
                        flush=True,
                    )
                    raise
                if save_logs_dir is not None:
                    safe_name = (
                        f"{Path(result['checkpoint_file']).name}"
                        f"_vs_{result['opponent_label']}.log"
                    )
                    (save_logs_dir / safe_name).write_text(result["output"])
                recap[result["task_idx"]] = result
                print(
                    f"[done task#{result['task_idx']:03d}] {task['display_name']}: "
                    f"win_rate={result['metrics']['win_rate']:.3f} "
                    f"({result['metrics']['wins']}W-{result['metrics']['losses']}L-"
                    f"{result['metrics']['draws']}T) "
                    f"elapsed={result['elapsed']:.1f}s",
                    flush=True,
                )

    total_elapsed = time.time() - t_start
    print("", flush=True)
    print("=== Official Suite Recap (parallel) ===", flush=True)
    for result in recap:
        if result is None:
            continue
        print(
            f"{result['checkpoint_file']} vs {result['opponent_label']}: "
            f"win_rate={result['metrics']['win_rate']:.3f} "
            f"({result['metrics']['wins']}W-{result['metrics']['losses']}L-"
            f"{result['metrics']['draws']}T)",
            flush=True,
        )
    print(
        f"[suite-parallel] total_elapsed={total_elapsed:.1f}s "
        f"tasks={len(tasks)} parallel={parallel}",
        flush=True,
    )


if __name__ == "__main__":
    main()
