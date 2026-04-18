#!/usr/bin/env python3
"""Filter noisy Ray 1.4 / ML-Agents lines while keeping training progress visible.

This is intentionally conservative about what it suppresses:
- Known Ray dashboard/metrics hostname noise on PACE
- Benign worker shutdown SystemExit traces during teardown
- Repetitive ML-Agents connection chatter

Full unfiltered logs still remain in Ray's result directories and `/tmp/ray/.../logs`.
"""

from __future__ import annotations

import re
import sys


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

SUPPRESS_LINE_SUBSTRINGS = (
    "Not all Ray CLI dependencies were found",
    "ray/autoscaler/_private/cli_logger.py:57: FutureWarning",
    "metric_exporter.cc:206: Export metrics to agent failed",
    "WARNING deprecation.py:33 -- DeprecationWarning: `simple_optimizer`",
    "INFO:mlagents_envs.environment:Connected to Unity environment",
    "INFO:mlagents_envs.environment:Connected new brain",
    "[INFO] Connected to Unity environment",
    "[INFO] Connected new brain",
    "fork_posix.cc:75] Other threads are currently calling into gRPC, skipping fork() handlers",
    "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR",
    "INFO trainer.py:696 -- Current log_level is WARN.",
    "warnings.warn(",
)

DASHBOARD_TRACEBACK_PATHS = (
    "ray/new_dashboard/agent.py",
    "ray/new_dashboard/modules/reporter/reporter_agent.py",
    "ray/_private/metrics_agent.py",
    "ray/_private/prometheus_exporter.py",
    "prometheus_client/exposition.py",
    "socket.gaierror: [Errno -2] Name or service not known",
)

WORKER_TEARDOWN_PATHS = (
    'File "python/ray/_raylet.pyx", line 591, in ray._raylet.task_execution_handler',
    "site-packages/ray/worker.py",
    "sigterm_handler",
    "SystemExit: 1",
)


def _clean(line: str) -> str:
    return ANSI_RE.sub("", line)


def _is_training_signal(line: str) -> bool:
    if not line.strip():
        return False
    if line.startswith("Training Configuration"):
        return True
    if line.startswith("Training Summary"):
        return True
    if line.startswith("Done training"):
        return True
    if line.startswith("  "):
        return True
    if "| it " in line and line.startswith("["):
        return True
    return False


def main() -> int:
    suppress_count = 0
    suppression_notice_printed = False

    for raw_line in sys.stdin:
        line = _clean(raw_line.rstrip("\n"))

        if suppress_count > 0:
            suppress_count -= 1
            if _is_training_signal(line):
                print(line, flush=True)
                continue
            if any(token in line for token in DASHBOARD_TRACEBACK_PATHS + WORKER_TEARDOWN_PATHS):
                continue
            continue

        if any(token in line for token in SUPPRESS_LINE_SUBSTRINGS):
            continue

        if "WARNING worker.py:1114 -- The agent on node" in line:
            suppress_count = 60
        elif "(raylet)" in line and "Traceback (most recent call last):" in line:
            suppress_count = 60
        elif any(token in line for token in DASHBOARD_TRACEBACK_PATHS):
            suppress_count = 60
        elif "ERROR worker.py:409 -- SystemExit was raised from the worker" in line:
            suppress_count = 20

        if suppress_count > 0:
            if not suppression_notice_printed:
                print("[filtered] Suppressing known Ray dashboard / teardown noise; full logs remain on disk.", flush=True)
                suppression_notice_printed = True
            continue

        print(line, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
