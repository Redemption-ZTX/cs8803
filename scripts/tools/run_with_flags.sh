#!/bin/bash
# run_with_flags.sh — universal flag manager for training / eval / capture / backfill.
#
# Usage:
#   bash scripts/tools/run_with_flags.sh <LANE_TAG> -- <command> [args...]
#
# Example:
#   bash scripts/tools/run_with_flags.sh 054 -- bash scripts/eval/_launch_054_mat_min_scratch.sh
#   bash scripts/tools/run_with_flags.sh 054_backfill -- python scripts/eval/backfill_run_eval.py --run-dir ...
#
# Manages:
#   docs/experiments/artifacts/slurm-logs/<LANE_TAG>.running   (during execution)
#   docs/experiments/artifacts/slurm-logs/<LANE_TAG>.done      (after exit, with accurate code)
#
# Guarantees (vs naked launchers):
#   - ${PIPESTATUS[0]} captures python's real exit (not tee's)
#   - trap cleanup fires on clean exit, SIGTERM, SIGINT, uncaught error
#   - .running is atomically removed and .done atomically written
#   - Works for ANY command (not just training) — eval/capture/backfill/H2H all supported
#
# Flag semantics:
#   - .running exists  = process alive (or crashed without cleanup — rare, only if kill -9 parent)
#   - .done exists     = process has exited; content = "EXIT_CODE=<N> at <timestamp>"
#   - Both absent      = lane never launched (or cleanup scrubbed both)
#   - Both present     = transient during cleanup transition (~<1ms); readers should re-check

set -o pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  echo "Usage: $0 <LANE_TAG> -- <command> [args...]" >&2
  exit 2
fi

LANE_TAG="$1"
shift 2  # drop LANE_TAG + --

# Resolve slurm-logs dir relative to repo root (follows symlinks if called from elsewhere)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SLURM_LOG_DIR="$REPO_ROOT/docs/experiments/artifacts/slurm-logs"
mkdir -p "$SLURM_LOG_DIR"

RUNNING_FLAG="$SLURM_LOG_DIR/${LANE_TAG}.running"
DONE_FLAG="$SLURM_LOG_DIR/${LANE_TAG}.done"
DONE_TMP="$SLURM_LOG_DIR/.${LANE_TAG}.done.tmp"

cleanup() {
  local rc=$?
  # Atomic done-flag write: stage to tmp then mv
  {
    echo "EXIT_CODE=$rc"
    echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "LANE_TAG=$LANE_TAG"
    echo "PID=$$"
  } > "$DONE_TMP" 2>/dev/null
  mv -f "$DONE_TMP" "$DONE_FLAG" 2>/dev/null || true
  rm -f "$RUNNING_FLAG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

# Clear stale done flag and raise running flag
rm -f "$DONE_FLAG"
touch "$RUNNING_FLAG"

# Execute the wrapped command — pipefail propagates python's exit code through tee
"$@"
exit_code=$?

# Fallthrough: return exit_code (trap will run)
exit $exit_code
