#!/usr/bin/env bash
# SNAPSHOT-048 §6 step 2 — R3 sanity check.
#
# Runs baseline vs baseline 100ep using existing evaluate_matches.
# Pass condition: WR(team0) in [0.42, 0.58]  (rough symmetry within ~2σ for n=100)
# Fail condition: WR(team0) significantly biased to one side -> baseline is NOT
# symmetric, hybrid takeover (which puts baseline in team0 slot) is unreliable.
#
# Usage:
#   bash scripts/eval/_hybrid_sanity_baseline_symmetry.sh [episodes] [base_port]
#
# Default: episodes=100, base_port=50101

set -euo pipefail

EPISODES="${1:-100}"
BASE_PORT="${2:-50101}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="docs/experiments/artifacts/hybrid-eval"
mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/sanity_baseline_symmetry_${EPISODES}ep.log"

PYTHON="${HYBRID_PYTHON:-/home/hice1/wsun377/.conda/envs/soccertwos/bin/python}"

echo "[R3-sanity] baseline vs baseline ${EPISODES}ep, base_port=${BASE_PORT}"
echo "[R3-sanity] log = ${LOG_PATH}"

set +e
"$PYTHON" -u -m cs8803drl.evaluation.evaluate_matches \
    --team0_module ceia_baseline_agent \
    --team1_module ceia_baseline_agent \
    --episodes "$EPISODES" \
    --base_port "$BASE_PORT" \
    2>&1 | tee "$LOG_PATH"
RC="${PIPESTATUS[0]}"
set -e

echo "[R3-sanity] exit=$RC"
echo
echo "[R3-sanity] interpretation:"
echo "  team0 WR in [0.42, 0.58]  -> PASS (baseline approximately symmetric)"
echo "  team0 WR outside that band -> FAIL (baseline has side bias)"
echo "                                hybrid takeover may be biased; flag in 048 verdict"
exit "$RC"
