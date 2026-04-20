#!/usr/bin/env bash
# SNAPSHOT-048 hybrid eval launcher (one condition).
#
# Usage:
#   bash scripts/eval/_hybrid_eval_run.sh \
#       <student_module> <checkpoint_path> <trigger {none,alpha,beta}> \
#       <episodes> <base_port> <out_json>
#
# Example (one condition):
#   bash scripts/eval/_hybrid_eval_run.sh \
#       cs8803drl.deployment.trained_team_ray_agent \
#       ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040 \
#       alpha 1000 50001 \
#       docs/experiments/artifacts/hybrid-eval/031A_alpha_1000.json

set -euo pipefail

if [[ "$#" -lt 6 ]]; then
    echo "Usage: $0 <student_module> <checkpoint_path> <trigger> <episodes> <base_port> <out_json>" >&2
    exit 1
fi

STUDENT_MODULE="$1"
CHECKPOINT="$2"
TRIGGER="$3"
EPISODES="$4"
BASE_PORT="$5"
OUT_JSON="$6"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$(dirname "$OUT_JSON")"
LOG_PATH="${OUT_JSON%.json}.log"

PYTHON="${HYBRID_PYTHON:-/home/hice1/wsun377/.conda/envs/soccertwos/bin/python}"

echo "[launch] student     = $STUDENT_MODULE"
echo "[launch] checkpoint  = $CHECKPOINT"
echo "[launch] trigger     = $TRIGGER"
echo "[launch] episodes    = $EPISODES"
echo "[launch] base_port   = $BASE_PORT"
echo "[launch] out_json    = $OUT_JSON"
echo "[launch] log         = $LOG_PATH"

set +e
"$PYTHON" -u -m cs8803drl.evaluation.evaluate_hybrid \
    --student-module "$STUDENT_MODULE" \
    --student-checkpoint "$CHECKPOINT" \
    --trigger "$TRIGGER" \
    --episodes "$EPISODES" \
    --base-port "$BASE_PORT" \
    --json-out "$OUT_JSON" \
    2>&1 | tee "$LOG_PATH"
RC="${PIPESTATUS[0]}"
set -e

echo "[done] exit=$RC  json=$OUT_JSON"
exit "$RC"
