#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_dir_or_checkpoint_file> [episodes] [base_port]" >&2
  echo "Example: $0 ray_results/.../checkpoint_000121 10 9100" >&2
  exit 2
fi

TARGET="$1"
EPISODES="${2:-10}"
BASE_PORT="${3:-9100}"

if [[ -d "$TARGET" ]]; then
  # Accept either checkpoint_000121/ or .../checkpoint_000121/checkpoint-121
  CKPT_FILE=$(ls -1 "$TARGET"/checkpoint-* 2>/dev/null | head -n 1 || true)
  if [[ -z "${CKPT_FILE}" ]]; then
    echo "No checkpoint-* file found in directory: $TARGET" >&2
    exit 1
  fi
else
  CKPT_FILE="$TARGET"
fi

export TRAINED_RAY_CHECKPOINT="$CKPT_FILE"

python evaluate_matches.py \
  -m1 trained_ray_agent \
  -m2 ceia_baseline_agent \
  -n "$EPISODES" \
  --base_port "$BASE_PORT"
