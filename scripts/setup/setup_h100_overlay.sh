#!/bin/bash
set -euo pipefail

# ============================================================
# H100-compatible overlay environment for Soccer-Twos
#
# This script keeps the original `soccertwos` environment intact and
# creates a small overlay venv that reuses its site-packages while
# overriding torch with an H100-compatible CUDA build.
#
# Usage:
#   bash scripts/setup/setup_h100_overlay.sh
#   bash scripts/setup/setup_h100_overlay.sh --verify-only
#   bash scripts/setup/setup_h100_overlay.sh --force
#
# Environment overrides:
#   BASE_PYTHON=/path/to/base/env/bin/python
#   TARGET_ENV=/path/to/overlay/venv
#   TORCH_VERSION=2.1.2+cu121
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# ============================================================

BASE_PYTHON="${BASE_PYTHON:-$HOME/.conda/envs/soccertwos/bin/python}"
TARGET_ENV="${TARGET_ENV:-$HOME/.venvs/soccertwos_h100}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2+cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PIP_VERSION="23.3.2"
SETUPTOOLS_VERSION="65.5.0"
WHEEL_VERSION="0.38.4"
TYPING_EXTENSIONS_VERSION="4.13.2"

VERIFY_ONLY=false
FORCE=false
REQUIRE_GPU_VERIFY=false

for arg in "$@"; do
    case "$arg" in
        --verify-only) VERIFY_ONLY=true ;;
        --force) FORCE=true ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }
error() { echo "[ERROR] $*" >&2; }

verify_overlay() {
    if [[ ! -x "$TARGET_ENV/bin/python" ]]; then
        error "Overlay env not found: $TARGET_ENV"
        exit 1
    fi

    info "Verifying overlay environment..."
    REQUIRE_GPU_VERIFY="$REQUIRE_GPU_VERIFY" "$TARGET_ENV/bin/python" - <<'PY'
import os
import torch
import ray
import soccer_twos
import mlagents

require_gpu = os.environ.get("REQUIRE_GPU_VERIFY") == "true"

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_file", torch.__file__)
print("ray_file", ray.__file__)
print("soccer_twos_file", soccer_twos.__file__)
print("mlagents_file", mlagents.__file__)

if torch.cuda.is_available():
    x = torch.tensor([1.0], device="cuda")
    print("cuda_tensor", x.item())
    print("cuda_device", torch.cuda.get_device_name(0))
elif require_gpu:
    raise SystemExit("torch.cuda.is_available() == False")
else:
    print("cuda_available", False)
    print("cuda_note", "No GPU visible in this shell; run --verify-only on a GPU node for CUDA validation.")
PY

    local pip_check_output
    pip_check_output="$("$TARGET_ENV/bin/python" -m pip check 2>&1 || true)"
    if [[ -n "$pip_check_output" ]]; then
        echo "$pip_check_output"
        if [[ "$pip_check_output" != *"mlagents 0.27.0 has requirement torch<1.9.0"* ]]; then
            error "Unexpected pip check output above."
            exit 1
        fi
        warn "Known metadata mismatch retained: mlagents 0.27.0 still pins torch<1.9.0."
    fi

    info "Overlay verification passed."
}

if [[ "$VERIFY_ONLY" == true ]]; then
    REQUIRE_GPU_VERIFY=true
    verify_overlay
    exit 0
fi

if [[ ! -x "$BASE_PYTHON" ]]; then
    error "Base python not found: $BASE_PYTHON"
    error "Set BASE_PYTHON=/path/to/existing/soccertwos/bin/python and retry."
    exit 1
fi

if [[ -d "$TARGET_ENV" && "$FORCE" == true ]]; then
    warn "Removing existing overlay env: $TARGET_ENV"
    rm -rf "$TARGET_ENV"
fi

if [[ ! -d "$TARGET_ENV" ]]; then
    info "Creating overlay venv at $TARGET_ENV"
    mkdir -p "$(dirname "$TARGET_ENV")"
    "$BASE_PYTHON" -m venv --system-site-packages "$TARGET_ENV"
else
    info "Reusing existing overlay venv at $TARGET_ENV"
fi

info "Upgrading build tools in overlay env"
"$TARGET_ENV/bin/python" -m pip install --upgrade \
    "pip==$PIP_VERSION" \
    "setuptools==$SETUPTOOLS_VERSION" \
    "wheel==$WHEEL_VERSION"

info "Installing H100-compatible torch into overlay env"
"$TARGET_ENV/bin/python" -m pip install --upgrade --ignore-installed \
    "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX_URL"

info "Restoring Python 3.8-compatible typing-extensions for shared deps"
"$TARGET_ENV/bin/python" -m pip install --upgrade \
    "typing-extensions==$TYPING_EXTENSIONS_VERSION"

verify_overlay

cat <<EOF

[INFO] Overlay env ready.
[INFO] Use this interpreter for GPU runs:
       $TARGET_ENV/bin/python

[INFO] Example GPU smoke test:
       RUN_NAME=PPO_smoke_rewardfix_gpu_h100_overlay \\
       TIMESTEPS_TOTAL=4000 TIME_TOTAL_S=1800 CHECKPOINT_FREQ=1 \\
       NUM_GPUS=1 FRAMEWORK=torch \\
       $TARGET_ENV/bin/python -m cs8803drl.training.train_ray_team_vs_random_shaping
EOF
