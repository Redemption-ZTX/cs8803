#!/bin/bash
set -euo pipefail

# ============================================================
# Soccer-Twos Environment Setup
# Usage:
#   bash scripts/setup.sh              # Full setup (conda + deps + baseline)
#   bash scripts/setup.sh --deps-only  # Skip conda create, just install deps
#   bash scripts/setup.sh --verify     # Only run verification checks
#   bash scripts/setup.sh --no-baseline # Skip baseline agent download
#
# Works on: local machine, PACE cluster, any Linux/macOS with conda
# ============================================================

CONDA_ENV_NAME="soccertwos"
PYTHON_VERSION="3.8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Google Drive file ID for ceia_baseline_agent.zip
BASELINE_GDRIVE_ID="1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE"
BASELINE_ZIP="ceia_baseline_agent.zip"
BASELINE_DIR="ceia_baseline_agent"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Parse args ----
DEPS_ONLY=false
VERIFY_ONLY=false
NO_BASELINE=false
for arg in "$@"; do
    case "$arg" in
        --deps-only)    DEPS_ONLY=true ;;
        --verify)       VERIFY_ONLY=true ;;
        --no-baseline)  NO_BASELINE=true ;;
    esac
done

# ---- Verification function ----
verify() {
    info "Running verification checks..."
    local fail=0

    # 1. Python version
    if python --version 2>&1 | grep -q "Python 3.8"; then
        info "Python version: $(python --version 2>&1) ✓"
    else
        error "Python version: $(python --version 2>&1) — expected 3.8.x"
        fail=1
    fi

    # 2. Key packages
    for pkg in ray soccer_twos gym numpy torch; do
        if python -c "import $pkg" 2>/dev/null; then
            info "import $pkg ✓"
        else
            error "import $pkg FAILED"
            fail=1
        fi
    done

    # 3. Ray version
    ray_ver=$(python -c "import ray; print(ray.__version__)" 2>/dev/null || echo "N/A")
    if [[ "$ray_ver" == "1.4.0" ]]; then
        info "Ray version: $ray_ver ✓"
    else
        error "Ray version: $ray_ver — expected 1.4.0"
        fail=1
    fi

    # 4. protobuf / pydantic versions
    proto_ver=$(python -c "import google.protobuf; print(google.protobuf.__version__)" 2>/dev/null || echo "N/A")
    pydantic_ver=$(python -c "import pydantic; print(pydantic.VERSION)" 2>/dev/null || echo "N/A")
    if [[ "$proto_ver" == "3.20.3" ]]; then
        info "protobuf: $proto_ver ✓"
    else
        warn "protobuf: $proto_ver — expected 3.20.3"
    fi
    if [[ "$pydantic_ver" == "1.10.13" ]]; then
        info "pydantic: $pydantic_ver ✓"
    else
        warn "pydantic: $pydantic_ver — expected 1.10.13"
    fi

    # 5. Baseline agent directory
    if [[ -d "$PROJECT_ROOT/$BASELINE_DIR" ]] && [[ -f "$PROJECT_ROOT/$BASELINE_DIR/agent_ray.py" ]]; then
        info "$BASELINE_DIR/ ✓"
    else
        error "$BASELINE_DIR/ not found — run setup without --no-baseline, or download manually"
        fail=1
    fi

    # 6. Baseline checkpoint (the actual model weights)
    local ckpt="$PROJECT_ROOT/$BASELINE_DIR/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449"
    if [[ -f "$ckpt" ]]; then
        info "Baseline checkpoint ✓"
    else
        error "Baseline checkpoint not found at expected path"
        error "  Expected: $ckpt"
        fail=1
    fi

    # 7. Baseline params.pkl (required by _get_baseline_policy)
    local params="$PROJECT_ROOT/$BASELINE_DIR/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/params.pkl"
    if [[ -f "$params" ]]; then
        info "Baseline params.pkl ✓"
    else
        error "Baseline params.pkl not found"
        fail=1
    fi

    # 8. Project structure
    for f in utils.py train_ray_selfplay.py train_ray_team_vs_random_shaping.py; do
        if [[ -f "$PROJECT_ROOT/$f" ]]; then
            info "$f ✓"
        else
            error "$f not found"
            fail=1
        fi
    done

    if [[ $fail -eq 0 ]]; then
        info "All checks passed. Environment is ready."
    else
        error "Some checks failed. See above."
        return 1
    fi
}

if $VERIFY_ONLY; then
    verify
    exit $?
fi

# ---- Step 1: Conda environment ----
if ! $DEPS_ONLY; then
    info "Creating conda environment: $CONDA_ENV_NAME (Python $PYTHON_VERSION)"

    # Detect if we need to source conda
    if ! command -v conda &>/dev/null; then
        for conda_sh in \
            ~/miniconda3/etc/profile.d/conda.sh \
            ~/anaconda3/etc/profile.d/conda.sh \
            /opt/conda/etc/profile.d/conda.sh; do
            if [[ -f "$conda_sh" ]]; then
                source "$conda_sh"
                break
            fi
        done
    fi

    if ! command -v conda &>/dev/null; then
        error "conda not found. Install Miniconda first."
        exit 1
    fi

    # Remove existing env if present
    if conda info --envs | grep -q "^$CONDA_ENV_NAME "; then
        warn "Environment '$CONDA_ENV_NAME' already exists. Removing..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$CONDA_ENV_NAME" -y
    fi

    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    conda activate "$CONDA_ENV_NAME"
fi

# ---- Step 2: Pin build tools ----
info "Pinning build tools for compatibility..."
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge

# ---- Step 3: Install requirements ----
info "Installing requirements..."
cd "$PROJECT_ROOT"
pip install -r requirements.txt

# ---- Step 4: Fix compatibility ----
info "Installing compatibility pins..."
pip install protobuf==3.20.3 pydantic==1.10.13

# ---- Step 5: Download baseline agent ----
if ! $NO_BASELINE; then
    if [[ -d "$PROJECT_ROOT/$BASELINE_DIR" ]] && [[ -f "$PROJECT_ROOT/$BASELINE_DIR/agent_ray.py" ]]; then
        info "Baseline agent already exists, skipping download."
    else
        info "Downloading baseline agent from Google Drive..."

        # Install gdown if not available
        if ! command -v gdown &>/dev/null; then
            pip install gdown
        fi

        cd "$PROJECT_ROOT"
        gdown --id "$BASELINE_GDRIVE_ID" -O "$BASELINE_ZIP"

        if [[ -f "$BASELINE_ZIP" ]]; then
            info "Extracting baseline agent..."
            unzip -o "$BASELINE_ZIP" -d "$PROJECT_ROOT"
            rm -f "$BASELINE_ZIP"
            info "Baseline agent extracted to $BASELINE_DIR/"
        else
            error "Download failed. Download manually from:"
            error "  https://drive.google.com/file/d/$BASELINE_GDRIVE_ID/view?usp=sharing"
            error "Extract to: $PROJECT_ROOT/$BASELINE_DIR/"
        fi
    fi
else
    info "Skipping baseline download (--no-baseline)."
fi

# ---- Step 6: Verify ----
verify

info "Setup complete. Activate with: conda activate $CONDA_ENV_NAME"
