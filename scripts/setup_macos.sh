#!/usr/bin/env bash
# One-shot setup for CorridorKey 2.0 on an Apple Silicon Mac.
#
#   bash scripts/setup_macos.sh
#
# Checks the machine, installs ffmpeg with Homebrew if it is missing, creates
# ./venv, installs PyTorch (with Metal/MPS support) and the remaining
# dependencies, then downloads the model weights (about 5 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script is for macOS. On Linux/Windows follow the README setup." >&2
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "CorridorKey on macOS needs an Apple Silicon (M-series) Mac." >&2
    echo "If you are on Apple Silicon, make sure this shell is not running under Rosetta." >&2
    exit 1
fi
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
if [ "$macos_major" -lt 14 ]; then
    echo "macOS 14 (Sonoma) or newer is required for bfloat16 on the GPU." >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        echo "Installing ffmpeg with Homebrew..."
        brew install ffmpeg
    else
        echo "ffmpeg not found. Install Homebrew (https://brew.sh) and run 'brew install ffmpeg'," >&2
        echo "or put an ffmpeg binary on your PATH, then re-run this script." >&2
        exit 1
    fi
fi

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10+ is required ($("$PYTHON" --version 2>&1) found)." >&2
    echo "Install one with 'brew install python@3.12' and re-run with PYTHON=python3.12." >&2
    exit 1
fi

if [ ! -x venv/bin/python ]; then
    "$PYTHON" -m venv venv
fi
venv/bin/python -m pip install --upgrade pip
# The default PyPI wheels for macOS arm64 include MPS support.
venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1
venv/bin/python -m pip install -r requirements.txt

venv/bin/python - <<'PY'
import torch
assert torch.backends.mps.is_available(), "PyTorch cannot see the Apple GPU (MPS)."
print(f"torch {torch.__version__}: MPS available, "
      f"{torch.mps.recommended_max_memory() / 2**30:.1f} GB Metal working set")
PY

venv/bin/python scripts/download_weights.py
echo
echo "Setup complete. Double-click CorridorKey_DRAG_CLIPS_HERE.command to start."
