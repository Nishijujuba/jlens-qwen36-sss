#!/usr/bin/env bash
set -euo pipefail

# Why a plain venv instead of a container?  VS Code's WSL debugger can attach
# directly to this interpreter, CUDA is exposed by the Windows driver, and the
# setup has fewer moving parts for a first interpretability experiment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script must run inside WSL/Linux." >&2
  exit 2
fi
if [[ "$ROOT" == /mnt/* ]]; then
  echo "WARNING: the repository is on a Windows-mounted drive ($ROOT)."
  echo "         For faster model/cache I/O, clone it under ~/src inside WSL."
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is missing. Install Python 3.11 or 3.12 in the WSL distribution." >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.11-3.13 required; found {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# The CUDA wheel already bundles the user-space CUDA runtime.  WSL receives
# the GPU driver from Windows; no Linux NVIDIA display driver is installed.
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch
python -m pip install --no-build-isolation -e '.[dev]'

if [[ "${INSTALL_FAST_KERNELS:-0}" == "1" ]]; then
  echo "Installing optional Gated DeltaNet kernels (may compile native code)..."
  python -m pip install causal-conv1d flash-linear-attention
fi

python scripts/check_wsl.py --require-wsl --require-cuda
python -m pytest tests_wsl

echo
echo "Setup complete. Start the debugger with:"
echo "  bash scripts/run_wsl.sh"
