#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run: bash scripts/setup_wsl.sh" >&2
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export JLENS_MODEL_ID="${JLENS_MODEL_ID:-Qwen/Qwen3.5-0.8B-Base}"
export JLENS_DEVICE="${JLENS_DEVICE:-cuda}"
export JLENS_DTYPE="${JLENS_DTYPE:-fp16}"
export JLENS_PATH="${JLENS_PATH:-none}"
export JLENS_EAGER_LOAD="${JLENS_EAGER_LOAD:-1}"
exec python -m jlens_wsl.cli serve --host "${JLENS_HOST:-0.0.0.0}" --port "${JLENS_PORT:-8765}" "$@"
