#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
EXTRAS="${EXTRAS:-test dev}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[info] repo root: ${ROOT_DIR}"
echo "[info] python: $("${PYTHON_BIN}" --version 2>&1)"

if command -v uv >/dev/null 2>&1; then
  echo "[info] using uv-managed environment"
  uv venv "${VENV_DIR}"
  uv sync --locked $(for extra in ${EXTRAS}; do printf -- '--extra %s ' "${extra}"; done)
  RUNNER=(uv run)
else
  echo "[info] uv not found; using stdlib venv + editable install"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip uninstall -y lerobot >/dev/null 2>&1 || true
  python -m pip install -e ".[dataset,training,test,dev]"
  RUNNER=(python)
fi

echo "[info] verifying imported lerobot path and version"
"${RUNNER[@]}" - <<'PY'
from pathlib import Path
import lerobot

print(f"lerobot.__file__={Path(lerobot.__file__).resolve()}")
print(f"lerobot.__version__={lerobot.__version__}")
PY

echo "[info] setup complete"
