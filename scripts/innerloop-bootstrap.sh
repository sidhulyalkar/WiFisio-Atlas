#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${PHYSIOATLAS_VENV:-.venv-physioatlas}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-physioatlas.txt
python -m pip install -e . --no-deps --no-build-isolation

python -m compileall -q physioatlas
python -m physioatlas list-adapters > /dev/null
python -m physioatlas doctor --repo .
python -m pytest -o addopts='' tests/physioatlas -q
python -m physioatlas smoke-test --output outputs/physioatlas/innerloop-smoke
python -m physioatlas verify-run outputs/physioatlas/innerloop-smoke/run
python -m physioatlas household-smoke-test --output outputs/physioatlas/household-smoke
python -m physioatlas household-fault-test --output outputs/physioatlas/household-fault
python -m physioatlas fault-test --output outputs/physioatlas/fault-test

echo "PhysioAtlas InnerLoop bootstrap passed."
echo "Report: outputs/physioatlas/innerloop-smoke/smoke-report.json"
