#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo "TRIDENT installer"
echo "=========================================="

choose_python() {
  for candidate in \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
    /opt/homebrew/bin/python3.13 \
    /usr/local/bin/python3.13 \
    python3.13 \
    python3
  do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)
PY
      if [ $? -eq 0 ]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [ -z "${PYTHON_BIN:-}" ]; then
  echo "ERROR: Python 3.13 was not found."
  echo "Install Python 3.13 from python.org, then rerun this installer."
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Removing incomplete environments..."
rm -rf .venv .venv-hdf

echo "Creating application environment..."
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .

echo "Creating isolated HDF4 environment..."
"$PYTHON_BIN" -m venv .venv-hdf
.venv-hdf/bin/python -m pip install --upgrade pip setuptools wheel
.venv-hdf/bin/python -m pip install -r requirements-hdf.lock

echo "Checking required imports..."
.venv/bin/python scripts/check_environment.py

echo "Checking isolated HDF4 imports..."
.venv-hdf/bin/python - <<'PY'
import numpy
from pyhdf.SD import SD, SDC
print("HDF4 environment OK")
PY

mkdir -p logs
date -u +"%Y-%m-%dT%H:%M:%SZ" > .venv/.trident-installed
date -u +"%Y-%m-%dT%H:%M:%SZ" > .venv-hdf/.trident-installed

echo
echo "=========================================="
echo "TRIDENT installation completed successfully"
echo "=========================================="
echo "Launch with: ./RUN_TRIDENT.command"
