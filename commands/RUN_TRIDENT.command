#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

APP_VERSION="3.7.2"
MAIN_ENV=".venv"
HDF_ENV=".venv-hdf"
INSTALL_LOG="logs/install.log"
mkdir -p logs

echo "=========================================="
echo "TRIDENT NPP Workbench v${APP_VERSION}"
echo "=========================================="

version_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if major == 3 and minor in (11, 12, 13) else 1)
PY
}

find_python() {
  local candidates=(
    "${TRIDENT_PYTHON:-}"
    python3.12
    /opt/homebrew/bin/python3.12
    /usr/local/bin/python3.12
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    python3.13
    /opt/homebrew/bin/python3.13
    /usr/local/bin/python3.13
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
    python3.11
    /opt/homebrew/bin/python3.11
    /usr/local/bin/python3.11
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    python3
  )

  local p full
  for p in "${candidates[@]}"; do
    [ -n "$p" ] || continue
    if command -v "$p" >/dev/null 2>&1; then
      full="$(command -v "$p")"
    elif [ -x "$p" ]; then
      full="$p"
    else
      continue
    fi
    if version_supported "$full"; then
      echo "$full"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(find_python)"; then
  cat <<'MSG'
TRIDENT could not find Python 3.11, 3.12, or 3.13.
Install Python from python.org and run RUN_TRIDENT.command again.
TRIDENT will not modify Homebrew or install Python automatically.
MSG
  exit 1
fi

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

file_hash() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
  fi
}

MAIN_HASH="$(cat requirements.lock pyproject.toml | shasum -a 256 | awk '{print $1}')"
HDF_HASH="$(file_hash requirements-hdf.lock)"

main_env_ready() {
  [ -x "$MAIN_ENV/bin/python" ] || return 1
  [ -f "$MAIN_ENV/.trident_lock_hash" ] || return 1
  [ "$(cat "$MAIN_ENV/.trident_lock_hash")" = "$MAIN_HASH" ] || return 1
  "$MAIN_ENV/bin/python" - <<'PY' >/dev/null 2>&1
import numpy, pandas, xarray, netCDF4, h5netcdf, streamlit, plotly
import earthaccess, copernicusmarine, dask
import trident
PY
}

hdf_env_ready() {
  [ -x "$HDF_ENV/bin/python" ] || return 1
  [ -f "$HDF_ENV/.trident_lock_hash" ] || return 1
  [ "$(cat "$HDF_ENV/.trident_lock_hash")" = "$HDF_HASH" ] || return 1
  "$HDF_ENV/bin/python" - <<'PY' >/dev/null 2>&1
from pyhdf.SD import SD, SDC
PY
}

install_main_env() {
  echo "Installing the pinned TRIDENT application environment (first launch or version change)..."
  rm -rf "$MAIN_ENV"
  "$PYTHON_BIN" -m venv "$MAIN_ENV"
  {
    "$MAIN_ENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    "$MAIN_ENV/bin/python" -m pip install --disable-pip-version-check -r requirements.lock
    "$MAIN_ENV/bin/python" -m pip install --disable-pip-version-check --no-deps -e .
  } >>"$INSTALL_LOG" 2>&1 || {
    echo "Installation failed. See: $PWD/$INSTALL_LOG"
    tail -80 "$INSTALL_LOG" || true
    exit 1
  }
  printf '%s' "$MAIN_HASH" > "$MAIN_ENV/.trident_lock_hash"
}

install_hdf_env() {
  echo "Installing the isolated HDF4 reader environment..."
  rm -rf "$HDF_ENV"
  "$PYTHON_BIN" -m venv "$HDF_ENV"
  {
    "$HDF_ENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    "$HDF_ENV/bin/python" -m pip install --disable-pip-version-check -r requirements-hdf.lock
  } >>"$INSTALL_LOG" 2>&1 || {
    echo "HDF4 installation failed. See: $PWD/$INSTALL_LOG"
    tail -80 "$INSTALL_LOG" || true
    exit 1
  }
  printf '%s' "$HDF_HASH" > "$HDF_ENV/.trident_lock_hash"
}

if main_env_ready; then
  echo "Application environment already installed — skipping pip."
else
  install_main_env
fi

if hdf_env_ready; then
  echo "HDF4 environment already installed — skipping pip."
else
  install_hdf_env
fi

export TRIDENT_HDF_PYTHON="$PWD/$HDF_ENV/bin/python"

echo "Running environment checks..."
"$MAIN_ENV/bin/python" scripts/check_environment.py
"$HDF_ENV/bin/python" - <<'PY'
from pyhdf.SD import SD, SDC
print("Isolated HDF4 environment OK")
PY

echo "Launching TRIDENT..."
exec "$MAIN_ENV/bin/python" -m streamlit run src/trident/app/workbench.py \
  --server.headless false \
  --server.fileWatcherType none
