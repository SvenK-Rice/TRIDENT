#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo "TRIDENT NPP Workbench"
echo "=========================================="

if [ ! -x ".venv/bin/python" ] || [ ! -f ".venv/.trident-installed" ]; then
  echo "TRIDENT is not installed yet."
  echo "Run: ./INSTALL_TRIDENT.command"
  exit 1
fi

.venv/bin/python scripts/check_environment.py

exec .venv/bin/python -m streamlit run \
  src/trident/app/workbench.py \
  --server.headless=false \
  --server.fileWatcherType=none
