#!/usr/bin/env bash
set -euo pipefail

# Run the FastAPI backend locally (with auto-reload)
# - Creates a virtualenv (./venv) if missing
# - Installs requirements
# - Starts uvicorn on port 8005

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"
VENV_DIR="venv"
PORT="8005"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[backend] Creating virtualenv in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[backend] Installing requirements"
pip install --upgrade pip
pip install -r requirements.local.txt

# Environment toggles
export USE_DEMO_MODE=${USE_DEMO_MODE:-false}
export USE_DOCKER_WORKER=${USE_DOCKER_WORKER:-false}
export DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data/projects"}

echo "[backend] Starting uvicorn on :$PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
