#!/usr/bin/env bash
set -euo pipefail

# Start backend and frontend together from the repo root.
# Usage:
#   ./run_local.sh
# Optional env vars:
#   BACKEND_DIR (default: bimba3d_backend)
#   FRONTEND_DIR (default: bimba3d_frontend)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${BACKEND_DIR:-bimba3d_backend}"
FRONTEND_DIR="${FRONTEND_DIR:-bimba3d_frontend}"

BACKEND_PATH="$ROOT_DIR/$BACKEND_DIR"
FRONTEND_PATH="$ROOT_DIR/$FRONTEND_DIR"

if [[ ! -x "$BACKEND_PATH/run_local.sh" ]]; then
  echo "Backend launcher not found or not executable: $BACKEND_PATH/run_local.sh" >&2
  exit 1
fi

if [[ ! -x "$FRONTEND_PATH/run_local.sh" ]]; then
  echo "Frontend launcher not found or not executable: $FRONTEND_PATH/run_local.sh" >&2
  exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "[root] Stopping services..."
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait || true
}

trap cleanup INT TERM EXIT

echo "[root] Starting backend from $BACKEND_DIR"
(
  cd "$BACKEND_PATH"
  ./run_local.sh
) &
BACKEND_PID=$!

echo "[root] Starting frontend from $FRONTEND_DIR"
(
  cd "$FRONTEND_PATH"
  ./run_local.sh
) &
FRONTEND_PID=$!

echo "[root] Backend:  http://localhost:8005"
echo "[root] Frontend: http://localhost:5173"

wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_CODE=$?

echo "[root] One service exited (code: $EXIT_CODE). Stopping the other..."
exit "$EXIT_CODE"
