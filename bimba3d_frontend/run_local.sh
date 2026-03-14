#!/usr/bin/env bash
set -euo pipefail

# Run the Vite + React frontend locally
# - Installs node modules via npm ci (or npm install)
# - Starts dev server on port 5173

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v npm >/dev/null 2>&1; then
  :
else
  echo "npm is not installed. Please install Node.js >= 18 and npm." >&2
  exit 1
fi

# Install dependencies (prefer clean install when lockfile exists)
if [[ -f package-lock.json ]]; then
  echo "[frontend] Installing dependencies (npm ci)"
  npm ci
else
  echo "[frontend] Installing dependencies (npm install)"
  npm install
fi

echo "[frontend] Starting dev server on :5173"
exec npm run dev
