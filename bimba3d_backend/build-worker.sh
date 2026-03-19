#!/bin/bash
set -e

echo "🚀 Building Gaussian Splatting Worker Docker Image..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build the worker image using repo root context (shared compatibility matrix)
docker build -f "$REPO_ROOT/bimba3d_backend/Dockerfile.worker" -t bimba3d-worker:latest "$REPO_ROOT"

echo ""
echo "✅ Build complete!"
echo ""
echo "To test the worker, run:"
echo "  docker run --rm bimba3d-worker:latest --help"
echo ""
echo "To process a project:"
echo "  docker run --rm --gpus all -v ./data:/data bimba3d-worker:latest <project_id> --params '{\"max_steps\": 300 }'"
