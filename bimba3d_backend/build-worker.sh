#!/bin/bash
set -e

echo "🚀 Building Gaussian Splatting Worker Docker Image..."
echo ""

# Build the worker image from within backend directory
docker build -f Dockerfile.worker -t bimba3d-worker:latest .

echo ""
echo "✅ Build complete!"
echo ""
echo "To test the worker, run:"
echo "  docker run --rm bimba3d-worker:latest --help"
echo ""
echo "To process a project:"
echo "  docker run --rm --gpus all -v ./data:/data bimba3d-worker:latest <project_id> --params '{\"max_steps\": 300 }'"
