# Bimba3d Frontend

React + TypeScript + Vite UI for project creation, upload, processing, logs, comparison, and viewer screens.

## Setup and Run
Canonical install/run instructions are maintained in the root README:
- [../README.md](../README.md)

Use that as the single source of truth to avoid duplicated instructions.

## Frontend-Specific Notes
- Uses Vite for development and build output to `dist/`.
- In monorepo single-URL mode, backend serves built frontend assets from `bimba3d_frontend/dist`.
- In dev mode (`npm run dev`), frontend runs on `5173` and calls backend API on `8005`.

## UX/Feature Areas
- Project list and creation flows.
- Image upload and per-project processing configuration.
- Live status/logs/progress during processing.
- Comparison view for run metrics and previews.
- 3D viewer tab and output download actions.

