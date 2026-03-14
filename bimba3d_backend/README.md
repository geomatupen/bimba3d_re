# Bimba3d Backend

FastAPI backend for project management, processing pipeline (COLMAP + training), and status/preview endpoints.

## Setup and Run
Canonical install/run instructions are maintained in the root README:
- [../README.md](../README.md)

Use that as the single source of truth to avoid duplicated instructions.

## Dependency Profiles
- `requirements.local.txt`
  - Host/local backend + local worker mode.
  - Includes FastAPI + training/runtime Python deps (`torch`, `torchvision`, `gsplat`, `torchmetrics`, etc.).
- `requirements.docker-worker.txt`
  - Docker worker Python deps reference.
  - In Docker builds, `torch`/`torchvision` and `gsplat` are installed separately in `Dockerfile.worker` with pinned CUDA-compatible wheels.
- `requirements.txt`
  - Backward-compatible alias pointing to `requirements.local.txt`.

## Key Endpoints
- `POST /projects` — create a project
- `POST /projects/{id}/images` — upload images
- `POST /projects/{id}/process` — start pipeline
  - Body params include:
    - `stage`: `full` | `colmap_only` | `train_only`
    - `max_steps`, `batch_size`
    - `splat_export_interval`, `png_export_interval`, `auto_early_stop`
- `POST /projects/{id}/stop` — request manual stop
- `GET /projects/{id}/status` — status with `stage`, `message`, `device`
- `GET /projects/{id}/preview` — latest preview PNG
- `GET /health/gpu` — GPU availability and device info

## Pipeline Stages
- `full`: COLMAP sparse + training
- `colmap_only`: only COLMAP sparse reconstruction
- `train_only`: only training (requires existing sparse outputs)

## Worker Runtime Modes
- `worker_mode` supports `docker` and `local`.
- If request body omits `worker_mode`, backend resolves in order:
  1) `WORKER_MODE` env (`docker` or `local`)
  2) legacy `USE_DOCKER_WORKER` env (`true` => `docker`, otherwise `local`)
  3) default `docker`
- Resolved mode is persisted in project status as `worker_mode`.

## Notes
- Default `max_steps` is 300; previews and splat exports occur at configured intervals.
- Manual stop triggers a final export and sets status to `stopped`.
- If CUDA is unavailable, training runs on CPU (slower).