# Bimba3d Backend

FastAPI backend for project management, processing pipeline (COLMAP + training), and status/preview endpoints.

## Quick Start
- Install local deps: `pip install -r requirements.local.txt`
- Run dev server: `uvicorn app.main:app --reload --port 8005`

## Dependency Profiles
- `requirements.local.txt`:
  - Use for host/local backend + local worker mode.
  - Includes FastAPI + training/runtime Python deps (`torch`, `torchvision`, `gsplat`, `torchmetrics`, etc.).
- `requirements.docker-worker.txt`:
  - Use for Docker worker Python deps reference.
  - In Docker builds, `torch`/`torchvision` and `gsplat` are installed separately in `Dockerfile.worker` with pinned CUDA-compatible wheels.
- `requirements.txt`:
  - Backward-compatible alias that points to `requirements.local.txt`.

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
- `colmap_only`: Only run COLMAP sparse reconstruction
- `train_only`: Only run training (requires existing sparse outputs)

## Worker Runtime Modes
- Runtime can be selected per request with `worker_mode`: `docker` | `local`.
- If request body does not provide `worker_mode`, backend resolves in this order:
  1) `WORKER_MODE` env (`docker` or `local`)
  2) legacy `USE_DOCKER_WORKER` env (`true` => `docker`, otherwise `local`)
  3) default `docker`
- Current run mode is persisted in project status as `worker_mode`.

### Switching Modes (default port `8005`)
- Docker mode (recommended baseline):
  - `export WORKER_MODE=docker`
  - `export USE_DOCKER_WORKER=true`
  - `uvicorn app.main:app --reload --port 8005`
- Local mode:
  - `export WORKER_MODE=local`
  - `export USE_DOCKER_WORKER=false`
  - `pip install -r requirements.local.txt`
  - `uvicorn app.main:app --reload --port 8005`
- Per-request override (takes precedence over env): set `worker_mode` in `POST /projects/{id}/process` body.

### Port Note
- Frontend expects backend on `8005`.
- Alternate ports (for example `8016`) are only for isolated smoke tests when you want to avoid interrupting an existing server.



## Notes
- Default `max_steps` is 300; previews and splat exports occur at configured intervals.
- Manual stop triggers a final export and sets status to `stopped`.
- GPU detection uses PyTorch; when CUDA is unavailable, training runs on CPU (slower).





./build-worker.sh
uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload