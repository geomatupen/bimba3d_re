# Bimba3d Monorepo

This repository contains both services:
- `bimba3d_backend` (FastAPI + worker pipeline)
- `bimba3d_frontend` (React + Vite)

## Prerequisites
- Python 3.12+
- Node.js 18+
- `colmap` installed on host and available on PATH (required for local mode)
- NVIDIA driver + CUDA-capable PyTorch build (optional, for GPU training)

### Important: what is installed in venv vs system
- `pip install -r bimba3d_backend/requirements.local.txt` installs Python packages only.
- COLMAP is **not** a pip package here; install it separately on the OS and make sure `colmap -h` works in your terminal.
- If Windows `colmap.exe` behaves oddly, set `COLMAP_EXE` to `...\\COLMAP.bat` before starting backend.
- On Windows, if `torch.cuda.is_available()` is false, reinstall CUDA-enabled PyTorch wheels explicitly (requirements file alone may install CPU wheels).
- For local `gsplat` training on Windows, CUDA Toolkit (`nvcc`) https://developer.nvidia.com/cuda-downloads is required in addition to NVIDIA drivers and CUDA-enabled PyTorch.

### Windows CUDA-safe install order (torch + gsplat)
If you need local GPU training on Windows, use this order to avoid `gsplat` replacing CUDA torch with an incompatible build:

```powershell
conda deactivate
.\.venv\Scripts\activate

python -m pip uninstall -y torch torchvision torchaudio gsplat
python -m pip cache purge

python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
python -m pip install --force-reinstall ninja
python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=gsplat gsplat==1.5.3 --no-build-isolation -v
python -m pip install -r bimba3d_backend\requirements.windows.txt
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import gsplat.cuda._wrapper as w; w._make_lazy_cuda_obj('CameraModelType'); print('gsplat lazy CUDA ok:', True)"
```

Expected: CUDA-enabled torch (`2.5.1+cu121`, CUDA version shown, `True`, device count >= 1) and `gsplat lazy CUDA ok: True`.

Compatibility profile probe:
- `python .\bimba3d_backend\scripts\resolve_compatibility_profile.py --matrix .\compatibility-matrix.json --format json`

If `nvcc --version` is not recognized, install CUDA Toolkit (12.x) from NVIDIA and ensure CUDA `bin` is on PATH (for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`).

## 1) Install
From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r bimba3d_backend/requirements.local.txt
```

Frontend dependencies:

```bash
cd bimba3d_frontend
npm install
cd ..
```

## 2) Run (Recommended for single URL)
Build frontend once, then serve both UI + API from backend (`uvicorn`):

```bash
cd bimba3d_frontend
npm run build
cd ..

source .venv/bin/activate
uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Open: `http://localhost:8005`

## 3) Frontend Dev Mode (optional)
If you want Vite hot-reload for UI work:

Terminal 1:
```bash
source .venv/bin/activate
uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Terminal 2:
```bash
cd bimba3d_frontend
npm run dev
```

Open: `http://localhost:5173`

## Worker mode
Backend supports both runtime modes for processing:
- `docker`
- `local`

Default mode behavior:
- All platforms: defaults to `local`

Set via env before starting backend:

```bash
export WORKER_MODE=local   # or docker
export USE_DOCKER_WORKER=false   # legacy flag, optional
```

PowerShell equivalent:

```powershell
$env:WORKER_MODE = "local"   # or "docker"
$env:USE_DOCKER_WORKER = "false"
$env:COLMAP_EXE = "D:\\Study\\4. Thesis\\colmap\\COLMAP-3.9.1-windows-cuda\\COLMAP.bat"
```

## PostgreSQL modes (both supported)

You can run database-backed auth/projects in two ways, both using an external/native PostgreSQL server:

1) Docker backend container + external/native PostgreSQL
- `docker-compose.yml` backend uses `DATABASE_URL` and defaults to host DB access:
	- `postgresql+psycopg://postgres:postgres@host.docker.internal:5432/bimba3d`
- By default, DB features remain OFF to keep current behavior:
	- `APP_MODE=desktop`
	- `DB_ENABLED=false`

To enable DB/auth in Compose backend, set:
- `APP_MODE=server`
- `DB_ENABLED=true`
- strong `JWT_SECRET_KEY`

2) Local `.venv` backend + external/native PostgreSQL
- Install/run Postgres on host OS (Ubuntu/Windows).
- Point backend to host DB URL and enable DB mode before starting uvicorn.

Recommended secure credential pattern:
- Set `PGPASSWORD` separately and keep `DATABASE_URL` without password.
- Avoids exposing DB password directly inside connection URL strings.

Copy starter env values from `.env.server.example`.

Linux/macOS example:
```bash
export APP_MODE=server
export DB_ENABLED=true
export DATABASE_URL='postgresql+psycopg://postgres@localhost:5432/bimba3d'
export PGPASSWORD='replace-with-db-password'
export JWT_SECRET_KEY='replace-with-strong-random-secret-at-least-32-chars'
uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

PowerShell example:
```powershell
$env:APP_MODE = "server"
$env:DB_ENABLED = "true"
$env:DATABASE_URL = "postgresql+psycopg://postgres@localhost:5432/bimba3d"
$env:PGPASSWORD = "replace-with-db-password"
$env:JWT_SECRET_KEY = "replace-with-strong-random-secret-at-least-32-chars"
uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Important:
- PostgreSQL server is NOT installed inside `.venv`.
- `.venv` only installs Python DB client packages (`psycopg`, `sqlalchemy`, etc.).
- This repo does not run PostgreSQL in Docker; both modes use external/native PostgreSQL.

Quick startup log check:
- Backend emits DB startup status logs with `db_startup state=`.
- Linux/WSL example: `grep -n "db_startup state=" backend.log`
- PowerShell example: `Select-String -Path backend.log -Pattern "db_startup state="`

Google OAuth (real sign-in):
- Set these env vars in server mode:
	- `GOOGLE_CLIENT_ID`
	- `GOOGLE_CLIENT_SECRET`
	- `GOOGLE_REDIRECT_URI` (example: `http://localhost:8005/auth/google/callback`)
- In Google Cloud Console, add the redirect URI exactly as above.
- Frontend Login modal uses “Continue with Google” and opens Google consent popup.
- Backend callback exchanges code, verifies Google token, and issues your app JWT tokens.

Database migrations (Alembic):
- For fresh-start environments, backend auto-creates schema on startup (`DB_AUTO_CREATE_SCHEMA=true`, default).
- Run migrations only when upgrading an existing DB or applying versioned schema changes.

Linux/WSL:
```bash
cd bimba3d_backend
../.venv/bin/alembic -c alembic.ini upgrade head
```

PowerShell:
```powershell
cd bimba3d_backend
..\.venv\Scripts\alembic.exe -c alembic.ini upgrade head
```

Current initial revision:
- `20260320_000001` (users, refresh_tokens, project_records)

## Public result API contract

Public project endpoints now expose a result page URL for website embedding/routing instead of direct splat URLs.

- `GET /projects/public`
- `GET /projects/public/{project_id}`

Response fields include:
- `project_id`
- `name`
- `description`
- `video_url`
- `category`
- `created_at`
- `visibility` (`public`)
- `thumbnail_url`
- `result_page_url` (example: `/result/<project_id>`)

Viewer routing behavior in frontend:
- Logged-in owner opening own project -> full processing page: `/project/:id`
- Logged-out user or non-owner opening a public project -> result-only page: `/result/:id`

## Windows checklist (local mode)
1. Install Python + Node.js.
2. Create venv and install Python deps from `requirements.windows.txt` for Windows CUDA local mode.
3. Install COLMAP natively on Windows and ensure `colmap -h` works.
4. (Optional GPU) Install NVIDIA driver and CUDA-enabled PyTorch wheel; verify:
	- `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
5. Build frontend and run backend:
	- `cd bimba3d_frontend; npm install; npm run build; cd ..`
	- `uvicorn bimba3d_backend.app.main:app --reload --port 8005`

## Notes
- Python import path uses underscore package names: `bimba3d_backend...`
- Do **not** use hyphen module names in uvicorn import strings.
