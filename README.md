# Bimba3d Monorepo

This repository contains both services:
- `bimba3d_backend` (FastAPI + worker pipeline)
- `bimba3d_frontend` (React + Vite)

## Prerequisites
- Python 3.12+
- Node.js 18+
- `colmap` installed on host (for local mode)
- NVIDIA/CUDA stack (optional, for GPU training)

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

Set via env before starting backend:

```bash
export WORKER_MODE=local   # or docker
export USE_DOCKER_WORKER=false   # legacy flag, optional
```

## Notes
- Python import path uses underscore package names: `bimba3d_backend...`
- Do **not** use hyphen module names in uvicorn import strings.
