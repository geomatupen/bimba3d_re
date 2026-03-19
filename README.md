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
python -m pip install --no-deps gsplat==1.5.3
```

Verify:

```powershell
python -c "import torch, gsplat.cuda._wrapper as w; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), getattr(w,'_C',None) is not None)"
```

Expected: `2.5.1+cu121 12.1 True True` (or equivalent CUDA-enabled values).

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

## Windows checklist (local mode)
1. Install Python + Node.js.
2. Create venv and install Python deps from `requirements.local.txt`.
3. Install COLMAP natively on Windows and ensure `colmap -h` works.
4. (Optional GPU) Install NVIDIA driver and CUDA-enabled PyTorch wheel; verify:
	- `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
5. Build frontend and run backend:
	- `cd bimba3d_frontend; npm install; npm run build; cd ..`
	- `uvicorn bimba3d_backend.app.main:app --reload --port 8005`

## Notes
- Python import path uses underscore package names: `bimba3d_backend...`
- Do **not** use hyphen module names in uvicorn import strings.
