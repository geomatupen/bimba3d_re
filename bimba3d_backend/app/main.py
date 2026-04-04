import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from bimba3d_backend.app.api.projects import router as projects_router
from bimba3d_backend.app.config import ALLOWED_ORIGINS
from bimba3d_backend.app.config import DATA_DIR
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Gaussian Splat Backend")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router, prefix="/projects")


@app.on_event("startup")
def mark_interrupted_projects():
    """On backend start, mark any projects that were 'processing' as stopped/resumable.

    This ensures the frontend doesn't continue to show 'processing' for jobs
    that were interrupted by a backend restart or crash.
    """
    note = "Backend restarted — processing interrupted. Please resume when ready."
    from bimba3d_backend.app.services.colmap import stop_project_worker_containers
    for proj_dir in DATA_DIR.iterdir():
        try:
            if not proj_dir.is_dir():
                continue
            stopped = stop_project_worker_containers(proj_dir.name)
            if stopped:
                logging.info("Stopped %d stale worker container(s) for %s", stopped, proj_dir.name)
            status_file = proj_dir / "status.json"
            if not status_file.exists():
                continue
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if data.get("status") == "processing":
                data["status"] = "stopped"
                data["progress"] = data.get("progress", 0)
                data["error"] = note
                data["stop_requested"] = True
                data["stopped_stage"] = data.get("stage", "unknown")
                data["resumable"] = True
                data["percentage"] = data.get("percentage", 0.0)
                # write atomically
                tmp = status_file.with_suffix('.tmp')
                with open(tmp, 'w') as f:
                    json.dump(data, f)
                tmp.replace(status_file)
        except Exception:
            logging.exception(f"Failed to mark interrupted project: {proj_dir}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/gpu")
def gpu_health():
    """Report GPU availability and basic CUDA/device info."""
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        devices = []
        for i in range(count):
            try:
                devices.append(torch.cuda.get_device_name(i))
            except Exception:
                devices.append(f"cuda:{i}")
        return {
            "gpu_available": available,
            "device_count": count,
            "devices": devices,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {
            "gpu_available": False,
            "device_count": 0,
            "devices": [],
            "cuda_version": None,
        }


@app.on_event("shutdown")
def signal_all_workers_to_stop():
    """Optionally signal workers to stop on backend shutdown.

    Disabled by default to avoid false stop requests during development reloads.
    Set BIMBA3D_SIGNAL_STOP_ON_SHUTDOWN=1 to re-enable legacy behavior.
    """
    should_signal = os.getenv("BIMBA3D_SIGNAL_STOP_ON_SHUTDOWN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not should_signal:
        logging.info("Backend shutdown: skip stop signaling (BIMBA3D_SIGNAL_STOP_ON_SHUTDOWN is off).")
        return

    logging.info("Backend shutdown: signaling all workers to stop.")
    for proj_dir in DATA_DIR.iterdir():
        try:
            if not proj_dir.is_dir():
                continue
            stop_flag = proj_dir / "stop_requested"
            stop_flag.write_text("stop requested by backend shutdown")
            logging.info(f"Created stop_requested for {proj_dir}")
        except Exception as e:
            logging.warning(f"Failed to create stop_requested for {proj_dir}: {e}")


DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "bimba3d_frontend" / "dist"
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(DEFAULT_FRONTEND_DIST))).resolve()
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logging.info("Serving frontend build from %s", FRONTEND_DIST)
else:
    logging.info("Frontend dist not found at %s; API-only mode", FRONTEND_DIST)
