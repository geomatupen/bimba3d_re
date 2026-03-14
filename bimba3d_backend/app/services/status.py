import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from bimba3d_backend.app.config import DATA_DIR


def get_status_file(project_id: str) -> Path:
    """Get path to project status file."""
    return DATA_DIR / project_id / "status.json"


def initialize_status(project_id: str, name: Optional[str] = None) -> None:
    """Initialize status file for a new project."""
    status_file = get_status_file(project_id)
    status_data = {
        "project_id": project_id,
        "name": name,
        "status": "pending",
        "progress": 0,
        "error": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "stage": None,
        "stage_progress": None,
        "message": None,
        "engine": None,
        "worker_mode": None,
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write atomically using a temporary file
    temp_file = status_file.with_suffix('.tmp')
    with open(temp_file, "w") as f:
        json.dump(status_data, f)
    temp_file.replace(status_file)


def get_status(project_id: str) -> dict:
    """Get current project status."""
    status_file = get_status_file(project_id)
    if not status_file.exists():
        return {"project_id": project_id, "status": "not_found", "progress": 0}
    
    try:
        with open(status_file, "r") as f:
            content = f.read()
            if not content.strip():
                # File is empty, return pending status
                return {"project_id": project_id, "status": "pending", "progress": 0, "name": None, "created_at": None, "error": None}
            data = json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        # Invalid JSON, return pending status as fallback
        return {"project_id": project_id, "status": "pending", "progress": 0, "name": None, "created_at": None, "error": None}

    # Backfill optional fields for older projects
    data.setdefault("project_id", project_id)
    data.setdefault("name", None)
    data.setdefault("created_at", None)
    data.setdefault("error", None)
    data.setdefault("progress", 0)
    data.setdefault("stage", None)
    data.setdefault("stage_progress", None)
    data.setdefault("engine", None)
    data.setdefault("worker_mode", None)
    return data


def update_project_name(project_id: str, name: Optional[str]) -> None:
    """Update only the project name in status metadata."""
    status_file = get_status_file(project_id)
    status_file.parent.mkdir(parents=True, exist_ok=True)

    current = get_status(project_id)
    current["name"] = name

    temp_file = status_file.with_suffix('.tmp')
    with open(temp_file, "w") as f:
        json.dump(current, f)
    temp_file.replace(status_file)


def update_status(
    project_id: str,
    status: str,
    progress: Optional[int] = None,
    error: Optional[str] = None,
    mode: Optional[str] = None,
    tuning_active: Optional[bool] = None,
    currentStep: Optional[int] = None,
    maxSteps: Optional[int] = None,
    last_tuning: Optional[dict] = None,
    stop_requested: Optional[bool] = None,
    stage: Optional[str] = None,
    stage_progress: Optional[int] = None,
    message: Optional[str] = None,
    device: Optional[str] = None,
    engine: Optional[str] = None,
    worker_mode: Optional[str] = None,
) -> None:
    """Update project status while preserving metadata fields."""
    status_file = get_status_file(project_id)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    current = get_status(project_id)
    current["status"] = status
    if progress is not None:
        current["progress"] = progress
    if error is not None:
        current["error"] = error
    if mode is not None:
        current["mode"] = mode
    if tuning_active is not None:
        current["tuning_active"] = tuning_active
    if currentStep is not None:
        current["currentStep"] = currentStep
    if maxSteps is not None:
        current["maxSteps"] = maxSteps
    if last_tuning is not None:
        current["last_tuning"] = last_tuning
    if stop_requested is not None:
        current["stop_requested"] = stop_requested
    if stage is not None:
        current["stage"] = stage
    if stage_progress is not None:
        current["stage_progress"] = stage_progress
    if message is not None:
        current["message"] = message
    if device is not None:
        current["device"] = device
    if engine is not None:
        current["engine"] = engine
    if worker_mode is not None:
        current["worker_mode"] = worker_mode
    
    # Write atomically using a temporary file
    temp_file = status_file.with_suffix('.tmp')
    with open(temp_file, "w") as f:
        json.dump(current, f)
    temp_file.replace(status_file)


def clear_stop_state(project_id: str) -> None:
    """Remove stale stop markers so a new run starts from a clean state."""
    status_file = get_status_file(project_id)
    status_file.parent.mkdir(parents=True, exist_ok=True)

    current = get_status(project_id)
    current["stop_requested"] = False
    current.pop("stopped_stage", None)
    current.pop("stopped_step", None)
    current.pop("stopped_percentage", None)

    # Remove stale stop message only if it came from a user stop action.
    message = current.get("message")
    if isinstance(message, str) and "stopped by user" in message.lower():
        current["message"] = None

    temp_file = status_file.with_suffix('.tmp')
    with open(temp_file, "w") as f:
        json.dump(current, f)
    temp_file.replace(status_file)
