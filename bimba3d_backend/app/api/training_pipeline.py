"""Training pipeline API endpoints for automated cross-project training."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services import training_pipeline_orchestrator
from bimba3d_backend.app.services import model_registry
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Request/Response Models ==========

class ScanDirectoryRequest(BaseModel):
    base_directory: str = Field(..., description="Path to directory containing dataset folders")


class DatasetInfo(BaseModel):
    name: str
    path: str
    image_count: int
    size_mb: float
    has_images: bool


class ScanDirectoryResponse(BaseModel):
    datasets: list[DatasetInfo]
    total: int


class ThermalConfig(BaseModel):
    enabled: bool = True
    strategy: str = "fixed_interval"  # or "temperature_based", "time_scheduled"
    cooldown_minutes: int = 10
    gpu_temp_threshold: int = 70
    check_interval_seconds: int = 30
    max_wait_minutes: int = 30


class PhaseConfig(BaseModel):
    phase_number: int
    name: str
    runs_per_project: int = 1
    passes: int = 1
    strategy_override: Optional[str] = None  # Override ai_selector_strategy for this phase
    preset_override: Optional[str] = None  # For baseline phase
    update_model: bool = True
    context_jitter: bool = False
    context_jitter_mode: str = "uniform"  # "uniform" (sample bounds), "mild" (±10%), "gaussian" (±15%)
    shuffle_order: bool = True
    session_execution_mode: str = "train"


class ProjectConfig(BaseModel):
    project_id: Optional[str] = None
    name: str
    dataset_path: str
    baseline_run_id: Optional[str] = None
    image_count: int
    created: bool = False
    colmap_source_project_id: Optional[str] = None


class CreatePipelineRequest(BaseModel):
    name: str
    base_directory: str
    pipeline_directory: Optional[str] = None  # Where to create pipeline folder (default: DATA_DIR)
    projects: list[ProjectConfig]
    shared_config: dict[str, Any]  # Training parameters shared across all runs
    phases: list[PhaseConfig]
    thermal_management: ThermalConfig
    failure_handling: dict[str, Any] = {
        "continue_on_failure": True,
        "max_retries_per_run": 1,
        "skip_project_after_failures": 3
    }


class PipelineStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    current_phase: int
    current_pass: int
    current_project_index: int
    total_runs: int
    completed_runs: int
    failed_runs: int
    mean_reward: Optional[float]
    success_rate: Optional[float]
    best_reward: Optional[float]
    last_error: Optional[str]
    cooldown_active: bool
    next_run_scheduled_at: Optional[str]
    config: dict[str, Any]
    runs: list[dict[str, Any]]


class BatchCreateProjectsRequest(BaseModel):
    datasets: list[DatasetInfo]
    shared_config: dict[str, Any]


class BatchCreateProjectsResponse(BaseModel):
    created: list[ProjectConfig]
    existing: list[ProjectConfig]
    failed: list[dict[str, str]]


# ========== Utility Functions ==========

def _scan_dataset_folder(folder_path: Path) -> Optional[DatasetInfo]:
    """Scan a folder to extract dataset information."""
    if not folder_path.is_dir():
        return None

    # Look for images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = [f for f in folder_path.glob("*") if f.suffix.lower() in image_exts]

    if not image_files:
        return None

    # Calculate size
    total_size = sum(f.stat().st_size for f in image_files)
    size_mb = total_size / (1024 * 1024)

    return DatasetInfo(
        name=folder_path.name,
        path=str(folder_path.absolute()),
        image_count=len(image_files),
        size_mb=round(size_mb, 2),
        has_images=True
    )


# ========== API Endpoints ==========

@router.post("/scan-directory", response_model=ScanDirectoryResponse)
async def scan_directory(request: ScanDirectoryRequest):
    """Scan a directory for dataset folders."""
    base_path = Path(request.base_directory)

    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")

    if not base_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    datasets = []
    for item in base_path.iterdir():
        if item.is_dir():
            dataset_info = _scan_dataset_folder(item)
            if dataset_info:
                datasets.append(dataset_info)

    # Sort by name
    datasets.sort(key=lambda d: d.name)

    return ScanDirectoryResponse(
        datasets=datasets,
        total=len(datasets)
    )


@router.post("/batch-create-projects", response_model=BatchCreateProjectsResponse)
async def batch_create_projects(request: BatchCreateProjectsRequest):
    """Create projects for multiple datasets in batch."""
    created = []
    existing = []
    failed = []

    for dataset in request.datasets:
        try:
            # Check if project already exists
            project_name = dataset.name
            project_dir = DATA_DIR / project_name

            if project_dir.exists():
                # Load existing project
                config_path = project_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        existing.append(ProjectConfig(
                            project_id=config.get("id"),
                            name=project_name,
                            dataset_path=dataset.path,
                            image_count=dataset.image_count,
                            created=False
                        ))
                continue

            # Create new project
            project_id = str(uuid.uuid4())
            project_dir.mkdir(parents=True, exist_ok=True)

            # Create config
            config = {
                "id": project_id,
                "name": project_name,
                "source_dir": dataset.path,
                "created_at": datetime.utcnow().isoformat() + "Z",
                **request.shared_config
            }

            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            created.append(ProjectConfig(
                project_id=project_id,
                name=project_name,
                dataset_path=dataset.path,
                image_count=dataset.image_count,
                created=True
            ))

        except Exception as e:
            logger.error(f"Failed to create project for {dataset.name}: {e}")
            failed.append({
                "dataset_name": dataset.name,
                "error": str(e)
            })

    return BatchCreateProjectsResponse(
        created=created,
        existing=existing,
        failed=failed
    )


@router.post("/create", response_model=PipelineStatusResponse)
async def create_pipeline(request: CreatePipelineRequest):
    """Create a new training pipeline."""
    try:
        config = request.dict()
        pipeline = training_pipeline_storage.create_pipeline(config)

        return PipelineStatusResponse(**pipeline)

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/start")
async def start_pipeline(pipeline_id: str):
    """Start pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] in ["running"]:
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    # Update status
    updates = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "current_phase": 1,
        "current_pass": 1,
        "current_project_index": 0,
    }

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, updates)

    # Start orchestrator in background thread
    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)

    return {"status": "running", "message": "Pipeline started"}


@router.post("/{pipeline_id}/pause")
async def pause_pipeline(pipeline_id: str):
    """Pause pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {"status": "paused"})

    # Signal orchestrator to pause
    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)

    return {"status": "paused", "message": "Pipeline paused"}


@router.post("/{pipeline_id}/resume")
async def resume_pipeline(pipeline_id: str):
    """Resume paused pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] != "paused":
        raise HTTPException(status_code=400, detail="Pipeline is not paused")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {"status": "running"})

    # Resume orchestrator
    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)

    return {"status": "running", "message": "Pipeline resumed"}


@router.post("/{pipeline_id}/stop")
async def stop_pipeline(pipeline_id: str):
    """Stop pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {
        "status": "stopped",
        "completed_at": datetime.utcnow().isoformat() + "Z"
    })

    # Signal orchestrator to stop
    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)

    return {"status": "stopped", "message": "Pipeline stopped"}


@router.get("/list")
async def list_pipelines(limit: int = 50):
    """List all pipelines."""
    pipelines = training_pipeline_storage.list_pipelines(limit=limit)

    # Convert pipeline dicts to response models
    pipeline_responses = []
    for p in pipelines:
        try:
            pipeline_responses.append(PipelineStatusResponse(**p))
        except Exception as e:
            logger.warning(f"Failed to serialize pipeline {p.get('id')}: {e}")
            continue

    return {"pipelines": pipeline_responses}


@router.get("/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline(pipeline_id: str):
    """Get pipeline status and details."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return PipelineStatusResponse(**pipeline)


@router.get("/{pipeline_id}/runs")
async def get_pipeline_runs(pipeline_id: str):
    """Get all runs for a pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {"runs": pipeline.get("runs", [])}


@router.get("/{pipeline_id}/learning-table")
async def get_pipeline_learning_table(pipeline_id: str):
    """Get aggregated AI learning table data from all projects in pipeline.

    This endpoint collects learning data from all projects that belong to this pipeline
    and returns a unified table showing:
    - All training runs across all projects
    - Quality metrics (PSNR, SSIM, LPIPS, Loss)
    - Learning scores (S_run, S_base, reward)
    - Parameter selections and learned inputs
    """
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    config = pipeline.get("config", {})
    pipeline_folder = Path(config.get("pipeline_folder"))

    if not pipeline_folder or not pipeline_folder.exists():
        raise HTTPException(status_code=404, detail="Pipeline folder not found")

    # Collect learning data from all project directories in pipeline folder
    learning_rows = []

    # Iterate through all project directories
    for project_dir in pipeline_folder.iterdir():
        if not project_dir.is_dir():
            continue

        # Skip non-project directories
        if project_dir.name in ["shared_models", "training_pipelines"]:
            continue

        # Skip if pipeline.json marker (this is the pipeline folder itself)
        if (project_dir / "pipeline.json").exists() and project_dir == pipeline_folder:
            continue

        # Check if it's a valid project directory with runs
        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue

        # Fetch learning data for this project
        try:
            # Use the same endpoint logic as /projects/{id}/ai_learning_table
            from bimba3d_backend.app.api.projects import _read_json_if_exists, _extract_eval_rows, _extract_eval_summary

            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                # Read run analytics (always available)
                analytics_file = run_dir / "analytics" / "run_analytics_v1.json"
                if not analytics_file.exists():
                    continue

                analytics_data = _read_json_if_exists(analytics_file)
                if not analytics_data:
                    continue

                # Extract summary data
                summary = analytics_data.get("summary", {})
                ai_insights = analytics_data.get("ai", {}).get("input_mode_insights", {})

                # Read learning results if available (only for non-baseline runs)
                learning_file = run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"
                learning_data = _read_json_if_exists(learning_file) if learning_file.exists() else {}

                # Determine if this is a baseline run (mode == "baseline")
                is_baseline = summary.get("mode") == "baseline"

                # Get AI selector strategy from project config
                project_config_file = project_dir / "config.json"
                ai_selector_strategy = None
                if project_config_file.exists():
                    try:
                        with open(project_config_file, "r") as f:
                            project_config = json.load(f)
                            ai_selector_strategy = project_config.get("ai_selector_strategy")
                    except Exception:
                        pass

                # Extract baseline_comparison sub-dict for score fields
                baseline_cmp = learning_data.get("baseline_comparison") or {}
                # Also check nested transition.baseline_comparison
                if not baseline_cmp:
                    baseline_cmp = (learning_data.get("transition") or {}).get("baseline_comparison") or {}

                # Extract eval summary for quality metrics (PSNR, SSIM, LPIPS, Loss)
                stats_dir = run_dir / "outputs" / "engines" / "gsplat" / "stats"
                eval_summary = _extract_eval_summary(stats_dir) if stats_dir.exists() else {}

                # Build row data structure
                row = {
                    "project_name": project_dir.name,
                    "run_id": run_dir.name,
                    "run_name": summary.get("run_name") or run_dir.name,
                    "ai_input_mode": ai_insights.get("ai_input_mode") or learning_data.get("mode"),
                    "ai_selector_strategy": ai_selector_strategy,
                    "baseline_run_id": learning_data.get("baseline_run_id") or ai_insights.get("baseline_session_id"),
                    "selected_preset": ai_insights.get("selected_preset") or learning_data.get("selected_preset"),
                    "phase": learning_data.get("phase"),
                    "is_baseline_row": is_baseline,
                    "is_warmup": learning_data.get("is_warmup", False),
                    # Quality metrics from eval stats
                    "best_loss": eval_summary.get("best_loss"),
                    "best_loss_step": eval_summary.get("best_loss_step"),
                    "final_loss": eval_summary.get("final_loss") or summary.get("metrics", {}).get("final_loss"),
                    "final_loss_step": eval_summary.get("final_loss_step") or summary.get("major_params", {}).get("total_steps_completed"),
                    "best_psnr": eval_summary.get("best_psnr"),
                    "best_psnr_step": eval_summary.get("best_psnr_step"),
                    "final_psnr": eval_summary.get("final_psnr"),
                    "final_psnr_step": eval_summary.get("final_psnr_step"),
                    "best_ssim": eval_summary.get("best_ssim"),
                    "best_ssim_step": eval_summary.get("best_ssim_step"),
                    "final_ssim": eval_summary.get("final_ssim"),
                    "final_ssim_step": eval_summary.get("final_ssim_step"),
                    "best_lpips": eval_summary.get("best_lpips"),
                    "best_lpips_step": eval_summary.get("best_lpips_step"),
                    "final_lpips": eval_summary.get("final_lpips"),
                    "final_lpips_step": eval_summary.get("final_lpips_step"),
                    # Learning scores from input_mode_learning_results.json
                    "t_best": learning_data.get("t_best"),
                    "t_eval_best": learning_data.get("t_eval_best"),
                    "t_end": learning_data.get("t_end"),
                    "s_best": learning_data.get("s_best"),
                    "s_end": learning_data.get("s_end"),
                    "s_run": learning_data.get("s_run"),
                    # s_base_best/s_base_end come from baseline_comparison sub-dict
                    "s_base_best": baseline_cmp.get("s_base_best") or baseline_cmp.get("s_run_best"),
                    "s_base_end": baseline_cmp.get("s_base_end") or baseline_cmp.get("s_run_end"),
                    "s_base": baseline_cmp.get("s_base") or learning_data.get("s_base"),
                    "reward": learning_data.get("reward") or learning_data.get("reward_signal") or ai_insights.get("reward"),
                    # Score components: run_best/run_end (l=loss, q=quality, t=time, s=score)
                    # These come from baseline_comparison in the learning results
                    "run_best_l": baseline_cmp.get("run_best_l"),
                    "run_best_q": baseline_cmp.get("run_best_q"),
                    "run_best_t": baseline_cmp.get("run_best_t"),
                    "run_best_s": baseline_cmp.get("s_run_best") or baseline_cmp.get("run_best_s"),
                    "run_end_l": baseline_cmp.get("run_end_l"),
                    "run_end_q": baseline_cmp.get("run_end_q"),
                    "run_end_t": baseline_cmp.get("run_end_t"),
                    "run_end_s": baseline_cmp.get("s_run_end") or baseline_cmp.get("run_end_s"),
                    # Additional info
                    "remarks": learning_data.get("remarks"),
                    "learned_input_params": learning_data.get("learned_input_params") or learning_data.get("yhat_scores"),
                    "learned_input_params_source": learning_data.get("learned_input_params_source"),
                    "learned_input_params_status": learning_data.get("learned_input_params_status"),
                }

                learning_rows.append(row)

        except Exception as e:
            logger.warning(f"Failed to load learning data from {project_dir.name}: {e}")
            continue

    # Sort by project name, then by run timestamp
    learning_rows.sort(key=lambda r: (r.get("project_name", ""), r.get("run_id", "")))

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline["name"],
        "rows": learning_rows,
        "total_rows": len(learning_rows)
    }


@router.get("/{pipeline_id}/worker-logs")
async def get_pipeline_worker_logs(pipeline_id: str):
    """Get worker processing logs from all projects in pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    config = pipeline.get("config", {})
    pipeline_folder = Path(config.get("pipeline_folder"))

    if not pipeline_folder or not pipeline_folder.exists():
        raise HTTPException(status_code=404, detail="Pipeline folder not found")

    logs = []

    # Collect logs from each project
    for project_dir in pipeline_folder.iterdir():
        if not project_dir.is_dir():
            continue

        # Skip non-project directories
        if project_dir.name in ["shared_models", "training_pipelines"]:
            continue

        # Skip pipeline folder marker
        if (project_dir / "pipeline.json").exists() and project_dir == pipeline_folder:
            continue

        project_name = project_dir.name

        # Read processing.log directly from pipeline project folder
        log_file = project_dir / "processing.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()
                    # Get last 1000 lines to avoid huge payloads
                    log_lines = log_content.splitlines()
                    if len(log_lines) > 1000:
                        log_content = "\n".join(log_lines[-1000:])

                    if log_lines:  # Only add if there's content
                        logs.append({
                            "project": project_name,
                            "logs": log_content,
                            "lines": len(log_lines)
                        })
            except Exception as e:
                logger.warning(f"Failed to read processing.log for {project_name}: {e}")

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline["name"],
        "logs": logs,
        "total_projects": len(logs)
    }


@router.get("/{pipeline_id}/models")
async def get_pipeline_models(pipeline_id: str):
    """Get shared models from pipeline's shared_models directory."""
    try:
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        config = pipeline.get("config", {})
        pipeline_folder = Path(config.get("pipeline_folder"))

        if not pipeline_folder or not pipeline_folder.exists():
            raise HTTPException(status_code=404, detail="Pipeline folder not found")

        shared_models_dir = pipeline_folder / "shared_models"
        if not shared_models_dir.exists():
            return {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline["name"],
                "models": [],
                "total": 0,
                "message": "No shared models directory found yet. Models will be created after training runs complete."
            }

        models = []

        # Check for contextual_continuous_selector models
        selector_dir = shared_models_dir / "contextual_continuous_selector"
        if selector_dir.exists() and selector_dir.is_dir():
            for model_file in selector_dir.glob("*.json"):
                try:
                    with open(model_file, "r", encoding="utf-8") as f:
                        model_data = json.load(f)
                    
                    # Get file stats
                    stats = model_file.stat()
                    
                    models.append({
                        "name": model_file.stem,
                        "type": "contextual_continuous_selector",
                        "mode": model_file.stem,
                        "path": str(model_file),
                        "size_bytes": stats.st_size,
                        "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        "data": model_data
                    })
                except Exception as e:
                    logger.warning(f"Failed to read model file {model_file}: {e}")

        # Check for other model types in shared_models
        for item in shared_models_dir.iterdir():
            if item.is_dir() and item.name != "contextual_continuous_selector":
                # Check for model files in subdirectories
                for model_file in item.glob("*.json"):
                    try:
                        with open(model_file, "r", encoding="utf-8") as f:
                            model_data = json.load(f)
                        
                        stats = model_file.stat()
                        
                        models.append({
                            "name": model_file.stem,
                            "type": item.name,
                            "mode": model_file.stem,
                            "path": str(model_file),
                            "size_bytes": stats.st_size,
                            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                            "data": model_data
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read model file {model_file}: {e}")

        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline["name"],
            "models": models,
            "total": len(models),
            "shared_models_path": str(shared_models_dir)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline models for {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/restart")
async def restart_pipeline(pipeline_id: str):
    """
    Restart a pipeline from scratch.

    Keeps per-project:
    - images/ directory (original uploads)
    - images_resized/ directory (resized copies)
    - outputs/sparse/ directory (COLMAP point clouds)
    - The baseline run directory (first run in runs/)

    Deletes per-project:
    - All non-baseline run directories under runs/
    - outputs/engines/ (trained splat models)
    - models/ (local learner weights)
    - analytics/ files
    - status.json (reset to pending)
    - .batch_lineage_latest.json
    - .project_model_state.json

    Also deletes pipeline-level:
    - shared_models/ directory (learner weights)
    - pipeline status reset to pending
    """
    import shutil
    import stat
    import os

    def _remove_readonly(func, path, _exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        func(path)

    def _rmtree(p: Path) -> None:
        if p.exists() or p.is_symlink():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, onerror=_remove_readonly)

    try:
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        config = pipeline.get("config", {})
        pipeline_folder = Path(config.get("pipeline_folder", ""))
        if not pipeline_folder.exists():
            raise HTTPException(status_code=404, detail="Pipeline folder not found on disk")

        # Stop any running orchestrator for this pipeline first
        try:
            training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)
        except Exception as exc:
            logger.warning("Could not stop pipeline orchestrator before restart for %s: %s", pipeline_id, exc)

        projects = config.get("projects", [])
        deleted_summary: list[dict] = []

        for project_cfg in projects:
            project_id = str(project_cfg.get("project_id") or "").strip()
            if not project_id:
                continue

            # Locate project directory — may be inside pipeline_folder or DATA_DIR
            project_dir: Path | None = None
            candidate_in_pipeline = pipeline_folder / project_id
            if candidate_in_pipeline.exists() and candidate_in_pipeline.is_dir():
                project_dir = candidate_in_pipeline
            else:
                candidate_in_data = DATA_DIR / project_id
                if candidate_in_data.exists() and candidate_in_data.is_dir():
                    project_dir = candidate_in_data
                else:
                    # Try scanning pipeline folder by config.json id
                    for sub in pipeline_folder.iterdir():
                        if not sub.is_dir():
                            continue
                        cfg_file = sub / "config.json"
                        if cfg_file.exists():
                            try:
                                with open(cfg_file, "r", encoding="utf-8") as fh:
                                    sub_cfg = json.load(fh)
                                if str(sub_cfg.get("id") or "").strip() == project_id:
                                    project_dir = sub
                                    break
                            except Exception:
                                continue

            if project_dir is None or not project_dir.exists():
                logger.warning("Restart: project dir not found for %s, skipping", project_id)
                continue

            # Determine baseline run id (first run alphabetically / by name)
            runs_root = project_dir / "runs"
            baseline_run_id: str | None = None
            if runs_root.exists() and runs_root.is_dir():
                run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
                if run_dirs:
                    baseline_run_id = run_dirs[0].name

            # Delete non-baseline run directories
            deleted_runs: list[str] = []
            if runs_root.exists() and runs_root.is_dir():
                for run_dir in list(runs_root.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    if run_dir.name == baseline_run_id:
                        continue  # keep baseline
                    _rmtree(run_dir)
                    deleted_runs.append(run_dir.name)

            # Delete engine outputs (trained splat models)
            _rmtree(project_dir / "outputs" / "engines")

            # Delete local learner weights
            _rmtree(project_dir / "models")

            # Delete batch/model state metadata
            for meta_file in (
                project_dir / ".batch_lineage_latest.json",
                project_dir / ".project_model_state.json",
            ):
                meta_file.unlink(missing_ok=True)

            # Reset project status to pending
            status_file = project_dir / "status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as fh:
                        existing_status = json.load(fh)
                except Exception:
                    existing_status = {}
                reset_status = {
                    "project_id": project_id,
                    "status": "pending",
                    "progress": 0,
                    "name": existing_status.get("name"),
                    "created_at": existing_status.get("created_at"),
                    "base_session_id": baseline_run_id,
                }
                try:
                    tmp = status_file.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(reset_status, fh, indent=2)
                    tmp.replace(status_file)
                except Exception as exc:
                    logger.warning("Failed to reset status for project %s: %s", project_id, exc)

            deleted_summary.append({
                "project_id": project_id,
                "baseline_kept": baseline_run_id,
                "deleted_runs": deleted_runs,
            })

        # Delete pipeline-level shared_models (learner weights)
        shared_models_dir = pipeline_folder / "shared_models"
        _rmtree(shared_models_dir)

        # Reset pipeline state to pending
        training_pipeline_storage.update_pipeline(pipeline_id, {
            "status": "pending",
            "current_phase": 0,
            "current_pass": 0,
            "current_run": 0,
            "current_project_index": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "mean_reward": None,
            "best_reward": None,
            "success_rate": None,
            "last_error": None,
            "started_at": None,
            "completed_at": None,
        })

        logger.info("Pipeline %s restarted: %d projects cleaned", pipeline_id, len(deleted_summary))
        return {
            "status": "restarted",
            "pipeline_id": pipeline_id,
            "projects_cleaned": len(deleted_summary),
            "details": deleted_summary,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to restart pipeline %s: %s", pipeline_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart pipeline: {exc}")


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline."""
    success = training_pipeline_storage.delete_pipeline(pipeline_id)

    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {"message": "Pipeline deleted successfully"}


class ElevateLearnerModelRequest(BaseModel):
    model_name: str = Field(..., description="User-friendly name for the elevated model")
    mode: str = Field(..., description="AI input mode (e.g., exif_only, exif_plus_flight_plan)")


@router.post("/{pipeline_id}/elevate-learner-model")
async def elevate_learner_model(pipeline_id: str, request: ElevateLearnerModelRequest):
    """
    Elevate a pipeline's shared learner model to global model registry.

    This allows the learned parameter selection model to be reused across
    other projects and pipelines.
    """
    try:
        # Get pipeline
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        config = pipeline["config"]
        pipeline_folder = Path(config.get("pipeline_folder"))
        if not pipeline_folder.exists():
            raise HTTPException(status_code=404, detail="Pipeline folder not found")

        # Locate shared_models directory
        shared_model_dir = pipeline_folder / "shared_models"
        if not shared_model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Shared models directory not found. Has the pipeline trained any projects?"
            )

        # Validate mode
        valid_modes = ["exif_only", "exif_plus_flight_plan", "exif_plus_flight_plan_plus_external"]
        if request.mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
            )

        # Check if learner model exists
        learner_model_path = shared_model_dir / "contextual_continuous_selector" / f"{request.mode}.json"
        if not learner_model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Learner model for mode '{request.mode}' not found. Has the pipeline completed any training runs?"
            )

        # Get pipeline projects and shared config for lineage tracking
        pipeline_projects = config.get("projects", [])
        shared_config = config.get("shared_config", {})

        # Elevate the model with lineage
        model_record = model_registry.elevate_learner_model(
            shared_model_dir=shared_model_dir,
            mode=request.mode,
            model_name=request.model_name,
            pipeline_id=pipeline["id"],
            pipeline_name=pipeline["name"],
            pipeline_projects=pipeline_projects,
            shared_config=shared_config,
        )

        logger.info(f"Elevated learner model from pipeline {pipeline_id}: {model_record['model_id']}")

        return {
            "success": True,
            "model_id": model_record["model_id"],
            "model_name": model_record["model_name"],
            "mode": request.mode,
            "created_at": model_record["created_at"],
            "paths": model_record["paths"],
            "provenance": model_record.get("provenance_summary", {}),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to elevate learner model for pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
