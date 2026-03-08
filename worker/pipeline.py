import logging
import os
from pathlib import Path
from app.services import colmap, storage, status, gsplat

logger = logging.getLogger(__name__)

# Use Docker worker for processing (recommended)
USE_DOCKER = os.getenv("USE_DOCKER_WORKER", "true").lower() == "true"

# Log configuration on startup
logger.info(f"Pipeline Configuration: USE_DOCKER_WORKER={USE_DOCKER}")
logger.info(f"  - Current mode: {'DOCKER' if USE_DOCKER else 'LOCAL (requires COLMAP/gsplat installed)'}")


def run_full_pipeline(project_id: str, params: dict | None = None):
    """
    Run the full Gaussian Splatting pipeline.
    Includes error handling and status tracking.
    
    Set USE_DOCKER_WORKER=true to use Docker worker (recommended).
    Set USE_DOCKER_WORKER=false to use local COLMAP/gsplat (must be installed).
    """
    # Extract mode from params (default to baseline if not specified)
    mode = params.get("mode", "baseline") if params else "baseline"
    max_steps = params.get("max_steps") if params else None
    stage = params.get("stage", "full") if params else "full"
    
    # Configure per-project file logging (local mode only)
    try:
        from app.services import storage as _storage
        _project_dir = _storage.get_project_dir(project_id)
        _log_file = _project_dir / "processing.log"
        _log_file.parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(_log_file, mode='a')
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        _root = logging.getLogger()
        _root.addHandler(_fh)
        logger.info(f"Initialized project log file: {_log_file}")
    except Exception as e:
        logger.warning(f"Failed to initialize project log file: {e}")

    # Use Docker worker for processing
    if USE_DOCKER:
        logger.info(f"Using DOCKER WORKER for project {project_id} (mode: {mode})")
        try:
            # Pre-Docker: provide useful, user-visible substep messages
            logger.info("DOCKER: Preparing Docker worker (check image, GPU, volumes)...")
            status.update_status(
                project_id,
                "processing",
                progress=5,
                mode=mode,
                maxSteps=max_steps,
                stage="docker",
                stage_progress=10,
                message="Preparing Docker worker (checking image, GPU, and volumes)...",
            )

            # Hint that image may be pulled/build required
            try:
                import subprocess as _sp
                _sp.run(["docker", "image", "inspect", "websplat-worker:latest"], check=True, capture_output=True)
                status.update_status(
                    project_id,
                    "processing",
                    progress=7,
                    stage="docker",
                    stage_progress=20,
                    message="Docker image found locally. Starting worker container...",
                )
                logger.info("DOCKER: Image present locally")
            except Exception:
                status.update_status(
                    project_id,
                    "processing",
                    progress=7,
                    stage="docker",
                    stage_progress=20,
                    message="Docker image not found locally. Pulling or using local build when starting...",
                )
                logger.info("DOCKER: Image not found locally; docker will pull if needed")

            # Launch the worker container (container will continue updating status.json)
            logger.info("DOCKER: Starting worker container")
            status.update_status(
                project_id,
                "processing",
                progress=9,
                stage="docker",
                stage_progress=50,
                message="Starting worker container...",
            )

            colmap.run_colmap_docker(project_id, params)

            # When docker finishes, the worker should have set final status.
            # Add a final docker-stage marker in case the worker couldn't start.
            logger.info("DOCKER: Worker container finished")
            status.update_status(
                project_id,
                "processing",
                stage="docker",
                stage_progress=100,
                message="Worker container finished. Finalizing...",
            )
            logger.info(f"Docker worker completed for project {project_id}")

            # Docker container writes the final status; respect stop state if present
            final_status = status.get_status(project_id)
            # Project dir and stop flag fallback
            try:
                project_dir = storage.get_project_dir(project_id)
                stop_flag = (project_dir / "stop_requested")
            except Exception:
                stop_flag = None

            if final_status.get("status") in {"stopped", "failed"}:
                logger.info(f"Pipeline ended with status {final_status.get('status')} for project {project_id}")
            elif final_status.get("stop_requested", False) or (stop_flag is not None and stop_flag.exists()):
                # If worker failed to persist a final stopped status, fall back to stop flag file
                s = status.get_status(project_id)
                pct = s.get("stopped_percentage") or s.get("percentage") or s.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                # Respect stopped_stage if worker recorded it, else fallback to stage
                stopped_stage = s.get("stopped_stage") or s.get("stage") or "colmap"
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage=stopped_stage, message="⏸️ Processing stopped by user.")
                logger.info(f"Pipeline stopped by user (detected stop_requested) for project {project_id}; stage={stopped_stage}")
            else:
                status.update_status(project_id, "completed", progress=100, stop_requested=False, stage="export", message="Finished")
                logger.info(f"✓ Pipeline completed for project {project_id}")
        except Exception as e:
            logger.error(f"Docker worker failed: {e}", exc_info=True)
            status.update_status(project_id, "failed", error=str(e))
            raise
        return
    
    try:
        logger.info(f"Starting pipeline for project: {project_id}")
        
        project_dir = storage.get_project_dir(project_id)
        image_dir = project_dir / "images"
        output_dir = project_dir / "outputs"
        
        # Ensure directories exist
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if stage == "colmap_only":
            # 1️⃣ COLMAP SfM only
            logger.info(f"Running COLMAP for project: {project_id}")
            status.update_status(project_id, "processing", progress=20, stage="colmap", message="Running COLMAP")
            sparse_dir = colmap.run_colmap(image_dir, output_dir, params)
            logger.info(f"COLMAP completed for project: {project_id}")
            # Only set completed if not stopped
            current_status = status.get_status(project_id)
            if current_status.get("stop_requested", False):
                pct = current_status.get("percentage") or current_status.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user.")
            else:
                status.update_status(project_id, "completed", progress=100, stop_requested=False, stage="colmap", message="Sparse reconstruction done")
            gs_output = None
        elif stage == "train_only":
            # 2️⃣ Training only, assuming sparse exists
            sparse_dir = output_dir / "sparse"
            if not (sparse_dir / "0").exists():
                raise FileNotFoundError("Sparse model not found. Run COLMAP first.")
            logger.info(f"Running Gaussian Splatting for project: {project_id}")
            status.update_status(project_id, "processing", progress=60, stage="training", message="Training gaussians")
            gs_output = gsplat.run_gsplat(image_dir, sparse_dir, output_dir, params or {})
            logger.info(f"Gaussian Splatting completed for project: {project_id}")
            current_status = status.get_status(project_id)
            if current_status.get("stop_requested", False):
                pct = current_status.get("percentage") or current_status.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage="training", message="⏸️ Processing stopped by user.")
            else:
                status.update_status(project_id, "completed", progress=100, stop_requested=False, stage="export", message="Finished")
        else:
            import time
            # Full pipeline
            logger.info(f"Running COLMAP for project: {project_id}")
            status.update_status(project_id, "processing", progress=20, stage="colmap", message="Running COLMAP")
            sparse_dir = colmap.run_colmap(image_dir, output_dir, params)
            logger.info(f"COLMAP completed for project: {project_id}")
            # Mark COLMAP as completed for frontend tick/green
            current_status = status.get_status(project_id)
            if current_status.get("stop_requested", False):
                pct = current_status.get("percentage") or current_status.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user.")
                return None
            status.update_status(project_id, "completed", progress=33, stop_requested=False, stage="colmap", message="Sparse reconstruction done")
            time.sleep(1.5)  # Give frontend time to observe

            logger.info(f"Running Gaussian Splatting for project: {project_id}")
            status.update_status(project_id, "processing", progress=60, stage="training", message="Training gaussians")
            gs_output = gsplat.run_gsplat(image_dir, sparse_dir, output_dir, params or {})
            logger.info(f"Gaussian Splatting completed for project: {project_id}")
            current_status = status.get_status(project_id)
            if current_status.get("stop_requested", False):
                pct = current_status.get("percentage") or current_status.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage="training", message="⏸️ Processing stopped by user.")
                return gs_output
            # Mark training as completed for frontend tick/green
            status.update_status(project_id, "completed", progress=90, stop_requested=False, stage="training", message="Training done")
            time.sleep(1.5)  # Give frontend time to observe

            # Export step
            status.update_status(project_id, "processing", progress=100, stage="export", message="Exporting outputs")
            # (Assume export is part of gsplat or handled here)
            # Final check before marking done
            current_status = status.get_status(project_id)
            if current_status.get("stop_requested", False):
                pct = current_status.get("percentage") or current_status.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage="export", message="⏸️ Processing stopped by user.")
            else:
                status.update_status(project_id, "done", progress=100, stop_requested=False, stage="export", message="Finished")
        
        return gs_output
    
    except Exception as e:
        logger.error(f"Pipeline failed for project {project_id}: {str(e)}", exc_info=True)
        status.update_status(project_id, "failed", error=str(e))
        raise
