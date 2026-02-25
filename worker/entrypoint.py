#!/usr/bin/env python3
"""
Docker worker entrypoint for Gaussian Splatting pipeline.
Runs COLMAP + faithful gsplat training (with research hooks) + export.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
import time

from .image_resize import prepare_training_images, normalize_max_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_status(
    project_dir: Path,
    status: str,
    progress: int = None,
    error: str = None,
    resumable: bool = None,
    mode: str = None,
    tuning_active: bool = None,
    currentStep: int = None,
    maxSteps: int = None,
    last_tuning: dict = None,
    stop_requested: bool = None,
    stage: str = None,
    stage_progress: int = None,
    message: str = None,
    timing: dict = None,
    stopped_stage: str = None,
    stopped_step: int | str = None,
    stopped_percentage: float = None,
):
    """Update status.json file."""
    status_file = project_dir / "status.json"

    try:
        if status_file.exists():
            with open(status_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"status": "pending", "progress": 0}

        data["status"] = status
        if progress is not None:
            data["progress"] = progress
        if error is not None:
            data["error"] = error
        if mode is not None:
            data["mode"] = mode
        if tuning_active is not None:
            data["tuning_active"] = tuning_active
        if resumable is not None:
            data["resumable"] = resumable
        if currentStep is not None:
            data["currentStep"] = currentStep
        if maxSteps is not None:
            data["maxSteps"] = maxSteps
        if last_tuning is not None:
            data["last_tuning"] = last_tuning
        if stop_requested is not None:
            data["stop_requested"] = stop_requested
        if stage is not None:
            data["stage"] = stage
        if stage_progress is not None:
            data["stage_progress"] = int(stage_progress)
        if message is not None:
            data["message"] = message
        if timing is not None:
            data["timing"] = timing
        if stopped_stage is not None:
            data["stopped_stage"] = stopped_stage
        if stopped_step is not None:
            data["stopped_step"] = stopped_step
        if stopped_percentage is not None:
            try:
                data["stopped_percentage"] = float(stopped_percentage)
            except Exception:
                data["stopped_percentage"] = stopped_percentage

        # Add percentage field: prefer overall 'progress' if available, else derive from currentStep/maxSteps
        percentage = None
        if data.get("progress") is not None:
            try:
                percentage = float(data["progress"])
            except Exception:
                percentage = None
        if percentage is None and data.get("currentStep") is not None and data.get("maxSteps"):
            try:
                step = float(data["currentStep"])
                maxs = float(data["maxSteps"])
                if maxs > 0:
                    percentage = max(0.0, min(100.0, (step / maxs) * 100.0))
            except Exception:
                percentage = None
        data["percentage"] = round(percentage, 2) if percentage is not None else 0.0

        temp_file = status_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        temp_file.replace(status_file)

    except Exception as e:
        logger.error(f"Failed to update status: {e}")


def write_metrics(project_dir: Path, metrics: dict):
    """Write training metrics to metrics.json."""
    metrics_file = project_dir / "outputs" / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        temp_file = metrics_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        temp_file.replace(metrics_file)
    except Exception as e:
        logger.error(f"Failed to write metrics: {e}")


def _run_cmd_with_retry(cmd: list[str], retries: int = 3, delay_sec: float = 2.0):
    """Run a subprocess command with retries on transient SQLite lock errors."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if res.stdout:
                logger.info(res.stdout.strip())
            if res.stderr:
                logger.debug(res.stderr.strip())
            return
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            stdout = (e.stdout or "")
            last_err = e
            # Detect common transient lock conditions
            if "database is locked" in stderr or "busy" in stderr:
                logger.warning(f"SQLite busy/locked detected (attempt {attempt}/{retries}). Retrying after {delay_sec}s...")
                time.sleep(delay_sec)
                continue
            # Non-retryable error
            logger.error(f"Command failed: {cmd}\nSTDOUT: {stdout}\nSTDERR: {e.stderr}")
            raise
    # Exhausted retries
    logger.error(f"Command failed after retries: {cmd}\nERR: {last_err}")
    raise last_err


def _cleanup_sqlite_sidecars(db_path: Path):
    """Remove SQLite -wal/-shm files that can linger after abrupt terminations."""
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            try:
                sidecar.unlink()
                logger.info(f"Removed stale SQLite sidecar: {sidecar}")
            except Exception as e:
                logger.warning(f"Failed to remove sidecar {sidecar}: {e}")


def run_colmap(image_dir: Path, output_dir: Path, params: dict | None = None) -> Path:
    """Run COLMAP Structure-from-Motion."""
    logger.info("Starting COLMAP...")
    colmap_start = time.time()

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Stop flag file used to request graceful stop from outside the worker
    stop_flag = output_dir.parent / "stop_requested"

    # Remove existing database and sidecars to prevent locking issues
    if database_path.exists():
        logger.info(f"Removing existing database: {database_path}")
        try:
            database_path.unlink()
        except Exception:
            # Try truncate if unlink fails
            database_path.write_bytes(b"")
    _cleanup_sqlite_sidecars(database_path)
    # allow colmap tuning via params.colmap
    p = params.get("colmap", {}) if isinstance(params, dict) else {}

    logger.info("COLMAP: Feature extraction...")
    update_status(
        output_dir.parent, "processing", progress=5, stage="colmap", stage_progress=10,
        message="📸 Extracting features from images (detecting keypoints and descriptors)...",
        timing={"start": colmap_start}
    )
    # Abort early if stop was requested before starting this substep
    if stop_flag.exists():
        update_status(output_dir.parent, "stopped", progress=0, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user before feature extraction.", stopped_stage="colmap")
        try:
            stop_flag.unlink()
        except Exception:
            pass
        sys.exit(0)
    feat_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
    ]
    if p.get("max_image_size"):
        feat_cmd += ["--SiftExtraction.max_image_size", str(p.get("max_image_size"))]
    else:
        feat_cmd += ["--SiftExtraction.max_image_size", "1600"]
    if p.get("peak_threshold") is not None:
        feat_cmd += ["--SiftExtraction.peak_threshold", str(p.get("peak_threshold"))]
    try:
        _run_cmd_with_retry(feat_cmd)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").lower()
        logger.error(f"COLMAP feature_extractor failed: {e.stderr}")
        if "cuda" in stderr or "driver" in stderr:
            # Surface a clear failure to the frontend and stop processing, but allow resume
            msg = "GPU error: CUDA driver/runtime mismatch. Update host NVIDIA drivers or use a compatible worker image."
            logger.error(msg)
            update_status(output_dir.parent, "stopped", progress=0, error=msg, stage="colmap", message=msg, stop_requested=True, stopped_stage="colmap", resumable=True)
            sys.exit(1)
        else:
            raise

    logger.info("COLMAP: Feature matching...")
    update_status(
        output_dir.parent, "processing", progress=15, stage="colmap", stage_progress=50,
        message="🔗 Matching features between images (finding correspondences)...",
        timing={"start": colmap_start, "elapsed": time.time() - colmap_start}
    )
    # Abort early if stop was requested before matching
    if stop_flag.exists():
        update_status(output_dir.parent, "stopped", progress=0, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user before feature matching.", stopped_stage="colmap")
        try:
            stop_flag.unlink()
        except Exception:
            pass
        sys.exit(0)
    # matching strategy
    guided = p.get("guided_matching")
    matching_type = p.get("matching_type", "exhaustive")
    if matching_type == "sequential":
        match_cmd = ["colmap", "sequential_matcher", "--database_path", str(database_path)]
    else:
        match_cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database_path)]
    if guided is not None:
        match_cmd += ["--SiftMatching.guided_matching", "1" if guided else "0"]
    try:
        _run_cmd_with_retry(match_cmd)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").lower()
        logger.error(f"COLMAP matcher failed: {e.stderr}")
        if "cuda" in stderr or "driver" in stderr:
            msg = "GPU error during matching: CUDA driver/runtime mismatch. Update host NVIDIA drivers or use a compatible worker image."
            logger.error(msg)
            update_status(output_dir.parent, "stopped", progress=0, error=msg, stage="colmap", message=msg, stop_requested=True, stopped_stage="colmap", resumable=True)
            sys.exit(1)
        else:
            raise

    logger.info("COLMAP: Sparse reconstruction...")
    update_status(
        output_dir.parent, "processing", progress=30, stage="colmap", stage_progress=85,
        message="📐 Building 3D structure (triangulating camera poses and points)...",
        timing={"start": colmap_start, "elapsed": time.time() - colmap_start}
    )
    # Run mapper with ability to interrupt if stop requested
    if stop_flag.exists():
        update_status(output_dir.parent, "stopped", progress=0, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user before sparse reconstruction.", stopped_stage="colmap")
        sys.exit(0)
    try:
        mapper_cmd = ["colmap", "mapper", "--database_path", str(database_path), "--image_path", str(image_dir), "--output_path", str(sparse_dir), "--Mapper.ba_refine_principal_point", "1", "--Mapper.ba_refine_focal_length", "1", "--Mapper.ba_refine_extra_params", "1"]
        if p.get("mapper_num_threads"):
            mapper_cmd += ["--Mapper.num_threads", str(p.get("mapper_num_threads"))]
        else:
            # Default to 2 mapper threads to reduce memory usage
            mapper_cmd += ["--Mapper.num_threads", "2"]
        # Do not capture stdout/stderr into pipes here; allow the child to inherit the
        # container's stdout/stderr so COLMAP can stream directly to Docker logs.
        # Capturing into pipes without continuously reading can lead to pipe buffer
        # deadlocks where COLMAP appears to hang while producing output.
        # Stream COLMAP mapper output into the application logger so it is
        # persisted to the project's processing.log and visible via frontend.
        # Use a pipe and continuously read to avoid deadlocks.
        proc = subprocess.Popen(
            mapper_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            # Read lines as they arrive and log them (will be written to file handler)
            for line in proc.stdout:
                try:
                    logger.info(line.rstrip())
                except Exception:
                    pass
                # If a stop has been requested externally, terminate mapper
                if stop_flag.exists():
                    logger.info("Stop requested during COLMAP mapper; terminating process")
                    try:
                        proc.terminate()
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    update_status(output_dir.parent, "stopped", progress=0, stop_requested=True, stage="colmap", message="⏸️ Processing stopped by user during sparse reconstruction.", stopped_stage="colmap")
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
                    try:
                        stop_flag.unlink()
                    except Exception:
                        pass
                    sys.exit(0)
            # After stdout is exhausted, wait for process exit and check return code
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, proc.args)
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").lower()
        logger.error(f"COLMAP mapper failed: {e}")
        if "cuda" in stderr or "driver" in stderr:
            msg = "GPU error during mapper: CUDA driver/runtime mismatch. Update host NVIDIA drivers or use a compatible worker image."
            logger.error(msg)
            update_status(output_dir.parent, "stopped", progress=0, error=msg, stage="colmap", message=msg, stop_requested=True, stopped_stage="colmap", resumable=True)
            sys.exit(1)
        else:
            raise

    reconstruction_dirs = list(sparse_dir.glob("*/"))
    if not reconstruction_dirs:
        raise RuntimeError("COLMAP reconstruction failed - no output")

    # Convert COLMAP outputs into lightweight points.bin files for the frontend
    converter = None
    try:
        from . import pointsbin as worker_pointsbin

        converter = worker_pointsbin
    except Exception as worker_conv_err:
        logger.debug("Worker pointsbin module unavailable: %s", worker_conv_err)
        try:
            from app.services import pointsbin as app_pointsbin

            converter = app_pointsbin
        except Exception as app_conv_err:
            logger.debug("Backend pointsbin module unavailable: %s", app_conv_err)

    if converter is not None:
        for recon_dir in reconstruction_dirs:
            try:
                count = converter.convert_colmap_recon_to_pointsbin(recon_dir)
                if count:
                    logger.info("Converted %s -> points.bin (%d points)", recon_dir, count)
            except Exception as conv_err:
                logger.warning("Failed to convert %s to points.bin: %s", recon_dir, conv_err)
    else:
        # Non-fatal: frontend simply falls back to COLMAP binaries if converter unavailable
        logger.debug("Skipping points.bin export (converter unavailable)")

    # Mark COLMAP done for substep progress
    colmap_end = time.time()
    update_status(
        output_dir.parent, "processing", stage="colmap", stage_progress=100, message="✅ COLMAP complete",
        timing={"start": colmap_start, "end": colmap_end, "elapsed": colmap_end - colmap_start}
    )
    # Mark COLMAP as completed for frontend tick/green
    update_status(
        output_dir.parent, "completed", progress=33, stage="colmap", message="Sparse reconstruction done"
    )
    time.sleep(1.5)
    return reconstruction_dirs[0]


def _export_with_gsplat(checkpoint_path: Path, output_dir: Path):
    """Load a checkpoint and export .splat and .ply using gsplat exporter."""
    import torch
    from gsplat.exporter import export_splats

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt if isinstance(ckpt, dict) else {}

    def grab(names, default=None):
        for n in names:
            if n in state:
                return state[n]
        return default

    means = grab(["means", "xyz", "_xyz"])
    scales = grab(["scales", "scaling", "_scaling"])
    quats = grab(["quats", "rotations", "_rotation", "rot"])
    opacities = grab(["opacities", "_opacity", "opacity"])
    sh0 = grab(["sh0", "sh_features", "_sh0"])
    shN = grab(["shN", "_shN"])

    if means is None:
        raise ValueError(f"Checkpoint missing means; available keys: {list(state.keys())}")

    means = means.cpu().float()
    scales = scales.cpu().float() if scales is not None else torch.zeros_like(means)
    quats = quats.cpu().float() if quats is not None else torch.zeros((means.shape[0], 4))
    opacities = opacities.cpu().float() if opacities is not None else torch.zeros(means.shape[0])

    if quats.numel() > 0:
        norm = torch.norm(quats, dim=1, keepdim=True).clamp_min(1e-7)
        quats = quats / norm

    if sh0 is None or (isinstance(sh0, torch.Tensor) and sh0.numel() == 0):
        sh0 = torch.ones((means.shape[0], 1, 3), dtype=torch.float32) * 0.5
    else:
        sh0 = sh0.cpu().float()
        if sh0.ndim == 2:
            sh0 = sh0.unsqueeze(1)

    if shN is None or (isinstance(shN, torch.Tensor) and shN.numel() == 0):
        shN = torch.zeros((means.shape[0], 0, 3), dtype=torch.float32)
    else:
        shN = shN.cpu().float()

    if opacities.ndim > 1:
        opacities = opacities.squeeze()

    logger.info(
        "Exporting with gsplat exporter | means %s, scales %s, quats %s, opacities %s, sh0 %s, shN %s",
        tuple(means.shape),
        tuple(scales.shape),
        tuple(quats.shape),
        tuple(opacities.shape),
        tuple(sh0.shape),
        tuple(shN.shape),
    )

    splat_path = output_dir / "splats.splat"
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="splat",
        save_to=str(splat_path),
    )
    logger.info("✓ Exported .splat -> %s (%d bytes)", splat_path, splat_path.stat().st_size)

    ply_path = output_dir / "splats.ply"
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(ply_path),
    )
    logger.info("✓ Exported .ply -> %s (%d bytes)", ply_path, ply_path.stat().st_size)


def run_gsplat_training(image_dir: Path, colmap_dir: Path, output_dir: Path, params: dict, resume: bool = False):
    """Run faithful gsplat training with research intervention hooks."""
    from .trainer import GsplatTrainer
    
    logger.info("Starting gsplat training...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract parameters
    p = params or {}
    mode = p.get("mode", "baseline")  # "baseline" or "modified"
    max_steps = p.get("max_steps", 300)
    project_dir = output_dir.parent
    training_image_dir = image_dir
    resize_stats = None
    training_max_size = normalize_max_size(p.get("training_image_max_size"))
    if training_max_size:
        resize_msg = f"🪄 Downscaling training images to ≤ {training_max_size}px before training..."
        logger.info("Starting training image resize pass (≤%d px)", training_max_size)
        update_status(
            project_dir,
            "processing",
            progress=54,
            stage="training",
            stage_progress=0,
            message=resize_msg,
            mode=mode,
        )
        try:
            training_image_dir, resize_stats = prepare_training_images(image_dir, project_dir, training_max_size)
            logger.info(
                "Finished resizing training images (%s)",
                resize_stats,
            )
            update_status(
                project_dir,
                "processing",
                stage="training",
                stage_progress=5,
                message=f"✅ Training images ready at ≤ {training_max_size}px",
                mode=mode,
            )
        except Exception as exc:
            training_image_dir = image_dir
            resize_stats = None
            logger.warning("Failed to prepare resized training images; using originals. Error: %s", exc)

    stop_flag = output_dir.parent / "stop_requested"

    def stop_checker() -> bool:
        return stop_flag.exists()
    
    # Track gsplat training timing
    gsplat_start = time.time()
    # Progress callback for status updates (single status text per substep)
    def progress_callback(step, progress, loss, last_tuning=None):
        # Only update every 100 steps to avoid I/O overhead
        if step % 100 == 0:
            requested_stop = stop_checker()
            status_text = "stopping" if requested_stop else "processing"
            # Compose a single, clear status message for this substep
            if requested_stop:
                msg = f"⏸️ Stopping after step {step}/{max_steps} completes (loss: {loss:.6f})..."
            elif progress < 0.1:
                msg = f"🎯 Training step {step}/{max_steps} - Initial optimization (loss: {loss:.6f})"
            elif progress < 0.3:
                msg = f"🔧 Training step {step}/{max_steps} - Refining gaussians (loss: {loss:.6f})"
            elif progress < 0.7:
                msg = f"✨ Training step {step}/{max_steps} - Optimizing quality (loss: {loss:.6f})"
            else:
                msg = f"🎨 Training step {step}/{max_steps} - Final refinement (loss: {loss:.6f})"
            if mode == "modified" and 50 <= step <= 300:
                msg += " [Adaptive tuning active]"
            # Estimate elapsed and ETA for gsplat
            now = time.time()
            elapsed = now - gsplat_start
            eta = (elapsed / (progress if progress > 0 else 1e-6)) * (1 - progress) if progress > 0 else None
            timing = {"start": gsplat_start, "elapsed": elapsed}
            if eta is not None:
                timing["eta"] = eta
            # Only one status/progress per substep update
            update_status(
                output_dir.parent,
                status_text,
                progress=60 + int(progress * 0.35),
                mode=mode,
                currentStep=step,
                maxSteps=max_steps,
                tuning_active=(mode == "modified" and 50 <= step <= 300),
                last_tuning=last_tuning,
                stop_requested=requested_stop,
                stage="training",
                stage_progress=int(max(0, min(100, progress * 100))),
                message=msg,
                timing=timing,
            )
            write_metrics(output_dir.parent, {
                "step": step,
                "loss": loss,
                "progress": progress,
            })
    
    # Initialize and run trainer
    # Determine device availability
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    device = "cuda" if (p.get("use_cuda", True) and cuda_ok) else "cpu"

    # Log the received params for debugging
    try:
        logger.info(f"Worker params: {p}")
        logger.info(f"Top-level params (raw): {params}")
    except Exception:
        pass

    init_message = f"🚀 Initializing Gaussian Splatting trainer ({'GPU ⚡' if device == 'cuda' else 'CPU'}, {mode} mode)..."
    if resize_stats:
        init_message = (
            f"{init_message}\n📐 Training images limited to {training_max_size}px"
        )

    update_status(
        project_dir,
        "processing",
        progress=55,
        stage="training",
        stage_progress=0,
        message=init_message,
        mode=mode,
        timing={"start": gsplat_start},
    )

    # Normalize colmap_dir: if provided dir lacks cameras.bin but has a '0' subdir with cameras.bin, use it
    try:
        cdir = Path(colmap_dir)
        if not (cdir / "cameras.bin").exists() and (cdir / "0" / "cameras.bin").exists():
            colmap_dir = cdir / "0"
            logger.info("Adjusted colmap_dir to %s (using subfolder 0)", colmap_dir)
    except Exception:
        pass

    trainer = GsplatTrainer(
        image_dir=training_image_dir,
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        mode=mode,
        max_steps=max_steps,
        max_init_gaussians=p.get("gsplat_max_gaussians", None),
        max_gaussians_cap=p.get("gsplat_hard_cap", None),
        amp_enabled=bool(p.get("amp", False)),
        pruning_enabled=bool(p.get("pruning_enabled", False)),
        pruning_policy=p.get("pruning_policy", "opacity"),
        pruning_weights=p.get("pruning_weights", {}),
        progress_callback=progress_callback,
        splat_export_interval=p.get("splat_export_interval"),
        png_export_interval=p.get("png_export_interval"),
        auto_early_stop=bool(p.get("auto_early_stop", False)),
        stop_checker=stop_checker,
        resume=resume,
    )
    
    logger.info(f"Training mode: {mode}")
    trainer.train()
    gsplat_end = time.time()

    # After training completes, save final metrics to metadata.json
    logger.info("Saving evaluation metrics...")
    try:
        # Load metrics from adaptive_tuning_results.json if available
        tuning_results_file = output_dir / "adaptive_tuning_results.json"
        if tuning_results_file.exists():
            with open(tuning_results_file) as f:
                tuning_data = json.load(f)
                final_metrics = tuning_data.get("final_evaluation", {})
                
                # Save to metadata.json for easy access
                metadata_file = output_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                
                metadata["evaluation_metrics"] = final_metrics
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved evaluation metrics: {final_metrics}")
    except Exception as e:
        logger.warning(f"Could not save evaluation metrics: {e}")
    
    # If stopped, do not export or mark as completed, just exit immediately

    if trainer.stop_reason:
        logger.info("Training stopped by user, skipping export and completion status.")
        # Set status to 'stopped' for clarity, and record stopped_stage/stopped_step
        # Try to read most recent metrics (written by progress_callback) for accurate step/progress
        metrics_file = output_dir.parent / "outputs" / "metrics.json"
        last_progress = None
        last_percentage = None
        last_step = None
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    m = json.load(f)
                    last_step = m.get("step")
                    mprogress = m.get("progress")
                    # mprogress expected to be fraction in [0,1] representing training completion
                    if isinstance(mprogress, (int, float)):
                        # Map training fraction to overall percentage (training occupies ~35% range from 60->95)
                        try:
                            last_percentage = 60 + (float(mprogress) * 35)
                        except Exception:
                            last_percentage = None
            except Exception:
                pass
        # Fallback to status.json if metrics not available
        if last_percentage is None:
            status_path = output_dir.parent / "status.json"
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        sd = json.load(f)
                        last_progress = sd.get("progress")
                        last_percentage = sd.get("percentage")
                except Exception:
                    pass
        if last_percentage is not None:
            try:
                progress_to_write = int(round(float(last_percentage)))
            except Exception:
                progress_to_write = last_progress if last_progress is not None else 0
        else:
            progress_to_write = int(last_progress) if last_progress is not None else 0

        update_status(
            output_dir.parent,
            "stopped",
            progress=progress_to_write,
            stage="training",
            message="⏸️ Training stopped by user.",
            stopped_stage="training",
            stopped_step=trainer.stop_reason if isinstance(trainer.stop_reason, int) else None,
            stopped_percentage=last_percentage,
        )
        if stop_flag.exists():
            stop_flag.unlink()
        return trainer.stop_reason

    # Only run export and completion logic if not stopped
    logger.info("Training complete, exporting final checkpoint...")
    update_status(
        output_dir.parent, "processing", stage="training", stage_progress=100,
        message="✅ Gsplat training complete",
        timing={"start": gsplat_start, "end": gsplat_end, "elapsed": gsplat_end - gsplat_start}
    )
    update_status(
        output_dir.parent, "completed", progress=90, stage="training", message="Training done"
    )
    time.sleep(1.5)
    update_status(output_dir.parent, "processing", stage="export", stage_progress=10, message="📦 Preparing export of final artifacts...")
    ckpt_dir = output_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if ckpts:
        latest = ckpts[-1]
        logger.info(f"Exporting checkpoint: {latest}")
        update_status(output_dir.parent, "processing", stage="export", stage_progress=40, message="📝 Exporting .splat file...")
        # Allow stop requests to interrupt export
        if stop_flag.exists():
            update_status(output_dir.parent, "stopped", progress=0, stop_requested=True, stage="export", message="⏸️ Processing stopped by user before export.", stopped_stage="export")
            try:
                stop_flag.unlink()
            except Exception:
                pass
            return None
        _export_with_gsplat(latest, output_dir)
        update_status(output_dir.parent, "processing", stage="export", stage_progress=100, message="✅ Export complete")
    else:
        logger.info("No saved checkpoints found; skipping checkpoint export")

    if stop_flag.exists():
        stop_flag.unlink()

    return trainer.stop_reason


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Worker")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("--data-dir", default="/data/projects", help="Data directory")
    parser.add_argument("--params", type=json.loads, default="{}", help="Training parameters as JSON")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "modified"], 
                        help="Training mode: baseline or modified (with step 200 optimization)")

    args = parser.parse_args()

    # Merge mode into params
    params = args.params
    if "mode" not in params:
        params["mode"] = args.mode

    project_dir = Path(args.data_dir) / args.project_id
    image_dir = project_dir / "images"
    output_dir = project_dir / "outputs"


    # Configure file logging per project
    try:
        logs_file = project_dir / "processing.log"
        logs_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info("Initialized project log file: %s", logs_file)
    except Exception as e:
        logger.warning(f"Failed to initialize project log file: {e}")

    stop_reason = None
    try:
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        stage = params.get("stage", "full")
        # Respect resume flag passed in params
        resume = False
        try:
            resume = bool(params.get("resume", False)) if params else False
        except Exception:
            resume = False
        
        if stage == "colmap_only":
            update_status(project_dir, "processing", progress=1, stage="colmap", message="🚀 Starting COLMAP structure-from-motion pipeline...")
            colmap_dir = run_colmap(image_dir, output_dir, params)
            logger.info("COLMAP completed")
            # Mark COLMAP as completed for frontend tick/green
            update_status(project_dir, "completed", progress=100, stage="colmap", message="✅ Sparse 3D reconstruction complete!")
            time.sleep(1.5)
            stop_reason = None
        elif stage == "train_only":
            # Always use outputs/sparse/0 if it exists, else outputs/sparse
            colmap_dir = output_dir / "sparse"
            if (colmap_dir / "0").exists():
                colmap_dir = colmap_dir / "0"
            elif not (colmap_dir).exists():
                raise RuntimeError("Sparse model not found. Run COLMAP first.")
            msg = "🎯 Starting Gaussian Splatting training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg)
            stop_reason = run_gsplat_training(image_dir, colmap_dir, output_dir, params, resume=resume)
        else:
            # Full pipeline
            update_status(project_dir, "processing", progress=1, stage="colmap", message="🚀 Starting full pipeline - Running COLMAP structure-from-motion...")
            colmap_dir = run_colmap(image_dir, output_dir, params)
            logger.info("COLMAP completed")

            # Always use outputs/sparse/0 if it exists, else outputs/sparse
            if (colmap_dir / "0").exists():
                colmap_dir = colmap_dir / "0"
            msg = "🎯 Starting Gaussian Splatting training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg)
            stop_reason = run_gsplat_training(image_dir, colmap_dir, output_dir, params, resume=resume)

        if stop_reason:
            # If stopped, set status to 'stopped' and include step info in message
            final_status = "stopped"
            # Try to infer stopped_stage and stopped_step
            stopped_stage = "training"
            stopped_step = stop_reason if isinstance(stop_reason, int) else None
            final_message = f"⏸️ Processing stopped by user at step {stop_reason}."
            # Read last known progress/percentage to preserve accurate overall progress
            status_path = Path(project_dir) / "status.json"
            last_progress = None
            last_percentage = None
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        sd = json.load(f)
                        last_progress = sd.get("progress")
                        last_percentage = sd.get("percentage")
                except Exception:
                    pass
            if last_percentage is not None:
                try:
                    progress_to_write = int(round(float(last_percentage)))
                except Exception:
                    progress_to_write = last_progress if last_progress is not None else 0
            else:
                progress_to_write = int(last_progress) if last_progress is not None else 0

            update_status(project_dir, final_status, progress=progress_to_write, stop_requested=True, stage=stopped_stage, message=final_message, stopped_stage=stopped_stage, stopped_step=stopped_step, stopped_percentage=last_percentage)
            logger.info(f"Pipeline stopped by user at step {stop_reason}")
        else:
            # Always check status.json for stop_requested before setting completed
            status_path = Path(project_dir) / "status.json"
            stop_requested_flag = False
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        stop_requested_flag = json.load(f).get("stop_requested", False)
                except Exception:
                    stop_requested_flag = False
            if stop_requested_flag:
                final_status = "stopped"
                final_message = "⏸️ Processing stopped by user."
                status_path = Path(project_dir) / "status.json"
                last_progress = None
                last_percentage = None
                if status_path.exists():
                    try:
                        with open(status_path) as f:
                            sd = json.load(f)
                            last_progress = sd.get("progress")
                            last_percentage = sd.get("percentage")
                    except Exception:
                        pass
                if last_percentage is not None:
                    try:
                        progress_to_write = int(round(float(last_percentage)))
                    except Exception:
                        progress_to_write = last_progress if last_progress is not None else 0
                else:
                    progress_to_write = int(last_progress) if last_progress is not None else 0

                update_status(project_dir, final_status, progress=progress_to_write, stop_requested=True, stage="training", message=final_message, stopped_percentage=last_percentage)
                logger.info("Pipeline stopped by user (detected stop_requested)")
            else:
                final_status = "completed"
                if stage == "colmap_only":
                    final_message = "✅ Sparse 3D reconstruction complete! Ready for training."
                    final_stage = "colmap"
                else:
                    final_message = "🎉 Processing complete! Your 3D Gaussian Splat is ready to view."
                    final_stage = "export"
                update_status(project_dir, final_status, progress=100, stop_requested=False, stage=final_stage, message=final_message)
                if stage == "train_only" or stage == "full":
                    logger.info("Training completed")
                logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        # If a stop was requested, still set status to 'stopped' for the frontend
        if stop_reason:
            final_status = "stopped"
            stopped_stage = "training"
            stopped_step = stop_reason if isinstance(stop_reason, int) else None
            final_message = f"⏸️ Processing stopped by user at step {stop_reason}."
            # Preserve last known progress/percentage
            status_path = Path(project_dir) / "status.json"
            last_progress = None
            last_percentage = None
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        sd = json.load(f)
                        last_progress = sd.get("progress")
                        last_percentage = sd.get("percentage")
                except Exception:
                    pass
            if last_percentage is not None:
                try:
                    progress_to_write = int(round(float(last_percentage)))
                except Exception:
                    progress_to_write = last_progress if last_progress is not None else 0
            else:
                progress_to_write = int(last_progress) if last_progress is not None else 0

            update_status(project_dir, final_status, progress=progress_to_write, stop_requested=True, stage=stopped_stage, message=final_message, stopped_stage=stopped_stage, stopped_step=stopped_step, stopped_percentage=last_percentage)
            logger.info(f"Pipeline stopped by user at step {stop_reason}")
        else:
            update_status(project_dir, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
