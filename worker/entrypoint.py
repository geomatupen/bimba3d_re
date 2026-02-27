#!/usr/bin/env python3
"""
Docker worker entrypoint for Gaussian Splatting pipeline.
Runs COLMAP + faithful gsplat training (with research hooks) + export.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import time

from .image_resize import prepare_training_images, normalize_max_size
from .colmap_loader import COLMAPDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENGINE_SUBDIR = "engines"
BEST_SPARSE_META = ".best_sparse_selection.json"


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
    device: str = None,
    engine: str = None,
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
        if device is not None:
            data["device"] = device
        if engine is not None:
            data["engine"] = engine
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


def write_metrics(project_dir: Path, metrics: dict, engine: str | None = None):
    """Write training metrics to metrics.json (engine-specific + legacy root)."""
    output_root = project_dir / "outputs"
    targets: list[Path] = []
    if engine:
        targets.append(output_root / ENGINE_SUBDIR / engine)
    targets.append(output_root)

    for root in targets:
        metrics_file = root / "metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            temp_file = metrics_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            temp_file.replace(metrics_file)
        except Exception as e:
            logger.error(f"Failed to write metrics for {root}: {e}")


def _ensure_symlink(source: Path, link_path: Path):
    """Create or replace a symlink pointing to source."""
    try:
        source_path = Path(source).resolve()
    except Exception:
        source_path = Path(source)
    link_path = Path(link_path)

    if link_path.exists() or link_path.is_symlink():
        try:
            existing_target = link_path.resolve()
        except Exception:
            existing_target = None
        if existing_target and existing_target == source_path:
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink(missing_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())


def _compute_sparse_signature(colmap_dir: Path) -> str:
    """Return a lightweight signature for cache validation."""
    colmap_dir = Path(colmap_dir)
    for name in ("cameras.bin", "cameras.txt"):
        candidate = colmap_dir / name
        if candidate.exists():
            stat = candidate.stat()
            return f"{stat.st_mtime_ns}-{stat.st_size}"
    # Fallback if no expected files are present
    return f"missing-{hash(colmap_dir)}"


def _convert_to_pinhole_params(model: str, params: list[float]) -> tuple[float, float, float, float]:
    """Map various COLMAP camera models to PINHOLE intrinsics."""
    if len(params) < 3:
        raise ValueError(f"Camera model {model} has too few parameters")

    model = model.upper()
    if model == "PINHOLE":
        if len(params) < 4:
            raise ValueError("PINHOLE camera missing parameters")
        return params[0], params[1], params[2], params[3]

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f = params[0]
        cx = params[1]
        cy = params[2]
        return f, f, cx, cy

    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        f = params[0]
        cx = params[1]
        cy = params[2]
        return f, f, cx, cy

    if model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        if len(params) < 4:
            raise ValueError(f"Camera model {model} missing fx/fy/cx/cy")
        return params[0], params[1], params[2], params[3]

    if model == "FOV":
        f = params[0]
        cx = params[1]
        cy = params[2]
        return f, f, cx, cy

    raise ValueError(f"Unsupported COLMAP camera model {model}")


def _rewrite_cameras_file_as_pinhole(cameras_txt: Path) -> tuple[bool, set[str]]:
    """Rewrite cameras.txt so every entry becomes PINHOLE."""
    models_seen: set[str] = set()
    converted = False
    updated_lines: list[str] = []

    with open(cameras_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                updated_lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) < 5:
                updated_lines.append(line)
                continue
            camera_id, model, width, height = parts[:4]
            params = list(map(float, parts[4:]))
            models_seen.add(model)
            fx, fy, cx, cy = _convert_to_pinhole_params(model, params)
            new_line = f"{camera_id} PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n"
            updated_lines.append(new_line)
            if model != "PINHOLE":
                converted = True

    if converted:
        with open(cameras_txt, "w", encoding="utf-8") as handle:
            handle.writelines(updated_lines)

    return converted, models_seen


def _prepare_pinhole_sparse_for_litegs(colmap_dir: Path, output_dir: Path) -> Path:
    """Ensure LiteGS sees a PINHOLE-only sparse reconstruction."""
    colmap_dir = Path(colmap_dir)
    cache_root = Path(output_dir) / "litegs" / "cache" / "pinhole_sparse"
    cache_root.mkdir(parents=True, exist_ok=True)

    signature = _compute_sparse_signature(colmap_dir)
    cached_dir = cache_root / colmap_dir.name
    cache_sig_file = cached_dir / ".source_signature"
    if cached_dir.exists() and cache_sig_file.exists():
        try:
            if cache_sig_file.read_text(encoding="utf-8").strip() == signature:
                logger.info("Using cached PINHOLE sparse model at %s", cached_dir)
                return cached_dir
        except Exception:
            logger.debug("Failed to read LiteGS cache signature at %s", cache_sig_file)

    with tempfile.TemporaryDirectory(dir=cache_root, prefix="litegs_sparse_") as tmp_parent:
        txt_dir = Path(tmp_parent) / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        _run_cmd_with_retry([
            "colmap", "model_converter",
            "--input_path", str(colmap_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ])

        cameras_txt = txt_dir / "cameras.txt"
        if not cameras_txt.exists():
            raise FileNotFoundError(f"COLMAP cameras.txt missing in {txt_dir}")

        converted, models = _rewrite_cameras_file_as_pinhole(cameras_txt)
        if not converted:
            logger.info("LiteGS sparse already PINHOLE-compatible (models: %s)", ", ".join(sorted(models)))
            return colmap_dir

        if cached_dir.exists():
            shutil.rmtree(cached_dir, ignore_errors=True)
        cached_dir.mkdir(parents=True, exist_ok=True)

        _run_cmd_with_retry([
            "colmap", "model_converter",
            "--input_path", str(txt_dir),
            "--output_path", str(cached_dir),
            "--output_type", "BIN",
        ])
        cache_sig_file.write_text(signature, encoding="utf-8")
        logger.info(
            "Prepared PINHOLE sparse model for LiteGS at %s (source models: %s)",
            cached_dir,
            ", ".join(sorted(models)),
        )
        return cached_dir


def _find_latest_litegs_checkpoint(model_root: Path) -> Path | None:
    ckpt_dir = Path(model_root) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("chkpnt*.pth"))
    return candidates[-1] if candidates else None


def _patch_litegs_opacity_decay(decay_rate: float):
    """Monkey patch LiteGS opacity decay to honor user shrink factor."""
    if decay_rate is None:
        return
    try:
        import litegs.training.densify as litegs_densify
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("LiteGS opacity patch skipped: %s", exc)
        return

    current_reset = litegs_densify.DensityControllerOfficial.reset_opacity
    base_reset = getattr(current_reset, "_bimba_original_reset", current_reset)
    if getattr(current_reset, "_bimba_decay_rate", None) == decay_rate:
        return

    def patched(self, optimizer, epoch):  # noqa: ANN001 - signature fixed by upstream
        if self.densify_params.opacity_reset_mode == 'decay':
            xyz, scale, rot, sh_0, sh_rest, opacity = self._get_params_from_optimizer(optimizer)

            def inverse_sigmoid(x):
                return torch.log(x / (1 - x))

            actived_opacities = opacity.sigmoid()
            opacity.data = inverse_sigmoid((actived_opacities * decay_rate).clamp_min(1.0 / 128))
            optimizer.state.clear()
        else:
            return base_reset(self, optimizer, epoch)

    patched._bimba_original_reset = base_reset  # type: ignore[attr-defined]
    patched._bimba_decay_rate = decay_rate      # type: ignore[attr-defined]
    litegs_densify.DensityControllerOfficial.reset_opacity = patched


def _export_litegs_outputs(model_root: Path, output_dir: Path, colmap_dir: Path, training_summary: dict):
    """Convert LiteGS point cloud outputs into splat artifacts and metadata."""
    from litegs.io_manager import load_ply  # pylint: disable=import-error
    import torch
    from gsplat.exporter import export_splats

    ply_path = Path(model_root) / "point_cloud" / "finish" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"LiteGS output missing: {ply_path}")

    sh_degree = int(training_summary.get("sh_degree", 3))
    xyz, scale, rot, sh0, sh_rest, opacity = load_ply(str(ply_path), sh_degree)

    means = torch.from_numpy(xyz.T).float().contiguous()
    scales = torch.from_numpy(scale.T).float().contiguous()
    quats = torch.from_numpy(rot.T).float().contiguous()
    opacities = torch.from_numpy(opacity.T).float().squeeze(-1).contiguous()
    sh0_tensor = torch.from_numpy(sh0).float().permute(2, 0, 1).contiguous()
    sh_rest_tensor = torch.from_numpy(sh_rest).float().permute(2, 0, 1).contiguous()

    splat_path = Path(output_dir) / "splats.splat"
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0_tensor,
        shN=sh_rest_tensor,
        format="splat",
        save_to=str(splat_path),
    )
    ply_export = Path(output_dir) / "splats.ply"
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0_tensor,
        shN=sh_rest_tensor,
        format="ply",
        save_to=str(ply_export),
    )

    metadata = {
        "version": "1.0",
        "type": "gaussian_splats",
        "training_engine": "litegs",
        "colmap_model": str(colmap_dir),
        "training_config": {
            "engine": "litegs",
            "target_primitives": training_summary.get("target_primitives"),
            "alpha_shrink": training_summary.get("alpha_shrink"),
            "iterations": training_summary.get("iterations"),
        },
    }
    metadata_path = Path(output_dir) / "metadata.json"
    tmp_metadata = metadata_path.with_suffix('.tmp')
    with open(tmp_metadata, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    tmp_metadata.replace(metadata_path)


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


def _persist_best_sparse_choice(sparse_root: Path, best_entry: dict, candidates: list[dict]):
    """Record the strongest sparse candidate so other components can reuse it."""
    sparse_root = Path(sparse_root)
    meta_file = sparse_root / BEST_SPARSE_META

    payload = {
        "relative_path": best_entry.get("relative_path"),
        "images": int(best_entry.get("images", 0) or 0),
        "points": int(best_entry.get("points", 0) or 0),
        "timestamp": time.time(),
        "candidates": [
            {
                "relative_path": item.get("relative_path"),
                "images": item.get("images"),
                "points": item.get("points"),
                "label": item.get("label"),
            }
            for item in candidates
        ],
    }

    try:
        tmp_meta = meta_file.with_suffix(meta_file.suffix + ".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_meta.replace(meta_file)
        logger.info(
            "Recorded best sparse reconstruction %s (%s images, %s points)",
            payload["relative_path"],
            payload["images"],
            payload["points"],
        )
    except Exception as exc:
        logger.debug("Failed to persist best sparse metadata at %s: %s", meta_file, exc)


def _evaluate_sparse_candidates(sparse_root: Path, image_dir: Path) -> list[dict]:
    sparse_root = Path(sparse_root)
    candidates: list[Path] = []
    if (sparse_root / "cameras.bin").exists():
        candidates.append(sparse_root)

    try:
        children = sorted(p for p in sparse_root.iterdir() if p.is_dir())
    except Exception as exc:
        logger.warning("Failed to enumerate sparse subdirectories under %s: %s", sparse_root, exc)
        children = []

    for child in children:
        if (child / "cameras.bin").exists():
            candidates.append(child)

    summaries: list[dict] = []
    for candidate in candidates:
        num_images = -1
        num_points = -1
        try:
            dataset = COLMAPDataset(candidate, image_dir)
            num_images = len(dataset)
            num_points = len(dataset.points)
            logger.info(
                "Sparse candidate %s contains %d images and %d points",
                candidate,
                num_images,
                num_points,
            )
        except Exception as exc:
            logger.warning("Failed to load sparse candidate %s: %s", candidate, exc)

        try:
            rel_path = os.path.relpath(candidate, sparse_root)
        except Exception:
            rel_path = candidate.name

        if rel_path in {".", ""}:
            label = "root"
            rel_path = "."
        else:
            label = Path(rel_path).name

        summaries.append(
            {
                "relative_path": rel_path,
                "label": label,
                "images": max(num_images, 0),
                "points": max(num_points, 0),
            }
        )

    return summaries


def _select_best_sparse_model(sparse_root: Path, image_dir: Path, preference: str | None = None) -> Path:
    """Return the sparse reconstruction honoring user preference when possible."""
    sparse_root = Path(sparse_root)
    if not sparse_root.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_root}")

    summaries = _evaluate_sparse_candidates(sparse_root, image_dir)
    if not summaries:
        raise RuntimeError(f"No COLMAP reconstructions found under {sparse_root}")

    def _score(entry: dict) -> tuple[int, int]:
        return int(entry.get("images", 0) or 0), int(entry.get("points", 0) or 0)

    best_entry = max(summaries, key=_score)
    _persist_best_sparse_choice(sparse_root, best_entry, summaries)

    chosen_entry = best_entry
    pref = (preference or "best").strip()
    if pref and pref.lower() != "best":
        norm = pref.lower()
        for entry in summaries:
            rel = (entry.get("relative_path") or "").lower()
            label = (entry.get("label") or "").lower()
            if norm in {rel, label, Path(rel or ".").name.lower()}:
                chosen_entry = entry
                logger.info(
                    "Using user-selected sparse candidate '%s' (%s images, %s points)",
                    entry.get("relative_path"),
                    entry.get("images"),
                    entry.get("points"),
                )
                break
        else:
            logger.warning(
                "Sparse preference '%s' not found; defaulting to best candidate %s",
                preference,
                best_entry.get("relative_path"),
            )
    else:
        logger.info(
            "Using best sparse candidate %s (%s images, %s points)",
            best_entry.get("relative_path"),
            best_entry.get("images"),
            best_entry.get("points"),
        )

    rel_path = chosen_entry.get("relative_path") or "."
    target = (sparse_root / rel_path).resolve()
    base = sparse_root.resolve()
    if target != base and base not in target.parents:
        raise RuntimeError(f"Unsafe sparse selection resolved outside project: {target}")
    return target


def _get_engine_output_dir(base_output_dir: Path, engine: str) -> Path:
    """Return (and ensure) the engine-specific outputs directory."""
    engine_dir = base_output_dir / ENGINE_SUBDIR / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    return engine_dir


def _sync_engine_artifact_dirs(base_output_dir: Path, engine_output_dir: Path, subdirs: tuple[str, ...]):
    """Move legacy artifact folders into the engine scope and remove old copies."""
    for name in subdirs:
        legacy_dir = base_output_dir / name
        engine_dir = engine_output_dir / name

        if legacy_dir.exists():
            if legacy_dir.is_symlink():
                legacy_dir.unlink(missing_ok=True)
            elif legacy_dir.is_dir():
                engine_dir.mkdir(parents=True, exist_ok=True)
                for item in legacy_dir.iterdir():
                    target = engine_dir / item.name
                    if target.exists():
                        continue
                    try:
                        item.rename(target)
                    except Exception:
                        shutil.move(str(item), str(target))
                shutil.rmtree(legacy_dir, ignore_errors=True)
            else:
                engine_dir.parent.mkdir(parents=True, exist_ok=True)
                target = engine_dir / legacy_dir.name
                try:
                    legacy_dir.rename(target)
                except Exception:
                    shutil.move(str(legacy_dir), str(target))

        engine_dir.mkdir(parents=True, exist_ok=True)


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
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    engine_name = "gsplat"
    engine_output_dir = _get_engine_output_dir(base_output_dir, engine_name)
    _sync_engine_artifact_dirs(base_output_dir, engine_output_dir, ("checkpoints", "previews", "snapshots"))
    
    # Extract parameters
    p = params or {}
    mode = p.get("mode", "baseline")  # "baseline" or "modified"
    max_steps = p.get("max_steps", 300)
    project_dir = base_output_dir.parent
    training_image_dir = image_dir
    images_max_size = normalize_max_size(p.get("images_max_size"))

    stop_flag = project_dir / "stop_requested"

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
                project_dir,
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
            write_metrics(project_dir, {
                "step": step,
                "loss": loss,
                "progress": progress,
            }, engine=engine_name)
    
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
    if images_max_size:
        init_message = (
            f"{init_message}\n📐 Input images limited to ≤ {images_max_size}px"
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

    trainer = GsplatTrainer(
        image_dir=training_image_dir,
        colmap_dir=colmap_dir,
        output_dir=engine_output_dir,
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
        tuning_results_file = engine_output_dir / "adaptive_tuning_results.json"
        if tuning_results_file.exists():
            with open(tuning_results_file) as f:
                tuning_data = json.load(f)
                final_metrics = tuning_data.get("final_evaluation", {})
                
                # Save to metadata.json for easy access
                metadata_file = engine_output_dir / "metadata.json"
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
        metrics_candidates = [
            engine_output_dir / "metrics.json",
            base_output_dir / "metrics.json",
        ]
        metrics_file = None
        for candidate in metrics_candidates:
            if candidate.exists():
                metrics_file = candidate
                break
        last_progress = None
        last_percentage = None
        last_step = None
        if metrics_file and metrics_file.exists():
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
            status_path = project_dir / "status.json"
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
            project_dir,
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
        project_dir, "processing", stage="training", stage_progress=100,
        message="✅ Gsplat training complete",
        timing={"start": gsplat_start, "end": gsplat_end, "elapsed": gsplat_end - gsplat_start}
    )
    update_status(
        project_dir, "completed", progress=90, stage="training", message="Training done"
    )
    time.sleep(1.5)
    update_status(project_dir, "processing", stage="export", stage_progress=10, message="📦 Preparing export of final artifacts...")
    ckpt_dir = engine_output_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if ckpts:
        latest = ckpts[-1]
        logger.info(f"Exporting checkpoint: {latest}")
        update_status(project_dir, "processing", stage="export", stage_progress=40, message="📝 Exporting .splat file...")
        # Allow stop requests to interrupt export
        if stop_flag.exists():
            update_status(project_dir, "stopped", progress=0, stop_requested=True, stage="export", message="⏸️ Processing stopped by user before export.", stopped_stage="export")
            try:
                stop_flag.unlink()
            except Exception:
                pass
            return None
        _export_with_gsplat(latest, engine_output_dir)
        update_status(project_dir, "processing", stage="export", stage_progress=100, message="✅ Export complete")
    else:
        logger.info("No saved checkpoints found; skipping checkpoint export")

    if stop_flag.exists():
        stop_flag.unlink()

    return trainer.stop_reason


def run_litegs_training(image_dir: Path, colmap_dir: Path, output_dir: Path, params: dict, resume: bool = False):
    """Run LiteGS training pipeline and export artifacts."""
    project_dir = output_dir.parent
    stop_flag = project_dir / "stop_requested"
    engine_label = "LiteGS"

    # Detect device availability for status updates
    try:
        import torch
        device_label = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch should exist but guard anyway
        device_label = "cuda"

    status_message = f"🎯 Starting {engine_label} training..."
    update_status(
        project_dir,
        "processing",
        progress=55,
        stage="training",
        stage_progress=5,
        message=status_message,
        device=device_label,
        engine="litegs",
    )

    if stop_flag.exists():
        update_status(
            project_dir,
            "stopped",
            progress=55,
            stage="training",
            message="⏸️ Processing stopped before LiteGS training.",
            stop_requested=True,
            stopped_stage="training",
        )
        try:
            stop_flag.unlink()
        except Exception:
            pass
        return 0

    try:
        import litegs  # pylint: disable=import-error
        from litegs import config as litegs_config  # pylint: disable=import-error
        from litegs import training as litegs_training  # pylint: disable=import-error
    except Exception as exc:
        logger.error("LiteGS import failed: %s", exc)
        update_status(project_dir, "failed", error=f"LiteGS not installed: {exc}")
        raise

    dataset_root = output_dir / "litegs" / "dataset"
    model_root = output_dir / "litegs" / "artifacts"
    dataset_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    _ensure_symlink(Path(image_dir), dataset_root / "images")
    colmap_sparse_root = Path(colmap_dir)

    processed_sparse = _prepare_pinhole_sparse_for_litegs(colmap_sparse_root, output_dir)

    litegs_sparse_root = dataset_root / "sparse"
    if litegs_sparse_root.is_symlink():
        litegs_sparse_root.unlink()
    litegs_sparse_root.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(processed_sparse, litegs_sparse_root / "0")

    lp, op, pp, dp = litegs_config.get_default_arg()
    lp.source_path = str(dataset_root)
    lp.model_path = str(model_root)
    if hasattr(lp, "images"):
        lp.images = "images"

    training_summary = {
        "iterations": getattr(op, "iterations", None),
        "target_primitives": getattr(dp, "target_primitives", None),
        "alpha_shrink": None,
        "sh_degree": getattr(lp, "sh_degree", 3),
    }

    user_steps = params.get("max_steps")
    if user_steps is not None:
        try:
            op.iterations = max(1, int(user_steps))
            training_summary["iterations"] = op.iterations
        except Exception:
            logger.warning("Invalid LiteGS max_steps override: %s", user_steps)

    target_override = params.get("litegs_target_primitives")
    if target_override is not None:
        try:
            dp.target_primitives = max(1, int(target_override))
        except Exception:
            logger.warning("Invalid LiteGS target primitive override: %s", target_override)
    training_summary["target_primitives"] = getattr(dp, "target_primitives", None)

    alpha_shrink = params.get("litegs_alpha_shrink", 0.95)
    try:
        alpha_shrink = float(alpha_shrink)
    except Exception:
        alpha_shrink = 0.95
    if alpha_shrink <= 0:
        alpha_shrink = 0.95
    training_summary["alpha_shrink"] = alpha_shrink
    _patch_litegs_opacity_decay(alpha_shrink)

    start_checkpoint = None
    if resume:
        ckpt_path = _find_latest_litegs_checkpoint(model_root)
        if ckpt_path:
            start_checkpoint = str(ckpt_path)
            logger.info("Resuming LiteGS from %s", ckpt_path)
        else:
            logger.info("LiteGS resume requested but no checkpoints found; starting fresh")

    litegs_start = time.time()
    try:
        litegs_training.start(lp, op, pp, dp, [], [], [], start_checkpoint)
    except Exception as exc:
        logger.error("LiteGS training failed: %s", exc, exc_info=True)
        update_status(project_dir, "failed", error=str(exc), stage="training", message=str(exc))
        raise

    if stop_flag.exists():
        logger.info("Stop requested after LiteGS training finished")
        return 0

    update_status(
        project_dir,
        "processing",
        stage="training",
        stage_progress=85,
        message=f"✅ {engine_label} training complete",
        device=device_label,
    )

    _export_litegs_outputs(model_root, output_dir, colmap_sparse_root, training_summary)

    litegs_end = time.time()
    update_status(
        project_dir,
        "processing",
        stage="export",
        stage_progress=100,
        message="✅ LiteGS export complete",
        timing={"start": litegs_start, "end": litegs_end, "elapsed": litegs_end - litegs_start},
        device=device_label,
    )

    return None


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Worker")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("--data-dir", default="/data/projects", help="Data directory")
    parser.add_argument("--params", type=json.loads, default="{}", help="Training parameters as JSON")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "modified"], 
                        help="Training mode: baseline or modified (with step 200 optimization)")

    args = parser.parse_args()

    # Merge mode into params
    params = dict(args.params or {})
    if "mode" not in params:
        params["mode"] = args.mode
    engine = params.get("engine", "gsplat")
    if engine not in {"gsplat", "litegs"}:
        engine = "gsplat"
    params["engine"] = engine

    project_dir = Path(args.data_dir) / args.project_id
    image_dir = project_dir / "images"
    output_dir = project_dir / "outputs"
    sparse_preference = params.get("sparse_preference") if isinstance(params, dict) else None
    if isinstance(sparse_preference, str):
        sparse_preference = sparse_preference.strip() or None


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

        active_image_dir = image_dir
        resize_stats = None
        images_max_size = normalize_max_size(params.get("images_max_size"))
        if images_max_size:
            resize_stage = "training" if stage == "train_only" else "colmap"
            logger.info("Preparing resized image set (≤%d px) for project %s", images_max_size, args.project_id)
            update_status(
                project_dir,
                "processing",
                progress=5 if resize_stage == "colmap" else 55,
                stage=resize_stage,
                stage_progress=2 if resize_stage == "training" else 5,
                message=f"📐 Resizing input images to ≤ {images_max_size}px...",
                engine=engine,
            )
            try:
                active_image_dir, resize_stats = prepare_training_images(image_dir, project_dir, images_max_size)
                logger.info("Prepared resized image set at %s (%s)", active_image_dir, resize_stats)
                update_status(
                    project_dir,
                    "processing",
                    stage=resize_stage,
                    stage_progress=6 if resize_stage == "training" else 8,
                    message=f"✅ Image set ready at ≤ {images_max_size}px",
                )
                colmap_cfg = dict(params.get("colmap") or {})
                colmap_cfg.setdefault("max_image_size", images_max_size)
                params["colmap"] = colmap_cfg
            except Exception as exc:
                logger.warning("Failed to prepare resized images; using originals. Error: %s", exc)
                active_image_dir = image_dir
        
        if stage == "colmap_only":
            update_status(project_dir, "processing", progress=1, stage="colmap", message="🚀 Starting COLMAP structure-from-motion pipeline...", engine=engine)
            colmap_dir = run_colmap(active_image_dir, output_dir, params)
            logger.info("COLMAP completed")
            try:
                _select_best_sparse_model(output_dir / "sparse", active_image_dir, sparse_preference)
            except Exception as exc:
                logger.warning("Failed to evaluate sparse candidates after COLMAP: %s", exc)
            # Mark COLMAP as completed for frontend tick/green
            update_status(project_dir, "completed", progress=100, stage="colmap", message="✅ Sparse 3D reconstruction complete!")
            time.sleep(1.5)
            stop_reason = None
        elif stage == "train_only":
            sparse_root = output_dir / "sparse"
            if not sparse_root.exists():
                raise RuntimeError("Sparse model not found. Run COLMAP first.")
            colmap_dir = _select_best_sparse_model(sparse_root, active_image_dir, sparse_preference)
            trainer_label = "LiteGS" if engine == "litegs" else "Gaussian Splatting"
            msg = f"🎯 Starting {trainer_label} training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg, engine=engine)
            if engine == "litegs":
                stop_reason = run_litegs_training(active_image_dir, colmap_dir, output_dir, params, resume=resume)
            else:
                stop_reason = run_gsplat_training(active_image_dir, colmap_dir, output_dir, params, resume=resume)
        else:
            # Full pipeline
            update_status(project_dir, "processing", progress=1, stage="colmap", message="🚀 Starting full pipeline - Running COLMAP structure-from-motion...", engine=engine)
            colmap_dir = run_colmap(active_image_dir, output_dir, params)
            logger.info("COLMAP completed")

            colmap_dir = _select_best_sparse_model(output_dir / "sparse", active_image_dir, sparse_preference)
            trainer_label = "LiteGS" if engine == "litegs" else "Gaussian Splatting"
            msg = f"🎯 Starting {trainer_label} training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg, engine=engine)
            if engine == "litegs":
                stop_reason = run_litegs_training(active_image_dir, colmap_dir, output_dir, params, resume=resume)
            else:
                stop_reason = run_gsplat_training(active_image_dir, colmap_dir, output_dir, params, resume=resume)

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
