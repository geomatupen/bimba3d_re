#!/usr/bin/env python3
"""
Docker worker entrypoint for Gaussian Splatting pipeline.
Runs COLMAP + faithful gsplat training (with research hooks) + export.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
import time

import numpy as np

from .image_resize import prepare_training_images, normalize_max_size
from .colmap_loader import COLMAPDataset, qvec2rotmat, read_images_binary, read_points3D_binary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLMAP_EXE = (os.getenv("COLMAP_EXE") or "colmap").strip() or "colmap"

ENGINE_SUBDIR = "engines"
SUPPORTED_ENGINES = {"gsplat", "litegs"}
ENGINE_LABELS = {
    "gsplat": "Gaussian Splatting",
    "litegs": "LiteGS",
}
BEST_SPARSE_META = ".best_sparse_selection.json"
SPARSE_IMAGE_MEMBERSHIP_META = ".sparse_image_membership.json"


def _colmap_cmd(*args: str) -> list[str]:
    return [COLMAP_EXE, *args]


def _prepare_subprocess_command(cmd: list[str]) -> tuple[list[str] | str, bool]:
    if os.name == "nt" and cmd:
        resolved = list(cmd)
        exe = str(resolved[0])
        exe_lower = exe.lower()
        if exe_lower.endswith("colmap.exe"):
            exe_path = Path(exe)
            candidates = [
                exe_path.with_name("COLMAP.bat"),
                exe_path.parent.parent / "COLMAP.bat",
            ]
            for candidate in candidates:
                if candidate.exists():
                    resolved[0] = str(candidate)
                    break
        if str(resolved[0]).lower().endswith((".bat", ".cmd")):
            return subprocess.list2cmdline(resolved), True
        return resolved, False
    return cmd, False


def _read_registered_image_names(images_bin_path: Path) -> list[str]:
    """Return sorted registered image names from a COLMAP images.bin file."""
    try:
        images = read_images_binary(images_bin_path)
    except Exception:
        return []

    names: list[str] = []
    for entry in images.values():
        name = entry.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    names.sort()
    return names


def _persist_sparse_image_membership(
    sparse_root: Path,
    candidate_membership: list[dict],
):
    """Persist sparse candidate -> image-name mapping for debugging/verification."""
    sparse_root = Path(sparse_root)
    target = sparse_root / SPARSE_IMAGE_MEMBERSHIP_META
    payload = {
        "updated_at": time.time(),
        "candidates": candidate_membership,
    }
    try:
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        with open(tmp_target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_target.replace(target)
    except Exception as exc:
        logger.debug("Failed to persist sparse image membership at %s: %s", target, exc)


def _run_colmap_image_registration_pass(
    database_path: Path,
    image_dir: Path,
    reconstruction_dir: Path,
    mapper_refine_focal_length: bool,
    mapper_refine_principal_point: bool,
    mapper_refine_extra_params: bool,
) -> bool:
    """Try to register additional images into an existing sparse model.

    Returns True when post-registration commands completed without error.
    """
    logger.info("COLMAP: image_registrator pass for %s", reconstruction_dir)
    try:
        _run_cmd_with_retry(_colmap_cmd(
            "image_registrator",
            "--database_path", str(database_path),
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
        ))
        _run_cmd_with_retry(_colmap_cmd(
            "point_triangulator",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
            "--Mapper.ba_refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--Mapper.ba_refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--Mapper.ba_refine_extra_params", "1" if mapper_refine_extra_params else "0",
        ))
        _run_cmd_with_retry(_colmap_cmd(
            "bundle_adjuster",
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
            "--BundleAdjustment.refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--BundleAdjustment.refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--BundleAdjustment.refine_extra_params", "1" if mapper_refine_extra_params else "0",
        ))
        return True
    except Exception as exc:
        logger.warning("Post-mapper image registration pass failed for %s: %s", reconstruction_dir, exc)
        return False


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
            with open(status_file, 'r', encoding='utf-8') as f:
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
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        temp_file.replace(status_file)

    except Exception as e:
        logger.error(f"Failed to update status: {e}")


def write_metrics(project_dir: Path, metrics: dict, engine: str | None = None):
    """Write training metrics to engine-scoped metrics.json."""
    if not engine:
        logger.warning("Skipping metrics write without engine scope")
        return

    metrics_root = project_dir / "outputs" / ENGINE_SUBDIR / engine
    metrics_file = metrics_root / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        temp_file = metrics_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        temp_file.replace(metrics_file)
    except Exception as e:
        logger.error(f"Failed to write metrics for {metrics_root}: {e}")


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
        _run_cmd_with_retry(_colmap_cmd(
            "model_converter",
            "--input_path", str(colmap_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ))

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

        _run_cmd_with_retry(_colmap_cmd(
            "model_converter",
            "--input_path", str(txt_dir),
            "--output_path", str(cached_dir),
            "--output_type", "BIN",
        ))
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
            run_cmd, use_shell = _prepare_subprocess_command(cmd)
            res = subprocess.run(run_cmd, check=True, capture_output=True, text=True, shell=use_shell)
            if res.stdout:
                logger.info(res.stdout.strip())
            if res.stderr:
                logger.debug(res.stderr.strip())
            return
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            stdout = (e.stdout or "")
            last_err = e
            if os.name == "nt" and getattr(e, "returncode", None) in (3221225781, 3221225786):
                logger.error(
                    "COLMAP crashed on Windows (code=%s). If COLMAP_EXE points to colmap.exe, set it to COLMAP.bat instead; also ensure Microsoft Visual C++ 2015-2022 Redistributable is installed.",
                    getattr(e, "returncode", None),
                )
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


def _clear_sparse_outputs(sparse_dir: Path):
    """Remove previous sparse outputs so COLMAP starts from a clean slate."""
    sparse_dir = Path(sparse_dir)
    if not sparse_dir.exists():
        return
    for child in sparse_dir.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to remove old sparse artifact %s: %s", child, exc)


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
    membership_rows: list[dict] = []
    for candidate in candidates:
        num_images = -1
        num_points = -1
        image_names: list[str] = []
        try:
            dataset = COLMAPDataset(candidate, image_dir)
            num_images = len(dataset)
            num_points = len(dataset.points)
            image_names = _read_registered_image_names(candidate / "images.bin")
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
        membership_rows.append(
            {
                "relative_path": rel_path,
                "label": label,
                "images": max(num_images, 0),
                "image_names": image_names,
            }
        )

    _persist_sparse_image_membership(sparse_root, membership_rows)

    return summaries


def _normalize_merge_selection(raw_selection) -> list[str]:
    if not isinstance(raw_selection, list):
        return []
    normalized: list[str] = []
    for item in raw_selection:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        normalized.append("." if value in {".", "root"} else value)
    # preserve order while deduplicating
    seen: set[str] = set()
    unique: list[str] = []
    for value in normalized:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _build_sparse_merge_signature(sparse_root: Path, selected_rel_paths: list[str]) -> str:
    pieces: list[str] = []
    for rel in selected_rel_paths:
        folder = (sparse_root / rel).resolve()
        points_path = folder / "points3D.bin"
        images_path = folder / "images.bin"
        points_sig = "missing"
        images_sig = "missing"
        if points_path.exists():
            stat = points_path.stat()
            points_sig = f"{stat.st_mtime_ns}-{stat.st_size}"
        if images_path.exists():
            stat = images_path.stat()
            images_sig = f"{stat.st_mtime_ns}-{stat.st_size}"
        pieces.append(f"{rel}|{points_sig}|{images_sig}")
    digest = hashlib.sha1("\n".join(pieces).encode("utf-8")).hexdigest()
    return digest[:14]


def _write_points3d_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    with open(path, "wb") as handle:
        handle.write(int(len(xyz)).to_bytes(8, byteorder="little", signed=False))
        for idx, (point_xyz, point_rgb) in enumerate(zip(xyz, rgb), start=1):
            handle.write(int(idx).to_bytes(8, byteorder="little", signed=False))
            handle.write(np.asarray(point_xyz, dtype=np.float64).tobytes())
            rgb_uint8 = np.asarray(np.clip(point_rgb, 0, 255), dtype=np.uint8)
            handle.write(rgb_uint8.tobytes())
            handle.write(np.float64(0.0).tobytes())
            handle.write((0).to_bytes(8, byteorder="little", signed=False))


def _load_camera_centers_by_name(images_bin_path: Path) -> dict[str, np.ndarray]:
    images = read_images_binary(images_bin_path)
    centers: dict[str, np.ndarray] = {}
    for image_data in images.values():
        name = image_data.get("name")
        if not isinstance(name, str) or not name:
            continue
        qvec = np.asarray(image_data.get("qvec"), dtype=np.float64)
        tvec = np.asarray(image_data.get("tvec"), dtype=np.float64)
        if qvec.shape != (4,) or tvec.shape != (3,):
            continue
        rotation = qvec2rotmat(qvec)
        center = -rotation.T @ tvec
        if np.all(np.isfinite(center)):
            centers[name] = center
    return centers


def _read_images_binary_with_ids(path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(path, "rb") as handle:
        num_images = struct.unpack("Q", handle.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", handle.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", handle.read(32))
            tx, ty, tz = struct.unpack("ddd", handle.read(24))
            camera_id = struct.unpack("I", handle.read(4))[0]
            name_bytes = b""
            while True:
                ch = handle.read(1)
                if ch == b"\x00":
                    break
                if not ch:
                    raise ValueError("Unexpected EOF while reading image name")
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            num_points = struct.unpack("Q", handle.read(8))[0]
            # Skip x,y,point3D_id tuples
            handle.read(24 * num_points)
            entries.append(
                {
                    "image_id": int(image_id),
                    "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
                    "tvec": np.array([tx, ty, tz], dtype=np.float64),
                    "camera_id": int(camera_id),
                    "name": name,
                }
            )
    return entries


def _write_images_binary(path: Path, images: list[dict]):
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for idx, image in enumerate(images, start=1):
            qvec = np.asarray(image["qvec"], dtype=np.float64)
            tvec = np.asarray(image["tvec"], dtype=np.float64)
            name = str(image["name"])
            camera_id = int(image["camera_id"])

            handle.write(struct.pack("<I", idx))
            handle.write(struct.pack("<dddd", float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])))
            handle.write(struct.pack("<ddd", float(tvec[0]), float(tvec[1]), float(tvec[2])))
            handle.write(struct.pack("<I", camera_id))
            handle.write(name.encode("utf-8") + b"\x00")
            # No point2D tracks in merged views.
            handle.write(struct.pack("<Q", 0))


def _rotmat2qvec(rot: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion (qw, qx, qy, qz)."""
    r = np.asarray(rot, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    trace = float(np.trace(r))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12))
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12))
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12))
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / q_norm


def _transform_source_image_pose(
    qvec: np.ndarray,
    tvec: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform source world pose to anchor world frame and return (qvec, tvec) world-to-camera."""
    r_cw_src = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    r_wc_src = r_cw_src.T
    c_src = -r_wc_src @ np.asarray(tvec, dtype=np.float64)

    c_dst = (float(scale) * (rot @ c_src)) + trans
    r_wc_dst = rot @ r_wc_src
    r_cw_dst = r_wc_dst.T
    t_dst = -r_cw_dst @ c_dst
    q_dst = _rotmat2qvec(r_cw_dst)
    return q_dst, np.asarray(t_dst, dtype=np.float64)


def _camera_pose_rmse(overlap: list[str], source_images: dict[int, dict], anchor_centers: dict[str, np.ndarray], scale: float, rot: np.ndarray, trans: np.ndarray) -> float:
    errs: list[float] = []
    by_name = {entry.get("name"): entry for entry in source_images.values() if isinstance(entry, dict)}
    for name in overlap:
        src_entry = by_name.get(name)
        dst_center = anchor_centers.get(name)
        if not src_entry or dst_center is None:
            continue
        r_cw_src = qvec2rotmat(np.asarray(src_entry.get("qvec"), dtype=np.float64))
        c_src = -r_cw_src.T @ np.asarray(src_entry.get("tvec"), dtype=np.float64)
        c_est = (float(scale) * (rot @ c_src)) + trans
        errs.append(float(np.linalg.norm(c_est - dst_center)))
    if not errs:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errs))))


def _estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Estimate dst ≈ s * R @ src + t using Umeyama alignment."""
    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < 3 or src_pts.shape[1] != 3:
        return None

    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    src_var = float(np.sum(src_centered ** 2) / src_pts.shape[0])
    if src_var <= 1e-12:
        return None

    cov = (dst_centered.T @ src_centered) / src_pts.shape[0]
    try:
        u, singular_vals, vh = np.linalg.svd(cov)
    except Exception:
        return None
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vh
    scale = float((singular_vals * d).sum() / src_var)
    t = dst_mean - scale * (r @ src_mean)

    if not np.isfinite(scale) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
        return None
    return scale, r, t


def _align_points_to_anchor(
    source_xyz: np.ndarray,
    source_images_bin: Path,
    anchor_centers: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, dict]:
    try:
        source_centers = _load_camera_centers_by_name(source_images_bin)
    except Exception:
        return None, {"aligned": False, "reason": "invalid_images_bin", "overlap_images": 0}
    overlap = sorted(set(anchor_centers.keys()) & set(source_centers.keys()))
    if len(overlap) < 3:
        return None, {"aligned": False, "reason": "insufficient_overlap", "overlap_images": len(overlap)}

    src = np.asarray([source_centers[name] for name in overlap], dtype=np.float64)
    dst = np.asarray([anchor_centers[name] for name in overlap], dtype=np.float64)
    transform = _estimate_similarity_transform(src, dst)
    if transform is None:
        return None, {"aligned": False, "reason": "failed_transform", "overlap_images": len(overlap)}

    scale, rot, trans = transform
    aligned = (scale * (rot @ source_xyz.T)).T + trans
    if not np.all(np.isfinite(aligned)):
        return None, {"aligned": False, "reason": "non_finite_aligned_points", "overlap_images": len(overlap)}

    return aligned, {
        "aligned": True,
        "overlap_images": len(overlap),
        "scale": float(scale),
    }


def _create_merged_sparse_model(sparse_root: Path, candidates: list[dict], selected_rel_paths: list[str]) -> Path:
    score_map = {entry.get("relative_path"): entry for entry in candidates}
    selected = [score_map[rel] for rel in selected_rel_paths if rel in score_map]
    if len(selected) < 2:
        raise RuntimeError("Sparse merge requested, but fewer than two valid candidates were selected")

    def _score(entry: dict) -> tuple[int, int]:
        return int(entry.get("images", 0) or 0), int(entry.get("points", 0) or 0)

    anchor = max(selected, key=_score)
    anchor_rel = anchor.get("relative_path") or "."
    anchor_dir = (sparse_root / anchor_rel).resolve()

    merge_root = sparse_root / "_merged"
    merge_root.mkdir(parents=True, exist_ok=True)
    signature = _build_sparse_merge_signature(sparse_root, selected_rel_paths)
    merged_dir = merge_root / f"selection_{signature}"

    required = [merged_dir / "cameras.bin", merged_dir / "images.bin", merged_dir / "points3D.bin"]
    if all(path.exists() for path in required):
        logger.info("Reusing cached merged sparse model: %s", merged_dir)
        return merged_dir

    tmp_dir = merge_root / f".tmp_selection_{signature}_{int(time.time())}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(anchor_dir / "cameras.bin", tmp_dir / "cameras.bin")
    shutil.copy2(anchor_dir / "images.bin", tmp_dir / "images.bin")
    anchor_centers = _load_camera_centers_by_name(anchor_dir / "images.bin")

    merged_xyz: list[np.ndarray] = []
    merged_rgb: list[np.ndarray] = []
    merged_images = _read_images_binary_with_ids(anchor_dir / "images.bin")
    known_image_names = {entry.get("name") for entry in merged_images}
    allowed_camera_ids = {int(entry.get("camera_id")) for entry in merged_images if entry.get("camera_id") is not None}
    source_details: list[dict] = []
    for entry in selected:
        rel = entry.get("relative_path") or "."
        source_dir = (sparse_root / rel).resolve()
        points_path = source_dir / "points3D.bin"
        if not points_path.exists():
            logger.warning("Skipping sparse candidate without points3D.bin: %s", source_dir)
            source_details.append({"relative_path": rel, "used": False, "reason": "missing_points3D.bin"})
            continue
        xyz, rgb = read_points3D_binary(points_path)
        if len(xyz) == 0:
            source_details.append({"relative_path": rel, "used": False, "reason": "empty_points"})
            continue

        xyz_source = np.asarray(xyz, dtype=np.float64)
        if rel == anchor_rel:
            aligned_xyz = xyz_source
            align_info = {"aligned": True, "anchor": True, "overlap_images": len(anchor_centers), "scale": 1.0}
        else:
            aligned_xyz, align_info = _align_points_to_anchor(
                xyz_source,
                source_dir / "images.bin",
                anchor_centers,
            )
            if aligned_xyz is None:
                logger.warning(
                    "Skipping sparse candidate %s due to alignment failure (%s)",
                    rel,
                    align_info.get("reason"),
                )
                source_details.append({"relative_path": rel, "used": False, **align_info})
                continue

            source_images = read_images_binary(source_dir / "images.bin")
            source_centers = _load_camera_centers_by_name(source_dir / "images.bin")
            overlap = sorted(set(anchor_centers.keys()) & set(source_centers.keys()))
            transform = _estimate_similarity_transform(
                np.asarray([source_centers[name] for name in overlap], dtype=np.float64),
                np.asarray([anchor_centers[name] for name in overlap], dtype=np.float64),
            )
            added_images = 0
            discarded_images = 0
            if transform is None:
                discarded_images = len(source_images)
                align_info["camera_merge"] = "skipped_failed_transform"
            else:
                scale, rot, trans = transform
                rmse = _camera_pose_rmse(overlap, source_images, anchor_centers, scale, rot, trans)
                align_info["camera_fit_rmse"] = rmse
                if not np.isfinite(rmse) or rmse > 2.0:
                    # Discard all source images if fit is poor.
                    discarded_images = len(source_images)
                    align_info["camera_merge"] = "discarded_high_fit_error"
                else:
                    for src in source_images.values():
                        name = src.get("name")
                        if name in known_image_names:
                            discarded_images += 1
                            continue
                        cam_id = int(src.get("camera_id"))
                        if cam_id not in allowed_camera_ids:
                            discarded_images += 1
                            continue
                        try:
                            q_dst, t_dst = _transform_source_image_pose(
                                np.asarray(src.get("qvec"), dtype=np.float64),
                                np.asarray(src.get("tvec"), dtype=np.float64),
                                scale,
                                rot,
                                trans,
                            )
                        except Exception:
                            discarded_images += 1
                            continue
                        if not (np.all(np.isfinite(q_dst)) and np.all(np.isfinite(t_dst))):
                            discarded_images += 1
                            continue
                        merged_images.append(
                            {
                                "qvec": q_dst,
                                "tvec": t_dst,
                                "camera_id": cam_id,
                                "name": str(name),
                            }
                        )
                        known_image_names.add(str(name))
                        added_images += 1
                    align_info["camera_merge"] = "ok"
            align_info["added_images"] = added_images
            align_info["discarded_images"] = discarded_images

        merged_xyz.append(aligned_xyz)
        merged_rgb.append(np.asarray(rgb, dtype=np.uint8))
        source_details.append(
            {
                "relative_path": rel,
                "used": True,
                "points": int(len(xyz_source)),
                **align_info,
            }
        )

    if len(merged_xyz) < 2:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            "Sparse merge needs at least two alignable reconstructions with overlapping registered images"
        )

    xyz_concat = np.concatenate(merged_xyz, axis=0)
    rgb_concat = np.concatenate(merged_rgb, axis=0)
    _write_points3d_binary(tmp_dir / "points3D.bin", xyz_concat, rgb_concat)
    _write_images_binary(tmp_dir / "images.bin", merged_images)

    merge_meta = {
        "anchor_relative_path": anchor_rel,
        "selected_relative_paths": selected_rel_paths,
        "merged_points": int(len(xyz_concat)),
        "merged_images": int(len(merged_images)),
        "created_at": time.time(),
        "alignment": "similarity_transform_from_overlapping_camera_centers",
        "source_details": source_details,
    }
    with open(tmp_dir / "merge_meta.json", "w", encoding="utf-8") as handle:
        json.dump(merge_meta, handle, indent=2)

    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    tmp_dir.rename(merged_dir)
    logger.info(
        "Created merged sparse model %s from %s folders (%s points total)",
        merged_dir,
        len(selected_rel_paths),
        len(xyz_concat),
    )
    return merged_dir


def _resolve_sparse_model_for_training(
    sparse_root: Path,
    image_dir: Path,
    preference: str | None = None,
    merge_selection: list[str] | None = None,
) -> Path:
    sparse_root = Path(sparse_root)
    pref = (preference or "best").strip().lower()
    if pref != "merge_selected":
        return _select_best_sparse_model(sparse_root, image_dir, preference)

    summaries = _evaluate_sparse_candidates(sparse_root, image_dir)
    if not summaries:
        raise RuntimeError(f"No COLMAP reconstructions found under {sparse_root}")

    selected_rel_paths = _normalize_merge_selection(merge_selection)
    if not selected_rel_paths:
        logger.warning("merge_selected requested, but no folders were provided; falling back to best")
        return _select_best_sparse_model(sparse_root, image_dir, "best")

    score_map = {entry.get("relative_path"): entry for entry in summaries}
    valid = [rel for rel in selected_rel_paths if rel in score_map]
    invalid = [rel for rel in selected_rel_paths if rel not in score_map]
    if invalid:
        logger.warning("Ignoring invalid sparse merge selections: %s", invalid)

    if len(valid) == 0:
        logger.warning("No valid sparse merge selections remained; falling back to best")
        return _select_best_sparse_model(sparse_root, image_dir, "best")
    if len(valid) == 1:
        logger.info("Only one sparse selection provided for merge; using %s directly", valid[0])
        return _select_best_sparse_model(sparse_root, image_dir, valid[0])

    _persist_best_sparse_choice(
        sparse_root,
        max(summaries, key=lambda entry: (int(entry.get("images", 0) or 0), int(entry.get("points", 0) or 0))),
        summaries,
    )
    return _create_merged_sparse_model(sparse_root, summaries, valid)


def _select_best_sparse_model(sparse_root: Path, image_dir: Path, preference: str | None = None) -> Path:
    """Return the sparse reconstruction honoring user preference when possible."""
    sparse_root = Path(sparse_root)
    if not sparse_root.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_root}")

    pref = (preference or "best").strip()
    if pref and pref.lower() != "best":
        direct_rel = "." if pref in {".", "root"} else pref
        direct_target = (sparse_root / direct_rel).resolve()
        base = sparse_root.resolve()
        if direct_target == base or base in direct_target.parents:
            if all((direct_target / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
                logger.info("Using explicitly selected sparse candidate path '%s'", direct_rel)
                return direct_target

    summaries = _evaluate_sparse_candidates(sparse_root, image_dir)
    if not summaries:
        raise RuntimeError(f"No COLMAP reconstructions found under {sparse_root}")

    def _score(entry: dict) -> tuple[int, int]:
        return int(entry.get("images", 0) or 0), int(entry.get("points", 0) or 0)

    best_entry = max(summaries, key=_score)
    _persist_best_sparse_choice(sparse_root, best_entry, summaries)

    chosen_entry = best_entry
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

    mapper_refine_principal_point = bool(p.get("mapper_refine_principal_point", True))
    mapper_refine_focal_length = bool(p.get("mapper_refine_focal_length", True))
    mapper_refine_extra_params = bool(p.get("mapper_refine_extra_params", True))
    clean_sparse_before_run = bool(p.get("clean_sparse_before_run", True))

    if clean_sparse_before_run:
        logger.info("Clearing previous sparse outputs under %s", sparse_dir)
        _clear_sparse_outputs(sparse_dir)

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
    feat_cmd = _colmap_cmd(
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
    )
    if p.get("max_image_size"):
        feat_cmd += ["--SiftExtraction.max_image_size", str(p.get("max_image_size"))]
    else:
        feat_cmd += ["--SiftExtraction.max_image_size", "1600"]
    if p.get("peak_threshold") is not None:
        feat_cmd += ["--SiftExtraction.peak_threshold", str(p.get("peak_threshold"))]
    if p.get("max_num_features") is not None:
        feat_cmd += ["--SiftExtraction.max_num_features", str(p.get("max_num_features"))]
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
        match_cmd = _colmap_cmd("sequential_matcher", "--database_path", str(database_path))
    else:
        match_cmd = _colmap_cmd("exhaustive_matcher", "--database_path", str(database_path))
    if guided is not None:
        match_cmd += ["--SiftMatching.guided_matching", "1" if guided else "0"]
    if p.get("sift_matching_min_num_inliers") is not None:
        # Compatibility: many COLMAP builds expose this under TwoViewGeometry, not SiftMatching.
        match_cmd += ["--TwoViewGeometry.min_num_inliers", str(p.get("sift_matching_min_num_inliers"))]
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
        mapper_cmd = _colmap_cmd(
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--Mapper.ba_refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--Mapper.ba_refine_extra_params", "1" if mapper_refine_extra_params else "0",
        )
        if p.get("mapper_num_threads"):
            mapper_cmd += ["--Mapper.num_threads", str(p.get("mapper_num_threads"))]
        else:
            # Default to 2 mapper threads to reduce memory usage
            mapper_cmd += ["--Mapper.num_threads", "2"]
        if p.get("mapper_min_num_matches") is not None:
            mapper_cmd += ["--Mapper.min_num_matches", str(p.get("mapper_min_num_matches"))]
        if p.get("mapper_abs_pose_min_num_inliers") is not None:
            mapper_cmd += ["--Mapper.abs_pose_min_num_inliers", str(p.get("mapper_abs_pose_min_num_inliers"))]
        if p.get("mapper_abs_pose_min_inlier_ratio") is not None:
            mapper_cmd += ["--Mapper.abs_pose_min_inlier_ratio", str(p.get("mapper_abs_pose_min_inlier_ratio"))]
        if p.get("mapper_init_min_num_inliers") is not None:
            mapper_cmd += ["--Mapper.init_min_num_inliers", str(p.get("mapper_init_min_num_inliers"))]
        if p.get("mapper_filter_max_reproj_error") is not None:
            mapper_cmd += ["--Mapper.filter_max_reproj_error", str(p.get("mapper_filter_max_reproj_error"))]
        # Do not capture stdout/stderr into pipes here; allow the child to inherit the
        # container's stdout/stderr so COLMAP can stream directly to Docker logs.
        # Capturing into pipes without continuously reading can lead to pipe buffer
        # deadlocks where COLMAP appears to hang while producing output.
        # Stream COLMAP mapper output into the application logger so it is
        # persisted to the project's processing.log and visible via frontend.
        # Use a pipe and continuously read to avoid deadlocks.
        popen_cmd, popen_shell = _prepare_subprocess_command(mapper_cmd)
        proc = subprocess.Popen(
            popen_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=popen_shell,
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

    # Optional second-pass registration recovers images that failed initial mapper pass.
    run_image_registrator = bool(p.get("run_image_registrator", True))
    if run_image_registrator:
        improved_dirs: list[Path] = []
        for recon_dir in reconstruction_dirs:
            if _run_colmap_image_registration_pass(
                database_path,
                image_dir,
                recon_dir,
                mapper_refine_focal_length=mapper_refine_focal_length,
                mapper_refine_principal_point=mapper_refine_principal_point,
                mapper_refine_extra_params=mapper_refine_extra_params,
            ):
                improved_dirs.append(recon_dir)
        if improved_dirs:
            logger.info("COLMAP: post-registration completed for %d sparse model(s)", len(improved_dirs))

    # Convert COLMAP outputs into lightweight points.bin files for the frontend
    converter = None
    try:
        from . import pointsbin as worker_pointsbin

        converter = worker_pointsbin
    except Exception as worker_conv_err:
        logger.debug("Worker pointsbin module unavailable: %s", worker_conv_err)
        try:
            from bimba3d_backend.app.services import pointsbin as app_pointsbin

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


def _export_with_gsplat(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    splat_name: str = "splats.splat",
    ply_name: str = "splats.ply",
    export_ply: bool = True,
):
    """Load a checkpoint and export .splat (and optional .ply) using gsplat exporter."""
    import torch
    from gsplat.exporter import export_splats

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "splats" in ckpt and isinstance(ckpt["splats"], dict):
        state = ckpt["splats"]
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

    splat_path = output_dir / splat_name
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

    if export_ply:
        ply_path = output_dir / ply_name
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


def _parse_step_from_name(name: str, prefix: str) -> int | None:
    """Parse zero-based trainer step from known filename patterns."""
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: dict | list):
    """Atomically write JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def _read_latest_training_loss(engine_output_dir: Path) -> float | None:
    metrics_path = engine_output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        value = payload.get("loss") if isinstance(payload, dict) else None
        return float(value) if value is not None else None
    except Exception:
        return None


def _compute_laplacian_variance(image_path: Path) -> float | None:
    if not image_path.exists():
        return None
    try:
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        lap = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        return float(np.var(lap))
    except Exception:
        return None


def _collect_eval_history(engine_output_dir: Path, params: dict, mode: str) -> list[dict]:
    """Build comparison-compatible eval history from upstream stats outputs."""
    stats_dir = engine_output_dir / "stats"
    if not stats_dir.exists():
        return []

    eval_history: list[dict] = []
    for stats_file in sorted(stats_dir.glob("val_step*.json")):
        step_zero = _parse_step_from_name(stats_file.stem, "val_step")
        if step_zero is None:
            continue
        try:
            with open(stats_file, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to parse eval stats %s: %s", stats_file, exc)
            continue

        eval_history.append({
            "step": int(step_zero + 1),
            "convergence_speed": stats.get("psnr"),
            "final_loss": None,
            "lpips_mean": stats.get("lpips"),
            "sharpness_mean": stats.get("ssim"),
            "num_gaussians": stats.get("num_GS"),
            "tuning_params": {
                "mode": mode,
                "eval_interval": params.get("eval_interval"),
                "save_interval": params.get("save_interval"),
                "splat_export_interval": params.get("splat_export_interval"),
            },
        })

    return eval_history


def _materialize_eval_previews(engine_output_dir: Path, eval_step: int | None = None):
    """Promote one representative eval render (index 0000) per eval step into previews."""
    render_dir = engine_output_dir / "renders"
    previews_dir = engine_output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    if eval_step is not None:
        expected_zero = max(0, int(eval_step) - 1)
        preview_candidates = [(expected_zero + 1, render_dir / f"val_step{expected_zero}_0000.png")]
    else:
        preview_candidates: list[tuple[int, Path]] = []
        for png_path in sorted(render_dir.glob("val_step*_0000.png")):
            step_zero = _parse_step_from_name(png_path.stem, "val_step")
            if step_zero is None:
                continue
            preview_candidates.append((step_zero + 1, png_path))

    existing_candidates = [(s, p) for s, p in preview_candidates if p.exists()]
    if not existing_candidates:
        return

    for step, source in existing_candidates:
        target = previews_dir / f"preview_{step:06d}.png"
        if target.exists():
            continue
        try:
            shutil.copy2(source, target)
        except Exception as exc:
            logger.warning("Failed to copy preview %s -> %s: %s", source, target, exc)

    latest_step, latest_source = existing_candidates[-1]
    latest_target = previews_dir / "preview_latest.png"
    try:
        shutil.copy2(latest_source, latest_target)
        logger.info("Updated preview_latest.png from eval step %s", latest_step)
    except Exception as exc:
        logger.warning("Failed to update preview_latest.png: %s", exc)


def run_gsplat_training(image_dir: Path, colmap_dir: Path, output_dir: Path, params: dict, resume: bool = False):
    """Run upstream simple_trainer-compatible gsplat training."""
    from .gsplat_upstream.simple_trainer import Config, DefaultStrategy, Runner

    logger.info("Starting gsplat training (upstream simple_trainer path)...")

    try:
        import gsplat.cuda._wrapper as _gsplat_cuda_wrapper
        _gsplat_cuda_wrapper._make_lazy_cuda_obj("CameraModelType")
    except Exception as exc:
        raise RuntimeError(
            "Local gsplat CUDA backend is unavailable (CUDA toolkit/runtime extensions not loaded). "
            "For local mode, install a CUDA-enabled gsplat environment; otherwise use worker_mode='docker'."
        ) from exc

    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    engine_name = "gsplat"
    engine_output_dir = _get_engine_output_dir(base_output_dir, engine_name)
    (engine_output_dir / "previews").mkdir(parents=True, exist_ok=True)
    (engine_output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    tuning_history_path = engine_output_dir / "rule_update_history.json"

    p = params or {}
    runtime_mode = str(p.get("worker_mode") or "").strip().lower()
    if runtime_mode not in {"docker", "local"}:
        docker_flag = str(os.getenv("BIMBA3D_DOCKER_WORKER", "")).strip().lower()
        runtime_mode = "docker" if docker_flag in {"1", "true", "yes", "on"} else "local"
    logger.info(
        "Worker runtime mode before training: %s (BIMBA3D_DOCKER_WORKER=%s)",
        runtime_mode,
        os.getenv("BIMBA3D_DOCKER_WORKER"),
    )

    mode = p.get("mode", "baseline")
    max_steps = int(p.get("max_steps", 30_000))
    raw_tune_end_step = p.get("tune_end_step", 200)
    try:
        modified_tune_end_step = max(1, int(raw_tune_end_step))
    except Exception:
        modified_tune_end_step = 200
    raw_tune_interval = p.get("tune_interval", 25)
    try:
        modified_tune_interval = max(1, int(raw_tune_interval))
    except Exception:
        modified_tune_interval = 25
    raw_tune_scope = str(p.get("tune_scope", "with_strategy") or "with_strategy").strip().lower()
    tune_scope = raw_tune_scope if raw_tune_scope in {"core_only", "with_strategy"} else "with_strategy"
    splat_interval = p.get("splat_export_interval")
    try:
        splat_interval = max(1, int(splat_interval)) if splat_interval is not None else None
    except Exception:
        splat_interval = None
    log_interval = p.get("log_interval", 100)
    try:
        log_interval = max(1, int(log_interval))
    except Exception:
        log_interval = 100
    project_dir = base_output_dir.parent
    stop_flag = project_dir / "stop_requested"
    gsplat_start = time.time()
    run_session_id = str(uuid.uuid4())
    host_boot_id = "n/a"
    host_uptime_text = "n/a"
    try:
        host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8", errors="ignore").strip() or "n/a"
    except Exception:
        pass
    try:
        uptime_raw = Path("/proc/uptime").read_text(encoding="utf-8", errors="ignore").strip().split()
        if uptime_raw:
            host_uptime_text = f"{float(uptime_raw[0]):.1f}s"
    except Exception:
        pass
    logger.info(
        "GSPLAT run marker: session_id=%s pid=%d boot_id=%s host_uptime=%s started_at=%s",
        run_session_id,
        os.getpid(),
        host_boot_id,
        host_uptime_text,
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gsplat_start)),
    )
    tuning_state: dict[str, object] = {
        "updates": 0,
        "last_event": None,
        "events": [],
        "last_tuned_step": None,
        "phase_complete_logged": False,
    }
    runner_ref: dict[str, object] = {"runner": None}
    last_snapshot_step: dict[str, int] = {"value": -1}
    cpu_usage_state: dict[str, tuple[int, int] | None] = {"last": None}
    telemetry_state: dict[str, object] = {
        "snapshot_count": 0,
        "gpu_usage": "n/a",
        "cpu_usage": "n/a",
        "ram_usage": "n/a",
        "gpu_temp": "n/a",
    }

    if mode == "modified":
        logger.info(
            "Modified training mode active: scope=%s, rule-based tuning runs every %d steps through step %d; afterwards training continues normally",
            tune_scope,
            modified_tune_interval,
            modified_tune_end_step,
        )
    else:
        logger.info("Baseline training mode active: no deterministic profile adjustments")

    def persist_rule_update_history() -> None:
        """Persist detailed rule-update history for reproducibility/debugging."""
        payload = {
            "mode": mode,
            "tune_scope": tune_scope,
            "tune_end_step": modified_tune_end_step,
            "tune_interval": modified_tune_interval,
            "updates_count": int(tuning_state.get("updates", 0) or 0),
            "updates": list(tuning_state.get("events") or []),
            "last_update": tuning_state.get("last_event"),
        }
        _write_json_atomic(tuning_history_path, payload)

    def _log_training_snapshot(
        step: int,
        max_steps_local: int,
        loss: float,
        progress_fraction: float,
        elapsed_seconds: float,
        eta_seconds: float | None,
    ):
        """Emit an infrequent, high-value training snapshot for project logs."""
        runner_obj = runner_ref.get("runner")
        if runner_obj is None:
            return

        try:
            gaussians = None
            gaussians_opacity_mean = None
            gaussians_scale_mean = None
            means_tensor = getattr(runner_obj, "splats", {}).get("means")
            if means_tensor is not None and hasattr(means_tensor, "shape") and len(means_tensor.shape) > 0:
                gaussians = int(means_tensor.shape[0])
            opacity_tensor = getattr(runner_obj, "splats", {}).get("opacities")
            if opacity_tensor is not None and hasattr(opacity_tensor, "mean"):
                try:
                    gaussians_opacity_mean = float(opacity_tensor.detach().mean().item())
                except Exception:
                    gaussians_opacity_mean = None
            scale_tensor = getattr(runner_obj, "splats", {}).get("scales")
            if scale_tensor is not None and hasattr(scale_tensor, "mean"):
                try:
                    gaussians_scale_mean = float(scale_tensor.detach().mean().item())
                except Exception:
                    gaussians_scale_mean = None

            strategy_obj = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
            strategy_vals: dict[str, object] = {}
            if strategy_obj is not None:
                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    if hasattr(strategy_obj, key):
                        value = getattr(strategy_obj, key)
                        if isinstance(value, float):
                            strategy_vals[key] = round(value, 8)
                        elif isinstance(value, int):
                            strategy_vals[key] = value

            optimizer_lrs: dict[str, float] = {}
            optimizers = getattr(runner_obj, "optimizers", {})
            for name in ("means", "opacities", "scales", "quats", "sh0", "shN"):
                optimizer = optimizers.get(name) if isinstance(optimizers, dict) else None
                if optimizer is None or not getattr(optimizer, "param_groups", None):
                    continue
                lr_val = optimizer.param_groups[0].get("lr")
                if lr_val is None:
                    continue
                optimizer_lrs[name] = float(lr_val)

            cfg_obj = getattr(runner_obj, "cfg", None)
            sh_degree = getattr(runner_obj, "sh_degree_to_use", None)
            eval_steps_cfg = list(getattr(cfg_obj, "eval_steps", []) or [])
            save_steps_cfg = list(getattr(cfg_obj, "save_steps", []) or [])
            next_eval_step = next((int(s) for s in eval_steps_cfg if int(s) >= int(step)), None)
            next_save_step = next((int(s) for s in save_steps_cfg if int(s) >= int(step)), None)

            steps_per_second = (float(step) / elapsed_seconds) if elapsed_seconds > 0 else None
            tuning_applied = bool(int(tuning_state.get("updates", 0) or 0) > 0)

            telemetry_state["snapshot_count"] = int(telemetry_state.get("snapshot_count", 0) or 0) + 1
            snapshot_count = int(telemetry_state["snapshot_count"])
            should_collect_telemetry = snapshot_count == 1 or (snapshot_count % 2 == 0)

            gpu_usage_text = str(telemetry_state.get("gpu_usage", "n/a"))
            gpu_temp_text = str(telemetry_state.get("gpu_temp", "n/a"))
            if should_collect_telemetry:
                try:
                    import torch
                    if torch.cuda.is_available():
                        selected_idx = int(torch.cuda.current_device())
                        smi_cmd = [
                            "nvidia-smi",
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ]
                        smi_out = subprocess.run(smi_cmd, check=True, capture_output=True, text=True)
                        for raw in (smi_out.stdout or "").splitlines():
                            parts = [part.strip() for part in raw.split(",")]
                            if len(parts) < 5:
                                continue
                            try:
                                gpu_idx = int(parts[0])
                            except Exception:
                                continue
                            if gpu_idx != selected_idx:
                                continue
                            try:
                                util_pct = int(float(parts[1]))
                            except Exception:
                                util_pct = 0
                            try:
                                mem_used = float(parts[2])
                                mem_total = max(1.0, float(parts[3]))
                                mem_pct = int(round((mem_used / mem_total) * 100.0))
                            except Exception:
                                mem_pct = 0
                            try:
                                gpu_temp_c = int(round(float(parts[4])))
                                gpu_temp_text = f"{gpu_temp_c}C"
                            except Exception:
                                gpu_temp_text = "n/a"
                            gpu_usage_text = f"gpu{gpu_idx}:util={util_pct}% mem={mem_pct}%"
                            break
                except Exception:
                    pass

            ram_usage_text = str(telemetry_state.get("ram_usage", "n/a"))
            if should_collect_telemetry:
                try:
                    meminfo: dict[str, int] = {}
                    with open("/proc/meminfo", "r", encoding="utf-8") as mem_file:
                        for raw_line in mem_file:
                            parts = raw_line.split(":", 1)
                            if len(parts) != 2:
                                continue
                            key = parts[0].strip()
                            value_field = parts[1].strip().split()[0]
                            try:
                                meminfo[key] = int(value_field)
                            except Exception:
                                continue

                    total_kb = int(meminfo.get("MemTotal", 0))
                    available_kb = int(meminfo.get("MemAvailable", 0))
                    if total_kb > 0:
                        used_kb = max(0, total_kb - available_kb)
                        ram_pct = int(round((used_kb / float(total_kb)) * 100.0))
                        ram_usage_text = f"util={ram_pct}%"
                except Exception:
                    pass

            cpu_usage_text = str(telemetry_state.get("cpu_usage", "n/a"))
            if should_collect_telemetry:
                try:
                    with open("/proc/stat", "r", encoding="utf-8") as stat_file:
                        first_line = stat_file.readline().strip()
                    parts = first_line.split()
                    if len(parts) >= 5 and parts[0] == "cpu":
                        values = [int(v) for v in parts[1:11]]
                        idle = values[3] + values[4]
                        total = sum(values)
                        previous = cpu_usage_state.get("last")
                        cpu_usage_state["last"] = (total, idle)
                        if previous is not None:
                            prev_total, prev_idle = previous
                            total_delta = max(1, total - prev_total)
                            idle_delta = max(0, idle - prev_idle)
                            usage_pct = int(round(((total_delta - idle_delta) / float(total_delta)) * 100.0))
                            usage_pct = max(0, min(100, usage_pct))
                            cpu_usage_text = f"util={usage_pct}%"
                except Exception:
                    pass

            telemetry_state["gpu_usage"] = gpu_usage_text
            telemetry_state["cpu_usage"] = cpu_usage_text
            telemetry_state["ram_usage"] = ram_usage_text
            telemetry_state["gpu_temp"] = gpu_temp_text

            logger.info(
                "[GSPLAT SNAPSHOT] step=%d/%d progress=%.2f%% loss=%.6f gs=%s opacity_mean=%s scale_mean=%s sh_degree=%s next_eval=%s next_save=%s elapsed=%.1fs eta=%s speed=%s tuning_applied=%s strategy=%s lrs=%s",
                int(step),
                int(max_steps_local),
                float(progress_fraction * 100.0),
                float(loss),
                str(gaussians) if gaussians is not None else "n/a",
                f"{gaussians_opacity_mean:.6f}" if gaussians_opacity_mean is not None else "n/a",
                f"{gaussians_scale_mean:.6f}" if gaussians_scale_mean is not None else "n/a",
                str(sh_degree) if sh_degree is not None else "n/a",
                str(next_eval_step) if next_eval_step is not None else "n/a",
                str(next_save_step) if next_save_step is not None else "n/a",
                float(elapsed_seconds),
                f"{float(eta_seconds):.1f}s" if eta_seconds is not None else "n/a",
                f"{steps_per_second:.3f} step/s" if steps_per_second is not None else "n/a",
                tuning_applied,
                strategy_vals,
                {k: round(v, 10) for k, v in optimizer_lrs.items()},
            )
            logger.info(
                "[GSPLAT TELEMETRY] step=%d gpu_usage=%s cpu_usage=%s ram_usage=%s gpu_temp=%s",
                int(step),
                gpu_usage_text,
                cpu_usage_text,
                ram_usage_text,
                gpu_temp_text,
            )
        except Exception as exc:
            logger.debug("Failed to emit gsplat training snapshot at step %s: %s", step, exc)

    def apply_modified_rules(step: int, loss: float) -> bool:
        """Apply rule-based tuning updates while modified mode is in its tuning window."""
        if mode != "modified":
            return False
        if step > modified_tune_end_step:
            return False
        if step < modified_tune_interval:
            return False
        if step % modified_tune_interval != 0 and step != modified_tune_end_step:
            return False
        if tuning_state.get("last_tuned_step") == step:
            return False

        runner_obj = runner_ref.get("runner")
        if runner_obj is None:
            return False

        try:
            try:
                loss_value = float(loss)
            except Exception:
                loss_value = 0.0

            if loss_value >= 0.20:
                profile = "high_loss"
                lr_multipliers = {
                    "means": 0.92,
                    "opacities": 0.86,
                    "scales": 1.00,
                    "quats": 1.00,
                    "sh0": 0.96,
                    "shN": 0.96,
                }
                strategy_multipliers = {
                    "grow_grad2d": 0.95,
                    "prune_opa": 1.00,
                    "refine_every": 0.96,
                    "reset_every": 0.98,
                }
            elif loss_value >= 0.08:
                profile = "mid_loss"
                lr_multipliers = {
                    "means": 0.96,
                    "opacities": 0.92,
                    "scales": 1.00,
                    "quats": 1.00,
                    "sh0": 1.00,
                    "shN": 1.00,
                }
                strategy_multipliers = {
                    "grow_grad2d": 0.97,
                    "prune_opa": 1.00,
                    "refine_every": 0.98,
                    "reset_every": 1.00,
                }
            else:
                profile = "low_loss"
                lr_multipliers = {
                    "means": 1.02,
                    "opacities": 0.99,
                    "scales": 1.00,
                    "quats": 1.00,
                    "sh0": 1.05,
                    "shN": 1.05,
                }
                strategy_multipliers = {
                    "grow_grad2d": 0.99,
                    "prune_opa": 1.00,
                    "refine_every": 0.99,
                    "reset_every": 1.00,
                }

            applied_multipliers = {
                "lr": float(lr_multipliers["means"]),
                "opacity_lr_mult": float(lr_multipliers["opacities"]),
                "sh_lr_mult": float((float(lr_multipliers["sh0"]) + float(lr_multipliers["shN"])) / 2.0),
                "position_lr_mult": float(lr_multipliers["means"]),
                "densify_threshold_mult": float(strategy_multipliers["grow_grad2d"]),
            }

            before_lrs: dict[str, float] = {}
            after_lrs: dict[str, float] = {}
            for name, mult in lr_multipliers.items():
                optimizer = getattr(runner_obj, "optimizers", {}).get(name)
                if optimizer is None or not optimizer.param_groups:
                    continue
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                before_lrs[name] = current_lr
                new_lr = max(1e-7, min(1.0, current_lr * float(mult)))
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                after_lrs[name] = new_lr

            strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
            strategy_before: dict[str, float] = {}
            strategy_after: dict[str, float] = {}
            if strategy is not None:
                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    strategy_before[key] = float(getattr(strategy, key))

                strategy.grow_grad2d = max(5e-5, min(5e-3, float(strategy.grow_grad2d) * float(strategy_multipliers["grow_grad2d"])))
                if tune_scope == "with_strategy":
                    strategy.prune_opa = max(1e-4, min(5e-2, float(strategy.prune_opa) * float(strategy_multipliers["prune_opa"])))
                    strategy.refine_every = max(25, min(300, int(float(strategy.refine_every) * float(strategy_multipliers["refine_every"]))))
                    strategy.reset_every = max(max(strategy.refine_every, 1000), min(6000, int(float(strategy.reset_every) * float(strategy_multipliers["reset_every"]))))

                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    strategy_after[key] = float(getattr(strategy, key))

            event = {
                "step": int(step),
                "loss": loss_value,
                "profile": profile,
                "scope": tune_scope,
                "rule_multipliers": applied_multipliers,
                "adjustments": [
                    "rule_based_lr_scaling",
                    "rule_based_strategy_scaling",
                ],
                "params": {
                    "learning_rates": after_lrs,
                    "strategy": strategy_after,
                },
                "before": {
                    "learning_rates": before_lrs,
                    "strategy": strategy_before,
                },
            }
            tuning_state["last_event"] = event
            tuning_state["events"].append(event)
            tuning_state["updates"] = int(tuning_state.get("updates", 0) or 0) + 1
            tuning_state["last_tuned_step"] = int(step)
            persist_rule_update_history()

            update_status(
                project_dir,
                "processing",
                mode=mode,
                tuning_active=True,
                last_tuning={
                    "step": int(step),
                    "action": f"Rule-based {profile} update",
                    "reason": f"Modified mode rule check (step {step} <= {modified_tune_end_step})",
                },
            )
            logger.info(
                "Modified rule update applied at step %d/%d (loss=%.6f, profile=%s, scope=%s)",
                step,
                modified_tune_end_step,
                loss_value,
                profile,
                tune_scope,
            )
            logger.info(
                "Modified rule details step=%d before_lrs=%s after_lrs=%s before_strategy=%s after_strategy=%s",
                step,
                json.dumps(before_lrs, sort_keys=True),
                json.dumps(after_lrs, sort_keys=True),
                json.dumps(strategy_before, sort_keys=True),
                json.dumps(strategy_after, sort_keys=True),
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed modified-mode rule update at step %d/%d: %s",
                step,
                modified_tune_end_step,
                exc,
            )
            return False

    def stop_checker() -> bool:
        return stop_flag.exists()

    def progress_callback(step: int, max_steps: int, loss: float) -> None:
        if mode == "modified" and step == modified_tune_interval:
            logger.info(
                "Modified tuning window started at step %d; rule checks every %d steps until step %d",
                step,
                modified_tune_interval,
                modified_tune_end_step,
            )
        if mode == "modified" and not tuning_state.get("phase_complete_logged") and step == modified_tune_end_step + 1:
            logger.info(
                "Modified tuning window finished at step %d; continuing with tuned parameters and no further rule updates",
                modified_tune_end_step,
            )
            tuning_state["phase_complete_logged"] = True
        apply_modified_rules(step, loss)
        requested_stop = stop_checker()
        status_text = "stopping" if requested_stop else "processing"
        progress_fraction = 0.0 if max_steps <= 0 else float(step) / float(max_steps)
        progress_fraction = max(0.0, min(1.0, progress_fraction))
        message = (
            f"⏸️ Stopping after step {step}/{max_steps} completes (loss: {loss:.6f})..."
            if requested_stop
            else f"🎯 Training step {step}/{max_steps} (loss: {loss:.6f})"
        )

        now = time.time()
        elapsed = now - gsplat_start
        eta = (
            (elapsed / progress_fraction) * (1 - progress_fraction)
            if progress_fraction > 0
            else None
        )
        timing = {"start": gsplat_start, "elapsed": elapsed}
        if eta is not None:
            timing["eta"] = eta

        update_status(
            project_dir,
            status_text,
            progress=60 + int(progress_fraction * 35),
            mode=mode,
            tuning_active=(mode == "modified" and step <= modified_tune_end_step),
            currentStep=step,
            maxSteps=max_steps,
            stop_requested=requested_stop,
            stage="training",
            stage_progress=int(progress_fraction * 100),
            message=message,
            timing=timing,
        )
        write_metrics(project_dir, {
            "step": step,
            "loss": loss,
            "progress": progress_fraction,
        }, engine=engine_name)

        should_log_snapshot = (
            step == 1
            or step == max_steps
            or step % log_interval == 0
        )
        if should_log_snapshot and step != last_snapshot_step["value"]:
            _log_training_snapshot(step, max_steps, loss, progress_fraction, elapsed, eta)
            last_snapshot_step["value"] = step

    gpu_inventory: list[dict[str, object]] = []
    selected_gpu_index: int | None = None
    selected_gpu_name: str | None = None
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            for gpu_idx in range(int(torch.cuda.device_count())):
                try:
                    gpu_inventory.append({
                        "index": int(gpu_idx),
                        "name": str(torch.cuda.get_device_name(gpu_idx)),
                    })
                except Exception:
                    gpu_inventory.append({
                        "index": int(gpu_idx),
                        "name": "unknown",
                    })
            selected_gpu_index = int(torch.cuda.current_device()) if torch.cuda.device_count() > 0 else None
            if selected_gpu_index is not None:
                selected_gpu_name = str(torch.cuda.get_device_name(selected_gpu_index))
    except Exception:
        cuda_ok = False
    device = "cuda" if (p.get("use_cuda", True) and cuda_ok) else "cpu"

    logger.info(
        "Training runtime selection: mode=%s device=%s use_cuda=%s cuda_available=%s gpu_count=%d",
        mode,
        device,
        bool(p.get("use_cuda", True)),
        bool(cuda_ok),
        len(gpu_inventory),
    )
    if gpu_inventory:
        logger.info("CUDA GPU inventory: %s", json.dumps(gpu_inventory))
    if device == "cuda":
        logger.info(
            "Selected CUDA device: index=%s name=%s",
            str(selected_gpu_index) if selected_gpu_index is not None else "unknown",
            selected_gpu_name or "unknown",
        )
        try:
            smi_cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            smi_out = subprocess.run(smi_cmd, check=True, capture_output=True, text=True)
            selected_usage_line = None
            for raw in (smi_out.stdout or "").splitlines():
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    gpu_idx = int(parts[0])
                except Exception:
                    continue
                if selected_gpu_index is not None and gpu_idx != selected_gpu_index:
                    continue
                try:
                    util_pct = int(float(parts[1]))
                except Exception:
                    util_pct = 0
                try:
                    mem_used = float(parts[2])
                    mem_total = max(1.0, float(parts[3]))
                    mem_pct = int(round((mem_used / mem_total) * 100.0))
                except Exception:
                    mem_pct = 0
                selected_usage_line = f"GPU usage now: util={util_pct}% mem={mem_pct}%"
                break
            if selected_usage_line:
                logger.info(selected_usage_line)
        except Exception:
            pass
    else:
        logger.info("Selected compute mode: CPU")

    if stop_flag.exists():
        update_status(project_dir, "stopped", progress=55, stage="training", message="⏸️ Processing stopped before gsplat training.", stop_requested=True, stopped_stage="training")
        return 0

    update_status(
        project_dir,
        "processing",
        progress=55,
        stage="training",
        stage_progress=0,
        message=f"🚀 Initializing upstream simple_trainer ({'GPU ⚡' if device == 'cuda' else 'CPU'})...",
        mode=mode,
        timing={"start": gsplat_start},
    )

    # Upstream simple_trainer expects data_dir/{images,sparse/0}. Reuse engine root directly.
    dataset_dir = engine_output_dir
    images_link = dataset_dir / "images"
    sparse_zero = dataset_dir / "sparse" / "0"
    sparse_zero.parent.mkdir(parents=True, exist_ok=True)
    for link_path, target in ((images_link, image_dir), (sparse_zero, colmap_dir)):
        if link_path.exists() or link_path.is_symlink():
            try:
                if link_path.is_symlink() or link_path.is_file():
                    link_path.unlink()
                else:
                    shutil.rmtree(link_path)
            except Exception:
                pass
        os.symlink(str(Path(target).resolve()), str(link_path), target_is_directory=True)

    def _build_steps(interval_value, fallback):
        if interval_value is None:
            return fallback
        try:
            interval = max(1, int(interval_value))
        except Exception:
            return fallback
        out = list(range(interval, max_steps + 1, interval))
        if max_steps not in out:
            out.append(max_steps)
        return sorted(set(out))

    strategy = DefaultStrategy(
        verbose=True,
        prune_opa=float(p.get("opacity_threshold", 0.005)),
        grow_grad2d=float(p.get("densify_grad_threshold", 0.0002)),
        grow_scale3d=float(p.get("percent_dense", 0.01)),
        refine_start_iter=int(p.get("densify_from_iter", 500)),
        refine_stop_iter=int(p.get("densify_until_iter", 15000)),
        refine_every=max(1, int(p.get("densification_interval", 100))),
        reset_every=max(1, int(p.get("opacity_reset_interval", 3000))),
    )

    feature_lr = float(p.get("feature_lr", 2.5e-3))
    eval_steps = _build_steps(p.get("eval_interval"), [7000, 30000])
    save_steps = sorted(set(
        _build_steps(p.get("save_interval"), [7000, 30000])
        + _build_steps(p.get("splat_export_interval"), [7000, 30000])
    ))

    cfg = Config(
        disable_viewer=True,
        disable_video=True,
        load_exposure=False,
        data_dir=str(dataset_dir),
        data_factor=1,
        result_dir=str(engine_output_dir),
        test_every=8,
        normalize_world_space=True,
        batch_size=1,
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_ply=False,
        ssim_lambda=float(p.get("lambda_dssim", 0.2)),
        means_lr=float(p.get("position_lr_init", 1.6e-4)),
        scales_lr=float(p.get("scaling_lr", 5.0e-3)),
        opacities_lr=float(p.get("opacity_lr", 5.0e-2)),
        quats_lr=float(p.get("rotation_lr", 1.0e-3)),
        sh0_lr=feature_lr,
        shN_lr=feature_lr / 20.0,
        strategy=strategy,
        tb_every=0,
    )
    cfg.disable_tqdm = not bool(p.get("enable_tqdm", False))
    cfg.progress_every = max(1, int(log_interval))
    if cfg.disable_tqdm:
        os.environ["TQDM_DISABLE"] = "1"
    else:
        os.environ.pop("TQDM_DISABLE", None)

    logger.info(
        "GSPLAT logging cadence: snapshot every %d steps (config key: log_interval); tqdm=%s",
        log_interval,
        "disabled" if cfg.disable_tqdm else "enabled",
    )

    # Hook worker-level stop/progress into vendored upstream runner.
    cfg.stop_checker = stop_checker
    cfg.progress_callback = progress_callback

    snapshots_dir = engine_output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    def eval_callback(step: int) -> None:
        _materialize_eval_previews(engine_output_dir, eval_step=step)

    def checkpoint_callback(step: int, checkpoint_path: str) -> None:
        if splat_interval and step % splat_interval != 0 and step != max_steps:
            return
        snapshot_name = f"snapshot_step_{step:06d}.splat"
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return
        _export_with_gsplat(
            Path(checkpoint_path),
            snapshots_dir,
            splat_name=snapshot_name,
            export_ply=False,
        )

    cfg.eval_callback = eval_callback
    cfg.checkpoint_callback = checkpoint_callback

    if device == "cpu":
        # Upstream runner assumes CUDA device naming; keep compatibility guard.
        raise RuntimeError("Upstream simple_trainer currently requires CUDA in this worker path")

    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    runner_ref["runner"] = runner
    stop_reason = runner.train()
    gsplat_end = time.time()

    if stop_reason is not None or stop_flag.exists():
        logger.info("Training stopped by user, skipping export and completion status.")
        update_status(
            project_dir,
            "stopped",
            progress=60,
            stage="training",
            message="⏸️ Training stopped by user.",
            stopped_stage="training",
            stopped_step=stop_reason if isinstance(stop_reason, int) else None,
        )
        if stop_flag.exists():
            stop_flag.unlink()
        return stop_reason if isinstance(stop_reason, int) else 1

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
    ckpt_dir = engine_output_dir / "ckpts"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    snapshots_dir = engine_output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    exported_snapshots = 0
    for ckpt in ckpts:
        step_zero = _parse_step_from_name(ckpt.stem, "ckpt_")
        if step_zero is None:
            continue
        step = step_zero + 1
        if splat_interval and step % splat_interval != 0 and step != max_steps:
            continue
        snapshot_name = f"snapshot_step_{step:06d}.splat"
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            continue
        try:
            _export_with_gsplat(
                ckpt,
                snapshots_dir,
                splat_name=snapshot_name,
                export_ply=False,
            )
            exported_snapshots += 1
        except Exception as exc:
            logger.warning("Failed snapshot export for %s: %s", ckpt.name, exc)

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

    if exported_snapshots:
        logger.info("Exported %d interval snapshot(s) to %s", exported_snapshots, snapshots_dir)

    # Build frontend/comparison-compatible artifacts from upstream outputs.
    _materialize_eval_previews(engine_output_dir)
    eval_history = _collect_eval_history(engine_output_dir, p, mode)
    if eval_history and mode == "modified":
        for row in eval_history:
            tuning_params = row.get("tuning_params") if isinstance(row, dict) else None
            if isinstance(tuning_params, dict):
                tuning_params["modified_rule_updates"] = int(tuning_state.get("updates", 0) or 0)
                if tuning_state.get("last_event"):
                    tuning_params["modified_last_rule_step"] = tuning_state["last_event"].get("step")
    if eval_history:
        _write_json_atomic(engine_output_dir / "eval_history.json", eval_history)
        final_eval = eval_history[-1]
        latest_training_loss = _read_latest_training_loss(engine_output_dir)
        final_loss_value = final_eval.get("final_loss")
        if final_loss_value is None:
            final_loss_value = latest_training_loss
        laplacian_variance = _compute_laplacian_variance(engine_output_dir / "previews" / "preview_latest.png")
        final_metrics = {
            "lpips_score": final_eval.get("lpips_mean"),
            "sharpness": final_eval.get("sharpness_mean"),
            "laplacian_variance": laplacian_variance,
            "convergence_speed": final_eval.get("convergence_speed"),
            "final_loss": final_loss_value,
            "gaussian_count": final_eval.get("num_gaussians"),
        }
        evaluation_parameters = {
            "fixed_iteration_step": final_eval.get("step"),
            "eval_interval": p.get("eval_interval"),
            "save_interval": p.get("save_interval"),
            "splat_export_interval": p.get("splat_export_interval"),
            "metrics_requested": [
                "convergence_speed",
                "final_loss",
                "lpips",
                "image_sharpness_laplacian_variance",
                "gaussian_count",
                "visual_comparison",
            ],
            "visual_comparison": {
                "latest_preview": "previews/preview_latest.png",
                "renders_dir": "renders",
                "snapshots_dir": "snapshots",
            },
        }
        adaptive_payload = {
            "mode": mode,
            "tune_scope": tune_scope if mode == "modified" else None,
            "final_evaluation": final_metrics,
            "evaluation_parameters": evaluation_parameters,
            "tune_end_step": modified_tune_end_step if mode == "modified" else final_eval.get("step"),
            "tune_interval": modified_tune_interval if mode == "modified" else None,
            "tuning_history": list(tuning_state.get("events") or []),
            "final_params": (tuning_state.get("last_event") or {}).get("params", {}),
        }
        _write_json_atomic(engine_output_dir / "adaptive_tuning_results.json", adaptive_payload)
        if mode == "modified":
            persist_rule_update_history()

        metadata_path = engine_output_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}
        metadata["evaluation_metrics"] = final_metrics
        metadata["final_metrics"] = {
            "convergence_speed": final_eval.get("convergence_speed"),
            "final_loss": final_loss_value,
            "lpips_mean": final_eval.get("lpips_mean"),
            "sharpness_mean": final_eval.get("sharpness_mean"),
            "laplacian_variance": laplacian_variance,
        }
        metadata["evaluation_parameters"] = evaluation_parameters
        metadata["num_gaussians"] = final_eval.get("num_gaussians")
        metadata["mode"] = mode
        metadata["tune_scope"] = tune_scope if mode == "modified" else None
        _write_json_atomic(metadata_path, metadata)

    if stop_flag.exists():
        stop_flag.unlink()

    return None


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


def _run_selected_training_engine(
    engine: str,
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    params: dict,
    *,
    resume: bool,
):
    """Run the selected engine training pipeline through one dispatch point."""
    runner_map = {
        "gsplat": run_gsplat_training,
        "litegs": run_litegs_training,
    }
    try:
        runner = runner_map[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported training engine: {engine}") from exc
    return runner(image_dir, colmap_dir, output_dir, params, resume=resume)


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Worker")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("--data-dir", default="/data/projects", help="Data directory")
    parser.add_argument("--params", type=json.loads, default="{}", help="Training parameters as JSON")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "modified"], 
                        help="Training mode: baseline or modified (deterministic profile applied at tune_end_step, default 200)")

    args = parser.parse_args()

    # Merge mode into params
    params = dict(args.params or {})
    if "mode" not in params:
        params["mode"] = args.mode
    engine = params.get("engine", "gsplat")
    if engine not in SUPPORTED_ENGINES:
        engine = "gsplat"
    params["engine"] = engine

    project_dir = Path(args.data_dir) / args.project_id
    image_dir = project_dir / "images"
    output_dir = project_dir / "outputs"
    sparse_preference = params.get("sparse_preference") if isinstance(params, dict) else None
    if isinstance(sparse_preference, str):
        sparse_preference = sparse_preference.strip() or None
    sparse_merge_selection = params.get("sparse_merge_selection") if isinstance(params, dict) else None
    if not isinstance(sparse_merge_selection, list):
        sparse_merge_selection = None


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
        output_dir.mkdir(parents=True, exist_ok=True)

        stage = params.get("stage", "full")
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
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
            logger.info("Preparing resized image set (<=%d px) for project %s", images_max_size, args.project_id)
            update_status(
                project_dir,
                "processing",
                progress=5 if resize_stage == "colmap" else 55,
                stage=resize_stage,
                stage_progress=2 if resize_stage == "training" else 5,
                message=f"📐 Resizing input images to <= {images_max_size}px...",
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
                    message=f"✅ Image set ready at <= {images_max_size}px",
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
            colmap_dir = _resolve_sparse_model_for_training(
                sparse_root,
                active_image_dir,
                sparse_preference,
                sparse_merge_selection,
            )
            trainer_label = ENGINE_LABELS.get(engine, ENGINE_LABELS["gsplat"])
            msg = f"🎯 Starting {trainer_label} training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg, engine=engine)
            stop_reason = _run_selected_training_engine(
                engine,
                active_image_dir,
                colmap_dir,
                output_dir,
                params,
                resume=resume,
            )
        else:
            # Full pipeline
            update_status(project_dir, "processing", progress=1, stage="colmap", message="🚀 Starting full pipeline - Running COLMAP structure-from-motion...", engine=engine)
            colmap_dir = run_colmap(active_image_dir, output_dir, params)
            logger.info("COLMAP completed")

            colmap_dir = _resolve_sparse_model_for_training(
                output_dir / "sparse",
                active_image_dir,
                sparse_preference,
                sparse_merge_selection,
            )
            trainer_label = ENGINE_LABELS.get(engine, ENGINE_LABELS["gsplat"])
            msg = f"🎯 Starting {trainer_label} training..."
            update_status(project_dir, "processing", progress=55, stage="training", message=msg, engine=engine)
            stop_reason = _run_selected_training_engine(
                engine,
                active_image_dir,
                colmap_dir,
                output_dir,
                params,
                resume=resume,
            )

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
