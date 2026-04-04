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
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .engines import ENGINE_LABELS, SUPPORTED_ENGINES, run_selected_engine
from .image_resize import prepare_training_images, normalize_max_size
from .colmap_loader import COLMAPDataset, qvec2rotmat, read_images_binary, read_points3D_binary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLMAP_EXE = (os.getenv("COLMAP_EXE") or "colmap").strip() or "colmap"

ENGINE_SUBDIR = "engines"
BEST_SPARSE_META = ".best_sparse_selection.json"
SPARSE_IMAGE_MEMBERSHIP_META = ".sparse_image_membership.json"

_UNRECOGNIZED_OPTION_RE = re.compile(r"unrecogni[sz]ed option '([^']+)'", re.IGNORECASE)

_COLMAP_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
}


def _parse_boolish(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_stop_source(stop_flag_path: Path) -> str:
    """Return stop source label based on stop flag content."""
    if not stop_flag_path.exists():
        return "none"
    try:
        payload = stop_flag_path.read_text(encoding="utf-8").strip().lower()
    except Exception:
        payload = ""
    if "backend" in payload and "shutdown" in payload:
        return "backend"
    return "user"


def _resolve_colmap_camera_model(value: object) -> str:
    candidate = str(value or "OPENCV").strip().upper()
    if candidate in _COLMAP_CAMERA_MODELS:
        return candidate
    logger.warning("Unsupported COLMAP camera_model '%s'; falling back to OPENCV", value)
    return "OPENCV"


def _resolve_colmap_executable() -> str:
    candidate = COLMAP_EXE

    if os.path.isabs(candidate) and Path(candidate).exists():
        if os.name == "nt" and candidate.lower().endswith("colmap.exe"):
            exe_path = Path(candidate)
            bat_candidates = [
                exe_path.with_name("COLMAP.bat"),
                exe_path.parent.parent / "COLMAP.bat",
            ]
            for bat in bat_candidates:
                if bat.exists():
                    return str(bat)
        return candidate

    found = shutil.which(candidate)
    if found:
        if os.name == "nt" and found.lower().endswith("colmap.exe"):
            exe_path = Path(found)
            bat_candidates = [
                exe_path.with_name("COLMAP.bat"),
                exe_path.parent.parent / "COLMAP.bat",
            ]
            for bat in bat_candidates:
                if bat.exists():
                    return str(bat)
        return found

    if os.name == "nt":
        program_data = os.environ.get("ProgramData", r"C:\ProgramData")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

        fallback_candidates = [
            Path(program_data) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.bat",
            Path(program_data) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.exe",
            Path(program_files) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.bat",
            Path(program_files) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.exe",
            Path(program_files_x86) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.bat",
            Path(program_files_x86) / "Bimba3D" / "third_party" / "colmap" / "COLMAP.exe",
        ]

        for fallback in fallback_candidates:
            if fallback.exists():
                return str(fallback)

    return candidate


def _prepare_subprocess_command(cmd: list[str]) -> tuple[list[str] | str, bool]:
    if os.name == "nt" and cmd and str(cmd[0]).lower().endswith((".bat", ".cmd")):
        return subprocess.list2cmdline(cmd), True
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
    colmap_exec: str,
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
        _run_cmd_with_retry([
            colmap_exec, "image_registrator",
            "--database_path", str(database_path),
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
        ])
        _run_cmd_with_retry([
            colmap_exec, "point_triangulator",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
            "--Mapper.ba_refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--Mapper.ba_refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--Mapper.ba_refine_extra_params", "1" if mapper_refine_extra_params else "0",
        ])
        _run_cmd_with_retry([
            colmap_exec, "bundle_adjuster",
            "--input_path", str(reconstruction_dir),
            "--output_path", str(reconstruction_dir),
            "--BundleAdjustment.refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--BundleAdjustment.refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--BundleAdjustment.refine_extra_params", "1" if mapper_refine_extra_params else "0",
        ])
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


def write_metrics(project_dir: Path, metrics: dict, engine: str | None = None, run_id: str | None = None):
    """Write training metrics to engine-scoped metrics.json."""
    if not engine:
        logger.warning("Skipping metrics write without engine scope")
        return

    if run_id:
        metrics_root = project_dir / "runs" / run_id / "outputs" / ENGINE_SUBDIR / engine
    else:
        metrics_root = project_dir / "outputs" / ENGINE_SUBDIR / engine
    metrics_file = metrics_root / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        temp_file = metrics_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        temp_file.replace(metrics_file)
    except Exception as e:
        logger.error(f"Failed to write metrics for {metrics_root}: {e}")


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
        except FileNotFoundError as e:
            program = cmd[0] if cmd else "<unknown>"
            raise RuntimeError(
                f"Command not found: {program}. Ensure it is installed and available on PATH."
            ) from e
    # Exhausted retries
    logger.error(f"Command failed after retries: {cmd}\nERR: {last_err}")
    raise last_err


def _remove_cli_option(cmd: list[str], option: str) -> list[str]:
    """Return a copy of cmd without the given option and its value (if any)."""
    updated: list[str] = []
    idx = 0
    removed = False
    while idx < len(cmd):
        token = cmd[idx]
        if token == option:
            removed = True
            idx += 1
            if idx < len(cmd) and not str(cmd[idx]).startswith("--"):
                idx += 1
            continue
        updated.append(token)
        idx += 1
    return updated if removed else cmd


def _run_colmap_cmd_with_option_fallback(cmd: list[str], stage_name: str) -> None:
    """Run COLMAP command and drop unsupported options when a build lacks them."""
    current_cmd = list(cmd)
    removed_options: list[str] = []
    for _ in range(6):
        try:
            _run_cmd_with_retry(current_cmd)
            if removed_options:
                logger.warning(
                    "%s succeeded after removing unsupported options: %s",
                    stage_name,
                    ", ".join(removed_options),
                )
            return
        except subprocess.CalledProcessError as exc:
            stderr_text = exc.stderr or ""
            match = _UNRECOGNIZED_OPTION_RE.search(stderr_text)
            if not match:
                raise
            bad_option = match.group(1).strip()
            if not bad_option or bad_option in removed_options:
                raise
            next_cmd = _remove_cli_option(current_cmd, bad_option)
            if next_cmd == current_cmd:
                raise
            removed_options.append(bad_option)
            logger.warning(
                "%s: COLMAP build does not support %s; retrying without it.",
                stage_name,
                bad_option,
            )
            current_cmd = next_cmd
    raise RuntimeError(f"{stage_name} failed due to repeated unsupported COLMAP options.")


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

    colmap_exec = _resolve_colmap_executable()
    colmap_exec_path = shutil.which(colmap_exec) if not os.path.isabs(colmap_exec) else colmap_exec
    if not colmap_exec_path:
        fallback_hint = ""
        if os.name == "nt":
            fallback_hint = " Tried default path C:\\ProgramData\\Bimba3D\\third_party\\colmap\\COLMAP.bat as well."
        msg = (
            f"COLMAP executable not found: {colmap_exec}. "
            "Install COLMAP and/or set COLMAP_EXE to full path (on Windows, COLMAP.bat is recommended)."
            + fallback_hint
        )
        logger.error(msg)
        update_status(output_dir.parent, "failed", progress=0, error=msg, stage="colmap", message=msg)
        raise RuntimeError(msg)
    colmap_exec = colmap_exec_path

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
    # Patch: Always pass explicit camera intrinsics if available
    camera_model = _resolve_colmap_camera_model(p.get("camera_model"))
    single_camera = _parse_boolish(p.get("single_camera"), True)

    feat_cmd = [
        colmap_exec, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model", camera_model,
    ]
    # If explicit camera params/intrinsics are provided in params, pass them to COLMAP.
    camera_params = p.get("camera_params")
    if isinstance(camera_params, str) and camera_params.strip():
        feat_cmd += ["--ImageReader.camera_params", camera_params.strip()]
    # Backward-compatible path for intrinsic dicts.
    cam_intrinsics = p.get("camera_intrinsics") or params.get("camera_intrinsics") if params else None
    # camera_intrinsics should be a dict: {"fx":..., "fy":..., "cx":..., "cy":...}
    if (not (isinstance(camera_params, str) and camera_params.strip())) and cam_intrinsics and all(k in cam_intrinsics for k in ("fx", "fy", "cx", "cy")):
        feat_cmd += [
            "--ImageReader.camera_params",
            f"{cam_intrinsics['fx']},{cam_intrinsics['fy']},{cam_intrinsics['cx']},{cam_intrinsics['cy']}"
        ]
    if p.get("max_image_size"):
        feat_cmd += ["--FeatureExtraction.max_image_size", str(p.get("max_image_size"))]
    else:
        feat_cmd += ["--FeatureExtraction.max_image_size", "1600"]
    if p.get("peak_threshold") is not None:
        feat_cmd += ["--SiftExtraction.peak_threshold", str(p.get("peak_threshold"))]
    if p.get("max_num_features") is not None:
        feat_cmd += ["--SiftExtraction.max_num_features", str(p.get("max_num_features"))]
    try:
        _run_colmap_cmd_with_option_fallback(feat_cmd, "COLMAP feature_extractor")
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
        match_cmd = [colmap_exec, "sequential_matcher", "--database_path", str(database_path)]
    else:
        match_cmd = [colmap_exec, "exhaustive_matcher", "--database_path", str(database_path)]
    if guided is not None:
        match_cmd += ["--FeatureMatching.guided_matching", "1" if guided else "0"]
    if p.get("sift_matching_min_num_inliers") is not None:
        # Compatibility: many COLMAP builds expose this under TwoViewGeometry, not SiftMatching.
        match_cmd += ["--TwoViewGeometry.min_num_inliers", str(p.get("sift_matching_min_num_inliers"))]
    try:
        _run_colmap_cmd_with_option_fallback(match_cmd, "COLMAP matcher")
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

    def _run_mapper_streaming(cmd: list[str]) -> None:
        """Run COLMAP mapper while streaming output to processing.log."""
        popen_cmd, popen_shell = _prepare_subprocess_command(cmd)
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
            for line in proc.stdout:
                try:
                    logger.info(line.rstrip())
                except Exception:
                    pass
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
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, proc.args)
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass

    try:
        mapper_cmd = [
            colmap_exec, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_refine_principal_point", "1" if mapper_refine_principal_point else "0",
            "--Mapper.ba_refine_focal_length", "1" if mapper_refine_focal_length else "0",
            "--Mapper.ba_refine_extra_params", "1" if mapper_refine_extra_params else "0",
        ]
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
        _run_mapper_streaming(mapper_cmd)
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
                colmap_exec,
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
    logger.info("Exported .splat -> %s (%d bytes)", splat_path, splat_path.stat().st_size)

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
        logger.info("Exported .ply -> %s (%d bytes)", ply_path, ply_path.stat().st_size)


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
    context = {
        "logger": logger,
        "update_status": update_status,
        "write_metrics": write_metrics,
        "get_engine_output_dir": _get_engine_output_dir,
        "materialize_eval_previews": _materialize_eval_previews,
        "export_with_gsplat": _export_with_gsplat,
        "parse_step_from_name": _parse_step_from_name,
        "collect_eval_history": _collect_eval_history,
        "write_json_atomic": _write_json_atomic,
        "patch_litegs_opacity_decay": _patch_litegs_opacity_decay,
        "find_latest_litegs_checkpoint": _find_latest_litegs_checkpoint,
        "export_litegs_outputs": _export_litegs_outputs,
    }
    try:
        return run_selected_engine(
            engine,
            image_dir,
            colmap_dir,
            output_dir,
            params,
            resume=resume,
            context=context,
        )
    except Exception as exc:
        if engine != "gsplat":
            raise

        project_dir = Path(image_dir).parent
        root_error = str(exc) if exc else "unknown error"

        if os.name == "nt":
            if "WinError 193" in root_error or "not a valid Win32 application" in root_error:
                guidance = (
                    "CUDA runtime binary mismatch detected for gsplat training on Windows. "
                    "PyTorch CUDA DLL load failed with WinError 193 (wrong-architecture dependency). "
                    "Ensure all CUDA/VS/NVIDIA components are x64 only, remove any x86 CUDA/VS toolchain from PATH, "
                    "and reinstall CUDA-enabled torch/gsplat in the runtime venv. "
                    "Recommended stack: NVIDIA driver + CUDA Toolkit x64 12.5 + VS 2022 Build Tools x64 C++ workload. "
                    "Before reinstalling gsplat, run: call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat\" amd64 . "
                    "CUDA downloads: https://developer.nvidia.com/cuda-downloads . "
                    "Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ ."
                )
            else:
                guidance = (
                    "CUDA is not available for gsplat training on this Windows machine. "
                    "Install compatible x64 components and retry. "
                    "Required: NVIDIA driver + CUDA Toolkit x64 (recommended 12.5 for this build) + Visual Studio 2022 Build Tools x64 with C++ workload. "
                    "Before reinstalling gsplat, run: call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat\" amd64 . "
                    "CUDA compatibility/downloads: https://developer.nvidia.com/cuda-downloads . "
                    "Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . "
                    "Do not use x86 installers."
                )
        else:
            guidance = (
                "CUDA is not available for gsplat training on this machine. "
                "Install compatible NVIDIA driver + CUDA toolkit and retry. "
                "CUDA compatibility/downloads: https://developer.nvidia.com/cuda-downloads ."
            )
        logger.error("gsplat training failed with CUDA/toolchain issue: %s", exc, exc_info=True)
        update_status(
            project_dir,
            "failed",
            progress=55,
            stage="training",
            stage_progress=5,
            message=f"{guidance} Root error: {root_error}",
            error=f"gsplat training requires CUDA + VS Build Tools x64. Root error: {root_error}",
            engine="gsplat",
        )
        raise RuntimeError(f"{guidance} Root error: {root_error}") from exc


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
    shared_output_dir = project_dir / "outputs"
    stop_flag = project_dir / "stop_requested"
    status_file = project_dir / "status.json"

    # Reset stale stop markers from previous runs so a fresh start does not auto-stop.
    if stop_flag.exists():
        try:
            stop_flag.unlink()
            logger.info("Cleared stale stop flag before processing start: %s", stop_flag)
        except Exception as exc:
            logger.warning("Failed to clear stale stop flag %s: %s", stop_flag, exc)

    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as handle:
                status_data = json.load(handle)
            if isinstance(status_data, dict) and status_data.get("stop_requested"):
                status_data["stop_requested"] = False
                status_data.pop("stopped_stage", None)
                status_data.pop("stopped_step", None)
                status_data.pop("stopped_percentage", None)
                status_data["updated_at"] = time.time()
                temp_status = status_file.with_suffix(".tmp")
                with open(temp_status, "w", encoding="utf-8") as handle:
                    json.dump(status_data, handle)
                temp_status.replace(status_file)
                logger.info("Cleared stale stop_requested state in %s", status_file)
        except Exception as exc:
            logger.warning("Failed to reset stale stop_requested state in %s: %s", status_file, exc)

    configured_run_id = str(params.get("run_id") or "").strip()
    if configured_run_id:
        output_dir = project_dir / "runs" / configured_run_id / "outputs"
    else:
        output_dir = shared_output_dir
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
        shared_output_dir.mkdir(parents=True, exist_ok=True)

        # Keep COLMAP sparse outputs shared at project level for all sessions.
        shared_sparse_dir = shared_output_dir / "sparse"
        shared_sparse_dir.mkdir(parents=True, exist_ok=True)

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
            colmap_dir = run_colmap(active_image_dir, shared_output_dir, params)
            logger.info("COLMAP completed")
            try:
                _select_best_sparse_model(shared_sparse_dir, active_image_dir, sparse_preference)
            except Exception as exc:
                logger.warning("Failed to evaluate sparse candidates after COLMAP: %s", exc)
            # Mark COLMAP as completed for frontend tick/green
            update_status(project_dir, "completed", progress=100, stage="colmap", message="✅ Sparse 3D reconstruction complete!")
            time.sleep(1.5)
            stop_reason = None
        elif stage == "train_only":
            sparse_root = shared_sparse_dir
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
            colmap_dir = run_colmap(active_image_dir, shared_output_dir, params)
            logger.info("COLMAP completed")

            colmap_dir = _resolve_sparse_model_for_training(
                shared_sparse_dir,
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
            stop_source = _resolve_stop_source(Path(project_dir) / "stop_requested")
            if stop_source == "backend":
                final_message = f"⏸️ Processing stopped by backend shutdown at step {stop_reason}."
            else:
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
            logger.info("Pipeline stopped by %s at step %s", stop_source, stop_reason)
        else:
            # Check actual stop marker file before setting completed.
            # status.json.stop_requested can be stale (e.g., backend restart/reload)
            # while a local worker still finishes successfully.
            stop_flag_path = Path(project_dir) / "stop_requested"
            status_path = Path(project_dir) / "status.json"
            stop_requested_flag = stop_flag_path.exists()
            if stop_requested_flag:
                final_status = "stopped"
                stop_source = _resolve_stop_source(stop_flag_path)
                if stop_source == "backend":
                    final_message = "⏸️ Processing stopped by backend shutdown."
                else:
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
                logger.info("Pipeline stopped by %s (detected stop_requested)", stop_source)
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
            stop_source = _resolve_stop_source(Path(project_dir) / "stop_requested")
            if stop_source == "backend":
                final_message = f"⏸️ Processing stopped by backend shutdown at step {stop_reason}."
            else:
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
            logger.info("Pipeline stopped by %s at step %s", stop_source, stop_reason)
        else:
            update_status(project_dir, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
