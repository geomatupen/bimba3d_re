import threading
import logging
import shutil
import json
import time
import re
import os
import stat
import struct
import ast
from typing import Optional
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import FileResponse
from bimba3d_backend.app.config import DATA_DIR, ALLOWED_IMAGE_EXTENSIONS
from bimba3d_backend.app.models.project import (
    ProjectResponse,
    StorageRootResponse,
    StatusResponse,
    ProcessParams,
    ProjectListItem,
    CreateProjectRequest,
    UpdateProjectRequest,
    EvaluationMetrics,
    ComparisonRequest,
    ComparisonStatus,
    SparseEditRequest,
    SparseMergeRequest,
    RenameRunRequest,
    CreateRunRequest,
)
from bimba3d_backend.app.services import status, storage, colmap, gsplat, files, sparse_edit, pointsbin
from bimba3d_backend.app.services.worker_mode import normalize_worker_mode, resolve_worker_mode
from bimba3d_backend.worker import pipeline

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)
BEST_SPARSE_META = ".best_sparse_selection.json"
SPARSE_IMAGE_MEMBERSHIP_META = ".sparse_image_membership.json"
SHARED_CONFIG_FILE = "shared_config.json"


def _get_project_shared_config_path(project_dir: Path) -> Path:
    return project_dir / SHARED_CONFIG_FILE


def _extract_shared_config_from_params(params: dict | None) -> dict:
    data = params if isinstance(params, dict) else {}
    shared: dict = {}

    if "images_resize_enabled" in data:
        shared["images_resize_enabled"] = bool(data.get("images_resize_enabled"))

    image_size = data.get("images_max_size")
    if isinstance(image_size, (int, float)):
        shared["images_max_size"] = int(image_size)

    colmap_in = data.get("colmap")
    if isinstance(colmap_in, dict):
        # Store full COLMAP object so base-owned shared behavior remains explicit.
        shared["colmap"] = json.loads(json.dumps(colmap_in))

    return shared


def _merge_shared_config_into_params(params: dict, shared: dict | None) -> dict:
    merged = dict(params)
    if not isinstance(shared, dict):
        return merged

    if "images_resize_enabled" in shared:
        merged["images_resize_enabled"] = bool(shared.get("images_resize_enabled"))

    if "images_max_size" in shared:
        merged["images_max_size"] = shared.get("images_max_size")

    shared_colmap = shared.get("colmap")
    if isinstance(shared_colmap, dict):
        current_colmap = merged.get("colmap") if isinstance(merged.get("colmap"), dict) else {}
        merged["colmap"] = {
            **current_colmap,
            **json.loads(json.dumps(shared_colmap)),
        }

    return merged


def _normalize_shared_doc(raw: dict | None, base_run_id: str | None = None) -> dict:
    doc = raw if isinstance(raw, dict) else {}
    shared = doc.get("shared") if isinstance(doc.get("shared"), dict) else {}
    version = doc.get("version")
    if not isinstance(version, int) or version < 1:
        version = 1

    active_shared = doc.get("active_shared") if isinstance(doc.get("active_shared"), dict) else None

    normalized = {
        "version": version,
        "base_run_id": doc.get("base_run_id") if isinstance(doc.get("base_run_id"), str) else base_run_id,
        "updated_at": doc.get("updated_at") if isinstance(doc.get("updated_at"), str) else None,
        "active_sparse_version": doc.get("active_sparse_version") if isinstance(doc.get("active_sparse_version"), int) else None,
        "active_sparse_updated_at": doc.get("active_sparse_updated_at") if isinstance(doc.get("active_sparse_updated_at"), str) else None,
        "active_shared": active_shared,
        "shared": shared,
    }
    if not normalized["base_run_id"] and base_run_id:
        normalized["base_run_id"] = base_run_id
    return normalized


def _read_project_shared_config(project_dir: Path, base_run_id: str | None = None) -> dict:
    path = _get_project_shared_config_path(project_dir)
    raw = _read_json_if_exists(path)
    normalized = _normalize_shared_doc(raw, base_run_id=base_run_id)

    if isinstance(normalized.get("shared"), dict) and normalized.get("shared"):
        return normalized

    if base_run_id:
        base_run_cfg = _read_json_if_exists(project_dir / "runs" / base_run_id / "run_config.json")
        if isinstance(base_run_cfg, dict):
            resolved = base_run_cfg.get("resolved_params") if isinstance(base_run_cfg.get("resolved_params"), dict) else {}
            inferred_shared = _extract_shared_config_from_params(resolved)
            if inferred_shared:
                normalized["shared"] = inferred_shared
                if not isinstance(normalized.get("active_shared"), dict):
                    normalized["active_shared"] = json.loads(json.dumps(inferred_shared))
                if not isinstance(normalized.get("active_sparse_version"), int):
                    normalized["active_sparse_version"] = int(normalized.get("version") or 1)

    return normalized


def _write_project_shared_config(project_dir: Path, doc: dict) -> None:
    path = _get_project_shared_config_path(project_dir)
    normalized = _normalize_shared_doc(doc)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
    tmp_path.replace(path)


def _read_json_if_exists(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse JSON %s: %s", path, exc)
        return None


def _close_project_log_handlers(project_dir: Path) -> None:
    """Best-effort close of file handlers writing under the project directory."""
    project_dir_resolved = project_dir.resolve()

    def _release_handlers(logger_obj: logging.Logger) -> None:
        for handler in list(logger_obj.handlers):
            base_filename = getattr(handler, "baseFilename", None)
            if not isinstance(base_filename, str):
                continue
            try:
                file_path = Path(base_filename).resolve()
            except Exception:
                continue
            if file_path == project_dir_resolved or project_dir_resolved in file_path.parents:
                try:
                    logger_obj.removeHandler(handler)
                except Exception:
                    pass
                try:
                    handler.flush()
                except Exception:
                    pass
                try:
                    handler.close()
                except Exception:
                    pass

    # Root logger
    _release_handlers(logging.getLogger())

    # Known named loggers
    for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            _release_handlers(logger_obj)


def _remove_readonly_then_retry(func, path, exc_info):
    """shutil.rmtree onerror callback: clear readonly and retry path removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)


def _is_windows_junction(path: Path) -> bool:
    checker = getattr(os.path, "isjunction", None)
    if callable(checker):
        try:
            return bool(checker(str(path)))
        except Exception:
            return False
    return False


def _delete_path_strict(path: Path) -> None:
    """Delete directory/symlink/junction at path if it exists."""
    if not path.exists() and not path.is_symlink():
        return

    if path.is_file():
        path.unlink(missing_ok=True)
        return

    if path.is_symlink():
        path.unlink(missing_ok=True)
        return

    if _is_windows_junction(path):
        os.rmdir(path)
        return

    shutil.rmtree(path, onerror=_remove_readonly_then_retry)


def _clear_restart_artifacts(
    project_dir: Path,
    run_dir: Path,
    *,
    stage: str,
    clear_project_level: bool,
) -> None:
    """Clear generated artifacts so restart begins from a clean state.

    Cleanup is stage-aware:
    - colmap_only: clear sparse artifacts only
    - train_only: clear engine/training artifacts only
    - full: clear both

    Project-level outputs are touched only when restarting the base session.
    """
    clear_colmap = stage in {"full", "colmap_only"}
    clear_training = stage in {"full", "train_only"}

    targets = [
        run_dir / "adaptive_ai",
        run_dir / "processing.log",
        run_dir / "resume_state.json",
    ]

    if clear_colmap:
        targets.append(run_dir / "outputs" / "sparse")
    if clear_training:
        targets.append(run_dir / "outputs" / "engines")

    if clear_project_level:
        if clear_colmap:
            targets.append(project_dir / "outputs" / "sparse")
        if clear_training:
            targets.append(project_dir / "outputs" / "engines")

    for target in targets:
        try:
            _delete_path_strict(target)
        except Exception as exc:
            logger.warning("Failed to clear restart artifact %s: %s", target, exc)


def _apply_run_jitter(params: dict, run_index: int, jitter_factor: float) -> dict:
    """Apply per-run multiplicative jitter to selected learning-rate parameters."""
    if jitter_factor <= 0:
        jitter_factor = 1.0
    if abs(jitter_factor - 1.0) < 1e-9:
        return dict(params)

    out = dict(params)
    multiplier = float(jitter_factor) ** int(run_index)
    lr_defaults = {
        "feature_lr": 2.5e-3,
        "opacity_lr": 5.0e-2,
        "scaling_lr": 5.0e-3,
        "rotation_lr": 1.0e-3,
        "position_lr_init": 1.6e-4,
        "position_lr_final": 1.6e-6,
    }
    for key, default_val in lr_defaults.items():
        base_val = out.get(key, default_val)
        if isinstance(base_val, (int, float)):
            out[key] = float(base_val) * multiplier
    return out


def _wait_for_run_completion(project_id: str, run_id: str, timeout_seconds: int = 0) -> dict:
    """Wait until the targeted run reaches a terminal project status."""
    started_at = time.time()
    terminal_states = {"completed", "done", "failed", "stopped"}
    while True:
        current = status.get_status(project_id)
        current_run = str(current.get("current_run_id") or "")
        current_state = str(current.get("status") or "")
        if current_run == run_id and current_state in terminal_states:
            return current
        if timeout_seconds > 0 and (time.time() - started_at) >= timeout_seconds:
            return current
        time.sleep(2)


def _run_batch_process(
    project_id: str,
    base_params: dict,
    run_count: int,
    run_name_prefix: str | None,
    jitter_factor: float,
    continue_on_failure: bool,
) -> None:
    """Run multiple sessions sequentially using the same base config."""
    try:
        prefix = _sanitize_run_token(run_name_prefix or "") if run_name_prefix else ""
        for idx in range(max(1, int(run_count))):
            run_params = json.loads(json.dumps(base_params))
            run_params["run_count"] = 1
            run_params["resume"] = False
            run_params["restart_fresh"] = False

            if idx > 0:
                run_params["stage"] = "train_only"

            if prefix:
                run_params["run_name"] = f"{prefix}_session{idx + 1}"
            elif run_params.get("run_name"):
                base_name = _sanitize_run_token(str(run_params.get("run_name") or "batch")) or "batch"
                run_params["run_name"] = f"{base_name}_{idx + 1}"

            run_params = _apply_run_jitter(run_params, idx, jitter_factor)
            logger.info("Batch %s/%s starting for %s (run_name=%s)", idx + 1, run_count, project_id, run_params.get("run_name"))

            response = process_project(project_id, ProcessParams(**run_params))
            run_id = str(response.get("run_id") or "")
            if not run_id:
                raise RuntimeError("Batch run started without run_id")

            final_status = _wait_for_run_completion(project_id, run_id)
            final_state = str(final_status.get("status") or "")
            if final_state in {"failed", "stopped"} and not continue_on_failure:
                logger.warning("Batch halted for %s after run %s ended with %s", project_id, run_id, final_state)
                break
    except Exception as exc:
        logger.error("Batch process failed for %s: %s", project_id, exc, exc_info=True)
        status.update_status(project_id, "failed", error=str(exc))


def _sanitize_run_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned[:80]


def _build_default_run_name(project_label: str | None, runs_root: Path | None = None) -> str:
    prefix = _sanitize_run_token(project_label or "project") or "project"
    if not runs_root:
        return f"{prefix}_session1"

    next_idx = 1
    pattern = re.compile(rf"^{re.escape(prefix)}_session(\d+)$")
    try:
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if not match:
                continue
            next_idx = max(next_idx, int(match.group(1)) + 1)
    except Exception:
        pass

    return f"{prefix}_session{next_idx}"


def _rewrite_auto_run_name_prefix(
    run_name_requested: str,
    project_id: str,
    project_label: str | None,
) -> str:
    """If request looks like auto id-based name, switch to project-name prefix."""
    requested = (run_name_requested or "").strip()
    if not requested:
        return requested

    project_id_prefix = _sanitize_run_token(project_id) or "project"
    preferred_prefix = _sanitize_run_token(project_label or project_id) or project_id_prefix
    if preferred_prefix == project_id_prefix:
        return requested

    # Match auto-generated shapes like:
    # <project_id_prefix>_YYYYMMDD_HHMMSS, <project_id_prefix>_YYYYMMDD_HHMMSS_01,
    # or <project_id_prefix>_sessionN
    match = re.fullmatch(
        rf"{re.escape(project_id_prefix)}_((?:\d{{8}}_\d{{6}}(?:_\d{{2}})?)|(?:session\d+))",
        requested,
    )
    if not match:
        return requested

    return f"{preferred_prefix}_{match.group(1)}"


def _resolve_unique_run_id(runs_root: Path, preferred_name: str) -> str:
    base = _sanitize_run_token(preferred_name) or _build_default_run_name("project")
    candidate = base
    idx = 1
    while (runs_root / candidate).exists():
        candidate = f"{base}_{idx:02d}"
        idx += 1
    return candidate


def _read_sparse_image_names(candidate_dir: Path) -> list[str]:
    images_bin = candidate_dir / "images.bin"
    if not images_bin.exists():
        return []
    try:
        from bimba3d_backend.worker.colmap_loader import read_images_binary  # pylint: disable=import-outside-toplevel
        images = read_images_binary(images_bin)
        names = [entry.get("name") for entry in images.values() if isinstance(entry.get("name"), str)]
        names.sort()
        return names
    except Exception as exc:
        logger.debug("Failed to read image names from %s: %s", images_bin, exc)
        return []
def _load_sparse_metadata(sparse_root: Path) -> tuple[dict | None, Path]:
    meta_path = sparse_root / BEST_SPARSE_META
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            logger.debug("Failed to parse sparse metadata at %s: %s", meta_path, exc)
    return meta, meta_path



def _colmap_to_opengl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax, ay, az = COLMAP_TO_OPENGL
    return float(ax * x), float(ay * y), float(az * z)


def _resolve_sparse_candidate_dir(sparse_root: Path, rel_path: str | None) -> Path:
    if not rel_path or rel_path in {".", ""}:
        return sparse_root
    try:
        candidate = (sparse_root / rel_path).resolve()
        base = sparse_root.resolve()
        if candidate == base or base in candidate.parents:
            return candidate
    except Exception:
        pass
    return sparse_root


def _read_sparse_stats(candidate_dir: Path) -> tuple[int | None, int | None]:
    images = None
    points = None
    try:
        with open(candidate_dir / "images.bin", "rb") as handle:
            header = handle.read(8)
            if len(header) == 8:
                images = int(struct.unpack("<Q", header)[0])
    except Exception:
        pass
    try:
        with open(candidate_dir / "points3D.bin", "rb") as handle:
            header = handle.read(8)
            if len(header) == 8:
                points = int(struct.unpack("<Q", header)[0])
    except Exception:
        pass
    return images, points


def _resolve_sparse_candidate_for_edit(project_dir: Path, requested: str | None) -> tuple[Path, str]:
    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    meta, _ = _load_sparse_metadata(sparse_root)
    token = (requested or "").strip()
    token_lower = token.lower()

    def candidate_for(rel_path: str) -> Path | None:
        resolved = _resolve_sparse_candidate_dir(sparse_root, rel_path)
        if resolved.exists() and (
            (resolved / "points3D.bin").exists() or (resolved / "points3D.txt").exists()
        ):
            return resolved
        return None

    if token and token_lower not in {"best", "auto"}:
        normalized = "." if token in {"", ".", "root"} else token
        candidate_dir = candidate_for(normalized)
        if not candidate_dir:
            raise HTTPException(status_code=404, detail="Requested sparse reconstruction not found")
        return candidate_dir, normalized

    preferred_rel = meta.get("relative_path") if isinstance(meta, dict) else None
    if preferred_rel:
        candidate_dir = candidate_for(preferred_rel)
        if candidate_dir:
            return candidate_dir, preferred_rel

    root_dir = candidate_for(".")
    if root_dir:
        return root_dir, "."

    try:
        for child in sorted(p for p in sparse_root.iterdir() if p.is_dir()):
            rel_path = os.path.relpath(child, sparse_root)
            candidate_dir = candidate_for(rel_path)
            if candidate_dir:
                if rel_path in {"", "."}:
                    rel_path = "."
                return candidate_dir, rel_path
    except Exception as exc:
        logger.debug("Failed to enumerate sparse directories for edit: %s", exc)

    raise HTTPException(status_code=404, detail="No sparse reconstruction available to edit")


def _update_sparse_candidate_points(project_dir: Path, candidate_rel: str, points: int | None) -> None:
    if points is None:
        return
    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        return
    meta, meta_path = _load_sparse_metadata(sparse_root)
    if not isinstance(meta, dict):
        meta = {}
    candidates = meta.setdefault("candidates", [])
    norm_rel = candidate_rel or "."
    updated = False
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        entry_rel = entry.get("relative_path") or "."
        if entry_rel == norm_rel:
            entry["points"] = points
            updated = True
            break
    if not updated:
        candidates.append({"relative_path": norm_rel, "points": points})
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        logger.debug("Failed to update sparse metadata at %s: %s", meta_path, exc)


def _is_colmap_reconstruction_dir(candidate_dir: Path) -> bool:
    """Return True when a directory looks like a valid COLMAP sparse reconstruction."""
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return False
    has_cameras = (candidate_dir / "cameras.bin").exists() or (candidate_dir / "cameras.txt").exists()
    has_images = (candidate_dir / "images.bin").exists() or (candidate_dir / "images.txt").exists()
    has_points = (candidate_dir / "points3D.bin").exists() or (candidate_dir / "points3D.txt").exists()
    return bool(has_cameras and has_images and has_points)


def _has_colmap_sparse_outputs(sparse_root: Path) -> bool:
    """Check whether sparse root contains at least one valid COLMAP reconstruction dir."""
    if not sparse_root.exists() or not sparse_root.is_dir():
        return False

    if _is_colmap_reconstruction_dir(sparse_root):
        return True

    try:
        for child in sparse_root.iterdir():
            if child.is_dir() and _is_colmap_reconstruction_dir(child):
                return True
    except Exception:
        return False

    return False


def _base_session_colmap_ready(project_dir: Path, base_session_id: str | None) -> bool:
    """True when base session has COLMAP sparse data (shared or run-local)."""
    if not base_session_id:
        return False

    shared_sparse_root = project_dir / "outputs" / "sparse"
    if _has_colmap_sparse_outputs(shared_sparse_root):
        return True

    base_run_sparse_root = project_dir / "runs" / base_session_id / "outputs" / "sparse"
    return _has_colmap_sparse_outputs(base_run_sparse_root)

logger = logging.getLogger(__name__)

router = APIRouter()

ENGINE_SUBDIR = "engines"
ENGINE_NAME_RE = re.compile(r"^[a-z0-9_-]+$", re.IGNORECASE)

# Map EXIF GPS tag id for quick lookup
EXIF_GPS_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == "GPSInfo":
        EXIF_GPS_TAG = k
        break


def _normalize_engine_name(engine: str | None) -> str | None:
    if engine is None:
        return None
    candidate = str(engine).strip()
    if not candidate:
        return None
    if not ENGINE_NAME_RE.fullmatch(candidate):
        return None
    return candidate.lower()


def _sanitize_engine(engine: str | None) -> str | None:
    if engine is None:
        return None
    normalized = _normalize_engine_name(engine)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid engine selector")
    return normalized


def _resolve_output_path(project_dir: Path, relative_path: str | Path, engine: str | None = None) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path request")
    base = project_dir / "outputs"
    if engine:
        base = base / ENGINE_SUBDIR / engine
    return base / rel


def _engine_search_order(project_id: str, sanitized_engine: str | None) -> tuple[list[str], str | None]:
    if sanitized_engine:
        return [sanitized_engine], sanitized_engine
    inferred = _infer_engine(project_id)
    order: list[str] = []
    if inferred:
        order.append(inferred)
    return order, inferred


def _infer_engine(project_id: str) -> str | None:
    try:
        info = status.get_status(project_id)
        normalized = _normalize_engine_name(info.get("engine"))
        if normalized:
            return normalized
    except Exception:
        pass
    project_dir = DATA_DIR / project_id
    engines_root = project_dir / "outputs" / ENGINE_SUBDIR
    if engines_root.exists() and engines_root.is_dir():
        for entry in sorted(p for p in engines_root.iterdir() if p.is_dir()):
            normalized = _normalize_engine_name(entry.name)
            if normalized:
                return normalized
    return None


def _find_existing_path(
    project_id: str,
    relative_path: str | Path,
    engine: str | None,
    run_id: str | None = None,
    *,
    expect_directory: bool = False,
) -> tuple[Path | None, str | None, str | None, str | None]:
    sanitized = _sanitize_engine(engine)
    search_order, inferred = _engine_search_order(project_id, sanitized)
    project_dir = DATA_DIR / project_id
    for candidate in search_order:
        if run_id:
            candidate_path = project_dir / "runs" / run_id / "outputs"
            if candidate:
                candidate_path = candidate_path / ENGINE_SUBDIR / candidate
            candidate_path = candidate_path / Path(relative_path)
        else:
            candidate_path = _resolve_output_path(project_dir, relative_path, candidate)
        if expect_directory:
            if candidate_path.exists() and candidate_path.is_dir():
                return candidate_path, candidate, sanitized, inferred
        else:
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path, candidate, sanitized, inferred
    return None, None, sanitized, inferred


def _rational_to_float(value: tuple) -> float:
    # Convert EXIF rational tuple (num, den) into float; guard against zero division
    # Handle both simplified float and rational tuple formats
    if isinstance(value, (int, float)):
        return float(value)
    
    # Pillow IFDRational has numerator/denominator attributes
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        den = value.denominator
        return float(value.numerator) / float(den) if den else 0.0
    
    # Tuple format (num, den)
    try:
        num, den = value
        return float(num) / float(den) if den else 0.0
    except Exception:
        return 0.0
    return float(num) / float(den) if den else 0.0


def _dms_to_dd(dms: tuple, ref: str) -> float:
    # Convert degrees/minutes/seconds to decimal degrees with hemisphere reference
    # Handle both simplified tuple (37.0, 49.0, 11.63) and rational ((37,1), (49,1), (1163,100))
    if isinstance(dms, (int, float)):
        # Already in decimal format
        dd = float(dms)
    elif len(dms) == 3:
        deg = _rational_to_float(dms[0])
        minutes = _rational_to_float(dms[1])
        seconds = _rational_to_float(dms[2])
        dd = deg + (minutes / 60.0) + (seconds / 3600.0)
    else:
        return 0.0
    
    if ref in ["S", "W"]:
        dd = -dd
    return dd


def extract_gps(filepath: Path) -> Optional[dict]:
    """Extract GPS lat/lon from image EXIF if available."""
    try:
        if EXIF_GPS_TAG is None:
            return None

        with Image.open(filepath) as img:
            exif = img._getexif()  # noqa: SLF001 Pillow private API is standard for EXIF
            if not exif or EXIF_GPS_TAG not in exif:
                return None

            gps_info = exif.get(EXIF_GPS_TAG, {})
            gps_data = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            lat = gps_data.get("GPSLatitude")
            lat_ref = gps_data.get("GPSLatitudeRef")
            lon = gps_data.get("GPSLongitude")
            lon_ref = gps_data.get("GPSLongitudeRef")

            if lat is None or lon is None or lat_ref is None or lon_ref is None:
                return None

            return {
                "lat": _dms_to_dd(lat, lat_ref),
                "lon": _dms_to_dd(lon, lon_ref),
            }
    except Exception as e:  # pragma: no cover - best effort extraction
        logger.debug(f"EXIF GPS extraction failed for {filepath}: {e}")
        return None


@router.get("", response_model=list[ProjectListItem], include_in_schema=False)
@router.get("/", response_model=list[ProjectListItem])
def list_projects():
    """List all projects with status and basic metadata."""
    try:
        projects: list[ProjectListItem] = []
        if not DATA_DIR.exists():
            return projects

        for project_dir in sorted(DATA_DIR.iterdir()):
            if not project_dir.is_dir():
                continue
            project_id = project_dir.name
            project_status = status.get_status(project_id)
            current_status = project_status.get("status", "pending")
            progress = int(project_status.get("progress", 0) or 0)
            has_outputs = (
                (project_dir / "outputs" / "engines" / "gsplat" / "splats.splat").exists()
                or (project_dir / "outputs" / "engines" / "gsplat" / "splats.ply").exists()
                or (project_dir / "outputs" / "engines" / "gsplat" / "metadata.json").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "splats.splat").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "splats.ply").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "metadata.json").exists()
            )
            runs_root = project_dir / "runs"
            session_count = 0
            if runs_root.exists() and runs_root.is_dir():
                try:
                    session_count = sum(1 for p in runs_root.iterdir() if p.is_dir())
                except Exception:
                    session_count = 0
            modified_at = None
            try:
                status_file = project_dir / "status.json"
                mtime_source = status_file if status_file.exists() else project_dir
                modified_at = datetime.utcfromtimestamp(mtime_source.stat().st_mtime).isoformat() + "Z"
            except Exception:
                modified_at = None
            projects.append(
                ProjectListItem(
                    project_id=project_id,
                    name=project_status.get("name"),
                    status=current_status,
                    progress=progress,
                    created_at=project_status.get("created_at"),
                    modified_at=modified_at,
                    has_outputs=has_outputs,
                    session_count=session_count,
                )
            )

        return projects
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list projects")


@router.post("", response_model=ProjectResponse, include_in_schema=False)
@router.post("/", response_model=ProjectResponse)
def create_project(payload: CreateProjectRequest | None = Body(None)):
    """Create a new project with optional human-friendly name."""
    try:
        storage_root_id = payload.storage_root_id if payload else None
        storage_path = payload.storage_path if payload else None
        try:
            project_root = storage.resolve_storage_root(storage_root_id=storage_root_id, storage_path=storage_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        project_id, project_dir = storage.create_project(base_dir=project_root)
        provided_name = (payload.name.strip() if payload and payload.name else None)

        # Initialize status file with name
        status.initialize_status(project_id, name=provided_name)

        logger.info(f"Created project: {project_id} name={provided_name}")
        project_status = status.get_status(project_id)
        return {
            "project_id": project_id,
            "name": project_status.get("name"),
            "created_at": project_status.get("created_at"),
        }
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create project")


@router.get("/storage-roots", response_model=list[StorageRootResponse])
def get_storage_roots():
    """List available storage roots for project creation."""
    try:
        roots = storage.list_storage_roots()
        return [StorageRootResponse(**entry) for entry in roots]
    except Exception as e:
        logger.error(f"Error listing storage roots: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list storage roots")


@router.post("/{project_id}/images")
async def upload_images(project_id: str, images: list[UploadFile] = File(...)):
    """Upload images to a project."""
    try:
        project_dir = DATA_DIR / project_id
        images_dir = project_dir / "images"
        thumbnails_dir = images_dir / "thumbnails"
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        images_dir.mkdir(parents=True, exist_ok=True)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        uploaded_count = 0
        invalid_files: list[str] = []
        allowed_ext_text = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        gps_records: dict[str, dict] = {}
        gps_file = images_dir / "locations.json"
        if gps_file.exists():
            try:
                gps_records = json.loads(gps_file.read_text())
            except Exception:
                gps_records = {}
        
        for img in images:
            # Validate file extension
            file_ext = Path(img.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(f"Skipped invalid image: {img.filename}")
                invalid_files.append(img.filename)
                continue
            
            # Read and save file
            content = await img.read()
            file_path = images_dir / img.filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Generate thumbnail
            try:
                with Image.open(file_path) as image:
                    # Create thumbnail (max 100x100 for faster loading)
                    image.thumbnail((100, 100), Image.Resampling.LANCZOS)
                    
                    # Save thumbnail with quality optimization
                    thumbnail_path = thumbnails_dir / img.filename
                    image.save(thumbnail_path, "JPEG", quality=80, optimize=True)
                    
                    logger.info(f"Generated thumbnail for: {img.filename}")
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for {img.filename}: {str(e)}")

            # Extract GPS if available
            gps = extract_gps(file_path)
            if gps:
                gps_records[img.filename] = gps
                logger.info(f"Captured GPS for {img.filename}: {gps}")
            
            uploaded_count += 1
            logger.info(f"Uploaded image: {img.filename} to {project_id}")
        
        if uploaded_count == 0:
            invalid_list = ", ".join(invalid_files[:10])
            if len(invalid_files) > 10:
                invalid_list = f"{invalid_list}, ..."
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No valid images uploaded. Allowed formats: {allowed_ext_text}. "
                    f"Invalid files: {invalid_list}"
                ),
            )
        
        # Persist GPS metadata if any
        try:
            if gps_records:
                gps_file.write_text(json.dumps(gps_records, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write GPS metadata for {project_id}: {e}")

        return {"status": "uploaded", "count": uploaded_count}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload images")


@router.post("/{project_id}/process")
def process_project(project_id: str, params: ProcessParams | None = Body(None)):
    """Start processing a project in background thread."""
    try:
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if images exist
        images_dir = project_dir / "images"
        if not images_dir.exists() or not list(images_dir.glob("*")):
            raise HTTPException(status_code=400, detail="No images in project")

        # Starting/resuming should always clear stale stop markers from previous runs.
        stop_flag = project_dir / "stop_requested"
        try:
            if stop_flag.exists():
                stop_flag.unlink()
        except Exception as exc:
            logger.warning("Failed to clear stale stop flag for %s: %s", project_id, exc)

        try:
            status.clear_stop_state(project_id)
        except Exception as exc:
            logger.warning("Failed to clear stale stop metadata for %s: %s", project_id, exc)

        # Prepare params payload with defaults (engine defaults to gsplat)
        requested_params = params.dict(exclude_none=True) if params else {}
        params_payload = dict(requested_params)

        # Repro defaults for provided COLMAP pipelines.
        params_payload.setdefault("stage", "train_only")
        params_payload.setdefault("max_steps", 15000)
        params_payload.setdefault("log_interval", 100)
        params_payload.setdefault("save_interval", 2500)
        params_payload.setdefault("splat_export_interval", 2500)
        params_payload.setdefault("tune_end_step", 15000)
        params_payload.setdefault("tune_interval", 100)
        params_payload.setdefault("batch_size", 1)
        params_payload.setdefault("densify_from_iter", 500)
        params_payload.setdefault("densify_until_iter", 10000)
        params_payload.setdefault("densification_interval", 100)
        params_payload.setdefault("densify_grad_threshold", 0.0002)
        params_payload.setdefault("opacity_reset_interval", 3000)
        params_payload.setdefault("lambda_dssim", 0.2)
        params_payload.setdefault("feature_lr", 2.5e-3)
        params_payload.setdefault("opacity_lr", 5.0e-2)
        params_payload.setdefault("scaling_lr", 5.0e-3)
        params_payload.setdefault("rotation_lr", 1.0e-3)
        params_payload.setdefault("percent_dense", 0.01)
        params_payload.setdefault("position_lr_init", 1.6e-4)
        params_payload.setdefault("position_lr_final", 1.6e-6)
        params_payload.setdefault("position_lr_delay_mult", 0.01)
        params_payload.setdefault("position_lr_max_steps", 30000)

        # Validate/resolve worker runtime mode
        try:
            normalize_worker_mode(params_payload.get("worker_mode"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        resolved_worker_mode = resolve_worker_mode(params_payload.get("worker_mode"))
        params_payload["worker_mode"] = resolved_worker_mode

        engine = params_payload.get("engine", "gsplat")
        if engine not in {"gsplat", "litegs"}:
            raise HTTPException(status_code=400, detail=f"Invalid training engine: {engine}")
        params_payload["engine"] = engine

        # Optional sequential batch orchestration.
        try:
            run_count = int(requested_params.get("run_count") or 1)
        except Exception:
            run_count = 1
        if run_count < 1:
            run_count = 1

        if run_count > 1:
            if bool(requested_params.get("resume")):
                raise HTTPException(status_code=400, detail="Batch resume is not supported. Use Batch Continue from a fresh start.")

            jitter_factor = float(requested_params.get("run_jitter_factor") or 1.0)
            run_name_prefix = str(requested_params.get("run_name_prefix") or "").strip() or None
            continue_on_failure = bool(requested_params.get("continue_on_failure", True))

            # Reuse existing conflict checks before launching the batch orchestrator.
            if resolved_worker_mode == "docker":
                running_workers = colmap.get_project_worker_container_ids(project_id)
                if running_workers:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A docker worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )
            else:
                if pipeline.is_local_project_active(project_id):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A local worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )

            status.update_status(
                project_id,
                "processing",
                progress=1,
                engine=engine,
                worker_mode=resolved_worker_mode,
                stop_requested=False,
                message=f"Batch queued: {run_count} runs (jitter={jitter_factor}).",
                error=None,
            )

            batch_seed_params = dict(requested_params)
            batch_seed_params["run_count"] = 1
            batch_seed_params["run_jitter_factor"] = jitter_factor
            batch_seed_params["run_name_prefix"] = run_name_prefix
            batch_seed_params["continue_on_failure"] = continue_on_failure

            thread = threading.Thread(
                target=_run_batch_process,
                args=(project_id, batch_seed_params, run_count, run_name_prefix, jitter_factor, continue_on_failure),
                daemon=True,
            )
            thread.start()

            logger.info("Started batch processing for %s with %s runs", project_id, run_count)
            return {
                "status": "batch_processing_started",
                "run_count": run_count,
                "jitter_factor": jitter_factor,
                "continue_on_failure": continue_on_failure,
                "run_name_prefix": run_name_prefix,
            }

        # Create a per-run session directory under project/runs/<run_name>.
        project_status = status.get_status(project_id)
        project_label = None
        if isinstance(project_status, dict):
            project_label = project_status.get("name") or project_status.get("project_id")
        raw_run_name_requested = str(requested_params.get("run_name") or "").strip()
        restart_fresh_requested = bool(requested_params.get("restart_fresh"))
        runs_root = project_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        default_run_name = _build_default_run_name(project_label, runs_root)
        if restart_fresh_requested and raw_run_name_requested and (runs_root / raw_run_name_requested).exists():
            run_name_requested = raw_run_name_requested
        else:
            run_name_requested = _rewrite_auto_run_name_prefix(raw_run_name_requested, project_id, project_label)
        requested_existing_run = runs_root / run_name_requested if run_name_requested else None
        if (
            requested_existing_run
            and requested_existing_run.exists()
            and requested_existing_run.is_dir()
        ):
            # If the client explicitly targets an existing session, keep using it.
            # This avoids accidental auto-cloning like "session3_01" when processing "session3".
            run_id = run_name_requested
        else:
            run_id = _resolve_unique_run_id(runs_root, run_name_requested or default_run_name)
        run_dir = project_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if restart_fresh_requested:
            stage_for_cleanup = str(params_payload.get("stage") or "train_only")
            base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
            is_base_run = bool(base_session_id) and run_id == str(base_session_id)
            logger.info("Restart requested from scratch for %s/%s; clearing previous artifacts", project_id, run_id)
            _clear_restart_artifacts(
                project_dir,
                run_dir,
                stage=stage_for_cleanup,
                clear_project_level=is_base_run,
            )

        params_payload["run_id"] = run_id
        params_payload["run_name"] = run_id

        # If no base session is defined yet, the first run becomes the base session.
        if not project_status.get("base_session_id"):
            try:
                status.update_base_session_id(project_id, run_id)
            except Exception as exc:
                logger.warning("Failed to set base session for %s: %s", project_id, exc)

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        is_base_run = bool(base_session_id) and run_id == str(base_session_id)

        # Canonical shared config is project-level and base-owned.
        # Non-base runs always inherit shared values from this source.
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else run_id)
        shared_doc["base_run_id"] = str(base_session_id) if base_session_id else run_id
        active_shared = shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else None
        inherited_shared = active_shared if active_shared else (shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {})

        incoming_shared = _extract_shared_config_from_params(params_payload)
        requested_stage = str(params_payload.get("stage") or "train_only")
        stage_includes_colmap = requested_stage in {"full", "colmap_only"}
        if is_base_run:
            if stage_includes_colmap and incoming_shared and incoming_shared != shared_doc.get("shared"):
                shared_doc["shared"] = incoming_shared
                shared_doc["version"] = int(shared_doc.get("version") or 1) + 1
                shared_doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
                try:
                    _write_project_shared_config(project_dir, shared_doc)
                except Exception as exc:
                    logger.warning("Failed to persist shared config for %s: %s", project_id, exc)
            elif not _get_project_shared_config_path(project_dir).exists():
                # Persist initial shared doc even if unchanged so non-base runs can inherit reliably.
                shared_doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
                try:
                    _write_project_shared_config(project_dir, shared_doc)
                except Exception as exc:
                    logger.warning("Failed to persist initial shared config for %s: %s", project_id, exc)
            # When base run does not include COLMAP this launch, keep using active shared values.
            if not stage_includes_colmap:
                params_payload = _merge_shared_config_into_params(params_payload, inherited_shared)
        else:
            # Ignore shared overrides coming from non-base runs.
            params_payload = _merge_shared_config_into_params(params_payload, inherited_shared)

        effective_shared_for_run = _extract_shared_config_from_params(params_payload)
        effective_shared_version = (
            int(shared_doc.get("version") or 1)
            if is_base_run and stage_includes_colmap
            else int(shared_doc.get("active_sparse_version") or shared_doc.get("version") or 1)
        )
        params_payload["shared_config_version"] = effective_shared_version
        params_payload["shared_base_run_id"] = shared_doc.get("base_run_id")

        # Persist run configuration for reproducibility (requested + resolved params).
        try:
            run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_config_payload = {
                "project_id": project_id,
                "run_id": run_id,
                "run_name": run_id,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "requested_params": requested_params,
                "resolved_params": params_payload,
                "shared_config_version": effective_shared_version,
                "shared_base_run_id": shared_doc.get("base_run_id"),
                "shared_config_snapshot": effective_shared_for_run,
            }

            run_config_latest = project_dir / "run_config.json"
            run_configs_dir = project_dir / "run_configs"
            run_configs_dir.mkdir(parents=True, exist_ok=True)
            run_config_versioned = run_configs_dir / f"run_config_{run_timestamp}.json"
            run_config_session = run_dir / "run_config.json"

            for target_path in (run_config_latest, run_config_versioned, run_config_session):
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(run_config_payload, handle, indent=2)
                tmp_path.replace(target_path)

            logger.info("Saved run configuration: %s", run_config_latest)
        except Exception as exc:
            logger.warning("Failed to persist run configuration for %s: %s", project_id, exc)

        # Prevent overlapping runs for the same project.
        if resolved_worker_mode == "docker":
            running_workers = colmap.get_project_worker_container_ids(project_id)
            if running_workers:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A docker worker is already running for this project. "
                        "Stop it first or wait for it to finish."
                    ),
                )
        else:
            if pipeline.is_local_project_active(project_id):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A local worker is already running for this project. "
                        "Stop it first or wait for it to finish."
                    ),
                )

        # Update status to processing with the resolved engine
        status.update_status(
            project_id,
            "processing",
            progress=5,
            engine=engine,
            worker_mode=resolved_worker_mode,
            current_run_id=run_id,
            stop_requested=False,
            message=f"Processing started ({resolved_worker_mode} mode).",
            error=None,
        )

        # Start processing in background thread
        # Pass optional parameters to pipeline
        thread = threading.Thread(
            target=pipeline.run_full_pipeline,
            args=(project_id, params_payload),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started processing for project: {project_id}")
        return {"status": "processing_started", "run_id": run_id, "run_name": run_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        status.update_status(project_id, "failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start processing")


@router.get("/{project_id}/status", response_model=StatusResponse)
def get_status_endpoint(project_id: str):
    """Get project status."""
    try:
        from bimba3d_backend.app.services.resume import can_resume_project
        
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_status = status.get_status(project_id)
        
        if project_status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Status not found")
        
        # Add resume info
        resume_info = can_resume_project(project_id)
        project_status["can_resume"] = resume_info["can_resume"]
        project_status["last_completed_step"] = resume_info.get("last_checkpoint_step")
        
        return StatusResponse(**project_status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@router.post("/{project_id}/stop")
def request_stop(project_id: str):
    """Signal an in-flight job to stop gracefully and export final artifacts."""
    try:
        project_dir = DATA_DIR / project_id

        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        stop_flag = project_dir / "stop_requested"
        stop_flag.write_text("stop")

        # Also force-stop docker worker container if one is still running.
        try:
            stopped = colmap.stop_project_worker_containers(project_id)
            if stopped:
                logger.info("Stopped %d active worker container(s) for project %s", stopped, project_id)
        except Exception as exc:
            logger.warning("Failed to force-stop worker container for %s: %s", project_id, exc)

        # Mark status so the UI can reflect stopping state
        status.update_status(
            project_id,
            "stopping",
            progress=status.get_status(project_id).get("progress", 0),
            stop_requested=True,
        )

        logger.info(f"Stop requested for project: {project_id}")
        return {"status": "stop_requested"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting stop: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request stop")


@router.get("/{project_id}/runs")
def list_project_runs(project_id: str):
    """List per-project run sessions with key metadata for UI run selection."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        base_colmap_ready = _base_session_colmap_ready(project_dir, base_session_id)
        can_create_session = bool(base_colmap_ready)
        can_create_session_reason = None if can_create_session else "Complete COLMAP on the base session before creating new sessions."

        if not runs_root.exists():
            return {
                "runs": [],
                "base_session_id": base_session_id,
                "base_colmap_ready": base_colmap_ready,
                "can_create_session": can_create_session,
                "can_create_session_reason": can_create_session_reason,
            }

        project_has_completed_outputs = any(
            p.exists()
            for p in (
                project_dir / "outputs" / "engines" / "gsplat" / "splats.splat",
                project_dir / "outputs" / "engines" / "gsplat" / "splats.ply",
                project_dir / "outputs" / "engines" / "gsplat" / "metadata.json",
                project_dir / "outputs" / "engines" / "litegs" / "splats.splat",
                project_dir / "outputs" / "engines" / "litegs" / "splats.ply",
                project_dir / "outputs" / "engines" / "litegs" / "metadata.json",
            )
        )
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        current_shared_version = int(shared_doc.get("version") or 1)
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None

        runs: list[dict] = []
        for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
            run_id = run_dir.name
            run_config_path = run_dir / "run_config.json"
            run_config = _read_json_if_exists(run_config_path)
            saved_at = run_config.get("saved_at") if isinstance(run_config, dict) else None
            resolved = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
            requested = run_config.get("requested_params") if isinstance(run_config, dict) and isinstance(run_config.get("requested_params"), dict) else {}
            has_completed_outputs = any(
                p.exists()
                for p in (
                    run_dir / "outputs" / "engines" / "gsplat" / "splats.splat",
                    run_dir / "outputs" / "engines" / "gsplat" / "splats.ply",
                    run_dir / "outputs" / "engines" / "gsplat" / "metadata.json",
                    run_dir / "outputs" / "engines" / "litegs" / "splats.splat",
                    run_dir / "outputs" / "engines" / "litegs" / "splats.ply",
                    run_dir / "outputs" / "engines" / "litegs" / "metadata.json",
                )
            )
            is_base_run = run_id == base_session_id
            is_completed = has_completed_outputs or (is_base_run and project_has_completed_outputs)
            run_shared_version = run_config.get("shared_config_version") if isinstance(run_config, dict) else None
            if not isinstance(run_shared_version, int):
                run_shared_version = None
            shared_outdated = bool(
                active_sparse_version is not None
                and run_shared_version is not None
                and run_shared_version < active_sparse_version
            )

            adaptive_runs_dir = run_dir / "adaptive_ai" / "runs"
            adaptive_logs = sorted(adaptive_runs_dir.glob("*.jsonl")) if adaptive_runs_dir.exists() else []
            adaptive_events = 0
            for log_path in adaptive_logs:
                try:
                    with log_path.open("r", encoding="utf-8") as f:
                        adaptive_events += sum(1 for _ in f)
                except Exception:
                    continue

            runs.append(
                {
                    "run_id": run_id,
                    "run_name": (
                        (run_config.get("run_name") if isinstance(run_config, dict) else None)
                        or (resolved.get("run_name") if isinstance(resolved, dict) else None)
                        or (requested.get("run_name") if isinstance(requested, dict) else None)
                        or run_id
                    ),
                    "saved_at": saved_at,
                    "mode": resolved.get("mode") or requested.get("mode"),
                    "stage": resolved.get("stage") or requested.get("stage"),
                    "engine": resolved.get("engine") or requested.get("engine"),
                    "max_steps": resolved.get("max_steps") or requested.get("max_steps"),
                    "tune_scope": resolved.get("tune_scope") or requested.get("tune_scope"),
                    "adaptive_event_count": adaptive_events,
                    "has_run_config": run_config_path.exists(),
                    "has_run_log": (run_dir / "processing.log").exists(),
                    "session_status": "completed" if is_completed else "pending",
                    "is_base": is_base_run,
                    "shared_config_version": run_shared_version,
                    "active_sparse_shared_version": active_sparse_version,
                    "shared_outdated": shared_outdated,
                }
            )

        # Fallback: if base session is missing/deleted, promote latest run as base.
        if runs and (not base_session_id or not any(r["run_id"] == base_session_id for r in runs)):
            fallback_base = runs[0]["run_id"]
            try:
                status.update_base_session_id(project_id, fallback_base)
                base_session_id = fallback_base
                shared_doc = _read_project_shared_config(project_dir, fallback_base)
                shared_doc["base_run_id"] = fallback_base
                _write_project_shared_config(project_dir, shared_doc)
                for item in runs:
                    item["is_base"] = item["run_id"] == base_session_id
            except Exception as exc:
                logger.warning("Failed to update fallback base session for %s: %s", project_id, exc)

        base_colmap_ready = _base_session_colmap_ready(project_dir, base_session_id)
        can_create_session = bool(base_colmap_ready)
        can_create_session_reason = None if can_create_session else "Complete COLMAP on the base session before creating new sessions."

        return {
            "runs": runs,
            "base_session_id": base_session_id,
            "base_colmap_ready": base_colmap_ready,
            "can_create_session": can_create_session,
            "can_create_session_reason": can_create_session_reason,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listing runs for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to list project runs")


@router.get("/{project_id}/runs/{run_id}/config")
def get_project_run_config(project_id: str, run_id: str):
    """Return the persisted run_config payload for a given run session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_config_path = project_dir / "runs" / run_id / "run_config.json"
        run_config = _read_json_if_exists(run_config_path)
        if not isinstance(run_config, dict):
            raise HTTPException(status_code=404, detail="Run config not found")

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        current_shared_version = int(shared_doc.get("version") or 1)
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None

        run_shared_version = run_config.get("shared_config_version")
        if not isinstance(run_shared_version, int):
            run_shared_version = None

        run_shared_snapshot = run_config.get("shared_config_snapshot")
        if not isinstance(run_shared_snapshot, dict):
            run_shared_snapshot = {}

        effective_shared = shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else run_shared_snapshot
        is_base_run = bool(base_session_id) and run_id == str(base_session_id)
        shared_outdated = bool(
            active_sparse_version is not None
            and run_shared_version is not None
            and run_shared_version < active_sparse_version
        )

        return {
            "project_id": project_id,
            "run_id": run_id,
            "run_config": run_config,
            "effective_shared_config": effective_shared,
            "shared_config_version": current_shared_version,
            "active_sparse_shared_version": active_sparse_version,
            "run_shared_config_version": run_shared_version,
            "shared_outdated": shared_outdated,
            "base_session_id": base_session_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading run config for %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read run config")


@router.post("/{project_id}/runs")
def create_project_run(project_id: str, payload: CreateRunRequest = Body(...)):
    """Create a new run/session directory and persist draft run config."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        if not _base_session_colmap_ready(project_dir, base_session_id):
            raise HTTPException(
                status_code=409,
                detail="Cannot create a new session until the base session has completed COLMAP sparse outputs.",
            )

        project_label = None
        if isinstance(project_status, dict):
            project_label = project_status.get("name") or project_status.get("project_id")

        requested_name = str(payload.run_name or "").strip()
        requested_name = _rewrite_auto_run_name_prefix(requested_name, project_id, project_label)
        preferred_name = requested_name or _build_default_run_name(project_label, runs_root)
        run_id = _resolve_unique_run_id(runs_root, preferred_name)

        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        requested_params: dict = {"run_name": run_id, "run_id": run_id}
        resolved_params: dict = {"run_name": run_id, "run_id": run_id}

        if isinstance(payload.resolved_params, dict):
            provided_params = json.loads(json.dumps(payload.resolved_params))
            requested_params.update(provided_params)
            resolved_params.update(provided_params)

        requested_params["run_name"] = run_id
        requested_params["run_id"] = run_id
        resolved_params["run_name"] = run_id
        resolved_params["run_id"] = run_id

        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        active_shared = shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else None
        inherited_shared = active_shared if active_shared else (shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {})
        inherited_shared_version = int(shared_doc.get("active_sparse_version") or shared_doc.get("version") or 1)

        run_config_payload = {
            "project_id": project_id,
            "run_id": run_id,
            "run_name": run_id,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "requested_params": requested_params,
            "resolved_params": resolved_params,
            "shared_config_version": inherited_shared_version,
            "shared_base_run_id": shared_doc.get("base_run_id") if isinstance(shared_doc.get("base_run_id"), str) else base_session_id,
            "shared_config_snapshot": {},
        }

        run_cfg_path = run_dir / "run_config.json"
        tmp = run_cfg_path.with_suffix(run_cfg_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(run_config_payload, f, indent=2)
        tmp.replace(run_cfg_path)

        latest_cfg_path = project_dir / "run_config.json"
        latest_tmp = latest_cfg_path.with_suffix(latest_cfg_path.suffix + ".tmp")
        with open(latest_tmp, "w", encoding="utf-8") as f:
            json.dump(run_config_payload, f, indent=2)
        latest_tmp.replace(latest_cfg_path)

        if not project_status.get("base_session_id"):
            try:
                status.update_base_session_id(project_id, run_id)
            except Exception as exc:
                logger.warning("Failed to set base session for %s: %s", project_id, exc)

        return {
            "status": "created",
            "project_id": project_id,
            "run_id": run_id,
            "run_name": run_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating run for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.patch("/{project_id}/runs/{run_id}")
def rename_project_run(project_id: str, run_id: str, payload: RenameRunRequest = Body(...)):
    """Rename a run session directory and update its run config metadata."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        source_dir = runs_root / run_id
        if not source_dir.exists() or not source_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        desired = _sanitize_run_token(payload.run_name or "")
        if not desired:
            raise HTTPException(status_code=400, detail="Run name cannot be empty")
        if desired == run_id:
            return {"status": "unchanged", "run_id": run_id, "run_name": run_id}

        current_status = status.get_status(project_id)
        if (
            isinstance(current_status, dict)
            and current_status.get("status") in {"processing", "stopping"}
            and current_status.get("current_run_id") == run_id
        ):
            raise HTTPException(status_code=409, detail="Cannot rename an active run")

        target_dir = runs_root / desired
        if target_dir.exists():
            raise HTTPException(status_code=409, detail="Run name already exists")

        source_dir.rename(target_dir)

        run_cfg_path = target_dir / "run_config.json"
        run_cfg = _read_json_if_exists(run_cfg_path)
        if isinstance(run_cfg, dict):
            run_cfg["run_id"] = desired
            run_cfg["run_name"] = desired
            requested = run_cfg.get("requested_params")
            if isinstance(requested, dict):
                requested["run_id"] = desired
                requested["run_name"] = desired
            resolved = run_cfg.get("resolved_params")
            if isinstance(resolved, dict):
                resolved["run_id"] = desired
                resolved["run_name"] = desired
            tmp = run_cfg_path.with_suffix(run_cfg_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(run_cfg, f, indent=2)
            tmp.replace(run_cfg_path)

        # Keep project-level latest config consistent when it points to this run.
        latest_cfg_path = project_dir / "run_config.json"
        latest_cfg = _read_json_if_exists(latest_cfg_path)
        if isinstance(latest_cfg, dict) and latest_cfg.get("run_id") == run_id:
            latest_cfg["run_id"] = desired
            latest_cfg["run_name"] = desired
            tmp = latest_cfg_path.with_suffix(latest_cfg_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(latest_cfg, f, indent=2)
            tmp.replace(latest_cfg_path)

        if isinstance(current_status, dict) and current_status.get("current_run_id") == run_id:
            status.update_status(project_id, current_status.get("status", "pending"), current_run_id=desired)

        if isinstance(current_status, dict) and current_status.get("base_session_id") == run_id:
            status.update_base_session_id(project_id, desired)
            try:
                shared_doc = _read_project_shared_config(project_dir, desired)
                shared_doc["base_run_id"] = desired
                _write_project_shared_config(project_dir, shared_doc)
            except Exception as exc:
                logger.warning("Failed to update shared config base run after rename for %s: %s", project_id, exc)

        return {"status": "renamed", "run_id": desired, "run_name": desired}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error renaming run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to rename run")


@router.patch("/{project_id}/runs/{run_id}/set-base")
def set_base_project_run(project_id: str, run_id: str):
    """Mark the selected run as the project's base session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_dir = project_dir / "runs" / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        status.update_base_session_id(project_id, run_id)
        try:
            shared_doc = _read_project_shared_config(project_dir, run_id)
            shared_doc["base_run_id"] = run_id
            _write_project_shared_config(project_dir, shared_doc)
        except Exception as exc:
            logger.warning("Failed to update shared config base run for %s: %s", project_id, exc)
        return {"status": "ok", "base_session_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error setting base run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to set base session")


@router.delete("/{project_id}/runs/{run_id}")
def delete_project_run(project_id: str, run_id: str):
    """Delete a completed/inactive run session and reassign base session if needed."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        target_dir = runs_root / run_id
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        current_status = status.get_status(project_id)
        if (
            isinstance(current_status, dict)
            and current_status.get("status") in {"processing", "stopping"}
            and current_status.get("current_run_id") == run_id
        ):
            worker_mode = str(current_status.get("worker_mode") or "")
            is_actively_running = False
            try:
                resolved_mode = resolve_worker_mode(worker_mode)
                if resolved_mode == "docker":
                    is_actively_running = bool(colmap.get_project_worker_container_ids(project_id))
                else:
                    is_actively_running = bool(pipeline.is_local_project_active(project_id))
            except Exception:
                # If we cannot confirm runtime activity, keep conservative behavior.
                is_actively_running = True

            if is_actively_running:
                raise HTTPException(status_code=409, detail="Cannot delete an active run")

            # Status says active, but no worker exists anymore; clear stale binding first.
            status.update_status(project_id, "stopped", current_run_id=None, stop_requested=True)
            current_status = status.get_status(project_id)

        _delete_path_strict(target_dir)

        base_session_id = current_status.get("base_session_id") if isinstance(current_status, dict) else None
        deleted_was_base = base_session_id == run_id
        new_base_session_id = base_session_id
        remaining = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True) if runs_root.exists() else []
        if deleted_was_base:
            new_base_session_id = remaining[0].name if remaining else None
            status.update_base_session_id(project_id, new_base_session_id)
            try:
                shared_doc = _read_project_shared_config(project_dir, new_base_session_id)
                shared_doc["base_run_id"] = new_base_session_id
                _write_project_shared_config(project_dir, shared_doc)
            except Exception as exc:
                logger.warning("Failed to update shared config base run after delete for %s: %s", project_id, exc)

        # Keep shared COLMAP/images artifacts on session deletion.
        # Shared data should only be refreshed via explicit restart stage selections.

        if isinstance(current_status, dict) and current_status.get("current_run_id") == run_id:
            status.update_status(project_id, current_status.get("status", "pending"), current_run_id=None)

        return {"status": "deleted", "run_id": run_id, "base_session_id": new_base_session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete run")


@router.get("/{project_id}/shared-config")
def get_project_shared_config(project_id: str):
    """Return project-level shared config anchored to the base session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)

        return {
            "project_id": project_id,
            "base_session_id": base_session_id,
            "version": int(shared_doc.get("version") or 1),
            "updated_at": shared_doc.get("updated_at"),
            "active_sparse_version": shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None,
            "active_sparse_updated_at": shared_doc.get("active_sparse_updated_at"),
            "active_shared": shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else {},
            "shared": shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading shared config for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read shared config")


@router.get("/{project_id}/files")
def get_files(project_id: str, run_id: str | None = Query(None)):
    """Get list of output files for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        requested_run_id = (run_id or "").strip() or None
        if requested_run_id:
            run_dir = project_dir / "runs" / requested_run_id
            if not run_dir.exists() or not run_dir.is_dir():
                raise HTTPException(status_code=404, detail="Run not found")

        output_files = files.get_output_files(project_id, run_id=requested_run_id)
        if requested_run_id:
            shared_files = files.get_output_files(project_id, run_id=None)
            # Uploaded images are shared across sessions.
            if "images" not in output_files and "images" in shared_files:
                output_files["images"] = shared_files["images"]

            # COLMAP sparse is shared across sessions; expose it for all run views.
            if "sparse" not in output_files and "sparse" in shared_files:
                output_files["sparse"] = shared_files["sparse"]
        return {"project_id": project_id, "files": output_files}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get files")


@router.get("/{project_id}/previews/{filename}")
def get_preview_image(
    project_id: str,
    filename: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """Serve a specific preview PNG from outputs/previews (optionally engine-scoped)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if previews_dir is None:
            missing_engine = sanitized_engine or inferred_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        # Prevent path traversal
        resolved_dir = previews_dir.resolve()
        resolved_img = img_path.resolve()
        if resolved_dir not in resolved_img.parents:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            resolved_img,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview image for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preview image")


@router.head("/{project_id}/previews/{filename}")
def head_preview_image(
    project_id: str,
    filename: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """HEAD probe for preview PNG (used by browsers for preflight)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if previews_dir is None:
            missing_engine = sanitized_engine or inferred_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        resolved_dir = previews_dir.resolve()
        resolved_img = img_path.resolve()
        if resolved_dir not in resolved_img.parents:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(resolved_img, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HEAD preview for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to head preview image")


@router.get("/{project_id}/images/locations")
def get_image_locations(project_id: str):
    """Return extracted GPS locations for project images, if available."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        gps_file = project_dir / "images" / "locations.json"
        if not gps_file.exists():
            return {"project_id": project_id, "locations": []}

        try:
            data = json.loads(gps_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to read GPS metadata for {project_id}: {e}")
            data = {}

        locations = [
            {"name": name, "lat": coords.get("lat"), "lon": coords.get("lon")}
            for name, coords in data.items()
            if coords and "lat" in coords and "lon" in coords
        ]

        return {"project_id": project_id, "locations": locations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image locations for {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get image locations")


@router.get("/{project_id}/preview")
def get_preview(project_id: str):
    """Get latest preview PNG."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        preview_file, _, _, _ = _find_existing_path(
            project_id,
            Path("previews") / "preview_latest.png",
            None,
        )
        if not preview_file:
            raise HTTPException(status_code=404, detail="Preview not available")

        return FileResponse(
            preview_file,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get preview")


@router.get("/{project_id}/logs")
def get_logs(project_id: str, lines: int = 500):
    """Get processing logs for a project."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        log_file = project_dir / "processing.log"
        if not log_file.exists():
            return {"project_id": project_id, "logs": "No logs available yet."}
        
        # Read lines from the latest run block (appended logs may contain older runs).
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            latest_run_start = 0
            marker = "Initialized project log file:"
            for idx, line in enumerate(all_lines):
                if marker in line:
                    latest_run_start = idx

            scoped_lines = all_lines[latest_run_start:] if latest_run_start < len(all_lines) else all_lines
            recent_lines = scoped_lines[-lines:] if len(scoped_lines) > lines else scoped_lines
            log_content = "".join(recent_lines)
        
        return {"project_id": project_id, "logs": log_content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get logs")


@router.get("/{project_id}/download/sparse.json")
def download_sparse_json(project_id: str):
    """Return a JSON representation of the first COLMAP sparse reconstruction (points only).

    This endpoint prefers `points3D.txt` (readable) and falls back to `points3D.bin` (best-effort parser).
    The returned shape is {"points": [{"x":..,"y":..,"z":..,"r":..,"g":..,"b":..}, ...]}.
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists() or not sparse_root.is_dir():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        # Pick the first reconstruction directory containing points3D
        recon_dir = None
        for d in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            if (d / "points3D.txt").exists() or (d / "points3D.bin").exists():
                recon_dir = d
                break

        if recon_dir is None:
            raise HTTPException(status_code=404, detail="No COLMAP reconstruction found")

        txt_path = recon_dir / "points3D.txt"
        bin_path = recon_dir / "points3D.bin"

        points = []

        if txt_path.exists():
            # Parse ASCII points3D.txt (format: id x y z r g b error track_length [track...])
            with open(txt_path, "r") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    try:
                        # parts[0]=id, [1..3]=xyz, [4..6]=rgb
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        x, y, z = _colmap_to_opengl_coords(x, y, z)
                        r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                        points.append({"x": x, "y": y, "z": z, "r": r, "g": g, "b": b})
                    except Exception:
                        continue
        elif bin_path.exists():
            # Best-effort binary parser. COLMAP's binary layout may vary; we attempt the common layout.
            import struct
            with open(bin_path, "rb") as f:
                try:
                    # Read number of points (uint64)
                    num_points_bytes = f.read(8)
                    if len(num_points_bytes) < 8:
                        raise ValueError("Invalid points3D.bin header")
                    num_points = struct.unpack("<Q", num_points_bytes)[0]
                except Exception:
                    # If header read fails, fall back to scanning (empty response)
                    num_points = 0

                for _ in range(num_points):
                    try:
                        pid = struct.unpack("<Q", f.read(8))[0]
                        x, y, z = struct.unpack("<ddd", f.read(24))
                        x, y, z = _colmap_to_opengl_coords(x, y, z)
                        r, g, b = struct.unpack("BBB", f.read(3))
                        error = struct.unpack("<d", f.read(8))[0]
                        track_len = struct.unpack("<Q", f.read(8))[0]
                        # skip track entries (image_id, point2d_idx) pairs
                        try:
                            f.read(track_len * 16)
                        except Exception:
                            # If sizes differ, try smaller element sizes
                            try:
                                f.read(track_len * 8)
                            except Exception:
                                pass
                        points.append({"x": x, "y": y, "z": z, "r": int(r), "g": int(g), "b": int(b)})
                    except Exception:
                        # On parse error just break to avoid infinite loop
                        break

        if not points:
            raise HTTPException(status_code=404, detail="No points parsed from COLMAP output")

        return {"points": points}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting sparse JSON for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export sparse points")










@router.get("/{project_id}/splat-format")
def get_splat_format(project_id: str, engine: str | None = Query(None)):
    """Check what splat format is available (ply or bin)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        sanitized_engine = _sanitize_engine(engine)
        search_order, inferred_engine = _engine_search_order(project_id, sanitized_engine)

        for candidate in search_order:
            ply_path = _resolve_output_path(project_dir, "splats.ply", candidate)
            bin_path = _resolve_output_path(project_dir, "splats.bin", candidate)
            if ply_path.exists():
                return {
                    "format": "ply",
                    "has_ply": True,
                    "has_bin": bin_path.exists(),
                    "engine": candidate,
                }
            if bin_path.exists():
                return {
                    "format": "bin",
                    "has_ply": ply_path.exists(),
                    "has_bin": True,
                    "engine": candidate,
                }
        return {"format": "none", "has_ply": False, "has_bin": False, "engine": sanitized_engine or inferred_engine}
    
    except Exception as e:
        logger.error(f"Error checking splat format: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check splat format")


@router.get("/{project_id}/download/splats.splat")
def download_splats_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download .splat file (optimized binary format for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        splat_path, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not splat_path:
            detail = ".splat file not found. Processing may not be complete."
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f".splat file not found for engine '{missing_engine}'."
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=splat_path,
            filename="splats.splat",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading .splat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download .splat")


@router.head("/{project_id}/download/splats.splat")
def head_splats_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """HEAD probe for .splat file (used by frontend to prefer native format)."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
        project_id,
        "splats.splat",
        engine,
        run_id=(run_id.strip() if run_id else None),
    )
    if splat_path:
        return FileResponse(path=splat_path, filename="splats.splat", media_type="application/octet-stream")
    missing_engine = sanitized_engine or inferred_engine
    detail = ".splat file not found"
    if missing_engine:
        detail = f".splat file not found for engine '{missing_engine}'"
    raise HTTPException(status_code=404, detail=detail)


@router.get("/{project_id}/download/splats.ply")
def download_splats_ply(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download PLY splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        ply_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.ply",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not ply_path:
            detail = "PLY file not found"
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f"PLY file not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=ply_path,
            filename="splats.ply",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PLY: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download PLY")


@router.get("/{project_id}/download/splats.bin")
def download_splats_bin(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download binary splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        bin_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.bin",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not bin_path:
            detail = "Binary file not found"
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f"Binary file not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=bin_path,
            filename="splats.bin",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading binary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download binary")


@router.get("/{project_id}/download/points.bin")
def download_points_bin(
    project_id: str,
    candidate: str | None = Query(None, description="best or specific sparse folder name"),
    mode: str = Query("view", regex="^(view|editable)$"),
):
    """Download compact `points.bin` generated from COLMAP reconstruction.

    The converter writes `points.bin` into each reconstruction directory (e.g. outputs/sparse/0/points.bin).
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        def try_serve(candidate_dir: Path):
            target_name = "points_editable.bin" if mode == "editable" else "points.bin"
            points_path = candidate_dir / target_name
            if not points_path.exists() and mode == "editable":
                try:
                    pointsbin.convert_colmap_recon_to_pointsbin(candidate_dir)
                    points_path = candidate_dir / target_name
                except Exception as exc:
                    logger.debug("Failed to refresh editable points for %s: %s", candidate_dir, exc)
            if points_path.exists():
                return FileResponse(path=points_path, filename=target_name, media_type="application/octet-stream")
            return None

        def resolve_relative(rel_path: str | None) -> Path | None:
            if rel_path in (None, "", ".", "root"):
                target = sparse_root
            else:
                target = sparse_root / rel_path
            try:
                resolved = target.resolve()
                base = sparse_root.resolve()
                if resolved == base or base in resolved.parents:
                    return resolved
            except Exception:
                return None
            return None

        def serve_relative(rel_path: str | None):
            target = resolve_relative(rel_path)
            if target and target.exists():
                served = try_serve(target)
                if served:
                    return served
            return None

        meta, _ = _load_sparse_metadata(sparse_root)

        preferred_rel = meta.get("relative_path") if isinstance(meta, dict) else None

        # Honor explicit candidate requests first
        if candidate:
            token = candidate.strip()
            if token:
                if token.lower() == "best" and preferred_rel:
                    served = serve_relative(preferred_rel)
                    if served:
                        return served
                else:
                    served = serve_relative(token)
                    if served:
                        return served

        # Default to best-known reconstruction
        if preferred_rel:
            served = serve_relative(preferred_rel)
            if served:
                return served

        # Fall back to first available reconstruction (original behavior)
        candidates = []
        if (sparse_root / "points.bin").exists():
            candidates.append(sparse_root)
        try:
            candidates.extend(sorted([p for p in sparse_root.iterdir() if p.is_dir()]))
        except Exception as exc:
            logger.warning("Failed to enumerate sparse directories under %s: %s", sparse_root, exc)

        for candidate_dir in candidates:
            served = try_serve(candidate_dir)
            if served:
                return served

        target_name = "points_editable.bin" if mode == "editable" else "points.bin"
        raise HTTPException(
            status_code=404,
            detail=f"{target_name} not found; reconstruction may not be converted yet",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading points.bin: {e}")
        raise HTTPException(status_code=500, detail="Failed to download points.bin")


@router.get("/{project_id}/sparse/candidates")
def list_sparse_candidates(project_id: str):
    """Return metadata about available sparse reconstructions for UI selection."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        return {"candidates": [], "best_relative_path": None, "updated_at": None}

    meta, _ = _load_sparse_metadata(sparse_root)

    candidates = []
    best_rel = None
    updated_at = None
    if isinstance(meta, dict):
        best_rel = meta.get("relative_path")
        updated_at = meta.get("timestamp")
        raw_candidates = meta.get("candidates") or []
        for entry in raw_candidates:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("relative_path") or "."
            candidate_dir = _resolve_sparse_candidate_dir(sparse_root, rel_path)
            images = entry.get("images")
            points = entry.get("points")
            if images is None or points is None:
                computed_images, computed_points = _read_sparse_stats(candidate_dir)
                if images is None:
                    images = computed_images
                if points is None:
                    points = computed_points
            candidates.append(
                {
                    "relative_path": rel_path,
                    "label": entry.get("label"),
                    "images": images,
                    "points": points,
                }
            )

    # Include cached merged sparse models so users can re-select them later.
    discovered: set[str] = {
        (entry.get("relative_path") or ".")
        for entry in candidates
        if isinstance(entry, dict)
    }
    merged_root = sparse_root / "_merged"
    if merged_root.exists() and merged_root.is_dir():
        try:
            for child in sorted(p for p in merged_root.iterdir() if p.is_dir()):
                rel_path = os.path.relpath(child, sparse_root)
                if rel_path in discovered:
                    continue
                if not all((child / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
                    continue
                images, points = _read_sparse_stats(child)
                label = f"merged/{child.name}"
                candidates.append(
                    {
                        "relative_path": rel_path,
                        "label": label,
                        "images": images,
                        "points": points,
                    }
                )
                discovered.add(rel_path)
        except Exception as exc:
            logger.debug("Failed to enumerate merged sparse candidates in %s: %s", merged_root, exc)

    if not candidates:
        try:
            for child in sorted(p for p in sparse_root.iterdir() if p.is_dir() and (p / "points.bin").exists()):
                try:
                    rel_path = os.path.relpath(child, sparse_root)
                except Exception:
                    rel_path = child.name
                if rel_path in {".", ""}:
                    rel_path = "."
                images, points = _read_sparse_stats(child)
                candidates.append({
                    "relative_path": rel_path,
                    "label": Path(rel_path).name if rel_path != "." else "root",
                    "images": images,
                    "points": points,
                })
        except Exception as exc:
            logger.debug("Failed to enumerate sparse fallbacks in %s: %s", sparse_root, exc)
        if not best_rel and candidates:
            best_rel = candidates[0].get("relative_path")

    return {
        "best_relative_path": best_rel,
        "candidates": candidates,
        "updated_at": updated_at,
    }


@router.get("/{project_id}/sparse/image-membership")
def get_sparse_image_membership(project_id: str):
    """Return registered image names for each sparse candidate."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists() or not sparse_root.is_dir():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    membership_path = sparse_root / SPARSE_IMAGE_MEMBERSHIP_META
    if membership_path.exists():
        try:
            payload = json.loads(membership_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.debug("Failed to parse sparse image membership JSON at %s: %s", membership_path, exc)

    # Fallback generation when worker metadata is not present yet.
    candidates_payload = list_sparse_candidates(project_id)
    rows = []
    for entry in candidates_payload.get("candidates", []):
        if not isinstance(entry, dict):
            continue
        rel_path = entry.get("relative_path") or "."
        candidate_dir = _resolve_sparse_candidate_dir(sparse_root, rel_path)
        image_names = _read_sparse_image_names(candidate_dir)
        rows.append(
            {
                "relative_path": rel_path,
                "label": entry.get("label"),
                "images": entry.get("images") if entry.get("images") is not None else len(image_names),
                "image_names": image_names,
            }
        )

    payload = {
        "updated_at": time.time(),
        "candidates": rows,
    }
    try:
        membership_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to persist sparse image membership metadata at %s: %s", membership_path, exc)
    return payload


@router.get("/{project_id}/sparse/merge-report")
def get_sparse_merge_report(project_id: str, candidate: str | None = Query(None)):
    """Return merge metadata for a cached merged sparse candidate.

    - If `candidate` is provided, it can be either `_merged/<name>` or `<name>`.
    - If omitted, the latest merged candidate with `merge_meta.json` is returned.
    """
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    merged_root = sparse_root / "_merged"
    if not merged_root.exists() or not merged_root.is_dir():
        return {"available": False, "candidate": None, "report": None}

    target_dir: Path | None = None
    if candidate:
        token = candidate.strip()
        if token.startswith("_merged/"):
            token = token.split("/", 1)[1]
        if token:
            resolved = (merged_root / token).resolve()
            root_resolved = merged_root.resolve()
            if resolved == root_resolved or root_resolved not in resolved.parents:
                raise HTTPException(status_code=400, detail="Invalid merge candidate")
            if resolved.exists() and resolved.is_dir():
                target_dir = resolved
            else:
                raise HTTPException(status_code=404, detail="Merge candidate not found")
    else:
        candidates = [
            p for p in merged_root.iterdir()
            if p.is_dir() and (p / "merge_meta.json").exists()
        ]
        if not candidates:
            return {"available": False, "candidate": None, "report": None}
        target_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    if target_dir is None:
        return {"available": False, "candidate": None, "report": None}

    meta_path = target_dir / "merge_meta.json"
    if not meta_path.exists():
        return {
            "available": False,
            "candidate": os.path.relpath(target_dir, sparse_root),
            "report": None,
        }

    try:
        report = json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning("Failed to parse merge metadata at %s: %s", meta_path, exc)
        raise HTTPException(status_code=500, detail="Failed to parse merge metadata")

    return {
        "available": True,
        "candidate": os.path.relpath(target_dir, sparse_root),
        "report": report,
    }


@router.post("/{project_id}/sparse/merge")
def build_sparse_merge(project_id: str, payload: SparseMergeRequest | None = Body(None)):
    """Build a merged sparse model from selected candidate folders without starting training."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    selections = payload.selections if payload else []
    if not isinstance(selections, list) or len(selections) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two sparse selections to merge")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists() or not sparse_root.is_dir():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    image_dir = project_dir / "images"
    if not image_dir.exists() or not image_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project images not found")

    try:
        # Imported lazily to avoid importing worker internals during API startup.
        from bimba3d_backend.worker.entrypoint import _resolve_sparse_model_for_training  # pylint: disable=import-outside-toplevel

        merged_dir = _resolve_sparse_model_for_training(
            sparse_root,
            image_dir,
            "merge_selected",
            selections,
        )
    except Exception as exc:
        logger.warning("Sparse merge build failed for %s: %s", project_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    merged_dir = Path(merged_dir)
    sparse_base = sparse_root.resolve()
    merged_resolved = merged_dir.resolve()
    if merged_resolved == sparse_base or sparse_base not in merged_resolved.parents:
        raise HTTPException(status_code=500, detail="Unexpected merged model location")

    rel_path = os.path.relpath(merged_resolved, sparse_root)
    report = None
    meta_path = merged_resolved / "merge_meta.json"
    if meta_path.exists():
        try:
            report = json.loads(meta_path.read_text())
        except Exception as exc:
            logger.debug("Failed to parse merge metadata after build (%s): %s", meta_path, exc)

    return {
        "status": "ok",
        "candidate": rel_path,
        "report": report,
    }


@router.post("/{project_id}/sparse/edit")
def edit_sparse_points(project_id: str, payload: SparseEditRequest | None = Body(None)):
    if payload is None or not payload.remove_point_ids:
        raise HTTPException(status_code=400, detail="No point ids provided")

    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    candidate_dir, candidate_rel = _resolve_sparse_candidate_for_edit(project_dir, payload.candidate)

    try:
        remove_ids = {int(pid) for pid in payload.remove_point_ids if isinstance(pid, int)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid point id provided") from None

    if not remove_ids:
        raise HTTPException(status_code=400, detail="No valid point ids provided")

    try:
        result = sparse_edit.apply_sparse_edits(
            project_dir=project_dir,
            candidate_dir=candidate_dir,
            candidate_rel=candidate_rel,
            remove_point_ids=remove_ids,
            create_backup=True if payload.create_backup is None else bool(payload.create_backup),
            reoptimize=bool(payload.reoptimize),
        )
    except sparse_edit.SparseEditError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _update_sparse_candidate_points(project_dir, candidate_rel, result.get("remaining_points"))

    return {
        **result,
        "candidate_relative_path": candidate_rel,
    }



@router.get("/{project_id}/download/splats")
def download_splats(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download splats file (.splat format optimized for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )
        if splat_path:
            return FileResponse(
                path=splat_path,
                filename="splats.splat",
                media_type="application/octet-stream"
            )

        ply_path, _, _, _ = _find_existing_path(project_id, "splats.ply", engine, run_id=(run_id.strip() if run_id else None))
        if ply_path:
            return FileResponse(
                path=ply_path,
                filename="splats.ply",
                media_type="application/octet-stream"
            )

        bin_path, _, _, _ = _find_existing_path(project_id, "splats.bin", engine, run_id=(run_id.strip() if run_id else None))
        if bin_path:
            return FileResponse(
                path=bin_path,
                filename="splats.bin",
                media_type="application/octet-stream"
            )
        
        detail = "Splats file not found. Processing may not be complete."
        missing_engine = sanitized_engine or inferred_engine
        if missing_engine:
            detail = f"Splats file not found for engine '{missing_engine}'."
        raise HTTPException(status_code=404, detail=detail)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading splats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download splats")


@router.get("/{project_id}/download/snapshots/{filename}")
def download_snapshot(project_id: str, filename: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download a specific intermediate splat snapshot exported during training."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        snapshots_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "snapshots",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if not snapshots_dir:
            missing_engine = sanitized_engine or inferred_engine
            detail = "No snapshots available"
            if missing_engine:
                detail = f"No snapshots available for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        snapshots_root = snapshots_dir.resolve()
        snap_path = (snapshots_dir / filename).resolve()
        if snapshots_root not in snap_path.parents:
            raise HTTPException(status_code=403, detail="Access denied")
        if not snap_path.exists() or not snap_path.is_file():
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Snapshot not found"
            if missing_engine:
                detail = f"Snapshot not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        media_type = "application/octet-stream"
        if snap_path.suffix.lower() == ".ply":
            media_type = "application/octet-stream"
        elif snap_path.suffix.lower() == ".splat":
            media_type = "application/octet-stream"

        return FileResponse(
            path=snap_path,
            filename=snap_path.name,
            media_type=media_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading snapshot for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download snapshot")


@router.get("/{project_id}/download/sparse.json")
def download_sparse_json(project_id: str):
    """Return a JSON representation of the first COLMAP sparse reconstruction (points only).

    This endpoint prefers `points3D.txt` (readable) and falls back to `points3D.bin` (best-effort parser).
    The returned shape is {"points": [{"x":..,"y":..,"z":..,"r":..,"g":..,"b":..}, ...]}.
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists() or not sparse_root.is_dir():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        # Pick the first reconstruction directory containing points3D
        recon_dir = None
        for d in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            if (d / "points3D.txt").exists() or (d / "points3D.bin").exists():
                recon_dir = d
                break

        if recon_dir is None:
            raise HTTPException(status_code=404, detail="No COLMAP reconstruction found")

        txt_path = recon_dir / "points3D.txt"
        bin_path = recon_dir / "points3D.bin"

        points = []

        if txt_path.exists():
            # Parse ASCII points3D.txt (format: id x y z r g b error track_length [track...])
            with open(txt_path, "r") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    try:
                        # parts[0]=id, [1..3]=xyz, [4..6]=rgb
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                        points.append({"x": x, "y": y, "z": z, "r": r, "g": g, "b": b})
                    except Exception:
                        continue
        elif bin_path.exists():
            # Best-effort binary parser. COLMAP's binary layout may vary; we attempt the common layout.
            import struct
            with open(bin_path, "rb") as f:
                try:
                    # Read number of points (uint64)
                    num_points_bytes = f.read(8)
                    if len(num_points_bytes) < 8:
                        raise ValueError("Invalid points3D.bin header")
                    num_points = struct.unpack("<Q", num_points_bytes)[0]
                except Exception:
                    # If header read fails, fall back to scanning (empty response)
                    num_points = 0

                for _ in range(num_points):
                    try:
                        pid = struct.unpack("<Q", f.read(8))[0]
                        x, y, z = struct.unpack("<ddd", f.read(24))
                        r, g, b = struct.unpack("BBB", f.read(3))
                        error = struct.unpack("<d", f.read(8))[0]
                        track_len = struct.unpack("<Q", f.read(8))[0]
                        # skip track entries (image_id, point2d_idx) pairs
                        try:
                            f.read(track_len * 16)
                        except Exception:
                            # If sizes differ, try smaller element sizes
                            try:
                                f.read(track_len * 8)
                            except Exception:
                                pass
                        points.append({"x": x, "y": y, "z": z, "r": int(r), "g": int(g), "b": int(b)})
                    except Exception:
                        # On parse error just break to avoid infinite loop
                        break

        if not points:
            raise HTTPException(status_code=404, detail="No points parsed from COLMAP output")

        return {"points": points}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting sparse JSON for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export sparse points")


@router.get("/{project_id}/metadata")
def get_metadata(project_id: str, engine: str | None = Query(None)):
    """Get metadata.json for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        metadata_path, _, _, _ = _find_existing_path(project_id, "metadata.json", engine)
        if not metadata_path:
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metadata")


@router.get("/{project_id}/images")
def list_images(project_id: str):
    """List all images in a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        images_dir = project_dir / "images"
        image_list = []
        
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*")):
                if img_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    image_list.append({
                        "name": img_path.name,
                        "size": img_path.stat().st_size,
                        "url": f"/projects/{project_id}/image/{img_path.name}"
                    })
        
        return {"project_id": project_id, "images": image_list}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list images")


@router.get("/{project_id}/image/{filename}")
def get_image(project_id: str, filename: str):
    """Get a specific image from a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        image_path = project_dir / "images" / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Verify file is actually in the images directory
        if not str(image_path).startswith(str(project_dir / "images")):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(path=image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get image")


@router.get("/{project_id}/thumbnail/{filename}")
def get_thumbnail(project_id: str, filename: str):
    """Get a thumbnail for a specific image."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        thumbnail_path = project_dir / "images" / "thumbnails" / filename
        
        # If thumbnail doesn't exist, return 404 (no fallback to full image)
        if not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        # Verify file is actually in the thumbnails directory
        if not str(thumbnail_path).startswith(str(project_dir / "images" / "thumbnails")):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(path=thumbnail_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get thumbnail")


@router.delete("/{project_id}")
def delete_project(project_id: str):
    """Delete a project and all associated files."""
    try:
        project_alias = DATA_DIR / project_id

        if not project_alias.exists() and not project_alias.is_symlink():
            raise HTTPException(status_code=404, detail="Project not found")

        # Resolve physical target (custom storage root projects are aliased under DATA_DIR).
        try:
            real_project_dir = project_alias.resolve(strict=True)
        except Exception:
            real_project_dir = project_alias

        # Request any in-flight workers to stop so they don't keep writing files.
        try:
            stop_flag = project_alias / "stop_requested"
            stop_flag.write_text("stop")
        except Exception:
            pass

        try:
            colmap.stop_project_worker_containers(project_id)
        except Exception as exc:
            logger.warning("Failed to stop worker containers for %s before delete: %s", project_id, exc)

        if pipeline.is_local_project_active(project_id):
            # Give local worker a short grace period to finish cleanup.
            for _ in range(15):
                if not pipeline.is_local_project_active(project_id):
                    break
                time.sleep(0.2)

        # Best-effort release of any file handlers still pointing into project logs.
        _close_project_log_handlers(project_alias)
        if real_project_dir != project_alias:
            _close_project_log_handlers(real_project_dir)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                # Remove alias under DATA_DIR first so project disappears from listings.
                _delete_path_strict(project_alias)

                # Also remove physical target dir when alias points elsewhere.
                if real_project_dir != project_alias:
                    _delete_path_strict(real_project_dir)

                last_error = None
                break
            except Exception as exc:
                last_error = exc
                # Retry after brief backoff for transient Windows file locks.
                _close_project_log_handlers(project_alias)
                if real_project_dir != project_alias:
                    _close_project_log_handlers(real_project_dir)
                time.sleep(0.2 * (attempt + 1))

        if last_error is not None:
            raise last_error

        # Ensure stale aliases are removed if target was deleted first.
        if project_alias.exists() or project_alias.is_symlink():
            _delete_path_strict(project_alias)

        alias_exists_after = project_alias.exists() or project_alias.is_symlink()
        target_exists_after = False
        if real_project_dir != project_alias:
            target_exists_after = real_project_dir.exists() or real_project_dir.is_symlink()
        if alias_exists_after or target_exists_after:
            logger.error(
                "Project deletion incomplete for %s (alias_exists=%s target_exists=%s)",
                project_id,
                alias_exists_after,
                target_exists_after,
            )
            raise HTTPException(status_code=500, detail="Project delete incomplete")

        logger.info(f"Deleted project: {project_id}")
        return {"status": "deleted", "project_id": project_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete project")


@router.patch("/{project_id}")
def update_project(project_id: str, payload: UpdateProjectRequest = Body(...)):
    """Update project metadata (currently only name)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        if payload.name is None:
            raise HTTPException(status_code=400, detail="Nothing to update")

        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if len(name) > 120:
            raise HTTPException(status_code=400, detail="Name too long (max 120 characters)")

        status.update_project_name(project_id, name)
        logger.info(f"Renamed project {project_id} -> {name}")
        return {"status": "updated", "project_id": project_id, "name": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update project")


@router.get("/{project_id}/metrics", response_model=EvaluationMetrics)
def get_metrics(project_id: str):
    """Get evaluation metrics for a completed project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check engine-aware metadata
        metadata_path, _, _, _ = _find_existing_path(project_id, "metadata.json", None)
        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Extract metrics from metadata
                metrics = metadata.get("evaluation_metrics", {})
                if metrics:
                    return EvaluationMetrics(**metrics)
        
        # Fallback: check adaptive_tuning_results.json (engine-aware)
        tuning_path, _, _, _ = _find_existing_path(project_id, "adaptive_tuning_results.json", None)
        if tuning_path and tuning_path.exists():
            with open(tuning_path) as f:
                tuning_data = json.load(f)
                final_metrics = tuning_data.get("final_evaluation", {})
                if final_metrics:
                    return EvaluationMetrics(**final_metrics)
        
        # No metrics available yet
        return EvaluationMetrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.get("/{project_id}/experiment-summary")
def get_experiment_summary(
    project_id: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """Return an engine-aware summary payload for comparing two runs."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        status_info = status.get_status(project_id)
        sanitized_engine = _sanitize_engine(engine) if engine is not None else None
        search_order, inferred_engine = _engine_search_order(project_id, sanitized_engine)
        selected_engine = search_order[0] if search_order else None
        requested_run_id = (run_id or "").strip()
        run_dir = (project_dir / "runs" / requested_run_id) if requested_run_id else None
        if requested_run_id and (run_dir is None or not run_dir.exists()):
            raise HTTPException(status_code=404, detail="Run not found")

        metadata_path, resolved_engine, _, _ = _find_existing_path(
            project_id,
            "metadata.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        eval_history_path, _, _, _ = _find_existing_path(
            project_id,
            "eval_history.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        tuning_results_path, _, _, _ = _find_existing_path(
            project_id,
            "adaptive_tuning_results.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        run_config_path = (run_dir / "run_config.json") if run_dir else (project_dir / "run_config.json")

        metadata = _read_json_if_exists(metadata_path)
        eval_history_raw = _read_json_if_exists(eval_history_path)
        eval_history = eval_history_raw if isinstance(eval_history_raw, list) else []
        tuning_results = _read_json_if_exists(tuning_results_path)
        run_config = _read_json_if_exists(run_config_path)

        latest_eval = eval_history[-1] if isinstance(eval_history, list) and eval_history else {}
        first_eval = eval_history[0] if isinstance(eval_history, list) and eval_history else {}

        # Prefer eval history fields, then final metadata fallback.
        metrics = {
            "convergence_speed": latest_eval.get("convergence_speed"),
            "final_loss": latest_eval.get("final_loss"),
            "lpips_mean": latest_eval.get("lpips_mean"),
            "sharpness_mean": latest_eval.get("sharpness_mean"),
            "num_gaussians": latest_eval.get("num_gaussians"),
            "total_time_seconds": None,
        }
        loss_milestones = {}
        if isinstance(latest_eval, dict):
            for k, v in latest_eval.items():
                if isinstance(k, str) and k.startswith("loss_at_"):
                    loss_milestones[k] = v

        eval_series = []
        eval_time_series = []
        runtime_tuning_series = []
        if isinstance(eval_history, list):
            for point in eval_history:
                if not isinstance(point, dict):
                    continue
                step_value = point.get("step")
                loss_value = point.get("final_loss")
                if isinstance(step_value, (int, float)) and isinstance(loss_value, (int, float)):
                    eval_series.append({"step": int(step_value), "loss": float(loss_value)})

                conv_speed = point.get("convergence_speed")
                if isinstance(step_value, (int, float)) and isinstance(conv_speed, (int, float)) and float(conv_speed) > 0:
                    eval_time_series.append({
                        "step": int(step_value),
                        "elapsed_seconds": float(step_value) / float(conv_speed),
                    })

        # Fallback for older runs where eval_history contains step but null final_loss.
        if not eval_series:
            processing_log = (run_dir / "processing.log") if run_dir else (project_dir / "processing.log")
            if processing_log.exists():
                try:
                    step_loss_map = {}
                    step_param_map = {}
                    pattern = re.compile(r"step=(\d+)/(?:\d+).*?loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
                    pattern_params = re.compile(r"step=(\d+)/(?:\d+).*?strategy=(\{[^\n]*?\})\s+lrs=(\{[^\n]*?\})")
                    with open(processing_log, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    # Run-specific logs are already isolated under runs/<run_id>/processing.log.
                    # For legacy project-level logs, keep parsing only the latest run segment.
                    start_idx = 0
                    end_idx = len(lines)
                    if not run_dir:
                        run_markers = [
                            idx for idx, line in enumerate(lines)
                            if "Running local worker:" in line and f" {project_id} " in line
                        ]
                        start_idx = run_markers[-1] if run_markers else 0
                        for idx in range(start_idx + 1, len(lines)):
                            line = lines[idx]
                            if "Running local worker:" in line and f" {project_id} " not in line:
                                end_idx = idx
                                break

                    for line in lines[start_idx:end_idx]:
                            match = pattern.search(line)
                            if not match:
                                pass
                            else:
                                step_value = int(match.group(1))
                                loss_value = float(match.group(2))
                                step_loss_map[step_value] = loss_value

                            match_params = pattern_params.search(line)
                            if match_params:
                                step_value = int(match_params.group(1))
                                try:
                                    strategy = ast.literal_eval(match_params.group(2))
                                    lrs = ast.literal_eval(match_params.group(3))
                                    if isinstance(strategy, dict) and isinstance(lrs, dict):
                                        step_param_map[step_value] = {
                                            "strategy": strategy,
                                            "learning_rates": lrs,
                                        }
                                except Exception:
                                    pass
                    if step_loss_map:
                        eval_series = [
                            {"step": s, "loss": step_loss_map[s]}
                            for s in sorted(step_loss_map.keys())
                        ]
                    if step_param_map:
                        runtime_tuning_series = [
                            {"step": s, "params": step_param_map[s]}
                            for s in sorted(step_param_map.keys())
                        ]
                except Exception as exc:
                    logger.warning("Failed to parse processing log for eval series %s: %s", processing_log, exc)

        if metrics.get("final_loss") is None and eval_series:
            metrics["final_loss"] = eval_series[-1].get("loss")

        timing = status_info.get("timing") if isinstance(status_info, dict) else None
        if isinstance(timing, dict):
            elapsed = timing.get("elapsed")
            if isinstance(elapsed, (int, float)) and float(elapsed) >= 0:
                metrics["total_time_seconds"] = float(elapsed)

        if metrics.get("total_time_seconds") is None and eval_time_series:
            try:
                metrics["total_time_seconds"] = max(
                    float(item.get("elapsed_seconds"))
                    for item in eval_time_series
                    if isinstance(item, dict) and isinstance(item.get("elapsed_seconds"), (int, float))
                )
            except ValueError:
                pass

        if metadata and isinstance(metadata, dict):
            final_metrics = metadata.get("final_metrics") if isinstance(metadata.get("final_metrics"), dict) else {}
            if metrics["convergence_speed"] is None:
                metrics["convergence_speed"] = final_metrics.get("convergence_speed")
            if metrics["final_loss"] is None:
                metrics["final_loss"] = final_metrics.get("final_loss")
            if metrics["lpips_mean"] is None:
                metrics["lpips_mean"] = final_metrics.get("lpips_mean")
            if metrics["sharpness_mean"] is None:
                metrics["sharpness_mean"] = final_metrics.get("sharpness_mean")
            if metrics["num_gaussians"] is None:
                metrics["num_gaussians"] = metadata.get("num_gaussians")

        final_tuning_params = {}
        initial_tuning_params = {}
        if isinstance(first_eval, dict):
            initial_tuning_params = first_eval.get("tuning_params") or {}
        if isinstance(latest_eval, dict):
            final_tuning_params = latest_eval.get("tuning_params") or {}
        if not final_tuning_params and isinstance(metadata, dict):
            meta_final = metadata.get("final_tuning_params")
            if isinstance(meta_final, dict):
                final_tuning_params = meta_final

        tuning_history_count = 0
        tuning_history = []
        tune_end_step = None
        tune_end_params = {}
        configured_tune_end_step = None
        if isinstance(run_config, dict):
            resolved_cfg = run_config.get("resolved_params")
            if isinstance(resolved_cfg, dict):
                configured_tune_end_step = resolved_cfg.get("tune_end_step")
        mode_value = status_info.get("mode") or (metadata.get("mode") if isinstance(metadata, dict) else None)

        if isinstance(tuning_results, dict):
            maybe_tuning_history = tuning_results.get("tuning_history")
            if isinstance(maybe_tuning_history, list):
                tuning_history = maybe_tuning_history
                tuning_history_count = len(tuning_history)
            tune_end_step = tuning_results.get("tune_end_step")
            maybe_final = tuning_results.get("final_params")
            if isinstance(maybe_final, dict):
                tune_end_params = maybe_final

        if mode_value != "modified":
            tune_end_step = None
            tune_end_params = {}

        if not runtime_tuning_series and isinstance(tuning_history, list):
            runtime_tuning_series = [
                {"step": item.get("step"), "params": item.get("params")}
                for item in tuning_history
                if isinstance(item, dict) and isinstance(item.get("step"), (int, float)) and isinstance(item.get("params"), dict)
            ]

        resolved_cfg = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
        tune_interval = resolved_cfg.get("tune_interval")
        log_interval = resolved_cfg.get("log_interval")
        major_params = {
            "max_steps": resolved_cfg.get("max_steps"),
            "total_steps_completed": status_info.get("currentStep") if status_info.get("currentStep") is not None else latest_eval.get("step"),
            "densify_from_iter": resolved_cfg.get("densify_from_iter"),
            "densify_until_iter": resolved_cfg.get("densify_until_iter"),
            "densification_interval": resolved_cfg.get("densification_interval"),
            "eval_interval": resolved_cfg.get("eval_interval"),
            "save_interval": resolved_cfg.get("save_interval"),
            "splat_export_interval": resolved_cfg.get("splat_export_interval"),
            "batch_size": resolved_cfg.get("batch_size"),
        }

        outputs = files.get_output_files(project_id, run_id=requested_run_id or None)
        preview_url = None
        if resolved_engine and isinstance(outputs.get("engines"), dict):
            engine_bundle = outputs["engines"].get(resolved_engine, {})
            previews = engine_bundle.get("previews", {}) if isinstance(engine_bundle, dict) else {}
            preview_url = previews.get("latest_url")

        return {
            "project_id": project_id,
            "run_id": requested_run_id or None,
            "run_name": run_config.get("run_name") if isinstance(run_config, dict) else (requested_run_id or None),
            "name": status_info.get("name"),
            "status": status_info.get("status"),
            "mode": mode_value,
            "engine": resolved_engine or inferred_engine,
            "metrics": metrics,
            "tuning": {
                "initial": initial_tuning_params,
                "final": final_tuning_params,
                "end_params": tune_end_params,
                "end_step": (
                    configured_tune_end_step
                    if configured_tune_end_step is not None
                    else tune_end_step
                    if tune_end_step is not None
                    else (metadata.get("tune_end_step") if isinstance(metadata, dict) else None)
                ),
                "runs": metadata.get("tuning_runs") if isinstance(metadata, dict) else None,
                "history_count": tuning_history_count,
                "history": tuning_history,
                "tune_interval": tune_interval,
                "log_interval": log_interval,
                "runtime_series": runtime_tuning_series,
            },
            "major_params": major_params,
            "loss_milestones": loss_milestones,
            "eval_series": eval_series,
            "eval_time_series": eval_time_series,
            "preview_url": preview_url,
            "eval_points": len(eval_history) if isinstance(eval_history, list) else 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building experiment summary for {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to build experiment summary")


# ==================== COMPARISON ENDPOINTS ====================

@router.post("/comparison", response_model=dict)
def create_comparison(payload: ComparisonRequest):
    """Create a comparison project (will run both baseline and optimized)."""
    try:
        comparison_id = f"cmp_{int(time.time())}"
        comparison_dir = DATA_DIR / comparison_id
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Store comparison metadata
        meta = {
            "comparison_id": comparison_id,
            "name": payload.name or "Comparison",
            "max_steps": payload.max_steps,
            "batch_size": payload.batch_size,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        logger.info(f"Created comparison: {comparison_id}")
        return {"comparison_id": comparison_id}
    except Exception as e:
        logger.error(f"Error creating comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@router.post("/comparison/{comparison_id}/images")
async def upload_comparison_images(
    comparison_id: str,
    images: list[UploadFile] = File(...)
):
    """Upload images for comparison."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        images_dir = comparison_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_count = 0
        invalid_files: list[str] = []
        allowed_ext_text = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        for img in images:
            # Validate file extension
            file_ext = Path(img.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(f"Skipped invalid image: {img.filename}")
                invalid_files.append(img.filename)
                continue
            
            content = await img.read()
            file_path = images_dir / img.filename
            with open(file_path, "wb") as f:
                f.write(content)
            uploaded_count += 1
            logger.info(f"Uploaded comparison image: {img.filename}")
        
        if uploaded_count == 0:
            invalid_list = ", ".join(invalid_files[:10])
            if len(invalid_files) > 10:
                invalid_list = f"{invalid_list}, ..."
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No valid images uploaded. Allowed formats: {allowed_ext_text}. "
                    f"Invalid files: {invalid_list}"
                ),
            )
        
        return {"status": "uploaded", "count": uploaded_count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading comparison images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload images")


@router.post("/comparison/{comparison_id}/start")
def start_comparison(comparison_id: str):
    """Start sequential comparison (baseline then optimized)."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Check if images exist
        images_dir = comparison_dir / "images"
        if not images_dir.exists() or not list(images_dir.glob("*")):
            raise HTTPException(status_code=400, detail="No images in comparison")
        
        # Start comparison in background thread
        thread = threading.Thread(
            target=run_comparison_pipeline,
            args=(comparison_id,),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started comparison: {comparison_id}")
        return {"status": "comparison_started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start comparison")


@router.get("/comparison/{comparison_id}/status", response_model=ComparisonStatus)
def get_comparison_status(comparison_id: str):
    """Get status of comparison run."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Load comparison metadata
        meta_path = comparison_dir / "comparison.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Comparison metadata not found")
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        return ComparisonStatus(**meta)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get comparison status")


def run_comparison_pipeline(comparison_id: str):
    """Background task to run both baseline and optimized sequentially."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        
        # Load metadata
        with open(comparison_dir / "comparison.json") as f:
            meta = json.load(f)
        
        images_dir = comparison_dir / "images"
        
        # Update status to running
        meta["status"] = "running"
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # 1. Create baseline project
        baseline_id, baseline_dir = storage.create_project()
        status.initialize_status(baseline_id, name=f"{meta['name']} - Baseline")
        shutil.copytree(images_dir, baseline_dir / "images")
        
        # Update comparison with baseline info
        meta["baseline"] = {"status": "running", "progress": 0}
        meta["baseline_project_id"] = baseline_id
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # Run baseline
        logger.info(f"Comparison {comparison_id}: Starting baseline run {baseline_id}")
        pipeline.run_full_pipeline(baseline_id, {
            "mode": "baseline",
            "max_steps": meta["max_steps"],
            "batch_size": meta["batch_size"]
        })
        
        # Update baseline status
        baseline_status = status.get_status(baseline_id)
        meta["baseline"] = baseline_status
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # 2. Create optimized project
        optimized_id, optimized_dir = storage.create_project()
        status.initialize_status(optimized_id, name=f"{meta['name']} - Optimized")
        shutil.copytree(images_dir, optimized_dir / "images")
        
        # Update comparison with optimized info
        meta["optimized"] = {"status": "running", "progress": 0}
        meta["optimized_project_id"] = optimized_id
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # Run optimized
        logger.info(f"Comparison {comparison_id}: Starting optimized run {optimized_id}")
        pipeline.run_full_pipeline(optimized_id, {
            "mode": "modified",
            "max_steps": meta["max_steps"],
            "batch_size": meta["batch_size"]
        })
        
        # Update optimized status
        optimized_status = status.get_status(optimized_id)
        meta["optimized"] = optimized_status
        
        # Mark comparison as completed
        meta["status"] = "completed"
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        logger.info(f"Comparison {comparison_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison pipeline failed: {str(e)}")
        # Update comparison status to failed
        try:
            with open(comparison_dir / "comparison.json") as f:
                meta = json.load(f)
            meta["status"] = "failed"
            meta["error"] = str(e)
            with open(comparison_dir / "comparison.json", "w") as f:
                json.dump(meta, f)
        except:
            pass



@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
