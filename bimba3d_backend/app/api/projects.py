import threading
import logging
import shutil
import json
import time
import re
import os
import struct
from typing import Optional
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import FileResponse
from bimba3d_backend.app.config import DATA_DIR, ALLOWED_IMAGE_EXTENSIONS
from bimba3d_backend.app.models.project import (
    ProjectResponse,
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
)
from bimba3d_backend.app.services import status, storage, colmap, gsplat, files, sparse_edit, pointsbin
from bimba3d_backend.app.services.worker_mode import normalize_worker_mode, resolve_worker_mode
from bimba3d_backend.worker import pipeline

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)
BEST_SPARSE_META = ".best_sparse_selection.json"
SPARSE_IMAGE_MEMBERSHIP_META = ".sparse_image_membership.json"


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
    *,
    expect_directory: bool = False,
) -> tuple[Path | None, str | None, str | None, str | None]:
    sanitized = _sanitize_engine(engine)
    search_order, inferred = _engine_search_order(project_id, sanitized)
    project_dir = DATA_DIR / project_id
    for candidate in search_order:
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
            projects.append(
                ProjectListItem(
                    project_id=project_id,
                    name=project_status.get("name"),
                    status=current_status,
                    progress=progress,
                    created_at=project_status.get("created_at"),
                    has_outputs=has_outputs,
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
        project_id, project_dir = storage.create_project()
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
            raise HTTPException(status_code=400, detail="No valid images uploaded")
        
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
        params_payload.setdefault("max_steps", 30000)
        params_payload.setdefault("log_interval", 100)
        params_payload.setdefault("batch_size", 1)
        params_payload.setdefault("densify_from_iter", 500)
        params_payload.setdefault("densify_until_iter", 15000)
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

        # Persist run configuration for reproducibility (requested + resolved params).
        try:
            run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            run_config_payload = {
                "project_id": project_id,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "requested_params": requested_params,
                "resolved_params": params_payload,
            }

            run_config_latest = project_dir / "run_config.json"
            run_configs_dir = project_dir / "run_configs"
            run_configs_dir.mkdir(parents=True, exist_ok=True)
            run_config_versioned = run_configs_dir / f"run_config_{run_timestamp}.json"

            for target_path in (run_config_latest, run_config_versioned):
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
        return {"status": "processing_started"}
    
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


@router.get("/{project_id}/files")
def get_files(project_id: str):
    """Get list of output files for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        output_files = files.get_output_files(project_id)
        return {"project_id": project_id, "files": output_files}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get files")


@router.get("/{project_id}/previews/{filename}")
def get_preview_image(project_id: str, filename: str, engine: str | None = Query(None)):
    """Serve a specific preview PNG from outputs/previews (optionally engine-scoped)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
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
def head_preview_image(project_id: str, filename: str, engine: str | None = Query(None)):
    """HEAD probe for preview PNG (used by browsers for preflight)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
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
def download_splats_splat(project_id: str, engine: str | None = Query(None)):
    """Download .splat file (optimized binary format for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        splat_path, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
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
def head_splats_splat(project_id: str, engine: str | None = Query(None)):
    """HEAD probe for .splat file (used by frontend to prefer native format)."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
        project_id,
        "splats.splat",
        engine,
    )
    if splat_path:
        return FileResponse(path=splat_path, filename="splats.splat", media_type="application/octet-stream")
    missing_engine = sanitized_engine or inferred_engine
    detail = ".splat file not found"
    if missing_engine:
        detail = f".splat file not found for engine '{missing_engine}'"
    raise HTTPException(status_code=404, detail=detail)


@router.get("/{project_id}/download/splats.ply")
def download_splats_ply(project_id: str, engine: str | None = Query(None)):
    """Download PLY splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        ply_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.ply",
            engine,
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
def download_splats_bin(project_id: str, engine: str | None = Query(None)):
    """Download binary splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        bin_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.bin",
            engine,
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
def download_splats(project_id: str, engine: str | None = Query(None)):
    """Download splats file (.splat format optimized for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
        )
        if splat_path:
            return FileResponse(
                path=splat_path,
                filename="splats.splat",
                media_type="application/octet-stream"
            )

        ply_path, _, _, _ = _find_existing_path(project_id, "splats.ply", engine)
        if ply_path:
            return FileResponse(
                path=ply_path,
                filename="splats.ply",
                media_type="application/octet-stream"
            )

        bin_path, _, _, _ = _find_existing_path(project_id, "splats.bin", engine)
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
def download_snapshot(project_id: str, filename: str, engine: str | None = Query(None)):
    """Download a specific intermediate splat snapshot exported during training."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        snapshots_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "snapshots",
            engine,
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
        project_dir = DATA_DIR / project_id

        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        shutil.rmtree(project_dir)
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
def get_experiment_summary(project_id: str, engine: str | None = Query(None)):
    """Return an engine-aware summary payload for comparing two runs."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        status_info = status.get_status(project_id)
        sanitized_engine = _sanitize_engine(engine) if engine is not None else None
        search_order, inferred_engine = _engine_search_order(project_id, sanitized_engine)
        selected_engine = search_order[0] if search_order else None

        def _read_json_if_exists(path: Path | None):
            if path is None or not path.exists():
                return None
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to parse JSON %s: %s", path, exc)
                return None

        metadata_path, resolved_engine, _, _ = _find_existing_path(project_id, "metadata.json", selected_engine)
        eval_history_path, _, _, _ = _find_existing_path(project_id, "eval_history.json", selected_engine)
        tuning_results_path, _, _, _ = _find_existing_path(project_id, "adaptive_tuning_results.json", selected_engine)

        metadata = _read_json_if_exists(metadata_path)
        eval_history = _read_json_if_exists(eval_history_path) or []
        tuning_results = _read_json_if_exists(tuning_results_path)

        latest_eval = eval_history[-1] if isinstance(eval_history, list) and eval_history else {}
        first_eval = eval_history[0] if isinstance(eval_history, list) and eval_history else {}

        # Prefer eval history fields, then final metadata fallback.
        metrics = {
            "convergence_speed": latest_eval.get("convergence_speed"),
            "final_loss": latest_eval.get("final_loss"),
            "lpips_mean": latest_eval.get("lpips_mean"),
            "sharpness_mean": latest_eval.get("sharpness_mean"),
            "num_gaussians": latest_eval.get("num_gaussians"),
        }
        loss_milestones = {}
        if isinstance(latest_eval, dict):
            for k, v in latest_eval.items():
                if isinstance(k, str) and k.startswith("loss_at_"):
                    loss_milestones[k] = v

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
        if isinstance(tuning_results, dict):
            maybe_tuning_history = tuning_results.get("tuning_history")
            if isinstance(maybe_tuning_history, list):
                tuning_history = maybe_tuning_history
                tuning_history_count = len(tuning_history)
            tune_end_step = tuning_results.get("tune_end_step")
            maybe_final = tuning_results.get("final_params")
            if isinstance(maybe_final, dict):
                tune_end_params = maybe_final

        outputs = files.get_output_files(project_id)
        preview_url = None
        if resolved_engine and isinstance(outputs.get("engines"), dict):
            engine_bundle = outputs["engines"].get(resolved_engine, {})
            previews = engine_bundle.get("previews", {}) if isinstance(engine_bundle, dict) else {}
            preview_url = previews.get("latest_url")

        return {
            "project_id": project_id,
            "name": status_info.get("name"),
            "status": status_info.get("status"),
            "mode": status_info.get("mode") or (metadata.get("mode") if isinstance(metadata, dict) else None),
            "engine": resolved_engine or inferred_engine,
            "metrics": metrics,
            "tuning": {
                "initial": initial_tuning_params,
                "final": final_tuning_params,
                "end_params": tune_end_params,
                "end_step": tune_end_step if tune_end_step is not None else (metadata.get("tune_end_step") if isinstance(metadata, dict) else None),
                "runs": metadata.get("tuning_runs") if isinstance(metadata, dict) else None,
                "history_count": tuning_history_count,
                "history": tuning_history,
            },
            "loss_milestones": loss_milestones,
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
        for img in images:
            # Validate file extension
            file_ext = Path(img.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(f"Skipped invalid image: {img.filename}")
                continue
            
            content = await img.read()
            file_path = images_dir / img.filename
            with open(file_path, "wb") as f:
                f.write(content)
            uploaded_count += 1
            logger.info(f"Uploaded comparison image: {img.filename}")
        
        if uploaded_count == 0:
            raise HTTPException(status_code=400, detail="No valid images uploaded")
        
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
