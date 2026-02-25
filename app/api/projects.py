import threading
import logging
import shutil
import json
import time
from typing import Optional
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from app.config import DATA_DIR, ALLOWED_IMAGE_EXTENSIONS
from app.models.project import (
    ProjectResponse,
    StatusResponse,
    ProcessParams,
    ProjectListItem,
    CreateProjectRequest,
    UpdateProjectRequest,
    EvaluationMetrics,
    ComparisonRequest,
    ComparisonStatus,
)
from app.services import status, storage, colmap, gsplat, files
from worker import pipeline

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)


def _colmap_to_opengl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax, ay, az = COLMAP_TO_OPENGL
    return float(ax * x), float(ay * y), float(az * z)

logger = logging.getLogger(__name__)

router = APIRouter()

# Map EXIF GPS tag id for quick lookup
EXIF_GPS_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == "GPSInfo":
        EXIF_GPS_TAG = k
        break


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
            has_outputs = (project_dir / "outputs" / "splats.bin").exists() or (
                project_dir / "outputs" / "metadata.json"
            ).exists()
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
        
        # Update status to processing
        status.update_status(project_id, "processing", progress=5)
        
        # Start processing in background thread
        # Pass optional parameters to pipeline
        thread = threading.Thread(
            target=pipeline.run_full_pipeline,
            args=(project_id, (params.dict(exclude_none=True) if params else None)),
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
        from app.services.resume import can_resume_project
        
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
def get_preview_image(project_id: str, filename: str):
    """Serve a specific preview PNG from outputs/previews."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir = project_dir / "outputs" / "previews"
        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            raise HTTPException(status_code=404, detail="Preview not found")

        # Prevent path traversal
        if not str(img_path).startswith(str(previews_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            img_path,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview image for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preview image")


@router.head("/{project_id}/previews/{filename}")
def head_preview_image(project_id: str, filename: str):
    """HEAD probe for preview PNG (used by browsers for preflight)."""
    try:
        project_dir = DATA_DIR / project_id
        previews_dir = project_dir / "outputs" / "previews"
        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            raise HTTPException(status_code=404, detail="Preview not found")

        if not str(img_path).startswith(str(previews_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(img_path, media_type="image/png")
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
        
        preview_file = project_dir / "output" / "previews" / "preview_latest.png"
        if not preview_file.exists():
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
        
        # Read last N lines
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
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
def get_splat_format(project_id: str):
    """Check what splat format is available (ply or bin)."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        ply_path = project_dir / "outputs" / "splats.ply"
        bin_path = project_dir / "outputs" / "splats.bin"
        
        if ply_path.exists():
            return {"format": "ply", "has_ply": True, "has_bin": bin_path.exists()}
        elif bin_path.exists():
            return {"format": "bin", "has_ply": False, "has_bin": True}
        else:
            return {"format": "none", "has_ply": False, "has_bin": False}
    
    except Exception as e:
        logger.error(f"Error checking splat format: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check splat format")


@router.get("/{project_id}/download/splats.splat")
def download_splats_splat(project_id: str):
    """Download .splat file (optimized binary format for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        splat_path = project_dir / "outputs" / "splats.splat"
        
        if not splat_path.exists():
            raise HTTPException(status_code=404, detail=".splat file not found. Processing may not be complete.")
        
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
def head_splats_splat(project_id: str):
    """HEAD probe for .splat file (used by frontend to prefer native format)."""
    project_dir = DATA_DIR / project_id
    splat_path = project_dir / "outputs" / "splats.splat"
    if splat_path.exists():
        return FileResponse(path=splat_path, filename="splats.splat", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail=".splat file not found")


@router.get("/{project_id}/download/splats.ply")
def download_splats_ply(project_id: str):
    """Download PLY splats file."""
    try:
        project_dir = DATA_DIR / project_id
        ply_path = project_dir / "outputs" / "splats.ply"
        
        if not ply_path.exists():
            raise HTTPException(status_code=404, detail="PLY file not found")
        
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
def download_splats_bin(project_id: str):
    """Download binary splats file."""
    try:
        project_dir = DATA_DIR / project_id
        bin_path = project_dir / "outputs" / "splats.bin"
        
        if not bin_path.exists():
            raise HTTPException(status_code=404, detail="Binary file not found")
        
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
def download_points_bin(project_id: str):
    """Download compact `points.bin` generated from COLMAP reconstruction.

    The converter writes `points.bin` into the reconstruction directory (e.g. outputs/sparse/0/points.bin).
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        # Find first recon dir with points.bin
        for d in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            p = d / "points.bin"
            if p.exists():
                return FileResponse(path=p, filename="points.bin", media_type="application/octet-stream")

        raise HTTPException(status_code=404, detail="points.bin not found; reconstruction may not be converted yet")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading points.bin: {e}")
        raise HTTPException(status_code=500, detail="Failed to download points.bin")



@router.get("/{project_id}/download/splats")
def download_splats(project_id: str):
    """Download splats file (.splat format optimized for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Prefer .splat format (optimized for web)
        splat_path = project_dir / "outputs" / "splats.splat"
        if splat_path.exists():
            return FileResponse(
                path=splat_path,
                filename="splats.splat",
                media_type="application/octet-stream"
            )
        
        # Fall back to PLY format if .splat not available
        ply_path = project_dir / "outputs" / "splats.ply"
        if ply_path.exists():
            return FileResponse(
                path=ply_path,
                filename="splats.ply",
                media_type="application/octet-stream"
            )
        
        raise HTTPException(status_code=404, detail="Splats file not found. Processing may not be complete.")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading splats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download splats")


@router.get("/{project_id}/download/snapshots/{filename}")
def download_snapshot(project_id: str, filename: str):
    """Download a specific intermediate splat snapshot exported during training."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        snapshots_dir = project_dir / "outputs" / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            raise HTTPException(status_code=404, detail="No snapshots available")

        snapshots_root = snapshots_dir.resolve()
        snap_path = (snapshots_dir / filename).resolve()
        if snapshots_root not in snap_path.parents:
            raise HTTPException(status_code=403, detail="Access denied")
        if not snap_path.exists() or not snap_path.is_file():
            raise HTTPException(status_code=404, detail="Snapshot not found")

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


@router.get("/{project_id}/download/{file_type}")
def get_preview_download(project_id: str, file_type: str):
    """Fallback preview endpoint for legacy clients requesting `download/{file_type}`.

    This handler is intentionally registered after specific download endpoints so
    requests for `.splat`, `.ply`, `.bin` and `sparse.json` are handled by
    their dedicated handlers. It serves the latest preview PNG as a harmless
    fallback for UI polling requests.
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        preview_path = project_dir / "outputs" / "previews" / "preview_latest.png"
        if not preview_path.exists():
            raise HTTPException(status_code=404, detail="Preview not available")

        return FileResponse(
            path=preview_path,
            media_type="image/png",
            headers={"Cache-Control": "no-store"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview for download/{file_type}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get preview")


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
def get_metadata(project_id: str):
    """Get metadata.json for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        metadata_path = project_dir / "outputs" / "metadata.json"
        
        if not metadata_path.exists():
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
        
        # Check if metrics exist in metadata.json
        metadata_path = project_dir / "outputs" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Extract metrics from metadata
                metrics = metadata.get("evaluation_metrics", {})
                if metrics:
                    return EvaluationMetrics(**metrics)
        
        # Fallback: check adaptive_tuning_results.json
        tuning_path = project_dir / "outputs" / "adaptive_tuning_results.json"
        if tuning_path.exists():
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
