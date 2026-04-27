import logging
import re
from pathlib import Path
from typing import Optional

from bimba3d_backend.app.config import DATA_DIR

logger = logging.getLogger(__name__)


def _find_project_dir(project_id: str) -> Optional[Path]:
    """Find project directory using the API layer's canonical resolver.

    The resolver knows about normal projects and registered pipeline folders.  This
    avoids the `/files` endpoint finding a pipeline project while this service then
    looks only under DATA_DIR and returns an empty image list.
    """
    try:
        from bimba3d_backend.app.api.projects import _find_project_dir as resolve_project_dir

        return resolve_project_dir(project_id)
    except Exception as exc:
        logger.warning("Canonical project lookup failed for %s: %s", project_id, exc)

    project_dir = DATA_DIR / project_id
    return project_dir if project_dir.exists() else None

ENGINE_SUBDIR = "engines"


def _collect_outputs(root_dir: Path, project_id: str, engine: str, run_id: str | None = None) -> dict:
    """Collect export artifacts for a specific engine output directory."""
    bundle: dict = {}
    query_parts = [f"engine={engine}"]
    if run_id:
        query_parts.append(f"run_id={run_id}")
    query_suffix = "?" + "&".join(query_parts)

    if not root_dir.exists():
        return bundle

    def add_splats_entry(candidate: Path, fmt: str):
        if not candidate.exists():
            return None
        download_target = {
            "splat": "splats.splat",
            "ply": "splats.ply",
            "bin": "splats.bin",
        }.get(fmt)
        if not download_target:
            return None
        entry = {
            "format": fmt,
            "path": str(candidate),
            "size": candidate.stat().st_size,
            "url": f"/projects/{project_id}/download/{download_target}{query_suffix}",
        }
        bundle["splats"] = entry
        return entry

    def add_best_splat_entry(candidate: Path):
        if not candidate.exists():
            return None
        entry = {
            "format": "splat",
            "path": str(candidate),
            "size": candidate.stat().st_size,
            "url": f"/projects/{project_id}/download/best.splat{query_suffix}",
        }
        bundle["best_splat"] = entry
        return entry

    # Final artifacts (splat/ply/bin)
    add_splats_entry(root_dir / "splats.splat", "splat")
    if "splats" not in bundle:
        add_splats_entry(root_dir / "splats.ply", "ply")
    if "splats" not in bundle:
        add_splats_entry(root_dir / "splats.bin", "bin")
    add_best_splat_entry(root_dir / "best.splat")

    metadata_path = root_dir / "metadata.json"
    if metadata_path.exists():
        bundle["metadata"] = {
            "path": str(metadata_path),
            "size": metadata_path.stat().st_size,
            "type": "json",
            "engine": engine,
        }

    metrics_path = root_dir / "metrics.json"
    if metrics_path.exists():
        bundle["metrics"] = {
            "path": str(metrics_path),
            "size": metrics_path.stat().st_size,
            "engine": engine,
        }

    previews_dir = root_dir / "previews"
    if previews_dir.exists():
        previews = []
        for preview in sorted(previews_dir.glob("preview_*.png")):
            previews.append({
                "name": preview.name,
                "path": str(preview),
                "size": preview.stat().st_size,
                "url": f"/projects/{project_id}/previews/{preview.name}{query_suffix}",
            })
        latest_preview = previews_dir / "preview_latest.png"
        if previews or latest_preview.exists():
            latest_url = None
            if latest_preview.exists():
                latest_url = f"/projects/{project_id}/previews/{latest_preview.name}{query_suffix}"
            bundle["previews"] = {
                "items": previews,
                "latest": str(latest_preview) if latest_preview.exists() else None,
                "latest_url": latest_url,
            }

    ckpt_dir = root_dir / "ckpts"
    if ckpt_dir.exists():
        checkpoints = []
        for ckpt in sorted(ckpt_dir.glob("*")):
            if not ckpt.is_file():
                continue
            if ckpt.suffix.lower() not in {".pt", ".pth"}:
                continue
            checkpoints.append({
                "name": ckpt.name,
                "path": str(ckpt),
                "size": ckpt.stat().st_size,
            })
        if checkpoints:
            bundle["checkpoints"] = checkpoints

    snapshots_dir = root_dir / "snapshots"
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        snapshots: list[dict] = []
        for snapshot in sorted(snapshots_dir.glob("*")):
            if not snapshot.is_file():
                continue
            file_suffix = snapshot.suffix.lower().lstrip(".") or "splat"
            step = None
            stem = snapshot.stem.split("_")[-1]
            if stem.isdigit():
                step = int(stem)
            else:
                match = re.search(r"(\d+)", snapshot.stem)
                if match:
                    step = int(match.group(1))
            snapshots.append({
                "name": snapshot.name,
                "path": str(snapshot),
                "size": snapshot.stat().st_size,
                "format": file_suffix,
                "step": step,
                "url": f"/projects/{project_id}/download/snapshots/{snapshot.name}{query_suffix}",
            })
        if snapshots:
            bundle["model_snapshots"] = snapshots

    return bundle


def get_output_files(project_id: str, run_id: str | None = None) -> dict:
    """
    Get list of output files for a project.
    """
    project_dir = _find_project_dir(project_id)
    if not project_dir:
        return {}
    
    files = {}

    # Check for images - handle both regular directories and Windows junctions (pipeline projects)
    images_dir = project_dir / "images"
    effective_images_dir = images_dir

    # For pipeline projects, the images dir may be a Windows junction that glob() can't traverse.
    # Fall back to source_dir from config.json if images_dir is empty.
    if images_dir.exists():
        images = []
        try:
            for img in sorted(images_dir.iterdir()):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"} and img.is_file():
                    images.append({
                        "name": img.name,
                        "path": str(img),
                        "size": img.stat().st_size,
                    })
        except Exception as exc:
            logger.debug("Failed to iterate images_dir %s: %s", images_dir, exc)

        # If no images found via junction, try source_dir from config.json
        if not images:
            config_file = project_dir / "config.json"
            if config_file.exists():
                try:
                    import json
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    source_dir = config.get("source_dir")
                    if source_dir:
                        source_path = Path(source_dir)
                        if source_path.exists() and source_path.is_dir():
                            effective_images_dir = source_path
                            for img in sorted(source_path.iterdir()):
                                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"} and img.is_file():
                                    images.append({
                                        "name": img.name,
                                        "path": str(img),
                                        "size": img.stat().st_size,
                                    })
                except Exception as exc:
                    logger.debug("Failed to read source_dir from config for %s: %s", project_id, exc)

        if images:
            files["images"] = images

    output_dir = project_dir / "outputs"
    if run_id:
        output_dir = project_dir / "runs" / run_id / "outputs"
    elif not (output_dir / ENGINE_SUBDIR).exists():
        # Pipeline/manual session artifacts are stored under runs/<run_id>/outputs.
        # When the caller asks for project-level files, expose the latest run's
        # artifacts so project tabs behave like an automated project run.
        runs_root = project_dir / "runs"
        if runs_root.exists() and runs_root.is_dir():
            try:
                run_dirs = sorted(
                    [p for p in runs_root.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in run_dirs:
                    candidate_outputs = run_dir / "outputs"
                    if candidate_outputs.exists():
                        output_dir = candidate_outputs
                        break
            except Exception as exc:
                logger.debug("Failed to locate latest run outputs for %s: %s", project_id, exc)
    
    if not output_dir.exists():
        return files

    # Check for COLMAP sparse reconstructions under outputs/sparse/*
    # For pipeline projects, sparse is at project level while engines are in runs
    # So check both the current output_dir and the project-level outputs/sparse
    sparse_roots_to_check = [output_dir / "sparse"]

    # If output_dir is in a run directory, also check project-level sparse
    if run_id or "/runs/" in str(output_dir) or "\\runs\\" in str(output_dir):
        project_level_sparse = project_dir / "outputs" / "sparse"
        if project_level_sparse != sparse_roots_to_check[0]:
            sparse_roots_to_check.insert(0, project_level_sparse)

    reconstructions = []
    for sparse_root in sparse_roots_to_check:
        if sparse_root.exists() and sparse_root.is_dir():
            for recon in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
                recon_info = {"name": recon.name, "path": str(recon), "files": []}
                # Look for common COLMAP outputs
                points = recon / "points3D.bin"
                cams = recon / "cameras.bin"
                imgs = recon / "images.bin"
                proj = recon / "project.ini"
                if points.exists():
                    recon_info["complete"] = True
                else:
                    recon_info["complete"] = False
                for f in (points, cams, imgs, proj):
                    if f.exists():
                        recon_info["files"].append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
                reconstructions.append(recon_info)
            # Only check first location with sparse data
            if reconstructions:
                break

    if reconstructions:
        files["sparse"] = reconstructions

    engines_root = output_dir / ENGINE_SUBDIR
    if engines_root.exists() and engines_root.is_dir():
        engine_entries = {}
        for engine_dir in sorted([p for p in engines_root.iterdir() if p.is_dir()]):
            engine_name = engine_dir.name
            bundle = _collect_outputs(engine_dir, project_id, engine_name, run_id=run_id)
            if bundle:
                engine_entries[engine_name] = bundle
        if engine_entries:
            files["engines"] = engine_entries
    
    return files
