import json
import logging
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Set

from bimba3d_backend.app.services import pointsbin

logger = logging.getLogger(__name__)


class SparseEditError(RuntimeError):
    """Domain-specific error raised when sparse edits cannot be applied."""


def apply_sparse_edits(
    project_dir: Path,
    candidate_dir: Path,
    candidate_rel: str,
    remove_point_ids: Set[int],
    *,
    create_backup: bool = True,
    reoptimize: bool = False,
) -> dict:
    """Apply point deletions (and optional COLMAP BA) to a sparse model."""
    if not remove_point_ids:
        raise SparseEditError("No point ids provided for deletion")

    if not candidate_dir.exists():
        raise SparseEditError("Sparse reconstruction directory not found")

    if create_backup:
        try:
            backup_path = _create_backup(candidate_dir)
            logger.info("Created sparse backup at %s", backup_path)
        except Exception as exc:
            logger.warning("Failed to create sparse backup: %s", exc)
            backup_path = None
    else:
        backup_path = None

    removed_points = 0
    try:
        with tempfile.TemporaryDirectory(prefix="sparse_edit_", dir=candidate_dir.parent) as tmp_root:
            tmp_path = Path(tmp_root)
            _run_model_converter(candidate_dir, tmp_path, "TXT")
            points_txt = tmp_path / "points3D.txt"
            images_txt = tmp_path / "images.txt"
            if not points_txt.exists():
                raise SparseEditError("points3D.txt missing after model conversion")
            removed_points, remaining_points_txt = _rewrite_points_txt(points_txt, remove_point_ids)
            if removed_points == 0:
                logger.info("No points matched requested deletions for %s", candidate_dir)
                return {
                    "removed_points": 0,
                    "remaining_points": remaining_points_txt,
                    "backup_path": str(backup_path) if backup_path else None,
                    "reoptimize_started": False,
                }
            if images_txt.exists():
                _rewrite_images_txt(images_txt, remove_point_ids)
            _run_model_converter(tmp_path, candidate_dir, "BIN")
    except SparseEditError:
        raise
    except subprocess.CalledProcessError as exc:
        logger.error("COLMAP model_converter failed: %s", exc)
        raise SparseEditError("COLMAP model_converter failed") from exc
    except Exception as exc:  # noqa: BLE001 - we want to wrap any failure
        logger.exception("Sparse edit failed")
        raise SparseEditError(str(exc)) from exc

    remaining_points = pointsbin.convert_colmap_recon_to_pointsbin(candidate_dir)
    _write_sparse_edit_log(
        project_dir,
        f"Removed {removed_points} sparse points from '{candidate_rel}' (remaining: {remaining_points}).",
    )

    reoptimize_started = False
    if reoptimize:
        thread = threading.Thread(
            target=_run_bundle_adjuster,
            args=(project_dir, candidate_dir, candidate_rel),
            name=f"bundle_adjuster_{candidate_dir.name}",
            daemon=True,
        )
        thread.start()
        reoptimize_started = True

    return {
        "removed_points": removed_points,
        "remaining_points": remaining_points,
        "backup_path": str(backup_path) if backup_path else None,
        "reoptimize_started": reoptimize_started,
    }


def _create_backup(candidate_dir: Path) -> Path:
    backup_root = candidate_dir.parent / ".backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = candidate_dir.name or "root"
    backup_path = backup_root / f"{name}-{stamp}"
    shutil.copytree(candidate_dir, backup_path, dirs_exist_ok=False)
    return backup_path


def _run_model_converter(input_path: Path, output_path: Path, output_type: str) -> None:
    cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        output_type,
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _rewrite_points_txt(points_txt: Path, remove_point_ids: Set[int]) -> tuple[int, int | None]:
    removed = 0
    remaining = 0
    lines_out: list[str] = []
    with open(points_txt, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines_out.append(line)
                continue
            parts = stripped.split()
            try:
                pid = int(parts[0])
            except Exception:
                lines_out.append(line)
                continue
            if pid in remove_point_ids:
                removed += 1
                continue
            remaining += 1
            lines_out.append(line)
    if removed == 0:
        return 0, remaining
    with open(points_txt, "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    return removed, remaining


def _rewrite_images_txt(images_txt: Path, remove_point_ids: Set[int]) -> None:
    lines_in = images_txt.read().splitlines()
    lines_out: list[str] = []
    idx = 0
    total_lines = len(lines_in)
    while idx < total_lines:
        line = lines_in[idx]
        lines_out.append(line)
        idx += 1
        if line.strip().startswith("#") or not line.strip():
            continue
        if idx >= total_lines:
            break
        points_line = lines_in[idx]
        tokens = points_line.strip().split()
        if tokens and len(tokens) % 3 == 0:
            for t in range(2, len(tokens), 3):
                try:
                    pid = int(tokens[t])
                except Exception:
                    continue
                if pid in remove_point_ids:
                    tokens[t] = "-1"
            lines_out.append(" ".join(tokens))
        else:
            lines_out.append(points_line)
        idx += 1
    images_txt.write_text("\n".join(lines_out) + "\n")


def _run_bundle_adjuster(project_dir: Path, candidate_dir: Path, candidate_rel: str) -> None:
    log_prefix = f"Bundle adjuster ({candidate_rel})"
    try:
        cmd = [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(candidate_dir),
            "--output_path",
            str(candidate_dir),
            "--BundleAdjustment.refine_extrinsics",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "1",
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        pointsbin.convert_colmap_recon_to_pointsbin(candidate_dir)
        _write_sparse_edit_log(project_dir, f"{log_prefix} completed successfully.")
    except subprocess.CalledProcessError as exc:
        logger.error("bundle_adjuster failed: %s", exc)
        _write_sparse_edit_log(project_dir, f"{log_prefix} failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("bundle_adjuster crashed")
        _write_sparse_edit_log(project_dir, f"{log_prefix} crashed: {exc}")


def _write_sparse_edit_log(project_dir: Path, message: str) -> None:
    log_file = project_dir / "processing.log"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        logger.debug("Unable to write sparse edit log for %s", project_dir)