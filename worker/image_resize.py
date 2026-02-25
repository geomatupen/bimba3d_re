"""Helpers for preparing downscaled training image sets."""
from __future__ import annotations

import logging
import shutil
import json
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from app.config import ALLOWED_IMAGE_EXTENSIONS as _ALLOWED_EXTENSIONS
    _SUPPORTED_SUFFIXES = {ext.lower() for ext in _ALLOWED_EXTENSIONS}
except Exception:  # pragma: no cover - fallback when app.config is unavailable
    _SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_FORMAT_BY_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

try:
    _RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:  # pragma: no cover - Pillow < 9 fallback
    _RESAMPLE = Image.LANCZOS

logger = logging.getLogger(__name__)


def normalize_max_size(value: object) -> Optional[int]:
    """Convert a loose value into a positive integer max size or None."""
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def prepare_training_images(source_dir: Path, project_dir: Path, max_size: int) -> Tuple[Path, dict]:
    """Return a directory where images are limited to ``max_size`` per dimension.

    Args:
        source_dir: Directory containing original uploads.
        project_dir: Project root (parent of images/ and outputs/).
        max_size: Maximum width/height in pixels for resized copies.

    Returns:
        (resized_dir, stats) tuple describing the prepared directory and
        high-level counters for logging/telemetry.
    """
    if max_size <= 0:
        return source_dir, {"total": 0, "resized": 0, "copied": 0, "reused": 0, "failed": 0, "removed": 0}
    if not source_dir.exists():
        raise FileNotFoundError(f"Source images directory not found: {source_dir}")

    resized_dir = project_dir / "images_resized"
    _reset_dir_if_needed(resized_dir, max_size)
    resized_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(resized_dir, max_size)

    stats = {"total": 0, "resized": 0, "copied": 0, "reused": 0, "failed": 0, "removed": 0}
    processed_names = set()

    for src_path in sorted(source_dir.iterdir()):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue

        processed_names.add(src_path.name)
        stats["total"] += 1
        dst_path = resized_dir / src_path.name

        if _is_up_to_date(src_path, dst_path):
            stats["reused"] += 1
            continue

        try:
            with Image.open(src_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                max_dim = max(width, height)

                if max_dim <= max_size:
                    _copy_source(src_path, dst_path)
                    stats["copied"] += 1
                    continue

                scale = max_size / float(max_dim)
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )

                fmt = _resolve_format(img.format, src_path.suffix)
                save_kwargs = {"optimize": True}
                if fmt in {"JPEG", "JPG"}:
                    save_kwargs["quality"] = 92

                resized = img.resize(new_size, _RESAMPLE)
                if fmt in {"JPEG", "JPG"} and resized.mode not in ("RGB", "L"):
                    resized = resized.convert("RGB")

                tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
                resized.save(tmp_path, fmt, **save_kwargs)
                tmp_path.replace(dst_path)
                stats["resized"] += 1
        except (UnidentifiedImageError, OSError) as exc:
            stats["failed"] += 1
            logger.warning("Skipping corrupt/unreadable image %s: %s", src_path.name, exc)
        except Exception as exc:  # pragma: no cover - defensive catch
            stats["failed"] += 1
            logger.warning("Could not resize %s: %s", src_path.name, exc)

    # Remove orphaned files from previous runs so dataset mirrors uploads exactly
    for existing in resized_dir.iterdir():
        if existing.name == ".meta.json":
            continue
        if existing.is_file() and existing.name not in processed_names:
            try:
                existing.unlink()
                stats["removed"] += 1
            except Exception:
                pass

    return resized_dir, stats


def _is_up_to_date(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    try:
        return dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size > 0
    except OSError:
        return False


def _copy_source(src: Path, dst: Path) -> None:
    tmp_path = dst.with_suffix(dst.suffix + ".tmpcopy")
    shutil.copy2(src, tmp_path)
    tmp_path.replace(dst)


def _reset_dir_if_needed(resized_dir: Path, max_size: int) -> None:
    meta_path = resized_dir / ".meta.json"
    if not resized_dir.exists():
        return
    previous_size = _read_metadata(meta_path)
    if previous_size == max_size:
        return
    try:
        shutil.rmtree(resized_dir)
    except FileNotFoundError:
        return


def _read_metadata(meta_path: Path) -> Optional[int]:
    try:
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        size = int(data.get("max_size"))
        return size if size > 0 else None
    except Exception:
        return None


def _write_metadata(resized_dir: Path, max_size: int) -> None:
    meta_path = resized_dir / ".meta.json"
    try:
        meta_path.write_text(json.dumps({"max_size": max_size}))
    except Exception:
        logger.debug("Failed to write resize metadata for %s", resized_dir)


def _resolve_format(found_format: Optional[str], suffix: str) -> str:
    if found_format:
        return found_format.upper()
    return _FORMAT_BY_SUFFIX.get(suffix.lower(), "PNG")
