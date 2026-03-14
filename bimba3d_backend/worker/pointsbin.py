"""Utility to convert COLMAP sparse reconstructions into compact points.bin files.

This mirrors app.services.pointsbin so the Docker worker can emit sparse point
clouds even when the full backend package is not present in the container.
"""

from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)

logger = logging.getLogger(__name__)


def _is_finite(value: float) -> bool:
    return value is not None and isinstance(value, float) and math.isfinite(value)


def _colmap_to_opengl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax, ay, az = COLMAP_TO_OPENGL
    return float(ax * x), float(ay * y), float(az * z)


def convert_colmap_recon_to_pointsbin(recon_dir: Path) -> int:
    """Convert COLMAP points3D files inside *recon_dir* into points.bin.

    The binary format is the same as the backend service expects:
      uint32 count
      count * float32 XYZ
      count * uint8 RGB
    Returns the number of points written (or 0 if none).
    """

    recon_dir = Path(recon_dir)
    txt = recon_dir / "points3D.txt"
    binf = recon_dir / "points3D.bin"
    out = recon_dir / "points.bin"

    points: list[tuple[float, float, float, int, int, int]] = []

    if txt.exists():
        logger.info("Parsing COLMAP ASCII points: %s", txt)
        try:
            with open(txt, "r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        r = int(parts[4]) & 0xFF
                        g = int(parts[5]) & 0xFF
                        b = int(parts[6]) & 0xFF
                    except Exception:
                        continue
                    if not (_is_finite(x) and _is_finite(y) and _is_finite(z)):
                        continue
                    ox, oy, oz = _colmap_to_opengl_coords(x, y, z)
                    points.append((ox, oy, oz, r, g, b))
        except Exception as err:
            logger.warning("Failed to parse points3D.txt: %s", err)

    elif binf.exists():
        logger.info("Parsing COLMAP binary points: %s", binf)
        try:
            with open(binf, "rb") as fh:
                header = fh.read(8)
                if len(header) < 8:
                    logger.warning("points3D.bin header too small")
                    return 0
                total = struct.unpack("<Q", header)[0]
                for _ in range(total):
                    pid_bytes = fh.read(8)
                    if len(pid_bytes) < 8:
                        break
                    xyz_bytes = fh.read(24)
                    if len(xyz_bytes) < 24:
                        break
                    x, y, z = struct.unpack("<ddd", xyz_bytes)
                    rgb_bytes = fh.read(3)
                    if len(rgb_bytes) < 3:
                        break
                    r, g, b = struct.unpack("BBB", rgb_bytes)
                    err_bytes = fh.read(8)
                    if len(err_bytes) < 8:
                        break
                    track_len_bytes = fh.read(8)
                    if len(track_len_bytes) < 8:
                        break
                    track_len = struct.unpack("<Q", track_len_bytes)[0]
                    # Skip track entries (best-effort: assume 8 bytes each)
                    try:
                        fh.read(int(track_len) * 8)
                    except Exception:
                        try:
                            fh.read(int(track_len) * 16)
                        except Exception:
                            pass
                    if not (_is_finite(float(x)) and _is_finite(float(y)) and _is_finite(float(z))):
                        continue
                    ox, oy, oz = _colmap_to_opengl_coords(float(x), float(y), float(z))
                    points.append((ox, oy, oz, int(r), int(g), int(b)))
        except Exception as err:
            logger.warning("Failed to parse points3D.bin: %s", err)
    else:
        logger.debug("No COLMAP points3D.txt or points3D.bin found in %s", recon_dir)

    count = len(points)
    if count == 0:
        logger.info("No valid points to write to points.bin in %s", recon_dir)
        if out.exists():
            try:
                out.unlink()
            except Exception:
                pass
        return 0

    try:
        with open(out, "wb") as fh:
            fh.write(struct.pack("<I", count))
            for x, y, z, _, _, _ in points:
                fh.write(struct.pack("<fff", float(x), float(y), float(z)))
            for _, _, _, r, g, b in points:
                fh.write(struct.pack("BBB", int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF))
        logger.info("WROTE %d points to %s", count, out)
        return count
    except Exception as err:
        logger.error("Failed to write points.bin for %s: %s", recon_dir, err)
        return 0
