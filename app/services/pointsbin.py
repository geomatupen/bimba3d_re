import struct
import math
import logging
from pathlib import Path

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)

logger = logging.getLogger(__name__)


def _is_finite(v: float) -> bool:
    return v is not None and isinstance(v, float) and math.isfinite(v)


def _colmap_to_opengl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax, ay, az = COLMAP_TO_OPENGL
    return float(ax * x), float(ay * y), float(az * z)


def convert_colmap_recon_to_pointsbin(recon_dir: Path) -> int:
    """Convert a COLMAP reconstruction into compact viewer/export formats.

    Outputs two files alongside the reconstruction directory:
        - points.bin (viewer format: positions + rgb bytes)
        - points_editable.bin (editor format: positions + normalized rgb + point ids)

    Both hand back the total number of finite points written.
    """
    recon_dir = Path(recon_dir)
    txt = recon_dir / "points3D.txt"
    binf = recon_dir / "points3D.bin"
    out = recon_dir / "points.bin"

    points = []  # list of (point_id, x, y, z, r, g, b)

    if txt.exists():
        logger.info(f"Parsing COLMAP ASCII points: {txt}")
        try:
            with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # Expect at least id + x y z r g b
                    if len(parts) < 7:
                        continue
                    try:
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        x, y, z = _colmap_to_opengl_coords(x, y, z)
                        r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                        if not (_is_finite(x) and _is_finite(y) and _is_finite(z)):
                            continue
                        pid = int(parts[0])
                        points.append((pid, x, y, z, r & 0xFF, g & 0xFF, b & 0xFF))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to parse points3D.txt: {e}")

    elif binf.exists():
        logger.info(f"Parsing COLMAP binary points: {binf}")
        try:
            with open(binf, "rb") as f:
                # Read header: number of points (uint64)
                hdr = f.read(8)
                if len(hdr) < 8:
                    logger.warning("points3D.bin header too small")
                else:
                    num_points = struct.unpack("<Q", hdr)[0]
                    for i in range(num_points):
                        # Each point typically: id(uint64), xyz(3*doubles), rgb(3*uint8), error(double), track_length(uint64), track_entries
                        pid_b = f.read(8)
                        if len(pid_b) < 8:
                            break
                        try:
                            point_id = struct.unpack("<Q", pid_b)[0]
                            xyz_b = f.read(24)
                            if len(xyz_b) < 24:
                                break
                            x, y, z = struct.unpack("<ddd", xyz_b)
                            rgb_b = f.read(3)
                            if len(rgb_b) < 3:
                                break
                            r, g, b = struct.unpack("BBB", rgb_b)
                            # skip error
                            err_b = f.read(8)
                            if len(err_b) < 8:
                                break
                            # track length
                            track_len_b = f.read(8)
                            if len(track_len_b) < 8:
                                break
                            track_len = struct.unpack("<Q", track_len_b)[0]
                            # skip track entries (most commonly 8 bytes each: uint32 image_id, uint32 point2d_idx)
                            try:
                                f.read(int(track_len) * 8)
                            except Exception:
                                # Best-effort; try 16 bytes each
                                try:
                                    f.read(int(track_len) * 16)
                                except Exception:
                                    pass

                            if not (_is_finite(x) and _is_finite(y) and _is_finite(z)):
                                continue
                            x, y, z = _colmap_to_opengl_coords(float(x), float(y), float(z))
                            points.append((int(point_id), x, y, z, int(r), int(g), int(b)))
                        except Exception:
                            # On parse error, try to continue to next point
                            continue
        except Exception as e:
            logger.warning(f"Failed to parse points3D.bin: {e}")
    else:
        logger.debug("No COLMAP points3D.txt or points3D.bin found in recon dir")

    # Write compact binary
    count = len(points)
    if count == 0:
        logger.info("No valid points to write to points.bin")
        if out.exists():
            try:
                out.unlink()
            except Exception:
                pass
        editable = recon_dir / "points_editable.bin"
        if editable.exists():
            try:
                editable.unlink()
            except Exception:
                pass
        return 0

    try:
        _write_points_bin(points, out)
        _write_points_editable(points, recon_dir / "points_editable.bin")
        logger.info(f"WROTE {count} points to {out}")
        return count
    except Exception as e:
        logger.error(f"Failed to write points data: {e}")
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        editable = recon_dir / "points_editable.bin"
        if editable.exists():
            try:
                editable.unlink()
            except Exception:
                pass
        return 0


def _write_points_bin(points: list[tuple[int, float, float, float, int, int, int]], out: Path) -> None:
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(points)))
        for _, x, y, z, _, _, _ in points:
            f.write(struct.pack("<fff", float(x), float(y), float(z)))
        for _, _, _, _, r, g, b in points:
            f.write(struct.pack("BBB", int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF))


def _write_points_editable(points: list[tuple[int, float, float, float, int, int, int]], out: Path) -> None:
    """Write editor-friendly format with normalized RGB + point ids.

    Format:
      uint32 count
      count * float32 positions (x, y, z)
      count * float32 colors (r, g, b normalized 0-1)
      count * uint32 point ids
    """
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(points)))
        for _, x, y, z, _, _, _ in points:
            f.write(struct.pack("<fff", float(x), float(y), float(z)))
        for _, _, _, _, r, g, b in points:
            f.write(struct.pack(
                "<fff",
                float(r) / 255.0,
                float(g) / 255.0,
                float(b) / 255.0,
            ))
        for point_id, *_ in points:
            if point_id < 0 or point_id > 0xFFFFFFFF:
                raise ValueError("point3D_id exceeds uint32 capacity")
            f.write(struct.pack("<I", int(point_id)))
