from __future__ import annotations

import math
import struct
from pathlib import Path
import re
from statistics import median
from typing import Any

from PIL import ExifTags, Image

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys


# ========== REDUCED FEATURE SET ==========
# Mode 1 (EXIF-only): 5 core features + 3 missing flags = 8 total
# Core features: focal_length_mm, shutter_s, iso, img_width_median, img_height_median
# NOTE: img_size_missing removed - image dimensions always available (real or safe default 4000×3000)
EXIF_ONLY_FEATURE_KEYS: set[str] = {
    "focal_length_mm",
    "focal_missing",
    "shutter_s",
    "shutter_missing",
    "iso",
    "iso_missing",
    "img_width_median",
    "img_height_median",
}

def _iter_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def _collect_processing_sizes(image_dir: Path, limit: int) -> tuple[list[int], list[int]]:
    widths: list[int] = []
    heights: list[int] = []
    files = _iter_images(image_dir)
    for path in files[: min(limit, len(files))]:
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception:
            continue
        widths.append(int(w))
        heights.append(int(h))
    return widths, heights


def _read_exif(path: Path) -> tuple[dict[str, Any], int, int]:
    with Image.open(path) as img:
        width, height = img.size
        raw = img.getexif() or {}
        exif: dict[str, Any] = {}
        for key, value in raw.items():
            name = ExifTags.TAGS.get(key, key)
            exif[str(name)] = value

        # Ensure GPSInfo is a parsed dict when available.
        if not isinstance(exif.get("GPSInfo"), dict) and hasattr(raw, "get_ifd"):
            try:
                gps_ifd = raw.get_ifd(0x8825)  # GPS IFD pointer tag
            except Exception:
                gps_ifd = None
            if isinstance(gps_ifd, dict):
                gps: dict[Any, Any] = {}
                for k, v in gps_ifd.items():
                    gps_name = ExifTags.GPSTAGS.get(k, k)
                    gps[gps_name] = v
                    gps[k] = v
                exif["GPSInfo"] = gps

        # DJI and similar cameras often keep useful capture metadata in XMP.
        xmp_attrs = _extract_xmp_attrs(img.info.get("xmp"))
        _apply_xmp_fallbacks(exif, xmp_attrs)
        return exif, width, height


def _extract_xmp_attrs(xmp_blob: Any) -> dict[str, str]:
    if isinstance(xmp_blob, (bytes, bytearray)):
        text = xmp_blob.decode("utf-8", errors="ignore")
    elif isinstance(xmp_blob, str):
        text = xmp_blob
    else:
        return {}

    attrs: dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z0-9_.:-]+)\s*=\s*"([^"]*)"', text):
        attrs[key] = value
    return attrs


def _xmp_pick(attrs: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _apply_xmp_fallbacks(exif: dict[str, Any], xmp_attrs: dict[str, str]) -> None:
    if not xmp_attrs:
        return

    if not exif.get("LensModel"):
        lens = _xmp_pick(xmp_attrs, ["aux:Lens", "exifEX:LensModel", "drone-dji:Lens"])
        if lens:
            exif["LensModel"] = lens

    if exif.get("FocalLength") is None:
        focal = _xmp_pick(xmp_attrs, ["exif:FocalLength", "tiff:FocalLength"])
        focal_val = _as_float(focal)
        if focal_val is None:
            # Calibrated focal is often in pixels; only accept if value looks like mm.
            calibrated = _as_float(_xmp_pick(xmp_attrs, ["drone-dji:CalibratedFocalLength"]))
            if calibrated is not None and 1.0 <= calibrated <= 100.0:
                focal_val = calibrated
        if focal_val is None:
            lens_text = str(exif.get("LensModel") or "")
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", lens_text, flags=re.IGNORECASE)
            if m:
                focal_val = _as_float(m.group(1))
        if focal_val is not None:
            exif["FocalLength"] = focal_val

    if exif.get("FNumber") is None:
        f_number = _as_float(_xmp_pick(xmp_attrs, ["exif:FNumber", "aux:LensInfoAperture"]))
        if f_number is None:
            lens_text = str(exif.get("LensModel") or "")
            m = re.search(r"f\s*/?\s*([0-9]+(?:\.[0-9]+)?)", lens_text, flags=re.IGNORECASE)
            if m:
                f_number = _as_float(m.group(1))
        if f_number is not None:
            exif["FNumber"] = f_number

    if exif.get("ExposureTime") is None:
        exp = _as_float(_xmp_pick(xmp_attrs, ["exif:ExposureTime"]))
        if exp is not None:
            exif["ExposureTime"] = exp

    if exif.get("ISOSpeedRatings") is None and exif.get("PhotographicSensitivity") is None:
        iso = _as_float(_xmp_pick(xmp_attrs, ["exif:ISOSpeedRatings", "exif:PhotographicSensitivity", "drone-dji:ISO"]))
        if iso is not None:
            exif["ISOSpeedRatings"] = iso

    if exif.get("DateTimeOriginal") is None:
        dt = _xmp_pick(xmp_attrs, ["exif:DateTimeOriginal", "xmp:CreateDate", "drone-dji:CreateDate"])
        if dt:
            exif["DateTimeOriginal"] = dt

    if exif.get("Pitch") is None and exif.get("CameraElevationAngle") is None:
        pitch = _as_float(_xmp_pick(xmp_attrs, ["drone-dji:GimbalPitchDegree", "drone-dji:FlightPitchDegree"]))
        if pitch is not None:
            exif["Pitch"] = pitch

    if not isinstance(exif.get("GPSInfo"), dict):
        lat = _as_float(_xmp_pick(xmp_attrs, ["drone-dji:GpsLatitude", "exif:GPSLatitude"]))
        lon = _as_float(_xmp_pick(xmp_attrs, ["drone-dji:GpsLongitude", "exif:GPSLongitude"]))
        alt = _as_float(_xmp_pick(xmp_attrs, ["drone-dji:AbsoluteAltitude", "drone-dji:RelativeAltitude", "exif:GPSAltitude"]))
        gps_payload: dict[str, float] = {}
        if lat is not None:
            gps_payload["lat"] = lat
        if lon is not None:
            gps_payload["lon"] = lon
        if alt is not None:
            gps_payload["alt"] = alt
        if gps_payload:
            exif["GPSInfo"] = gps_payload


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            if float(den) == 0.0:
                return None
            return float(num) / float(den)
        if isinstance(value, str) and "/" in value:
            num_s, den_s = value.split("/", 1)
            den = float(den_s.strip())
            if den == 0.0:
                return None
            return float(num_s.strip()) / den
        return float(value)
    except Exception:
        return None


def _extract_gps(exif: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    gps = exif.get("GPSInfo")
    if not isinstance(gps, dict):
        return None, None, None

    # Support direct decimal GPS payloads (used by XMP fallback path).
    lat_direct = _as_float(gps.get("lat"))
    lon_direct = _as_float(gps.get("lon"))
    alt_direct = _as_float(gps.get("alt"))
    if lat_direct is not None or lon_direct is not None or alt_direct is not None:
        return lat_direct, lon_direct, alt_direct

    def _to_deg(ref_key: int, val_key: int) -> float | None:
        ref = gps.get(ref_key)
        val = gps.get(val_key)
        if not isinstance(val, (tuple, list)) or len(val) != 3:
            return None
        d = _as_float(val[0])
        m = _as_float(val[1])
        s = _as_float(val[2])
        if d is None or m is None or s is None:
            return None
        out = d + (m / 60.0) + (s / 3600.0)
        if str(ref).upper() in {"S", "W"}:
            out = -out
        return out

    lat = _to_deg(1, 2)
    lon = _to_deg(3, 4)
    alt_raw = gps.get(6)
    alt = _as_float(alt_raw)
    return lat, lon, alt


def _angle_bucket(exif: dict[str, Any]) -> str:
    # Best-effort bucket; many datasets do not expose gimbal pitch in standard EXIF.
    angle = _as_float(exif.get("CameraElevationAngle"))
    if angle is None:
        angle = _as_float(exif.get("Pitch"))
    if angle is None:
        return "unknown_angle_bucket"
    return _angle_bucket_from_pitch(angle)


def _angle_bucket_from_pitch(angle: float) -> str:
    """Classify pitch angle into bucket: nadir / oblique / high_oblique."""
    if angle <= -80:
        return "nadir"
    if angle <= -60:
        return "oblique"
    return "high_oblique"


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def _candidate_colmap_images_bin(colmap_dir: Path) -> list[Path]:
    base = Path(colmap_dir)
    candidates = [
        base / "images.bin",
        base / "0" / "images.bin",
        base / "sparse" / "images.bin",
        base / "sparse" / "0" / "images.bin",
    ]
    return [p for p in candidates if p.exists() and p.is_file()]


def _collect_colmap_pitch_angles(colmap_dir: Path, limit: int = 48) -> list[float]:
    candidates = _candidate_colmap_images_bin(colmap_dir)
    if not candidates:
        return []

    # Use the first valid COLMAP images.bin candidate.
    path = candidates[0]
    entries: list[tuple[str, float]] = []
    try:
        with path.open("rb") as f:
            num_images = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_images):
                _image_id = struct.unpack("I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
                _tx, _ty, _tz = struct.unpack("ddd", f.read(24))
                _camera_id = struct.unpack("I", f.read(4))[0]

                name_bytes = bytearray()
                while True:
                    ch = f.read(1)
                    if ch == b"" or ch == b"\x00":
                        break
                    name_bytes.extend(ch)
                name = name_bytes.decode("utf-8", errors="ignore")

                num_points = struct.unpack("Q", f.read(8))[0]
                f.read(24 * num_points)

                r = _qvec_to_rotmat(qw, qx, qy, qz)
                # COLMAP qvec is world->camera rotation. Camera forward in world is R^T * [0,0,1],
                # which equals the 3rd row of R. Map Z component to EXIF-like pitch range [-90, 90].
                fz = max(-1.0, min(1.0, r[2][2]))
                pitch_deg = math.degrees(math.asin(fz))
                entries.append((name, pitch_deg))
    except Exception:
        return []

    entries.sort(key=lambda t: t[0])
    return [pitch for _, pitch in entries[: min(limit, len(entries))]]


def build_preset(ctx: ModeContext) -> PresetResult:
    files = _iter_images(ctx.metadata_image_dir)
    sample = files[: min(24, len(files))]

    widths: list[int] = []
    heights: list[int] = []
    makes: list[str] = []
    models: list[str] = []
    lenses: list[str] = []
    focal_lengths: list[float] = []
    apertures: list[float] = []
    exposure_times: list[float] = []
    iso_values: list[float] = []
    gps_lat: list[float] = []
    gps_lon: list[float] = []
    gps_alt: list[float] = []
    timestamps: list[str] = []

    for path in sample:
        try:
            exif, width, height = _read_exif(path)
        except Exception:
            continue
        make = str(exif.get("Make") or "").strip()
        model = str(exif.get("Model") or "").strip()
        lens = str(exif.get("LensModel") or "").strip()
        if make:
            makes.append(make)
        if model:
            models.append(model)
        if lens:
            lenses.append(lens)

        focal = _as_float(exif.get("FocalLength"))
        if focal is not None:
            focal_lengths.append(focal)
        aperture = _as_float(exif.get("FNumber"))
        if aperture is not None:
            apertures.append(aperture)
        exposure = _as_float(exif.get("ExposureTime"))
        if exposure is not None:
            exposure_times.append(exposure)
        iso = _as_float(exif.get("ISOSpeedRatings"))
        if iso is not None:
            iso_values.append(iso)

        lat, lon, alt = _extract_gps(exif)
        if lat is not None:
            gps_lat.append(lat)
        if lon is not None:
            gps_lon.append(lon)
        if alt is not None:
            gps_alt.append(alt)

        ts = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "").strip()
        if ts:
            timestamps.append(ts)

    # ========== REDUCED FEATURE CALCULATION ==========

    # 1. FOCAL LENGTH (primary image scale parameter)
    med_focal = float(median(focal_lengths)) if focal_lengths else 24.0
    # Clamp to reasonable range: 8mm (wide) to 300mm (telephoto)
    med_focal = max(8.0, min(300.0, med_focal))
    focal_missing = 0 if focal_lengths else 1

    # 2. SHUTTER SPEED (exposure duration - affects motion blur)
    med_exposure = float(median(exposure_times)) if exposure_times else 0.004  # 4ms typical
    # Clamp to reasonable range: 1/10000s to 1s
    med_exposure = max(0.0001, min(1.0, med_exposure))
    shutter_missing = 0 if exposure_times else 1

    # 3. ISO VALUE (sensor sensitivity)
    med_iso = float(median(iso_values)) if iso_values else 100.0
    # Clamp to reasonable range: 50 to 102400
    med_iso = max(50.0, min(102400.0, med_iso))
    iso_missing = 0 if iso_values else 1

    # 4. IMAGE DIMENSIONS (resolution affects feature extraction quality)
    widths, heights = _collect_processing_sizes(ctx.processing_image_dir, limit=24)
    img_width_med = int(median(widths)) if widths else 4000
    img_height_med = int(median(heights)) if heights else 3000
    # Clamp to reasonable range: 640x480 to 8000x6000
    img_width_med = max(640, min(8000, img_width_med))
    img_height_med = max(480, min(6000, img_height_med))
    # NOTE: No img_size_missing flag - dimensions always available (real or safe default)

    # ========== PRESET SELECTION LOGIC ==========
    low_light_index = med_exposure * (med_iso / 100.0)
    if low_light_index > 0.010:
        preset = "conservative"
    elif med_focal >= 30.0 and low_light_index < 0.006:
        preset = "geometry_fast"
    elif med_focal < 18.0 and low_light_index < 0.007:
        preset = "appearance_fast"
    else:
        preset = "balanced"

    updates = apply_preset_updates(ctx.params, preset)

    # ========== BUILD FEATURE DICTIONARY ==========
    features = {
        "focal_length_mm": med_focal,
        "focal_missing": focal_missing,
        "shutter_s": med_exposure,
        "shutter_missing": shutter_missing,
        "iso": med_iso,
        "iso_missing": iso_missing,
        "img_width_median": img_width_med,
        "img_height_median": img_height_med,
    }
    features = keep_only_feature_keys(features, EXIF_ONLY_FEATURE_KEYS)

    notes = [
        "MODE 1 (EXIF-only): 8-feature reduced set.",
        "Features: focal_length_mm, shutter_s, iso, img_width_median, img_height_median + 3 missing flags.",
        "Image dimensions always available (real or safe default).",
    ]
    return PresetResult(mode="exif_only", updates=updates, features=features, notes=notes)
