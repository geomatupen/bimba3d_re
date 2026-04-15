from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

from PIL import ExifTags, Image

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys


EXIF_ONLY_FEATURE_KEYS: set[str] = {
    "camera_make",
    "camera_model",
    "camera_meta_missing",
    "lens_model",
    "lens_missing",
    "focal_length_mm",
    "focal_missing",
    "aperture_f",
    "aperture_missing",
    "shutter_s",
    "shutter_missing",
    "iso",
    "iso_missing",
    "camera_angle_bucket",
    "angle_missing",
    "gps_lat_mean",
    "gps_lon_mean",
    "gps_alt_mean",
    "gps_missing",
    "timestamp_mode",
    "timestamp_missing",
    "img_width_median",
    "img_height_median",
    "img_orientation",
    "orientation_missing",
    "img_size_missing",
}

def _iter_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def _read_exif(path: Path) -> tuple[dict[str, Any], int, int]:
    with Image.open(path) as img:
        width, height = img.size
        raw = img.getexif() or {}
        exif: dict[str, Any] = {}
        for key, value in raw.items():
            name = ExifTags.TAGS.get(key, key)
            exif[str(name)] = value
        return exif, width, height


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            if float(den) == 0.0:
                return None
            return float(num) / float(den)
        return float(value)
    except Exception:
        return None


def _extract_gps(exif: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    gps = exif.get("GPSInfo")
    if not isinstance(gps, dict):
        return None, None, None

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
    if angle <= -80:
        return "nadir"
    if angle <= -60:
        return "oblique"
    return "high_oblique"


def build_preset(ctx: ModeContext) -> PresetResult:
    files = _iter_images(ctx.image_dir)
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
    angle_buckets: list[str] = []
    gps_lat: list[float] = []
    gps_lon: list[float] = []
    gps_alt: list[float] = []
    timestamps: list[str] = []

    for path in sample:
        try:
            exif, width, height = _read_exif(path)
        except Exception:
            continue
        widths.append(int(width))
        heights.append(int(height))

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

        angle_buckets.append(_angle_bucket(exif))

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

    med_focal = float(median(focal_lengths)) if focal_lengths else 24.0
    med_aperture = float(median(apertures)) if apertures else 2.8
    med_exposure = float(median(exposure_times)) if exposure_times else 0.004
    med_iso = float(median(iso_values)) if iso_values else 100.0
    low_light_index = med_exposure * (med_iso / 100.0)

    orientations: list[str] = []
    for path in sample:
        try:
            exif, _, _ = _read_exif(path)
        except Exception:
            continue
        raw_orientation = exif.get("Orientation")
        if raw_orientation is None:
            continue
        try:
            orientations.append(str(int(raw_orientation)))
        except Exception:
            orientations.append(str(raw_orientation))

    orientation_value = orientations[0] if orientations else "1"
    orientation_missing = 0 if orientations else 1

    if low_light_index > 0.010:
        preset = "conservative"
    elif med_focal >= 30.0 and low_light_index < 0.006:
        preset = "geometry_fast"
    elif med_focal < 18.0 and low_light_index < 0.007:
        preset = "appearance_fast"
    else:
        preset = "balanced"

    updates = apply_preset_updates(ctx.params, preset)

    features = {
        "camera_make": makes[0] if makes else "unknown_make",
        "camera_model": models[0] if models else "unknown_model",
        "camera_meta_missing": 0 if (makes or models) else 1,
        "lens_model": lenses[0] if lenses else "unknown_lens",
        "lens_missing": 0 if lenses else 1,
        "focal_length_mm": med_focal,
        "focal_missing": 0 if focal_lengths else 1,
        "aperture_f": med_aperture,
        "aperture_missing": 0 if apertures else 1,
        "shutter_s": med_exposure,
        "shutter_missing": 0 if exposure_times else 1,
        "iso": med_iso,
        "iso_missing": 0 if iso_values else 1,
        "camera_angle_bucket": next((b for b in angle_buckets if b != "unknown_angle_bucket"), "unknown_angle_bucket"),
        "angle_missing": 0 if any(b != "unknown_angle_bucket" for b in angle_buckets) else 1,
        "gps_lat_mean": float(median(gps_lat)) if gps_lat else 0.0,
        "gps_lon_mean": float(median(gps_lon)) if gps_lon else 0.0,
        "gps_alt_mean": float(median(gps_alt)) if gps_alt else 0.0,
        "gps_missing": 0 if (gps_lat and gps_lon) else 1,
        "timestamp_mode": "exif" if timestamps else "filename",
        "timestamp_missing": 0 if timestamps else 1,
        "img_width_median": int(median(widths)) if widths else 4000,
        "img_height_median": int(median(heights)) if heights else 3000,
        "img_orientation": orientation_value,
        "orientation_missing": orientation_missing,
        "img_size_missing": 0 if (widths and heights) else 1,
    }
    features = keep_only_feature_keys(features, EXIF_ONLY_FEATURE_KEYS)

    notes = [
        "Derived from direct EXIF/file metadata with explicit fallback flags.",
        "Preset selected from conservative/balanced/geometry_fast/appearance_fast.",
    ]
    return PresetResult(mode="exif_only", updates=updates, features=features, notes=notes)
