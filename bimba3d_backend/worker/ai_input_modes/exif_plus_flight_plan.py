from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import ExifTags, Image

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys
from .exif_only import EXIF_ONLY_FEATURE_KEYS, build_preset as build_exif_only_preset


FLIGHT_PLAN_FEATURE_KEYS: set[str] = {
    "flight_type",
    "flight_type_missing",
    "camera_angle_profile",
    "angle_profile_missing",
    "average_altitude",
    "altitude_missing",
    "heading_consistency",
    "heading_missing",
    "coverage_spread",
    "coverage_missing",
    "overlap_proxy",
    "overlap_missing",
}


def _iter_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files[: min(48, len(files))]


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            den_f = float(den)
            if den_f == 0.0:
                return None
            return float(num) / den_f
        return float(value)
    except Exception:
        return None


def _read_exif(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        raw = img.getexif() or {}
    exif: dict[str, Any] = {}
    for key, value in raw.items():
        name = ExifTags.TAGS.get(key, key)
        exif[str(name)] = value
    return exif


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
    alt = _as_float(gps.get(6))
    return lat, lon, alt


def _extract_angle(exif: dict[str, Any]) -> float | None:
    for key in ("CameraElevationAngle", "Pitch"):
        val = _as_float(exif.get(key))
        if val is not None:
            return val
    return None


def _extract_timestamp(exif: dict[str, Any]) -> datetime | None:
    raw = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "").strip()
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    return None


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return deg


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def build_preset(ctx: ModeContext) -> PresetResult:
    base = build_exif_only_preset(ctx)
    updates = dict(base.updates)
    features = dict(base.features)
    notes = list(base.notes)

    gps_missing = int(features.get("gps_missing", 1) or 1)
    angle_missing = int(features.get("angle_missing", 1) or 1)
    timestamp_missing = int(features.get("timestamp_missing", 1) or 1)

    images = _iter_images(ctx.image_dir)
    points: list[tuple[float, float]] = []
    altitudes: list[float] = []
    timestamps: list[datetime] = []
    angles: list[float] = []

    for path in images:
        try:
            exif = _read_exif(path)
        except Exception:
            continue
        lat, lon, alt = _extract_gps(exif)
        ts = _extract_timestamp(exif)
        ang = _extract_angle(exif)
        if lat is not None and lon is not None:
            points.append((lat, lon))
        if alt is not None:
            altitudes.append(alt)
        if ts is not None:
            timestamps.append(ts)
        if ang is not None:
            angles.append(ang)

    if gps_missing or timestamp_missing or len(points) < 3:
        flight_type = "unknown_flight_type"
        heading_consistency = 0.5
        coverage_spread = 0.0
        overlap_proxy = 0.5
        flight_type_missing = 1
        heading_missing = 1
        coverage_missing = 1
        overlap_missing = 1
    else:
        bearings: list[float] = []
        steps_m: list[float] = []
        for idx in range(1, len(points)):
            (lat1, lon1), (lat2, lon2) = points[idx - 1], points[idx]
            bearings.append(_bearing_deg(lat1, lon1, lat2, lon2))
            steps_m.append(_haversine_m(lat1, lon1, lat2, lon2))

        if len(bearings) >= 2:
            deltas = []
            for i in range(1, len(bearings)):
                d = abs(bearings[i] - bearings[i - 1])
                deltas.append(min(d, 360.0 - d))
            turn_std = pstdev(deltas) if len(deltas) >= 2 else (deltas[0] if deltas else 0.0)
            heading_consistency = max(0.0, min(1.0, 1.0 - (turn_std / 90.0)))
        else:
            heading_consistency = 0.5

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        lat_span_m = _haversine_m(min(lats), mean(lons), max(lats), mean(lons)) if lats else 0.0
        lon_span_m = _haversine_m(mean(lats), min(lons), mean(lats), max(lons)) if lons else 0.0
        bbox_diag_m = math.sqrt((lat_span_m ** 2) + (lon_span_m ** 2))
        coverage_spread = max(0.0, min(1.0, bbox_diag_m / 600.0))

        avg_step = mean(steps_m) if steps_m else 0.0
        avg_alt = mean(altitudes) if altitudes else 120.0
        footprint = max(10.0, 1.2 * avg_alt)
        overlap_proxy = max(0.0, min(1.0, 1.0 - (avg_step / footprint)))

        end_distance = _haversine_m(points[0][0], points[0][1], points[-1][0], points[-1][1]) if len(points) > 1 else 0.0
        aspect = (max(lat_span_m, lon_span_m) / max(1.0, min(lat_span_m, lon_span_m))) if min(lat_span_m, lon_span_m) > 0 else 1.0
        angle_mean = mean(angles) if angles else -75.0

        if end_distance < max(20.0, 0.15 * bbox_diag_m) and heading_consistency < 0.55:
            flight_type = "orbit"
        elif aspect >= 3.0:
            flight_type = "corridor"
        elif coverage_spread >= 0.55 and heading_consistency >= 0.65:
            flight_type = "grid"
        elif angle_mean > -65.0:
            flight_type = "oblique"
        else:
            flight_type = "mixed"
        flight_type_missing = 0
        heading_missing = 0
        coverage_missing = 0
        overlap_missing = 0

    if angle_missing:
        camera_angle_profile = "unknown-angle profile"
        angle_profile_missing = 1
    else:
        if angles:
            angle_mean = mean(angles)
            angle_spread = pstdev(angles) if len(angles) >= 2 else 0.0
            camera_angle_profile = f"mean={angle_mean:.1f},spread={angle_spread:.1f}"
        else:
            bucket = str(features.get("camera_angle_bucket") or "unknown_angle_bucket")
            camera_angle_profile = "mostly_nadir" if bucket == "nadir" else "mostly_oblique"
        angle_profile_missing = 0

    avg_altitude = float(features.get("gps_alt_mean", 0.0) or 0.0)
    altitude_missing = 1 if gps_missing else 0

    # Decide preset using flight-aware logic from chapter intent.
    preset = str(base.updates.get("preset_name") or "balanced")
    if flight_type in {"corridor", "grid"} and heading_consistency >= 0.7 and angle_profile_missing == 0:
        preset = "geometry_fast"
    elif flight_type == "mixed":
        preset = "balanced"

    updates = apply_preset_updates(ctx.params, preset)

    features["flight_type"] = flight_type
    features["flight_type_missing"] = flight_type_missing
    features["camera_angle_profile"] = camera_angle_profile
    features["angle_profile_missing"] = angle_profile_missing
    features["average_altitude"] = avg_altitude
    features["altitude_missing"] = altitude_missing
    features["heading_consistency"] = heading_consistency
    features["heading_missing"] = heading_missing
    features["coverage_spread"] = coverage_spread
    features["coverage_missing"] = coverage_missing
    features["overlap_proxy"] = overlap_proxy
    features["overlap_missing"] = overlap_missing
    features = keep_only_feature_keys(features, EXIF_ONLY_FEATURE_KEYS | FLIGHT_PLAN_FEATURE_KEYS)

    notes.append("Added flight-plan features derived from image sequence metadata only.")
    return PresetResult(mode="exif_plus_flight_plan", updates=updates, features=features, notes=notes)
