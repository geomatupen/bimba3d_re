from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys
from .exif_only import (
    EXIF_ONLY_FEATURE_KEYS,
    build_preset as build_exif_only_preset,
    _angle_bucket_from_pitch,
    _collect_colmap_pitch_angles,
    _extract_gps as _extract_gps_shared,
    _read_exif as _read_exif_shared,
)
from .feature_utils import calculate_gsd, infer_sensor_width


# ========== REDUCED FLIGHT PLAN FEATURE SET ==========
# Mode 2 additions (stacked on Mode 1): 5 core features + 5 missing flags = 10 total
# Core features: gsd_median, overlap_proxy, coverage_spread, camera_angle_bucket, heading_consistency
# NOTE: Replaces average_altitude with GSD (Ground Sampling Distance) for better scene characterization
# NOTE: camera_angle_bucket is numeric-encoded in this mode:
#   0=missing, 1=nadir, 2=oblique, 3=mixed
#   where "oblique" includes previous oblique/high_oblique classes.
FLIGHT_PLAN_FEATURE_KEYS: set[str] = {
    "gsd_median",
    "gsd_missing",
    "overlap_proxy",
    "overlap_missing",
    "coverage_spread",
    "coverage_missing",
    "camera_angle_bucket",
    "angle_missing",
    "heading_consistency",
    "heading_missing",
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
    timestamp_missing = int(features.get("timestamp_missing", 1) or 1)

    images = _iter_images(ctx.metadata_image_dir)
    points: list[tuple[float, float]] = []
    altitudes: list[float] = []
    timestamps: list[datetime] = []
    angles: list[float] = []

    for path in images:
        try:
            exif, _, _ = _read_exif_shared(path)
        except Exception:
            continue
        lat, lon, alt = _extract_gps_shared(exif)
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

    # ========== REDUCED FEATURE CALCULATION (10 features: 5 core + 5 missing flags) ==========
    # Extract altitude data for GSD calculation (separate from feature set)
    avg_altitude = mean(altitudes) if altitudes else 120.0
    gps_data_available = len(points) >= 3 and not gps_missing
    
    # 1. GSD (GROUND SAMPLING DISTANCE) - replaces raw altitude
    # GSD represents effective resolution at ground: lower = more detail
    if gps_data_available:
        # Get image dimensions from Mode 1 features
        img_width = int(features.get("img_width_median", 4000))
        # Extract focal length and infer sensor width
        focal_mm = float(features.get("focal_length_mm", 24.0))
        # Try to infer sensor width from EXIF metadata
        camera_model = ""
        try:
            first_img = _iter_images(ctx.metadata_image_dir)[0] if _iter_images(ctx.metadata_image_dir) else None
            if first_img:
                exif, _, _ = _read_exif_shared(first_img)
                camera_model = str(exif.get("Model", ""))
        except Exception:
            pass
        sensor_width = infer_sensor_width(camera_model)
        
        # Calculate GSD for median altitude
        gsd_median = calculate_gsd(avg_altitude, focal_mm, sensor_width, img_width)
        gsd_missing = 0
    else:
        gsd_median = 0.0
        gsd_missing = 1

    # 2. OVERLAP PROXY - degree of frame overlap (0=none, 1=high overlap)
    if gps_data_available:
        # Calculate from flight path geometry
        bearings: list[float] = []
        steps_m: list[float] = []
        for idx in range(1, len(points)):
            (lat1, lon1), (lat2, lon2) = points[idx - 1], points[idx]
            bearings.append(_bearing_deg(lat1, lon1, lat2, lon2))
            steps_m.append(_haversine_m(lat1, lon1, lat2, lon2))
        
        avg_step = mean(steps_m) if steps_m else 0.0
        # Footprint = ground coverage per image (roughly proportional to altitude)
        footprint = max(10.0, 1.2 * avg_altitude)
        overlap_proxy_val = max(0.0, min(1.0, 1.0 - (avg_step / footprint)))
        overlap_missing = 0
    else:
        overlap_proxy_val = 0.5
        overlap_missing = 1

    # 3. COVERAGE SPREAD - geographic extent of survey (0=tight, 1=wide)
    if gps_data_available:
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        lat_span_m = _haversine_m(min(lats), mean(lons), max(lats), mean(lons)) if lats else 0.0
        lon_span_m = _haversine_m(mean(lats), min(lons), mean(lats), max(lons)) if lons else 0.0
        bbox_diag_m = math.sqrt((lat_span_m ** 2) + (lon_span_m ** 2))
        coverage_spread_val = max(0.0, min(1.0, bbox_diag_m / 600.0))
        coverage_missing = 0
    else:
        coverage_spread_val = 0.0
        coverage_missing = 1

    # 4. CAMERA ANGLE BUCKET - numeric encoding from COLMAP poses
    # 0=missing, 1=nadir, 2=oblique, 3=mixed
    angle_samples = _collect_colmap_pitch_angles(ctx.colmap_dir, limit=48)
    if angle_samples:
        # Collapse prior oblique/high_oblique into a single oblique class.
        coarse_buckets: list[str] = []
        for angle in angle_samples:
            raw_bucket = _angle_bucket_from_pitch(angle)
            if raw_bucket == "nadir":
                coarse_buckets.append("nadir")
            elif raw_bucket in {"oblique", "high_oblique"}:
                coarse_buckets.append("oblique")

        if coarse_buckets:
            unique_buckets = set(coarse_buckets)
            if len(unique_buckets) > 1:
                camera_angle_bucket_val = 3  # mixed
            elif "nadir" in unique_buckets:
                camera_angle_bucket_val = 1
            else:
                camera_angle_bucket_val = 2
            angle_missing = 0
        else:
            camera_angle_bucket_val = 0
            angle_missing = 1
    else:
        camera_angle_bucket_val = 0
        angle_missing = 1

    # 5. HEADING CONSISTENCY - regularity of flight path direction
    if gps_data_available and len(points) >= 3:
        bearings: list[float] = []
        for idx in range(1, len(points)):
            (lat1, lon1), (lat2, lon2) = points[idx - 1], points[idx]
            bearings.append(_bearing_deg(lat1, lon1, lat2, lon2))
        
        if len(bearings) >= 2:
            deltas = []
            for i in range(1, len(bearings)):
                d = abs(bearings[i] - bearings[i - 1])
                deltas.append(min(d, 360.0 - d))
            turn_std = pstdev(deltas) if len(deltas) >= 2 else (deltas[0] if deltas else 0.0)
            heading_consistency_val = max(0.0, min(1.0, 1.0 - (turn_std / 90.0)))
        else:
            heading_consistency_val = 0.5
        heading_missing = 0
    else:
        heading_consistency_val = 0.5
        heading_missing = 1

    # ========== PRESET SELECTION LOGIC ==========
    # Simple preset selection based on coverage and heading patterns
    preset = str(base.updates.get("preset_name", "balanced"))
    if gps_data_available and coverage_spread_val >= 0.55 and heading_consistency_val >= 0.65:
        # Likely grid/corridor with good overlap
        preset = "geometry_fast"
    else:
        preset = "balanced"

    updates = apply_preset_updates(ctx.params, preset)

    # ========== BUILD FEATURE DICTIONARY ==========
    # Preserve Mode 1 features and add Mode 2 features
    features["gsd_median"] = gsd_median
    features["gsd_missing"] = gsd_missing
    features["overlap_proxy"] = overlap_proxy_val
    features["overlap_missing"] = overlap_missing
    features["coverage_spread"] = coverage_spread_val
    features["coverage_missing"] = coverage_missing
    features["camera_angle_bucket"] = camera_angle_bucket_val
    features["angle_missing"] = angle_missing
    features["heading_consistency"] = heading_consistency_val
    features["heading_missing"] = heading_missing
    features = keep_only_feature_keys(features, EXIF_ONLY_FEATURE_KEYS | FLIGHT_PLAN_FEATURE_KEYS)

    notes.append("MODE 2 (Flight Plan): 10-feature reduced set with GSD replacement and numeric camera_angle_bucket (0 missing, 1 nadir, 2 oblique, 3 mixed).")
    notes.append("GSD replaces raw altitude; provides direct measure of image resolution at ground level.")
    return PresetResult(mode="exif_plus_flight_plan", updates=updates, features=features, notes=notes)
