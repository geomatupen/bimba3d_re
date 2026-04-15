from __future__ import annotations

from pathlib import Path
from statistics import median

from PIL import Image

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys
from .exif_plus_flight_plan import (
    FLIGHT_PLAN_FEATURE_KEYS,
    build_preset as build_exif_plus_flight_plan_preset,
)
from .exif_only import EXIF_ONLY_FEATURE_KEYS


EXTERNAL_IMAGE_FEATURE_KEYS: set[str] = {
    "vegetation_cover_percentage",
    "green_area_missing",
    "vegetation_complexity_score",
    "veg_complexity_missing",
    "terrain_roughness_proxy",
    "roughness_missing",
    "texture_density",
    "texture_missing",
    "blur_motion_risk",
    "blur_missing",
}


def _iter_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files[: min(20, len(files))]


def _image_metrics(path: Path) -> tuple[float, float, float, float, float]:
    # returns: green_cover, veg_complexity, roughness_proxy, texture_density, blur_risk
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((128, 128))
        w, h = rgb.size
        px = rgb.load()
        total = max(1, w * h)

        green_hits = 0
        luma_vals: list[float] = []
        edge_sum = 0.0
        lap_sum = 0.0

        def _luma(r: int, g: int, b: int) -> float:
            return 0.299 * r + 0.587 * g + 0.114 * b

        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                if g > r * 1.05 and g > b * 1.05:
                    green_hits += 1
                luma_vals.append(_luma(r, g, b))

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                c = _luma(*px[x, y])
                gx = _luma(*px[x + 1, y]) - _luma(*px[x - 1, y])
                gy = _luma(*px[x, y + 1]) - _luma(*px[x, y - 1])
                edge_sum += abs(gx) + abs(gy)
                lap = (
                    _luma(*px[x - 1, y])
                    + _luma(*px[x + 1, y])
                    + _luma(*px[x, y - 1])
                    + _luma(*px[x, y + 1])
                    - 4.0 * c
                )
                lap_sum += abs(lap)

        green_cover = float(green_hits) / float(total)
        mean_luma = sum(luma_vals) / float(len(luma_vals)) if luma_vals else 0.0
        var_luma = sum((v - mean_luma) ** 2 for v in luma_vals) / float(len(luma_vals)) if luma_vals else 0.0
        std_luma = var_luma ** 0.5
        texture_density = min(1.0, edge_sum / float(max(1.0, (w - 2) * (h - 2) * 255.0)))
        roughness_proxy = min(1.0, std_luma / 64.0)
        blur_sharpness = min(1.0, lap_sum / float(max(1.0, (w - 2) * (h - 2) * 255.0)))
        blur_risk = 1.0 - blur_sharpness
        # Keep complexity tied to vegetation and texture spread.
        veg_complexity = min(1.0, 0.55 * green_cover + 0.45 * texture_density)

        return green_cover, veg_complexity, roughness_proxy, texture_density, blur_risk


def build_preset(ctx: ModeContext) -> PresetResult:
    base = build_exif_plus_flight_plan_preset(ctx)
    updates = dict(base.updates)
    features = dict(base.features)
    notes = list(base.notes)

    img_files = _iter_images(ctx.image_dir)
    metrics = []
    for path in img_files:
        try:
            metrics.append(_image_metrics(path))
        except Exception:
            continue

    if metrics:
        green_cover = float(median([m[0] for m in metrics]))
        veg_complexity = float(median([m[1] for m in metrics]))
        roughness_proxy = float(median([m[2] for m in metrics]))
        texture_density = float(median([m[3] for m in metrics]))
        blur_risk = float(median([m[4] for m in metrics]))
        green_area_missing = 0
        veg_complexity_missing = 0
        roughness_missing = 0
        texture_missing = 0
        blur_missing = 0
    else:
        green_cover = 0.0
        veg_complexity = 0.5
        roughness_proxy = 0.5
        texture_density = 0.5
        blur_risk = 0.5
        green_area_missing = 1
        veg_complexity_missing = 1
        roughness_missing = 1
        texture_missing = 1
        blur_missing = 1

    # Select preset from richer scene understanding.
    if blur_risk >= 0.55 or (green_cover >= 0.60 and veg_complexity >= 0.60):
        preset = "conservative"
    elif roughness_proxy <= 0.35 and texture_density >= 0.60:
        preset = "geometry_fast"
    elif texture_density >= 0.68 and blur_risk <= 0.35:
        preset = "appearance_fast"
    else:
        preset = "balanced"

    updates = apply_preset_updates(ctx.params, preset)

    features["vegetation_cover_percentage"] = green_cover
    features["green_area_missing"] = green_area_missing
    features["vegetation_complexity_score"] = veg_complexity
    features["veg_complexity_missing"] = veg_complexity_missing
    features["terrain_roughness_proxy"] = roughness_proxy
    features["roughness_missing"] = roughness_missing
    features["texture_density"] = texture_density
    features["texture_missing"] = texture_missing
    features["blur_motion_risk"] = blur_risk
    features["blur_missing"] = blur_missing
    features = keep_only_feature_keys(
        features,
        EXIF_ONLY_FEATURE_KEYS | FLIGHT_PLAN_FEATURE_KEYS | EXTERNAL_IMAGE_FEATURE_KEYS,
    )

    notes.append("Added external image-derived features (vegetation, roughness, texture, blur) without manual input.")
    return PresetResult(mode="exif_plus_flight_plan_plus_external", updates=updates, features=features, notes=notes)
