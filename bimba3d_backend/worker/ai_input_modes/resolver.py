from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import ModeContext, apply_preset_updates
from .exif_only import build_preset as build_exif_only_preset
from .exif_plus_flight_plan import build_preset as build_exif_plus_flight_plan_preset
from .exif_plus_flight_plan_plus_external import (
    build_preset as build_exif_plus_flight_plan_plus_external_preset,
)
from .continuous_learner import select_continuous
from .contextual_continuous_learner import select_contextual_continuous
from .learner import select_preset

# Import jitter functions for multi-pass learning
try:
    from bimba3d_backend.app.services.context_jitter import apply_context_jitter
except ImportError:
    # Fallback if import path changes
    apply_context_jitter = None

VALID_AI_INPUT_MODES = {
    "exif_only",
    "exif_plus_flight_plan",
    "exif_plus_flight_plan_plus_external",
}

CACHE_VERSION = 1
VALID_PRESET_OVERRIDES = {"conservative", "balanced", "geometry_fast", "appearance_fast"}
VALID_SELECTOR_STRATEGIES = {"preset_bias", "continuous_bandit_linear", "contextual_continuous"}


def _normalize_preset_override(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in VALID_PRESET_OVERRIDES else ""


def _normalize_selector_strategy(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in VALID_SELECTOR_STRATEGIES:
        return token
    return "preset_bias"


def _feature_cache_dir(project_dir: Path) -> Path:
    return project_dir / "outputs" / "ai_input_modes"


def _feature_cache_path(project_dir: Path, mode: str) -> Path:
    return _feature_cache_dir(project_dir) / f"{mode}.json"


def _image_fingerprint(image_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [p for p in Path(image_dir).glob("*") if p.is_file()]
    files.sort()
    for path in files:
        try:
            stat = path.stat()
        except Exception:
            continue
        rel_name = path.name.lower().encode("utf-8", errors="ignore")
        digest.update(rel_name)
        digest.update(str(int(stat.st_size)).encode("utf-8"))
        digest.update(str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))).encode("utf-8"))
    return digest.hexdigest()


def _combined_image_fingerprint(metadata_image_dir: Path, processing_image_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_image_fingerprint(metadata_image_dir).encode("utf-8"))
    digest.update(_image_fingerprint(processing_image_dir).encode("utf-8"))
    return digest.hexdigest()


def _load_feature_cache(project_dir: Path, mode: str, fingerprint: str) -> dict[str, Any] | None:
    cache_path = _feature_cache_path(project_dir, mode)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", 0) or 0) != CACHE_VERSION:
        return None
    if str(payload.get("fingerprint") or "") != fingerprint:
        return None
    return payload


def _save_feature_cache(project_dir: Path, mode: str, fingerprint: str, payload: dict[str, Any]) -> Path:
    cache_dir = _feature_cache_dir(project_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _feature_cache_path(project_dir, mode)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_payload = {
        "version": CACHE_VERSION,
        "mode": mode,
        "fingerprint": fingerprint,
        "features": dict(payload.get("features") or {}),
        "notes": list(payload.get("notes") or []),
        "heuristic_preset": str(payload.get("heuristic_preset") or "balanced"),
    }
    tmp_path.write_text(json.dumps(tmp_payload, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)
    return cache_path


def normalize_ai_input_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in VALID_AI_INPUT_MODES:
        return mode
    return ""


def _count_supported_images(image_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    try:
        return sum(1 for p in Path(image_dir).glob("*") if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return 0


def _build_feature_log_details(mode: str, features: dict[str, Any], image_count: int) -> dict[str, Any]:
    details: dict[str, Any] = {
        "image_count": int(image_count),
    }

    # Emit the complete extracted feature set (including *_missing flags) so
    # telemetry UIs can render one row per parameter with status tags.
    for key in sorted(features.keys()):
        details[key] = features.get(key)

    return details


def _build_initial_params_log(params: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tune_start_step",
        "tune_end_step",
        "trend_scope",
        "feature_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
        "position_lr_init",
        "position_lr_final",
        "densification_interval",
        "densify_grad_threshold",
        "opacity_threshold",
        "lambda_dssim",
    ]
    details: dict[str, Any] = {}
    for key in keys:
        value = params.get(key)
        if value is not None:
            details[key] = value
    return details


def apply_initial_preset(
    params: dict[str, Any],
    *,
    image_dir: Path,
    colmap_dir: Path,
    logger,
) -> dict[str, Any]:
    """Apply optional initial parameter preset from selected AI input mode.

    This function is intentionally additive: it keeps legacy behavior when no
    mode is selected and only updates a small bounded set of initial params.
    """
    mode = normalize_ai_input_mode(params.get("ai_input_mode"))
    selector_strategy = _normalize_selector_strategy(params.get("ai_selector_strategy"))
    params["ai_selector_strategy"] = selector_strategy
    if not mode:
        return {
            "mode": "legacy",
            "applied": False,
            "updates": {},
            "features": {},
            "notes": ["No ai_input_mode selected; using existing behavior."],
            "cache_used": False,
            "selector_strategy": selector_strategy,
        }

    image_dir_path = Path(image_dir)
    project_dir = image_dir_path.resolve().parent

    # Keep metadata extraction tied to original uploads when resized copies exist,
    # because some EXIF/XMP fields may be dropped during resize.
    metadata_image_dir = image_dir_path
    if image_dir_path.name == "images_resized":
        original_dir = project_dir / "images"
        if original_dir.exists() and original_dir.is_dir():
            metadata_image_dir = original_dir

    processing_image_dir = image_dir_path

    ctx = ModeContext(
        metadata_image_dir=metadata_image_dir,
        processing_image_dir=processing_image_dir,
        colmap_dir=Path(colmap_dir),
        params=params,
    )
    fingerprint = _combined_image_fingerprint(metadata_image_dir, processing_image_dir)
    cached = _load_feature_cache(project_dir, mode, fingerprint)
    cache_used = cached is not None

    if cached is not None:
        result_features = dict(cached.get("features") or {})
        result_notes = list(cached.get("notes") or [])
        heuristic_preset = str(cached.get("heuristic_preset") or "balanced")
    else:
        if mode == "exif_only":
            result = build_exif_only_preset(ctx)
        elif mode == "exif_plus_flight_plan":
            result = build_exif_plus_flight_plan_preset(ctx)
        else:
            result = build_exif_plus_flight_plan_plus_external_preset(ctx)

        result_features = dict(result.features)
        result_notes = list(result.notes)
        heuristic_preset = str(result.updates.get("preset_name") or "balanced")
        _save_feature_cache(
            project_dir,
            mode,
            fingerprint,
            {
                "features": result_features,
                "notes": result_notes,
                "heuristic_preset": heuristic_preset,
            },
        )

    if selector_strategy == "contextual_continuous":
        # Apply context jitter if enabled (for multi-pass pipeline learning)
        features_for_selection = result_features
        if params.get("context_jitter_enabled", False) and apply_context_jitter is not None:
            jitter_mode = str(params.get("context_jitter_mode", "uniform")).strip().lower()
            # Support both "uniform" mode and simplified boolean mode
            if jitter_mode in ["uniform", "mild", "gaussian", "true", "1"]:
                actual_mode = jitter_mode if jitter_mode in ["uniform", "mild", "gaussian"] else "uniform"
                features_for_selection = apply_context_jitter(result_features, jitter_mode=actual_mode)
                logger.info(
                    "CONTEXT_JITTER_APPLIED mode=%s jitter_mode=%s original_features=%s jittered_features=%s",
                    mode,
                    actual_mode,
                    json.dumps({k: v for k, v in result_features.items() if not k.endswith("_missing")}, sort_keys=True),
                    json.dumps({k: v for k, v in features_for_selection.items() if not k.endswith("_missing")}, sort_keys=True),
                )

        selection = select_contextual_continuous(
            project_dir=project_dir,
            mode=mode,
            x_features=features_for_selection,
            params=params,
            exploration_mode="thompson",
        )
        selected_updates = dict(selection.get("updates") or {})
        selected_preset = str(selection.get("selected_preset") or "contextual_continuous")
        preset_forced = False
    elif selector_strategy == "continuous_bandit_linear":
        selection = select_continuous(
            project_dir=project_dir,
            mode=mode,
            params=params,
        )
        selected_updates = dict(selection.get("updates") or {})
        selected_preset = str(selection.get("selected_preset") or "continuous_bandit_linear")
        preset_forced = False
    else:
        selection = select_preset(
            project_dir=project_dir,
            mode=mode,
            heuristic_preset=heuristic_preset,
            params=params,
        )
        selected_updates = dict(selection.get("updates") or {})
        selected_preset = str(selection.get("selected_preset") or heuristic_preset)
        forced_preset = _normalize_preset_override(params.get("ai_preset_override"))
        preset_forced = bool(forced_preset)
        if preset_forced:
            # Optional override for controlled exploration experiments; default path remains adaptive.
            selected_preset = forced_preset
            selected_updates = apply_preset_updates(params, selected_preset)

    for key, value in selected_updates.items():
        if key == "preset_name":
            continue
        params[key] = value

    feature_details = _build_feature_log_details(mode, result_features, _count_supported_images(processing_image_dir))
    logger.info(
        "AI_INPUT_MODE_FEATURES mode=%s details=%s",
        mode,
        json.dumps(feature_details, sort_keys=True),
    )
    logger.info(
        "AI_INPUT_MODE_PRESET mode=%s heuristic=%s selected=%s cache_used=%s forced=%s strategy=%s",
        mode,
        heuristic_preset,
        selected_preset,
        str(bool(cache_used)).lower(),
        str(preset_forced).lower(),
        selector_strategy,
    )
    logger.info(
        "AI_INPUT_MODE_INITIAL_PARAMS mode=%s params=%s",
        mode,
        json.dumps(_build_initial_params_log(params), sort_keys=True),
    )

    logger.info(
        "AI input preset applied mode=%s selected_preset=%s cache_used=%s updates=%s features=%s",
        mode,
        selected_preset,
        cache_used,
        selected_updates,
        result_features,
    )

    return {
        "mode": mode,
        "applied": True,
        "updates": selected_updates,
        "features": result_features,
        "notes": result_notes,
        "heuristic_preset": heuristic_preset,
        "selected_preset": selected_preset,
        "preset_forced": preset_forced,
        "selector_strategy": selector_strategy,
        "yhat_scores": dict(selection.get("yhat_scores") or {}),
        "project_dir": str(project_dir),
        "cache_used": cache_used,
        "cache_path": str(_feature_cache_path(project_dir, mode)),
    }
