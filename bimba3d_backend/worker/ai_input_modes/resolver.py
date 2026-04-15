from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import ModeContext
from .exif_only import build_preset as build_exif_only_preset
from .exif_plus_flight_plan import build_preset as build_exif_plus_flight_plan_preset
from .exif_plus_flight_plan_plus_external import (
    build_preset as build_exif_plus_flight_plan_plus_external_preset,
)
from .learner import select_preset

VALID_AI_INPUT_MODES = {
    "exif_only",
    "exif_plus_flight_plan",
    "exif_plus_flight_plan_plus_external",
}

CACHE_VERSION = 1


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
    if not mode:
        return {
            "mode": "legacy",
            "applied": False,
            "updates": {},
            "features": {},
            "notes": ["No ai_input_mode selected; using existing behavior."],
            "cache_used": False,
        }

    ctx = ModeContext(image_dir=Path(image_dir), colmap_dir=Path(colmap_dir), params=params)
    project_dir = Path(image_dir).resolve().parent
    fingerprint = _image_fingerprint(Path(image_dir))
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

    selection = select_preset(
        project_dir=project_dir,
        mode=mode,
        heuristic_preset=heuristic_preset,
        params=params,
    )
    selected_updates = dict(selection.get("updates") or {})
    selected_preset = str(selection.get("selected_preset") or heuristic_preset)

    for key, value in selected_updates.items():
        if key == "preset_name":
            continue
        params[key] = value

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
        "yhat_scores": dict(selection.get("yhat_scores") or {}),
        "project_dir": str(project_dir),
        "cache_used": cache_used,
        "cache_path": str(_feature_cache_path(project_dir, mode)),
    }
