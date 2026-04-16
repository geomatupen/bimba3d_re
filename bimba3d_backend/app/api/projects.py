import threading
import logging
import shutil
import json
import time
import random
import re
import os
import stat
import struct
import ast
import uuid
from typing import Optional, Any
from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import FileResponse
from bimba3d_backend.app.config import DATA_DIR, ALLOWED_IMAGE_EXTENSIONS
from bimba3d_backend.app.models.project import (
    ProjectResponse,
    StorageRootResponse,
    StatusResponse,
    ProcessParams,
    ProjectListItem,
    CreateProjectRequest,
    UpdateProjectRequest,
    EvaluationMetrics,
    ComparisonRequest,
    ComparisonStatus,
    SparseEditRequest,
    SparseMergeRequest,
    RenameRunRequest,
    CreateRunRequest,
    UpdateRunConfigRequest,
    UpdateSharedConfigRequest,
    ElevateModelRequest,
    RenameModelRequest,
    ContinueBatchRequest,
)
from bimba3d_backend.app.services import status, storage, colmap, gsplat, files, sparse_edit, pointsbin
from bimba3d_backend.app.services import model_registry
from bimba3d_backend.app.services.worker_mode import normalize_worker_mode, resolve_worker_mode
from bimba3d_backend.worker import pipeline

COLMAP_TO_OPENGL = (1.0, -1.0, -1.0)
BEST_SPARSE_META = ".best_sparse_selection.json"
SPARSE_IMAGE_MEMBERSHIP_META = ".sparse_image_membership.json"
SHARED_CONFIG_FILE = "shared_config.json"
BATCH_LINEAGE_FILE = ".batch_lineage_latest.json"
RUN_CONFIG_HISTORY_KEEP = 5
GSPLAT_SNAPSHOT_RE = re.compile(
    r"\[GSPLAT SNAPSHOT\]\s+step=(?P<step>\d+)/(?:\s*)?(?P<max_steps>\d+).*?loss=(?P<loss>[0-9.eE+-]+).*?elapsed=(?P<elapsed>[0-9.eE+-]+)s.*?eta=(?P<eta>[^\s]+).*?speed=(?P<speed>[^\s]+)",
    re.IGNORECASE,
)
GSPLAT_STEP_RE = re.compile(
    r"Training step\s+(?P<step>\d+)/(?:\s*)?(?P<max_steps>\d+)\s+\(loss:\s*(?P<loss>[0-9.eE+-]+)\)",
    re.IGNORECASE,
)
VAL_STEP_RE = re.compile(r"^val_step(?P<step>\d+)\.json$", re.IGNORECASE)
BEST_SPLAT_UPDATE_RE = re.compile(
    r"BEST_SPLAT_UPDATE\s+step=(?P<step>\d+).*?loss=(?P<loss>[0-9.eE+-]+).*?improvement=(?P<improvement>[0-9.eE+-]+|n/a)",
    re.IGNORECASE,
)
EARLY_STOP_TRIGGER_RE = re.compile(
    r"EARLY_STOP_TRIGGER\s+step=(?P<step>\d+).*?rel_improve=(?P<rel>[0-9.eE+-]+).*?volatility=(?P<vol>[0-9.eE+-]+)",
    re.IGNORECASE,
)
CORE_AI_DECISION_RE = re.compile(
    r"Core-AI adaptive decision\s+step=(?P<step>\d+)\s+action=(?P<action>[^\s]+)",
    re.IGNORECASE,
)
CORE_AI_REASON_RE = re.compile(r"\breason=(?P<reason>[^\s]+)", re.IGNORECASE)
CORE_AI_LOSS_RE = re.compile(r"\bloss=(?P<loss>[0-9.eE+-]+|n/a)", re.IGNORECASE)
CORE_AI_REL_IMPROVE_RE = re.compile(r"\brel_improve=(?P<rel>[0-9.eE+-]+|n/a)", re.IGNORECASE)
CORE_AI_REWARD_PREV_RE = re.compile(r"\breward_prev=(?P<reward>[0-9.eE+-]+|n/a)", re.IGNORECASE)
RULE_UPDATE_RE = re.compile(
    r"Modified rule update applied at step\s+(?P<step>\d+)",
    re.IGNORECASE,
)
AI_INPUT_MODE_PRESET_RE = re.compile(
    r"AI_INPUT_MODE_PRESET\s+mode=(?P<mode>[^\s]+)\s+heuristic=(?P<heuristic>[^\s]+)\s+selected=(?P<selected>[^\s]+)\s+cache_used=(?P<cache>[^\s]+)",
    re.IGNORECASE,
)
AI_INPUT_MODE_FEATURES_RE = re.compile(
    r"AI_INPUT_MODE_FEATURES\s+mode=(?P<mode>[^\s]+)\s+details=(?P<details>\{.*\})",
    re.IGNORECASE,
)
AI_INPUT_MODE_LEARN_RE = re.compile(
    r"AI_INPUT_MODE_LEARN\s+mode=(?P<mode>[^\s]+)\s+preset=(?P<preset>[^\s]+)\s+s_best=(?P<s_best>[0-9.eE+-]+)\s+s_end=(?P<s_end>[0-9.eE+-]+)\s+s_run=(?P<s_run>[0-9.eE+-]+)\s+reward=(?P<reward>[0-9.eE+-]+)",
    re.IGNORECASE,
)
AI_INPUT_MODE_INITIAL_PARAMS_RE = re.compile(
    r"AI_INPUT_MODE_INITIAL_PARAMS\s+mode=(?P<mode>[^\s]+)\s+params=(?P<params>\{.*\})",
    re.IGNORECASE,
)
AI_INPUT_MODE_REWARD_OUTCOME_RE = re.compile(
    r"AI_INPUT_MODE_REWARD_OUTCOME\s+mode=(?P<mode>[^\s]+)\s+preset=(?P<preset>[^\s]+)\s+reward=(?P<reward>[0-9.eE+-]+)\s+rewarded=(?P<rewarded>true|false)",
    re.IGNORECASE,
)

WARMUP_PHASE_PLAN = [
    {
        "name": "phase_a_explore",
        "runs": 16,
        "preset_sequence": ["balanced", "conservative", "geometry_fast", "appearance_fast"],
        "jitter_mode": "random",
        "jitter_min": 0.5,
        "jitter_max": 1.5,
    },
    {
        "name": "phase_b_stability",
        "runs": 8,
        "jitter_mode": "random",
        "jitter_min": 0.75,
        "jitter_max": 1.25,
    },
    {
        "name": "phase_c_adaptive",
        "runs": 12,
        "jitter_mode": "random",
        "jitter_min": 0.9,
        "jitter_max": 1.1,
    },
]


def _resolve_warmup_phase_plan(total_runs: int) -> list[dict[str, Any]]:
    """Scale the 3-phase warmup template to a requested total run count."""
    target = max(3, int(total_runs or 0))
    weights = [int(phase.get("runs") or 1) for phase in WARMUP_PHASE_PLAN]
    weight_sum = sum(weights) or len(weights)

    raw = [target * (w / weight_sum) for w in weights]
    alloc = [max(1, int(v)) for v in raw]
    allocated = sum(alloc)

    # Distribute remaining runs to phases with largest fractional remainder.
    if allocated < target:
        remainder_order = sorted(
            range(len(raw)),
            key=lambda i: (raw[i] - int(raw[i])),
            reverse=True,
        )
        idx = 0
        while allocated < target:
            alloc[remainder_order[idx % len(remainder_order)]] += 1
            allocated += 1
            idx += 1

    # Remove excess runs from largest phases first while keeping >=1 each.
    if allocated > target:
        shrink_order = sorted(range(len(alloc)), key=lambda i: alloc[i], reverse=True)
        idx = 0
        while allocated > target and any(v > 1 for v in alloc):
            candidate = shrink_order[idx % len(shrink_order)]
            if alloc[candidate] > 1:
                alloc[candidate] -= 1
                allocated -= 1
            idx += 1

    plan: list[dict[str, Any]] = []
    for i, phase in enumerate(WARMUP_PHASE_PLAN):
        clone = dict(phase)
        clone["runs"] = int(alloc[i])
        plan.append(clone)
    return plan


def _get_project_shared_config_path(project_dir: Path) -> Path:
    return project_dir / SHARED_CONFIG_FILE


def _extract_shared_config_from_params(params: dict | None) -> dict:
    data = params if isinstance(params, dict) else {}
    shared: dict = {}

    if "images_resize_enabled" in data:
        shared["images_resize_enabled"] = bool(data.get("images_resize_enabled"))

    image_size = data.get("images_max_size")
    if isinstance(image_size, (int, float)):
        shared["images_max_size"] = int(image_size)

    colmap_in = data.get("colmap")
    if isinstance(colmap_in, dict):
        # Store full COLMAP object so base-owned shared behavior remains explicit.
        shared["colmap"] = json.loads(json.dumps(colmap_in))

    return shared


def _merge_shared_config_into_params(params: dict, shared: dict | None) -> dict:
    merged = dict(params)
    if not isinstance(shared, dict):
        return merged

    if "images_resize_enabled" in shared:
        merged["images_resize_enabled"] = bool(shared.get("images_resize_enabled"))

    if "images_max_size" in shared:
        merged["images_max_size"] = shared.get("images_max_size")

    shared_colmap = shared.get("colmap")
    if isinstance(shared_colmap, dict):
        current_colmap = merged.get("colmap") if isinstance(merged.get("colmap"), dict) else {}
        merged["colmap"] = {
            **current_colmap,
            **json.loads(json.dumps(shared_colmap)),
        }

    return merged


def _normalize_shared_doc(raw: dict | None, base_run_id: str | None = None) -> dict:
    doc = raw if isinstance(raw, dict) else {}
    shared = doc.get("shared") if isinstance(doc.get("shared"), dict) else {}
    version = doc.get("version")
    if not isinstance(version, int) or version < 1:
        version = 1

    active_shared = doc.get("active_shared") if isinstance(doc.get("active_shared"), dict) else None

    normalized = {
        "version": version,
        "base_run_id": doc.get("base_run_id") if isinstance(doc.get("base_run_id"), str) else base_run_id,
        "updated_at": doc.get("updated_at") if isinstance(doc.get("updated_at"), str) else None,
        "active_sparse_version": doc.get("active_sparse_version") if isinstance(doc.get("active_sparse_version"), int) else None,
        "active_sparse_updated_at": doc.get("active_sparse_updated_at") if isinstance(doc.get("active_sparse_updated_at"), str) else None,
        "active_shared": active_shared,
        "shared": shared,
    }
    if not normalized["base_run_id"] and base_run_id:
        normalized["base_run_id"] = base_run_id
    return normalized


def _read_project_shared_config(project_dir: Path, base_run_id: str | None = None) -> dict:
    path = _get_project_shared_config_path(project_dir)
    raw = _read_json_if_exists(path)
    normalized = _normalize_shared_doc(raw, base_run_id=base_run_id)

    if isinstance(normalized.get("shared"), dict) and normalized.get("shared"):
        return normalized

    if base_run_id:
        base_run_cfg = _read_json_if_exists(project_dir / "runs" / base_run_id / "run_config.json")
        if isinstance(base_run_cfg, dict):
            resolved = base_run_cfg.get("resolved_params") if isinstance(base_run_cfg.get("resolved_params"), dict) else {}
            inferred_shared = _extract_shared_config_from_params(resolved)
            if inferred_shared:
                normalized["shared"] = inferred_shared
                if not isinstance(normalized.get("active_shared"), dict):
                    normalized["active_shared"] = json.loads(json.dumps(inferred_shared))
                if not isinstance(normalized.get("active_sparse_version"), int):
                    normalized["active_sparse_version"] = int(normalized.get("version") or 1)

    return normalized


def _write_project_shared_config(project_dir: Path, doc: dict) -> None:
    path = _get_project_shared_config_path(project_dir)
    normalized = _normalize_shared_doc(doc)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
    tmp_path.replace(path)


def _read_json_if_exists(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse JSON %s: %s", path, exc)
        return None


def _read_run_analytics(run_dir: Path | None) -> dict[str, Any] | None:
    if not isinstance(run_dir, Path):
        return None
    payload = _read_json_if_exists(run_dir / "analytics" / "run_analytics_v1.json")
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def _best_metric(rows: list[dict[str, Any]], key: str, prefer: str) -> tuple[int | None, float | None]:
    best_step = None
    best_value = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        step_raw = row.get("step")
        value_raw = row.get(key)
        if not isinstance(step_raw, (int, float)):
            continue
        if not isinstance(value_raw, (int, float)):
            continue
        step = int(step_raw)
        value = float(value_raw)
        if best_value is None:
            best_value = value
            best_step = step
            continue
        if prefer == "max" and value > best_value:
            best_value = value
            best_step = step
        if prefer == "min" and value < best_value:
            best_value = value
            best_step = step
    return best_step, best_value


def _ensure_run_analytics(
    *,
    run_dir: Path,
    run_config: dict[str, Any] | None,
    ai_insights: dict[str, Any] | None,
) -> dict[str, Any] | None:
    existing = _read_run_analytics(run_dir)
    if isinstance(existing, dict):
        return existing

    learning_payload = _read_json_if_exists(run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json")
    if not isinstance(learning_payload, dict):
        learning_payload = None

    loss_summary = _extract_loss_summary_from_log(run_dir / "processing.log")
    eval_rows = _extract_eval_rows(run_dir / "outputs" / "engines" / "gsplat" / "stats", eval_limit=200000)
    if not eval_rows:
        eval_history = _read_json_if_exists(run_dir / "outputs" / "engines" / "gsplat" / "eval_history.json")
        if isinstance(eval_history, list):
            rows: list[dict[str, Any]] = []
            for item in eval_history:
                if not isinstance(item, dict):
                    continue
                step = item.get("step")
                if not isinstance(step, (int, float)):
                    continue
                rows.append(
                    {
                        "step": int(step),
                        "psnr": item.get("convergence_speed"),
                        "lpips": item.get("lpips_mean"),
                        "ssim": item.get("sharpness_mean"),
                        "num_gaussians": item.get("num_gaussians"),
                    }
                )
            eval_rows = sorted(rows, key=lambda r: int(r.get("step", 0)))

    best_psnr_step, best_psnr = _best_metric(eval_rows, "psnr", "max")
    best_ssim_step, best_ssim = _best_metric(eval_rows, "ssim", "max")
    best_lpips_step, best_lpips = _best_metric(eval_rows, "lpips", "min")
    final_eval = eval_rows[-1] if eval_rows else {}

    run_id = run_dir.name
    resolved_cfg = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
    requested_cfg = run_config.get("requested_params") if isinstance(run_config, dict) and isinstance(run_config.get("requested_params"), dict) else {}
    run_name = (
        (run_config.get("run_name") if isinstance(run_config, dict) else None)
        or (run_config.get("name") if isinstance(run_config, dict) else None)
        or requested_cfg.get("run_name")
        or run_id
    )

    payload = {
        "schema": "run_analytics_v1",
        "version": 1,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "project_id": run_dir.parent.parent.name,
        "run_id": run_id,
        "run_name": run_name,
        "engine": str(resolved_cfg.get("engine") or requested_cfg.get("engine") or "gsplat"),
        "mode": str(resolved_cfg.get("mode") or requested_cfg.get("mode") or "baseline"),
        "metrics": {
            "best_loss_step": loss_summary.get("best_loss_step"),
            "best_loss": loss_summary.get("best_loss"),
            "final_loss_step": loss_summary.get("final_loss_step"),
            "final_loss": loss_summary.get("final_loss"),
            "best_psnr_step": best_psnr_step,
            "best_psnr": best_psnr,
            "final_psnr_step": final_eval.get("step") if isinstance(final_eval, dict) else None,
            "final_psnr": final_eval.get("psnr") if isinstance(final_eval, dict) else None,
            "best_ssim_step": best_ssim_step,
            "best_ssim": best_ssim,
            "final_ssim_step": final_eval.get("step") if isinstance(final_eval, dict) else None,
            "final_ssim": final_eval.get("ssim") if isinstance(final_eval, dict) else None,
            "best_lpips_step": best_lpips_step,
            "best_lpips": best_lpips,
            "final_lpips_step": final_eval.get("step") if isinstance(final_eval, dict) else None,
            "final_lpips": final_eval.get("lpips") if isinstance(final_eval, dict) else None,
        },
        "ai": {
            "input_mode_learning": learning_payload,
            "input_mode_insights": ai_insights if isinstance(ai_insights, dict) else None,
            "controller": None,
        },
    }
    try:
        _write_json_atomic(run_dir / "analytics" / "run_analytics_v1.json", payload)
    except Exception as exc:
        logger.warning("Failed to backfill run analytics for %s: %s", run_dir, exc)
    return payload


def _get_batch_lineage_path(project_dir: Path) -> Path:
    return project_dir / BATCH_LINEAGE_FILE


def _read_batch_lineage(project_dir: Path) -> dict | None:
    payload = _read_json_if_exists(_get_batch_lineage_path(project_dir))
    return payload if isinstance(payload, dict) else None


def _write_batch_lineage(project_dir: Path, payload: dict) -> None:
    target = _get_batch_lineage_path(project_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(target)


def _prune_run_config_history(run_configs_dir: Path, keep: int = RUN_CONFIG_HISTORY_KEEP) -> None:
    """Keep only the most recent versioned run_config snapshots."""
    try:
        files = sorted(run_configs_dir.glob("run_config_*.json"), key=lambda p: p.name, reverse=True)
        if len(files) <= keep:
            return
        for stale in files[keep:]:
            try:
                stale.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to prune stale run config snapshot %s: %s", stale, exc)
    except Exception as exc:
        logger.warning("Failed to prune run config history in %s: %s", run_configs_dir, exc)


def _close_project_log_handlers(project_dir: Path) -> None:
    """Best-effort close of file handlers writing under the project directory."""
    project_dir_resolved = project_dir.resolve()

    def _release_handlers(logger_obj: logging.Logger) -> None:
        for handler in list(logger_obj.handlers):
            base_filename = getattr(handler, "baseFilename", None)
            if not isinstance(base_filename, str):
                continue
            try:
                file_path = Path(base_filename).resolve()
            except Exception:
                continue
            if file_path == project_dir_resolved or project_dir_resolved in file_path.parents:
                try:
                    logger_obj.removeHandler(handler)
                except Exception:
                    pass
                try:
                    handler.flush()
                except Exception:
                    pass
                try:
                    handler.close()
                except Exception:
                    pass

    # Root logger
    _release_handlers(logging.getLogger())

    # Known named loggers
    for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            _release_handlers(logger_obj)


def _remove_readonly_then_retry(func, path, exc_info):
    """shutil.rmtree onerror callback: clear readonly and retry path removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)


def _is_windows_junction(path: Path) -> bool:
    checker = getattr(os.path, "isjunction", None)
    if callable(checker):
        try:
            return bool(checker(str(path)))
        except Exception:
            return False
    return False


def _delete_path_strict(path: Path) -> None:
    """Delete directory/symlink/junction at path if it exists."""
    if not path.exists() and not path.is_symlink():
        return

    if path.is_file():
        path.unlink(missing_ok=True)
        return

    if path.is_symlink():
        path.unlink(missing_ok=True)
        return

    if _is_windows_junction(path):
        os.rmdir(path)
        return

    shutil.rmtree(path, onerror=_remove_readonly_then_retry)


def _clear_restart_artifacts(
    project_dir: Path,
    run_dir: Path,
    *,
    stage: str,
    clear_project_level: bool,
) -> None:
    """Clear generated artifacts so restart begins from a clean state.

    Cleanup is stage-aware:
    - colmap_only: clear sparse artifacts only
    - train_only: clear engine/training artifacts only
    - full: clear both

    Project-level outputs are touched only when restarting the base session.
    """
    clear_colmap = stage in {"full", "colmap_only"}
    clear_training = stage in {"full", "train_only"}

    targets = [
        run_dir / "adaptive_ai",
        run_dir / "processing.log",
        run_dir / "resume_state.json",
    ]

    if clear_colmap:
        targets.append(run_dir / "outputs" / "sparse")
    if clear_training:
        targets.append(run_dir / "outputs" / "engines")

    if clear_project_level:
        if clear_colmap:
            targets.append(project_dir / "outputs" / "sparse")
        if clear_training:
            targets.append(project_dir / "outputs" / "engines")

    for target in targets:
        try:
            _delete_path_strict(target)
        except Exception as exc:
            logger.warning("Failed to clear restart artifact %s: %s", target, exc)


def _normalize_jitter_mode(raw_mode: Any) -> str:
    candidate = str(raw_mode or "fixed").strip().lower()
    return candidate if candidate in {"fixed", "random"} else "fixed"


def _parse_jitter_value(raw_value: Any, default_value: float) -> float:
    try:
        value = float(raw_value)
    except Exception:
        value = float(default_value)
    return max(1e-6, value)


def _resolve_jitter_settings(params: dict) -> tuple[str, float, float, float]:
    mode = _normalize_jitter_mode(params.get("run_jitter_mode"))
    factor = _parse_jitter_value(params.get("run_jitter_factor"), 1.0)

    default_min = min(1.0, factor)
    default_max = max(1.0, factor)
    jitter_min = _parse_jitter_value(params.get("run_jitter_min"), default_min)
    jitter_max = _parse_jitter_value(params.get("run_jitter_max"), default_max)
    if jitter_min > jitter_max:
        jitter_min, jitter_max = jitter_max, jitter_min

    return mode, factor, jitter_min, jitter_max


def _apply_run_jitter(
    params: dict,
    run_index: int,
    jitter_factor: float,
    *,
    jitter_mode: str = "fixed",
    jitter_min: float = 1.0,
    jitter_max: float = 1.0,
) -> dict:
    """Apply per-run multiplicative jitter to learnable optimization parameters.

    Run index is zero-based; run 1 (index 0) is intentionally left unchanged.
    """
    out = dict(params)

    idx = int(run_index)
    if idx <= 0:
        out["run_jitter_multiplier"] = 1.0
        return out

    if _normalize_jitter_mode(jitter_mode) == "random":
        lo = _parse_jitter_value(jitter_min, 1.0)
        hi = _parse_jitter_value(jitter_max, 1.0)
        if lo > hi:
            lo, hi = hi, lo
        multiplier = random.uniform(lo, hi)
    else:
        factor = _parse_jitter_value(jitter_factor, 1.0)
        if abs(factor - 1.0) < 1e-9:
            out["run_jitter_multiplier"] = 1.0
            return out
        multiplier = float(factor) ** idx

    lr_defaults = {
        "feature_lr": 2.5e-3,
        "opacity_lr": 5.0e-2,
        "scaling_lr": 5.0e-3,
        "rotation_lr": 1.0e-3,
        "position_lr_init": 1.6e-4,
        "position_lr_final": 1.6e-6,
    }

    # Keep jittered values in the same safety envelope used by the selector presets.
    bounded_float_defaults = {
        "densify_grad_threshold": (2.0e-4, 5.0e-5, 5.0e-4),
        "opacity_threshold": (0.005, 0.001, 0.02),
        "lambda_dssim": (0.2, 0.05, 0.5),
        "tune_min_improvement": (0.005, 0.001, 0.02),
    }

    bounded_int_defaults = {
        "tune_interval": (100, 50, 400),
    }

    def _clamp_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _clamp_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

    for key, default_val in lr_defaults.items():
        base_val = out.get(key, default_val)
        if isinstance(base_val, (int, float)):
            out[key] = float(base_val) * multiplier

    for key, (default_val, low, high) in bounded_float_defaults.items():
        base_val = out.get(key, default_val)
        if isinstance(base_val, (int, float)):
            out[key] = _clamp_float(float(base_val) * multiplier, low, high)

    for key, (default_val, low, high) in bounded_int_defaults.items():
        base_val = out.get(key, default_val)
        if isinstance(base_val, (int, float)):
            out[key] = _clamp_int(round(float(base_val) * multiplier), low, high)

    out["run_jitter_multiplier"] = float(multiplier)
    return out


def _wait_for_run_completion(project_id: str, run_id: str, timeout_seconds: int = 0) -> dict:
    """Wait until the targeted run reaches a terminal project status."""
    started_at = time.time()
    terminal_states = {"completed", "done", "failed", "stopped"}
    terminal_seen_at: float | None = None
    teardown_grace_seconds = 90
    while True:
        current = status.get_status(project_id)
        current_run = str(current.get("current_run_id") or "")
        current_state = str(current.get("status") or "")
        if current_run == run_id and current_state in terminal_states:
            if terminal_seen_at is None:
                terminal_seen_at = time.time()

            worker_mode = str(current.get("worker_mode") or "local").strip().lower()
            try:
                if worker_mode == "docker":
                    worker_active = bool(colmap.get_project_worker_container_ids(project_id))
                else:
                    worker_active = bool(pipeline.is_local_project_active(project_id))
            except Exception:
                worker_active = False

            if not worker_active:
                return current

            if (time.time() - terminal_seen_at) >= teardown_grace_seconds:
                logger.warning(
                    "Proceeding after terminal status for %s/%s despite active-worker check still true.",
                    project_id,
                    run_id,
                )
                return current
        if timeout_seconds > 0 and (time.time() - started_at) >= timeout_seconds:
            return current
        time.sleep(2)


def _tail_text_lines(path: Path, max_lines: int) -> list[str]:
    if not path.exists() or max_lines <= 0:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=max_lines)]
    except Exception:
        return []


def _read_text_lines(path: Path, max_lines: int, from_start: bool = False) -> list[str]:
    """Read lines from text file. If from_start=True, read first N lines; else read last N lines."""
    if not path.exists() or max_lines <= 0:
        return []
    try:
        if from_start:
            # Read from the beginning of file
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = []
                for i, line in enumerate(handle):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip("\n"))
                return lines
        else:
            # Read from the end of file (original tail behavior)
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return [line.rstrip("\n") for line in deque(handle, maxlen=max_lines)]
    except Exception:
        return []


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"", "n/a", "na", "none", "null"}:
        return None
    try:
        return float(token)
    except Exception:
        return None


def _extract_ai_run_insights(run_dir: Path | None, run_config: dict | None) -> dict[str, Any] | None:
    if not isinstance(run_config, dict):
        return None

    resolved_cfg = run_config.get("resolved_params") if isinstance(run_config.get("resolved_params"), dict) else {}
    requested_cfg = run_config.get("requested_params") if isinstance(run_config.get("requested_params"), dict) else {}

    ai_mode = str(resolved_cfg.get("ai_input_mode") or requested_cfg.get("ai_input_mode") or "").strip().lower()
    if not ai_mode:
        return None

    baseline_session_id = str(
        resolved_cfg.get("baseline_session_id")
        or requested_cfg.get("baseline_session_id")
        or ""
    ).strip() or None

    initial_param_keys = [
        "tune_start_step",
        "tune_end_step",
        "tune_interval",
        "tune_min_improvement",
        "run_jitter_mode",
        "run_jitter_factor",
        "run_jitter_min",
        "run_jitter_max",
        "run_jitter_multiplier",
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

    initial_params: dict[str, Any] = {}
    for key in initial_param_keys:
        if key in resolved_cfg and resolved_cfg.get(key) is not None:
            initial_params[key] = resolved_cfg.get(key)
        elif key in requested_cfg and requested_cfg.get(key) is not None:
            initial_params[key] = requested_cfg.get(key)

    features_details: dict[str, Any] = {}
    features_from_log = False
    initial_params_from_log: dict[str, Any] = {}
    selected_preset: str | None = None
    heuristic_preset: str | None = None
    cache_used: bool | None = None
    reward_value: float | None = None
    reward_positive: bool | None = None
    reward_mode: str | None = None
    reward_preset: str | None = None
    learn_snapshot: dict[str, Any] = {}

    cache_features: dict[str, Any] = {}

    if isinstance(run_dir, Path):
        log_path = run_dir / "processing.log"
        lines = _read_text_lines(log_path, max_lines=5000)

        for line in lines:
            features_match = AI_INPUT_MODE_FEATURES_RE.search(line)
            if features_match:
                details_raw = str(features_match.group("details") or "{}").strip()
                try:
                    parsed = json.loads(details_raw)
                    if isinstance(parsed, dict):
                        features_details = parsed
                        features_from_log = True
                except Exception:
                    pass

            preset_match = AI_INPUT_MODE_PRESET_RE.search(line)
            if preset_match:
                selected_preset = str(preset_match.group("selected") or "").strip() or selected_preset
                heuristic_preset = str(preset_match.group("heuristic") or "").strip() or heuristic_preset
                cache_token = str(preset_match.group("cache") or "").strip().lower()
                if cache_token in {"true", "false"}:
                    cache_used = cache_token == "true"

            init_params_match = AI_INPUT_MODE_INITIAL_PARAMS_RE.search(line)
            if init_params_match:
                params_raw = str(init_params_match.group("params") or "{}").strip()
                try:
                    parsed = json.loads(params_raw)
                    if isinstance(parsed, dict):
                        initial_params_from_log = parsed
                except Exception:
                    pass

            learn_match = AI_INPUT_MODE_LEARN_RE.search(line)
            if learn_match:
                reward_value = _parse_optional_float(learn_match.group("reward"))
                learn_snapshot = {
                    "s_best": _parse_optional_float(learn_match.group("s_best")),
                    "s_end": _parse_optional_float(learn_match.group("s_end")),
                    "s_run": _parse_optional_float(learn_match.group("s_run")),
                }
                reward_mode = str(learn_match.group("mode") or "").strip() or reward_mode
                reward_preset = str(learn_match.group("preset") or "").strip() or reward_preset
                if reward_value is not None:
                    reward_positive = reward_value > 0

            reward_outcome_match = AI_INPUT_MODE_REWARD_OUTCOME_RE.search(line)
            if reward_outcome_match:
                reward_value = _parse_optional_float(reward_outcome_match.group("reward"))
                rewarded_token = str(reward_outcome_match.group("rewarded") or "").strip().lower()
                if rewarded_token in {"true", "false"}:
                    reward_positive = rewarded_token == "true"
                reward_mode = str(reward_outcome_match.group("mode") or "").strip() or reward_mode
                reward_preset = str(reward_outcome_match.group("preset") or "").strip() or reward_preset

        cache_path = run_dir.parent.parent / "outputs" / "ai_input_modes" / f"{ai_mode}.json"
        cache_payload = _read_json_if_exists(cache_path)
        if isinstance(cache_payload, dict):
            raw_features = cache_payload.get("features")
            if isinstance(raw_features, dict):
                cache_features = raw_features

    if isinstance(initial_params_from_log, dict) and initial_params_from_log:
        initial_params.update(initial_params_from_log)

    if not features_details and cache_features:
        features_details = dict(cache_features)

    missing_flags = {
        key: value
        for key, value in features_details.items()
        if isinstance(key, str) and key.endswith("_missing")
    }

    reward_label = "unknown"
    if reward_positive is True:
        reward_label = "rewarded"
    elif reward_positive is False:
        reward_label = "penalized_or_neutral"

    return {
        "ai_input_mode": ai_mode,
        "baseline_session_id": baseline_session_id,
        "selected_preset": selected_preset,
        "heuristic_preset": heuristic_preset,
        "cache_used": cache_used,
        "initial_params": initial_params,
        "feature_details": features_details,
        "missing_flags": missing_flags,
        "reward": reward_value,
        "reward_positive": reward_positive,
        "reward_label": reward_label,
        "learn_snapshot": learn_snapshot,
        "reward_mode": reward_mode,
        "reward_preset": reward_preset,
        "feature_source": "log" if features_from_log else ("cache" if features_details else "none"),
    }


def _extract_training_rows(lines: list[str], row_limit: int, from_start: bool = False) -> list[dict[str, Any]]:
    snapshot_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for line in lines:
        if "[GSPLAT SNAPSHOT]" in line:
            match = GSPLAT_SNAPSHOT_RE.search(line)
            if not match:
                continue
            timestamp = line.split(" - ", 1)[0] if " - " in line else None
            try:
                step = int(match.group("step"))
                max_steps = int(match.group("max_steps"))
                loss = float(match.group("loss"))
            except Exception:
                continue
            elapsed_raw = match.group("elapsed")
            speed_raw = match.group("speed")
            eta_raw = match.group("eta")
            snapshot_rows.append(
                {
                    "timestamp": timestamp,
                    "step": step,
                    "max_steps": max_steps,
                    "loss": loss,
                    "elapsed_seconds": float(elapsed_raw) if re.match(r"^[0-9.eE+-]+$", elapsed_raw or "") else None,
                    "eta": None if str(eta_raw).lower() in {"none", "n/a"} else eta_raw,
                    "speed": None if str(speed_raw).lower() in {"none", "n/a"} else speed_raw,
                    "source": "snapshot",
                }
            )
            continue

        step_match = GSPLAT_STEP_RE.search(line)
        if step_match:
            timestamp = line.split(" - ", 1)[0] if " - " in line else None
            try:
                step = int(step_match.group("step"))
                max_steps = int(step_match.group("max_steps"))
                loss = float(step_match.group("loss"))
            except Exception:
                continue
            step_rows.append(
                {
                    "timestamp": timestamp,
                    "step": step,
                    "max_steps": max_steps,
                    "loss": loss,
                    "elapsed_seconds": None,
                    "eta": None,
                    "speed": None,
                    "source": "step",
                }
            )

    # Show log-interval snapshots in the modal. If snapshots are absent, fallback to step rows.
    rows = snapshot_rows if snapshot_rows else step_rows

    # Deduplicate by step while preserving first-seen order from the selected source.
    deduped: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for row in rows:
        step = row.get("step")
        if isinstance(step, int):
            if step in seen_steps:
                continue
            seen_steps.add(step)
        deduped.append(row)

    rows = deduped
    if row_limit > 0 and len(rows) > row_limit:
        rows = rows[:row_limit] if from_start else rows[-row_limit:]
    return rows


def _parse_log_timestamp_to_epoch(raw: Any) -> float | None:
    if not isinstance(raw, str):
        return None
    token = raw.strip()
    if not token:
        return None
    # Common logger format: "YYYY-MM-DD HH:MM:SS,mmm"
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(token, fmt).timestamp()
        except Exception:
            continue
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _build_training_summary(training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not training_rows:
        return {
            "first_step": None,
            "last_step": None,
            "start_timestamp": None,
            "end_timestamp": None,
            "total_elapsed_seconds": None,
            "row_count": 0,
        }

    first_step = None
    last_step = None
    start_timestamp = None
    end_timestamp = None
    first_epoch = None
    last_epoch = None
    max_elapsed = None
    best_loss = None
    best_loss_step = None

    for row in training_rows:
        step = row.get("step")
        if isinstance(step, int):
            if first_step is None or step < first_step:
                first_step = step
            if last_step is None or step > last_step:
                last_step = step

        ts = row.get("timestamp")
        epoch = _parse_log_timestamp_to_epoch(ts)
        if epoch is not None:
            if first_epoch is None or epoch < first_epoch:
                first_epoch = epoch
                start_timestamp = ts
            if last_epoch is None or epoch > last_epoch:
                last_epoch = epoch
                end_timestamp = ts

        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_val = float(elapsed)
            if max_elapsed is None or elapsed_val > max_elapsed:
                max_elapsed = elapsed_val

        loss = row.get("loss")
        if isinstance(loss, (int, float)):
            loss_val = float(loss)
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_loss_step = step if isinstance(step, int) else best_loss_step

    total_elapsed_seconds = None
    if max_elapsed is not None:
        total_elapsed_seconds = max_elapsed
    elif first_epoch is not None and last_epoch is not None and last_epoch >= first_epoch:
        total_elapsed_seconds = float(last_epoch - first_epoch)

    return {
        "first_step": first_step,
        "last_step": last_step,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "total_elapsed_seconds": total_elapsed_seconds,
        "row_count": len(training_rows),
        "best_loss": best_loss,
        "best_loss_step": best_loss_step,
    }


def _extract_eval_rows(stats_dir: Path, eval_limit: int) -> list[dict[str, Any]]:
    if not stats_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, Path]] = []
    for stats_file in stats_dir.glob("val_step*.json"):
        match = VAL_STEP_RE.match(stats_file.name)
        if not match:
            continue
        try:
            step_zero = int(match.group("step"))
        except Exception:
            continue
        candidates.append((step_zero, stats_file))

    for step_zero, stats_file in sorted(candidates, key=lambda item: item[0]):
        payload = _read_json_if_exists(stats_file)
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "step": int(step_zero + 1),
                "psnr": payload.get("psnr"),
                "lpips": payload.get("lpips"),
                "ssim": payload.get("ssim"),
                "num_gaussians": payload.get("num_GS"),
            }
        )

    if eval_limit > 0 and len(rows) > eval_limit:
        rows = rows[-eval_limit:]
    return rows


def _extract_event_rows(lines: list[str], row_limit: int) -> list[dict[str, Any]]:
    def _parse_optional_float(raw: str | None) -> float | None:
        if raw is None:
            return None
        token = str(raw).strip().lower()
        if token in {"", "n/a", "na", "none", "null"}:
            return None
        try:
            return float(token)
        except Exception:
            return None

    rows: list[dict[str, Any]] = []
    for line in lines:
        timestamp = line.split(" - ", 1)[0] if " - " in line else None

        best_match = BEST_SPLAT_UPDATE_RE.search(line)
        if best_match:
            step = int(best_match.group("step"))
            loss = float(best_match.group("loss"))
            improvement_raw = best_match.group("improvement")
            improvement = (
                float(improvement_raw)
                if isinstance(improvement_raw, str) and re.match(r"^[0-9.eE+-]+$", improvement_raw)
                else None
            )
            summary = (
                f"Best splat updated at step {step:,} (loss {loss:.6f})"
                + (f", improvement {improvement:.6f}" if improvement is not None else "")
            )
            rows.append({
                "timestamp": timestamp,
                "type": "best_splat_update",
                "step": step,
                "summary": summary,
            })
            continue

        early_match = EARLY_STOP_TRIGGER_RE.search(line)
        if early_match:
            step = int(early_match.group("step"))
            rel = float(early_match.group("rel"))
            vol = float(early_match.group("vol"))
            rows.append({
                "timestamp": timestamp,
                "type": "early_stop_trigger",
                "step": step,
                "summary": f"Early stop triggered at step {step:,} (rel {rel:.6f}, vol {vol:.6f})",
            })
            continue

        ai_match = CORE_AI_DECISION_RE.search(line)
        if ai_match:
            step = int(ai_match.group("step"))
            action = str(ai_match.group("action") or "keep")
            reason_match = CORE_AI_REASON_RE.search(line)
            reason = str(reason_match.group("reason")) if reason_match else None
            loss_match = CORE_AI_LOSS_RE.search(line)
            loss = _parse_optional_float(loss_match.group("loss") if loss_match else None)
            rel_match = CORE_AI_REL_IMPROVE_RE.search(line)
            rel_improve = _parse_optional_float(rel_match.group("rel") if rel_match else None)
            reward_match = CORE_AI_REWARD_PREV_RE.search(line)
            reward_prev = _parse_optional_float(reward_match.group("reward") if reward_match else None)

            summary = f"Core-AI decision at step {step:,}: {action}"
            if reason:
                summary += f" ({reason})"

            rows.append({
                "timestamp": timestamp,
                "type": "core_ai_decision",
                "step": step,
                "summary": summary,
                "action": action,
                "reason": reason,
                "loss": loss,
                "relative_improvement": rel_improve,
                "reward_from_previous": reward_prev,
            })
            continue

        rule_match = RULE_UPDATE_RE.search(line)
        if rule_match:
            step = int(rule_match.group("step"))
            rows.append({
                "timestamp": timestamp,
                "type": "rule_update",
                "step": step,
                "summary": f"Rule update applied at step {step:,}",
            })
            continue

        preset_match = AI_INPUT_MODE_PRESET_RE.search(line)
        if preset_match:
            mode = str(preset_match.group("mode") or "")
            selected = str(preset_match.group("selected") or "")
            heuristic = str(preset_match.group("heuristic") or "")
            cache_used = str(preset_match.group("cache") or "")
            rows.append(
                {
                    "timestamp": timestamp,
                    "type": "ai_input_mode_preset",
                    "step": None,
                    "summary": (
                        f"AI mode {mode}: selected preset {selected} "
                        f"(heuristic {heuristic}, cache_used={cache_used})"
                    ),
                    "mode": mode,
                    "selected_preset": selected,
                    "heuristic_preset": heuristic,
                    "cache_used": cache_used,
                }
            )
            continue

        features_match = AI_INPUT_MODE_FEATURES_RE.search(line)
        if features_match:
            mode = str(features_match.group("mode") or "")
            details_raw = str(features_match.group("details") or "{}").strip()
            details = {}
            try:
                parsed = json.loads(details_raw)
                if isinstance(parsed, dict):
                    details = parsed
            except Exception:
                details = {}

            missing_bits = []
            for key in sorted(details.keys()):
                if key.endswith("_missing"):
                    missing_bits.append(f"{key}={details.get(key)}")
            missing_text = ", ".join(missing_bits[:6]) if missing_bits else "none"
            rows.append(
                {
                    "timestamp": timestamp,
                    "type": "ai_input_mode_features",
                    "step": None,
                    "summary": f"AI mode {mode}: extracted input features (missing/default flags: {missing_text})",
                    "mode": mode,
                    "details": details,
                }
            )
            continue

        initial_match = AI_INPUT_MODE_INITIAL_PARAMS_RE.search(line)
        if initial_match:
            mode = str(initial_match.group("mode") or "")
            params_raw = str(initial_match.group("params") or "{}").strip()
            params_payload = {}
            try:
                parsed = json.loads(params_raw)
                if isinstance(parsed, dict):
                    params_payload = parsed
            except Exception:
                params_payload = {}
            rows.append(
                {
                    "timestamp": timestamp,
                    "type": "ai_input_mode_initial_params",
                    "step": None,
                    "summary": f"AI mode {mode}: initial parameter set applied",
                    "mode": mode,
                    "params": params_payload,
                }
            )
            continue

        learn_match = AI_INPUT_MODE_LEARN_RE.search(line)
        if learn_match:
            mode = str(learn_match.group("mode") or "")
            preset = str(learn_match.group("preset") or "")
            s_run = _parse_optional_float(learn_match.group("s_run"))
            reward = _parse_optional_float(learn_match.group("reward"))
            rows.append(
                {
                    "timestamp": timestamp,
                    "type": "ai_input_mode_learn",
                    "step": None,
                    "summary": (
                        f"AI mode learner update ({mode}): preset {preset}, "
                        f"s_run={s_run if s_run is not None else 'n/a'}, reward={reward if reward is not None else 'n/a'}"
                    ),
                    "mode": mode,
                    "selected_preset": preset,
                    "s_best": _parse_optional_float(learn_match.group("s_best")),
                    "s_end": _parse_optional_float(learn_match.group("s_end")),
                    "s_run": s_run,
                    "reward": reward,
                }
            )
            continue

        reward_outcome_match = AI_INPUT_MODE_REWARD_OUTCOME_RE.search(line)
        if reward_outcome_match:
            mode = str(reward_outcome_match.group("mode") or "")
            preset = str(reward_outcome_match.group("preset") or "")
            reward = _parse_optional_float(reward_outcome_match.group("reward"))
            rewarded_token = str(reward_outcome_match.group("rewarded") or "").strip().lower()
            rewarded = rewarded_token == "true"
            rows.append(
                {
                    "timestamp": timestamp,
                    "type": "ai_input_mode_reward_outcome",
                    "step": None,
                    "summary": (
                        f"AI mode reward outcome ({mode}): preset {preset}, "
                        f"reward={reward if reward is not None else 'n/a'}, rewarded={rewarded}"
                    ),
                    "mode": mode,
                    "selected_preset": preset,
                    "reward": reward,
                    "rewarded": rewarded,
                }
            )
            continue

    if row_limit > 0 and len(rows) > row_limit:
        rows = rows[-row_limit:]
    return rows


def _extract_loss_summary_from_log(path: Path) -> dict[str, Any]:
    best_loss_step: int | None = None
    best_loss_value: float | None = None
    final_loss_step: int | None = None
    final_loss_value: float | None = None

    if not path.exists():
        return {
            "best_loss_step": None,
            "best_loss": None,
            "final_loss_step": None,
            "final_loss": None,
        }

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                best_match = BEST_SPLAT_UPDATE_RE.search(line)
                if best_match:
                    try:
                        step = int(best_match.group("step"))
                        loss = float(best_match.group("loss"))
                        if best_loss_value is None or loss < best_loss_value:
                            best_loss_value = loss
                            best_loss_step = step
                    except Exception:
                        pass

                snap_match = GSPLAT_SNAPSHOT_RE.search(line)
                if snap_match:
                    try:
                        final_loss_step = int(snap_match.group("step"))
                        final_loss_value = float(snap_match.group("loss"))
                    except Exception:
                        pass
                    continue

                step_match = GSPLAT_STEP_RE.search(line)
                if step_match:
                    try:
                        final_loss_step = int(step_match.group("step"))
                        final_loss_value = float(step_match.group("loss"))
                    except Exception:
                        pass
    except Exception:
        pass

    if best_loss_value is None and final_loss_value is not None:
        best_loss_value = final_loss_value
        best_loss_step = final_loss_step

    return {
        "best_loss_step": best_loss_step,
        "best_loss": best_loss_value,
        "final_loss_step": final_loss_step,
        "final_loss": final_loss_value,
    }


def _extract_eval_summary(stats_dir: Path) -> dict[str, Any]:
    rows = _extract_eval_rows(stats_dir, eval_limit=200000)
    if not rows:
        return {
            "best_psnr_step": None,
            "best_psnr": None,
            "final_psnr_step": None,
            "final_psnr": None,
            "best_ssim_step": None,
            "best_ssim": None,
            "final_ssim_step": None,
            "final_ssim": None,
            "best_lpips_step": None,
            "best_lpips": None,
            "final_lpips_step": None,
            "final_lpips": None,
        }

    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    final_row = rows[-1]

    psnr_candidates = [r for r in rows if _safe_float(r.get("psnr")) is not None]
    ssim_candidates = [r for r in rows if _safe_float(r.get("ssim")) is not None]
    lpips_candidates = [r for r in rows if _safe_float(r.get("lpips")) is not None]

    best_psnr_row = max(psnr_candidates, key=lambda r: float(r.get("psnr")), default=None)
    best_ssim_row = max(ssim_candidates, key=lambda r: float(r.get("ssim")), default=None)
    best_lpips_row = min(lpips_candidates, key=lambda r: float(r.get("lpips")), default=None)

    return {
        "best_psnr_step": int(best_psnr_row.get("step")) if isinstance(best_psnr_row, dict) and isinstance(best_psnr_row.get("step"), int) else None,
        "best_psnr": _safe_float(best_psnr_row.get("psnr")) if isinstance(best_psnr_row, dict) else None,
        "final_psnr_step": int(final_row.get("step")) if isinstance(final_row.get("step"), int) else None,
        "final_psnr": _safe_float(final_row.get("psnr")),
        "best_ssim_step": int(best_ssim_row.get("step")) if isinstance(best_ssim_row, dict) and isinstance(best_ssim_row.get("step"), int) else None,
        "best_ssim": _safe_float(best_ssim_row.get("ssim")) if isinstance(best_ssim_row, dict) else None,
        "final_ssim_step": int(final_row.get("step")) if isinstance(final_row.get("step"), int) else None,
        "final_ssim": _safe_float(final_row.get("ssim")),
        "best_lpips_step": int(best_lpips_row.get("step")) if isinstance(best_lpips_row, dict) and isinstance(best_lpips_row.get("step"), int) else None,
        "best_lpips": _safe_float(best_lpips_row.get("lpips")) if isinstance(best_lpips_row, dict) else None,
        "final_lpips_step": int(final_row.get("step")) if isinstance(final_row.get("step"), int) else None,
        "final_lpips": _safe_float(final_row.get("lpips")),
    }


def _build_project_ai_learning_table(project_id: str) -> dict[str, Any]:
    """Build run-wise AI learning comparison rows for Logs tab tables."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            return {
                "project_id": project_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "ai_enabled": False,
                "rows": [],
                "message": "No runs found for this project.",
            }

        rows: list[dict[str, Any]] = []

        for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            run_id = run_dir.name
            run_analytics = _read_run_analytics(run_dir)
            if isinstance(run_analytics, dict):
                metrics = run_analytics.get("metrics") if isinstance(run_analytics.get("metrics"), dict) else {}
                ai_block = run_analytics.get("ai") if isinstance(run_analytics.get("ai"), dict) else {}
                input_mode_learning = ai_block.get("input_mode_learning") if isinstance(ai_block.get("input_mode_learning"), dict) else {}
                transition = input_mode_learning.get("transition") if isinstance(input_mode_learning.get("transition"), dict) else {}
                outcomes = transition.get("outcomes") if isinstance(transition.get("outcomes"), dict) else {}
                baseline_comparison = transition.get("baseline_comparison") if isinstance(transition.get("baseline_comparison"), dict) else {}
                best_anchor = outcomes.get("best_anchor") if isinstance(outcomes.get("best_anchor"), dict) else {}
                end_anchor = outcomes.get("end_anchor") if isinstance(outcomes.get("end_anchor"), dict) else {}
                insight = ai_block.get("input_mode_insights") if isinstance(ai_block.get("input_mode_insights"), dict) else {}

                if insight.get("ai_input_mode") or input_mode_learning:
                    rows.append(
                        {
                            "run_id": run_id,
                            "run_name": run_analytics.get("run_name"),
                            "ai_input_mode": insight.get("ai_input_mode"),
                            "baseline_run_id": insight.get("baseline_session_id"),
                            "selected_preset": input_mode_learning.get("selected_preset") or insight.get("selected_preset"),
                            "phase": "phase_a" if run_id.startswith("warmup_phase_a") else (
                                "phase_b" if run_id.startswith("warmup_phase_b") else (
                                    "phase_c" if run_id.startswith("warmup_phase_c") else "other"
                                )
                            ),
                            "is_warmup": run_id.startswith("warmup_phase_"),
                            "best_loss_step": metrics.get("best_loss_step"),
                            "best_loss": metrics.get("best_loss"),
                            "final_loss_step": metrics.get("final_loss_step"),
                            "final_loss": metrics.get("final_loss"),
                            "best_psnr_step": metrics.get("best_psnr_step"),
                            "best_psnr": metrics.get("best_psnr"),
                            "final_psnr_step": metrics.get("final_psnr_step"),
                            "final_psnr": metrics.get("final_psnr"),
                            "best_ssim_step": metrics.get("best_ssim_step"),
                            "best_ssim": metrics.get("best_ssim"),
                            "final_ssim_step": metrics.get("final_ssim_step"),
                            "final_ssim": metrics.get("final_ssim"),
                            "best_lpips_step": metrics.get("best_lpips_step"),
                            "best_lpips": metrics.get("best_lpips"),
                            "final_lpips_step": metrics.get("final_lpips_step"),
                            "final_lpips": metrics.get("final_lpips"),
                            "t_best": input_mode_learning.get("t_best"),
                            "t_eval_best": input_mode_learning.get("t_eval_best"),
                            "t_end": input_mode_learning.get("t_end"),
                            "s_best": input_mode_learning.get("s_best"),
                            "s_end": input_mode_learning.get("s_end"),
                            "s_run": input_mode_learning.get("s_run"),
                            "s_base_best": baseline_comparison.get("s_base_best"),
                            "s_base_end": baseline_comparison.get("s_base_end"),
                            "s_base": baseline_comparison.get("s_base"),
                            "reward": input_mode_learning.get("reward_signal"),
                            "baseline_best_anchor_step": baseline_comparison.get("baseline_best_anchor_step"),
                            "baseline_end_anchor_step": baseline_comparison.get("baseline_end_anchor_step"),
                            "score_weights": baseline_comparison.get("score_weights"),
                            "run_best_l": best_anchor.get("l"),
                            "run_best_q": best_anchor.get("q"),
                            "run_best_t": best_anchor.get("t"),
                            "run_best_s": best_anchor.get("s"),
                            "run_end_l": end_anchor.get("l"),
                            "run_end_q": end_anchor.get("q"),
                            "run_end_t": end_anchor.get("t"),
                            "run_end_s": end_anchor.get("s"),
                        }
                    )
                    continue

            run_config = _read_json_if_exists(run_dir / "run_config.json")
            resolved_cfg = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
            requested_cfg = run_config.get("requested_params") if isinstance(run_config, dict) and isinstance(run_config.get("requested_params"), dict) else {}

            ai_mode = str(resolved_cfg.get("ai_input_mode") or requested_cfg.get("ai_input_mode") or "").strip().lower()
            learning_path = run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"
            learning_payload = _read_json_if_exists(learning_path)

            if not ai_mode and not isinstance(learning_payload, dict):
                continue

            if not isinstance(run_analytics, dict):
                run_analytics = _ensure_run_analytics(
                    run_dir=run_dir,
                    run_config=run_config if isinstance(run_config, dict) else None,
                    ai_insights=None,
                )
                if isinstance(run_analytics, dict):
                    metrics = run_analytics.get("metrics") if isinstance(run_analytics.get("metrics"), dict) else {}
                    ai_block = run_analytics.get("ai") if isinstance(run_analytics.get("ai"), dict) else {}
                    input_mode_learning = ai_block.get("input_mode_learning") if isinstance(ai_block.get("input_mode_learning"), dict) else {}
                    transition = input_mode_learning.get("transition") if isinstance(input_mode_learning.get("transition"), dict) else {}
                    outcomes = transition.get("outcomes") if isinstance(transition.get("outcomes"), dict) else {}
                    baseline_comparison = transition.get("baseline_comparison") if isinstance(transition.get("baseline_comparison"), dict) else {}
                    best_anchor = outcomes.get("best_anchor") if isinstance(outcomes.get("best_anchor"), dict) else {}
                    end_anchor = outcomes.get("end_anchor") if isinstance(outcomes.get("end_anchor"), dict) else {}
                    insight = ai_block.get("input_mode_insights") if isinstance(ai_block.get("input_mode_insights"), dict) else {}
                    if insight.get("ai_input_mode") or input_mode_learning:
                        rows.append(
                            {
                                "run_id": run_id,
                                "run_name": run_analytics.get("run_name"),
                                "ai_input_mode": insight.get("ai_input_mode"),
                                "baseline_run_id": insight.get("baseline_session_id"),
                                "selected_preset": input_mode_learning.get("selected_preset") or insight.get("selected_preset"),
                                "phase": "phase_a" if run_id.startswith("warmup_phase_a") else (
                                    "phase_b" if run_id.startswith("warmup_phase_b") else (
                                        "phase_c" if run_id.startswith("warmup_phase_c") else "other"
                                    )
                                ),
                                "is_warmup": run_id.startswith("warmup_phase_"),
                                "best_loss_step": metrics.get("best_loss_step"),
                                "best_loss": metrics.get("best_loss"),
                                "final_loss_step": metrics.get("final_loss_step"),
                                "final_loss": metrics.get("final_loss"),
                                "best_psnr_step": metrics.get("best_psnr_step"),
                                "best_psnr": metrics.get("best_psnr"),
                                "final_psnr_step": metrics.get("final_psnr_step"),
                                "final_psnr": metrics.get("final_psnr"),
                                "best_ssim_step": metrics.get("best_ssim_step"),
                                "best_ssim": metrics.get("best_ssim"),
                                "final_ssim_step": metrics.get("final_ssim_step"),
                                "final_ssim": metrics.get("final_ssim"),
                                "best_lpips_step": metrics.get("best_lpips_step"),
                                "best_lpips": metrics.get("best_lpips"),
                                "final_lpips_step": metrics.get("final_lpips_step"),
                                "final_lpips": metrics.get("final_lpips"),
                                "t_best": input_mode_learning.get("t_best"),
                                "t_eval_best": input_mode_learning.get("t_eval_best"),
                                "t_end": input_mode_learning.get("t_end"),
                                "s_best": input_mode_learning.get("s_best"),
                                "s_end": input_mode_learning.get("s_end"),
                                "s_run": input_mode_learning.get("s_run"),
                                "s_base_best": baseline_comparison.get("s_base_best"),
                                "s_base_end": baseline_comparison.get("s_base_end"),
                                "s_base": baseline_comparison.get("s_base"),
                                "reward": input_mode_learning.get("reward_signal"),
                                "baseline_best_anchor_step": baseline_comparison.get("baseline_best_anchor_step"),
                                "baseline_end_anchor_step": baseline_comparison.get("baseline_end_anchor_step"),
                                "score_weights": baseline_comparison.get("score_weights"),
                                "run_best_l": best_anchor.get("l"),
                                "run_best_q": best_anchor.get("q"),
                                "run_best_t": best_anchor.get("t"),
                                "run_best_s": best_anchor.get("s"),
                                "run_end_l": end_anchor.get("l"),
                                "run_end_q": end_anchor.get("q"),
                                "run_end_t": end_anchor.get("t"),
                                "run_end_s": end_anchor.get("s"),
                            }
                        )
                        continue

            transition = learning_payload.get("transition") if isinstance(learning_payload, dict) and isinstance(learning_payload.get("transition"), dict) else {}
            outcomes = transition.get("outcomes") if isinstance(transition.get("outcomes"), dict) else {}
            baseline_comparison = transition.get("baseline_comparison") if isinstance(transition.get("baseline_comparison"), dict) else {}
            best_anchor = outcomes.get("best_anchor") if isinstance(outcomes.get("best_anchor"), dict) else {}
            end_anchor = outcomes.get("end_anchor") if isinstance(outcomes.get("end_anchor"), dict) else {}

            loss_summary = _extract_loss_summary_from_log(run_dir / "processing.log")
            eval_summary = _extract_eval_summary(run_dir / "outputs" / "engines" / "gsplat" / "stats")

            run_name = None
            if isinstance(run_config, dict):
                run_name = run_config.get("run_name") or run_config.get("name")
                if not run_name and isinstance(requested_cfg, dict):
                    run_name = requested_cfg.get("run_name")

            rows.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "ai_input_mode": ai_mode or None,
                    "baseline_run_id": str(resolved_cfg.get("baseline_session_id") or requested_cfg.get("baseline_session_id") or "").strip() or None,
                    "selected_preset": learning_payload.get("selected_preset") if isinstance(learning_payload, dict) else None,
                    "phase": "phase_a" if run_id.startswith("warmup_phase_a") else (
                        "phase_b" if run_id.startswith("warmup_phase_b") else (
                            "phase_c" if run_id.startswith("warmup_phase_c") else "other"
                        )
                    ),
                    "is_warmup": run_id.startswith("warmup_phase_"),
                    **loss_summary,
                    **eval_summary,
                    "t_best": learning_payload.get("t_best") if isinstance(learning_payload, dict) else None,
                    "t_eval_best": learning_payload.get("t_eval_best") if isinstance(learning_payload, dict) else None,
                    "t_end": learning_payload.get("t_end") if isinstance(learning_payload, dict) else None,
                    "s_best": learning_payload.get("s_best") if isinstance(learning_payload, dict) else None,
                    "s_end": learning_payload.get("s_end") if isinstance(learning_payload, dict) else None,
                    "s_run": learning_payload.get("s_run") if isinstance(learning_payload, dict) else None,
                    "s_base_best": baseline_comparison.get("s_base_best") if isinstance(baseline_comparison, dict) else None,
                    "s_base_end": baseline_comparison.get("s_base_end") if isinstance(baseline_comparison, dict) else None,
                    "s_base": baseline_comparison.get("s_base") if isinstance(baseline_comparison, dict) else None,
                    "reward": learning_payload.get("reward_signal") if isinstance(learning_payload, dict) else None,
                    "baseline_best_anchor_step": baseline_comparison.get("baseline_best_anchor_step") if isinstance(baseline_comparison, dict) else None,
                    "baseline_end_anchor_step": baseline_comparison.get("baseline_end_anchor_step") if isinstance(baseline_comparison, dict) else None,
                    "score_weights": baseline_comparison.get("score_weights") if isinstance(baseline_comparison, dict) else None,
                    "run_best_l": best_anchor.get("l") if isinstance(best_anchor, dict) else None,
                    "run_best_q": best_anchor.get("q") if isinstance(best_anchor, dict) else None,
                    "run_best_t": best_anchor.get("t") if isinstance(best_anchor, dict) else None,
                    "run_best_s": best_anchor.get("s") if isinstance(best_anchor, dict) else None,
                    "run_end_l": end_anchor.get("l") if isinstance(end_anchor, dict) else None,
                    "run_end_q": end_anchor.get("q") if isinstance(end_anchor, dict) else None,
                    "run_end_t": end_anchor.get("t") if isinstance(end_anchor, dict) else None,
                    "run_end_s": end_anchor.get("s") if isinstance(end_anchor, dict) else None,
                }
            )

        return {
            "project_id": project_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "ai_enabled": len(rows) > 0,
            "rows": rows,
            "message": None if rows else "No AI-applied runs with learning artifacts found.",
        }
    except HTTPException:
        raise


def _run_batch_process(
    project_id: str,
    base_params: dict,
    run_count: int,
    jitter_factor: float,
    jitter_mode: str,
    jitter_min: float,
    jitter_max: float,
    continue_on_failure: bool,
    start_index: int = 1,
    initial_previous_success_run_id: str | None = None,
    initial_completed_runs: int = 0,
    preset_sequence: list[str] | None = None,
) -> None:
    """Run multiple sessions sequentially using the same base config."""
    try:
        project_dir = DATA_DIR / project_id
        completed_runs = max(0, int(initial_completed_runs))
        batch_connect_runs = bool(base_params.get("batch_connect_runs", True))
        previous_success_run_id: str | None = str(initial_previous_success_run_id or "") or None
        successful_run_ids: list[str] = []
        last_started_run_id: str | None = None
        run_count_int = max(1, int(run_count))
        start_idx = max(1, min(int(start_index), run_count_int))
        batch_plan_id = _sanitize_run_token(str(base_params.get("batch_plan_id") or ""))
        seed_run_name = _sanitize_run_token(str(base_params.get("run_name") or "")) or None

        def _build_followup_run_name(seed_name: str | None, ordinal: int) -> str:
            if not seed_name:
                return f"batch_run{ordinal}"
            base_name = re.sub(r"[_-]run\d+$", "", seed_name, flags=re.IGNORECASE)
            base_name = base_name or seed_name
            return f"{base_name}_run{ordinal}"

        for idx in range(start_idx - 1, run_count_int):
            status.update_status(
                project_id,
                "processing",
                batch_total=run_count_int,
                batch_completed=completed_runs,
                batch_current_index=idx + 1,
                message=f"Batch session {idx + 1}/{run_count_int} starting...",
                error=None,
            )

            run_params = json.loads(json.dumps(base_params))
            run_params["run_count"] = 1
            run_params["resume"] = False
            run_params["restart_fresh"] = False
            run_params["batch_connect_runs"] = batch_connect_runs
            if batch_plan_id:
                run_params["batch_plan_id"] = batch_plan_id
            run_params["batch_index"] = idx + 1
            run_params["batch_total"] = run_count_int
            run_params["batch_completed"] = completed_runs
            run_params["batch_continue_on_failure"] = bool(continue_on_failure)
            if idx > 0:
                run_params["stage"] = "train_only"

            if batch_connect_runs and previous_success_run_id:
                run_params["start_model_mode"] = "reuse"
                run_params["source_run_id"] = previous_success_run_id
                run_params.pop("source_model_id", None)

            if idx == 0 and seed_run_name:
                run_params["run_name"] = seed_run_name
            else:
                run_params["run_name"] = _build_followup_run_name(seed_run_name, idx + 1)

            if preset_sequence:
                forced = str(preset_sequence[idx % len(preset_sequence)] or "").strip().lower()
                if forced:
                    run_params["ai_preset_override"] = forced
            else:
                run_params.pop("ai_preset_override", None)

            run_params = _apply_run_jitter(
                run_params,
                idx,
                jitter_factor,
                jitter_mode=jitter_mode,
                jitter_min=jitter_min,
                jitter_max=jitter_max,
            )
            logger.info("Batch %s/%s starting for %s (run_name=%s)", idx + 1, run_count_int, project_id, run_params.get("run_name"))

            response = process_project(project_id, ProcessParams(**run_params))
            run_id = str(response.get("run_id") or "")
            if not run_id:
                raise RuntimeError("Batch run started without run_id")
            last_started_run_id = run_id

            final_status = _wait_for_run_completion(project_id, run_id)
            final_state = str(final_status.get("status") or "")
            if final_state in {"completed", "done"}:
                completed_runs += 1
                previous_success_run_id = run_id
                successful_run_ids.append(run_id)

            is_last_run = (idx + 1) >= run_count_int
            halted_by_stop = final_state == "stopped"
            halted_by_failure = final_state == "failed" and not continue_on_failure
            will_continue = not is_last_run and not halted_by_stop and not halted_by_failure

            status_value = (
                "processing"
                if will_continue
                else (final_state if final_state in {"processing", "stopping", "completed", "done", "failed", "stopped"} else "processing")
            )
            message_value = f"Batch progress: {completed_runs}/{run_count_int} sessions completed."
            if will_continue:
                message_value = f"{message_value} Starting next session..."

            status.update_status(
                project_id,
                status_value,
                batch_total=run_count_int,
                batch_completed=completed_runs,
                batch_current_index=idx + 1,
                message=message_value,
            )

            # A user stop must always terminate the remaining batch chain.
            if final_state == "stopped":
                logger.warning("Batch halted for %s after run %s was stopped", project_id, run_id)
                break

            if final_state == "failed" and not continue_on_failure:
                logger.warning("Batch halted for %s after run %s ended with %s", project_id, run_id, final_state)
                break

        status.update_status(
            project_id,
            str(status.get_status(project_id).get("status") or "pending"),
            batch_total=run_count_int,
            batch_completed=completed_runs,
            batch_current_index=run_count_int,
            message=f"Batch finished: {completed_runs}/{run_count_int} sessions completed.",
        )

        final_run_id = previous_success_run_id or last_started_run_id
        if final_run_id:
            try:
                _write_batch_lineage(
                    project_dir,
                    {
                        "project_id": project_id,
                        "captured_at": datetime.utcnow().isoformat() + "Z",
                        "batch_connect_runs": batch_connect_runs,
                        "run_count_requested": int(run_count_int),
                        "start_index": int(start_idx),
                        "completed_count": int(completed_runs),
                        "successful_run_ids": successful_run_ids,
                        "final_run_id": final_run_id,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to persist batch lineage manifest for %s: %s", project_id, exc)
    except Exception as exc:
        logger.error("Batch process failed for %s: %s", project_id, exc, exc_info=True)
        status.update_status(project_id, "failed", error=str(exc), message=f"Batch failed: {exc}")


def _build_warmup_seed_params(project_dir: Path, requested_params: dict) -> tuple[dict, str]:
    project_status = status.get_status(project_dir.name)
    base_session_id = str(project_status.get("base_session_id") or "").strip()
    if not base_session_id:
        raise RuntimeError("Warmup requires a base session on the project.")

    base_cfg_path = project_dir / "runs" / base_session_id / "run_config.json"
    base_cfg = _read_json_if_exists(base_cfg_path)
    if not isinstance(base_cfg, dict):
        raise RuntimeError("Warmup requires a valid base session run_config.")

    base_resolved = base_cfg.get("resolved_params") if isinstance(base_cfg.get("resolved_params"), dict) else {}
    base_requested = base_cfg.get("requested_params") if isinstance(base_cfg.get("requested_params"), dict) else {}
    seed = json.loads(json.dumps(base_resolved if base_resolved else base_requested))
    if not isinstance(seed, dict) or not seed:
        raise RuntimeError("Warmup could not derive seed parameters from base session.")

    # Keep base configuration as source of truth; allow a small set of runtime overrides from request.
    for key in (
        "worker_mode",
        "engine",
        "mode",
        "tune_scope",
        "trend_scope",
        "ai_input_mode",
        "baseline_session_id",
        "start_model_mode",
        "source_model_id",
        "source_run_id",
        "source_model_checkpoint",
        "ai_preset_override",
    ):
        if key in requested_params and requested_params.get(key) not in {None, ""}:
            seed[key] = requested_params.get(key)

    # Keep the training horizon from base session (user expects 5k/4k to remain unchanged).
    if isinstance(base_resolved.get("max_steps"), (int, float)):
        seed["max_steps"] = int(base_resolved.get("max_steps"))
    if isinstance(base_resolved.get("densify_until_iter"), (int, float)):
        seed["densify_until_iter"] = int(base_resolved.get("densify_until_iter"))

    seed["stage"] = "train_only"
    seed["resume"] = False
    seed["restart_fresh"] = False
    seed["run_count"] = 1
    seed["batch_connect_runs"] = True
    seed["continue_on_failure"] = bool(requested_params.get("continue_on_failure", True))
    seed.pop("warmup_at_start", None)

    for key in (
        "run_jitter_mode",
        "run_jitter_factor",
        "run_jitter_min",
        "run_jitter_max",
        "run_jitter_multiplier",
        "batch_plan_id",
        "batch_index",
        "batch_total",
        "batch_completed",
        "batch_continue_on_failure",
        "batch_run_name_prefix",
    ):
        seed.pop(key, None)

    return seed, base_session_id


def _run_warmup_experiment(project_id: str, requested_params: dict) -> None:
    project_dir = DATA_DIR / project_id
    continue_on_failure = bool(requested_params.get("continue_on_failure", True))
    requested_total_runs = int(requested_params.get("run_count") or sum(int(p.get("runs") or 0) for p in WARMUP_PHASE_PLAN))
    warmup_plan = _resolve_warmup_phase_plan(requested_total_runs)

    try:
        seed_params, base_session_id = _build_warmup_seed_params(project_dir, requested_params)
        previous_success_run_id: str | None = None
        if str(seed_params.get("start_model_mode") or "").strip().lower() == "reuse":
            previous_success_run_id = str(seed_params.get("source_run_id") or "").strip() or None

        for phase_idx, phase in enumerate(warmup_plan, start=1):
            runs = int(phase.get("runs") or 1)
            jitter_mode = _normalize_jitter_mode(phase.get("jitter_mode"))
            jitter_min = _parse_jitter_value(phase.get("jitter_min"), 1.0)
            jitter_max = _parse_jitter_value(phase.get("jitter_max"), 1.0)
            if jitter_min > jitter_max:
                jitter_min, jitter_max = jitter_max, jitter_min

            phase_name = str(phase.get("name") or f"phase_{phase_idx}")
            phase_sequence_raw = phase.get("preset_sequence")
            phase_sequence = [
                str(item).strip().lower()
                for item in (phase_sequence_raw if isinstance(phase_sequence_raw, list) else [])
                if str(item).strip()
            ] or None
            phase_seed = json.loads(json.dumps(seed_params))
            phase_seed["batch_plan_id"] = f"warmup_{uuid.uuid4().hex[:12]}"
            phase_seed["run_name"] = f"warmup_{phase_name}"

            if previous_success_run_id and bool(phase_seed.get("batch_connect_runs", True)):
                phase_seed["start_model_mode"] = "reuse"
                phase_seed["source_run_id"] = previous_success_run_id
                phase_seed.pop("source_model_id", None)

            status.update_status(
                project_id,
                "processing",
                message=(
                    f"Warmup {phase_idx}/{len(warmup_plan)} ({phase_name}) queued: "
                    f"{runs} runs, jitter={jitter_mode} [{jitter_min}, {jitter_max}]"
                ),
                stop_requested=False,
                error=None,
                batch_total=runs,
                batch_completed=0,
                batch_current_index=0,
            )

            _run_batch_process(
                project_id,
                phase_seed,
                runs,
                1.0,
                jitter_mode,
                jitter_min,
                jitter_max,
                continue_on_failure,
                1,
                previous_success_run_id,
                0,
                phase_sequence,
            )

            lineage = _read_batch_lineage(project_dir)
            final_run_id = str((lineage or {}).get("final_run_id") or "").strip()
            if final_run_id:
                previous_success_run_id = final_run_id

            phase_status = status.get_status(project_id)
            if str(phase_status.get("status") or "") in {"failed", "stopped"}:
                break

        latest = status.get_status(project_id)
        latest_state = str(latest.get("status") or "pending")
        if latest_state not in {"failed", "stopped"}:
            status.update_status(
                project_id,
                latest_state,
                message=(
                    f"Warmup experiment finished ({sum(int(p.get('runs') or 0) for p in warmup_plan)} runs). "
                    f"Base session: {base_session_id}."
                ),
            )
    except Exception as exc:
        logger.error("Warmup experiment failed for %s: %s", project_id, exc, exc_info=True)
        status.update_status(project_id, "failed", error=str(exc), message=f"Warmup failed: {exc}")


def _sanitize_run_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned[:80]


def _build_default_run_name(project_label: str | None, runs_root: Path | None = None) -> str:
    prefix = _sanitize_run_token(project_label or "project") or "project"
    if not runs_root:
        return f"{prefix}_session1"

    next_idx = 1
    pattern = re.compile(rf"^{re.escape(prefix)}_session(\d+)$")
    try:
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if not match:
                continue
            next_idx = max(next_idx, int(match.group(1)) + 1)
    except Exception:
        pass

    return f"{prefix}_session{next_idx}"


def _rewrite_auto_run_name_prefix(
    run_name_requested: str,
    project_id: str,
    project_label: str | None,
) -> str:
    """If request looks like auto id-based name, switch to project-name prefix."""
    requested = (run_name_requested or "").strip()
    if not requested:
        return requested

    project_id_prefix = _sanitize_run_token(project_id) or "project"
    preferred_prefix = _sanitize_run_token(project_label or project_id) or project_id_prefix
    if preferred_prefix == project_id_prefix:
        return requested

    # Match auto-generated shapes like:
    # <project_id_prefix>_YYYYMMDD_HHMMSS, <project_id_prefix>_YYYYMMDD_HHMMSS_01,
    # or <project_id_prefix>_sessionN
    match = re.fullmatch(
        rf"{re.escape(project_id_prefix)}_((?:\d{{8}}_\d{{6}}(?:_\d{{2}})?)|(?:session\d+))",
        requested,
    )
    if not match:
        return requested

    return f"{preferred_prefix}_{match.group(1)}"


def _resolve_unique_run_id(runs_root: Path, preferred_name: str) -> str:
    base = _sanitize_run_token(preferred_name) or _build_default_run_name("project")
    candidate = base
    idx = 1
    while (runs_root / candidate).exists():
        candidate = f"{base}_{idx:02d}"
        idx += 1
    return candidate


def _read_sparse_image_names(candidate_dir: Path) -> list[str]:
    images_bin = candidate_dir / "images.bin"
    if not images_bin.exists():
        return []
    try:
        from bimba3d_backend.worker.colmap_loader import read_images_binary  # pylint: disable=import-outside-toplevel
        images = read_images_binary(images_bin)
        names = [entry.get("name") for entry in images.values() if isinstance(entry.get("name"), str)]
        names.sort()
        return names
    except Exception as exc:
        logger.debug("Failed to read image names from %s: %s", images_bin, exc)
        return []
def _load_sparse_metadata(sparse_root: Path) -> tuple[dict | None, Path]:
    meta_path = sparse_root / BEST_SPARSE_META
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            logger.debug("Failed to parse sparse metadata at %s: %s", meta_path, exc)
    return meta, meta_path



def _colmap_to_opengl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax, ay, az = COLMAP_TO_OPENGL
    return float(ax * x), float(ay * y), float(az * z)


def _resolve_sparse_candidate_dir(sparse_root: Path, rel_path: str | None) -> Path:
    if not rel_path or rel_path in {".", ""}:
        return sparse_root
    try:
        candidate = (sparse_root / rel_path).resolve()
        base = sparse_root.resolve()
        if candidate == base or base in candidate.parents:
            return candidate
    except Exception:
        pass
    return sparse_root


def _read_sparse_stats(candidate_dir: Path) -> tuple[int | None, int | None]:
    images = None
    points = None
    try:
        with open(candidate_dir / "images.bin", "rb") as handle:
            header = handle.read(8)
            if len(header) == 8:
                images = int(struct.unpack("<Q", header)[0])
    except Exception:
        pass
    try:
        with open(candidate_dir / "points3D.bin", "rb") as handle:
            header = handle.read(8)
            if len(header) == 8:
                points = int(struct.unpack("<Q", header)[0])
    except Exception:
        pass
    return images, points


def _resolve_sparse_candidate_for_edit(project_dir: Path, requested: str | None) -> tuple[Path, str]:
    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    meta, _ = _load_sparse_metadata(sparse_root)
    token = (requested or "").strip()
    token_lower = token.lower()

    def candidate_for(rel_path: str) -> Path | None:
        resolved = _resolve_sparse_candidate_dir(sparse_root, rel_path)
        if resolved.exists() and (
            (resolved / "points3D.bin").exists() or (resolved / "points3D.txt").exists()
        ):
            return resolved
        return None

    if token and token_lower not in {"best", "auto"}:
        normalized = "." if token in {"", ".", "root"} else token
        candidate_dir = candidate_for(normalized)
        if not candidate_dir:
            raise HTTPException(status_code=404, detail="Requested sparse reconstruction not found")
        return candidate_dir, normalized

    preferred_rel = meta.get("relative_path") if isinstance(meta, dict) else None
    if preferred_rel:
        candidate_dir = candidate_for(preferred_rel)
        if candidate_dir:
            return candidate_dir, preferred_rel

    root_dir = candidate_for(".")
    if root_dir:
        return root_dir, "."

    try:
        for child in sorted(p for p in sparse_root.iterdir() if p.is_dir()):
            rel_path = os.path.relpath(child, sparse_root)
            candidate_dir = candidate_for(rel_path)
            if candidate_dir:
                if rel_path in {"", "."}:
                    rel_path = "."
                return candidate_dir, rel_path
    except Exception as exc:
        logger.debug("Failed to enumerate sparse directories for edit: %s", exc)

    raise HTTPException(status_code=404, detail="No sparse reconstruction available to edit")


def _update_sparse_candidate_points(project_dir: Path, candidate_rel: str, points: int | None) -> None:
    if points is None:
        return
    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        return
    meta, meta_path = _load_sparse_metadata(sparse_root)
    if not isinstance(meta, dict):
        meta = {}
    candidates = meta.setdefault("candidates", [])
    norm_rel = candidate_rel or "."
    updated = False
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        entry_rel = entry.get("relative_path") or "."
        if entry_rel == norm_rel:
            entry["points"] = points
            updated = True
            break
    if not updated:
        candidates.append({"relative_path": norm_rel, "points": points})
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        logger.debug("Failed to update sparse metadata at %s: %s", meta_path, exc)


def _is_colmap_reconstruction_dir(candidate_dir: Path) -> bool:
    """Return True when a directory looks like a valid COLMAP sparse reconstruction."""
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return False
    has_cameras = (candidate_dir / "cameras.bin").exists() or (candidate_dir / "cameras.txt").exists()
    has_images = (candidate_dir / "images.bin").exists() or (candidate_dir / "images.txt").exists()
    has_points = (candidate_dir / "points3D.bin").exists() or (candidate_dir / "points3D.txt").exists()
    return bool(has_cameras and has_images and has_points)


def _has_colmap_sparse_outputs(sparse_root: Path) -> bool:
    """Check whether sparse root contains at least one valid COLMAP reconstruction dir."""
    if not sparse_root.exists() or not sparse_root.is_dir():
        return False

    if _is_colmap_reconstruction_dir(sparse_root):
        return True

    try:
        for child in sparse_root.iterdir():
            if child.is_dir() and _is_colmap_reconstruction_dir(child):
                return True
    except Exception:
        return False

    return False


def _base_session_colmap_ready(project_dir: Path, base_session_id: str | None) -> bool:
    """True when base session has COLMAP sparse data (shared or run-local)."""
    if not base_session_id:
        return False

    shared_sparse_root = project_dir / "outputs" / "sparse"
    if _has_colmap_sparse_outputs(shared_sparse_root):
        return True

    base_run_sparse_root = project_dir / "runs" / base_session_id / "outputs" / "sparse"
    return _has_colmap_sparse_outputs(base_run_sparse_root)

logger = logging.getLogger(__name__)

router = APIRouter()

ENGINE_SUBDIR = "engines"
ENGINE_NAME_RE = re.compile(r"^[a-z0-9_-]+$", re.IGNORECASE)

# Map EXIF GPS tag id for quick lookup
EXIF_GPS_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == "GPSInfo":
        EXIF_GPS_TAG = k
        break


def _normalize_engine_name(engine: str | None) -> str | None:
    if engine is None:
        return None
    candidate = str(engine).strip()
    if not candidate:
        return None
    if not ENGINE_NAME_RE.fullmatch(candidate):
        return None
    return candidate.lower()


def _sanitize_engine(engine: str | None) -> str | None:
    if engine is None:
        return None
    normalized = _normalize_engine_name(engine)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid engine selector")
    return normalized


def _resolve_output_path(project_dir: Path, relative_path: str | Path, engine: str | None = None) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid path request")
    base = project_dir / "outputs"
    if engine:
        base = base / ENGINE_SUBDIR / engine
    return base / rel


def _engine_search_order(project_id: str, sanitized_engine: str | None) -> tuple[list[str], str | None]:
    if sanitized_engine:
        return [sanitized_engine], sanitized_engine
    inferred = _infer_engine(project_id)
    order: list[str] = []
    if inferred:
        order.append(inferred)
    return order, inferred


def _infer_engine(project_id: str) -> str | None:
    try:
        info = status.get_status(project_id)
        normalized = _normalize_engine_name(info.get("engine"))
        if normalized:
            return normalized
    except Exception:
        pass
    project_dir = DATA_DIR / project_id
    engines_root = project_dir / "outputs" / ENGINE_SUBDIR
    if engines_root.exists() and engines_root.is_dir():
        for entry in sorted(p for p in engines_root.iterdir() if p.is_dir()):
            normalized = _normalize_engine_name(entry.name)
            if normalized:
                return normalized
    return None


def _find_existing_path(
    project_id: str,
    relative_path: str | Path,
    engine: str | None,
    run_id: str | None = None,
    *,
    expect_directory: bool = False,
) -> tuple[Path | None, str | None, str | None, str | None]:
    sanitized = _sanitize_engine(engine)
    search_order, inferred = _engine_search_order(project_id, sanitized)
    project_dir = DATA_DIR / project_id
    for candidate in search_order:
        if run_id:
            candidate_path = project_dir / "runs" / run_id / "outputs"
            if candidate:
                candidate_path = candidate_path / ENGINE_SUBDIR / candidate
            candidate_path = candidate_path / Path(relative_path)
        else:
            candidate_path = _resolve_output_path(project_dir, relative_path, candidate)
        if expect_directory:
            if candidate_path.exists() and candidate_path.is_dir():
                return candidate_path, candidate, sanitized, inferred
        else:
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path, candidate, sanitized, inferred
    return None, None, sanitized, inferred


def _rational_to_float(value: tuple) -> float:
    # Convert EXIF rational tuple (num, den) into float; guard against zero division
    # Handle both simplified float and rational tuple formats
    if isinstance(value, (int, float)):
        return float(value)
    
    # Pillow IFDRational has numerator/denominator attributes
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        den = value.denominator
        return float(value.numerator) / float(den) if den else 0.0
    
    # Tuple format (num, den)
    try:
        num, den = value
        return float(num) / float(den) if den else 0.0
    except Exception:
        return 0.0
    return float(num) / float(den) if den else 0.0


def _dms_to_dd(dms: tuple, ref: str) -> float:
    # Convert degrees/minutes/seconds to decimal degrees with hemisphere reference
    # Handle both simplified tuple (37.0, 49.0, 11.63) and rational ((37,1), (49,1), (1163,100))
    if isinstance(dms, (int, float)):
        # Already in decimal format
        dd = float(dms)
    elif len(dms) == 3:
        deg = _rational_to_float(dms[0])
        minutes = _rational_to_float(dms[1])
        seconds = _rational_to_float(dms[2])
        dd = deg + (minutes / 60.0) + (seconds / 3600.0)
    else:
        return 0.0
    
    if ref in ["S", "W"]:
        dd = -dd
    return dd


def extract_gps(filepath: Path) -> Optional[dict]:
    """Extract GPS lat/lon from image EXIF if available."""
    try:
        if EXIF_GPS_TAG is None:
            return None

        with Image.open(filepath) as img:
            exif = img._getexif()  # noqa: SLF001 Pillow private API is standard for EXIF
            if not exif or EXIF_GPS_TAG not in exif:
                return None

            gps_info = exif.get(EXIF_GPS_TAG, {})
            gps_data = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            lat = gps_data.get("GPSLatitude")
            lat_ref = gps_data.get("GPSLatitudeRef")
            lon = gps_data.get("GPSLongitude")
            lon_ref = gps_data.get("GPSLongitudeRef")

            if lat is None or lon is None or lat_ref is None or lon_ref is None:
                return None

            return {
                "lat": _dms_to_dd(lat, lat_ref),
                "lon": _dms_to_dd(lon, lon_ref),
            }
    except Exception as e:  # pragma: no cover - best effort extraction
        logger.debug(f"EXIF GPS extraction failed for {filepath}: {e}")
        return None


@router.get("", response_model=list[ProjectListItem], include_in_schema=False)
@router.get("/", response_model=list[ProjectListItem])
def list_projects():
    """List all projects with status and basic metadata."""
    try:
        projects: list[ProjectListItem] = []
        if not DATA_DIR.exists():
            return projects

        for project_dir in sorted(DATA_DIR.iterdir()):
            if not project_dir.is_dir():
                continue
            project_id = project_dir.name
            project_status = status.get_status(project_id)
            current_status = project_status.get("status", "pending")
            progress = int(project_status.get("progress", 0) or 0)
            has_outputs = (
                (project_dir / "outputs" / "engines" / "gsplat" / "splats.splat").exists()
                or (project_dir / "outputs" / "engines" / "gsplat" / "splats.ply").exists()
                or (project_dir / "outputs" / "engines" / "gsplat" / "metadata.json").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "splats.splat").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "splats.ply").exists()
                or (project_dir / "outputs" / "engines" / "litegs" / "metadata.json").exists()
            )
            runs_root = project_dir / "runs"
            session_count = 0
            if runs_root.exists() and runs_root.is_dir():
                try:
                    session_count = sum(1 for p in runs_root.iterdir() if p.is_dir())
                except Exception:
                    session_count = 0
            modified_at = None
            try:
                status_file = project_dir / "status.json"
                mtime_source = status_file if status_file.exists() else project_dir
                modified_at = datetime.utcfromtimestamp(mtime_source.stat().st_mtime).isoformat() + "Z"
            except Exception:
                modified_at = None
            projects.append(
                ProjectListItem(
                    project_id=project_id,
                    name=project_status.get("name"),
                    status=current_status,
                    progress=progress,
                    created_at=project_status.get("created_at"),
                    modified_at=modified_at,
                    has_outputs=has_outputs,
                    session_count=session_count,
                )
            )

        return projects
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list projects")


@router.post("", response_model=ProjectResponse, include_in_schema=False)
@router.post("/", response_model=ProjectResponse)
def create_project(payload: CreateProjectRequest | None = Body(None)):
    """Create a new project with optional human-friendly name."""
    try:
        storage_root_id = payload.storage_root_id if payload else None
        storage_path = payload.storage_path if payload else None
        try:
            project_root = storage.resolve_storage_root(storage_root_id=storage_root_id, storage_path=storage_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        project_id, project_dir = storage.create_project(base_dir=project_root)
        provided_name = (payload.name.strip() if payload and payload.name else None)

        # Initialize status file with name
        status.initialize_status(project_id, name=provided_name)

        logger.info(f"Created project: {project_id} name={provided_name}")
        project_status = status.get_status(project_id)
        return {
            "project_id": project_id,
            "name": project_status.get("name"),
            "created_at": project_status.get("created_at"),
        }
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create project")


@router.get("/storage-roots", response_model=list[StorageRootResponse])
def get_storage_roots():
    """List available storage roots for project creation."""
    try:
        roots = storage.list_storage_roots()
        return [StorageRootResponse(**entry) for entry in roots]
    except Exception as e:
        logger.error(f"Error listing storage roots: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list storage roots")


@router.get("/models")
def list_reusable_models():
    """List globally elevated reusable gsplat models."""
    try:
        items = model_registry.load_models_index()
        items = sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"models": items}
    except Exception as exc:
        logger.error("Error listing reusable models: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list reusable models")


@router.patch("/models/{model_id}")
def rename_reusable_model(model_id: str, payload: RenameModelRequest):
    """Rename one reusable model display name."""
    model_name = str(payload.model_name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    try:
        updated = model_registry.rename_model(model_id, model_name)
        if not isinstance(updated, dict):
            raise HTTPException(status_code=404, detail="Reusable model not found")
        return {"status": "renamed", "model": updated}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error renaming reusable model %s: %s", model_id, exc)
        raise HTTPException(status_code=500, detail="Failed to rename reusable model")


@router.delete("/models/{model_id}")
def delete_reusable_model(model_id: str):
    """Delete one reusable model from global registry and remove its folder."""
    try:
        existing = model_registry.get_model_record(model_id)
        if not isinstance(existing, dict):
            raise HTTPException(status_code=404, detail="Reusable model not found")

        paths = existing.get("paths") if isinstance(existing.get("paths"), dict) else {}
        model_dir_raw = paths.get("model_dir") if isinstance(paths, dict) else None
        model_dir = Path(str(model_dir_raw)).expanduser() if model_dir_raw else (model_registry.MODELS_DIR / model_id)

        if model_dir.exists() or model_dir.is_symlink():
            _delete_path_strict(model_dir)

        removed = model_registry.remove_model(model_id)
        if not isinstance(removed, dict):
            raise HTTPException(status_code=404, detail="Reusable model not found")

        return {"status": "deleted", "model_id": model_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting reusable model %s: %s", model_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete reusable model")


@router.get("/models/{model_id}/lineage")
def get_reusable_model_lineage(model_id: str):
    """Return lineage metadata and copied contributor config tree for one model."""
    try:
        detail = model_registry.get_model_lineage_summary(model_id)
        if not isinstance(detail, dict):
            raise HTTPException(status_code=404, detail="Reusable model not found")
        return detail
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error loading reusable model lineage for %s: %s", model_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load reusable model lineage")


@router.get("/models/{model_id}/configs/{project_id}/{run_id}/{filename}")
def get_reusable_model_config_snapshot(model_id: str, project_id: str, run_id: str, filename: str):
    """Read copied config snapshot JSON for a model contributor."""
    try:
        target = model_registry.resolve_config_snapshot_file(model_id, project_id, run_id, filename)
        if target is None:
            raise HTTPException(status_code=404, detail="Config snapshot file not found")

        payload = model_registry.read_json_if_exists(target)
        if not isinstance(payload, (dict, list)):
            raise HTTPException(status_code=400, detail="Snapshot file is not valid JSON")

        return {
            "model_id": model_id,
            "project_id": project_id,
            "run_id": run_id,
            "filename": filename,
            "size": target.stat().st_size,
            "content": payload,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error loading config snapshot for model=%s project=%s run=%s file=%s: %s",
            model_id,
            project_id,
            run_id,
            filename,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to load config snapshot")


@router.get("/models/{model_id}/configs/{project_id}/{run_id}/{filename}/download")
def download_reusable_model_config_snapshot(model_id: str, project_id: str, run_id: str, filename: str):
    """Download copied config snapshot file for a model contributor."""
    try:
        target = model_registry.resolve_config_snapshot_file(model_id, project_id, run_id, filename)
        if target is None:
            raise HTTPException(status_code=404, detail="Config snapshot file not found")
        return FileResponse(path=target, filename=filename, media_type="application/json")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error downloading config snapshot for model=%s project=%s run=%s file=%s: %s",
            model_id,
            project_id,
            run_id,
            filename,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to download config snapshot")


@router.post("/{project_id}/runs/{run_id}/elevate-model")
def elevate_project_run_model(project_id: str, run_id: str, payload: ElevateModelRequest | None = Body(None)):
    """Promote a completed gsplat run checkpoint into global reusable model storage."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_dir = project_dir / "runs" / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        checkpoint_path = model_registry.find_latest_gsplat_checkpoint(run_dir)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise HTTPException(status_code=400, detail="No gsplat checkpoint found for this run")

        metadata_path = run_dir / "outputs" / "engines" / "gsplat" / "metadata.json"
        run_config_path = run_dir / "run_config.json"
        run_cfg = _read_json_if_exists(run_config_path)
        project_status = status.get_status(project_id)
        project_name = project_status.get("name") if isinstance(project_status, dict) else None

        model_name = ""
        if payload and payload.model_name:
            model_name = str(payload.model_name).strip()
        if not model_name:
            model_name = run_id

        model_id = model_registry.build_model_id(model_name)
        model_dir = model_registry.MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=False)
        captured_at = model_registry.utc_now_iso()

        target_ckpt = model_dir / "source_checkpoint.pt"
        shutil.copy2(checkpoint_path, target_ckpt)

        if metadata_path.exists():
            try:
                shutil.copy2(metadata_path, model_dir / "metadata.json")
            except Exception as exc:
                logger.warning("Failed to copy metadata for elevated model %s: %s", model_id, exc)

        source_model_id = None
        if isinstance(run_cfg, dict):
            resolved = run_cfg.get("resolved_params") if isinstance(run_cfg.get("resolved_params"), dict) else {}
            if str(resolved.get("start_model_mode") or "").strip().lower() == "reuse":
                source_model_id = str(resolved.get("source_model_id") or "").strip() or None

        parent_model_record = model_registry.resolve_reusable_model(source_model_id or "") if source_model_id else None
        model_registry.import_parent_configs_into_model(parent_model_record, model_dir)
        parent_contributors = model_registry.load_parent_lineage_contributors(parent_model_record)
        current_contributor = model_registry.snapshot_contributor_configs(
            model_dir=model_dir,
            project_dir=project_dir,
            run_dir=run_dir,
            project_id=project_id,
            project_name=str(project_name) if project_name else None,
            run_id=run_id,
            captured_at=captured_at,
        )

        batch_contributors: list[dict] = []
        try:
            batch_doc = _read_batch_lineage(project_dir)
            if (
                isinstance(batch_doc, dict)
                and str(batch_doc.get("project_id") or "") == project_id
                and str(batch_doc.get("final_run_id") or "") == run_id
            ):
                successful_ids = batch_doc.get("successful_run_ids") if isinstance(batch_doc.get("successful_run_ids"), list) else []
                for candidate in successful_ids:
                    candidate_run_id = str(candidate or "").strip()
                    if not candidate_run_id or candidate_run_id == run_id:
                        continue
                    candidate_run_dir = project_dir / "runs" / candidate_run_id
                    if not candidate_run_dir.exists() or not candidate_run_dir.is_dir():
                        continue
                    batch_contributors.append(
                        model_registry.snapshot_contributor_configs(
                            model_dir=model_dir,
                            project_dir=project_dir,
                            run_dir=candidate_run_dir,
                            project_id=project_id,
                            project_name=str(project_name) if project_name else None,
                            run_id=candidate_run_id,
                            captured_at=captured_at,
                        )
                    )
        except Exception as exc:
            logger.warning("Failed to include batch lineage contributors for %s/%s: %s", project_id, run_id, exc)

        lineage_doc = model_registry.write_lineage(
            model_dir=model_dir,
            model_id=model_id,
            source_model_id=source_model_id,
            contributors=[*parent_contributors, *batch_contributors, current_contributor],
            captured_at=captured_at,
        )
        provenance_summary = model_registry.summarize_lineage(lineage_doc.get("contributors") or [])

        model_record = {
            "model_id": model_id,
            "model_name": model_name,
            "engine": "gsplat",
            "created_at": captured_at,
            "source": {
                "project_id": project_id,
                "project_name": project_name,
                "run_id": run_id,
            },
            "paths": {
                "checkpoint": str(target_ckpt),
                "artifact": None,
                "model_dir": str(model_dir),
                "configs_dir": str(model_dir / "configs"),
                "lineage": str(model_dir / "lineage.json"),
            },
            "artifact_format": None,
            "provenance_summary": provenance_summary,
        }

        records = model_registry.load_models_index()
        records.append(model_record)
        model_registry.save_models_index(records)

        model_registry.write_json_atomic(model_dir / "model.json", model_record)

        return {"status": "model_elevated", "model": model_record}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Model id collision, retry elevation")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to elevate model from %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to elevate model")


@router.post("/{project_id}/images")
async def upload_images(project_id: str, images: list[UploadFile] = File(...)):
    """Upload images to a project."""
    try:
        project_dir = DATA_DIR / project_id
        images_dir = project_dir / "images"
        thumbnails_dir = images_dir / "thumbnails"
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        images_dir.mkdir(parents=True, exist_ok=True)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        uploaded_count = 0
        invalid_files: list[str] = []
        allowed_ext_text = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        gps_records: dict[str, dict] = {}
        gps_file = images_dir / "locations.json"
        if gps_file.exists():
            try:
                gps_records = json.loads(gps_file.read_text())
            except Exception:
                gps_records = {}
        
        for img in images:
            # Validate file extension
            file_ext = Path(img.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(f"Skipped invalid image: {img.filename}")
                invalid_files.append(img.filename)
                continue
            
            # Read and save file
            content = await img.read()
            file_path = images_dir / img.filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Generate thumbnail
            try:
                with Image.open(file_path) as image:
                    # Create thumbnail (max 100x100 for faster loading)
                    image.thumbnail((100, 100), Image.Resampling.LANCZOS)
                    
                    # Save thumbnail with quality optimization
                    thumbnail_path = thumbnails_dir / img.filename
                    image.save(thumbnail_path, "JPEG", quality=80, optimize=True)
                    
                    logger.info(f"Generated thumbnail for: {img.filename}")
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for {img.filename}: {str(e)}")

            # Extract GPS if available
            gps = extract_gps(file_path)
            if gps:
                gps_records[img.filename] = gps
                logger.info(f"Captured GPS for {img.filename}: {gps}")
            
            uploaded_count += 1
            logger.info(f"Uploaded image: {img.filename} to {project_id}")
        
        if uploaded_count == 0:
            invalid_list = ", ".join(invalid_files[:10])
            if len(invalid_files) > 10:
                invalid_list = f"{invalid_list}, ..."
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No valid images uploaded. Allowed formats: {allowed_ext_text}. "
                    f"Invalid files: {invalid_list}"
                ),
            )
        
        # Persist GPS metadata if any
        try:
            if gps_records:
                gps_file.write_text(json.dumps(gps_records, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write GPS metadata for {project_id}: {e}")

        return {"status": "uploaded", "count": uploaded_count}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload images")


@router.post("/{project_id}/process")
def process_project(project_id: str, params: ProcessParams | None = Body(None)):
    """Start processing a project in background thread."""
    try:
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if images exist
        images_dir = project_dir / "images"
        if not images_dir.exists() or not list(images_dir.glob("*")):
            raise HTTPException(status_code=400, detail="No images in project")

        # Starting/resuming should always clear stale stop markers from previous runs.
        stop_flag = project_dir / "stop_requested"
        try:
            if stop_flag.exists():
                stop_flag.unlink()
        except Exception as exc:
            logger.warning("Failed to clear stale stop flag for %s: %s", project_id, exc)

        try:
            status.clear_stop_state(project_id)
        except Exception as exc:
            logger.warning("Failed to clear stale stop metadata for %s: %s", project_id, exc)

        # Prepare params payload with defaults (engine defaults to gsplat)
        requested_params = params.dict(exclude_none=True) if params else {}
        params_payload = dict(requested_params)

        # Repro defaults for provided COLMAP pipelines.
        params_payload.setdefault("stage", "train_only")
        params_payload.setdefault("max_steps", 15000)
        params_payload.setdefault("log_interval", 100)
        params_payload.setdefault("save_interval", 31000)
        params_payload.setdefault("splat_export_interval", 31000)
        params_payload.setdefault("best_splat_interval", 100)
        params_payload.setdefault("best_splat_start_step", 2000)
        params_payload.setdefault("save_best_splat", False)
        params_payload.setdefault("auto_early_stop", False)
        params_payload.setdefault("early_stop_monitor_interval", 200)
        params_payload.setdefault("early_stop_decision_points", 10)
        params_payload.setdefault("early_stop_min_eval_points", 6)
        params_payload.setdefault("early_stop_min_step_ratio", 0.25)
        params_payload.setdefault("early_stop_monitor_min_relative_improvement", 0.0015)
        params_payload.setdefault("early_stop_eval_min_relative_improvement", 0.003)
        params_payload.setdefault("early_stop_max_volatility_ratio", 0.01)
        params_payload.setdefault("early_stop_ema_alpha", 0.1)
        params_payload.setdefault("tune_end_step", 15000)
        params_payload.setdefault("tune_interval", 100)
        params_payload.setdefault("trend_scope", "run")
        params_payload.setdefault("batch_size", 1)
        params_payload.setdefault("densify_from_iter", 500)
        params_payload.setdefault("densify_until_iter", 10000)
        params_payload.setdefault("densification_interval", 100)
        params_payload.setdefault("densify_grad_threshold", 0.0002)
        params_payload.setdefault("opacity_reset_interval", 3000)
        params_payload.setdefault("lambda_dssim", 0.2)
        params_payload.setdefault("feature_lr", 2.5e-3)
        params_payload.setdefault("opacity_lr", 5.0e-2)
        params_payload.setdefault("scaling_lr", 5.0e-3)
        params_payload.setdefault("rotation_lr", 1.0e-3)
        params_payload.setdefault("percent_dense", 0.01)
        params_payload.setdefault("position_lr_init", 1.6e-4)
        params_payload.setdefault("position_lr_final", 1.6e-6)
        params_payload.setdefault("position_lr_delay_mult", 0.01)
        params_payload.setdefault("position_lr_max_steps", 30000)

        # Validate/resolve worker runtime mode
        try:
            normalize_worker_mode(params_payload.get("worker_mode"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        resolved_worker_mode = resolve_worker_mode(params_payload.get("worker_mode"))
        params_payload["worker_mode"] = resolved_worker_mode

        engine = params_payload.get("engine", "gsplat")
        if engine not in {"gsplat", "litegs"}:
            raise HTTPException(status_code=400, detail=f"Invalid training engine: {engine}")
        params_payload["engine"] = engine

        # Optional AI input mode for initial presets in core_ai_optimization.
        mode_value = str(params_payload.get("mode") or "baseline").strip().lower()
        tune_scope_value = str(params_payload.get("tune_scope") or "").strip().lower()
        requested_ai_input_mode = str(requested_params.get("ai_input_mode") or "").strip().lower()
        requested_preset_override = str(requested_params.get("ai_preset_override") or "").strip().lower()
        valid_ai_input_modes = {
            "exif_only",
            "exif_plus_flight_plan",
            "exif_plus_flight_plan_plus_external",
        }
        valid_ai_preset_overrides = {"conservative", "balanced", "geometry_fast", "appearance_fast"}
        if requested_ai_input_mode and requested_ai_input_mode not in valid_ai_input_modes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "ai_input_mode must be one of: exif_only, exif_plus_flight_plan, "
                    "exif_plus_flight_plan_plus_external"
                ),
            )
        if requested_preset_override and requested_preset_override not in valid_ai_preset_overrides:
            raise HTTPException(
                status_code=400,
                detail=(
                    "ai_preset_override must be one of: conservative, balanced, geometry_fast, appearance_fast"
                ),
            )

        if engine == "gsplat" and mode_value == "modified" and tune_scope_value == "core_ai_optimization":
            chosen_mode = requested_ai_input_mode or str(params_payload.get("ai_input_mode") or "").strip().lower()
            if chosen_mode:
                if chosen_mode not in valid_ai_input_modes:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "ai_input_mode must be one of: exif_only, exif_plus_flight_plan, "
                            "exif_plus_flight_plan_plus_external"
                        ),
                    )
                params_payload["ai_input_mode"] = chosen_mode

                baseline_session_id = str(requested_params.get("baseline_session_id") or "").strip()
                if not baseline_session_id:
                    raise HTTPException(
                        status_code=400,
                        detail="baseline_session_id is required for core_ai_optimization with ai_input_mode.",
                    )

                baseline_run_dir = project_dir / "runs" / baseline_session_id
                if not baseline_run_dir.exists() or not baseline_run_dir.is_dir():
                    raise HTTPException(status_code=404, detail="Selected baseline session not found")

                baseline_eval_path = baseline_run_dir / "outputs" / "engines" / "gsplat" / "eval_history.json"
                if not baseline_eval_path.exists():
                    raise HTTPException(
                        status_code=400,
                        detail="Selected baseline session has no gsplat eval history",
                    )

                baseline_run_cfg = _read_json_if_exists(baseline_run_dir / "run_config.json")
                baseline_mode = str(
                    (baseline_run_cfg.get("resolved_params") or {}).get("mode")
                    or (baseline_run_cfg.get("requested_params") or {}).get("mode")
                    or ""
                ).strip().lower()
                if baseline_mode and baseline_mode != "baseline":
                    raise HTTPException(
                        status_code=400,
                        detail="Selected baseline session must be a baseline-mode run",
                    )
                params_payload["baseline_session_id"] = baseline_session_id
                if requested_preset_override:
                    params_payload["ai_preset_override"] = requested_preset_override
            else:
                params_payload.pop("ai_input_mode", None)
                params_payload.pop("baseline_session_id", None)
                params_payload.pop("ai_preset_override", None)
        else:
            params_payload.pop("ai_input_mode", None)
            params_payload.pop("baseline_session_id", None)
            params_payload.pop("ai_preset_override", None)

        # Optional warm-start from globally elevated model.
        start_model_mode = str(requested_params.get("start_model_mode") or "scratch").strip().lower()
        if start_model_mode not in {"scratch", "reuse"}:
            raise HTTPException(status_code=400, detail="start_model_mode must be 'scratch' or 'reuse'")

        if start_model_mode == "reuse":
            source_model_id = str(requested_params.get("source_model_id") or "").strip()
            source_run_id = str(requested_params.get("source_run_id") or "").strip()
            if not source_model_id and not source_run_id:
                raise HTTPException(status_code=400, detail="source_model_id or source_run_id is required when start_model_mode='reuse'")

            mode = str(params_payload.get("mode") or "baseline")
            tune_scope = str(params_payload.get("tune_scope") or "")
            if engine != "gsplat" or mode != "modified" or tune_scope != "core_ai_optimization":
                raise HTTPException(
                    status_code=400,
                    detail="Model reuse is supported only for gsplat + modified + core_ai_optimization.",
                )

            checkpoint_path: Path | None = None
            if source_model_id:
                model_record = model_registry.resolve_reusable_model(source_model_id)
                if not isinstance(model_record, dict):
                    raise HTTPException(status_code=404, detail="Reusable model not found")

                checkpoint_path = Path(str((model_record.get("paths") or {}).get("checkpoint") or "")).expanduser()
                if not checkpoint_path.exists():
                    raise HTTPException(status_code=400, detail="Reusable model checkpoint file is missing")
                params_payload["source_model_id"] = source_model_id
                params_payload.pop("source_run_id", None)
            else:
                source_run_dir = project_dir / "runs" / source_run_id
                if not source_run_dir.exists() or not source_run_dir.is_dir():
                    raise HTTPException(status_code=404, detail="Source run for warm-start not found")
                checkpoint_path = model_registry.find_latest_gsplat_checkpoint(source_run_dir)
                if checkpoint_path is None or not checkpoint_path.exists():
                    raise HTTPException(status_code=400, detail="Source run has no gsplat checkpoint")
                params_payload["source_run_id"] = source_run_id
                params_payload.pop("source_model_id", None)

            params_payload["start_model_mode"] = "reuse"
            params_payload["source_model_checkpoint"] = str(checkpoint_path)
        else:
            params_payload["start_model_mode"] = "scratch"
            params_payload.pop("source_model_id", None)
            params_payload.pop("source_run_id", None)
            params_payload.pop("source_model_checkpoint", None)

        # Optional sequential batch orchestration.
        try:
            run_count = int(requested_params.get("run_count") or 1)
        except Exception:
            run_count = 1
        if run_count < 1:
            run_count = 1

        warmup_at_start = bool(requested_params.get("warmup_at_start", False))

        # Restart-from-scratch is scoped to one existing selected session.
        # Never treat restart_fresh requests as batch orchestration.
        if bool(requested_params.get("restart_fresh")):
            run_count = 1

        if warmup_at_start:
            if bool(requested_params.get("resume")):
                raise HTTPException(status_code=400, detail="Warmup start does not support resume.")
            if bool(requested_params.get("restart_fresh")):
                raise HTTPException(status_code=400, detail="Warmup start does not support restart_fresh.")

            warmup_mode = str(params_payload.get("mode") or "baseline").strip().lower()
            warmup_scope = str(params_payload.get("tune_scope") or "").strip().lower()
            warmup_ai_mode = str(params_payload.get("ai_input_mode") or "").strip().lower()
            if engine != "gsplat" or warmup_mode != "modified" or warmup_scope != "core_ai_optimization" or not warmup_ai_mode:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Warmup start requires gsplat + modified + core_ai_optimization with ai_input_mode enabled."
                    ),
                )

            if resolved_worker_mode == "docker":
                running_workers = colmap.get_project_worker_container_ids(project_id)
                if running_workers:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A docker worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )
            else:
                if pipeline.is_local_project_active(project_id):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A local worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )

            warmup_total_runs = max(3, int(run_count or 0))
            status.update_status(
                project_id,
                "processing",
                progress=1,
                engine=engine,
                worker_mode=resolved_worker_mode,
                stop_requested=False,
                message=(
                    f"Warmup experiment queued: {warmup_total_runs} total runs "
                    "(base config + phased jitter)."
                ),
                error=None,
                batch_total=warmup_total_runs,
                batch_completed=0,
                batch_current_index=0,
            )

            warmup_seed_params = dict(requested_params)
            warmup_seed_params["worker_mode"] = resolved_worker_mode
            warmup_seed_params["engine"] = engine
            warmup_seed_params["mode"] = params_payload.get("mode")
            warmup_seed_params["tune_scope"] = params_payload.get("tune_scope")
            warmup_seed_params["ai_input_mode"] = params_payload.get("ai_input_mode")
            warmup_seed_params["run_count"] = warmup_total_runs
            if params_payload.get("baseline_session_id"):
                warmup_seed_params["baseline_session_id"] = params_payload.get("baseline_session_id")

            thread = threading.Thread(
                target=_run_warmup_experiment,
                args=(project_id, warmup_seed_params),
                daemon=True,
            )
            thread.start()

            return {
                "status": "warmup_processing_started",
                "run_count": warmup_total_runs,
                "phase_count": len(WARMUP_PHASE_PLAN),
            }

        # Orchestration-only flag; do not pass to worker runtime.
        params_payload.pop("warmup_at_start", None)

        if run_count > 1:
            if bool(requested_params.get("resume")):
                raise HTTPException(status_code=400, detail="Batch resume is not supported. Use Batch Continue from a fresh start.")

            jitter_factor = float(requested_params.get("run_jitter_factor") or 1.0)
            jitter_mode, jitter_factor, jitter_min, jitter_max = _resolve_jitter_settings(requested_params)
            continue_on_failure = bool(requested_params.get("continue_on_failure", True))
            batch_plan_id = f"batch_{uuid.uuid4().hex[:12]}"

            if jitter_mode == "random":
                batch_message = f"Batch queued: {run_count} runs (random jitter {jitter_min} to {jitter_max}, starts from run 2)."
            else:
                batch_message = f"Batch queued: {run_count} runs (fixed jitter={jitter_factor}, starts from run 2)."
            seed_run_raw = str(requested_params.get("run_name") or "").strip()
            seed_run_id = _sanitize_run_token(seed_run_raw)
            seed_action_note = (
                f"Using selected session '{seed_run_id}' as batch run 1."
                if seed_run_id
                else ""
            )
            if seed_action_note:
                batch_message = f"{batch_message} {seed_action_note}"

            # Reuse existing conflict checks before launching the batch orchestrator.
            if resolved_worker_mode == "docker":
                running_workers = colmap.get_project_worker_container_ids(project_id)
                if running_workers:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A docker worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )
            else:
                if pipeline.is_local_project_active(project_id):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "A local worker is already running for this project. "
                            "Stop it first or wait for it to finish."
                        ),
                    )

            status.update_status(
                project_id,
                "processing",
                progress=1,
                engine=engine,
                worker_mode=resolved_worker_mode,
                stop_requested=False,
                message=batch_message,
                error=None,
                batch_total=run_count,
                batch_completed=0,
                batch_current_index=0,
            )

            batch_seed_params = dict(requested_params)
            batch_seed_params["run_count"] = 1
            batch_seed_params["run_jitter_factor"] = jitter_factor
            batch_seed_params["run_jitter_mode"] = jitter_mode
            batch_seed_params["run_jitter_min"] = jitter_min
            batch_seed_params["run_jitter_max"] = jitter_max
            batch_seed_params["continue_on_failure"] = continue_on_failure
            batch_seed_params["batch_plan_id"] = batch_plan_id
            batch_seed_params["batch_total"] = int(run_count)
            batch_seed_params["batch_continue_on_failure"] = bool(continue_on_failure)
            batch_seed_params.pop("run_name_prefix", None)
            batch_seed_params.pop("batch_run_name_prefix", None)

            thread = threading.Thread(
                target=_run_batch_process,
                args=(
                    project_id,
                    batch_seed_params,
                    run_count,
                    jitter_factor,
                    jitter_mode,
                    jitter_min,
                    jitter_max,
                    continue_on_failure,
                ),
                daemon=True,
            )
            thread.start()

            logger.info("Started batch processing for %s with %s runs", project_id, run_count)
            return {
                "status": "batch_processing_started",
                "batch_plan_id": batch_plan_id,
                "run_count": run_count,
                "jitter_factor": jitter_factor,
                "jitter_mode": jitter_mode,
                "jitter_min": jitter_min,
                "jitter_max": jitter_max,
                "continue_on_failure": continue_on_failure,
                "seed_action_note": seed_action_note or None,
            }

        # Create a per-run session directory under project/runs/<run_name>.
        project_status = status.get_status(project_id)
        project_label = None
        if isinstance(project_status, dict):
            project_label = project_status.get("name") or project_status.get("project_id")
        raw_run_name_requested = str(requested_params.get("run_name") or "").strip()
        restart_fresh_requested = bool(requested_params.get("restart_fresh"))
        runs_root = project_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        default_run_name = _build_default_run_name(project_label, runs_root)
        if restart_fresh_requested and raw_run_name_requested and (runs_root / raw_run_name_requested).exists():
            run_name_requested = raw_run_name_requested
        else:
            run_name_requested = _rewrite_auto_run_name_prefix(raw_run_name_requested, project_id, project_label)
        requested_existing_run = runs_root / run_name_requested if run_name_requested else None
        if (
            requested_existing_run
            and requested_existing_run.exists()
            and requested_existing_run.is_dir()
        ):
            # If the client explicitly targets an existing session, keep using it.
            # This avoids accidental auto-cloning like "session3_01" when processing "session3".
            run_id = run_name_requested
        else:
            run_id = _resolve_unique_run_id(runs_root, run_name_requested or default_run_name)
        run_dir = project_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if restart_fresh_requested:
            stage_for_cleanup = str(params_payload.get("stage") or "train_only")
            base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
            is_base_run = bool(base_session_id) and run_id == str(base_session_id)
            logger.info("Restart requested from scratch for %s/%s; clearing previous artifacts", project_id, run_id)
            _clear_restart_artifacts(
                project_dir,
                run_dir,
                stage=stage_for_cleanup,
                clear_project_level=is_base_run,
            )

        params_payload["run_id"] = run_id
        params_payload["run_name"] = run_id

        # If no base session is defined yet, the first run becomes the base session.
        if not project_status.get("base_session_id"):
            try:
                status.update_base_session_id(project_id, run_id)
            except Exception as exc:
                logger.warning("Failed to set base session for %s: %s", project_id, exc)

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        is_base_run = bool(base_session_id) and run_id == str(base_session_id)

        # Canonical shared config is project-level and base-owned.
        # Non-base runs always inherit shared values from this source.
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else run_id)
        shared_doc["base_run_id"] = str(base_session_id) if base_session_id else run_id
        active_shared = shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else None
        inherited_shared = active_shared if active_shared else (shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {})

        incoming_shared = _extract_shared_config_from_params(params_payload)
        requested_stage = str(params_payload.get("stage") or "train_only")
        stage_includes_colmap = requested_stage in {"full", "colmap_only"}
        if is_base_run:
            if stage_includes_colmap and incoming_shared and incoming_shared != shared_doc.get("shared"):
                shared_doc["shared"] = incoming_shared
                shared_doc["version"] = int(shared_doc.get("version") or 1) + 1
                shared_doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
                try:
                    _write_project_shared_config(project_dir, shared_doc)
                except Exception as exc:
                    logger.warning("Failed to persist shared config for %s: %s", project_id, exc)
            elif not _get_project_shared_config_path(project_dir).exists():
                # Persist initial shared doc even if unchanged so non-base runs can inherit reliably.
                shared_doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
                try:
                    _write_project_shared_config(project_dir, shared_doc)
                except Exception as exc:
                    logger.warning("Failed to persist initial shared config for %s: %s", project_id, exc)
            # When base run does not include COLMAP this launch, keep using active shared values.
            if not stage_includes_colmap:
                params_payload = _merge_shared_config_into_params(params_payload, inherited_shared)
        else:
            # Ignore shared overrides coming from non-base runs.
            params_payload = _merge_shared_config_into_params(params_payload, inherited_shared)

        effective_shared_for_run = _extract_shared_config_from_params(params_payload)
        effective_shared_version = (
            int(shared_doc.get("version") or 1)
            if is_base_run and stage_includes_colmap
            else int(shared_doc.get("active_sparse_version") or shared_doc.get("version") or 1)
        )
        params_payload["shared_config_version"] = effective_shared_version
        params_payload["shared_base_run_id"] = shared_doc.get("base_run_id")

        # Persist run configuration for reproducibility (requested + resolved params).
        try:
            run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_config_payload = {
                "project_id": project_id,
                "run_id": run_id,
                "run_name": run_id,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "requested_params": requested_params,
                "resolved_params": params_payload,
                "shared_config_version": effective_shared_version,
                "shared_base_run_id": shared_doc.get("base_run_id"),
                "shared_config_snapshot": effective_shared_for_run,
            }

            run_config_latest = project_dir / "run_config.json"
            run_configs_dir = project_dir / "run_configs"
            run_configs_dir.mkdir(parents=True, exist_ok=True)
            run_config_versioned = run_configs_dir / f"run_config_{run_timestamp}.json"
            run_config_session = run_dir / "run_config.json"

            for target_path in (run_config_latest, run_config_versioned, run_config_session):
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(run_config_payload, handle, indent=2)
                tmp_path.replace(target_path)

            _prune_run_config_history(run_configs_dir)

            logger.info("Saved run configuration: %s", run_config_latest)
        except Exception as exc:
            logger.warning("Failed to persist run configuration for %s: %s", project_id, exc)

        # Prevent overlapping runs for the same project.
        if resolved_worker_mode == "docker":
            running_workers = colmap.get_project_worker_container_ids(project_id)
            if running_workers:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A docker worker is already running for this project. "
                        "Stop it first or wait for it to finish."
                    ),
                )
        else:
            if pipeline.is_local_project_active(project_id):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A local worker is already running for this project. "
                        "Stop it first or wait for it to finish."
                    ),
                )

        batch_total_value = int(params_payload.get("batch_total") or 1)
        if batch_total_value < 1:
            batch_total_value = 1

        batch_current_index_value = int(params_payload.get("batch_index") or 1)
        if batch_current_index_value < 1:
            batch_current_index_value = 1
        if batch_current_index_value > batch_total_value:
            batch_current_index_value = batch_total_value

        batch_completed_value = int(params_payload.get("batch_completed") or 0)
        if batch_completed_value < 0:
            batch_completed_value = 0
        if batch_completed_value > batch_total_value:
            batch_completed_value = batch_total_value

        # Update status to processing with the resolved engine
        status.update_status(
            project_id,
            "processing",
            progress=5,
            engine=engine,
            worker_mode=resolved_worker_mode,
            current_run_id=run_id,
            stop_requested=False,
            message=f"Processing started ({resolved_worker_mode} mode).",
            error=None,
            batch_total=batch_total_value,
            batch_completed=batch_completed_value,
            batch_current_index=batch_current_index_value,
        )

        # Start processing in background thread
        # Pass optional parameters to pipeline
        thread = threading.Thread(
            target=pipeline.run_full_pipeline,
            args=(project_id, params_payload),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started processing for project: {project_id}")
        return {"status": "processing_started", "run_id": run_id, "run_name": run_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        status.update_status(project_id, "failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start processing")


@router.get("/{project_id}/status", response_model=StatusResponse)
def get_status_endpoint(project_id: str):
    """Get project status."""
    try:
        from bimba3d_backend.app.services.resume import can_resume_project
        
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_status = status.get_status(project_id)
        
        if project_status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Status not found")
        
        # Add resume info
        resume_info = can_resume_project(project_id)
        project_status["can_resume"] = resume_info["can_resume"]
        project_status["last_completed_step"] = resume_info.get("last_checkpoint_step")
        
        return StatusResponse(**project_status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@router.get("/{project_id}/telemetry")
def get_project_telemetry(
    project_id: str,
    run_id: Optional[str] = Query(default=None),
    log_limit: int = Query(default=150, ge=20, le=5000),
    eval_limit: int = Query(default=20, ge=1, le=100),
    from_start: int = Query(default=0, ge=0, le=1),
):
    """Return lightweight telemetry snapshots for the Process tab modal.
    
    Args:
        from_start: 0 (default) = read from end/tail (quick status), 1 = read from beginning (complete logs).
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        project_status = status.get_status(project_id)
        project_name = project_status.get("name") if isinstance(project_status, dict) else None
        if not isinstance(project_name, str) or not project_name.strip():
            project_name = project_id
        resolved_run_id = (run_id or project_status.get("current_run_id") or "").strip()
        if not resolved_run_id:
            return {
                "project_id": project_id,
                "project_name": project_name,
                "run_id": None,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "training_rows": [],
                "event_rows": [],
                "eval_rows": [],
                "latest_eval": None,
                "training_summary": _build_training_summary([]),
                "run_config": None,
                "effective_shared_config": None,
                "shared_config_version": None,
                "active_sparse_shared_version": None,
                "run_shared_config_version": None,
                "shared_outdated": None,
                "base_session_id": None,
                "ai_insights": None,
                "status": {
                    "stage": project_status.get("stage"),
                    "message": project_status.get("message"),
                    "currentStep": project_status.get("currentStep"),
                    "maxSteps": project_status.get("maxSteps"),
                    "current_loss": project_status.get("current_loss"),
                },
            }

        run_dir = project_dir / "runs" / resolved_run_id
        run_log_path = run_dir / "processing.log"
        stats_dir = run_dir / "outputs" / "engines" / "gsplat" / "stats"
        run_config_path = run_dir / "run_config.json"
        run_analytics = _read_run_analytics(run_dir)

        text_lines = _read_text_lines(run_log_path, max_lines=log_limit, from_start=bool(from_start))
        training_rows = _extract_training_rows(text_lines, row_limit=log_limit, from_start=bool(from_start))
        event_rows = _extract_event_rows(text_lines, row_limit=max(10, min(100, log_limit)))
        eval_rows = _extract_eval_rows(stats_dir, eval_limit=eval_limit)
        training_summary = _build_training_summary(training_rows)
        run_config = _read_json_if_exists(run_config_path)
        if not isinstance(run_config, dict):
            run_config = None
        ai_insights = _extract_ai_run_insights(run_dir, run_config)
        if not isinstance(run_analytics, dict):
            run_analytics = _ensure_run_analytics(
                run_dir=run_dir,
                run_config=run_config if isinstance(run_config, dict) else None,
                ai_insights=ai_insights if isinstance(ai_insights, dict) else None,
            )
        if isinstance(run_analytics, dict):
            metrics = run_analytics.get("metrics") if isinstance(run_analytics.get("metrics"), dict) else {}
            if isinstance(metrics.get("best_loss"), (int, float)):
                training_summary["best_loss"] = float(metrics.get("best_loss"))
            if isinstance(metrics.get("best_loss_step"), (int, float)):
                training_summary["best_loss_step"] = int(metrics.get("best_loss_step"))

            ai_block = run_analytics.get("ai") if isinstance(run_analytics.get("ai"), dict) else {}
            canonical_ai = ai_block.get("input_mode_insights") if isinstance(ai_block.get("input_mode_insights"), dict) else None
            if isinstance(canonical_ai, dict) and canonical_ai.get("ai_input_mode"):
                ai_insights = canonical_ai

        project_base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(project_base_session_id) if project_base_session_id else None)
        current_shared_version = int(shared_doc.get("version") or 1)
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None

        run_shared_version = None
        shared_outdated = None
        effective_shared = None
        if isinstance(run_config, dict):
            raw_run_shared_version = run_config.get("shared_config_version")
            if isinstance(raw_run_shared_version, int):
                run_shared_version = raw_run_shared_version

            run_shared_snapshot = run_config.get("shared_config_snapshot")
            if not isinstance(run_shared_snapshot, dict):
                run_shared_snapshot = {}

            effective_shared = shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else run_shared_snapshot
            shared_outdated = bool(
                active_sparse_version is not None
                and run_shared_version is not None
                and run_shared_version < active_sparse_version
            )

        return {
            "project_id": project_id,
            "project_name": project_name,
            "run_id": resolved_run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "training_rows": training_rows,
            "event_rows": event_rows,
            "eval_rows": eval_rows,
            "latest_eval": eval_rows[-1] if eval_rows else None,
            "training_summary": training_summary,
            "run_analytics": run_analytics,
            "run_config": run_config,
            "effective_shared_config": effective_shared,
            "shared_config_version": current_shared_version,
            "active_sparse_shared_version": active_sparse_version,
            "run_shared_config_version": run_shared_version,
            "shared_outdated": shared_outdated,
            "base_session_id": project_base_session_id,
            "ai_insights": ai_insights,
            "status": {
                "stage": project_status.get("stage"),
                "message": project_status.get("message"),
                "currentStep": project_status.get("currentStep"),
                "maxSteps": project_status.get("maxSteps"),
                "current_loss": project_status.get("current_loss"),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting telemetry for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to get telemetry")


@router.get("/{project_id}/ai-learning-table")
def get_project_ai_learning_table(project_id: str):
    try:
        return _build_project_ai_learning_table(project_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error building AI learning table for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to build AI learning table")


@router.post("/{project_id}/stop")
def request_stop(project_id: str):
    """Signal an in-flight job to stop gracefully and export final artifacts."""
    try:
        project_dir = DATA_DIR / project_id

        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        stop_flag = project_dir / "stop_requested"
        stop_flag.write_text("stop")

        # Also force-stop docker worker container if one is still running.
        try:
            stopped = colmap.stop_project_worker_containers(project_id)
            if stopped:
                logger.info("Stopped %d active worker container(s) for project %s", stopped, project_id)
        except Exception as exc:
            logger.warning("Failed to force-stop worker container for %s: %s", project_id, exc)

        current_status = status.get_status(project_id)
        worker_mode = resolve_worker_mode(str(current_status.get("worker_mode") or ""))
        is_active = False
        try:
            if worker_mode == "docker":
                is_active = bool(colmap.get_project_worker_container_ids(project_id))
            else:
                is_active = bool(pipeline.is_local_project_active(project_id))
        except Exception as exc:
            logger.warning("Failed to probe active worker while stopping %s: %s", project_id, exc)

        if is_active:
            # Worker is still shutting down; keep transient stopping state.
            status.update_status(
                project_id,
                "stopping",
                progress=current_status.get("progress", 0),
                stop_requested=True,
                message="Will stop after current step completes...",
            )
        else:
            # No worker remains; finalize stop immediately so frontend exits processing state.
            status.update_status(
                project_id,
                "stopped",
                progress=current_status.get("progress", 0),
                current_run_id=None,
                stop_requested=True,
                message="Stopped by user.",
            )

        logger.info(f"Stop requested for project: {project_id}")
        return {"status": "stop_requested"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting stop: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request stop")


@router.get("/{project_id}/runs")
def list_project_runs(project_id: str):
    """List per-project run sessions with key metadata for UI run selection."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        base_colmap_ready = _base_session_colmap_ready(project_dir, base_session_id)
        can_create_session = bool(base_colmap_ready)
        can_create_session_reason = None if can_create_session else "Complete COLMAP on the base session before creating new sessions."

        if not runs_root.exists():
            return {
                "runs": [],
                "base_session_id": base_session_id,
                "base_colmap_ready": base_colmap_ready,
                "can_create_session": can_create_session,
                "can_create_session_reason": can_create_session_reason,
            }

        project_has_completed_outputs = any(
            p.exists()
            for p in (
                project_dir / "outputs" / "engines" / "gsplat" / "splats.splat",
                project_dir / "outputs" / "engines" / "gsplat" / "splats.ply",
                project_dir / "outputs" / "engines" / "gsplat" / "metadata.json",
                project_dir / "outputs" / "engines" / "litegs" / "splats.splat",
                project_dir / "outputs" / "engines" / "litegs" / "splats.ply",
                project_dir / "outputs" / "engines" / "litegs" / "metadata.json",
            )
        )
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        current_shared_version = int(shared_doc.get("version") or 1)
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None

        runs: list[dict] = []
        for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
            run_id = run_dir.name
            run_config_path = run_dir / "run_config.json"
            run_config = _read_json_if_exists(run_config_path)
            saved_at = run_config.get("saved_at") if isinstance(run_config, dict) else None
            resolved = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
            requested = run_config.get("requested_params") if isinstance(run_config, dict) and isinstance(run_config.get("requested_params"), dict) else {}
            has_completed_outputs = any(
                p.exists()
                for p in (
                    run_dir / "outputs" / "engines" / "gsplat" / "splats.splat",
                    run_dir / "outputs" / "engines" / "gsplat" / "splats.ply",
                    run_dir / "outputs" / "engines" / "gsplat" / "metadata.json",
                    run_dir / "outputs" / "engines" / "litegs" / "splats.splat",
                    run_dir / "outputs" / "engines" / "litegs" / "splats.ply",
                    run_dir / "outputs" / "engines" / "litegs" / "metadata.json",
                )
            )
            has_comparison_summary = (run_dir / "comparison" / "experiment_summary.json").exists()
            is_base_run = run_id == base_session_id
            is_completed = has_completed_outputs or has_comparison_summary or (is_base_run and project_has_completed_outputs)
            run_shared_version = run_config.get("shared_config_version") if isinstance(run_config, dict) else None
            if not isinstance(run_shared_version, int):
                run_shared_version = None
            shared_outdated = bool(
                active_sparse_version is not None
                and run_shared_version is not None
                and run_shared_version < active_sparse_version
            )

            adaptive_runs_dir = run_dir / "adaptive_ai" / "runs"
            adaptive_events = 0
            try:
                adaptive_logs = sorted(adaptive_runs_dir.glob("*.jsonl")) if adaptive_runs_dir.is_dir() else []
                for log_path in adaptive_logs:
                    try:
                        with log_path.open("r", encoding="utf-8") as f:
                            adaptive_events += sum(1 for _ in f)
                    except Exception:
                        continue

                run_log_path = run_dir / "processing.log"
                if run_log_path.exists():
                    try:
                        with run_log_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                if CORE_AI_DECISION_RE.search(line) or "AI_INPUT_MODE_" in line:
                                    adaptive_events += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Skipping adaptive logs for run %s due to access error: %s", run_id, exc)

            runs.append(
                {
                    "run_id": run_id,
                    "run_name": (
                        (run_config.get("run_name") if isinstance(run_config, dict) else None)
                        or (resolved.get("run_name") if isinstance(resolved, dict) else None)
                        or (requested.get("run_name") if isinstance(requested, dict) else None)
                        or run_id
                    ),
                    "saved_at": saved_at,
                    "mode": resolved.get("mode") or requested.get("mode"),
                    "stage": resolved.get("stage") or requested.get("stage"),
                    "engine": resolved.get("engine") or requested.get("engine"),
                    "max_steps": resolved.get("max_steps") or requested.get("max_steps"),
                    "tune_scope": resolved.get("tune_scope") or requested.get("tune_scope"),
                    "trend_scope": resolved.get("trend_scope") or requested.get("trend_scope"),
                    "adaptive_event_count": adaptive_events,
                    "has_run_config": run_config_path.exists(),
                    "has_run_log": (run_dir / "processing.log").exists(),
                    "session_status": "completed" if is_completed else "pending",
                    "is_base": is_base_run,
                    "shared_config_version": run_shared_version,
                    "active_sparse_shared_version": active_sparse_version,
                    "shared_outdated": shared_outdated,
                    "batch_plan_id": resolved.get("batch_plan_id") or requested.get("batch_plan_id"),
                    "batch_index": resolved.get("batch_index") or requested.get("batch_index"),
                    "batch_total": resolved.get("batch_total") or requested.get("batch_total"),
                }
            )

        # Fallback: if base session is missing/deleted, promote latest run as base.
        if runs and (not base_session_id or not any(r["run_id"] == base_session_id for r in runs)):
            fallback_base = runs[0]["run_id"]
            try:
                status.update_base_session_id(project_id, fallback_base)
                base_session_id = fallback_base
                shared_doc = _read_project_shared_config(project_dir, fallback_base)
                shared_doc["base_run_id"] = fallback_base
                _write_project_shared_config(project_dir, shared_doc)
                for item in runs:
                    item["is_base"] = item["run_id"] == base_session_id
            except Exception as exc:
                logger.warning("Failed to update fallback base session for %s: %s", project_id, exc)

        base_colmap_ready = _base_session_colmap_ready(project_dir, base_session_id)
        can_create_session = bool(base_colmap_ready)
        can_create_session_reason = None if can_create_session else "Complete COLMAP on the base session before creating new sessions."

        return {
            "runs": runs,
            "base_session_id": base_session_id,
            "base_colmap_ready": base_colmap_ready,
            "can_create_session": can_create_session,
            "can_create_session_reason": can_create_session_reason,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listing runs for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to list project runs")


@router.get("/{project_id}/runs/{run_id}/config")
def get_project_run_config(project_id: str, run_id: str):
    """Return the persisted run_config payload for a given run session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_config_path = project_dir / "runs" / run_id / "run_config.json"
        run_config = _read_json_if_exists(run_config_path)
        if not isinstance(run_config, dict):
            raise HTTPException(status_code=404, detail="Run config not found")

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        current_shared_version = int(shared_doc.get("version") or 1)
        active_sparse_version = shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None

        run_shared_version = run_config.get("shared_config_version")
        if not isinstance(run_shared_version, int):
            run_shared_version = None

        run_shared_snapshot = run_config.get("shared_config_snapshot")
        if not isinstance(run_shared_snapshot, dict):
            run_shared_snapshot = {}

        effective_shared = shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else run_shared_snapshot
        is_base_run = bool(base_session_id) and run_id == str(base_session_id)
        shared_outdated = bool(
            active_sparse_version is not None
            and run_shared_version is not None
            and run_shared_version < active_sparse_version
        )

        return {
            "project_id": project_id,
            "run_id": run_id,
            "run_config": run_config,
            "effective_shared_config": effective_shared,
            "shared_config_version": current_shared_version,
            "active_sparse_shared_version": active_sparse_version,
            "run_shared_config_version": run_shared_version,
            "shared_outdated": shared_outdated,
            "base_session_id": base_session_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading run config for %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read run config")


@router.patch("/{project_id}/runs/{run_id}/config")
def update_project_run_config(
    project_id: str,
    run_id: str,
    payload: UpdateRunConfigRequest = Body(...),
):
    """Persist run-level training config for a specific session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_dir = project_dir / "runs" / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        run_config_path = run_dir / "run_config.json"
        existing = _read_json_if_exists(run_config_path)
        run_config = existing if isinstance(existing, dict) else {}

        existing_requested = run_config.get("requested_params") if isinstance(run_config.get("requested_params"), dict) else {}
        existing_resolved = run_config.get("resolved_params") if isinstance(run_config.get("resolved_params"), dict) else {}

        incoming_requested = payload.requested_params if isinstance(payload.requested_params, dict) else {}
        incoming_resolved = payload.resolved_params if isinstance(payload.resolved_params, dict) else {}

        requested_params = {**existing_requested, **json.loads(json.dumps(incoming_requested))}
        resolved_params = {**existing_resolved, **json.loads(json.dumps(incoming_resolved))}

        requested_params["run_id"] = run_id
        requested_params["run_name"] = run_id
        resolved_params["run_id"] = run_id
        resolved_params["run_name"] = run_id

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)

        saved_at = datetime.utcnow().isoformat() + "Z"
        run_config_payload = {
            "project_id": project_id,
            "run_id": run_id,
            "run_name": run_id,
            "saved_at": saved_at,
            "requested_params": requested_params,
            "resolved_params": resolved_params,
            "shared_config_version": run_config.get("shared_config_version")
            if isinstance(run_config.get("shared_config_version"), int)
            else int(shared_doc.get("active_sparse_version") or shared_doc.get("version") or 1),
            "shared_base_run_id": run_config.get("shared_base_run_id")
            if isinstance(run_config.get("shared_base_run_id"), str)
            else (shared_doc.get("base_run_id") if isinstance(shared_doc.get("base_run_id"), str) else base_session_id),
            "shared_config_snapshot": run_config.get("shared_config_snapshot")
            if isinstance(run_config.get("shared_config_snapshot"), dict)
            else {},
        }

        run_config_tmp = run_config_path.with_suffix(run_config_path.suffix + ".tmp")
        with open(run_config_tmp, "w", encoding="utf-8") as handle:
            json.dump(run_config_payload, handle, indent=2)
        run_config_tmp.replace(run_config_path)

        latest_path = project_dir / "run_config.json"
        latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
        with open(latest_tmp, "w", encoding="utf-8") as handle:
            json.dump(run_config_payload, handle, indent=2)
        latest_tmp.replace(latest_path)

        return {
            "status": "saved",
            "project_id": project_id,
            "run_id": run_id,
            "saved_at": saved_at,
            "run_config": run_config_payload,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error saving run config for %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to save run config")


@router.post("/{project_id}/runs/{run_id}/continue-batch")
def continue_batch_from_run(
    project_id: str,
    run_id: str,
    payload: ContinueBatchRequest = Body(default=ContinueBatchRequest()),
):
    """Restart one run and continue remaining runs in its batch chain."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_dir = project_dir / "runs" / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        run_cfg = _read_json_if_exists(run_dir / "run_config.json")
        if not isinstance(run_cfg, dict):
            raise HTTPException(status_code=404, detail="Run config not found")

        requested = run_cfg.get("requested_params") if isinstance(run_cfg.get("requested_params"), dict) else {}
        resolved = run_cfg.get("resolved_params") if isinstance(run_cfg.get("resolved_params"), dict) else {}

        raw_total = resolved.get("batch_total", requested.get("batch_total"))
        raw_index = resolved.get("batch_index", requested.get("batch_index"))
        try:
            batch_total = int(raw_total)
            batch_index = int(raw_index)
        except Exception:
            raise HTTPException(status_code=409, detail="Selected run is not part of a resumable batch chain.")

        if batch_total < 2 or batch_index < 1 or batch_index > batch_total:
            raise HTTPException(status_code=409, detail="Selected run is not part of a valid batch chain.")

        continue_on_failure = bool(
            resolved.get("batch_continue_on_failure", requested.get("batch_continue_on_failure", resolved.get("continue_on_failure", requested.get("continue_on_failure", True))))
        )
        batch_connect_runs = bool(resolved.get("batch_connect_runs", requested.get("batch_connect_runs", True)))
        jitter_mode, jitter_factor, jitter_min, jitter_max = _resolve_jitter_settings({
            "run_jitter_mode": resolved.get("run_jitter_mode", requested.get("run_jitter_mode", "fixed")),
            "run_jitter_factor": resolved.get("run_jitter_factor", requested.get("run_jitter_factor", 1.0)),
            "run_jitter_min": resolved.get("run_jitter_min", requested.get("run_jitter_min")),
            "run_jitter_max": resolved.get("run_jitter_max", requested.get("run_jitter_max")),
        })
        batch_plan_id = str(resolved.get("batch_plan_id") or requested.get("batch_plan_id") or f"batch_{uuid.uuid4().hex[:12]}")

        restart_current = bool(payload.restart_current if payload is not None else True)
        completed_before = max(0, batch_index - 1)
        previous_success_run_id: str | None = run_id if batch_connect_runs else None

        base_params = json.loads(json.dumps(requested if requested else resolved))
        base_params["run_name"] = run_id
        base_params["run_count"] = 1
        base_params["continue_on_failure"] = continue_on_failure
        base_params["batch_connect_runs"] = batch_connect_runs
        base_params["run_jitter_factor"] = jitter_factor
        base_params["run_jitter_mode"] = jitter_mode
        base_params["run_jitter_min"] = jitter_min
        base_params["run_jitter_max"] = jitter_max
        base_params["batch_plan_id"] = batch_plan_id
        base_params["batch_total"] = batch_total
        base_params["batch_index"] = batch_index
        base_params.pop("run_name_prefix", None)
        base_params.pop("batch_run_name_prefix", None)

        if restart_current:
            restart_params = dict(base_params)
            restart_params["run_name"] = run_id
            restart_params["restart_fresh"] = True
            restart_params["resume"] = False
            restart_params["run_count"] = 1
            process_project(project_id, ProcessParams(**restart_params))

            final_status = _wait_for_run_completion(project_id, run_id)
            final_state = str(final_status.get("status") or "")
            if final_state in {"completed", "done"}:
                completed_before = batch_index
                previous_success_run_id = run_id if batch_connect_runs else None
            elif final_state == "stopped":
                return {
                    "status": "batch_continue_aborted",
                    "reason": "Restarted run was stopped by user.",
                    "run_id": run_id,
                    "batch_index": batch_index,
                    "batch_total": batch_total,
                }
            elif not continue_on_failure:
                return {
                    "status": "batch_continue_aborted",
                    "reason": f"Restarted run ended with state '{final_state}'",
                    "run_id": run_id,
                    "batch_index": batch_index,
                    "batch_total": batch_total,
                }
            else:
                previous_success_run_id = None

        next_index = batch_index + 1
        if next_index > batch_total:
            return {
                "status": "batch_continue_complete",
                "message": "No remaining sessions after selected run.",
                "run_id": run_id,
                "batch_index": batch_index,
                "batch_total": batch_total,
            }

        status.update_status(
            project_id,
            "processing",
            progress=1,
            stop_requested=False,
            batch_total=batch_total,
            batch_completed=completed_before,
            batch_current_index=next_index,
            message=f"Batch continue queued from session {batch_index}; next session {next_index}/{batch_total}.",
            error=None,
        )

        thread = threading.Thread(
            target=_run_batch_process,
            args=(
                project_id,
                base_params,
                batch_total,
                jitter_factor,
                jitter_mode,
                jitter_min,
                jitter_max,
                continue_on_failure,
                next_index,
                previous_success_run_id,
                completed_before,
            ),
            daemon=True,
        )
        thread.start()

        return {
            "status": "batch_continue_started",
            "run_id": run_id,
            "batch_plan_id": batch_plan_id,
            "batch_index": batch_index,
            "batch_total": batch_total,
            "next_index": next_index,
            "restart_current": restart_current,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error continuing batch for %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to continue batch")


@router.post("/{project_id}/runs")
def create_project_run(project_id: str, payload: CreateRunRequest = Body(...)):
    """Create a new run/session directory and persist draft run config."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        if base_session_id and not _base_session_colmap_ready(project_dir, base_session_id):
            raise HTTPException(
                status_code=409,
                detail="Cannot create a new session until the base session has completed COLMAP sparse outputs.",
            )

        project_label = None
        if isinstance(project_status, dict):
            project_label = project_status.get("name") or project_status.get("project_id")

        requested_name = str(payload.run_name or "").strip()
        requested_name = _rewrite_auto_run_name_prefix(requested_name, project_id, project_label)
        preferred_name = requested_name or _build_default_run_name(project_label, runs_root)
        run_id = _resolve_unique_run_id(runs_root, preferred_name)

        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        requested_params: dict = {"run_name": run_id, "run_id": run_id}
        resolved_params: dict = {"run_name": run_id, "run_id": run_id}

        if isinstance(payload.resolved_params, dict):
            provided_params = json.loads(json.dumps(payload.resolved_params))
            requested_params.update(provided_params)
            resolved_params.update(provided_params)

        requested_params["run_name"] = run_id
        requested_params["run_id"] = run_id
        resolved_params["run_name"] = run_id
        resolved_params["run_id"] = run_id

        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)
        active_shared = shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else None
        inherited_shared = active_shared if active_shared else (shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {})
        inherited_shared_version = int(shared_doc.get("active_sparse_version") or shared_doc.get("version") or 1)

        run_config_payload = {
            "project_id": project_id,
            "run_id": run_id,
            "run_name": run_id,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "requested_params": requested_params,
            "resolved_params": resolved_params,
            "shared_config_version": inherited_shared_version,
            "shared_base_run_id": shared_doc.get("base_run_id") if isinstance(shared_doc.get("base_run_id"), str) else base_session_id,
            "shared_config_snapshot": {},
        }

        run_cfg_path = run_dir / "run_config.json"
        tmp = run_cfg_path.with_suffix(run_cfg_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(run_config_payload, f, indent=2)
        tmp.replace(run_cfg_path)

        latest_cfg_path = project_dir / "run_config.json"
        latest_tmp = latest_cfg_path.with_suffix(latest_cfg_path.suffix + ".tmp")
        with open(latest_tmp, "w", encoding="utf-8") as f:
            json.dump(run_config_payload, f, indent=2)
        latest_tmp.replace(latest_cfg_path)

        if not project_status.get("base_session_id"):
            try:
                status.update_base_session_id(project_id, run_id)
            except Exception as exc:
                logger.warning("Failed to set base session for %s: %s", project_id, exc)

        return {
            "status": "created",
            "project_id": project_id,
            "run_id": run_id,
            "run_name": run_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating run for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.patch("/{project_id}/runs/{run_id}")
def rename_project_run(project_id: str, run_id: str, payload: RenameRunRequest = Body(...)):
    """Rename a run session directory and update its run config metadata."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        source_dir = runs_root / run_id
        if not source_dir.exists() or not source_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        desired = _sanitize_run_token(payload.run_name or "")
        if not desired:
            raise HTTPException(status_code=400, detail="Run name cannot be empty")
        if desired == run_id:
            return {"status": "unchanged", "run_id": run_id, "run_name": run_id}

        current_status = status.get_status(project_id)
        if (
            isinstance(current_status, dict)
            and current_status.get("status") in {"processing", "stopping"}
            and current_status.get("current_run_id") == run_id
        ):
            raise HTTPException(status_code=409, detail="Cannot rename an active run")

        target_dir = runs_root / desired
        if target_dir.exists():
            raise HTTPException(status_code=409, detail="Run name already exists")

        source_dir.rename(target_dir)

        run_cfg_path = target_dir / "run_config.json"
        run_cfg = _read_json_if_exists(run_cfg_path)
        if isinstance(run_cfg, dict):
            run_cfg["run_id"] = desired
            run_cfg["run_name"] = desired
            requested = run_cfg.get("requested_params")
            if isinstance(requested, dict):
                requested["run_id"] = desired
                requested["run_name"] = desired
            resolved = run_cfg.get("resolved_params")
            if isinstance(resolved, dict):
                resolved["run_id"] = desired
                resolved["run_name"] = desired
            tmp = run_cfg_path.with_suffix(run_cfg_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(run_cfg, f, indent=2)
            tmp.replace(run_cfg_path)

        # Keep project-level latest config consistent when it points to this run.
        latest_cfg_path = project_dir / "run_config.json"
        latest_cfg = _read_json_if_exists(latest_cfg_path)
        if isinstance(latest_cfg, dict) and latest_cfg.get("run_id") == run_id:
            latest_cfg["run_id"] = desired
            latest_cfg["run_name"] = desired
            tmp = latest_cfg_path.with_suffix(latest_cfg_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(latest_cfg, f, indent=2)
            tmp.replace(latest_cfg_path)

        if isinstance(current_status, dict) and current_status.get("current_run_id") == run_id:
            status.update_status(project_id, current_status.get("status", "pending"), current_run_id=desired)

        if isinstance(current_status, dict) and current_status.get("base_session_id") == run_id:
            status.update_base_session_id(project_id, desired)
            try:
                shared_doc = _read_project_shared_config(project_dir, desired)
                shared_doc["base_run_id"] = desired
                _write_project_shared_config(project_dir, shared_doc)
            except Exception as exc:
                logger.warning("Failed to update shared config base run after rename for %s: %s", project_id, exc)

        return {"status": "renamed", "run_id": desired, "run_name": desired}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error renaming run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to rename run")


@router.patch("/{project_id}/runs/{run_id}/set-base")
def set_base_project_run(project_id: str, run_id: str):
    """Mark the selected run as the project's base session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        run_dir = project_dir / "runs" / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        status.update_base_session_id(project_id, run_id)
        try:
            shared_doc = _read_project_shared_config(project_dir, run_id)
            shared_doc["base_run_id"] = run_id
            _write_project_shared_config(project_dir, shared_doc)
        except Exception as exc:
            logger.warning("Failed to update shared config base run for %s: %s", project_id, exc)
        return {"status": "ok", "base_session_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error setting base run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to set base session")


@router.delete("/{project_id}/runs/{run_id}")
def delete_project_run(project_id: str, run_id: str):
    """Delete a completed/inactive run session and reassign base session if needed."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        runs_root = project_dir / "runs"
        target_dir = runs_root / run_id
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")

        current_status = status.get_status(project_id)
        if (
            isinstance(current_status, dict)
            and current_status.get("status") in {"processing", "stopping"}
            and current_status.get("current_run_id") == run_id
        ):
            worker_mode = str(current_status.get("worker_mode") or "")
            is_actively_running = False
            try:
                resolved_mode = resolve_worker_mode(worker_mode)
                if resolved_mode == "docker":
                    is_actively_running = bool(colmap.get_project_worker_container_ids(project_id))
                else:
                    is_actively_running = bool(pipeline.is_local_project_active(project_id))
            except Exception:
                # If we cannot confirm runtime activity, keep conservative behavior.
                is_actively_running = True

            if is_actively_running:
                raise HTTPException(status_code=409, detail="Cannot delete an active run")

            # Status says active, but no worker exists anymore; clear stale binding first.
            status.update_status(project_id, "stopped", current_run_id=None, stop_requested=True)
            current_status = status.get_status(project_id)

        try:
            _delete_path_strict(target_dir)
        except PermissionError as exc:
            logger.warning("Delete blocked by permissions for %s/%s: %s", project_id, run_id, exc)
            raise HTTPException(
                status_code=423,
                detail="Cannot delete session because files are locked or access is denied. Close open previews/File Explorer handles and retry.",
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in {5, 32}:
                logger.warning("Delete blocked by Windows file lock for %s/%s: %s", project_id, run_id, exc)
                raise HTTPException(
                    status_code=423,
                    detail="Cannot delete session because files are locked by another process. Close apps using the session files and retry.",
                )
            raise

        base_session_id = current_status.get("base_session_id") if isinstance(current_status, dict) else None
        deleted_was_base = base_session_id == run_id
        new_base_session_id = base_session_id
        remaining = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True) if runs_root.exists() else []
        if deleted_was_base:
            new_base_session_id = remaining[0].name if remaining else None
            status.update_base_session_id(project_id, new_base_session_id)
            try:
                shared_doc = _read_project_shared_config(project_dir, new_base_session_id)
                shared_doc["base_run_id"] = new_base_session_id
                _write_project_shared_config(project_dir, shared_doc)
            except Exception as exc:
                logger.warning("Failed to update shared config base run after delete for %s: %s", project_id, exc)

        # Keep shared COLMAP/images artifacts on session deletion.
        # Shared data should only be refreshed via explicit restart stage selections.

        if isinstance(current_status, dict) and current_status.get("current_run_id") == run_id:
            status.update_status(project_id, current_status.get("status", "pending"), current_run_id=None)

        return {"status": "deleted", "run_id": run_id, "base_session_id": new_base_session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting run %s/%s: %s", project_id, run_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete run")


@router.get("/{project_id}/shared-config")
def get_project_shared_config(project_id: str):
    """Return project-level shared config anchored to the base session."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        shared_doc = _read_project_shared_config(project_dir, str(base_session_id) if base_session_id else None)

        return {
            "project_id": project_id,
            "base_session_id": base_session_id,
            "version": int(shared_doc.get("version") or 1),
            "updated_at": shared_doc.get("updated_at"),
            "active_sparse_version": shared_doc.get("active_sparse_version") if isinstance(shared_doc.get("active_sparse_version"), int) else None,
            "active_sparse_updated_at": shared_doc.get("active_sparse_updated_at"),
            "active_shared": shared_doc.get("active_shared") if isinstance(shared_doc.get("active_shared"), dict) else {},
            "shared": shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading shared config for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read shared config")


@router.patch("/{project_id}/shared-config")
def update_project_shared_config(project_id: str, payload: UpdateSharedConfigRequest = Body(...)):
    """Persist base-owned shared config (images/COLMAP) for the project."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        project_status = status.get_status(project_id)
        base_session_id = project_status.get("base_session_id") if isinstance(project_status, dict) else None
        if not isinstance(base_session_id, str) or not base_session_id:
            raise HTTPException(status_code=409, detail="Base session not set")

        if isinstance(payload.run_id, str) and payload.run_id and payload.run_id != base_session_id:
            raise HTTPException(status_code=409, detail="Shared config can only be saved from the base session")

        if not isinstance(payload.shared, dict):
            raise HTTPException(status_code=400, detail="Invalid shared config payload")

        shared_doc = _read_project_shared_config(project_dir, base_session_id)
        current_shared = shared_doc.get("shared") if isinstance(shared_doc.get("shared"), dict) else {}
        merged_shared = {**current_shared, **json.loads(json.dumps(payload.shared))}

        shared_doc["base_run_id"] = base_session_id
        shared_doc["shared"] = merged_shared
        shared_doc["version"] = int(shared_doc.get("version") or 1) + 1
        shared_doc["updated_at"] = datetime.utcnow().isoformat() + "Z"

        _write_project_shared_config(project_dir, shared_doc)

        return {
            "status": "saved",
            "project_id": project_id,
            "base_session_id": base_session_id,
            "version": shared_doc.get("version"),
            "updated_at": shared_doc.get("updated_at"),
            "shared": merged_shared,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error saving shared config for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to save shared config")


@router.get("/{project_id}/files")
def get_files(project_id: str, run_id: str | None = Query(None)):
    """Get list of output files for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        # Verify project exists
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        requested_run_id = (run_id or "").strip() or None
        if requested_run_id:
            run_dir = project_dir / "runs" / requested_run_id
            if not run_dir.exists() or not run_dir.is_dir():
                raise HTTPException(status_code=404, detail="Run not found")

        output_files = files.get_output_files(project_id, run_id=requested_run_id)
        if requested_run_id:
            shared_files = files.get_output_files(project_id, run_id=None)
            # Uploaded images are shared across sessions.
            if "images" not in output_files and "images" in shared_files:
                output_files["images"] = shared_files["images"]

            # COLMAP sparse is shared across sessions; expose it for all run views.
            if "sparse" not in output_files and "sparse" in shared_files:
                output_files["sparse"] = shared_files["sparse"]
        return {"project_id": project_id, "files": output_files}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get files")


@router.get("/{project_id}/previews/{filename}")
def get_preview_image(
    project_id: str,
    filename: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """Serve a specific preview PNG from outputs/previews (optionally engine-scoped)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if previews_dir is None:
            missing_engine = sanitized_engine or inferred_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        # Prevent path traversal
        resolved_dir = previews_dir.resolve()
        resolved_img = img_path.resolve()
        if resolved_dir not in resolved_img.parents:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            resolved_img,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview image for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preview image")


@router.head("/{project_id}/previews/{filename}")
def head_preview_image(
    project_id: str,
    filename: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """HEAD probe for preview PNG (used by browsers for preflight)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        previews_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "previews",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if previews_dir is None:
            missing_engine = sanitized_engine or inferred_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        img_path = previews_dir / filename

        if not img_path.exists() or img_path.suffix.lower() != ".png":
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Preview not found"
            if missing_engine:
                detail = f"Preview not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        resolved_dir = previews_dir.resolve()
        resolved_img = img_path.resolve()
        if resolved_dir not in resolved_img.parents:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(resolved_img, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HEAD preview for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to head preview image")


@router.get("/{project_id}/images/locations")
def get_image_locations(project_id: str):
    """Return extracted GPS locations for project images, if available."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        gps_file = project_dir / "images" / "locations.json"
        if not gps_file.exists():
            return {"project_id": project_id, "locations": []}

        try:
            data = json.loads(gps_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to read GPS metadata for {project_id}: {e}")
            data = {}

        locations = [
            {"name": name, "lat": coords.get("lat"), "lon": coords.get("lon")}
            for name, coords in data.items()
            if coords and "lat" in coords and "lon" in coords
        ]

        return {"project_id": project_id, "locations": locations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image locations for {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get image locations")


@router.get("/{project_id}/preview")
def get_preview(project_id: str):
    """Get latest preview PNG."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        preview_file, _, _, _ = _find_existing_path(
            project_id,
            Path("previews") / "preview_latest.png",
            None,
        )
        if not preview_file:
            raise HTTPException(status_code=404, detail="Preview not available")

        return FileResponse(
            preview_file,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get preview")


@router.get("/{project_id}/logs")
def get_logs(project_id: str, lines: int = 500):
    """Get processing logs for a project."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        log_file = project_dir / "processing.log"
        if not log_file.exists():
            return {"project_id": project_id, "logs": "No logs available yet."}
        
        # Read lines from the latest run block (appended logs may contain older runs).
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            latest_run_start = 0
            marker = "Initialized project log file:"
            for idx, line in enumerate(all_lines):
                if marker in line:
                    latest_run_start = idx

            scoped_lines = all_lines[latest_run_start:] if latest_run_start < len(all_lines) else all_lines
            recent_lines = scoped_lines[-lines:] if len(scoped_lines) > lines else scoped_lines
            log_content = "".join(recent_lines)
        
        return {"project_id": project_id, "logs": log_content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get logs")


@router.get("/{project_id}/download/sparse.json")
def download_sparse_json(project_id: str):
    """Return a JSON representation of the first COLMAP sparse reconstruction (points only).

    This endpoint prefers `points3D.txt` (readable) and falls back to `points3D.bin` (best-effort parser).
    The returned shape is {"points": [{"x":..,"y":..,"z":..,"r":..,"g":..,"b":..}, ...]}.
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists() or not sparse_root.is_dir():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        # Pick the first reconstruction directory containing points3D
        recon_dir = None
        for d in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            if (d / "points3D.txt").exists() or (d / "points3D.bin").exists():
                recon_dir = d
                break

        if recon_dir is None:
            raise HTTPException(status_code=404, detail="No COLMAP reconstruction found")

        txt_path = recon_dir / "points3D.txt"
        bin_path = recon_dir / "points3D.bin"

        points = []

        if txt_path.exists():
            # Parse ASCII points3D.txt (format: id x y z r g b error track_length [track...])
            with open(txt_path, "r") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    try:
                        # parts[0]=id, [1..3]=xyz, [4..6]=rgb
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        x, y, z = _colmap_to_opengl_coords(x, y, z)
                        r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                        points.append({"x": x, "y": y, "z": z, "r": r, "g": g, "b": b})
                    except Exception:
                        continue
        elif bin_path.exists():
            # Best-effort binary parser. COLMAP's binary layout may vary; we attempt the common layout.
            import struct
            with open(bin_path, "rb") as f:
                try:
                    # Read number of points (uint64)
                    num_points_bytes = f.read(8)
                    if len(num_points_bytes) < 8:
                        raise ValueError("Invalid points3D.bin header")
                    num_points = struct.unpack("<Q", num_points_bytes)[0]
                except Exception:
                    # If header read fails, fall back to scanning (empty response)
                    num_points = 0

                for _ in range(num_points):
                    try:
                        pid = struct.unpack("<Q", f.read(8))[0]
                        x, y, z = struct.unpack("<ddd", f.read(24))
                        x, y, z = _colmap_to_opengl_coords(x, y, z)
                        r, g, b = struct.unpack("BBB", f.read(3))
                        error = struct.unpack("<d", f.read(8))[0]
                        track_len = struct.unpack("<Q", f.read(8))[0]
                        # skip track entries (image_id, point2d_idx) pairs
                        try:
                            f.read(track_len * 16)
                        except Exception:
                            # If sizes differ, try smaller element sizes
                            try:
                                f.read(track_len * 8)
                            except Exception:
                                pass
                        points.append({"x": x, "y": y, "z": z, "r": int(r), "g": int(g), "b": int(b)})
                    except Exception:
                        # On parse error just break to avoid infinite loop
                        break

        if not points:
            raise HTTPException(status_code=404, detail="No points parsed from COLMAP output")

        return {"points": points}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting sparse JSON for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export sparse points")










@router.get("/{project_id}/splat-format")
def get_splat_format(project_id: str, engine: str | None = Query(None)):
    """Check what splat format is available (ply or bin)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        sanitized_engine = _sanitize_engine(engine)
        search_order, inferred_engine = _engine_search_order(project_id, sanitized_engine)

        for candidate in search_order:
            ply_path = _resolve_output_path(project_dir, "splats.ply", candidate)
            bin_path = _resolve_output_path(project_dir, "splats.bin", candidate)
            if ply_path.exists():
                return {
                    "format": "ply",
                    "has_ply": True,
                    "has_bin": bin_path.exists(),
                    "engine": candidate,
                }
            if bin_path.exists():
                return {
                    "format": "bin",
                    "has_ply": ply_path.exists(),
                    "has_bin": True,
                    "engine": candidate,
                }
        return {"format": "none", "has_ply": False, "has_bin": False, "engine": sanitized_engine or inferred_engine}
    
    except Exception as e:
        logger.error(f"Error checking splat format: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check splat format")


@router.get("/{project_id}/download/splats.splat")
def download_splats_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download .splat file (optimized binary format for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        splat_path, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not splat_path:
            detail = ".splat file not found. Processing may not be complete."
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f".splat file not found for engine '{missing_engine}'."
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=splat_path,
            filename="splats.splat",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading .splat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download .splat")


@router.head("/{project_id}/download/splats.splat")
def head_splats_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """HEAD probe for .splat file (used by frontend to prefer native format)."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
        project_id,
        "splats.splat",
        engine,
        run_id=(run_id.strip() if run_id else None),
    )
    if splat_path:
        return FileResponse(path=splat_path, filename="splats.splat", media_type="application/octet-stream")
    missing_engine = sanitized_engine or inferred_engine
    detail = ".splat file not found"
    if missing_engine:
        detail = f".splat file not found for engine '{missing_engine}'"
    raise HTTPException(status_code=404, detail=detail)


@router.get("/{project_id}/download/best.splat")
def download_best_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download best model .splat file selected during training."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        best_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "best.splat",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not best_path:
            detail = "best.splat file not found. Best checkpoint may not be available yet."
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f"best.splat file not found for engine '{missing_engine}'."
            raise HTTPException(status_code=404, detail=detail)

        return FileResponse(
            path=best_path,
            filename="best.splat",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading best.splat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download best.splat")


@router.head("/{project_id}/download/best.splat")
def head_best_splat(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """HEAD probe for best.splat file."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    best_path, _, sanitized_engine, inferred_engine = _find_existing_path(
        project_id,
        "best.splat",
        engine,
        run_id=(run_id.strip() if run_id else None),
    )
    if best_path:
        return FileResponse(path=best_path, filename="best.splat", media_type="application/octet-stream")
    missing_engine = sanitized_engine or inferred_engine
    detail = "best.splat file not found"
    if missing_engine:
        detail = f"best.splat file not found for engine '{missing_engine}'"
    raise HTTPException(status_code=404, detail=detail)


@router.get("/{project_id}/download/splats.ply")
def download_splats_ply(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download PLY splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        ply_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.ply",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not ply_path:
            detail = "PLY file not found"
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f"PLY file not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=ply_path,
            filename="splats.ply",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PLY: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download PLY")


@router.get("/{project_id}/download/splats.bin")
def download_splats_bin(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download binary splats file."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        bin_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.bin",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )

        if not bin_path:
            detail = "Binary file not found"
            missing_engine = sanitized_engine or inferred_engine
            if missing_engine:
                detail = f"Binary file not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)
        
        return FileResponse(
            path=bin_path,
            filename="splats.bin",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading binary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download binary")


@router.get("/{project_id}/download/points.bin")
def download_points_bin(
    project_id: str,
    candidate: str | None = Query(None, description="best or specific sparse folder name"),
    mode: str = Query("view", regex="^(view|editable)$"),
):
    """Download compact `points.bin` generated from COLMAP reconstruction.

    The converter writes `points.bin` into each reconstruction directory (e.g. outputs/sparse/0/points.bin).
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        def try_serve(candidate_dir: Path):
            target_name = "points_editable.bin" if mode == "editable" else "points.bin"
            points_path = candidate_dir / target_name
            if not points_path.exists() and mode == "editable":
                try:
                    pointsbin.convert_colmap_recon_to_pointsbin(candidate_dir)
                    points_path = candidate_dir / target_name
                except Exception as exc:
                    logger.debug("Failed to refresh editable points for %s: %s", candidate_dir, exc)
            if points_path.exists():
                return FileResponse(path=points_path, filename=target_name, media_type="application/octet-stream")
            return None

        def resolve_relative(rel_path: str | None) -> Path | None:
            if rel_path in (None, "", ".", "root"):
                target = sparse_root
            else:
                target = sparse_root / rel_path
            try:
                resolved = target.resolve()
                base = sparse_root.resolve()
                if resolved == base or base in resolved.parents:
                    return resolved
            except Exception:
                return None
            return None

        def serve_relative(rel_path: str | None):
            target = resolve_relative(rel_path)
            if target and target.exists():
                served = try_serve(target)
                if served:
                    return served
            return None

        meta, _ = _load_sparse_metadata(sparse_root)

        preferred_rel = meta.get("relative_path") if isinstance(meta, dict) else None

        # Honor explicit candidate requests first
        if candidate:
            token = candidate.strip()
            if token:
                if token.lower() == "best" and preferred_rel:
                    served = serve_relative(preferred_rel)
                    if served:
                        return served
                else:
                    served = serve_relative(token)
                    if served:
                        return served

        # Default to best-known reconstruction
        if preferred_rel:
            served = serve_relative(preferred_rel)
            if served:
                return served

        # Fall back to first available reconstruction (original behavior)
        candidates = []
        if (sparse_root / "points.bin").exists():
            candidates.append(sparse_root)
        try:
            candidates.extend(sorted([p for p in sparse_root.iterdir() if p.is_dir()]))
        except Exception as exc:
            logger.warning("Failed to enumerate sparse directories under %s: %s", sparse_root, exc)

        for candidate_dir in candidates:
            served = try_serve(candidate_dir)
            if served:
                return served

        target_name = "points_editable.bin" if mode == "editable" else "points.bin"
        raise HTTPException(
            status_code=404,
            detail=f"{target_name} not found; reconstruction may not be converted yet",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading points.bin: {e}")
        raise HTTPException(status_code=500, detail="Failed to download points.bin")


@router.get("/{project_id}/sparse/candidates")
def list_sparse_candidates(project_id: str):
    """Return metadata about available sparse reconstructions for UI selection."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists():
        return {"candidates": [], "best_relative_path": None, "updated_at": None}

    meta, _ = _load_sparse_metadata(sparse_root)

    candidates = []
    best_rel = None
    updated_at = None
    if isinstance(meta, dict):
        best_rel = meta.get("relative_path")
        updated_at = meta.get("timestamp")
        raw_candidates = meta.get("candidates") or []
        for entry in raw_candidates:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("relative_path") or "."
            candidate_dir = _resolve_sparse_candidate_dir(sparse_root, rel_path)
            images = entry.get("images")
            points = entry.get("points")
            if images is None or points is None:
                computed_images, computed_points = _read_sparse_stats(candidate_dir)
                if images is None:
                    images = computed_images
                if points is None:
                    points = computed_points
            candidates.append(
                {
                    "relative_path": rel_path,
                    "label": entry.get("label"),
                    "images": images,
                    "points": points,
                }
            )

    # Include cached merged sparse models so users can re-select them later.
    discovered: set[str] = {
        (entry.get("relative_path") or ".")
        for entry in candidates
        if isinstance(entry, dict)
    }
    merged_root = sparse_root / "_merged"
    if merged_root.exists() and merged_root.is_dir():
        try:
            for child in sorted(p for p in merged_root.iterdir() if p.is_dir()):
                rel_path = os.path.relpath(child, sparse_root)
                if rel_path in discovered:
                    continue
                if not all((child / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
                    continue
                images, points = _read_sparse_stats(child)
                label = f"merged/{child.name}"
                candidates.append(
                    {
                        "relative_path": rel_path,
                        "label": label,
                        "images": images,
                        "points": points,
                    }
                )
                discovered.add(rel_path)
        except Exception as exc:
            logger.debug("Failed to enumerate merged sparse candidates in %s: %s", merged_root, exc)

    if not candidates:
        try:
            for child in sorted(p for p in sparse_root.iterdir() if p.is_dir() and (p / "points.bin").exists()):
                try:
                    rel_path = os.path.relpath(child, sparse_root)
                except Exception:
                    rel_path = child.name
                if rel_path in {".", ""}:
                    rel_path = "."
                images, points = _read_sparse_stats(child)
                candidates.append({
                    "relative_path": rel_path,
                    "label": Path(rel_path).name if rel_path != "." else "root",
                    "images": images,
                    "points": points,
                })
        except Exception as exc:
            logger.debug("Failed to enumerate sparse fallbacks in %s: %s", sparse_root, exc)
        if not best_rel and candidates:
            best_rel = candidates[0].get("relative_path")

    return {
        "best_relative_path": best_rel,
        "candidates": candidates,
        "updated_at": updated_at,
    }


@router.get("/{project_id}/sparse/image-membership")
def get_sparse_image_membership(project_id: str):
    """Return registered image names for each sparse candidate."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists() or not sparse_root.is_dir():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    membership_path = sparse_root / SPARSE_IMAGE_MEMBERSHIP_META
    if membership_path.exists():
        try:
            payload = json.loads(membership_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.debug("Failed to parse sparse image membership JSON at %s: %s", membership_path, exc)

    # Fallback generation when worker metadata is not present yet.
    candidates_payload = list_sparse_candidates(project_id)
    rows = []
    for entry in candidates_payload.get("candidates", []):
        if not isinstance(entry, dict):
            continue
        rel_path = entry.get("relative_path") or "."
        candidate_dir = _resolve_sparse_candidate_dir(sparse_root, rel_path)
        image_names = _read_sparse_image_names(candidate_dir)
        rows.append(
            {
                "relative_path": rel_path,
                "label": entry.get("label"),
                "images": entry.get("images") if entry.get("images") is not None else len(image_names),
                "image_names": image_names,
            }
        )

    payload = {
        "updated_at": time.time(),
        "candidates": rows,
    }
    try:
        membership_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to persist sparse image membership metadata at %s: %s", membership_path, exc)
    return payload


@router.get("/{project_id}/sparse/merge-report")
def get_sparse_merge_report(project_id: str, candidate: str | None = Query(None)):
    """Return merge metadata for a cached merged sparse candidate.

    - If `candidate` is provided, it can be either `_merged/<name>` or `<name>`.
    - If omitted, the latest merged candidate with `merge_meta.json` is returned.
    """
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    sparse_root = project_dir / "outputs" / "sparse"
    merged_root = sparse_root / "_merged"
    if not merged_root.exists() or not merged_root.is_dir():
        return {"available": False, "candidate": None, "report": None}

    target_dir: Path | None = None
    if candidate:
        token = candidate.strip()
        if token.startswith("_merged/"):
            token = token.split("/", 1)[1]
        if token:
            resolved = (merged_root / token).resolve()
            root_resolved = merged_root.resolve()
            if resolved == root_resolved or root_resolved not in resolved.parents:
                raise HTTPException(status_code=400, detail="Invalid merge candidate")
            if resolved.exists() and resolved.is_dir():
                target_dir = resolved
            else:
                raise HTTPException(status_code=404, detail="Merge candidate not found")
    else:
        candidates = [
            p for p in merged_root.iterdir()
            if p.is_dir() and (p / "merge_meta.json").exists()
        ]
        if not candidates:
            return {"available": False, "candidate": None, "report": None}
        target_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    if target_dir is None:
        return {"available": False, "candidate": None, "report": None}

    meta_path = target_dir / "merge_meta.json"
    if not meta_path.exists():
        return {
            "available": False,
            "candidate": os.path.relpath(target_dir, sparse_root),
            "report": None,
        }

    try:
        report = json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning("Failed to parse merge metadata at %s: %s", meta_path, exc)
        raise HTTPException(status_code=500, detail="Failed to parse merge metadata")

    return {
        "available": True,
        "candidate": os.path.relpath(target_dir, sparse_root),
        "report": report,
    }


@router.post("/{project_id}/sparse/merge")
def build_sparse_merge(project_id: str, payload: SparseMergeRequest | None = Body(None)):
    """Build a merged sparse model from selected candidate folders without starting training."""
    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    selections = payload.selections if payload else []
    if not isinstance(selections, list) or len(selections) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two sparse selections to merge")

    sparse_root = project_dir / "outputs" / "sparse"
    if not sparse_root.exists() or not sparse_root.is_dir():
        raise HTTPException(status_code=404, detail="Sparse outputs not found")

    image_dir = project_dir / "images"
    if not image_dir.exists() or not image_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project images not found")

    try:
        # Imported lazily to avoid importing worker internals during API startup.
        from bimba3d_backend.worker.entrypoint import _resolve_sparse_model_for_training  # pylint: disable=import-outside-toplevel

        merged_dir = _resolve_sparse_model_for_training(
            sparse_root,
            image_dir,
            "merge_selected",
            selections,
        )
    except Exception as exc:
        logger.warning("Sparse merge build failed for %s: %s", project_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    merged_dir = Path(merged_dir)
    sparse_base = sparse_root.resolve()
    merged_resolved = merged_dir.resolve()
    if merged_resolved == sparse_base or sparse_base not in merged_resolved.parents:
        raise HTTPException(status_code=500, detail="Unexpected merged model location")

    rel_path = os.path.relpath(merged_resolved, sparse_root)
    report = None
    meta_path = merged_resolved / "merge_meta.json"
    if meta_path.exists():
        try:
            report = json.loads(meta_path.read_text())
        except Exception as exc:
            logger.debug("Failed to parse merge metadata after build (%s): %s", meta_path, exc)

    return {
        "status": "ok",
        "candidate": rel_path,
        "report": report,
    }


@router.post("/{project_id}/sparse/edit")
def edit_sparse_points(project_id: str, payload: SparseEditRequest | None = Body(None)):
    if payload is None or not payload.remove_point_ids:
        raise HTTPException(status_code=400, detail="No point ids provided")

    project_dir = DATA_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    candidate_dir, candidate_rel = _resolve_sparse_candidate_for_edit(project_dir, payload.candidate)

    try:
        remove_ids = {int(pid) for pid in payload.remove_point_ids if isinstance(pid, int)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid point id provided") from None

    if not remove_ids:
        raise HTTPException(status_code=400, detail="No valid point ids provided")

    try:
        result = sparse_edit.apply_sparse_edits(
            project_dir=project_dir,
            candidate_dir=candidate_dir,
            candidate_rel=candidate_rel,
            remove_point_ids=remove_ids,
            create_backup=True if payload.create_backup is None else bool(payload.create_backup),
            reoptimize=bool(payload.reoptimize),
        )
    except sparse_edit.SparseEditError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _update_sparse_candidate_points(project_dir, candidate_rel, result.get("remaining_points"))

    return {
        **result,
        "candidate_relative_path": candidate_rel,
    }



@router.get("/{project_id}/download/splats")
def download_splats(project_id: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download splats file (.splat format optimized for web rendering)."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        splat_path, _, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "splats.splat",
            engine,
            run_id=(run_id.strip() if run_id else None),
        )
        if splat_path:
            return FileResponse(
                path=splat_path,
                filename="splats.splat",
                media_type="application/octet-stream"
            )

        ply_path, _, _, _ = _find_existing_path(project_id, "splats.ply", engine, run_id=(run_id.strip() if run_id else None))
        if ply_path:
            return FileResponse(
                path=ply_path,
                filename="splats.ply",
                media_type="application/octet-stream"
            )

        bin_path, _, _, _ = _find_existing_path(project_id, "splats.bin", engine, run_id=(run_id.strip() if run_id else None))
        if bin_path:
            return FileResponse(
                path=bin_path,
                filename="splats.bin",
                media_type="application/octet-stream"
            )
        
        detail = "Splats file not found. Processing may not be complete."
        missing_engine = sanitized_engine or inferred_engine
        if missing_engine:
            detail = f"Splats file not found for engine '{missing_engine}'."
        raise HTTPException(status_code=404, detail=detail)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading splats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download splats")


@router.get("/{project_id}/download/snapshots/{filename}")
def download_snapshot(project_id: str, filename: str, engine: str | None = Query(None), run_id: str | None = Query(None)):
    """Download a specific intermediate splat snapshot exported during training."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        snapshots_dir, resolved_engine, sanitized_engine, inferred_engine = _find_existing_path(
            project_id,
            "snapshots",
            engine,
            run_id=(run_id.strip() if run_id else None),
            expect_directory=True,
        )
        if not snapshots_dir:
            missing_engine = sanitized_engine or inferred_engine
            detail = "No snapshots available"
            if missing_engine:
                detail = f"No snapshots available for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        snapshots_root = snapshots_dir.resolve()
        snap_path = (snapshots_dir / filename).resolve()
        if snapshots_root not in snap_path.parents:
            raise HTTPException(status_code=403, detail="Access denied")
        if not snap_path.exists() or not snap_path.is_file():
            missing_engine = sanitized_engine or inferred_engine or resolved_engine
            detail = "Snapshot not found"
            if missing_engine:
                detail = f"Snapshot not found for engine '{missing_engine}'"
            raise HTTPException(status_code=404, detail=detail)

        media_type = "application/octet-stream"
        if snap_path.suffix.lower() == ".ply":
            media_type = "application/octet-stream"
        elif snap_path.suffix.lower() == ".splat":
            media_type = "application/octet-stream"

        return FileResponse(
            path=snap_path,
            filename=snap_path.name,
            media_type=media_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading snapshot for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download snapshot")


@router.get("/{project_id}/download/sparse.json")
def download_sparse_json(project_id: str):
    """Return a JSON representation of the first COLMAP sparse reconstruction (points only).

    This endpoint prefers `points3D.txt` (readable) and falls back to `points3D.bin` (best-effort parser).
    The returned shape is {"points": [{"x":..,"y":..,"z":..,"r":..,"g":..,"b":..}, ...]}.
    """
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        sparse_root = project_dir / "outputs" / "sparse"
        if not sparse_root.exists() or not sparse_root.is_dir():
            raise HTTPException(status_code=404, detail="Sparse outputs not found")

        # Pick the first reconstruction directory containing points3D
        recon_dir = None
        for d in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            if (d / "points3D.txt").exists() or (d / "points3D.bin").exists():
                recon_dir = d
                break

        if recon_dir is None:
            raise HTTPException(status_code=404, detail="No COLMAP reconstruction found")

        txt_path = recon_dir / "points3D.txt"
        bin_path = recon_dir / "points3D.bin"

        points = []

        if txt_path.exists():
            # Parse ASCII points3D.txt (format: id x y z r g b error track_length [track...])
            with open(txt_path, "r") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    try:
                        # parts[0]=id, [1..3]=xyz, [4..6]=rgb
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                        points.append({"x": x, "y": y, "z": z, "r": r, "g": g, "b": b})
                    except Exception:
                        continue
        elif bin_path.exists():
            # Best-effort binary parser. COLMAP's binary layout may vary; we attempt the common layout.
            import struct
            with open(bin_path, "rb") as f:
                try:
                    # Read number of points (uint64)
                    num_points_bytes = f.read(8)
                    if len(num_points_bytes) < 8:
                        raise ValueError("Invalid points3D.bin header")
                    num_points = struct.unpack("<Q", num_points_bytes)[0]
                except Exception:
                    # If header read fails, fall back to scanning (empty response)
                    num_points = 0

                for _ in range(num_points):
                    try:
                        pid = struct.unpack("<Q", f.read(8))[0]
                        x, y, z = struct.unpack("<ddd", f.read(24))
                        r, g, b = struct.unpack("BBB", f.read(3))
                        error = struct.unpack("<d", f.read(8))[0]
                        track_len = struct.unpack("<Q", f.read(8))[0]
                        # skip track entries (image_id, point2d_idx) pairs
                        try:
                            f.read(track_len * 16)
                        except Exception:
                            # If sizes differ, try smaller element sizes
                            try:
                                f.read(track_len * 8)
                            except Exception:
                                pass
                        points.append({"x": x, "y": y, "z": z, "r": int(r), "g": int(g), "b": int(b)})
                    except Exception:
                        # On parse error just break to avoid infinite loop
                        break

        if not points:
            raise HTTPException(status_code=404, detail="No points parsed from COLMAP output")

        return {"points": points}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting sparse JSON for {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export sparse points")


@router.get("/{project_id}/metadata")
def get_metadata(project_id: str, engine: str | None = Query(None)):
    """Get metadata.json for a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        metadata_path, _, _, _ = _find_existing_path(project_id, "metadata.json", engine)
        if not metadata_path:
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metadata")


@router.get("/{project_id}/images")
def list_images(project_id: str):
    """List all images in a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        images_dir = project_dir / "images"
        image_list = []
        
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*")):
                if img_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    image_list.append({
                        "name": img_path.name,
                        "size": img_path.stat().st_size,
                        "url": f"/projects/{project_id}/image/{img_path.name}"
                    })
        
        return {"project_id": project_id, "images": image_list}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list images")


@router.get("/{project_id}/image/{filename}")
def get_image(project_id: str, filename: str):
    """Get a specific image from a project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        image_path = project_dir / "images" / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Verify file is actually in the images directory
        if not str(image_path).startswith(str(project_dir / "images")):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(path=image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get image")


@router.get("/{project_id}/thumbnail/{filename}")
def get_thumbnail(project_id: str, filename: str):
    """Get a thumbnail for a specific image."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        thumbnail_path = project_dir / "images" / "thumbnails" / filename
        
        # If thumbnail doesn't exist, return 404 (no fallback to full image)
        if not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        # Verify file is actually in the thumbnails directory
        if not str(thumbnail_path).startswith(str(project_dir / "images" / "thumbnails")):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(path=thumbnail_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get thumbnail")


@router.delete("/{project_id}")
def delete_project(project_id: str):
    """Delete a project and all associated files."""
    try:
        project_alias = DATA_DIR / project_id

        if not project_alias.exists() and not project_alias.is_symlink():
            raise HTTPException(status_code=404, detail="Project not found")

        # Resolve physical target (custom storage root projects are aliased under DATA_DIR).
        try:
            real_project_dir = project_alias.resolve(strict=True)
        except Exception:
            real_project_dir = project_alias

        # Request any in-flight workers to stop so they don't keep writing files.
        try:
            stop_flag = project_alias / "stop_requested"
            stop_flag.write_text("stop")
        except Exception:
            pass

        try:
            colmap.stop_project_worker_containers(project_id)
        except Exception as exc:
            logger.warning("Failed to stop worker containers for %s before delete: %s", project_id, exc)

        if pipeline.is_local_project_active(project_id):
            # Give local worker a short grace period to finish cleanup.
            for _ in range(15):
                if not pipeline.is_local_project_active(project_id):
                    break
                time.sleep(0.2)

        # Best-effort release of any file handlers still pointing into project logs.
        _close_project_log_handlers(project_alias)
        if real_project_dir != project_alias:
            _close_project_log_handlers(real_project_dir)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                # Remove alias under DATA_DIR first so project disappears from listings.
                _delete_path_strict(project_alias)

                # Also remove physical target dir when alias points elsewhere.
                if real_project_dir != project_alias:
                    _delete_path_strict(real_project_dir)

                last_error = None
                break
            except Exception as exc:
                last_error = exc
                # Retry after brief backoff for transient Windows file locks.
                _close_project_log_handlers(project_alias)
                if real_project_dir != project_alias:
                    _close_project_log_handlers(real_project_dir)
                time.sleep(0.2 * (attempt + 1))

        if last_error is not None:
            raise last_error

        # Ensure stale aliases are removed if target was deleted first.
        if project_alias.exists() or project_alias.is_symlink():
            _delete_path_strict(project_alias)

        alias_exists_after = project_alias.exists() or project_alias.is_symlink()
        target_exists_after = False
        if real_project_dir != project_alias:
            target_exists_after = real_project_dir.exists() or real_project_dir.is_symlink()
        if alias_exists_after or target_exists_after:
            logger.error(
                "Project deletion incomplete for %s (alias_exists=%s target_exists=%s)",
                project_id,
                alias_exists_after,
                target_exists_after,
            )
            raise HTTPException(status_code=500, detail="Project delete incomplete")

        logger.info(f"Deleted project: {project_id}")
        return {"status": "deleted", "project_id": project_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete project")


@router.patch("/{project_id}")
def update_project(project_id: str, payload: UpdateProjectRequest = Body(...)):
    """Update project metadata (currently only name)."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        if payload.name is None:
            raise HTTPException(status_code=400, detail="Nothing to update")

        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if len(name) > 120:
            raise HTTPException(status_code=400, detail="Name too long (max 120 characters)")

        status.update_project_name(project_id, name)
        logger.info(f"Renamed project {project_id} -> {name}")
        return {"status": "updated", "project_id": project_id, "name": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update project")


@router.get("/{project_id}/metrics", response_model=EvaluationMetrics)
def get_metrics(project_id: str):
    """Get evaluation metrics for a completed project."""
    try:
        project_dir = DATA_DIR / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check engine-aware metadata
        metadata_path, _, _, _ = _find_existing_path(project_id, "metadata.json", None)
        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Extract metrics from metadata
                metrics = metadata.get("evaluation_metrics", {})
                if metrics:
                    return EvaluationMetrics(**metrics)
        
        # Fallback: check adaptive_tuning_results.json (engine-aware)
        tuning_path, _, _, _ = _find_existing_path(project_id, "adaptive_tuning_results.json", None)
        if tuning_path and tuning_path.exists():
            with open(tuning_path) as f:
                tuning_data = json.load(f)
                final_metrics = tuning_data.get("final_evaluation", {})
                if final_metrics:
                    return EvaluationMetrics(**final_metrics)
        
        # No metrics available yet
        return EvaluationMetrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.get("/{project_id}/experiment-summary")
def get_experiment_summary(
    project_id: str,
    engine: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """Return an engine-aware summary payload for comparing two runs."""
    try:
        project_dir = DATA_DIR / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")

        status_info = status.get_status(project_id)
        sanitized_engine = _sanitize_engine(engine) if engine is not None else None
        search_order, inferred_engine = _engine_search_order(project_id, sanitized_engine)
        selected_engine = search_order[0] if search_order else None
        requested_run_id = (run_id or "").strip()
        run_dir = (project_dir / "runs" / requested_run_id) if requested_run_id else None
        if requested_run_id and (run_dir is None or not run_dir.exists()):
            raise HTTPException(status_code=404, detail="Run not found")

        if requested_run_id and run_dir:
            persisted_summary_path = run_dir / "comparison" / "experiment_summary.json"
            persisted_summary = _read_json_if_exists(persisted_summary_path)
            if not isinstance(persisted_summary, dict):
                raise HTTPException(
                    status_code=404,
                    detail="Run comparison summary not found. Re-run the session to generate comparison artifacts.",
                )

            outputs = files.get_output_files(project_id, run_id=requested_run_id)
            summary_engine = persisted_summary.get("engine")
            preview_url = None
            if isinstance(outputs.get("engines"), dict) and isinstance(summary_engine, str):
                engine_bundle = outputs["engines"].get(summary_engine, {})
                previews = engine_bundle.get("previews", {}) if isinstance(engine_bundle, dict) else {}
                preview_url = previews.get("latest_url")

            response_payload = dict(persisted_summary)
            response_payload["project_id"] = project_id
            response_payload["run_id"] = requested_run_id
            if not response_payload.get("run_name"):
                response_payload["run_name"] = requested_run_id
            metrics_payload = response_payload.get("metrics") if isinstance(response_payload.get("metrics"), dict) else {}
            comparison_metadata = _read_json_if_exists(run_dir / "comparison" / "metadata.json")
            if not isinstance(comparison_metadata, dict):
                comparison_metadata = _read_json_if_exists(run_dir / "outputs" / "engines" / "gsplat" / "metadata.json")
            best_splat = comparison_metadata.get("best_splat") if isinstance(comparison_metadata, dict) and isinstance(comparison_metadata.get("best_splat"), dict) else {}
            early_stop = comparison_metadata.get("early_stop") if isinstance(comparison_metadata, dict) and isinstance(comparison_metadata.get("early_stop"), dict) else {}
            if metrics_payload.get("best_splat_step") is None and isinstance(best_splat.get("step"), (int, float)):
                metrics_payload["best_splat_step"] = int(best_splat.get("step"))
            if metrics_payload.get("best_splat_loss") is None and isinstance(best_splat.get("loss"), (int, float)):
                metrics_payload["best_splat_loss"] = float(best_splat.get("loss"))
            if metrics_payload.get("stopped_early") is None and "triggered" in early_stop:
                metrics_payload["stopped_early"] = bool(early_stop.get("triggered"))
            if metrics_payload.get("early_stop_step") is None and isinstance(early_stop.get("trigger_step"), (int, float)):
                metrics_payload["early_stop_step"] = int(early_stop.get("trigger_step"))
            response_payload["metrics"] = metrics_payload
            if response_payload.get("early_stop") is None and isinstance(early_stop, dict) and early_stop:
                response_payload["early_stop"] = early_stop

            summary_major = response_payload.get("major_params") if isinstance(response_payload.get("major_params"), dict) else {}
            run_cfg = _read_json_if_exists(run_dir / "run_config.json")
            resolved_cfg = run_cfg.get("resolved_params") if isinstance(run_cfg, dict) and isinstance(run_cfg.get("resolved_params"), dict) else {}
            requested_cfg = run_cfg.get("requested_params") if isinstance(run_cfg, dict) and isinstance(run_cfg.get("requested_params"), dict) else {}
            ai_insights = _extract_ai_run_insights(run_dir, run_cfg)
            major_keys = [
                "max_steps",
                "densify_from_iter",
                "densify_until_iter",
                "densification_interval",
                "eval_interval",
                "save_interval",
                "splat_export_interval",
                "best_splat_interval",
                "best_splat_start_step",
                "auto_early_stop",
                "early_stop_monitor_interval",
                "early_stop_decision_points",
                "early_stop_min_eval_points",
                "early_stop_min_step_ratio",
                "early_stop_monitor_min_relative_improvement",
                "early_stop_eval_min_relative_improvement",
                "early_stop_max_volatility_ratio",
                "early_stop_ema_alpha",
                "trend_scope",
                "batch_size",
            ]
            for key in major_keys:
                if summary_major.get(key) is not None:
                    continue
                if resolved_cfg.get(key) is not None:
                    summary_major[key] = resolved_cfg.get(key)
                    continue
                if requested_cfg.get(key) is not None:
                    summary_major[key] = requested_cfg.get(key)

            if summary_major:
                response_payload["major_params"] = summary_major

            if ai_insights:
                response_payload["ai_insights"] = ai_insights

            if (
                response_payload.get("eval_psnr_series") is None
                or response_payload.get("eval_ssim_series") is None
                or response_payload.get("eval_lpips_series") is None
            ):
                eval_history_for_series = _read_json_if_exists(run_dir / "comparison" / "eval_history.json")
                if not isinstance(eval_history_for_series, list):
                    eval_history_for_series = _read_json_if_exists(run_dir / "outputs" / "engines" / "gsplat" / "eval_history.json")
                eval_rows_for_series = (
                    sorted(
                        [item for item in eval_history_for_series if isinstance(item, dict)],
                        key=lambda item: float(item.get("step")) if isinstance(item.get("step"), (int, float)) else float("inf"),
                    )
                    if isinstance(eval_history_for_series, list)
                    else []
                )
                if eval_rows_for_series:
                    eval_psnr_series = []
                    eval_ssim_series = []
                    eval_lpips_series = []
                    for point in eval_rows_for_series:
                        step_value = point.get("step")
                        if not isinstance(step_value, (int, float)):
                            continue
                        step_int = int(step_value)
                        psnr_value = point.get("convergence_speed")
                        if isinstance(psnr_value, (int, float)):
                            eval_psnr_series.append({"step": step_int, "value": float(psnr_value)})
                        ssim_value = point.get("sharpness_mean")
                        if isinstance(ssim_value, (int, float)):
                            eval_ssim_series.append({"step": step_int, "value": float(ssim_value)})
                        lpips_value = point.get("lpips_mean")
                        if isinstance(lpips_value, (int, float)):
                            eval_lpips_series.append({"step": step_int, "value": float(lpips_value)})
                    if response_payload.get("eval_psnr_series") is None:
                        response_payload["eval_psnr_series"] = eval_psnr_series
                    if response_payload.get("eval_ssim_series") is None:
                        response_payload["eval_ssim_series"] = eval_ssim_series
                    if response_payload.get("eval_lpips_series") is None:
                        response_payload["eval_lpips_series"] = eval_lpips_series

            response_payload["preview_url"] = preview_url
            return response_payload

        metadata_path, resolved_engine, _, _ = _find_existing_path(
            project_id,
            "metadata.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        eval_history_path, _, _, _ = _find_existing_path(
            project_id,
            "eval_history.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        tuning_results_path, _, _, _ = _find_existing_path(
            project_id,
            "adaptive_tuning_results.json",
            selected_engine,
            run_id=requested_run_id or None,
        )
        run_config_path = (run_dir / "run_config.json") if run_dir else (project_dir / "run_config.json")

        metadata = _read_json_if_exists(metadata_path)
        eval_history_raw = _read_json_if_exists(eval_history_path)
        eval_history = eval_history_raw if isinstance(eval_history_raw, list) else []
        eval_history_sorted = sorted(
            [item for item in eval_history if isinstance(item, dict)],
            key=lambda item: float(item.get("step")) if isinstance(item.get("step"), (int, float)) else float("inf"),
        )
        tuning_results = _read_json_if_exists(tuning_results_path)
        run_config = _read_json_if_exists(run_config_path)

        resolved_run_dir = run_dir
        if resolved_run_dir is None and isinstance(run_config, dict):
            resolved_candidate = str(
                run_config.get("run_id")
                or (run_config.get("resolved_params") or {}).get("run_id")
                or ""
            ).strip()
            if resolved_candidate:
                candidate_dir = project_dir / "runs" / resolved_candidate
                if candidate_dir.exists() and candidate_dir.is_dir():
                    resolved_run_dir = candidate_dir

        latest_eval = eval_history_sorted[-1] if eval_history_sorted else {}
        first_eval = eval_history_sorted[0] if eval_history_sorted else {}

        # Prefer eval history fields, then final metadata fallback.
        metrics = {
            "convergence_speed": latest_eval.get("convergence_speed"),
            "final_loss": latest_eval.get("final_loss"),
            "lpips_mean": latest_eval.get("lpips_mean"),
            "sharpness_mean": latest_eval.get("sharpness_mean"),
            "num_gaussians": latest_eval.get("num_gaussians"),
            "total_time_seconds": None,
        }
        loss_milestones = {}
        for point in eval_history_sorted:
            for k, v in point.items():
                if isinstance(k, str) and k.startswith("loss_at_") and isinstance(v, (int, float)):
                    loss_milestones[k] = float(v)

        eval_series = []
        eval_time_series = []
        eval_psnr_series = []
        eval_ssim_series = []
        eval_lpips_series = []
        runtime_tuning_series = []
        if eval_history_sorted:
            for point in eval_history_sorted:
                step_value = point.get("step")
                loss_value = point.get("final_loss")
                if not isinstance(loss_value, (int, float)):
                    loss_value = point.get("lpips_mean")
                if isinstance(step_value, (int, float)) and isinstance(loss_value, (int, float)):
                    eval_series.append({"step": int(step_value), "loss": float(loss_value)})

                conv_speed = point.get("convergence_speed")
                if isinstance(step_value, (int, float)) and isinstance(conv_speed, (int, float)) and float(conv_speed) > 0:
                    eval_time_series.append({
                        "step": int(step_value),
                        "elapsed_seconds": float(step_value) / float(conv_speed),
                    })

                if isinstance(step_value, (int, float)) and isinstance(conv_speed, (int, float)):
                    eval_psnr_series.append({"step": int(step_value), "value": float(conv_speed)})

                ssim_value = point.get("sharpness_mean")
                if isinstance(step_value, (int, float)) and isinstance(ssim_value, (int, float)):
                    eval_ssim_series.append({"step": int(step_value), "value": float(ssim_value)})

                lpips_value = point.get("lpips_mean")
                if isinstance(step_value, (int, float)) and isinstance(lpips_value, (int, float)):
                    eval_lpips_series.append({"step": int(step_value), "value": float(lpips_value)})

        if len(eval_series) < 2 and loss_milestones:
            milestone_points = []
            for key, value in loss_milestones.items():
                try:
                    step = int(str(key).replace("loss_at_", ""))
                except Exception:
                    continue
                milestone_points.append({"step": step, "loss": float(value)})
            if milestone_points:
                eval_series = sorted(milestone_points, key=lambda p: p["step"])

        if metrics.get("final_loss") is None and eval_series:
            metrics["final_loss"] = eval_series[-1].get("loss")

        if metrics.get("total_time_seconds") is None and eval_time_series:
            try:
                metrics["total_time_seconds"] = max(
                    float(item.get("elapsed_seconds"))
                    for item in eval_time_series
                    if isinstance(item, dict) and isinstance(item.get("elapsed_seconds"), (int, float))
                )
            except ValueError:
                pass

        if metrics.get("total_time_seconds") is None and not requested_run_id:
            timing = status_info.get("timing") if isinstance(status_info, dict) else None
            if isinstance(timing, dict):
                elapsed = timing.get("elapsed")
                if isinstance(elapsed, (int, float)) and float(elapsed) >= 0:
                    metrics["total_time_seconds"] = float(elapsed)

        if metadata and isinstance(metadata, dict):
            final_metrics = metadata.get("final_metrics") if isinstance(metadata.get("final_metrics"), dict) else {}
            if metrics["convergence_speed"] is None:
                metrics["convergence_speed"] = final_metrics.get("convergence_speed")
            if metrics["final_loss"] is None:
                metrics["final_loss"] = final_metrics.get("final_loss")
            if metrics["lpips_mean"] is None:
                metrics["lpips_mean"] = final_metrics.get("lpips_mean")
            if metrics["sharpness_mean"] is None:
                metrics["sharpness_mean"] = final_metrics.get("sharpness_mean")
            if metrics["num_gaussians"] is None:
                metrics["num_gaussians"] = metadata.get("num_gaussians")
            best_splat = metadata.get("best_splat") if isinstance(metadata.get("best_splat"), dict) else {}
            early_stop = metadata.get("early_stop") if isinstance(metadata.get("early_stop"), dict) else {}
            if metrics.get("best_splat_step") is None and isinstance(best_splat.get("step"), (int, float)):
                metrics["best_splat_step"] = int(best_splat.get("step"))
            if metrics.get("best_splat_loss") is None and isinstance(best_splat.get("loss"), (int, float)):
                metrics["best_splat_loss"] = float(best_splat.get("loss"))
            if metrics.get("stopped_early") is None and "triggered" in early_stop:
                metrics["stopped_early"] = bool(early_stop.get("triggered"))
            if metrics.get("early_stop_step") is None and isinstance(early_stop.get("trigger_step"), (int, float)):
                metrics["early_stop_step"] = int(early_stop.get("trigger_step"))

        final_tuning_params = {}
        initial_tuning_params = {}
        if isinstance(first_eval, dict):
            initial_tuning_params = first_eval.get("tuning_params") or {}
        if isinstance(latest_eval, dict):
            final_tuning_params = latest_eval.get("tuning_params") or {}
        if not final_tuning_params and isinstance(metadata, dict):
            meta_final = metadata.get("final_tuning_params")
            if isinstance(meta_final, dict):
                final_tuning_params = meta_final

        tuning_history_count = 0
        tuning_history = []
        tune_end_step = None
        tune_end_params = {}
        configured_tune_end_step = None
        if isinstance(run_config, dict):
            resolved_cfg = run_config.get("resolved_params")
            if isinstance(resolved_cfg, dict):
                configured_tune_end_step = resolved_cfg.get("tune_end_step")
        mode_value = status_info.get("mode") or (metadata.get("mode") if isinstance(metadata, dict) else None)

        if isinstance(tuning_results, dict):
            maybe_tuning_history = tuning_results.get("tuning_history")
            if isinstance(maybe_tuning_history, list):
                tuning_history = maybe_tuning_history
                tuning_history_count = len(tuning_history)
            tune_end_step = tuning_results.get("tune_end_step")
            maybe_final = tuning_results.get("final_params")
            if isinstance(maybe_final, dict):
                tune_end_params = maybe_final

        if mode_value != "modified":
            tune_end_step = None
            tune_end_params = {}

        if not runtime_tuning_series and isinstance(tuning_history, list):
            runtime_tuning_series = [
                {"step": item.get("step"), "params": item.get("params")}
                for item in tuning_history
                if isinstance(item, dict) and isinstance(item.get("step"), (int, float)) and isinstance(item.get("params"), dict)
            ]

        resolved_cfg = run_config.get("resolved_params") if isinstance(run_config, dict) and isinstance(run_config.get("resolved_params"), dict) else {}
        tune_interval = resolved_cfg.get("tune_interval")
        trend_scope = resolved_cfg.get("trend_scope")
        log_interval = resolved_cfg.get("log_interval")
        major_params = {
            "max_steps": resolved_cfg.get("max_steps"),
            "total_steps_completed": status_info.get("currentStep") if status_info.get("currentStep") is not None else latest_eval.get("step"),
            "densify_from_iter": resolved_cfg.get("densify_from_iter"),
            "densify_until_iter": resolved_cfg.get("densify_until_iter"),
            "densification_interval": resolved_cfg.get("densification_interval"),
            "eval_interval": resolved_cfg.get("eval_interval"),
            "save_interval": resolved_cfg.get("save_interval"),
            "splat_export_interval": resolved_cfg.get("splat_export_interval"),
            "best_splat_interval": resolved_cfg.get("best_splat_interval"),
            "best_splat_start_step": resolved_cfg.get("best_splat_start_step"),
            "auto_early_stop": resolved_cfg.get("auto_early_stop"),
            "early_stop_monitor_interval": resolved_cfg.get("early_stop_monitor_interval"),
            "early_stop_decision_points": resolved_cfg.get("early_stop_decision_points"),
            "early_stop_min_eval_points": resolved_cfg.get("early_stop_min_eval_points"),
            "early_stop_min_step_ratio": resolved_cfg.get("early_stop_min_step_ratio"),
            "early_stop_monitor_min_relative_improvement": resolved_cfg.get("early_stop_monitor_min_relative_improvement"),
            "early_stop_eval_min_relative_improvement": resolved_cfg.get("early_stop_eval_min_relative_improvement"),
            "early_stop_max_volatility_ratio": resolved_cfg.get("early_stop_max_volatility_ratio"),
            "early_stop_ema_alpha": resolved_cfg.get("early_stop_ema_alpha"),
            "batch_size": resolved_cfg.get("batch_size"),
        }

        outputs = files.get_output_files(project_id, run_id=requested_run_id or None)
        preview_url = None
        if resolved_engine and isinstance(outputs.get("engines"), dict):
            engine_bundle = outputs["engines"].get(resolved_engine, {})
            previews = engine_bundle.get("previews", {}) if isinstance(engine_bundle, dict) else {}
            preview_url = previews.get("latest_url")

        return {
            "project_id": project_id,
            "run_id": requested_run_id or None,
            "run_name": run_config.get("run_name") if isinstance(run_config, dict) else (requested_run_id or None),
            "name": status_info.get("name"),
            "status": status_info.get("status"),
            "mode": mode_value,
            "engine": resolved_engine or inferred_engine,
            "metrics": metrics,
            "tuning": {
                "initial": initial_tuning_params,
                "final": final_tuning_params,
                "end_params": tune_end_params,
                "end_step": (
                    configured_tune_end_step
                    if configured_tune_end_step is not None
                    else tune_end_step
                    if tune_end_step is not None
                    else (metadata.get("tune_end_step") if isinstance(metadata, dict) else None)
                ),
                "runs": metadata.get("tuning_runs") if isinstance(metadata, dict) else None,
                "history_count": tuning_history_count,
                "history": tuning_history,
                "tune_interval": tune_interval,
                "trend_scope": trend_scope,
                "log_interval": log_interval,
                "runtime_series": runtime_tuning_series,
            },
            "major_params": major_params,
            "loss_milestones": loss_milestones,
            "eval_series": eval_series,
            "eval_time_series": eval_time_series,
            "eval_psnr_series": eval_psnr_series,
            "eval_ssim_series": eval_ssim_series,
            "eval_lpips_series": eval_lpips_series,
            "preview_url": preview_url,
            "eval_points": len(eval_history) if isinstance(eval_history, list) else 0,
            "early_stop": metadata.get("early_stop") if isinstance(metadata, dict) else None,
            "ai_insights": _extract_ai_run_insights(resolved_run_dir, run_config),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building experiment summary for {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to build experiment summary")


# ==================== COMPARISON ENDPOINTS ====================

@router.post("/comparison", response_model=dict)
def create_comparison(payload: ComparisonRequest):
    """Create a comparison project (will run both baseline and optimized)."""
    try:
        comparison_id = f"cmp_{int(time.time())}"
        comparison_dir = DATA_DIR / comparison_id
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Store comparison metadata
        meta = {
            "comparison_id": comparison_id,
            "name": payload.name or "Comparison",
            "max_steps": payload.max_steps,
            "batch_size": payload.batch_size,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        logger.info(f"Created comparison: {comparison_id}")
        return {"comparison_id": comparison_id}
    except Exception as e:
        logger.error(f"Error creating comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@router.post("/comparison/{comparison_id}/images")
async def upload_comparison_images(
    comparison_id: str,
    images: list[UploadFile] = File(...)
):
    """Upload images for comparison."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        images_dir = comparison_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_count = 0
        invalid_files: list[str] = []
        allowed_ext_text = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        for img in images:
            # Validate file extension
            file_ext = Path(img.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(f"Skipped invalid image: {img.filename}")
                invalid_files.append(img.filename)
                continue
            
            content = await img.read()
            file_path = images_dir / img.filename
            with open(file_path, "wb") as f:
                f.write(content)
            uploaded_count += 1
            logger.info(f"Uploaded comparison image: {img.filename}")
        
        if uploaded_count == 0:
            invalid_list = ", ".join(invalid_files[:10])
            if len(invalid_files) > 10:
                invalid_list = f"{invalid_list}, ..."
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No valid images uploaded. Allowed formats: {allowed_ext_text}. "
                    f"Invalid files: {invalid_list}"
                ),
            )
        
        return {"status": "uploaded", "count": uploaded_count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading comparison images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload images")


@router.post("/comparison/{comparison_id}/start")
def start_comparison(comparison_id: str):
    """Start sequential comparison (baseline then optimized)."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Check if images exist
        images_dir = comparison_dir / "images"
        if not images_dir.exists() or not list(images_dir.glob("*")):
            raise HTTPException(status_code=400, detail="No images in comparison")
        
        # Start comparison in background thread
        thread = threading.Thread(
            target=run_comparison_pipeline,
            args=(comparison_id,),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started comparison: {comparison_id}")
        return {"status": "comparison_started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start comparison")


@router.get("/comparison/{comparison_id}/status", response_model=ComparisonStatus)
def get_comparison_status(comparison_id: str):
    """Get status of comparison run."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        if not comparison_dir.exists():
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Load comparison metadata
        meta_path = comparison_dir / "comparison.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Comparison metadata not found")
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        return ComparisonStatus(**meta)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get comparison status")


def run_comparison_pipeline(comparison_id: str):
    """Background task to run both baseline and optimized sequentially."""
    try:
        comparison_dir = DATA_DIR / comparison_id
        
        # Load metadata
        with open(comparison_dir / "comparison.json") as f:
            meta = json.load(f)
        
        images_dir = comparison_dir / "images"
        
        # Update status to running
        meta["status"] = "running"
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # 1. Create baseline project
        baseline_id, baseline_dir = storage.create_project()
        status.initialize_status(baseline_id, name=f"{meta['name']} - Baseline")
        shutil.copytree(images_dir, baseline_dir / "images")
        
        # Update comparison with baseline info
        meta["baseline"] = {"status": "running", "progress": 0}
        meta["baseline_project_id"] = baseline_id
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # Run baseline
        logger.info(f"Comparison {comparison_id}: Starting baseline run {baseline_id}")
        pipeline.run_full_pipeline(baseline_id, {
            "mode": "baseline",
            "max_steps": meta["max_steps"],
            "batch_size": meta["batch_size"]
        })
        
        # Update baseline status
        baseline_status = status.get_status(baseline_id)
        meta["baseline"] = baseline_status
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # 2. Create optimized project
        optimized_id, optimized_dir = storage.create_project()
        status.initialize_status(optimized_id, name=f"{meta['name']} - Optimized")
        shutil.copytree(images_dir, optimized_dir / "images")
        
        # Update comparison with optimized info
        meta["optimized"] = {"status": "running", "progress": 0}
        meta["optimized_project_id"] = optimized_id
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        # Run optimized
        logger.info(f"Comparison {comparison_id}: Starting optimized run {optimized_id}")
        pipeline.run_full_pipeline(optimized_id, {
            "mode": "modified",
            "max_steps": meta["max_steps"],
            "batch_size": meta["batch_size"]
        })
        
        # Update optimized status
        optimized_status = status.get_status(optimized_id)
        meta["optimized"] = optimized_status
        
        # Mark comparison as completed
        meta["status"] = "completed"
        with open(comparison_dir / "comparison.json", "w") as f:
            json.dump(meta, f)
        
        logger.info(f"Comparison {comparison_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison pipeline failed: {str(e)}")
        # Update comparison status to failed
        try:
            with open(comparison_dir / "comparison.json") as f:
                meta = json.load(f)
            meta["status"] = "failed"
            meta["error"] = str(e)
            with open(comparison_dir / "comparison.json", "w") as f:
                json.dump(meta, f)
        except:
            pass



@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
