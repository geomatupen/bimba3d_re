from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

from .common import clamp_float, clamp_int, safe_ratio

MULT_KEYS = [
    "feature_lr_mult",
    "position_lr_init_mult",
    "scaling_lr_mult",
    "opacity_lr_mult",
    "rotation_lr_mult",
    "densify_grad_threshold_mult",
    "opacity_threshold_mult",
    "lambda_dssim_mult",
]

SAFE_BOUNDS = {
    "feature_lr_mult": (0.5, 1.5),
    "position_lr_init_mult": (0.5, 1.5),
    "scaling_lr_mult": (0.5, 1.5),
    "opacity_lr_mult": (0.5, 1.5),
    "rotation_lr_mult": (0.5, 1.5),
    "densify_grad_threshold_mult": (0.7, 1.3),
    "opacity_threshold_mult": (0.7, 1.3),
    "lambda_dssim_mult": (0.7, 1.3),
}

# Mode-specific context dimensions
MODE_CONTEXT_DIMS = {
    "exif_only": 9,  # 1 intercept + 5 primary + 3 missing flags
    "exif_plus_flight_plan": 19,  # +5 primary + 5 missing flags (coverage, angle, heading)
    "exif_plus_flight_plan_plus_external": 29,  # +5 primary + 5 missing flags (veg complexity, green, texture, blur, veg_complexity)
}


def _selector_dir(project_dir: Path) -> Path:
    return project_dir / "models" / "contextual_continuous_selector"


def _selector_path(project_dir: Path, mode: str) -> Path:
    return _selector_dir(project_dir) / f"{mode}.json"


def _default_model(mode: str) -> dict[str, Any]:
    d = MODE_CONTEXT_DIMS.get(mode, 16)
    return {
        "version": 2,
        "mode": mode,
        "context_dim": d,
        "lambda_ridge": 2.0,
        "runs": 0,
        "reward_mean": 0.0,
        "models": {
            key: {
                "A": np.eye(d, dtype=np.float64).tolist(),
                "b": np.zeros(d, dtype=np.float64).tolist(),
                "n": 0,
            }
            for key in MULT_KEYS
        },
    }


def _load_model(project_dir: Path, mode: str) -> dict[str, Any]:
    path = _selector_path(project_dir, mode)
    if not path.exists():
        return _default_model(mode)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == 2:
            return data
    except Exception:
        pass
    return _default_model(mode)


def _save_model(project_dir: Path, mode: str, model: dict[str, Any]) -> None:
    out_dir = _selector_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _selector_path(project_dir, mode)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(model, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_context_vector(features: dict[str, Any], mode: str) -> np.ndarray:
    """Build normalized context vector from extracted features.

    Returns vector with:
    - Intercept term (1.0) at position 0
    - Normalized continuous features (log-scale for wide ranges)
    - Binary missing flags

    The intercept allows the model to predict non-zero values when all features
    are at baseline. Without it, neutral feature values would force prediction to 0.
    """
    x = [1.0]  # Intercept term for bias (allows baseline offset)

    # Mode 1: EXIF-only features (5 primary + 3 missing flags)
    focal = features.get("focal_length_mm", 50.0)
    focal_norm = (focal - 50.0) / 150.0  # [8-300] → roughly [-0.28, 1.67]
    x.append(focal_norm)

    shutter = features.get("shutter_s", 0.001)
    shutter_norm = np.log10(shutter + 1e-6) / 3.0  # Log scale for wide range
    x.append(shutter_norm)

    iso = features.get("iso", 400.0)
    iso_norm = (np.log10(iso) - 2.0) / 3.0  # [50-102400] → log scale
    x.append(iso_norm)

    img_w = features.get("img_width_median", 4000.0)
    img_w_norm = (img_w - 4000.0) / 2000.0
    x.append(img_w_norm)

    img_h = features.get("img_height_median", 3000.0)
    img_h_norm = (img_h - 3000.0) / 1500.0
    x.append(img_h_norm)

    # Missing flags (binary)
    x.append(float(features.get("focal_missing", 0)))
    x.append(float(features.get("shutter_missing", 0)))
    x.append(float(features.get("iso_missing", 0)))

    if mode in ["exif_plus_flight_plan", "exif_plus_flight_plan_plus_external"]:
        # Mode 2: Flight plan features (5 primary + 5 missing flags)
        gsd = features.get("gsd_median", 0.05)
        gsd_norm = gsd / 0.5  # [0.001-0.5] → [0.002, 1.0]
        x.append(gsd_norm)

        overlap = features.get("overlap_proxy", 0.5)
        x.append(overlap)  # Already [0-1]

        coverage = features.get("coverage_spread", 0.5)
        x.append(coverage)  # Already [0-1]

        angle_bucket = features.get("camera_angle_bucket", 0)
        angle_norm = float(angle_bucket) / 3.0  # {0,1,2,3} → [0, 0.33, 0.67, 1.0]
        x.append(angle_norm)

        heading = features.get("heading_consistency", 0.5)
        x.append(heading)  # Already [0-1]

        # ALL missing flags from flight plan
        x.append(float(features.get("gsd_missing", 0)))
        x.append(float(features.get("overlap_missing", 0)))
        x.append(float(features.get("coverage_missing", 0)))
        x.append(float(features.get("angle_missing", 0)))
        x.append(float(features.get("heading_missing", 0)))

    if mode == "exif_plus_flight_plan_plus_external":
        # Mode 3: External image features (5 primary + 5 missing flags)
        veg_cover = features.get("vegetation_cover_percentage", 0.5)
        x.append(veg_cover)  # Already [0-1]

        veg_complexity = features.get("vegetation_complexity_score", 0.5)
        x.append(veg_complexity)  # Already [0-1]

        terrain_rough = features.get("terrain_roughness_proxy", 0.5)
        x.append(terrain_rough)  # Already [0-1]

        texture = features.get("texture_density", 0.5)
        x.append(texture)  # Already [0-1]

        blur_risk = features.get("blur_motion_risk", 0.5)
        x.append(blur_risk)  # Already [0-1]

        # ALL missing flags from external features
        x.append(float(features.get("green_area_missing", 0)))
        x.append(float(features.get("veg_complexity_missing", 0)))
        x.append(float(features.get("roughness_missing", 0)))
        x.append(float(features.get("texture_missing", 0)))
        x.append(float(features.get("blur_missing", 0)))

    return np.array(x, dtype=np.float32)


def _predict_multiplier(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    n: int,
    exploration_mode: str,
    lambda_ridge: float,
) -> float:
    """Predict single multiplier value with exploration.

    Args:
        A: Ridge regression matrix (d×d)
        b: Target accumulator (d,)
        x: Context vector (d,)
        n: Number of observations
        exploration_mode: "thompson" or "greedy"
        lambda_ridge: Ridge regularization parameter

    Returns:
        Predicted multiplier value
    """
    d = len(x)

    # Compute posterior mean
    A_inv = np.linalg.inv(A)
    theta_mean = A_inv @ b

    if exploration_mode == "thompson":
        # Thompson Sampling: sample from posterior
        # Posterior variance decreases as n increases
        sigma_sq = 1.0 / (n + 1)
        theta_cov = sigma_sq * A_inv

        # Sample θ ~ N(θ_mean, θ_cov)
        try:
            theta_sample = np.random.multivariate_normal(theta_mean, theta_cov)
            predicted = float(x @ theta_sample)
        except np.linalg.LinAlgError:
            # Fallback to mean if covariance is singular
            predicted = float(x @ theta_mean)
    else:
        # Greedy: use posterior mean only
        predicted = float(x @ theta_mean)

    return predicted


def _build_updates(params: dict[str, Any], multipliers: dict[str, float]) -> dict[str, Any]:
    """Build parameter updates from multipliers."""
    feature_lr = float(params.get("feature_lr", 2.5e-3))
    position_lr_init = float(params.get("position_lr_init", 1.6e-4))
    scaling_lr = float(params.get("scaling_lr", 5.0e-3))
    opacity_lr = float(params.get("opacity_lr", 5.0e-2))
    rotation_lr = float(params.get("rotation_lr", 1.0e-3))
    densify_grad_threshold = float(params.get("densify_grad_threshold", 2.0e-4))
    opacity_threshold = float(params.get("opacity_threshold", 0.005))
    lambda_dssim = float(params.get("lambda_dssim", 0.2))

    return {
        "preset_name": "contextual_continuous",
        "feature_lr": clamp_float(feature_lr * multipliers["feature_lr_mult"], 5e-4, 8e-3),
        "position_lr_init": clamp_float(position_lr_init * multipliers["position_lr_init_mult"], 5e-5, 5e-4),
        "scaling_lr": clamp_float(scaling_lr * multipliers["scaling_lr_mult"], 1e-4, 2e-2),
        "opacity_lr": clamp_float(opacity_lr * multipliers["opacity_lr_mult"], 1e-3, 1e-1),
        "rotation_lr": clamp_float(rotation_lr * multipliers["rotation_lr_mult"], 1e-4, 1e-2),
        "densify_grad_threshold": clamp_float(
            densify_grad_threshold * multipliers["densify_grad_threshold_mult"],
            5e-5,
            5e-4,
        ),
        "opacity_threshold": clamp_float(
            opacity_threshold * multipliers["opacity_threshold_mult"],
            0.001,
            0.02,
        ),
        "lambda_dssim": clamp_float(lambda_dssim * multipliers["lambda_dssim_mult"], 0.05, 0.5),
    }


def select_contextual_continuous(
    *,
    project_dir: Path,
    mode: str,
    x_features: dict[str, Any],
    params: dict[str, Any],
    exploration_mode: str = "thompson",
) -> dict[str, Any]:
    """Select multipliers using contextual linear model.

    Args:
        project_dir: Project directory for model persistence
        mode: AI input mode (determines context dimension)
        x_features: Extracted feature dictionary
        params: Current training parameters
        exploration_mode: "thompson" for exploration, "greedy" for exploitation

    Returns:
        Selection result with predicted multipliers and updates
    """
    model = _load_model(project_dir, mode)
    lambda_ridge = float(model.get("lambda_ridge", 2.0))

    # Build context vector
    x = build_context_vector(x_features, mode)

    # Predict each multiplier
    multipliers: dict[str, float] = {}
    model_states: dict[str, dict[str, float]] = {}

    for key in MULT_KEYS:
        model_data = model["models"][key]
        A = np.array(model_data["A"], dtype=np.float64)
        b = np.array(model_data["b"], dtype=np.float64)
        n = int(model_data["n"])

        predicted = _predict_multiplier(A, b, x, n, exploration_mode, lambda_ridge)

        # Clamp to safe bounds
        lo, hi = SAFE_BOUNDS[key]
        multipliers[key] = clamp_float(predicted, lo, hi)

        # Track model state for logging
        theta = np.linalg.inv(A) @ b
        model_states[key] = {
            "predicted_raw": predicted,
            "predicted_clamped": multipliers[key],
            "theta_norm": float(np.linalg.norm(theta)),
            "n": n,
        }

    updates = _build_updates(params, multipliers)

    return {
        "selected_preset": "contextual_continuous",
        "yhat_scores": multipliers,
        "updates": updates,
        "context_vector": x.tolist(),
        "context_norm": float(np.linalg.norm(x)),
        "model_states": model_states,
        "exploration_mode": exploration_mode,
    }


def _normalize_series(values: list[float], invert: bool = False) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        out = [0.5 for _ in values]
    else:
        out = [(v - lo) / (hi - lo) for v in values]
    if invert:
        out = [1.0 - v for v in out]
    return out


def _step_value_with_neighbors(values: dict[int, float], step: int) -> float | None:
    for candidate in (step, step + 1, step - 1):
        value = values.get(int(candidate))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def update_from_run_contextual_continuous(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    eval_history: list[dict[str, Any]],
    baseline_eval_history: list[dict[str, Any]] | None,
    loss_by_step: dict[int, float],
    elapsed_by_step: dict[int, float],
    x_features: dict[str, Any] | None,
    run_id: str,
    logger,
    apply_update: bool = True,
) -> dict[str, Any]:
    """Update contextual models with observed reward.

    This follows the same reward calculation as the existing learners
    but updates all 8 multiplier models with the context vector.
    """
    if not eval_history:
        return {"updated": False, "reason": "no_eval_history"}

    eval_rows = [row for row in eval_history if isinstance(row, dict) and isinstance(row.get("step"), (int, float))]
    if not eval_rows:
        return {"updated": False, "reason": "no_eval_steps"}

    eval_rows.sort(key=lambda r: int(r.get("step", 0)))
    eval_steps = [int(r["step"]) for r in eval_rows]

    if loss_by_step:
        t_best = int(min(loss_by_step.keys(), key=lambda s: float(loss_by_step[s])))
    else:
        t_best = int(eval_steps[-1])

    eval_ge = [s for s in eval_steps if s >= t_best]
    t_eval_best = int(min(eval_ge)) if eval_ge else int(max(eval_steps))
    t_end = int(max(eval_steps))

    psnr_vals = [float(r.get("convergence_speed", 0.0) or 0.0) for r in eval_rows]
    ssim_vals = [float(r.get("sharpness_mean", 0.0) or 0.0) for r in eval_rows]
    lpips_vals = [float(r.get("lpips_mean", 0.0) or 0.0) for r in eval_rows]

    loss_by_step_num = {
        int(k): float(v)
        for k, v in (loss_by_step or {}).items()
        if isinstance(k, int) and isinstance(v, (int, float))
    }
    elapsed_by_step_num = {
        int(k): float(v)
        for k, v in (elapsed_by_step or {}).items()
        if isinstance(k, int) and isinstance(v, (int, float))
    }
    eval_loss_by_step = {
        int(r.get("step")): float(r.get("final_loss"))
        for r in eval_rows
        if isinstance(r, dict)
        and isinstance(r.get("step"), (int, float))
        and isinstance(r.get("final_loss"), (int, float))
    }
    eval_elapsed_by_step = {
        int(r.get("step")): float(r.get("elapsed_seconds"))
        for r in eval_rows
        if isinstance(r, dict)
        and isinstance(r.get("step"), (int, float))
        and isinstance(r.get("elapsed_seconds"), (int, float))
    }

    loss_vals: list[float] = []
    elapsed_vals: list[float] = []
    for row in eval_rows:
        step = int(row.get("step", 0) or 0)
        loss_value = _step_value_with_neighbors(loss_by_step_num, step)
        if loss_value is None:
            loss_value = _step_value_with_neighbors(eval_loss_by_step, step)
        elapsed_value = _step_value_with_neighbors(elapsed_by_step_num, step)
        if elapsed_value is None:
            elapsed_value = _step_value_with_neighbors(eval_elapsed_by_step, step)
        loss_vals.append(float(loss_value) if isinstance(loss_value, (int, float)) else 0.0)
        elapsed_vals.append(float(elapsed_value) if isinstance(elapsed_value, (int, float)) else 0.0)

    baseline_rows: list[dict[str, Any]] = []
    if isinstance(baseline_eval_history, list):
        baseline_rows = [row for row in baseline_eval_history if isinstance(row, dict) and isinstance(row.get("step"), (int, float))]
        baseline_rows.sort(key=lambda r: int(r.get("step", 0)))

    baseline_elapsed_vals = [float(r.get("elapsed_seconds") or 0.0) for r in baseline_rows] if baseline_rows else []

    b_steps: list[int] = []
    b_psnr_vals: list[float] = []
    b_ssim_vals: list[float] = []
    b_lpips_vals: list[float] = []
    b_loss_vals: list[float] = []
    b_elapsed_vals: list[float] = []
    if baseline_rows:
        b_steps = [int(r["step"]) for r in baseline_rows]
        b_psnr_vals = [float(r.get("convergence_speed", 0.0) or 0.0) for r in baseline_rows]
        b_ssim_vals = [float(r.get("sharpness_mean", 0.0) or 0.0) for r in baseline_rows]
        b_lpips_vals = [float(r.get("lpips_mean", 0.0) or 0.0) for r in baseline_rows]
        baseline_loss_by_step = {
            int(r.get("step")): float(r.get("final_loss"))
            for r in baseline_rows
            if isinstance(r.get("step"), (int, float)) and isinstance(r.get("final_loss"), (int, float))
        }
        baseline_elapsed_by_step = {
            int(r.get("step")): float(r.get("elapsed_seconds"))
            for r in baseline_rows
            if isinstance(r.get("step"), (int, float)) and isinstance(r.get("elapsed_seconds"), (int, float))
        }
        b_loss_vals = [
            float(_step_value_with_neighbors(baseline_loss_by_step, int(r.get("step", 0) or 0)) or 0.0)
            for r in baseline_rows
        ]
        b_elapsed_vals = [
            float(_step_value_with_neighbors(baseline_elapsed_by_step, int(r.get("step", 0) or 0)) or 0.0)
            for r in baseline_rows
        ]

    if baseline_rows:
        joint_psnr_norm = _normalize_series(psnr_vals + b_psnr_vals)
        joint_ssim_norm = _normalize_series(ssim_vals + b_ssim_vals)
        joint_lpips_norm = _normalize_series(lpips_vals + b_lpips_vals, invert=True)
        joint_loss_norm = _normalize_series(loss_vals + b_loss_vals, invert=True)

        split = len(psnr_vals)
        psnr_norm = joint_psnr_norm[:split]
        b_psnr_norm = joint_psnr_norm[split:]

        split = len(ssim_vals)
        ssim_norm = joint_ssim_norm[:split]
        b_ssim_norm = joint_ssim_norm[split:]

        split = len(lpips_vals)
        lpips_norm = joint_lpips_norm[:split]
        b_lpips_norm = joint_lpips_norm[split:]

        split = len(loss_vals)
        loss_norm = joint_loss_norm[:split]
        b_loss_norm = joint_loss_norm[split:]
    else:
        psnr_norm = _normalize_series(psnr_vals)
        ssim_norm = _normalize_series(ssim_vals)
        lpips_norm = _normalize_series(lpips_vals, invert=True)
        loss_norm = _normalize_series(loss_vals, invert=True)
        b_psnr_norm = []
        b_ssim_norm = []
        b_lpips_norm = []
        b_loss_norm = []

    baseline_time_ref = max(baseline_elapsed_vals) if baseline_elapsed_vals else 0.0
    time_ref = baseline_time_ref if baseline_time_ref > 0.0 else (max(elapsed_vals) if elapsed_vals else 1.0)
    time_ref = max(time_ref, 1e-6)

    by_step: dict[int, dict[str, float]] = {}
    for idx, row in enumerate(eval_rows):
        step = int(row["step"])
        q = 0.4 * psnr_norm[idx] + 0.3 * ssim_norm[idx] + 0.3 * lpips_norm[idx]
        t_ratio = safe_ratio(elapsed_vals[idx], time_ref)
        t_score = 1.0 - clamp_float(t_ratio, 0.0, 1.0)
        l_score = loss_norm[idx]
        s = 0.5 * l_score + 0.25 * q + 0.25 * t_score
        by_step[step] = {"l": l_score, "q": q, "t": t_score, "s": s}

    s_best = float(by_step.get(t_eval_best, {}).get("s", 0.0))
    s_end = float(by_step.get(t_end, {}).get("s", 0.0))
    s_run = max(s_best, s_end)

    baseline_comparison: dict[str, Any] | None = None
    reward_signal = s_run
    if baseline_rows:
        b_by_step: dict[int, dict[str, float]] = {}
        for idx, row in enumerate(baseline_rows):
            step = int(row["step"])
            q = 0.4 * b_psnr_norm[idx] + 0.3 * b_ssim_norm[idx] + 0.3 * b_lpips_norm[idx]
            t_ratio = safe_ratio(b_elapsed_vals[idx], time_ref)
            t_score = 1.0 - clamp_float(t_ratio, 0.0, 1.0)
            l_score = b_loss_norm[idx]
            s = 0.5 * l_score + 0.25 * q + 0.25 * t_score
            b_by_step[step] = {"l": l_score, "q": q, "t": t_score, "s": s}

        def _anchor_step(target: int) -> int:
            ge = [s for s in b_steps if s >= target]
            return int(min(ge)) if ge else int(max(b_steps))

        b_best_anchor_step = _anchor_step(t_eval_best)
        b_end_anchor_step = _anchor_step(t_end)
        s_base_best = float(b_by_step.get(b_best_anchor_step, {}).get("s", 0.0))
        s_base_end = float(b_by_step.get(b_end_anchor_step, {}).get("s", 0.0))
        s_base = max(s_base_best, s_base_end)
        reward_signal = s_run - s_base
        baseline_comparison = {
            "baseline_best_anchor_step": b_best_anchor_step,
            "baseline_end_anchor_step": b_end_anchor_step,
            "s_base_best": s_base_best,
            "s_base_end": s_base_end,
            "s_base": s_base,
            "s_run_relative": reward_signal,
            "score_weights": {
                "loss": 0.5,
                "quality": 0.25,
                "time": 0.25,
            },
        }

    if apply_update and x_features is not None:
        model = _load_model(project_dir, mode)
        x = build_context_vector(x_features, mode)

        # Update global statistics
        runs = int(model.get("runs", 0) or 0)
        reward_mean = float(model.get("reward_mean", 0.0) or 0.0)
        delta = reward_signal - reward_mean
        alpha = 1.0 / float(runs + 1)
        reward_mean = reward_mean + alpha * delta
        model["reward_mean"] = reward_mean
        model["runs"] = runs + 1

        # Update all 8 multiplier models with ridge regression
        theta_norms = {}
        for key in MULT_KEYS:
            model_data = model["models"][key]
            A = np.array(model_data["A"], dtype=np.float64)
            b = np.array(model_data["b"], dtype=np.float64)

            # Ridge regression update
            A += np.outer(x, x)
            b += reward_signal * x

            model_data["A"] = A.tolist()
            model_data["b"] = b.tolist()
            model_data["n"] = int(model_data["n"]) + 1

            # Track theta norm for logging
            theta = np.linalg.inv(A) @ b
            theta_norms[key] = float(np.linalg.norm(theta))

        model["last"] = {
            "run_id": run_id,
            "selected_preset": selected_preset,
            "t_best": t_best,
            "t_eval_best": t_eval_best,
            "t_end": t_end,
            "s_best": s_best,
            "s_end": s_end,
            "s_run": s_run,
            "yhat_scores": yhat_scores,
            "reward_signal": reward_signal,
            "context_vector": x.tolist(),
            "context_norm": float(np.linalg.norm(x)),
            "theta_norms": theta_norms,
        }

        _save_model(project_dir, mode, model)

        logger.info(
            "CONTEXTUAL_CONTINUOUS_UPDATE mode=%s s_best=%.4f s_end=%.4f s_run=%.4f reward=%.4f context_norm=%.3f",
            mode,
            s_best,
            s_end,
            s_run,
            reward_signal,
            np.linalg.norm(x),
        )
        logger.info(
            "CONTEXTUAL_CONTINUOUS_THETA_NORMS mode=%s theta_norms=%s",
            mode,
            json.dumps({k: f"{v:.4f}" for k, v in theta_norms.items()}),
        )
    else:
        logger.info(
            "CONTEXTUAL_CONTINUOUS_COMPARE_ONLY mode=%s s_best=%.4f s_end=%.4f s_run=%.4f reward=%.4f",
            mode,
            s_best,
            s_end,
            s_run,
            reward_signal,
        )

    logger.info(
        "CONTEXTUAL_CONTINUOUS_REWARD mode=%s preset=%s reward=%.4f rewarded=%s",
        mode,
        selected_preset,
        reward_signal,
        str(reward_signal > 0.0).lower(),
    )

    transition = {
        "x": dict(x_features or {}),
        "yhat": dict(yhat_scores),
        "k_star": selected_preset,
        "s_run": s_run,
        "baseline_comparison": baseline_comparison,
        "reward_signal": reward_signal,
    }

    return {
        "updated": bool(apply_update),
        "mode": mode,
        "selected_preset": selected_preset,
        "t_best": t_best,
        "t_eval_best": t_eval_best,
        "t_end": t_end,
        "s_best": s_best,
        "s_end": s_end,
        "s_run": s_run,
        "yhat_scores": yhat_scores,
        "transition": transition,
        "baseline_comparison": baseline_comparison,
        "reward_signal": reward_signal,
        "compare_only": not bool(apply_update),
    }


def record_run_penalty_contextual_continuous(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    penalty_reward: float,
    x_features: dict[str, Any],
    reason: str,
    run_id: str,
    logger,
) -> dict[str, Any]:
    """Record penalty for failed run."""
    model = _load_model(project_dir, mode)
    x = build_context_vector(x_features, mode)

    reward_signal = float(penalty_reward)

    # Update global statistics
    runs = int(model.get("runs", 0) or 0)
    reward_mean = float(model.get("reward_mean", 0.0) or 0.0)
    delta = reward_signal - reward_mean
    alpha = 1.0 / float(runs + 1)
    reward_mean = reward_mean + alpha * delta
    model["reward_mean"] = reward_mean
    model["runs"] = runs + 1

    # Update all models with penalty
    for key in MULT_KEYS:
        model_data = model["models"][key]
        A = np.array(model_data["A"], dtype=np.float64)
        b = np.array(model_data["b"], dtype=np.float64)

        A += np.outer(x, x)
        b += reward_signal * x

        model_data["A"] = A.tolist()
        model_data["b"] = b.tolist()
        model_data["n"] = int(model_data["n"]) + 1

    model["last"] = {
        "run_id": run_id,
        "selected_preset": selected_preset,
        "yhat_scores": yhat_scores,
        "reward_signal": reward_signal,
        "reason": reason,
        "penalty": True,
    }

    _save_model(project_dir, mode, model)

    logger.info(
        "CONTEXTUAL_CONTINUOUS_PENALTY mode=%s preset=%s reward=%.4f reason=%s",
        mode,
        selected_preset,
        reward_signal,
        reason,
    )

    return {
        "updated": True,
        "mode": mode,
        "selected_preset": selected_preset,
        "yhat_scores": yhat_scores,
        "reward_signal": reward_signal,
        "reason": reason,
    }
