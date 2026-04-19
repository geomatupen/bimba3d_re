from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import apply_preset_updates, clamp_float

PRESETS = ["conservative", "balanced", "geometry_fast", "appearance_fast"]
HEURISTIC_PRESET_BONUS = 0.002


def _selector_dir(project_dir: Path) -> Path:
    return project_dir / "models" / "input_mode_selector"


def _selector_path(project_dir: Path) -> Path:
    return _selector_dir(project_dir) / "selector_model.json"


def _default_model() -> dict[str, Any]:
    return {
        "version": 1,
        "modes": {},
    }


def _load_model(project_dir: Path) -> dict[str, Any]:
    path = _selector_path(project_dir)
    if not path.exists():
        return _default_model()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _default_model()


def _save_model(project_dir: Path, model: dict[str, Any]) -> None:
    out_dir = _selector_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _selector_path(project_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(model, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mode_entry(model: dict[str, Any], mode: str) -> dict[str, Any]:
    modes = model.setdefault("modes", {})
    entry = modes.setdefault(
        mode,
        {
            "bias": {k: 0.0 for k in PRESETS},
            "runs": 0,
            "reward_mean": 0.0,
            "last": {},
        },
    )
    if not isinstance(entry.get("bias"), dict):
        entry["bias"] = {k: 0.0 for k in PRESETS}
    for p in PRESETS:
        if p not in entry["bias"]:
            entry["bias"][p] = 0.0
    return entry


def select_preset(
    *,
    project_dir: Path,
    mode: str,
    heuristic_preset: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    model = _load_model(project_dir)
    entry = _mode_entry(model, mode)
    bias = entry.get("bias", {})

    scores: dict[str, float] = {}
    for preset in PRESETS:
        base = float(bias.get(preset, 0.0) or 0.0)
        if preset == heuristic_preset:
            base += HEURISTIC_PRESET_BONUS
        scores[preset] = base

    selected_preset = max(PRESETS, key=lambda p: scores.get(p, 0.0))

    updates = apply_preset_updates(params, selected_preset)

    return {
        "selected_preset": selected_preset,
        "yhat_scores": scores,
        "updates": updates,
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


def update_from_run(
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

    # IMPORTANT: Normalize run and baseline on a shared min/max reference frame.
    #
    # If we normalize each side independently, scores become relative to each side's own spread:
    #   run_norm(v)  = (v - min(run))  / (max(run)  - min(run))
    #   base_norm(v) = (v - min(base)) / (max(base) - min(base))
    # This can produce misleading reward signs because the same absolute metric value can map to
    # very different normalized values depending on local variance.
    #
    # To compare outcomes fairly, we compute one joint normalization per metric using run+baseline:
    #   joint_norm(v) = (v - min(run ∪ base)) / (max(run ∪ base) - min(run ∪ base))
    # Then split back into run and baseline sequences.
    #
    # Metric direction handling stays unchanged:
    # - PSNR, SSIM: higher is better (direct normalization)
    # - LPIPS, Loss: lower is better (inverted normalization)
    #
    # This keeps reward = s_run - s_base on a consistent scale and avoids false positive/negative
    # rewards caused by independent scaling artifacts.
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
        t_ratio = elapsed_vals[idx] / time_ref
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
            t_ratio = b_elapsed_vals[idx] / time_ref
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

    outcomes = {
        "t_best": t_best,
        "t_eval_best": t_eval_best,
        "t_end": t_end,
        "best_anchor": dict(by_step.get(t_eval_best, {})),
        "end_anchor": dict(by_step.get(t_end, {})),
    }
    transition = {
        "x": dict(x_features or {}),
        "yhat": dict(yhat_scores),
        "k_star": selected_preset,
        "outcomes": outcomes,
        "s_run": s_run,
        "baseline_comparison": baseline_comparison,
        "reward_signal": reward_signal,
    }

    if apply_update:
        model = _load_model(project_dir)
        entry = _mode_entry(model, mode)
        runs = int(entry.get("runs", 0) or 0)
        reward_mean = float(entry.get("reward_mean", 0.0) or 0.0)
        delta = reward_signal - reward_mean

        alpha = 1.0 / float(runs + 1)
        reward_mean = reward_mean + alpha * delta
        entry["reward_mean"] = reward_mean
        entry["runs"] = runs + 1

        bias = entry.get("bias", {})
        lr = 0.10

        for preset in PRESETS:
            cur = float(bias.get(preset, 0.0) or 0.0)
            if preset == selected_preset:
                cur += lr * delta
            else:
                cur -= lr * 0.15 * delta
            bias[preset] = float(clamp_float(cur, -3.0, 3.0))
        entry["bias"] = bias
        entry["last"] = {
            "run_id": run_id,
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
        }

        _save_model(project_dir, model)

        logger.info(
            "AI_INPUT_MODE_LEARN mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f reward=%.4f",
            mode,
            selected_preset,
            s_best,
            s_end,
            s_run,
            reward_signal,
        )
    else:
        logger.info(
            "AI_INPUT_MODE_COMPARE_ONLY mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f reward=%.4f",
            mode,
            selected_preset,
            s_best,
            s_end,
            s_run,
            reward_signal,
        )
    logger.info(
        "AI_INPUT_MODE_REWARD_OUTCOME mode=%s preset=%s reward=%.4f rewarded=%s",
        mode,
        selected_preset,
        reward_signal,
        str(reward_signal > 0.0).lower(),
    )
    logger.info(
        "Input-mode selector updated run_id=%s mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f",
        run_id,
        mode,
        selected_preset,
        s_best,
        s_end,
        s_run,
    )

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


def record_run_penalty(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    penalty_reward: float,
    reason: str,
    run_id: str,
    logger,
) -> dict[str, Any]:
    model = _load_model(project_dir)
    entry = _mode_entry(model, mode)
    runs = int(entry.get("runs", 0) or 0)
    reward_mean = float(entry.get("reward_mean", 0.0) or 0.0)
    reward_signal = float(penalty_reward)
    delta = reward_signal - reward_mean

    alpha = 1.0 / float(runs + 1)
    reward_mean = reward_mean + alpha * delta
    entry["reward_mean"] = reward_mean
    entry["runs"] = runs + 1

    bias = entry.get("bias", {})
    lr = 0.10
    for preset in PRESETS:
        cur = float(bias.get(preset, 0.0) or 0.0)
        if preset == selected_preset:
            cur += lr * delta
        else:
            cur -= lr * 0.15 * delta
        bias[preset] = float(clamp_float(cur, -3.0, 3.0))

    entry["bias"] = bias
    entry["last"] = {
        "run_id": run_id,
        "selected_preset": selected_preset,
        "yhat_scores": yhat_scores,
        "reward_signal": reward_signal,
        "reason": reason,
        "penalty": True,
    }

    _save_model(project_dir, model)

    logger.info(
        "AI_INPUT_MODE_PENALTY mode=%s preset=%s reward=%.4f reason=%s",
        mode,
        selected_preset,
        reward_signal,
        reason,
    )

    return {
        "updated": True,
        "mode": mode,
        "selected_preset": selected_preset,
        "reward_signal": reward_signal,
        "reason": reason,
        "penalty": True,
        "yhat_scores": yhat_scores,
    }
