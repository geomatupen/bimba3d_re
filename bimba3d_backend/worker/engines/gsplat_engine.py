import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from ..modified_rule_scopes import (
    apply_tune_scope,
    build_rule_multiplier_summary,
    normalize_tune_scope,
    select_rule_profile,
)
from ..ai_adaptive_light import ACTION_KEEP, CoreAIAdaptiveController
from ..ai_input_modes import apply_initial_preset
from ..ai_input_modes.learner import update_from_run


def _find_vswhere_exe() -> Path | None:
    candidates = []
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_msvc_build_env(logger) -> bool:
    """Ensure cl.exe is available in current process env on Windows.

    Returns True when cl.exe can be resolved after bootstrapping.
    """
    if os.name != "nt":
        return True
    if shutil.which("cl"):
        return True

    vswhere = _find_vswhere_exe()
    if not vswhere:
        logger.warning("vswhere.exe not found; cannot auto-load MSVC build environment.")
        return False

    try:
        install_query = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        install_path = (install_query.stdout or "").strip().splitlines()
        if not install_path:
            logger.warning("Visual Studio C++ build tools installation not found via vswhere.")
            return False
        root = Path(install_path[0].strip())
    except Exception as exc:
        logger.warning("Failed querying Visual Studio installation with vswhere: %s", exc)
        return False

    vcvars = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    devcmd = root / "Common7" / "Tools" / "VsDevCmd.bat"

    bootstrap_cmds: list[str] = []
    if vcvars.exists():
        bootstrap_cmds.append(f'call "{vcvars}" && set')
    if devcmd.exists():
        bootstrap_cmds.append(f'call "{devcmd}" -arch=x64 -host_arch=x64 -no_logo && set')

    if not bootstrap_cmds:
        logger.warning("No vcvars64.bat or VsDevCmd.bat found under %s", root)
        return False

    loaded = False
    for bootstrap_cmd in bootstrap_cmds:
        try:
            env_dump = subprocess.run(
                ["cmd.exe", "/d", "/s", "/c", bootstrap_cmd],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in (env_dump.stdout or "").splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key:
                    os.environ[key] = value
            loaded = True
            break
        except Exception as exc:
            logger.warning("MSVC env command failed (%s): %s", bootstrap_cmd, exc)

    if not loaded:
        msvc_glob = root / "VC" / "Tools" / "MSVC"
        if msvc_glob.exists():
            versions = sorted([p for p in msvc_glob.iterdir() if p.is_dir()])
            if versions:
                latest = versions[-1]
                cl_dir = latest / "bin" / "Hostx64" / "x64"
                if (cl_dir / "cl.exe").exists():
                    os.environ["PATH"] = str(cl_dir) + os.pathsep + os.environ.get("PATH", "")
                    logger.warning(
                        "Fell back to direct cl.exe PATH injection from %s (INCLUDE/LIB env may be incomplete).",
                        cl_dir,
                    )

    if shutil.which("cl"):
        logger.info("Loaded MSVC build environment for gsplat CUDA extension.")
        return True

    logger.warning("MSVC environment bootstrap completed but cl.exe is still not on PATH.")
    return False


def run_training(
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    params: dict,
    *,
    resume: bool = False,
    context: dict,
):
    """Run upstream simple_trainer-compatible gsplat training."""
    from ..gsplat_upstream.simple_trainer import Config, DefaultStrategy, Runner

    logger = context["logger"]
    update_status = context["update_status"]
    write_metrics = context["write_metrics"]
    get_engine_output_dir = context["get_engine_output_dir"]
    materialize_eval_previews = context["materialize_eval_previews"]
    export_with_gsplat = context["export_with_gsplat"]
    parse_step_from_name = context["parse_step_from_name"]
    collect_eval_history = context["collect_eval_history"]
    write_json_atomic = context["write_json_atomic"]

    logger.info("Starting gsplat training (upstream simple_trainer path)...")
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    engine_name = "gsplat"
    engine_output_dir = get_engine_output_dir(base_output_dir, engine_name)
    (engine_output_dir / "previews").mkdir(parents=True, exist_ok=True)
    (engine_output_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    p = dict(params or {})

    # Optional initial preset for core-ai runs; leaves legacy behavior unchanged
    # when ai_input_mode is not selected.
    preset_summary = apply_initial_preset(
        p,
        image_dir=Path(image_dir),
        colmap_dir=Path(colmap_dir),
        logger=logger,
    )
    mode = p.get("mode", "baseline")
    max_steps = int(p.get("max_steps", 15_000))
    raw_tune_start_step = p.get("tune_start_step", 100)
    try:
        modified_tune_start_step = max(1, int(raw_tune_start_step))
    except Exception:
        modified_tune_start_step = 100
    raw_tune_end_step = p.get("tune_end_step", 15000)
    try:
        modified_tune_end_step = max(1, int(raw_tune_end_step))
    except Exception:
        modified_tune_end_step = 15000
    if modified_tune_start_step > modified_tune_end_step:
        modified_tune_start_step = modified_tune_end_step
    raw_tune_interval = p.get("tune_interval", 100)
    try:
        modified_tune_interval = max(1, int(raw_tune_interval))
    except Exception:
        modified_tune_interval = 100
    raw_tune_min_improvement = p.get("tune_min_improvement", 0.005)
    try:
        tune_min_improvement = max(0.0, min(1.0, float(raw_tune_min_improvement)))
    except Exception:
        tune_min_improvement = 0.005
    tune_scope = normalize_tune_scope(p.get("tune_scope", "with_strategy"))
    trend_scope_raw = str(p.get("trend_scope") or "run").strip().lower()
    trend_scope = trend_scope_raw if trend_scope_raw in {"run", "phase"} else "run"
    raw_densify_start = p.get("densify_from_iter", 500)
    raw_densify_end = p.get("densify_until_iter", 10000)
    try:
        strategy_tune_start_step = max(1, int(raw_densify_start))
    except Exception:
        strategy_tune_start_step = 500
    try:
        strategy_tune_end_step = max(strategy_tune_start_step, int(raw_densify_end))
    except Exception:
        strategy_tune_end_step = max(strategy_tune_start_step, 10000)
    splat_interval = p.get("splat_export_interval", 31000)
    try:
        splat_interval = max(1, int(splat_interval))
    except Exception:
        splat_interval = 31000
    checkpoint_interval = p.get("save_interval", 31000)
    try:
        checkpoint_interval = max(1, int(checkpoint_interval))
    except Exception:
        checkpoint_interval = 31000
    best_splat_interval = p.get("best_splat_interval", 100)
    try:
        best_splat_interval = max(1, int(best_splat_interval))
    except Exception:
        best_splat_interval = 100
    best_splat_start_step = p.get("best_splat_start_step")
    try:
        best_splat_start_step = int(best_splat_start_step) if best_splat_start_step is not None else None
    except Exception:
        best_splat_start_step = None
    save_best_splat_raw = p.get("save_best_splat", p.get("saveBestSplat", True))
    if isinstance(save_best_splat_raw, str):
        save_best_splat = save_best_splat_raw.strip().lower() not in {"0", "false", "no", "off"}
    else:
        save_best_splat = bool(save_best_splat_raw)
    log_interval = p.get("log_interval", 100)
    try:
        log_interval = max(1, int(log_interval))
    except Exception:
        log_interval = 100
    auto_early_stop = bool(p.get("auto_early_stop", False))
    try:
        early_stop_monitor_interval = max(1, int(p.get("early_stop_monitor_interval", 200)))
    except Exception:
        early_stop_monitor_interval = 200
    try:
        early_stop_decision_points = max(3, int(p.get("early_stop_decision_points", 10)))
    except Exception:
        early_stop_decision_points = 10
    try:
        early_stop_min_eval_points = max(2, int(p.get("early_stop_min_eval_points", 6)))
    except Exception:
        early_stop_min_eval_points = 6
    try:
        early_stop_min_step_ratio = max(0.0, min(1.0, float(p.get("early_stop_min_step_ratio", 0.25))))
    except Exception:
        early_stop_min_step_ratio = 0.25
    try:
        early_stop_monitor_min_rel_improvement = max(0.0, float(p.get("early_stop_monitor_min_relative_improvement", 0.0015)))
    except Exception:
        early_stop_monitor_min_rel_improvement = 0.0015
    try:
        early_stop_eval_min_rel_improvement = max(0.0, float(p.get("early_stop_eval_min_relative_improvement", 0.003)))
    except Exception:
        early_stop_eval_min_rel_improvement = 0.003
    try:
        early_stop_max_volatility_ratio = max(0.0, float(p.get("early_stop_max_volatility_ratio", 0.01)))
    except Exception:
        early_stop_max_volatility_ratio = 0.01
    try:
        early_stop_ema_alpha = max(0.001, min(1.0, float(p.get("early_stop_ema_alpha", 0.1))))
    except Exception:
        early_stop_ema_alpha = 0.1
    if best_splat_start_step is None:
        best_splat_start_step = max(
            int(strategy_tune_start_step),
            int(early_stop_monitor_interval) * int(early_stop_decision_points),
        )
    best_splat_start_step = max(1, int(best_splat_start_step))
    project_dir = Path(image_dir).parent
    stop_flag = project_dir / "stop_requested"
    gsplat_start = time.time()
    configured_run_id = str(p.get("run_id") or "").strip()
    run_session_id = configured_run_id or f"engine-{os.getpid()}-{int(gsplat_start * 1000)}"
    tuning_state: dict[str, object] = {
        "updates": 0,
        "last_event": None,
        "events": [],
        "last_tuned_step": None,
        "last_checked_loss": None,
        "phase_complete_logged": False,
        "adaptive_schedule": None,
        "runtime_samples": [],
        "last_callback_step": None,
        "last_callback_elapsed": None,
        "last_gaussians": None,
        "strategy_frozen": False,
        "strategy_frozen_reason": None,
        "elapsed_by_step": {},
        "loss_by_step": {},
        "best_splat": {"step": None, "loss": None, "path": None},
        "input_mode_preset": preset_summary,
        "early_stop": {
            "enabled": bool(auto_early_stop),
            "candidate": False,
            "candidate_since_step": None,
            "triggered": False,
            "trigger_step": None,
            "reason": None,
            "ema_loss": None,
            "monitor_points": [],
            "eval_points": [],
            "monitor_relative_improvement": None,
            "eval_relative_improvement": None,
            "eval_volatility_ratio": None,
        },
    }
    use_html_input_mode_flow = (
        mode == "modified"
        and tune_scope == "core_ai_optimization"
        and bool(isinstance(preset_summary, dict) and preset_summary.get("applied"))
    )

    def _clamp_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

    def _clamp_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    core_ai_controller: CoreAIAdaptiveController | None = None
    if mode == "modified" and tune_scope == "core_ai_optimization" and not use_html_input_mode_flow:
        bounded_start = _clamp_int(modified_tune_start_step, 100, 1500)
        bounded_end = _clamp_int(modified_tune_end_step, bounded_start + 5000, max_steps)
        bounded_interval = _clamp_int(modified_tune_interval, 50, 400)
        bounded_min_improve = _clamp_float(tune_min_improvement, 0.001, 0.02)

        modified_tune_start_step = bounded_start
        modified_tune_end_step = bounded_end
        modified_tune_interval = bounded_interval
        tune_min_improvement = bounded_min_improve

        tuning_state["adaptive_schedule"] = {
            "start_step": int(bounded_start),
            "end_step": int(bounded_end),
            "interval": int(bounded_interval),
            "min_improvement": float(bounded_min_improve),
            "trend_scope": trend_scope,
        }

        # Extract AI tunable parameters from request payload with defaults matching CoreAIAdaptiveController signature.
        ai_reward_step_weight = p.get("ai_reward_step_weight", 0.70)
        try:
            ai_reward_step_weight = max(0.0, float(ai_reward_step_weight))
        except Exception:
            ai_reward_step_weight = 0.70

        ai_reward_trend_weight = p.get("ai_reward_trend_weight", 0.30)
        try:
            ai_reward_trend_weight = max(0.0, float(ai_reward_trend_weight))
        except Exception:
            ai_reward_trend_weight = 0.30

        ai_lr_up_multiplier = p.get("ai_lr_up_multiplier", 1.30)
        try:
            ai_lr_up_multiplier = float(ai_lr_up_multiplier)
        except Exception:
            ai_lr_up_multiplier = 1.30

        ai_lr_down_multiplier = p.get("ai_lr_down_multiplier", 0.90)
        try:
            ai_lr_down_multiplier = float(ai_lr_down_multiplier)
        except Exception:
            ai_lr_down_multiplier = 0.90

        ai_gate_alpha = p.get("ai_gate_alpha", 0.30)
        try:
            ai_gate_alpha = float(ai_gate_alpha)
        except Exception:
            ai_gate_alpha = 0.30

        ai_cooldown_intervals = p.get("ai_cooldown_intervals", 2)
        try:
            ai_cooldown_intervals = int(ai_cooldown_intervals)
        except Exception:
            ai_cooldown_intervals = 2

        ai_small_change_band = p.get("ai_small_change_band", 0.015)
        try:
            ai_small_change_band = float(ai_small_change_band)
        except Exception:
            ai_small_change_band = 0.015

        core_ai_controller = CoreAIAdaptiveController(
            project_dir=project_dir,
            run_id=run_session_id,
            max_steps=max_steps,
            tune_start_step=bounded_start,
            tune_end_step=bounded_end,
            strategy_start_step=strategy_tune_start_step,
            strategy_end_step=strategy_tune_end_step,
            base_min_improvement=bounded_min_improve,
            decision_interval=bounded_interval,
            reward_step_weight=ai_reward_step_weight,
            reward_trend_weight=ai_reward_trend_weight,
            trend_scope=trend_scope,
            lr_up_multiplier=ai_lr_up_multiplier,
            lr_down_multiplier=ai_lr_down_multiplier,
            gate_alpha=ai_gate_alpha,
            cooldown_intervals=ai_cooldown_intervals,
            small_change_band=ai_small_change_band,
        )
    runner_ref: dict[str, object] = {"runner": None}
    last_snapshot_step: dict[str, int] = {"value": -1}

    def _log_training_snapshot(
        step: int,
        max_steps_local: int,
        loss: float,
        progress_fraction: float,
        elapsed_seconds: float,
        eta_seconds: float | None,
    ):
        runner_obj = runner_ref.get("runner")
        if runner_obj is None:
            return

        try:
            gaussians = None
            gaussians_opacity_mean = None
            gaussians_scale_mean = None
            means_tensor = getattr(runner_obj, "splats", {}).get("means")
            if means_tensor is not None and hasattr(means_tensor, "shape") and len(means_tensor.shape) > 0:
                gaussians = int(means_tensor.shape[0])
            opacity_tensor = getattr(runner_obj, "splats", {}).get("opacities")
            if opacity_tensor is not None and hasattr(opacity_tensor, "mean"):
                try:
                    gaussians_opacity_mean = float(opacity_tensor.detach().mean().item())
                except Exception:
                    gaussians_opacity_mean = None
            scale_tensor = getattr(runner_obj, "splats", {}).get("scales")
            if scale_tensor is not None and hasattr(scale_tensor, "mean"):
                try:
                    gaussians_scale_mean = float(scale_tensor.detach().mean().item())
                except Exception:
                    gaussians_scale_mean = None

            strategy_obj = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
            strategy_vals: dict[str, object] = {}
            if strategy_obj is not None:
                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    if hasattr(strategy_obj, key):
                        value = getattr(strategy_obj, key)
                        if isinstance(value, float):
                            strategy_vals[key] = round(value, 8)
                        elif isinstance(value, int):
                            strategy_vals[key] = value

            optimizer_lrs: dict[str, float] = {}
            optimizers = getattr(runner_obj, "optimizers", {})
            for name in ("means", "opacities", "scales", "quats", "sh0", "shN"):
                optimizer = optimizers.get(name) if isinstance(optimizers, dict) else None
                if optimizer is None or not getattr(optimizer, "param_groups", None):
                    continue
                lr_val = optimizer.param_groups[0].get("lr")
                if lr_val is None:
                    continue
                optimizer_lrs[name] = float(lr_val)

            cfg_obj = getattr(runner_obj, "cfg", None)
            sh_degree = getattr(runner_obj, "sh_degree_to_use", None)
            eval_steps_cfg = list(getattr(cfg_obj, "eval_steps", []) or [])
            save_steps_cfg = list(getattr(cfg_obj, "save_steps", []) or [])
            next_eval_step = next((int(s) for s in eval_steps_cfg if int(s) >= int(step)), None)
            next_save_step = next((int(s) for s in save_steps_cfg if int(s) >= int(step)), None)

            steps_per_second = (float(step) / elapsed_seconds) if elapsed_seconds > 0 else None
            tuning_applied = bool(int(tuning_state.get("updates", 0) or 0) > 0)

            logger.info(
                "[GSPLAT SNAPSHOT] step=%d/%d progress=%.2f%% loss=%.6f gs=%s opacity_mean=%s scale_mean=%s sh_degree=%s next_eval=%s next_save=%s elapsed=%.1fs eta=%s speed=%s tuning_applied=%s strategy=%s lrs=%s",
                int(step),
                int(max_steps_local),
                float(progress_fraction * 100.0),
                float(loss),
                str(gaussians) if gaussians is not None else "n/a",
                f"{gaussians_opacity_mean:.6f}" if gaussians_opacity_mean is not None else "n/a",
                f"{gaussians_scale_mean:.6f}" if gaussians_scale_mean is not None else "n/a",
                str(sh_degree) if sh_degree is not None else "n/a",
                str(next_eval_step) if next_eval_step is not None else "n/a",
                str(next_save_step) if next_save_step is not None else "n/a",
                float(elapsed_seconds),
                f"{float(eta_seconds):.1f}s" if eta_seconds is not None else "n/a",
                f"{steps_per_second:.3f} step/s" if steps_per_second is not None else "n/a",
                tuning_applied,
                strategy_vals,
                {k: round(v, 10) for k, v in optimizer_lrs.items()},
            )
        except Exception as exc:
            logger.debug("Failed to emit gsplat training snapshot at step %s: %s", step, exc)

    def apply_modified_rules(step: int, loss: float) -> bool:
        if mode != "modified":
            return False
        if use_html_input_mode_flow and tune_scope == "core_ai_optimization":
            return False

        schedule = tuning_state.get("adaptive_schedule") if isinstance(tuning_state.get("adaptive_schedule"), dict) else None
        tune_start = int(schedule.get("start_step")) if isinstance(schedule, dict) else int(modified_tune_start_step)
        tune_end = int(schedule.get("end_step")) if isinstance(schedule, dict) else int(modified_tune_end_step)
        tune_interval = int(schedule.get("interval")) if isinstance(schedule, dict) else int(modified_tune_interval)
        min_improve = float(schedule.get("min_improvement")) if isinstance(schedule, dict) else float(tune_min_improvement)

        effective_tune_end = max(tune_end, strategy_tune_end_step)
        if step > effective_tune_end:
            return False
        if step < min(tune_start, strategy_tune_start_step):
            return False
        if step % max(1, tune_interval) != 0 and step not in {tune_end, strategy_tune_end_step}:
            return False
        if tuning_state.get("last_tuned_step") == step:
            return False

        runner_obj = runner_ref.get("runner")
        if runner_obj is None:
            return False

        try:
            try:
                loss_value = float(loss)
            except Exception:
                loss_value = 0.0

            previous_loss = tuning_state.get("last_checked_loss")
            tuning_state["last_checked_loss"] = float(loss_value)
            relative_improvement = None
            if previous_loss is not None:
                try:
                    denom = max(abs(float(previous_loss)), 1e-8)
                    relative_improvement = (float(previous_loss) - float(loss_value)) / denom
                except Exception:
                    relative_improvement = None

            if tune_scope == "core_individual":
                if relative_improvement is None:
                    return False
                if float(relative_improvement) >= float(min_improve):
                    return False

            apply_lr = tune_start <= step <= tune_end
            apply_strategy = strategy_tune_start_step <= step <= strategy_tune_end_step
            if bool(tuning_state.get("strategy_frozen")):
                apply_strategy = False
            if tune_scope == "core_individual":
                apply_strategy = False
            if not apply_lr and not apply_strategy:
                return False

            def _learn_schedule() -> None:
                if tune_scope != "core_ai_optimization" or not isinstance(schedule, dict):
                    return

                nonlocal tune_start, tune_end, tune_interval, min_improve

                updated = False
                if relative_improvement is not None:
                    rel = float(relative_improvement)
                    if rel < min_improve * 0.5 and step < tune_start:
                        tune_start = _clamp_int(tune_start - 50, 100, 1500)
                        updated = True

                    if rel < min_improve * 0.4:
                        tune_interval = _clamp_int(tune_interval - 10, 50, 400)
                        min_improve = _clamp_float(min_improve * 0.95, 0.001, 0.02)
                        updated = True
                    elif rel > min_improve * 1.6:
                        tune_interval = _clamp_int(tune_interval + 10, 50, 400)
                        min_improve = _clamp_float(min_improve * 1.05, 0.001, 0.02)
                        updated = True

                    if rel < min_improve * 0.3 and step < tune_end - 500:
                        tune_end = _clamp_int(tune_end - 100, tune_start + 5000, max_steps)
                        updated = True
                    elif rel > min_improve * 2.0 and step > tune_end - 2000:
                        tune_end = _clamp_int(tune_end + 100, tune_start + 5000, max_steps)
                        updated = True

                runtime_samples = tuning_state.get("runtime_samples") if isinstance(tuning_state.get("runtime_samples"), list) else []
                means_tensor = getattr(runner_obj, "splats", {}).get("means")
                gaussians = int(means_tensor.shape[0]) if means_tensor is not None and hasattr(means_tensor, "shape") and len(means_tensor.shape) > 0 else 0
                last_gaussians = tuning_state.get("last_gaussians")
                tuning_state["last_gaussians"] = gaussians

                if (
                    not bool(tuning_state.get("strategy_frozen"))
                    and len(runtime_samples) >= 6
                    and gaussians >= 200000
                    and isinstance(last_gaussians, (int, float))
                ):
                    baseline = sum(float(v) for v in runtime_samples[:3]) / 3.0
                    recent = sum(float(v) for v in runtime_samples[-3:]) / 3.0
                    growth = (float(gaussians) - float(last_gaussians)) / max(abs(float(last_gaussians)), 1.0)
                    if baseline > 0 and recent >= baseline * 3.0 and growth > 0.02:
                        tuning_state["strategy_frozen"] = True
                        tuning_state["strategy_frozen_reason"] = (
                            f"runtime_slowdown baseline={baseline:.4f}s/step recent={recent:.4f}s/step gaussians={gaussians}"
                        )

                if updated:
                    schedule["start_step"] = int(tune_start)
                    schedule["end_step"] = int(tune_end)
                    schedule["interval"] = int(tune_interval)
                    schedule["min_improvement"] = float(min_improve)

            _learn_schedule()

            if tune_scope == "core_ai_optimization" and core_ai_controller is not None:
                decision = core_ai_controller.decide_and_apply(
                    step=int(step),
                    loss=loss_value,
                    runner_obj=runner_obj,
                    apply_lr=bool(apply_lr),
                    apply_strategy=bool(apply_strategy),
                )
                event = {
                    "action_adjustment_tag": decision.action_adjustment_tag,
                    "step": int(step),
                    "loss": loss_value,
                    "previous_loss": float(previous_loss) if previous_loss is not None else None,
                    "relative_improvement": float(relative_improvement) if relative_improvement is not None else None,
                    "required_min_improvement": float(tune_min_improvement),
                    "adaptive_schedule": {
                        "start_step": int(tune_start),
                        "end_step": int(tune_end),
                        "interval": int(tune_interval),
                        "min_improvement": float(min_improve),
                    },
                    "apply_lr": bool(apply_lr),
                    "apply_strategy": bool(apply_strategy),
                    "profile": "ai_adaptive_light",
                    "scope": tune_scope,
                    "rule_multipliers": {},
                    "scope_multipliers": {},
                    "adjustments": [f"ai_action_{decision.action_adjustment_tag}"],
                    "lr_changes": {},
                    "params": {
                        "learning_rates": {},
                        "strategy": {},
                    },
                    "before": {
                        "learning_rates": {},
                        "strategy": {},
                    },
                    "ai_decision": {
                        "action": decision.action,
                        "action_adjustment_tag": decision.action_adjustment_tag,
                        "reason": decision.reason,
                        "gate_threshold": decision.gate_threshold,
                        "reward_from_previous": decision.reward_from_previous,
                        "relative_improvement": decision.relative_improvement,
                        "scores": decision.action_scores,
                    },
                }
                tuning_state["last_event"] = event
                tuning_state["events"].append(event)
                tuning_state["updates"] = int(tuning_state.get("updates", 0) or 0) + 1
                tuning_state["last_tuned_step"] = int(step)

                update_status(
                    project_dir,
                    "processing",
                    mode=mode,
                    tuning_active=True,
                    last_tuning={
                        "step": int(step),
                        "action": f"AI adaptive action: {decision.action}",
                        "reason": decision.reason,
                        "scope": tune_scope,
                        "profile": "ai_adaptive_light",
                        "adjustments": [f"ai_action_{decision.action}"],
                        "adaptive_schedule": {
                            "start_step": int(tune_start),
                            "end_step": int(tune_end),
                            "interval": int(tune_interval),
                            "min_improvement": float(min_improve),
                        },
                        "strategy_frozen": bool(tuning_state.get("strategy_frozen")),
                        "strategy_frozen_reason": tuning_state.get("strategy_frozen_reason"),
                    },
                )
                logger.info(
                    "Core-AI adaptive decision step=%d action=%s reason=%s apply_lr=%s apply_strategy=%s loss=%.6f prev_loss=%s rel_improve=%s gate=%.6f reward_prev=%s",
                    step,
                    decision.action,
                    decision.reason,
                    str(bool(apply_lr)),
                    str(bool(apply_strategy)),
                    loss_value,
                    f"{float(previous_loss):.6f}" if previous_loss is not None else "n/a",
                    f"{float(relative_improvement):.6f}" if relative_improvement is not None else "n/a",
                    float(decision.gate_threshold),
                    f"{float(decision.reward_from_previous):.6f}" if decision.reward_from_previous is not None else "n/a",
                )
                return decision.action != ACTION_KEEP

            profile_data = select_rule_profile(loss_value)
            profile = profile_data.name
            applied_multipliers = build_rule_multiplier_summary(tune_scope, profile_data)
            scope_multipliers = {
                "lr": dict(profile_data.lr_multipliers),
                "strategy": dict(profile_data.strategy_multipliers),
            }

            scope_updates = apply_tune_scope(
                tune_scope,
                runner_obj,
                profile_data,
                apply_lr=apply_lr,
                apply_strategy=apply_strategy,
            )
            before_lrs = dict(scope_updates.get("before_lrs") or {})
            after_lrs = dict(scope_updates.get("after_lrs") or {})
            strategy_before = dict(scope_updates.get("strategy_before") or {})
            strategy_after = dict(scope_updates.get("strategy_after") or {})
            adjustments = list(scope_updates.get("adjustments") or [])

            lr_change_details: dict[str, dict[str, float]] = {}
            for name, before_val in before_lrs.items():
                after_val = after_lrs.get(name)
                if after_val is None:
                    continue
                multiplier = 1.0
                try:
                    if float(before_val) != 0.0:
                        multiplier = float(after_val) / float(before_val)
                except Exception:
                    multiplier = 1.0
                lr_change_details[name] = {
                    "before": float(before_val),
                    "after": float(after_val),
                    "multiplier": float(multiplier),
                }

            strategy_change_details: dict[str, dict[str, float]] = {}
            for key, before_val in strategy_before.items():
                after_val = strategy_after.get(key)
                if after_val is None:
                    continue
                strategy_change_details[key] = {
                    "before": float(before_val),
                    "after": float(after_val),
                }

            has_lr_updates = any(
                abs(float(v.get("after", 0.0)) - float(v.get("before", 0.0))) > 1e-15
                for v in lr_change_details.values()
            )
            has_strategy_updates = any(
                abs(float(v.get("after", 0.0)) - float(v.get("before", 0.0))) > 1e-12
                for v in strategy_change_details.values()
            )
            if not has_lr_updates and not has_strategy_updates:
                tuning_state["last_tuned_step"] = int(step)
                return False

            event = {
                "step": int(step),
                "loss": loss_value,
                "previous_loss": float(previous_loss) if previous_loss is not None else None,
                "relative_improvement": float(relative_improvement) if relative_improvement is not None else None,
                "required_min_improvement": float(tune_min_improvement),
                "adaptive_schedule": {
                    "start_step": int(tune_start),
                    "end_step": int(tune_end),
                    "interval": int(tune_interval),
                    "min_improvement": float(min_improve),
                },
                "apply_lr": bool(apply_lr),
                "apply_strategy": bool(apply_strategy),
                "profile": profile,
                "scope": tune_scope,
                "rule_multipliers": applied_multipliers,
                "scope_multipliers": scope_multipliers,
                "adjustments": adjustments,
                "lr_changes": lr_change_details,
                "params": {
                    "learning_rates": after_lrs,
                    "strategy": strategy_after,
                },
                "before": {
                    "learning_rates": before_lrs,
                    "strategy": strategy_before,
                },
            }
            tuning_state["last_event"] = event
            tuning_state["events"].append(event)
            tuning_state["updates"] = int(tuning_state.get("updates", 0) or 0) + 1
            tuning_state["last_tuned_step"] = int(step)

            update_status(
                project_dir,
                "processing",
                mode=mode,
                tuning_active=True,
                last_tuning={
                    "step": int(step),
                    "action": f"Rule-based {profile} update",
                    "reason": f"Modified mode rule check (LR {modified_tune_start_step}-{modified_tune_end_step}, strategy {strategy_tune_start_step}-{strategy_tune_end_step})",
                    "adaptive_schedule": {
                        "start_step": int(tune_start),
                        "end_step": int(tune_end),
                        "interval": int(tune_interval),
                        "min_improvement": float(min_improve),
                    },
                    "strategy_frozen": bool(tuning_state.get("strategy_frozen")),
                    "strategy_frozen_reason": tuning_state.get("strategy_frozen_reason"),
                    "scope": tune_scope,
                    "profile": profile,
                    "lr_changes": lr_change_details,
                    "adjustments": adjustments,
                },
            )
            logger.info(
                "Modified rule update applied at step %d (lr_window=%d-%d, strategy_window=%d-%d, apply_lr=%s, apply_strategy=%s, loss=%.6f, prev_loss=%s, rel_improve=%s, min_improve=%.4f, profile=%s, scope=%s)",
                step,
                tune_start,
                tune_end,
                strategy_tune_start_step,
                strategy_tune_end_step,
                str(bool(apply_lr)),
                str(bool(apply_strategy)),
                loss_value,
                f"{float(previous_loss):.6f}" if previous_loss is not None else "n/a",
                f"{float(relative_improvement):.6f}" if relative_improvement is not None else "n/a",
                min_improve,
                profile,
                tune_scope,
            )
            logger.info(
                "Modified rule multipliers step=%d profile=%s scope=%s lr=%s strategy=%s",
                step,
                profile,
                tune_scope,
                json.dumps(scope_multipliers.get("lr", {}), sort_keys=True),
                json.dumps(scope_multipliers.get("strategy", {}), sort_keys=True),
            )
            logger.info(
                "Modified rule details step=%d lr_changes=%s before_strategy=%s after_strategy=%s",
                step,
                json.dumps(lr_change_details, sort_keys=True),
                json.dumps(strategy_before, sort_keys=True),
                json.dumps(strategy_after, sort_keys=True),
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed modified-mode rule update at step %d/%d: %s",
                step,
                modified_tune_end_step,
                exc,
            )
            return False

    def stop_checker() -> bool:
        return stop_flag.exists()

    def _evaluate_early_stop_on_eval(step: int, max_steps_local: int) -> None:
        early_state = tuning_state.get("early_stop") if isinstance(tuning_state.get("early_stop"), dict) else {}
        if not isinstance(early_state, dict) or not bool(early_state.get("enabled")):
            return
        if bool(early_state.get("triggered")):
            return

        loss_by_step = tuning_state.get("loss_by_step") if isinstance(tuning_state.get("loss_by_step"), dict) else {}
        eval_loss = loss_by_step.get(int(step))
        if not isinstance(eval_loss, (int, float)):
            return

        eval_points = early_state.get("eval_points") if isinstance(early_state.get("eval_points"), list) else []
        eval_points.append({"step": int(step), "loss": float(eval_loss)})
        if len(eval_points) > 200:
            del eval_points[:-200]
        early_state["eval_points"] = eval_points

        required_points = max(int(early_stop_min_eval_points), int(early_stop_decision_points))
        if len(eval_points) < required_points:
            tuning_state["early_stop"] = early_state
            return

        min_step_gate = int(max_steps_local * float(early_stop_min_step_ratio))
        if int(step) < min_step_gate:
            tuning_state["early_stop"] = early_state
            return

        candidate_since = early_state.get("candidate_since_step")
        if not isinstance(candidate_since, int):
            tuning_state["early_stop"] = early_state
            return
        if int(step) - int(candidate_since) < int(p.get("eval_interval") or 0):
            tuning_state["early_stop"] = early_state
            return

        window = eval_points[-int(early_stop_decision_points):]
        losses = [float(item.get("loss")) for item in window if isinstance(item.get("loss"), (int, float))]
        if len(losses) < int(early_stop_decision_points):
            tuning_state["early_stop"] = early_state
            return

        first_loss = losses[0]
        last_loss = losses[-1]
        denom = max(abs(first_loss), 1e-8)
        rel_improve = (first_loss - last_loss) / denom

        mean_loss = sum(losses) / max(len(losses), 1)
        std_loss = statistics.pstdev(losses) if len(losses) > 1 else 0.0
        volatility_ratio = std_loss / max(abs(mean_loss), 1e-8)

        early_state["eval_relative_improvement"] = float(rel_improve)
        early_state["eval_volatility_ratio"] = float(volatility_ratio)

        should_stop = (
            float(rel_improve) < float(early_stop_eval_min_rel_improvement)
            and float(volatility_ratio) <= float(early_stop_max_volatility_ratio)
        )

        if should_stop:
            early_state["triggered"] = True
            early_state["trigger_step"] = int(step)
            early_state["reason"] = (
                f"plateau_confirmed rel_improve={rel_improve:.6f} "
                f"volatility={volatility_ratio:.6f} points={len(losses)}"
            )
            stop_flag.write_text("early_stop")
            logger.info(
                "EARLY_STOP_TRIGGER step=%d rel_improve=%.6f volatility=%.6f points=%d candidate_since=%s min_step_ratio=%.3f",
                int(step),
                float(rel_improve),
                float(volatility_ratio),
                len(losses),
                str(early_state.get("candidate_since_step")),
                float(early_stop_min_step_ratio),
            )
            update_status(
                project_dir,
                "stopping",
                progress=60 + int((float(step) / max(float(max_steps_local), 1.0)) * 35),
                stage="training",
                stage_progress=int((float(step) / max(float(max_steps_local), 1.0)) * 100),
                message=(
                    "🛑 Early stop candidate confirmed on eval; finishing current step and exporting outputs "
                    f"(step {int(step)})."
                ),
                early_stop={
                    "candidate": bool(early_state.get("candidate")),
                    "candidate_since_step": early_state.get("candidate_since_step"),
                    "triggered": True,
                    "trigger_step": int(step),
                    "reason": early_state.get("reason"),
                    "eval_relative_improvement": float(rel_improve),
                    "eval_volatility_ratio": float(volatility_ratio),
                },
            )

        tuning_state["early_stop"] = early_state

    def progress_callback(
        step: int,
        max_steps_local: int | None = None,
        loss: float = 0.0,
        **kwargs: object,
    ) -> None:
        schedule = tuning_state.get("adaptive_schedule") if isinstance(tuning_state.get("adaptive_schedule"), dict) else None
        tune_end_for_phase = int(schedule.get("end_step")) if isinstance(schedule, dict) else int(modified_tune_end_step)
        if max_steps_local is None:
            raw_max_steps = kwargs.get("max_steps", max_steps)
            try:
                max_steps_local = int(raw_max_steps)
            except Exception:
                max_steps_local = int(max_steps)
            if mode == "modified" and not tuning_state.get("phase_complete_logged") and step == tune_end_for_phase + 1:
                tuning_state["phase_complete_logged"] = True
        apply_modified_rules(step, loss)
        progress_fraction = 0.0 if max_steps_local <= 0 else float(step) / float(max_steps_local)
        progress_fraction = max(0.0, min(1.0, progress_fraction))

        now = time.time()
        elapsed = now - gsplat_start

        elapsed_by_step = tuning_state.get("elapsed_by_step") if isinstance(tuning_state.get("elapsed_by_step"), dict) else {}
        loss_by_step = tuning_state.get("loss_by_step") if isinstance(tuning_state.get("loss_by_step"), dict) else {}
        elapsed_by_step[int(step)] = float(elapsed)
        loss_by_step[int(step)] = float(loss)
        tuning_state["elapsed_by_step"] = elapsed_by_step
        tuning_state["loss_by_step"] = loss_by_step

        early_state = tuning_state.get("early_stop") if isinstance(tuning_state.get("early_stop"), dict) else {}
        if isinstance(early_state, dict) and bool(early_state.get("enabled")) and int(step) % int(early_stop_monitor_interval) == 0:
            prev_ema = early_state.get("ema_loss")
            if isinstance(prev_ema, (int, float)):
                ema_loss = float(early_stop_ema_alpha) * float(loss) + (1.0 - float(early_stop_ema_alpha)) * float(prev_ema)
            else:
                ema_loss = float(loss)
            early_state["ema_loss"] = float(ema_loss)

            monitor_points = early_state.get("monitor_points") if isinstance(early_state.get("monitor_points"), list) else []
            monitor_points.append({"step": int(step), "ema_loss": float(ema_loss)})
            if len(monitor_points) > 200:
                del monitor_points[:-200]
            early_state["monitor_points"] = monitor_points

            if len(monitor_points) >= int(early_stop_decision_points):
                window = monitor_points[-int(early_stop_decision_points):]
                first_ema = float(window[0].get("ema_loss"))
                last_ema = float(window[-1].get("ema_loss"))
                monitor_rel_improve = (first_ema - last_ema) / max(abs(first_ema), 1e-8)
                early_state["monitor_relative_improvement"] = float(monitor_rel_improve)

                if float(monitor_rel_improve) < float(early_stop_monitor_min_rel_improvement):
                    if not bool(early_state.get("candidate")):
                        early_state["candidate"] = True
                        early_state["candidate_since_step"] = int(step)
                else:
                    early_state["candidate"] = False
                    early_state["candidate_since_step"] = None

            tuning_state["early_stop"] = early_state

        last_step = tuning_state.get("last_callback_step")
        last_elapsed = tuning_state.get("last_callback_elapsed")
        if isinstance(last_step, int) and isinstance(last_elapsed, (int, float)) and step > last_step:
            delta_steps = step - last_step
            delta_time = float(elapsed - float(last_elapsed))
            if delta_time > 0:
                sec_per_step = delta_time / float(delta_steps)
                samples = tuning_state.get("runtime_samples") if isinstance(tuning_state.get("runtime_samples"), list) else []
                samples.append(float(sec_per_step))
                if len(samples) > 20:
                    del samples[:-20]
                tuning_state["runtime_samples"] = samples
        tuning_state["last_callback_step"] = int(step)
        tuning_state["last_callback_elapsed"] = float(elapsed)

        requested_stop = stop_checker()

        eta = (
            (elapsed / progress_fraction) * (1 - progress_fraction)
            if progress_fraction > 0
            else None
        )
        timing = {"start": gsplat_start, "elapsed": elapsed}
        if eta is not None:
            timing["eta"] = eta

        message = (
            f"⏸️ Stopping after step {step}/{max_steps_local} completes (loss: {loss:.6f})..."
            if requested_stop
            else f"🎯 Training step {step}/{max_steps_local} (loss: {loss:.6f})"
        )

        update_status(
            project_dir,
            "stopping" if requested_stop else "processing",
            progress=60 + int(progress_fraction * 35),
            mode=mode,
            tuning_active=(mode == "modified" and step <= max(tune_end_for_phase, strategy_tune_end_step)),
            currentStep=step,
            maxSteps=max_steps_local,
            current_loss=float(loss),
            stop_requested=requested_stop,
            stage="training",
            stage_progress=int(progress_fraction * 100),
            message=message,
            timing=timing,
            early_stop=(
                {
                    "candidate": bool(early_state.get("candidate")),
                    "candidate_since_step": early_state.get("candidate_since_step"),
                    "triggered": bool(early_state.get("triggered")),
                    "trigger_step": early_state.get("trigger_step"),
                    "reason": early_state.get("reason"),
                    "monitor_relative_improvement": early_state.get("monitor_relative_improvement"),
                    "eval_relative_improvement": early_state.get("eval_relative_improvement"),
                    "eval_volatility_ratio": early_state.get("eval_volatility_ratio"),
                }
                if isinstance(early_state, dict) and bool(early_state.get("enabled"))
                else None
            ),
        )
        write_metrics(project_dir, {
            "step": step,
            "loss": loss,
            "progress": progress_fraction,
        }, engine=engine_name)

        should_log_snapshot = (
            step == 1
            or step == max_steps_local
            or step % log_interval == 0
        )
        if should_log_snapshot and step != last_snapshot_step["value"]:
            _log_training_snapshot(step, max_steps_local, loss, progress_fraction, elapsed, eta)
            last_snapshot_step["value"] = step

    torch_version = None
    torch_cuda_version = None
    try:
        import torch
        torch_version = getattr(torch, "__version__", None)
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    device = "cuda" if (p.get("use_cuda", True) and cuda_ok) else "cpu"

    if stop_flag.exists():
        update_status(project_dir, "stopped", progress=55, stage="training", message="⏸️ Processing stopped before gsplat training.", stop_requested=True, stopped_stage="training")
        return 0

    update_status(
        project_dir,
        "processing",
        progress=55,
        stage="training",
        stage_progress=0,
        message=f"🚀 Initializing upstream simple_trainer ({'GPU ⚡' if device == 'cuda' else 'CPU'})...",
        mode=mode,
        timing={"start": gsplat_start},
    )

    dataset_dir = Path(image_dir).parent

    def _build_steps(interval_value, fallback):
        if interval_value is None:
            return fallback
        try:
            interval = max(1, int(interval_value))
        except Exception:
            return fallback
        out = list(range(interval, max_steps + 1, interval))
        if max_steps not in out:
            out.append(max_steps)
        return sorted(set(out))

    strategy = DefaultStrategy(
        verbose=True,
        prune_opa=float(p.get("opacity_threshold", 0.005)),
        grow_grad2d=float(p.get("densify_grad_threshold", 0.0002)),
        grow_scale3d=float(p.get("percent_dense", 0.01)),
        refine_start_iter=int(p.get("densify_from_iter", 500)),
        refine_stop_iter=int(p.get("densify_until_iter", 15000)),
        refine_every=max(1, int(p.get("densification_interval", 100))),
        reset_every=max(1, int(p.get("opacity_reset_interval", 3000))),
    )

    feature_lr = float(p.get("feature_lr", 2.5e-3))
    eval_steps = _build_steps(p.get("eval_interval"), [7000, 30000])
    save_steps = sorted(set(
        _build_steps(checkpoint_interval, [31000])
        + _build_steps(splat_interval, [31000])
        + _build_steps(best_splat_interval, [7000, 30000])
    ))

    cfg = Config(
        disable_viewer=True,
        disable_video=True,
        load_exposure=False,
        data_dir=str(dataset_dir),
        image_dir_override=str(Path(image_dir)),
        sparse_dir_override=str(Path(colmap_dir)),
        data_factor=1,
        result_dir=str(engine_output_dir),
        test_every=8,
        normalize_world_space=True,
        batch_size=1,
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_ply=False,
        ssim_lambda=float(p.get("lambda_dssim", 0.2)),
        means_lr=float(p.get("position_lr_init", 1.6e-4)),
        scales_lr=float(p.get("scaling_lr", 5.0e-3)),
        opacities_lr=float(p.get("opacity_lr", 5.0e-2)),
        quats_lr=float(p.get("rotation_lr", 1.0e-3)),
        sh0_lr=feature_lr,
        shN_lr=feature_lr / 20.0,
        strategy=strategy,
        tb_every=0,
    )
    cfg.disable_tqdm = not bool(p.get("enable_tqdm", False))
    progress_every = max(1, int(log_interval))
    if mode == "modified":
        progress_every = min(progress_every, max(1, int(modified_tune_interval)))
    progress_every = min(progress_every, best_splat_interval)
    cfg.progress_every = progress_every
    if cfg.disable_tqdm:
        os.environ["TQDM_DISABLE"] = "1"
    else:
        os.environ.pop("TQDM_DISABLE", None)

    logger.info(
        "GSPLAT logging cadence: snapshot every %d steps (log_interval), callback every %d steps, modified_tune_interval=%d, best_splat_start_step=%d; tqdm=%s",
        log_interval,
        cfg.progress_every,
        modified_tune_interval,
        best_splat_start_step,
        "disabled" if cfg.disable_tqdm else "enabled",
    )

    cfg.stop_checker = stop_checker
    cfg.progress_callback = progress_callback

    snapshots_dir = engine_output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    def eval_callback(step: int) -> None:
        materialize_eval_previews(engine_output_dir, eval_step=step)
        _evaluate_early_stop_on_eval(int(step), int(max_steps))

    def checkpoint_callback(step: int, checkpoint_path: str) -> None:
        if step >= best_splat_start_step and (step % best_splat_interval == 0 or step == max_steps):
            try:
                loss_by_step = tuning_state.get("loss_by_step") if isinstance(tuning_state.get("loss_by_step"), dict) else {}
                best_state = tuning_state.get("best_splat") if isinstance(tuning_state.get("best_splat"), dict) else {}
                current_loss = loss_by_step.get(int(step))
                best_loss = best_state.get("loss") if isinstance(best_state, dict) else None
                should_update_best = isinstance(current_loss, (int, float)) and (
                    not isinstance(best_loss, (int, float)) or float(current_loss) < float(best_loss)
                )
                if should_update_best:
                    previous_best = float(best_loss) if isinstance(best_loss, (int, float)) else None
                    if save_best_splat:
                        export_with_gsplat(
                            Path(checkpoint_path),
                            engine_output_dir,
                            splat_name="best.splat",
                            export_ply=False,
                            log_details=False,
                        )
                        tuning_state["best_splat"] = {
                            "step": int(step),
                            "loss": float(current_loss),
                            "path": str(engine_output_dir / "best.splat"),
                        }
                        best_path = engine_output_dir / "best.splat"
                        best_size = best_path.stat().st_size if best_path.exists() else None
                    else:
                        tuning_state["best_splat"] = {
                            "step": int(step),
                            "loss": float(current_loss),
                            "path": None,
                        }
                        best_size = None
                    improvement = (previous_best - float(current_loss)) if previous_best is not None else None
                    logger.info(
                        "BEST_SPLAT_UPDATE step=%d loss=%.6f prev_best=%s improvement=%s bytes=%s save=%s",
                        int(step),
                        float(current_loss),
                        f"{previous_best:.6f}" if previous_best is not None else "n/a",
                        f"{improvement:.6f}" if improvement is not None else "n/a",
                        str(best_size) if best_size is not None else "n/a",
                        str(save_best_splat),
                    )
            except Exception as exc:
                logger.warning("Failed to export best.splat at step %s: %s", step, exc)

        if splat_interval and step % splat_interval != 0 and step != max_steps:
            return
        snapshot_name = f"snapshot_step_{step:06d}.splat"
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return
        export_with_gsplat(
            Path(checkpoint_path),
            snapshots_dir,
            splat_name=snapshot_name,
            export_ply=False,
        )

    cfg.eval_callback = eval_callback
    cfg.checkpoint_callback = checkpoint_callback

    if device == "cpu":
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "<unset>"
        nvcc_path = shutil.which("nvcc") or "<not-found>"
        cl_path = shutil.which("cl") or "<not-found>"
        msg = (
            "Upstream simple_trainer requires CUDA in this worker path. "
            f"torch={torch_version or '<unknown>'}, torch.version.cuda={torch_cuda_version or '<none>'}, "
            f"torch.cuda.is_available={cuda_ok}, CUDA_HOME/CUDA_PATH={cuda_home}, nvcc={nvcc_path}, cl={cl_path}."
        )
        update_status(project_dir, "failed", progress=55, stage="training", message=msg, error=msg)
        raise RuntimeError(msg)

    _load_msvc_build_env(logger)

    gsplat_cuda_error: str | None = None
    try:
        import gsplat.cuda._wrapper as _gsplat_cuda_wrapper
        if hasattr(_gsplat_cuda_wrapper, "_make_lazy_cuda_obj"):
            _gsplat_cuda_wrapper._make_lazy_cuda_obj("CameraModelType")
            _gsplat_cuda_ready = True
        else:
            _gsplat_cuda_ext = getattr(_gsplat_cuda_wrapper, "_C", None)
            _gsplat_cuda_ready = _gsplat_cuda_ext is not None
    except Exception as exc:
        _gsplat_cuda_ready = False
        gsplat_cuda_error = str(exc)

    if not _gsplat_cuda_ready:
        msg = (
            "gsplat CUDA extension is unavailable. Install CUDA Toolkit and a CUDA-enabled PyTorch build, "
            "then reinstall gsplat so CUDA extensions can be built/loaded."
        )
        if gsplat_cuda_error:
            msg = f"{msg} Details: {gsplat_cuda_error}"
        update_status(project_dir, "failed", progress=55, stage="training", message=msg, error=msg)
        raise RuntimeError(msg)

    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)

    start_model_mode = str(p.get("start_model_mode") or "scratch").strip().lower()
    source_model_id = str(p.get("source_model_id") or "").strip()
    source_model_checkpoint = str(p.get("source_model_checkpoint") or "").strip()
    if start_model_mode == "reuse" and source_model_checkpoint:
        try:
            ckpt = torch.load(source_model_checkpoint, map_location=runner.device)
            splats_state = ckpt.get("splats") if isinstance(ckpt, dict) else None
            if not isinstance(splats_state, dict):
                raise ValueError("Checkpoint missing 'splats' state")
            for key in runner.splats.keys():
                if key not in splats_state:
                    raise ValueError(f"Checkpoint missing splat tensor '{key}'")
                runner.splats[key].data = splats_state[key].to(runner.device).clone()

            update_status(
                project_dir,
                "processing",
                progress=55,
                stage="training",
                stage_progress=2,
                message=f"Loaded reusable model '{source_model_id or 'selected-model'}' for warm-start.",
            )
            logger.info(
                "Warm-started gsplat from reusable model checkpoint %s (model_id=%s)",
                source_model_checkpoint,
                source_model_id or "<unknown>",
            )
        except Exception as exc:
            msg = f"Failed to load reusable model checkpoint: {exc}"
            update_status(project_dir, "failed", progress=55, stage="training", message=msg, error=msg)
            raise RuntimeError(msg) from exc

    runner_ref["runner"] = runner
    stop_reason = runner.train()
    gsplat_end = time.time()
    early_stop_state = tuning_state.get("early_stop") if isinstance(tuning_state.get("early_stop"), dict) else {}
    early_stop_triggered = bool(isinstance(early_stop_state, dict) and early_stop_state.get("triggered"))
    stop_flag_reason = None
    if stop_flag.exists():
        try:
            stop_flag_reason = stop_flag.read_text(encoding="utf-8").strip() or None
        except Exception:
            stop_flag_reason = None

    if stop_reason is not None or stop_flag_reason:
        if not early_stop_triggered:
            stop_message = "⏸️ Training stopped by user."
            logger.info("Training stopped before export; reason=%s", stop_flag_reason or stop_reason)
            update_status(
                project_dir,
                "stopped",
                progress=60,
                stage="training",
                message=stop_message,
                stopped_stage="training",
                stopped_step=stop_reason if isinstance(stop_reason, int) else None,
            )
            if stop_flag.exists():
                stop_flag.unlink()
            return stop_reason if isinstance(stop_reason, int) else 1

        logger.info(
            "Early stop confirmed at step %s; continuing to export artifacts.",
            early_stop_state.get("trigger_step") if isinstance(early_stop_state, dict) else None,
        )

    logger.info("Training complete, exporting final checkpoint...")
    update_status(
        project_dir, "processing", stage="training", stage_progress=100,
        message="✅ Gsplat training complete",
        timing={"start": gsplat_start, "end": gsplat_end, "elapsed": gsplat_end - gsplat_start}
    )
    update_status(project_dir, "completed", progress=90, stage="training", message="Training done")
    time.sleep(1.5)
    update_status(project_dir, "processing", stage="export", stage_progress=10, message="📦 Preparing export of final artifacts...")
    ckpt_dir = engine_output_dir / "ckpts"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    snapshots_dir = engine_output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    exported_snapshots = 0
    for ckpt in ckpts:
        step_zero = parse_step_from_name(ckpt.stem, "ckpt_")
        if step_zero is None:
            continue
        step = step_zero + 1
        if splat_interval and step % splat_interval != 0 and step != max_steps:
            continue
        snapshot_name = f"snapshot_step_{step:06d}.splat"
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            continue
        try:
            export_with_gsplat(
                ckpt,
                snapshots_dir,
                splat_name=snapshot_name,
                export_ply=False,
            )
            exported_snapshots += 1
        except Exception as exc:
            logger.warning("Failed snapshot export for %s: %s", ckpt.name, exc)

    if ckpts:
        latest = ckpts[-1]
        logger.info("Exporting checkpoint: %s", latest)
        update_status(project_dir, "processing", stage="export", stage_progress=40, message="📝 Exporting .splat file...")
        if stop_flag.exists() and not early_stop_triggered:
            update_status(project_dir, "stopped", progress=0, stop_requested=True, stage="export", message="⏸️ Processing stopped by user before export.", stopped_stage="export")
            try:
                stop_flag.unlink()
            except Exception:
                pass
            return None
        export_with_gsplat(latest, engine_output_dir)
        update_status(project_dir, "processing", stage="export", stage_progress=100, message="✅ Export complete")
    else:
        logger.info("No saved checkpoints found; skipping checkpoint export")

    if exported_snapshots:
        logger.info("Exported %d interval snapshot(s) to %s", exported_snapshots, snapshots_dir)

    materialize_eval_previews(engine_output_dir)
    eval_history = collect_eval_history(engine_output_dir, p, mode)
    elapsed_by_step = tuning_state.get("elapsed_by_step") if isinstance(tuning_state.get("elapsed_by_step"), dict) else {}
    loss_by_step = tuning_state.get("loss_by_step") if isinstance(tuning_state.get("loss_by_step"), dict) else {}
    if eval_history:
        for row in eval_history:
            if not isinstance(row, dict):
                continue
            step_value = row.get("step")
            if not isinstance(step_value, (int, float)):
                continue
            step_int = int(step_value)
            measured_loss = loss_by_step.get(step_int)
            measured_elapsed = elapsed_by_step.get(step_int)
            row["final_loss"] = float(measured_loss) if isinstance(measured_loss, (int, float)) else None
            row["elapsed_seconds"] = float(measured_elapsed) if isinstance(measured_elapsed, (int, float)) else None
    if eval_history and mode == "modified":
        for row in eval_history:
            tuning_params = row.get("tuning_params") if isinstance(row, dict) else None
            if isinstance(tuning_params, dict):
                tuning_params["modified_rule_updates"] = int(tuning_state.get("updates", 0) or 0)
                if tuning_state.get("last_event"):
                    tuning_params["modified_last_rule_step"] = tuning_state["last_event"].get("step")
    if eval_history:
        final_eval = eval_history[-1]
        write_json_atomic(engine_output_dir / "eval_history.json", eval_history)
        final_metrics = {
            "lpips_score": final_eval.get("lpips_mean"),
            "sharpness": final_eval.get("sharpness_mean"),
            "convergence_speed": final_eval.get("convergence_speed"),
            "final_loss": final_eval.get("final_loss"),
            "gaussian_count": final_eval.get("num_gaussians"),
        }
        adaptive_payload = {
            "mode": mode,
            "tune_scope": tune_scope if mode == "modified" else None,
            "final_evaluation": final_metrics,
            "tune_start_step": modified_tune_start_step if mode == "modified" else None,
            "tune_end_step": modified_tune_end_step if mode == "modified" else final_eval.get("step"),
            "tune_interval": modified_tune_interval if mode == "modified" else None,
            "tune_min_improvement": tune_min_improvement if mode == "modified" else None,
            "strategy_tune_start_step": strategy_tune_start_step if mode == "modified" else None,
            "strategy_tune_end_step": strategy_tune_end_step if mode == "modified" else None,
            "tuning_history": list(tuning_state.get("events") or []),
            "final_params": (tuning_state.get("last_event") or {}).get("params", {}),
        }
        write_json_atomic(engine_output_dir / "adaptive_tuning_results.json", adaptive_payload)

        metadata_path = engine_output_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}
        metadata["evaluation_metrics"] = final_metrics
        metadata["final_metrics"] = {
            "convergence_speed": final_eval.get("convergence_speed"),
            "final_loss": final_eval.get("final_loss"),
            "lpips_mean": final_eval.get("lpips_mean"),
            "sharpness_mean": final_eval.get("sharpness_mean"),
        }

        input_mode_learning_payload = None
        if use_html_input_mode_flow:
            selected_preset = str((preset_summary or {}).get("selected_preset") or "")
            yhat_scores = dict((preset_summary or {}).get("yhat_scores") or {})
            mode_name = str((preset_summary or {}).get("mode") or "")
            baseline_eval_history: list[dict] = []
            baseline_session_id = str(p.get("baseline_session_id") or "").strip()
            if baseline_session_id:
                baseline_eval_path = (
                    project_dir / "runs" / baseline_session_id / "outputs" / "engines" / "gsplat" / "eval_history.json"
                )
                if baseline_eval_path.exists():
                    try:
                        raw = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
                        if isinstance(raw, list):
                            baseline_eval_history = [row for row in raw if isinstance(row, dict)]
                    except Exception as exc:
                        logger.warning("Failed loading baseline eval history from %s: %s", baseline_eval_path, exc)
            if selected_preset and mode_name:
                try:
                    input_mode_learning_payload = update_from_run(
                        project_dir=project_dir,
                        mode=mode_name,
                        selected_preset=selected_preset,
                        yhat_scores=yhat_scores,
                        eval_history=eval_history,
                        baseline_eval_history=baseline_eval_history,
                        loss_by_step={int(k): float(v) for k, v in (loss_by_step or {}).items()},
                        elapsed_by_step={int(k): float(v) for k, v in (elapsed_by_step or {}).items()},
                        x_features=dict((preset_summary or {}).get("features") or {}),
                        run_id=run_session_id,
                        logger=logger,
                    )
                    write_json_atomic(
                        engine_output_dir / "input_mode_learning_results.json",
                        input_mode_learning_payload,
                    )
                except Exception as exc:
                    logger.warning("Failed input-mode learner update: %s", exc)

        metadata["num_gaussians"] = final_eval.get("num_gaussians")
        metadata["mode"] = mode
        metadata["tune_scope"] = tune_scope if mode == "modified" else None
        metadata["best_splat"] = tuning_state.get("best_splat")
        metadata["early_stop"] = tuning_state.get("early_stop")
        metadata["stop_reason"] = stop_flag_reason or (f"runner_stop_step={stop_reason}" if isinstance(stop_reason, int) else None)
        if input_mode_learning_payload is not None:
            metadata["input_mode_learning"] = input_mode_learning_payload
        write_json_atomic(metadata_path, metadata)

        loss_milestones: dict[str, float] = {}
        eval_series: list[dict] = []
        eval_time_series: list[dict] = []
        eval_psnr_series: list[dict] = []
        eval_ssim_series: list[dict] = []
        eval_lpips_series: list[dict] = []
        log_loss_series: list[dict] = []
        log_time_series: list[dict] = []
        for point in eval_history:
            if not isinstance(point, dict):
                continue
            step_value = point.get("step")
            if isinstance(step_value, (int, float)):
                step_int = int(step_value)
                loss_value = point.get("final_loss")
                if isinstance(loss_value, (int, float)):
                    eval_series.append({"step": step_int, "loss": float(loss_value)})

                elapsed_value = point.get("elapsed_seconds")
                if isinstance(elapsed_value, (int, float)) and float(elapsed_value) >= 0:
                    eval_time_series.append({
                        "step": step_int,
                        "elapsed_seconds": float(elapsed_value),
                    })

                psnr_value = point.get("convergence_speed")
                if isinstance(psnr_value, (int, float)):
                    eval_psnr_series.append({"step": step_int, "value": float(psnr_value)})

                ssim_value = point.get("sharpness_mean")
                if isinstance(ssim_value, (int, float)):
                    eval_ssim_series.append({"step": step_int, "value": float(ssim_value)})

                lpips_value = point.get("lpips_mean")
                if isinstance(lpips_value, (int, float)):
                    eval_lpips_series.append({"step": step_int, "value": float(lpips_value)})

            for key, value in point.items():
                if isinstance(key, str) and key.startswith("loss_at_") and isinstance(value, (int, float)):
                    loss_milestones[key] = float(value)

        if isinstance(loss_by_step, dict):
            for step_value, loss_value in loss_by_step.items():
                try:
                    step_int = int(step_value)
                except Exception:
                    continue
                if isinstance(loss_value, (int, float)):
                    log_loss_series.append({"step": step_int, "loss": float(loss_value)})
        if log_loss_series:
            log_loss_series.sort(key=lambda row: int(row.get("step", 0)))

        if isinstance(elapsed_by_step, dict):
            for step_value, elapsed_value in elapsed_by_step.items():
                try:
                    step_int = int(step_value)
                except Exception:
                    continue
                if isinstance(elapsed_value, (int, float)) and float(elapsed_value) >= 0:
                    log_time_series.append({"step": step_int, "elapsed_seconds": float(elapsed_value)})
        if log_time_series:
            log_time_series.sort(key=lambda row: int(row.get("step", 0)))

        runtime_tuning_series = [
            {"step": item.get("step"), "params": item.get("params")}
            for item in list(tuning_state.get("events") or [])
            if isinstance(item, dict) and isinstance(item.get("step"), (int, float)) and isinstance(item.get("params"), dict)
        ]

        final_loss_value = final_eval.get("final_loss")
        if not isinstance(final_loss_value, (int, float)):
            final_loss_value = None

        total_time_seconds = max(0.0, float(time.time() - gsplat_start))
        run_name = str(p.get("run_name") or p.get("run_id") or project_dir.name)
        run_id_value = str(p.get("run_id") or project_dir.name)

        project_id = None
        try:
            if project_dir.parent.name == "runs":
                project_id = project_dir.parent.parent.name
        except Exception:
            project_id = None

        summary_payload = {
            "project_id": project_id,
            "run_id": run_id_value,
            "run_name": run_name,
            "name": project_id,
            "status": "completed",
            "mode": mode,
            "engine": "gsplat",
            "metrics": {
                "convergence_speed": final_eval.get("convergence_speed"),
                "final_loss": final_loss_value,
                "lpips_mean": final_eval.get("lpips_mean"),
                "sharpness_mean": final_eval.get("sharpness_mean"),
                "num_gaussians": final_eval.get("num_gaussians"),
                "total_time_seconds": total_time_seconds,
                "best_splat_step": (
                    (tuning_state.get("best_splat") or {}).get("step")
                    if isinstance(tuning_state.get("best_splat"), dict)
                    else None
                ),
                "best_splat_loss": (
                    (tuning_state.get("best_splat") or {}).get("loss")
                    if isinstance(tuning_state.get("best_splat"), dict)
                    else None
                ),
                "stopped_early": bool(
                    isinstance(tuning_state.get("early_stop"), dict)
                    and (tuning_state.get("early_stop") or {}).get("triggered")
                ),
                "early_stop_step": (
                    (tuning_state.get("early_stop") or {}).get("trigger_step")
                    if isinstance(tuning_state.get("early_stop"), dict)
                    else None
                ),
            },
            "tuning": {
                "initial": (eval_history[0].get("tuning_params") if isinstance(eval_history[0], dict) else {}) or {},
                "final": (final_eval.get("tuning_params") if isinstance(final_eval, dict) else {}) or {},
                "end_params": (tuning_state.get("last_event") or {}).get("params", {}) if mode == "modified" else {},
                "end_step": modified_tune_end_step if mode == "modified" else final_eval.get("step"),
                "runs": metadata.get("tuning_runs") if isinstance(metadata, dict) else None,
                "history_count": len(list(tuning_state.get("events") or [])),
                "history": list(tuning_state.get("events") or []),
                "tune_interval": p.get("tune_interval"),
                "log_interval": p.get("log_interval"),
                "runtime_series": runtime_tuning_series,
            },
            "major_params": {
                "max_steps": p.get("max_steps"),
                "total_steps_completed": final_eval.get("step"),
                "densify_from_iter": p.get("densify_from_iter"),
                "densify_until_iter": p.get("densify_until_iter"),
                "densification_interval": p.get("densification_interval"),
                "eval_interval": p.get("eval_interval"),
                "save_interval": p.get("save_interval"),
                "splat_export_interval": p.get("splat_export_interval"),
                "best_splat_interval": p.get("best_splat_interval"),
                "best_splat_start_step": p.get("best_splat_start_step"),
                "auto_early_stop": p.get("auto_early_stop"),
                "early_stop_monitor_interval": p.get("early_stop_monitor_interval"),
                "early_stop_decision_points": p.get("early_stop_decision_points"),
                "early_stop_min_eval_points": p.get("early_stop_min_eval_points"),
                "early_stop_min_step_ratio": p.get("early_stop_min_step_ratio"),
                "early_stop_monitor_min_relative_improvement": p.get("early_stop_monitor_min_relative_improvement"),
                "early_stop_eval_min_relative_improvement": p.get("early_stop_eval_min_relative_improvement"),
                "early_stop_max_volatility_ratio": p.get("early_stop_max_volatility_ratio"),
                "early_stop_ema_alpha": p.get("early_stop_ema_alpha"),
                "batch_size": p.get("batch_size"),
            },
            "loss_milestones": loss_milestones,
            "log_loss_series": log_loss_series,
            "log_time_series": log_time_series,
            "eval_series": eval_series,
            "eval_time_series": eval_time_series,
            "eval_psnr_series": eval_psnr_series,
            "eval_ssim_series": eval_ssim_series,
            "eval_lpips_series": eval_lpips_series,
            "preview_url": None,
            "eval_points": len(eval_history),
            "early_stop": tuning_state.get("early_stop") if isinstance(tuning_state.get("early_stop"), dict) else None,
        }

        run_artifact_root = project_dir
        if configured_run_id:
            candidate_run_root = project_dir / "runs" / configured_run_id
            if candidate_run_root.exists():
                run_artifact_root = candidate_run_root

        comparison_dir = run_artifact_root / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(comparison_dir / "experiment_summary.json", summary_payload)

        for artifact_name in ("eval_history.json", "adaptive_tuning_results.json", "metadata.json"):
            source_path = engine_output_dir / artifact_name
            if source_path.exists():
                try:
                    shutil.copy2(source_path, comparison_dir / artifact_name)
                except Exception as exc:
                    logger.warning("Failed to copy %s into comparison folder: %s", artifact_name, exc)
        run_cfg_source = run_artifact_root / "run_config.json"
        if run_cfg_source.exists():
            try:
                shutil.copy2(run_cfg_source, comparison_dir / "run_config.json")
            except Exception as exc:
                logger.warning("Failed to copy run_config.json into comparison folder: %s", exc)

    # Persist durable training timing into metadata.json so total elapsed survives log rotation.
    training_time_payload = {
        "start_unix": float(gsplat_start),
        "end_unix": float(gsplat_end),
        "total_elapsed_seconds": max(0.0, float(gsplat_end - gsplat_start)),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gsplat_start)),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gsplat_end)),
    }
    metadata_path = engine_output_dir / "metadata.json"
    metadata_timing = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata_timing = json.load(handle)
        except Exception:
            metadata_timing = {}
    if not isinstance(metadata_timing, dict):
        metadata_timing = {}
    metadata_timing["training_time"] = training_time_payload
    metadata_timing["total_time_seconds"] = float(training_time_payload["total_elapsed_seconds"])
    write_json_atomic(metadata_path, metadata_timing)

    if stop_flag.exists():
        stop_flag.unlink()

    return None
