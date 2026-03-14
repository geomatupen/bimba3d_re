import json
import os
import shutil
import time
from pathlib import Path


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

    p = params or {}
    mode = p.get("mode", "baseline")
    max_steps = int(p.get("max_steps", 30_000))
    splat_interval = p.get("splat_export_interval")
    try:
        splat_interval = max(1, int(splat_interval)) if splat_interval is not None else None
    except Exception:
        splat_interval = None
    log_interval = p.get("log_interval", 100)
    try:
        log_interval = max(1, int(log_interval))
    except Exception:
        log_interval = 100
    project_dir = base_output_dir.parent
    stop_flag = project_dir / "stop_requested"
    gsplat_start = time.time()
    tuning_state: dict[str, object] = {"applied": False, "event": None}
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
            tuning_applied = bool(tuning_state.get("applied"))

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

    def apply_modified_profile(step: int) -> bool:
        if mode != "modified" or tuning_state.get("applied"):
            return False
        if step < 200:
            return False

        runner_obj = runner_ref.get("runner")
        if runner_obj is None:
            return False

        try:
            lr_multipliers = {
                "means": 0.85,
                "opacities": 0.80,
                "scales": 0.95,
                "quats": 0.95,
                "sh0": 1.10,
                "shN": 1.10,
            }
            before_lrs: dict[str, float] = {}
            after_lrs: dict[str, float] = {}
            for name, mult in lr_multipliers.items():
                optimizer = getattr(runner_obj, "optimizers", {}).get(name)
                if optimizer is None or not optimizer.param_groups:
                    continue
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                before_lrs[name] = current_lr
                new_lr = current_lr * float(mult)
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                after_lrs[name] = new_lr

            strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
            strategy_before: dict[str, float] = {}
            strategy_after: dict[str, float] = {}
            if strategy is not None:
                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    strategy_before[key] = float(getattr(strategy, key))

                strategy.grow_grad2d = max(5e-5, float(strategy.grow_grad2d) * 0.85)
                strategy.prune_opa = max(1e-4, float(strategy.prune_opa) * 0.90)
                strategy.refine_every = max(25, int(float(strategy.refine_every) * 0.80))
                strategy.reset_every = max(strategy.refine_every, int(float(strategy.reset_every) * 0.85))

                for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
                    strategy_after[key] = float(getattr(strategy, key))

            event = {
                "step": int(step),
                "adjustments": [
                    "step200_deterministic_profile",
                    "lr_rebalance_means_opacity_sh",
                    "densification_threshold_and_cadence_shift",
                ],
                "params": {
                    "learning_rates": after_lrs,
                    "strategy": strategy_after,
                },
                "before": {
                    "learning_rates": before_lrs,
                    "strategy": strategy_before,
                },
            }
            tuning_state["event"] = event
            tuning_state["applied"] = True

            update_status(
                project_dir,
                "processing",
                mode=mode,
                tuning_active=True,
                last_tuning={
                    "step": int(step),
                    "action": "Step-200 profile update",
                    "reason": "Modified mode deterministic tuning",
                },
            )
            logger.info("Applied modified-mode profile at step %d", step)
            return True
        except Exception as exc:
            logger.warning("Failed to apply modified-mode profile at step %d: %s", step, exc)
            return False

    def stop_checker() -> bool:
        return stop_flag.exists()

    def progress_callback(step: int, max_steps_local: int, loss: float) -> None:
        apply_modified_profile(step)
        requested_stop = stop_checker()
        status_text = "stopping" if requested_stop else "processing"
        progress_fraction = 0.0 if max_steps_local <= 0 else float(step) / float(max_steps_local)
        progress_fraction = max(0.0, min(1.0, progress_fraction))
        message = (
            f"⏸️ Stopping after step {step}/{max_steps_local} completes (loss: {loss:.6f})..."
            if requested_stop
            else f"🎯 Training step {step}/{max_steps_local} (loss: {loss:.6f})"
        )

        now = time.time()
        elapsed = now - gsplat_start
        eta = (
            (elapsed / progress_fraction) * (1 - progress_fraction)
            if progress_fraction > 0
            else None
        )
        timing = {"start": gsplat_start, "elapsed": elapsed}
        if eta is not None:
            timing["eta"] = eta

        update_status(
            project_dir,
            status_text,
            progress=60 + int(progress_fraction * 35),
            mode=mode,
            currentStep=step,
            maxSteps=max_steps_local,
            stop_requested=requested_stop,
            stage="training",
            stage_progress=int(progress_fraction * 100),
            message=message,
            timing=timing,
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

    try:
        import torch
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

    dataset_dir = engine_output_dir
    images_link = dataset_dir / "images"
    sparse_zero = dataset_dir / "sparse" / "0"
    sparse_zero.parent.mkdir(parents=True, exist_ok=True)
    for link_path, target in ((images_link, image_dir), (sparse_zero, colmap_dir)):
        if link_path.exists() or link_path.is_symlink():
            try:
                if link_path.is_symlink() or link_path.is_file():
                    link_path.unlink()
                else:
                    shutil.rmtree(link_path)
            except Exception:
                pass
        os.symlink(str(Path(target).resolve()), str(link_path), target_is_directory=True)

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
        _build_steps(p.get("save_interval"), [7000, 30000])
        + _build_steps(p.get("splat_export_interval"), [7000, 30000])
    ))

    cfg = Config(
        disable_viewer=True,
        disable_video=True,
        load_exposure=False,
        data_dir=str(dataset_dir),
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
    cfg.progress_every = max(1, int(log_interval))
    if cfg.disable_tqdm:
        os.environ["TQDM_DISABLE"] = "1"
    else:
        os.environ.pop("TQDM_DISABLE", None)

    logger.info(
        "GSPLAT logging cadence: snapshot every %d steps (config key: log_interval); tqdm=%s",
        log_interval,
        "disabled" if cfg.disable_tqdm else "enabled",
    )

    cfg.stop_checker = stop_checker
    cfg.progress_callback = progress_callback

    snapshots_dir = engine_output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    def eval_callback(step: int) -> None:
        materialize_eval_previews(engine_output_dir, eval_step=step)

    def checkpoint_callback(step: int, checkpoint_path: str) -> None:
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
        raise RuntimeError("Upstream simple_trainer currently requires CUDA in this worker path")

    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    runner_ref["runner"] = runner
    stop_reason = runner.train()
    gsplat_end = time.time()

    if stop_reason is not None or stop_flag.exists():
        logger.info("Training stopped by user, skipping export and completion status.")
        update_status(
            project_dir,
            "stopped",
            progress=60,
            stage="training",
            message="⏸️ Training stopped by user.",
            stopped_stage="training",
            stopped_step=stop_reason if isinstance(stop_reason, int) else None,
        )
        if stop_flag.exists():
            stop_flag.unlink()
        return stop_reason if isinstance(stop_reason, int) else 1

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
        if stop_flag.exists():
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
    if eval_history and mode == "modified":
        for row in eval_history:
            tuning_params = row.get("tuning_params") if isinstance(row, dict) else None
            if isinstance(tuning_params, dict):
                tuning_params["modified_profile_applied"] = bool(tuning_state.get("applied"))
                if tuning_state.get("event"):
                    tuning_params["modified_profile_step"] = tuning_state["event"].get("step")
    if eval_history:
        write_json_atomic(engine_output_dir / "eval_history.json", eval_history)
        final_eval = eval_history[-1]
        final_metrics = {
            "lpips_score": final_eval.get("lpips_mean"),
            "sharpness": final_eval.get("sharpness_mean"),
            "convergence_speed": final_eval.get("convergence_speed"),
            "final_loss": final_eval.get("final_loss"),
            "gaussian_count": final_eval.get("num_gaussians"),
        }
        adaptive_payload = {
            "mode": mode,
            "final_evaluation": final_metrics,
            "tune_end_step": final_eval.get("step"),
            "tuning_history": [tuning_state["event"]] if tuning_state.get("event") else [],
            "final_params": (tuning_state.get("event") or {}).get("params", {}),
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
        metadata["num_gaussians"] = final_eval.get("num_gaussians")
        metadata["mode"] = mode
        write_json_atomic(metadata_path, metadata)

    if stop_flag.exists():
        stop_flag.unlink()

    return None
