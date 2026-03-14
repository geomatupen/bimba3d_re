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
    """Run LiteGS training pipeline and export artifacts."""
    logger = context["logger"]
    update_status = context["update_status"]
    ensure_symlink = context["ensure_symlink"]
    prepare_pinhole_sparse_for_litegs = context["prepare_pinhole_sparse_for_litegs"]
    patch_litegs_opacity_decay = context["patch_litegs_opacity_decay"]
    find_latest_litegs_checkpoint = context["find_latest_litegs_checkpoint"]
    export_litegs_outputs = context["export_litegs_outputs"]

    project_dir = output_dir.parent
    stop_flag = project_dir / "stop_requested"
    engine_label = "LiteGS"

    try:
        import torch
        device_label = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_label = "cuda"

    status_message = f"🎯 Starting {engine_label} training..."
    update_status(
        project_dir,
        "processing",
        progress=55,
        stage="training",
        stage_progress=5,
        message=status_message,
        device=device_label,
        engine="litegs",
    )

    if stop_flag.exists():
        update_status(
            project_dir,
            "stopped",
            progress=55,
            stage="training",
            message="⏸️ Processing stopped before LiteGS training.",
            stop_requested=True,
            stopped_stage="training",
        )
        try:
            stop_flag.unlink()
        except Exception:
            pass
        return 0

    try:
        import litegs  # pylint: disable=import-error
        from litegs import config as litegs_config  # pylint: disable=import-error
        from litegs import training as litegs_training  # pylint: disable=import-error
    except Exception as exc:
        logger.error("LiteGS import failed: %s", exc)
        update_status(project_dir, "failed", error=f"LiteGS not installed: {exc}")
        raise

    dataset_root = output_dir / "litegs" / "dataset"
    model_root = output_dir / "litegs" / "artifacts"
    dataset_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    ensure_symlink(Path(image_dir), dataset_root / "images")
    colmap_sparse_root = Path(colmap_dir)

    processed_sparse = prepare_pinhole_sparse_for_litegs(colmap_sparse_root, output_dir)

    litegs_sparse_root = dataset_root / "sparse"
    if litegs_sparse_root.is_symlink():
        litegs_sparse_root.unlink()
    litegs_sparse_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(processed_sparse, litegs_sparse_root / "0")

    lp, op, pp, dp = litegs_config.get_default_arg()
    lp.source_path = str(dataset_root)
    lp.model_path = str(model_root)
    if hasattr(lp, "images"):
        lp.images = "images"

    training_summary = {
        "iterations": getattr(op, "iterations", None),
        "target_primitives": getattr(dp, "target_primitives", None),
        "alpha_shrink": None,
        "sh_degree": getattr(lp, "sh_degree", 3),
    }

    user_steps = params.get("max_steps")
    if user_steps is not None:
        try:
            op.iterations = max(1, int(user_steps))
            training_summary["iterations"] = op.iterations
        except Exception:
            logger.warning("Invalid LiteGS max_steps override: %s", user_steps)

    target_override = params.get("litegs_target_primitives")
    if target_override is not None:
        try:
            dp.target_primitives = max(1, int(target_override))
        except Exception:
            logger.warning("Invalid LiteGS target primitive override: %s", target_override)
    training_summary["target_primitives"] = getattr(dp, "target_primitives", None)

    alpha_shrink = params.get("litegs_alpha_shrink", 0.95)
    try:
        alpha_shrink = float(alpha_shrink)
    except Exception:
        alpha_shrink = 0.95
    if alpha_shrink <= 0:
        alpha_shrink = 0.95
    training_summary["alpha_shrink"] = alpha_shrink
    patch_litegs_opacity_decay(alpha_shrink)

    start_checkpoint = None
    if resume:
        ckpt_path = find_latest_litegs_checkpoint(model_root)
        if ckpt_path:
            start_checkpoint = str(ckpt_path)
            logger.info("Resuming LiteGS from %s", ckpt_path)
        else:
            logger.info("LiteGS resume requested but no checkpoints found; starting fresh")

    litegs_start = time.time()
    try:
        litegs_training.start(lp, op, pp, dp, [], [], [], start_checkpoint)
    except Exception as exc:
        logger.error("LiteGS training failed: %s", exc, exc_info=True)
        update_status(project_dir, "failed", error=str(exc), stage="training", message=str(exc))
        raise

    if stop_flag.exists():
        logger.info("Stop requested after LiteGS training finished")
        return 0

    update_status(
        project_dir,
        "processing",
        stage="training",
        stage_progress=85,
        message=f"✅ {engine_label} training complete",
        device=device_label,
    )

    export_litegs_outputs(model_root, output_dir, colmap_sparse_root, training_summary)

    litegs_end = time.time()
    update_status(
        project_dir,
        "processing",
        stage="export",
        stage_progress=100,
        message="✅ LiteGS export complete",
        timing={"start": litegs_start, "end": litegs_end, "elapsed": litegs_end - litegs_start},
        device=device_label,
    )

    return None
