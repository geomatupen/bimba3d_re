from pathlib import Path

from .gsplat_engine import run_training as run_gsplat_training
from .litegs_engine import run_training as run_litegs_training

SUPPORTED_ENGINES = {"gsplat", "litegs"}
ENGINE_LABELS = {
    "gsplat": "Gaussian Splatting",
    "litegs": "LiteGS",
}


ENGINE_RUNNERS = {
    "gsplat": run_gsplat_training,
    "litegs": run_litegs_training,
}


def run_selected_engine(
    engine: str,
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    params: dict,
    *,
    resume: bool,
    context: dict,
):
    try:
        runner = ENGINE_RUNNERS[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported training engine: {engine}") from exc
    return runner(
        image_dir,
        colmap_dir,
        output_dir,
        params,
        resume=resume,
        context=context,
    )
