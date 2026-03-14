"""Resume detection utilities for interrupted pipelines."""
from pathlib import Path
from typing import Optional, Dict
from bimba3d_backend.app.config import DATA_DIR


def _collect_checkpoint_files(output_dir: Path) -> list[Path]:
    """Collect checkpoints from engine-scoped directories."""
    candidates = [
        output_dir / "engines" / "gsplat" / "ckpts",
        output_dir / "engines" / "litegs" / "checkpoints",
    ]
    files: list[Path] = []
    for ckpt_dir in candidates:
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            continue
        files.extend(sorted(ckpt_dir.glob("ckpt_*.pt")))
        files.extend(sorted(ckpt_dir.glob("chkpnt*.pth")))
    return files


def _has_final_outputs(output_dir: Path) -> bool:
    """Check for final splat outputs in engine-scoped locations."""
    candidates = [
        output_dir / "engines" / "gsplat",
        output_dir / "engines" / "litegs",
    ]
    for root in candidates:
        if (root / "splats.splat").exists() or (root / "splats.ply").exists() or (root / "splats.bin").exists():
            return True
    return False


def can_resume_project(project_id: str) -> Dict[str, any]:
    """
    Check if a project has resumable state.
    Returns dict with:
      - can_resume: bool
      - has_sparse: bool (COLMAP completed)
      - has_checkpoints: bool (training checkpoints exist)
      - last_checkpoint_step: Optional[int]
    """
    project_dir = DATA_DIR / project_id
    output_dir = project_dir / "outputs"
    
    # Check for COLMAP sparse output
    sparse_dir = output_dir / "sparse" / "0"
    has_sparse = sparse_dir.exists() and any(sparse_dir.iterdir())
    
    # Check for training checkpoints
    has_checkpoints = False
    last_checkpoint_step = None

    checkpoints = _collect_checkpoint_files(output_dir)
    if checkpoints:
        has_checkpoints = True
        latest_ckpt = checkpoints[-1]
        try:
            if latest_ckpt.stem.startswith("ckpt_"):
                step_str = latest_ckpt.stem.split("_")[1]
                last_checkpoint_step = int(step_str) + 1
            else:
                digits = "".join(ch for ch in latest_ckpt.stem if ch.isdigit())
                if digits:
                    last_checkpoint_step = int(digits)
        except (IndexError, ValueError):
            pass
    
    # Check for full completion: status.json status=="completed" and all outputs present
    status_file = project_dir / "status.json"
    fully_completed = False
    if status_file.exists():
        try:
            import json
            with open(status_file) as f:
                s = json.load(f)
            if s.get("status") == "completed":
                if _has_final_outputs(output_dir):
                    fully_completed = True
        except Exception:
            pass
    can_resume = (has_sparse or has_checkpoints) and not fully_completed
    return {
        "can_resume": can_resume,
        "has_sparse": has_sparse,
        "has_checkpoints": has_checkpoints,
        "last_checkpoint_step": last_checkpoint_step,
    }
