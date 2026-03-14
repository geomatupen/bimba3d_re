"""Background watcher that mirrors LiteGS artifacts into frontend-friendly outputs."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from litegs.io_manager import load_ply
from gsplat.exporter import export_splats

logger = logging.getLogger(__name__)


class LiteGSOutputWatcher:
    """Polls the LiteGS artifact directory for new point clouds/checkpoints."""

    def __init__(
        self,
        model_root: Path,
        output_dir: Path,
        status_callback: Optional[Callable[[str], None]] = None,
        poll_interval: float = 15.0,
    ) -> None:
        self.model_root = Path(model_root)
        self.output_dir = Path(output_dir)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.previews_dir = self.output_dir / "previews"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.status_callback = status_callback
        self.poll_interval = poll_interval

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()
        self._processed_plys: set[Path] = set()
        self._processed_ckpts: set[Path] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        logger.info("Starting LiteGS artifact watcher at %s", self.model_root)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        # Run one last scan to pick up trailing artifacts
        self.drain_once()
        self._thread.join(timeout=timeout)

    def drain_once(self) -> None:
        try:
            self._scan_point_clouds()
            self._scan_checkpoints()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("LiteGS watcher drain failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._scan_point_clouds()
                self._scan_checkpoints()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("LiteGS watcher iteration failed: %s", exc)
            self._stop_event.wait(self.poll_interval)

    def _scan_point_clouds(self) -> None:
        point_cloud_root = self.model_root / "point_cloud"
        if not point_cloud_root.exists():
            return
        for ply_path in sorted(point_cloud_root.glob("**/point_cloud.ply")):
            if ply_path in self._processed_plys:
                continue
            self._convert_point_cloud(ply_path)
            self._processed_plys.add(ply_path)

    def _scan_checkpoints(self) -> None:
        ckpt_dir = self.model_root / "checkpoints"
        if not ckpt_dir.exists():
            return
        for ckpt_path in sorted(ckpt_dir.glob("chkpnt*.pth")):
            if ckpt_path in self._processed_ckpts:
                continue
            target = self.checkpoints_dir / ckpt_path.name
            try:
                shutil.copy2(ckpt_path, target)
                self._processed_ckpts.add(ckpt_path)
                logger.info("Mirrored LiteGS checkpoint -> %s", target)
                self._notify_status(f"🧊 Checkpoint available ({ckpt_path.name})")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to copy checkpoint %s: %s", ckpt_path, exc)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _convert_point_cloud(self, ply_path: Path) -> None:
        try:
            sh_degree = 3
            xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(str(ply_path), sh_degree)
            means = torch.from_numpy(xyz.T).float().contiguous()
            scales = torch.from_numpy(scale.T).float().contiguous()
            quats = torch.from_numpy(rot.T).float().contiguous()
            opacities = torch.from_numpy(opacity.T).float().squeeze(-1).contiguous()
            sh0_tensor = torch.from_numpy(sh_0).float().permute(2, 1, 0).contiguous()
            sh_rest_tensor = torch.from_numpy(sh_rest).float().permute(2, 1, 0).contiguous()

            tag = ply_path.parent.name
            snapshot_name = f"litegs_{tag}.splat"
            snapshot_path = self.snapshots_dir / snapshot_name
            tmp_path = snapshot_path.with_suffix(".tmp")
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0_tensor,
                shN=sh_rest_tensor,
                format="splat",
                save_to=str(tmp_path),
            )
            tmp_path.replace(snapshot_path)
            self._refresh_latest(snapshot_path)
            self._write_preview(means.numpy(), tag)
            logger.info("Converted LiteGS point cloud -> %s", snapshot_name)
            self._notify_status(f"🌀 LiteGS snapshot ready ({tag})")
        except Exception as exc:  # pragma: no cover - conversion best effort
            logger.warning("Failed to convert LiteGS point cloud %s: %s", ply_path, exc)

    def _refresh_latest(self, snapshot_path: Path) -> None:
        splat_latest = self.output_dir / "splats.splat"
        try:
            if splat_latest.exists() or splat_latest.is_symlink():
                splat_latest.unlink()
            shutil.copy2(snapshot_path, splat_latest)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to refresh splats.splat: %s", exc)

    def _write_preview(self, points_xyz: np.ndarray, tag: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pts = points_xyz.T
            if pts.shape[0] > 150_000:
                idx = np.random.choice(pts.shape[0], 150_000, replace=False)
                pts = pts[idx]
            colors = pts[:, 2]
            colors = (colors - colors.min()) / (colors.ptp() + 1e-6)
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=0.2, cmap="viridis")
            ax.set_axis_off()
            ax.set_title(f"LiteGS {tag}")
            fig.tight_layout(pad=0)
            preview_path = self.previews_dir / f"preview_{tag}.png"
            fig.savefig(preview_path, bbox_inches="tight", pad_inches=0)
            fig.savefig(self.previews_dir / "preview_latest.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - preview best effort
            logger.debug("Failed to save LiteGS preview for %s: %s", tag, exc)

    def _notify_status(self, message: str) -> None:
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:  # pragma: no cover - callback best effort
                pass
