"""
Gaussian Splatting Trainer based on nerfstudio-project/gsplat.
Faithful implementation with research intervention hooks.
"""

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

# Torch Hub downloads (pulled in by torchvision/torchmetrics) spam tqdm progress bars
# into project processing logs by default. Monkey patch the helpers once so any
# downstream model weights download quietly.
try:  # pragma: no cover - defensive guard only
    import torch.hub as _torch_hub
except Exception:  # torch hub may not exist in stripped builds
    _torch_hub = None
else:
    _orig_download_url_to_file = _torch_hub.download_url_to_file

    def _download_url_to_file_no_progress(url, dst, hash_prefix=None, progress=True):
        return _orig_download_url_to_file(url, dst, hash_prefix, False)

    _torch_hub.download_url_to_file = _download_url_to_file_no_progress

    _orig_load_state_dict_from_url = _torch_hub.load_state_dict_from_url

    def _load_state_dict_from_url_no_progress(
        url,
        model_dir=None,
        map_location=None,
        progress=True,
        check_hash=False,
        file_name=None,
        weights_only=False,
        **kwargs,
    ):
        return _orig_load_state_dict_from_url(
            url,
            model_dir,
            map_location,
            False,
            check_hash,
            file_name,
            weights_only,
            **kwargs,
        )

    _torch_hub.load_state_dict_from_url = _load_state_dict_from_url_no_progress

    try:  # legacy alias used by some deps (lpips, torchvision <= 0.16)
        import torch.utils.model_zoo as _model_zoo

        _model_zoo.load_url = _torch_hub.load_state_dict_from_url
    except Exception:
        pass

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from .utils import rgb_to_sh, knn, set_random_seed
from .colmap_loader import COLMAPDataset

logger = logging.getLogger(__name__)


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
):
    """Kerbl-style exponential LR schedule with optional delayed warmup multiplier."""

    def helper(step: int) -> float:
        if step < 0:
            return 0.0
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(max(step / lr_delay_steps, 0.0), 1.0)
            )
        else:
            delay_rate = 1.0

        t = min(max(step / max_steps, 0.0), 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class GsplatTrainer:
    """
    Gaussian Splatting trainer based on upstream gsplat implementation.
    Supports baseline and modified modes for research comparison.
    """
    
    def __init__(
        self,
        image_dir: Path,
        colmap_dir: Path,
        output_dir: Path,
        # --- ORIGINAL KERBL PARAMETERS ---
        mode: str = "baseline",  # [custom]
        max_steps: int = 30_000,  # [original]
        eval_interval: Optional[int] = 1000,  # [original]
        splat_export_interval: Optional[int] = 150,  # [original]
        png_export_interval: Optional[int] = 50,  # [original]
        checkpoint_interval: Optional[int] = None,  # [original]
        densify_from_iter: int | None = None,  # [original]
        densify_until_iter: int | None = None,  # [original]
        densification_interval: int | None = None,  # [original]
        densify_grad_threshold: float | None = None,  # [original]
        opacity_threshold: float | None = None,  # [original]
        lambda_dssim: float | None = None,  # [original]
        position_lr_init: float = 1.6e-4,  # [original]
        position_lr_final: float = 1.6e-6,  # [original]
        position_lr_delay_mult: float = 0.01,  # [original]
        position_lr_max_steps: int = 30_000,  # [original]
        test_iterations: Optional[list[int]] = None,  # [original]
        save_iterations: Optional[list[int]] = None,  # [original]
        # --- CUSTOM PARAMETERS ---
        device: str = "cuda",  # [custom]
        progress_callback: Optional[Callable] = None,  # [custom]
        auto_early_stop: bool = False,  # [custom]
        stop_checker: Optional[Callable[[], bool]] = None,  # [custom]
        resume: bool = False,  # [custom]
        max_init_gaussians: int | None = None,  # [custom]
        amp_enabled: bool = False,  # [custom]
        pruning_enabled: bool = False,  # [custom]
        pruning_policy: str = "opacity",  # [custom]
        pruning_weights: dict | None = None,  # [custom]
        opacity_reset_interval: int | None = None,  # [custom]
    ):
        set_random_seed(42)
        
        self.image_dir = Path(image_dir)
        self.colmap_dir = Path(colmap_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode  # "baseline" or "modified"
        self.max_steps = max_steps
        self.eval_interval = int(eval_interval) if eval_interval is not None else None
        self.position_lr_init = float(position_lr_init)
        self.position_lr_final = float(position_lr_final)
        self.position_lr_delay_mult = float(position_lr_delay_mult)
        self.position_lr_max_steps = int(position_lr_max_steps)
        default_test_iterations = [7_000, 30_000]
        default_save_iterations = [7_000, 30_000, int(self.max_steps)]
        self.test_iterations = sorted(
            {
                int(v)
                for v in (test_iterations if test_iterations is not None else default_test_iterations)
                if int(v) > 0
            }
        )
        self.save_iterations = sorted(
            {
                int(v)
                for v in (save_iterations if save_iterations is not None else default_save_iterations)
                if int(v) > 0
            }
        )

        self.device = device
        self.progress_callback = progress_callback
        self.splat_export_interval = splat_export_interval
        self.png_export_interval = png_export_interval
        self.checkpoint_interval = int(checkpoint_interval) if checkpoint_interval is not None else None
        self.auto_early_stop = auto_early_stop
        self.stop_checker = stop_checker
        self.resume = resume
        # Maximum number of Gaussians to initialize from COLMAP points.
        # If None, use all points (may OOM for large scenes).
        self.max_init_gaussians = max_init_gaussians
        logger.info(f"Trainer configured max_init_gaussians={self.max_init_gaussians}")
        # AMP (automatic mixed precision) flag
        self.amp_enabled = bool(amp_enabled)
        if self.amp_enabled:
            logger.info("AMP (mixed precision) enabled for training")

        # Pruning policy configuration
        self.pruning_enabled = bool(pruning_enabled)
        self.pruning_policy = pruning_policy or "opacity"
        # Pruning weights: dict may contain keys 'opacity', 'scale', 'age'
        self.pruning_weights = pruning_weights if isinstance(pruning_weights, dict) else {}
        # Defaults
        self.pruning_weights.setdefault("opacity", 1.0)
        self.pruning_weights.setdefault("scale", 0.0)
        self.pruning_weights.setdefault("age", 0.0)
        logger.info(f"Pruning enabled={self.pruning_enabled}, policy={self.pruning_policy}, weights={self.pruning_weights}")
        self.stop_reason: Optional[str] = None
        self.last_tuning_info = None  # Track last tuning action for status updates
        # Match upstream behavior: progressively unlock SH bands up to degree 3.
        self.sh_degree = 3
        self.sh_degree_interval = 1000

        self.densify_from_iter = max(0, int(densify_from_iter)) if densify_from_iter is not None else 500
        default_stop = 15_000 if densify_until_iter is None else int(densify_until_iter)
        self.densify_until_iter = max(self.densify_from_iter + 1, default_stop)
        interval_value = densification_interval if densification_interval is not None else 100
        self.densification_interval = max(1, int(interval_value))
        reset_every_value = opacity_reset_interval if opacity_reset_interval is not None else 3000
        self.opacity_reset_interval = max(self.densification_interval, int(reset_every_value))

        default_opacity_threshold = 0.005
        try:
            self.opacity_threshold = float(opacity_threshold) if opacity_threshold is not None else default_opacity_threshold
        except Exception:
            self.opacity_threshold = default_opacity_threshold
        if self.opacity_threshold <= 0:
            self.opacity_threshold = default_opacity_threshold

        default_lambda = 0.2
        try:
            self.lambda_dssim = float(lambda_dssim) if lambda_dssim is not None else default_lambda
        except Exception:
            self.lambda_dssim = default_lambda
        self.lambda_dssim = min(1.0, max(0.0, self.lambda_dssim))

        default_densify_grad_threshold = 0.0002
        try:
            self.densify_grad_threshold = (
                float(densify_grad_threshold)
                if densify_grad_threshold is not None
                else default_densify_grad_threshold
            )
        except Exception:
            self.densify_grad_threshold = default_densify_grad_threshold
        if self.densify_grad_threshold <= 0:
            self.densify_grad_threshold = default_densify_grad_threshold
        logger.info(
            "Densify schedule configured: start=%d, stop=%d, interval=%d, reset_every=%d",
            self.densify_from_iter,
            self.densify_until_iter,
            self.densification_interval,
            self.opacity_reset_interval,
        )
        logger.info("Trainer configured opacity_threshold=%f, lambda_dssim=%f", self.opacity_threshold, self.lambda_dssim)
        logger.info("Trainer configured densify_grad_threshold=%f", self.densify_grad_threshold)
        logger.info(
            "Position LR schedule: init=%g final=%g delay_mult=%g max_steps=%d",
            self.position_lr_init,
            self.position_lr_final,
            self.position_lr_delay_mult,
            self.position_lr_max_steps,
        )
        logger.info(
            "Milestone defaults: test_iterations=%s save_iterations=%s",
            self.test_iterations,
            self.save_iterations,
        )
        
        # Setup directories
        self.ckpt_dir = output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir = output_dir / "previews"
        if self.png_export_interval:
            self.preview_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COLMAP data
        logger.info("Loading COLMAP data...")
        self.dataset = COLMAPDataset(colmap_dir, image_dir)
        self.scene_scale = self.dataset.scene_scale * 1.1
        logger.info(f"Loaded {len(self.dataset)} images, {len(self.dataset.points)} points")
        logger.info(f"Scene scale: {self.scene_scale:.3f}")
        
        # Initialize Gaussian parameters
        self._init_gaussians()

        # If AMP enabled and on CUDA, create GradScaler
        self.scaler = None
        if self.amp_enabled and torch.cuda.is_available():
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception:
                self.scaler = None
        # Debug: log AMP / GradScaler status for runtime verification
        try:
            logger.info(
                f"AMP flag: {self.amp_enabled}; CUDA available: {torch.cuda.is_available()}; GradScaler initialized: {self.scaler is not None}"
            )
        except Exception:
            # Best-effort logging; do not crash on logging errors
            pass
        
        # Adaptive tuning parameters (your thesis approach)
        self.tuning_params = {
            "lr_mult": 1.0,
            "opacity_lr_mult": 1.0,
            "sh_lr_mult": 1.0,
            "position_lr_mult": 1.0,
            "densify_threshold": self.densify_grad_threshold,
        }
        
        # Setup training strategy
        self.strategy_config = dict(
            verbose=True,
            prune_opa=self.opacity_threshold,
            grow_grad2d=self.tuning_params["densify_threshold"],
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            refine_scale2d_stop_iter=0,
            refine_start_iter=self.densify_from_iter,
            refine_stop_iter=self.densify_until_iter,
            reset_every=self.opacity_reset_interval,
            refine_every=self.densification_interval,
            pause_refine_after_reset=0,
            absgrad=False,
            revised_opacity=False,
            key_for_gradient="means2d",
        )
        self.strategy = DefaultStrategy(**self.strategy_config)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        
        # Store convergence data for analysis
        self.convergence_history = []
        self.tuning_history = []
        self.tuner_runs = 0
        self.eval_history: list[dict] = []

    def _init_gaussians(self):
        """Initialize Gaussian parameters from COLMAP points."""
        points = torch.from_numpy(self.dataset.points).float()
        rgbs = torch.from_numpy(self.dataset.points_rgb / 255.0).float()

        N = points.shape[0]
        # Optionally subsample to avoid excessive memory usage
        if self.max_init_gaussians is not None and N > self.max_init_gaussians:
            logger.info(f"COLMAP produced {N} points; subsampling to {self.max_init_gaussians} Gaussians to reduce memory")
            idx = torch.randperm(N)[: self.max_init_gaussians]
            points = points[idx].contiguous()
            rgbs = rgbs[idx].contiguous()
            N = points.shape[0]

        logger.info(f"Initializing {N} Gaussians from COLMAP points")
        
        # Initialize scales based on k-nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)
        
        # Initialize rotations as random quaternions (rasterizer normalizes internally).
        quats = torch.rand((N, 4))
        
        # Initialize opacities (logit space)
        opacities = torch.logit(torch.full((N,), 0.1))
        
        # Initialize full SH storage and enable bands progressively during training.
        sh0 = rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]
        shN = torch.zeros((N, (self.sh_degree + 1) ** 2 - 1, 3))
        
        # Create parameter dict
        self.splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }).to(self.device)
        
        # Setup optimizers (same as upstream)
        self._setup_optimizers()

    def _setup_optimizers(self):
        """(Re)initialize optimizers and scheduler for current splats."""
        lr_scale = math.sqrt(1)  # batch_size=1
        self.optimizers = {
            "means": torch.optim.Adam(
                [{"params": self.splats["means"], "lr": 1.6e-4 * self.scene_scale * lr_scale, "name": "means"}],
                eps=1e-15,
            ),
            "scales": torch.optim.Adam(
                [{"params": self.splats["scales"], "lr": 5e-3 * lr_scale, "name": "scales"}],
                eps=1e-15,
            ),
            "quats": torch.optim.Adam(
                [{"params": self.splats["quats"], "lr": 1e-3 * lr_scale, "name": "quats"}],
                eps=1e-15,
            ),
            "opacities": torch.optim.Adam(
                [{"params": self.splats["opacities"], "lr": 2.5e-2 * lr_scale, "name": "opacities"}],
                eps=1e-15,
            ),
            "sh0": torch.optim.Adam(
                [{"params": self.splats["sh0"], "lr": 2.5e-3 * lr_scale, "name": "sh0"}],
                eps=1e-15,
            ),
            "shN": torch.optim.Adam(
                [{"params": self.splats["shN"], "lr": 2.5e-3 / 20 * lr_scale, "name": "shN"}],
                eps=1e-15,
            ),
        }

        # Learning rate scheduler for means
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.max_steps)
        )

        self.position_lr_fn = get_expon_lr_func(
            lr_init=self.position_lr_init * self.scene_scale * lr_scale,
            lr_final=self.position_lr_final * self.scene_scale * lr_scale,
            lr_delay_steps=0,
            lr_delay_mult=self.position_lr_delay_mult,
            max_steps=self.position_lr_max_steps,
        )

    def _reset_strategy_state(self):
        """Rebuild gsplat strategy so its buffers stay in sync after pruning."""
        self.strategy = DefaultStrategy(**self.strategy_config)
        self.strategy.grow_grad2d = self.tuning_params["densify_threshold"]
        try:
            self.strategy.check_sanity(self.splats, self.optimizers)
        except Exception as exc:
            logger.warning("Strategy sanity check failed; continuing with fresh state: %s", exc)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

    def rasterize(
        self,
        camtoworld: Tensor,
        K: Tensor,
        width: int,
        height: int,
        sh_degree_to_use: Optional[int] = None,
    ) -> Tensor:
        """Rasterize Gaussians to image."""
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        
        if sh_degree_to_use is None:
            sh_degree_to_use = self.sh_degree
        sh_degree_to_use = max(0, min(int(sh_degree_to_use), self.sh_degree))
        active_bands = (sh_degree_to_use + 1) ** 2
        shN_active = self.splats["shN"][:, : max(0, active_bands - 1), :]

        # Keep SH unflattened: gsplat SH rasterization expects [N, bands, 3].
        colors = torch.cat([self.splats["sh0"], shN_active], 1)
        
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree_to_use,
            viewmats=torch.linalg.inv(camtoworld)[None],
            Ks=K[None],
            width=width,
            height=height,
            packed=False,
        )
        
        return render_colors, render_alphas, info
    
    def detect_convergence_issues(self, window_size: int = 50) -> Dict:
        """
        Detect convergence issues: slow convergence, loss plateau.
        Part of Rule-Based Adaptive Tuner.
        """
        if len(self.convergence_history) < window_size:
            return {"has_issues": False, "reason": "insufficient_data"}
        
        recent = self.convergence_history[-window_size:]
        losses = [h["loss"] for h in recent]
        
        # Check for loss plateau (variance too low)
        loss_variance = np.var(losses)
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]  # slope
        
        issues = {
            "has_issues": False,
            "loss_plateau": loss_variance < 1e-6,
            "slow_convergence": abs(loss_trend) < 1e-5,
            "loss_variance": loss_variance,
            "loss_trend": loss_trend,
        }
        
        issues["has_issues"] = issues["loss_plateau"] or issues["slow_convergence"]
        return issues
    
    def detect_instability(self, window_size: int = 50) -> Dict:
        """
        Detect training instability: gradient explosion, loss spikes, NaN.
        Part of Rule-Based Adaptive Tuner.
        """
        if len(self.convergence_history) < window_size:
            return {"has_issues": False, "reason": "insufficient_data"}
        
        recent = self.convergence_history[-window_size:]
        losses = [h["loss"] for h in recent]
        grad_means = [h["grad_means"] for h in recent]
        grad_opacities = [h["grad_opacities"] for h in recent]
        
        # Check for instability indicators
        loss_mean = np.mean(losses)
        loss_spikes = sum(1 for l in losses if l > loss_mean * 2.0)
        
        grad_explosion = any(g > 100.0 for g in grad_means)
        has_nan = any(np.isnan(l) or np.isinf(l) for l in losses)
        
        instability = {
            "has_issues": False,
            "loss_spikes": loss_spikes > window_size * 0.1,  # >10% spikes
            "gradient_explosion": grad_explosion,
            "has_nan": has_nan,
            "max_grad": max(grad_means) if grad_means else 0,
        }
        
        instability["has_issues"] = any([
            instability["loss_spikes"],
            instability["gradient_explosion"],
            instability["has_nan"]
        ])
        return instability
    
    def adaptive_tune_parameters(self, step: int) -> bool:
        """
        Rule-Based Adaptive Tuner: Adjust hyperparameters based on convergence/instability.
        Returns True if parameters were updated.
        """
        logger.info(f"[TUNER] Run #{self.tuner_runs + 1} at step {step}")
        
        convergence = self.detect_convergence_issues()
        instability = self.detect_instability()
        
        updated = False
        adjustments = []
        
        # Rule 1: If instability detected, reduce learning rates
        if instability["has_issues"]:
            logger.warning(f"[TUNER] Instability detected: {instability}")
            if instability["gradient_explosion"]:
                self.tuning_params["lr_mult"] *= 0.7
                self.tuning_params["position_lr_mult"] *= 0.7
                adjustments.append("reduced_lr_explosion")
                updated = True
            if instability["loss_spikes"]:
                self.tuning_params["opacity_lr_mult"] *= 0.8
                adjustments.append("reduced_opacity_lr_spikes")
                updated = True
        
        # Rule 2: If slow convergence, increase learning rates cautiously
        elif convergence["has_issues"]:
            logger.warning(f"[TUNER] Convergence issues detected: {convergence}")
            if convergence["loss_plateau"]:
                self.tuning_params["lr_mult"] *= 1.2
                self.tuning_params["sh_lr_mult"] *= 1.15
                adjustments.append("increased_lr_plateau")
                updated = True
            if convergence["slow_convergence"]:
                self.tuning_params["densify_threshold"] *= 0.8  # more aggressive
                adjustments.append("reduced_densify_threshold")
                updated = True
        
        # Rule 3: Ensure parameters stay in valid ranges
        self.tuning_params["lr_mult"] = np.clip(self.tuning_params["lr_mult"], 0.1, 3.0)
        self.tuning_params["opacity_lr_mult"] = np.clip(self.tuning_params["opacity_lr_mult"], 0.1, 2.0)
        self.tuning_params["sh_lr_mult"] = np.clip(self.tuning_params["sh_lr_mult"], 0.1, 2.0)
        self.tuning_params["position_lr_mult"] = np.clip(self.tuning_params["position_lr_mult"], 0.1, 2.0)
        self.tuning_params["densify_threshold"] = np.clip(self.tuning_params["densify_threshold"], 0.0001, 0.001)
        
        if updated:
            # Apply updated parameters to optimizers
            self._apply_tuning_params()
            
            tuning_record = {
                "step": step,
                "run": self.tuner_runs + 1,
                "convergence": convergence,
                "instability": instability,
                "adjustments": adjustments,
                "params": self.tuning_params.copy(),
            }
            self.tuning_history.append(tuning_record)
            
            # Save last tuning info for status updates
            action_text = ", ".join(adjustments).replace("_", " ").title()
            reason_parts = []
            if instability["has_issues"]:
                if instability["gradient_explosion"]:
                    reason_parts.append("gradient explosion")
                if instability["loss_spikes"]:
                    reason_parts.append("loss spikes")
            if convergence["has_issues"]:
                if convergence["loss_plateau"]:
                    reason_parts.append("loss plateau")
                if convergence["slow_convergence"]:
                    reason_parts.append("slow convergence")
            
            self.last_tuning_info = {
                "step": step,
                "action": action_text if action_text else "Parameter adjustment",
                "reason": ", ".join(reason_parts).title() if reason_parts else "Optimization"
            }
            
            logger.info(f"[TUNER] Updated parameters: {self.tuning_params}")
            logger.info(f"[TUNER] Adjustments: {adjustments}")
        else:
            logger.info(f"[TUNER] No parameter updates needed")
        
        self.tuner_runs += 1
        return updated
    
    def _apply_tuning_params(self):
        """Apply tuned parameters to optimizers and strategy."""
        # Update optimizer learning rates
        for name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                base_lr = param_group['lr']
                if name == "means":
                    param_group['lr'] = base_lr * self.tuning_params["position_lr_mult"]
                elif name == "opacities":
                    param_group['lr'] = base_lr * self.tuning_params["opacity_lr_mult"]
                elif name in ["sh0", "shN"]:
                    param_group['lr'] = base_lr * self.tuning_params["sh_lr_mult"]
                else:
                    param_group['lr'] = base_lr * self.tuning_params["lr_mult"]
        
        # Update densification threshold
        self.strategy.grow_grad2d = self.tuning_params["densify_threshold"]

    def _export_current_splats(self, step: int):
        """Export current in-memory splats to .splat and .ply snapshots for live viewing."""
        try:
            from gsplat.exporter import export_splats

            means = self.splats["means"].detach().cpu().float()
            scales = torch.exp(self.splats["scales"].detach().cpu().float())
            quats = self.splats["quats"].detach().cpu().float()
            opacities = torch.sigmoid(self.splats["opacities"].detach().cpu().float())

            sh0 = self.splats["sh0"].detach().cpu().float()
            shN = self.splats["shN"].detach().cpu().float()

            snapshot_dir = self.output_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            step_tag = f"{step:06d}"

            splat_snapshot = snapshot_dir / f"3dmodel_{step_tag}.splat"
            splat_tmp = splat_snapshot.with_name(splat_snapshot.name + ".tmp")
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format="splat",
                save_to=str(splat_tmp),
            )
            splat_tmp.replace(splat_snapshot)
            self._update_latest_link(splat_snapshot, self.output_dir / "splats.splat")

            # Export PLY alongside SPLAT so viewers can reliably fall back.
            ply_snapshot = snapshot_dir / f"3dmodel_{step_tag}.ply"
            ply_tmp = ply_snapshot.with_name(ply_snapshot.name + ".tmp")
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format="ply",
                save_to=str(ply_tmp),
            )
            ply_tmp.replace(ply_snapshot)
            self._update_latest_link(ply_snapshot, self.output_dir / "splats.ply")

            logger.info(
                "Live export complete at step %d (snapshots %s, %s)",
                step,
                splat_snapshot.name,
                ply_snapshot.name,
            )
        except Exception as e:
            logger.warning("Live export failed at step %d: %s", step, e)

    def _update_latest_link(self, snapshot_path: Path, latest_path: Path):
        """Update canonical splats.* pointer without duplicating large files."""
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            os.link(snapshot_path, latest_path)
        except OSError:
            try:
                shutil.copy2(snapshot_path, latest_path)
            except Exception as copy_err:
                logger.warning(
                    "Failed to refresh %s from %s: %s",
                    latest_path.name,
                    snapshot_path.name,
                    copy_err,
                )

    def _save_preview_image(self, step: int, colors: Tensor):
        """Persist a PNG preview from the current render."""
        try:
            if not self.preview_dir.exists():
                self.preview_dir.mkdir(parents=True, exist_ok=True)

            colors_np = colors.detach().clamp(0, 1).cpu().numpy()
            img = Image.fromarray((colors_np * 255).astype(np.uint8))
            preview_path = self.preview_dir / f"preview_{step:06d}.png"
            img.save(preview_path)

            latest_path = self.preview_dir / "preview_latest.png"
            img.save(latest_path)
            logger.info("Saved preview PNG at step %d", step)
        except Exception as e:
            logger.warning("Could not save preview at step %d: %s", step, e)

    def _record_periodic_eval(self, step: int):
        """Run lightweight periodic evaluation and persist it to disk."""
        try:
            metrics = self.compute_evaluation_metrics(step)
            self.eval_history.append(metrics)
            out_path = self.output_dir / "eval_history.json"
            with open(out_path, "w") as f:
                json.dump(self.eval_history, f, indent=2)
            logger.info(
                "Eval %d | Loss: %.4f | GS: %d",
                step,
                float(metrics.get("final_loss", 0.0)),
                int(metrics.get("num_gaussians", len(self.splats["means"]))),
            )
        except Exception as exc:
            logger.warning("Periodic eval failed at step %d: %s", step, exc)
    
    def compute_evaluation_metrics(self, step: int, validation_images: list = None) -> Dict:
        """
        Compute comprehensive evaluation metrics for your thesis:
        - Convergence Speed
        - Final Loss at fixed iterations
        - Reprojection Error (LPIPS)
        - Image Sharpness (Laplacian Variance)
        - Gaussian Count
        """
        metrics = {
            "step": step,
            "num_gaussians": len(self.splats["means"]),
            "tuning_params": self.tuning_params.copy(),
        }
        
        # 1. Loss and Convergence Speed
        if len(self.convergence_history) > 0:
            all_losses = [h["loss"] for h in self.convergence_history]
            metrics["final_loss"] = all_losses[-1] if all_losses else 0
            metrics["loss_std"] = np.std(all_losses[-100:]) if len(all_losses) >= 100 else 0
            
            # Convergence speed: loss reduction rate
            if len(all_losses) > 100:
                early_loss = np.mean(all_losses[50:100])
                late_loss = np.mean(all_losses[-100:])
                metrics["convergence_speed"] = (early_loss - late_loss) / step
                metrics["loss_reduction_percent"] = ((early_loss - late_loss) / early_loss) * 100
            
            # Loss at fixed iterations (e.g., 1k, 5k, 10k)
            for milestone in [1000, 5000, 10000, 20000]:
                if step >= milestone and milestone < len(all_losses):
                    metrics[f"loss_at_{milestone}"] = all_losses[milestone]
        
        # 2. LPIPS and Image Sharpness (if validation images provided)
        if validation_images and len(validation_images) > 0:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
                
                lpips_scores = []
                sharpness_scores = []
                
                for img_idx in validation_images[:5]:  # Sample 5 images for speed
                    if img_idx >= len(self.dataset):
                        continue
                    
                    img_path = self.image_dir / self.dataset.image_names[img_idx]
                    try:
                        image = Image.open(img_path).convert("RGB")
                        pixels = torch.from_numpy(np.array(image)).float() / 255.0
                        pixels = pixels.to(self.device)
                        height, width = pixels.shape[0], pixels.shape[1]
                        
                        camtoworld = torch.from_numpy(self.dataset.camtoworlds[img_idx]).float().to(self.device)
                        K = torch.from_numpy(self.dataset.Ks[img_idx]).float().to(self.device)
                        
                        # Render
                        with torch.no_grad():
                            colors, _, _ = self.rasterize(camtoworld, K, width, height)
                            colors = colors[0].clamp(0, 1)
                        
                        # LPIPS (Reprojection Error)
                        pixels_p = pixels.permute(2, 0, 1).unsqueeze(0)
                        colors_p = colors.permute(2, 0, 1).unsqueeze(0)
                        lpips_score = lpips(colors_p, pixels_p).item()
                        lpips_scores.append(lpips_score)
                        
                        # Image Sharpness (Laplacian Variance)
                        sharpness = self._compute_sharpness(colors)
                        sharpness_scores.append(sharpness)
                        
                    except Exception as e:
                        logger.warning(f"Failed to evaluate image {img_idx}: {e}")
                        continue
                
                if lpips_scores:
                    metrics["lpips_mean"] = np.mean(lpips_scores)
                    metrics["lpips_std"] = np.std(lpips_scores)
                if sharpness_scores:
                    metrics["sharpness_mean"] = np.mean(sharpness_scores)
                    metrics["sharpness_std"] = np.std(sharpness_scores)
                    
            except ImportError:
                logger.warning("LPIPS not available, skipping perceptual metrics")
        
        return metrics
    
    def _compute_sharpness(self, image: Tensor) -> float:
        """
        Compute image sharpness using Laplacian variance.
        Higher values indicate sharper images.
        """
        # Convert to grayscale
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        # Apply Laplacian
        gray_padded = gray.unsqueeze(0).unsqueeze(0)
        laplacian = F.conv2d(gray_padded, laplacian_kernel, padding=1)
        
        # Variance of Laplacian
        sharpness = laplacian.var().item()
        return sharpness
    
    def save_tuning_results(self, step: int):
        """
        Save tuning results for ML/RL model training (Phase 2 of your thesis).
        This creates the dataset for your small ML model.
        """
        # Get validation image indices (use test split)
        num_images = len(self.dataset)
        val_indices = list(range(0, num_images, 8))  # Every 8th image
        
        results = {
            "final_params": self.tuning_params,
            "tuning_history": self.tuning_history,
            "scene_metadata": {
                "scene_scale": self.scene_scale,
                "num_images": num_images,
                "initial_points": len(self.dataset.points),
            },
            "intermediate_metrics": self.compute_evaluation_metrics(step, val_indices),
        }
        
        results_path = self.output_dir / "adaptive_tuning_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[TUNER] Saved tuning results to {results_path}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training in {self.mode} mode for {self.max_steps} steps")
        
        # Check for resume
        start_step = 0
        if self.resume:
            loaded_step = self._load_checkpoint()
            if loaded_step is not None:
                start_step = loaded_step + 1
                logger.info(f"Resuming training from step {start_step}")
        
        # Prepare image indices for training
        num_images = len(self.dataset)
        train_indices = list(range(num_images))
        
        start_time = time.time()
        last_step = start_step
        
        for step in range(start_step, self.max_steps):
            step_start = time.time()
            # Early exit if a stop signal was requested
            if self.stop_checker and self.stop_checker():
                self.stop_reason = "manual_stop"
                logger.info("Stop requested before step %d, exiting immediately without export or completion.", step)
                last_step = step
                return  # Exit immediately, do not continue or save/export anything

            # Random image selection
            idx = np.random.choice(train_indices)
            
            # Load image
            img_path = self.image_dir / self.dataset.image_names[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                pixels = torch.from_numpy(np.array(image)).float() / 255.0
                pixels = pixels.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
            
            height, width = pixels.shape[0], pixels.shape[1]
            camtoworld = torch.from_numpy(self.dataset.camtoworlds[idx]).float().to(self.device)
            K = torch.from_numpy(self.dataset.Ks[idx]).float().to(self.device)
            
            # Match upstream behavior: progressively unlock SH degree every 1000 steps.
            sh_degree_to_use = min(step // self.sh_degree_interval, self.sh_degree)
            t_before_raster = time.time()

            # Forward pass (support AMP if enabled)
            if self.amp_enabled and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    colors, alphas, info = self.rasterize(
                        camtoworld,
                        K,
                        width,
                        height,
                        sh_degree_to_use=sh_degree_to_use,
                    )
            else:
                colors, alphas, info = self.rasterize(
                    camtoworld,
                    K,
                    width,
                    height,
                    sh_degree_to_use=sh_degree_to_use,
                )
            colors = colors[0]  # Remove batch dim
            t_after_raster = time.time()
            
            # Compute loss (L1 + SSIM, same as upstream)
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            t_after_pre_backward = time.time()
            
            l1_loss = F.l1_loss(colors, pixels)
            ssim_loss = 1.0 - self._ssim(colors, pixels)
            l1_weight = 1.0 - self.lambda_dssim
            loss = (l1_weight * l1_loss) + (self.lambda_dssim * ssim_loss)
            
            # Backward (use GradScaler when AMP enabled)
            if self.amp_enabled and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            t_after_backward = time.time()
            
            # Store convergence data
            with torch.no_grad():
                grad_means = self.splats["means"].grad.norm().item() if self.splats["means"].grad is not None else 0
                grad_scales = self.splats["scales"].grad.norm().item() if self.splats["scales"].grad is not None else 0
                grad_opacities = self.splats["opacities"].grad.norm().item() if self.splats["opacities"].grad is not None else 0
                self.convergence_history.append({
                    "step": step,
                    "loss": loss.item(),
                    "l1_loss": l1_loss.item(),
                    "ssim_loss": ssim_loss.item(),
                    "grad_means": grad_means,
                    "grad_scales": grad_scales,
                    "grad_opacities": grad_opacities,
                })
            
            # Research intervention: Rule-Based Adaptive Tuner
            # Check every step in tuning window (first 200-300 iterations)
            if self.mode == "modified" and step >= 50 and step <= 300:
                # Only tune every 10 steps to avoid overhead
                if step % 10 == 0:
                    updated = self.adaptive_tune_parameters(step)
                
                # Save tuning results at end of tuning phase
                if step == 300:
                    self.save_tuning_results(step)
            
            # Optimizer step (handle AMP stepping when enabled)
            if self.amp_enabled and self.scaler is not None:
                for optimizer in self.optimizers.values():
                    try:
                        self.scaler.step(optimizer)
                    except Exception:
                        # Fallback to regular step if scaler.step fails
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                try:
                    self.scaler.update()
                except Exception:
                    pass
            else:
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            t_after_optim = time.time()

            # Match Kerbl schedule: update means LR with delayed exponential profile.
            means_lr = self.position_lr_fn(step + 1)
            for param_group in self.optimizers["means"].param_groups:
                param_group["lr"] = means_lr
            
            # Densification
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=False,
            )
            t_after_post_backward = time.time()

            # Live exports for viewer refresh
            if self.splat_export_interval and step % self.splat_export_interval == 0 and step > 0:
                self._export_current_splats(step)

            t_after_exports = time.time()

            if self.png_export_interval and step % self.png_export_interval == 0 and step > 0:
                self._save_preview_image(step, colors)
            
            # Logging and checkpoints
            if step % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step}/{self.max_steps} | Loss: {loss.item():.4f} | "
                    f"L1: {l1_loss.item():.4f} | SSIM: {ssim_loss.item():.4f} | "
                    f"GS: {len(self.splats['means'])} | Time: {elapsed:.1f}s"
                )

                step_total = t_after_exports - step_start
                logger.info(
                    "Step timing %d | total=%.3fs raster=%.3fs pre=%.3fs back=%.3fs opt=%.3fs post=%.3fs export=%.3fs",
                    step,
                    step_total,
                    t_after_raster - t_before_raster,
                    t_after_pre_backward - t_after_raster,
                    t_after_backward - t_after_pre_backward,
                    t_after_optim - t_after_backward,
                    t_after_post_backward - t_after_optim,
                    t_after_exports - t_after_post_backward,
                )
                
                if self.progress_callback:
                    progress = int((step / self.max_steps) * 100)
                    self.progress_callback(step, progress, loss.item(), self.last_tuning_info)

            # Periodic evaluation (native-style cadence control)
            iter_num = step + 1
            should_eval = (
                (self.eval_interval and step > 0 and step % self.eval_interval == 0)
                or (not self.eval_interval and iter_num in self.test_iterations)
            )
            if should_eval:
                self._record_periodic_eval(step)

            # Auto early-stop (plateau detection) if enabled
            if self.auto_early_stop and step > 1000:
                convergence = self.detect_convergence_issues(window_size=120)
                if convergence.get("has_issues") and abs(convergence.get("loss_trend", 0)) < 1e-6:
                    self.stop_reason = "auto_early_stop"
                    logger.info("Auto early-stop triggered at step %d due to plateau", step)
                    self._export_current_splats(step)
                    self._save_preview_image(step, colors)
                    last_step = step
                    break
            
            # Save checkpoints on configured cadence and always at final step.
            should_checkpoint = (
                (self.checkpoint_interval and step > 0 and step % self.checkpoint_interval == 0)
                or (not self.checkpoint_interval and iter_num in self.save_iterations)
                or step == self.max_steps - 1
            )
            if (
                should_checkpoint
            ):
                self._save_checkpoint(step)

            last_step = step
        
        logger.info(f"Training complete in {time.time() - start_time:.1f}s")
        
        # Compute final comprehensive evaluation metrics
        logger.info("Computing final evaluation metrics...")
        num_images = len(self.dataset)
        val_indices = list(range(0, num_images, 8))  # Every 8th image for validation
        final_metrics = self.compute_evaluation_metrics(last_step, val_indices)
        
        # Save final metadata with comprehensive metrics
        metadata = {
            "mode": self.mode,
            "num_gaussians": len(self.splats["means"]),
            "training_steps": last_step + 1,
            "scene_scale": self.scene_scale,
            "training_time_seconds": time.time() - start_time,
            "final_metrics": final_metrics,
            "training_engine": "gsplat",
        }
        
        if self.mode == "modified":
            metadata["tuning_runs"] = self.tuner_runs
            metadata["final_tuning_params"] = self.tuning_params

        if self.stop_reason:
            metadata["stop_reason"] = self.stop_reason
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Final evaluation metrics: {final_metrics}")
    
    def _ssim(self, img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
        """Simple SSIM implementation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1.permute(2, 0, 1)[None], window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2.permute(2, 0, 1)[None], window_size, 1, window_size // 2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1.permute(2, 0, 1)[None].pow(2), window_size, 1, window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2.permute(2, 0, 1)[None].pow(2), window_size, 1, window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1.permute(2, 0, 1)[None] * img2.permute(2, 0, 1)[None], window_size, 1, window_size // 2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def _save_checkpoint(self, step: int):
        """Save checkpoint."""
        ckpt_path = self.ckpt_dir / f"ckpt_{step:06d}.pt"
        checkpoint = {
            "step": step,
            "means": self.splats["means"].data,
            "scales": self.splats["scales"].data,
            "quats": self.splats["quats"].data,
            "opacities": self.splats["opacities"].data,
            "sh0": self.splats["sh0"].data,
            "shN": self.splats["shN"].data,
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
    
    def _load_checkpoint(self) -> Optional[int]:
        """Load latest checkpoint if available. Returns starting step or None."""
        checkpoints = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if not checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return None
        
        latest_ckpt = checkpoints[-1]
        logger.info(f"Resuming from checkpoint: {latest_ckpt}")
        
        try:
            checkpoint = torch.load(latest_ckpt, map_location=self.device)
            step = checkpoint["step"]
            
            # Load Gaussian parameters
            self.splats["means"].data = checkpoint["means"].to(self.device)
            self.splats["scales"].data = checkpoint["scales"].to(self.device)
            self.splats["quats"].data = checkpoint["quats"].to(self.device)
            self.splats["opacities"].data = checkpoint["opacities"].to(self.device)
            self.splats["sh0"].data = checkpoint["sh0"].to(self.device)
            self.splats["shN"].data = checkpoint["shN"].to(self.device)
            
            logger.info(f"Loaded checkpoint from step {step}")
            return step
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

