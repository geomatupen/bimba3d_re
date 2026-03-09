import subprocess
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def run_gsplat(image_dir: Path, colmap_sparse: Path, output_dir: Path, params: dict | None = None):
    """
    Run gsplat training to create Gaussian Splat model.
    
    Args:
        image_dir: Directory with input images
        colmap_sparse: Path to COLMAP sparse reconstruction (0 folder)
        output_dir: Directory for training outputs
        
    Returns:
        Path to output directory containing splats.bin and metadata.json
        
    Raises:
        subprocess.CalledProcessError: If training fails
        FileNotFoundError: If gsplat is not installed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if COLMAP output exists
    colmap_model_dir = colmap_sparse / "0"
    if not colmap_model_dir.exists():
        raise FileNotFoundError(f"COLMAP model not found at {colmap_model_dir}")
    
    try:
        logger.info("Running Gaussian Splatting training...")
        
        # Train with gsplat
        # This uses gsplat's standard training command
        # Resolve parameters with defaults
        p = params or {}
        cmd = [
            "python", "-m", "gsplat.train",
            "--image_path", str(image_dir),
            "--colmap_path", str(colmap_model_dir),
            "--output_dir", str(output_dir),
            "--max_steps", str(p.get("max_steps", 30000)),
            "--batch_size", str(p.get("batch_size", 1)),
            "--eval_interval", str(p.get("eval_interval", 1000)),
            "--save_interval", str(p.get("save_interval", 150)),
            "--densify_from_iter", str(p.get("densify_from_iter", 500)),
            "--densify_until_iter", str(p.get("densify_until_iter", 15000)),
            "--densification_interval", str(p.get("densification_interval", 100)),
            "--densify_grad_threshold", str(p.get("densify_grad_threshold", 0.0002)),
            "--opacity_threshold", str(p.get("opacity_threshold", 0.005)),
            "--lambda_dssim", str(p.get("lambda_dssim", 0.2)),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("✓ Gaussian Splatting training completed")
        
        # The trained model is typically saved as a checkpoint
        # We need to export it to splats.bin and metadata.json
        create_output_artifacts(output_dir, colmap_model_dir, params or {})
        
        logger.info(f"Outputs saved to: {output_dir}")
        return output_dir
        
    except FileNotFoundError as e:
        logger.error(f"gsplat not found. Install with: pip install gsplat")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Gaussian Splatting training failed: {e.stderr}")
        raise


def create_output_artifacts(output_dir: Path, colmap_model_dir: Path, params: dict):
    """
    Create splats.splat (.splat format) and metadata.json from trained model.
    """
    # Check for trained model checkpoint
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        # Find latest checkpoint
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            logger.info(f"Found checkpoint: {latest_ckpt}")
            
            # Export to .splat format (optimized for web)
            export_checkpoint_to_splats(latest_ckpt, output_dir)
    
    # Create metadata.json with camera and model info
    metadata = create_metadata(colmap_model_dir)
    # Attach training config used (merge defaults with overrides)
    p = params or {}
    # --- ORIGINAL KERBL PARAMETERS ---
    metadata["training_config"] = {
        "max_steps": int(p.get("max_steps", 30000)),  # [original]
        "batch_size": int(p.get("batch_size", 1)),  # [original]
        "eval_interval": int(p.get("eval_interval", 1000)),  # [original]
        "save_interval": int(p.get("save_interval", 150)),  # [original]
        "densify_from_iter": int(p.get("densify_from_iter", 500)),  # [original]
        "densify_until_iter": int(p.get("densify_until_iter", 15000)),  # [original]
        "densification_interval": int(p.get("densification_interval", 100)),  # [original]
        "densify_grad_threshold": float(p.get("densify_grad_threshold", 0.0002)),  # [original]
        "opacity_threshold": float(p.get("opacity_threshold", 0.005)),  # [original]
        "lambda_dssim": float(p.get("lambda_dssim", 0.2)),  # [original]
        # --- CUSTOM PARAMETERS ---
        "gsplat_max_gaussians": p.get("gsplat_max_gaussians"),  # [custom]
        "amp": p.get("amp"),  # [custom]
        "auto_early_stop": p.get("auto_early_stop"),  # [custom]
        "pruning_enabled": p.get("pruning_enabled"),  # [custom]
        "pruning_policy": p.get("pruning_policy"),  # [custom]
        "pruning_weights": p.get("pruning_weights"),  # [custom]
        "litegs_target_primitives": p.get("litegs_target_primitives"),  # [custom]
        "litegs_alpha_shrink": p.get("litegs_alpha_shrink"),  # [custom]
        "sparse_preference": p.get("sparse_preference"),  # [custom]
        "images_max_size": p.get("images_max_size"),  # [custom]
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Created metadata.json")


def export_checkpoint_to_splats(ckpt_path: Path, output_dir: Path):
    """
    Export trained model checkpoint using gsplat's built-in exporter.
    This handles all the proper quantization, sorting, and format conversion.
    """
    try:
        import torch
        from gsplat.exporter import export_splats
        
        logger.info(f"Exporting checkpoint using gsplat exporter: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint structures
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            else:
                state = checkpoint
        else:
            state = checkpoint
        
        # Extract parameters - try multiple possible key names
        means = state.get('means', state.get('xyz', state.get('_xyz', None)))
        scales = state.get('scales', state.get('scaling', state.get('_scaling', None)))
        quats = state.get('quats', state.get('rotations', state.get('_rotation', None)))
        opacities = state.get('opacities', state.get('_opacity', None))
        sh0 = state.get('sh0', state.get('sh_features', state.get('_sh0', None)))
        shN = state.get('shN', state.get('_shN', None))
        
        if means is None:
            logger.error(f"Could not find means in checkpoint. Available keys: {list(state.keys())}")
            raise ValueError("Missing Gaussian means in checkpoint")
        
        logger.info(f"Extracted {means.shape[0]} Gaussians from checkpoint")
        
        # Ensure tensors are on CPU and correct dtype
        means = means.cpu().float()
        scales = scales.cpu().float() if scales is not None else torch.zeros_like(means)
        quats = quats.cpu().float() if quats is not None else torch.zeros((means.shape[0], 4))
        opacities = opacities.cpu().float() if opacities is not None else torch.zeros(means.shape[0])
        
        # Handle SH features
        if sh0 is None or (isinstance(sh0, torch.Tensor) and sh0.numel() == 0):
            sh0 = torch.ones((means.shape[0], 1, 3), dtype=torch.float32) * 0.5
        else:
            sh0 = sh0.cpu().float()
            if sh0.ndim == 2:
                sh0 = sh0.unsqueeze(1)
        
        if shN is None or (isinstance(shN, torch.Tensor) and shN.numel() == 0):
            shN = torch.zeros((means.shape[0], 0, 3), dtype=torch.float32)
        else:
            shN = shN.cpu().float()
        
        # Ensure opacities is 1D
        if opacities.ndim > 1:
            opacities = opacities.squeeze()
        
        logger.info(f"Shapes - means: {means.shape}, scales: {scales.shape}, quats: {quats.shape}, opacities: {opacities.shape}, sh0: {sh0.shape}")
        
        # Export to both formats using gsplat's exporter
        try:
            splat_path = output_dir / "splats.splat"
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format="splat",
                save_to=str(splat_path)
            )
            logger.info(f"✓ Exported to .splat: {splat_path} ({splat_path.stat().st_size} bytes)")
        except Exception as e:
            logger.warning(f"Could not export to .splat format: {e}")
        
        # Always export PLY for compatibility
        try:
            ply_path = output_dir / "splats.ply"
            export_splats(
                means=means,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format="ply",
                save_to=str(ply_path)
            )
            logger.info(f"✓ Exported to PLY: {ply_path} ({ply_path.stat().st_size} bytes)")
        except Exception as e:
            logger.error(f"PLY export failed: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise


def export_to_ply_format(ckpt_path: Path, output_dir: Path):
    """
    Fallback: Export checkpoint to PLY format using gsplat's exporter.
    """
    try:
        import torch
        from gsplat.exporter import export_splats
        
        logger.info("Exporting checkpoint to PLY format using gsplat exporter (fallback)...")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        
        means = state.get('means', state.get('xyz', None))
        scales = state.get('scales', state.get('scaling', None))
        quats = state.get('quats', state.get('rot', None))
        opacities = state.get('opacities', state.get('opacity', None))
        sh0 = state.get('sh0', state.get('sh_features', None))
        shN = state.get('shN', None)
        
        if means is None:
            raise ValueError("Cannot extract means from checkpoint")
        
        # Ensure correct shapes
        if sh0 is not None and sh0.ndim == 2:
            sh0 = sh0.unsqueeze(1)
        elif sh0 is None:
            sh0 = torch.ones((means.shape[0], 1, 3)) * 0.5
        
        if shN is None:
            shN = torch.zeros((means.shape[0], 0, 3))
        
        if opacities is not None and opacities.ndim > 1:
            opacities = opacities.squeeze()
        
        ply_path = output_dir / "splats.ply"
        export_splats(
            means=means.cpu().float(),
            scales=scales.cpu().float() if scales is not None else torch.zeros_like(means),
            quats=quats.cpu().float() if quats is not None else torch.zeros((means.shape[0], 4)),
            opacities=opacities.cpu().float() if opacities is not None else torch.zeros(means.shape[0]),
            sh0=sh0.cpu().float(),
            shN=shN.cpu().float(),
            format="ply",
            save_to=str(ply_path)
        )
        logger.info(f"✓ Exported to {ply_path}")
        
    except Exception as e:
        logger.error(f"PLY fallback export failed: {e}", exc_info=True)
        raise


def create_metadata(colmap_model_dir: Path) -> dict:
    """
    Create metadata.json with model and camera information.
    """
    try:
        import struct
        
        cameras_path = colmap_model_dir / "cameras.bin"
        images_path = colmap_model_dir / "images.bin"
        points_path = colmap_model_dir / "points3D.bin"
        
        # Count points from binary file
        num_points = 0
        if points_path.exists():
            file_size = points_path.stat().st_size
            # XYZ (24 bytes) + RGB (3 bytes) + error (8 bytes) + num_track_elements (4 bytes)
            point_size = 39  # Base size, then variable track elements
            # This is approximate - actual size depends on tracks
            num_points = file_size // 50  # Conservative estimate
        
        metadata = {
            "version": "1.0",
            "type": "gaussian_splats",
            "num_points": max(num_points, 0),
            "colmap_model": str(colmap_model_dir),
            "training_config": {
                "max_steps": 30000,
                "batch_size": 1,
                "densification_interval": 100,
                "splat_export_interval": 150,
                "png_export_interval": 50,
            }
        }
        return metadata
        
    except Exception as e:
        logger.warning(f"Could not read COLMAP metadata: {e}")
        return {
            "version": "1.0",
            "type": "gaussian_splats",
            "error": str(e)
        }
