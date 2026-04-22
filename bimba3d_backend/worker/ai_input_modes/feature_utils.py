"""
Utility functions for feature calculation in reduced AI input modes.

This module provides concrete, verified calculations for:
- Ground Sampling Distance (GSD): primary metric for flight altitude impact
- Terrain roughness from COLMAP sparse points: geometric variance measure
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import struct


def calculate_gsd(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """
    Calculate Ground Sampling Distance (GSD) in meters/pixel.
    
    Formula: GSD = (H * sensor_width) / (f * image_width)
    
    Where:
    - H: flight altitude (meters above ground)
    - sensor_width: physical sensor width (mm)
    - f: focal length (mm)
    - image_width: image width in pixels
    
    GSD is inverse to resolution quality: lower GSD = higher detail.
    Typical ranges: 1-10 cm/px for UAV reconstruction.
    
    Args:
        altitude_m: Flight altitude in meters
        focal_length_mm: Lens focal length in mm
        sensor_width_mm: Camera sensor physical width in mm
        image_width_px: Image width in pixels
        
    Returns:
        GSD value in meters/pixel, clamped to [0.001, 0.5]
        (0.001 = 1mm/px for high-altitude detailed imaging)
        (0.5 = 50cm/px for low-resolution or high-altitude overview)
    """
    # Guard against division by zero and invalid inputs
    if focal_length_mm <= 0.0 or image_width_px <= 0:
        return 0.0  # Missing data default
    
    if altitude_m <= 0.0:
        return 0.0  # No altitude data
    
    if sensor_width_mm <= 0.0:
        return 0.0  # Invalid sensor data
    
    # GSD calculation (safe division)
    gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * float(image_width_px))
    
    # Clamp to realistic range for UAV/drone imaging
    # Min: 1mm/pixel (very low altitude, high zoom)
    # Max: 50cm/pixel (very high altitude or poor conditions)
    gsd = max(0.001, min(0.5, gsd))
    
    return float(gsd)


def read_colmap_points3d(colmap_dir: Path) -> Optional[np.ndarray]:
    """
    Read COLMAP points3D.bin sparse reconstruction data.
    
    Returns:
        numpy array of shape (N, 3) with XYZ coordinates, or None if unavailable.
    """
    points_file = colmap_dir / "sparse" / "0" / "points3D.bin"
    
    if not points_file.exists():
        return None
    
    try:
        xyz = []
        with open(points_file, "rb") as f:
            num_points = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_points):
                _pid = struct.unpack("Q", f.read(8))[0]
                x, y, z = struct.unpack("ddd", f.read(24))
                # Skip color and error
                f.read(3)  # RGB
                f.read(8)  # reprojection error
                # Skip track
                track_len = struct.unpack("Q", f.read(8))[0]
                f.read(8 * track_len)
                xyz.append([x, y, z])
        
        if xyz:
            return np.array(xyz, dtype=np.float64)
        return None
    except Exception:
        return None


def calculate_terrain_roughness(
    xyz_points: np.ndarray,
    grid_size: int = 20,
    min_points_per_cell: int = 3,
) -> float:
    """
    Calculate terrain roughness as median of per-cell plane-fit residuals.
    
    Algorithm:
    1. Bin points into XY grid (e.g., 20x20 cells)
    2. For each cell with >= min_points: fit plane z = ax + by + c via least squares
    3. Compute residuals r_i = |z_i - z_fit_i| for each point
    4. Cell score = median(residuals)
    5. Final score = median(cell scores)
    
    This approach is:
    - Slope-invariant: distinguishes slope from undulation
    - Robust to outliers: uses median aggregation
    - O(N) complexity: vectorized NumPy implementation
    - No KD-trees needed: simple bin-based approach
    
    Args:
        xyz_points: numpy array of shape (N, 3) with XYZ coordinates
        grid_size: dimension of XY grid (default 20x20)
        min_points_per_cell: minimum points to fit a plane per cell
        
    Returns:
        Terrain roughness score in [0.0, 1.0], where:
        - 0.0 = perfectly flat terrain (no variation)
        - 1.0 = highly rough/undulating terrain
        Clamped to [0.0, 1.0] for bounded output.
    """
    if xyz_points is None or len(xyz_points) < min_points_per_cell:
        return 0.0
    
    try:
        # Extract XYZ coordinates
        xyz = np.array(xyz_points, dtype=np.float64)
        if xyz.shape[1] != 3:
            return 0.0
        
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        
        # Normalize XY to [0, grid_size)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
            # Degenerate case: all points in a line or single point
            return 0.0
        
        x_norm = (x - x_min) / (x_max - x_min) * (grid_size - 0.0001)
        y_norm = (y - y_min) / (y_max - y_min) * (grid_size - 0.0001)
        
        # Assign points to grid cells
        cell_x = np.floor(x_norm).astype(np.int32)
        cell_y = np.floor(y_norm).astype(np.int32)
        cell_x = np.clip(cell_x, 0, grid_size - 1)
        cell_y = np.clip(cell_y, 0, grid_size - 1)
        cell_id = cell_y * grid_size + cell_x
        
        # Process each cell
        cell_residuals = []
        for cid in range(grid_size * grid_size):
            mask = cell_id == cid
            if mask.sum() < min_points_per_cell:
                continue
            
            x_cell = x[mask]
            y_cell = y[mask]
            z_cell = z[mask]
            
            # Fit plane z = ax + by + c via least squares
            # [x1 y1 1] [a]   [z1]
            # [x2 y2 1] [b] = [z2]
            # [...  ] [c]   [...]
            A = np.column_stack([x_cell, y_cell, np.ones(len(x_cell))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z_cell, rcond=None)
                a, b, c = coeffs
                z_fit = a * x_cell + b * y_cell + c
                residuals = np.abs(z_cell - z_fit)
                cell_score = float(np.median(residuals))
                cell_residuals.append(cell_score)
            except Exception:
                continue
        
        if not cell_residuals:
            return 0.0
        
        # Final score = median across all cells
        roughness = float(np.median(cell_residuals))
        
        # Normalize: assume max realistic roughness ~10m for typical scenes
        # This gives reasonable [0, 1] output range
        roughness_normalized = min(1.0, roughness / 10.0)
        
        return max(0.0, min(1.0, roughness_normalized))
    
    except Exception:
        return 0.0


def get_sensor_dimensions() -> dict[str, float]:
    """
    Common UAV/camera sensor dimensions (mm).
    Used as fallbacks when sensor metadata is unavailable.
    
    Returns dictionary with camera model keys -> (sensor_width, sensor_height).
    """
    return {
        # DJI drones (most common)
        "DJI_FC6310": (13.3, 8.8),    # Mavic 3 main camera
        "DJI_FC3170": (13.3, 8.8),    # Mavic 2 main
        "DJI_FC6510": (13.3, 8.8),    # Air 3
        "DJI_FC6D": (13.3, 8.8),      # Mavic 3 Classic
        
        # DJI Phantom
        "DJI_FC330": (13.2, 8.8),     # Phantom 3
        "DJI_FC350": (13.2, 8.8),     # Phantom 4
        
        # Generic full-frame equivalent
        "GENERIC_DSLR_FF": (36.0, 24.0),
        # Generic APS-C equivalent
        "GENERIC_DSLR_APSC": (23.5, 15.6),
        # Smartphone equivalent
        "GENERIC_PHONE": (8.0, 6.0),
    }


def infer_sensor_width(camera_model: str) -> float:
    """
    Infer sensor width in mm from camera model name (best-effort).
    
    Args:
        camera_model: Camera model string from EXIF
        
    Returns:
        Estimated sensor width in mm, or 13.3 (DJI default) if unknown.
    """
    sensors = get_sensor_dimensions()
    
    # Exact match
    for model, (w, _) in sensors.items():
        if model in camera_model:
            return w
    
    # Substring matching for common brands
    model_lower = camera_model.lower()
    if "mavic" in model_lower or "dji" in model_lower:
        return 13.3  # Default DJI sensor width
    elif "phantom" in model_lower:
        return 13.2
    elif "canon" in model_lower or "nikon" in model_lower:
        # DSLR/mirrorless: assume APS-C
        return 23.5
    elif "iphone" in model_lower or "samsung" in model_lower:
        return 8.0
    
    # Default to DJI (most common for UAV photogrammetry)
    return 13.3
