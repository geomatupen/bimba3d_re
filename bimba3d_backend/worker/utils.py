"""
Utility functions for gsplat training.
Adapted from: https://github.com/nerfstudio-project/gsplat
"""

import torch
import numpy as np
import random


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors to spherical harmonics DC coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors distances."""
    dists = torch.cdist(points, points)
    return torch.topk(dists, k, largest=False, sorted=True)[0]
