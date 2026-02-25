"""
COLMAP data loader for gsplat training.
Simplified version adapted from gsplat examples.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

COLMAP_TO_OPENGL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
COLMAP_TO_OPENGL_3X3 = COLMAP_TO_OPENGL[:3, :3]


def read_cameras_binary(path: Path) -> Dict:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("I", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            width = struct.unpack("Q", f.read(8))[0]
            height = struct.unpack("Q", f.read(8))[0]
            params = struct.unpack("d" * 4, f.read(8 * 4))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> Dict:
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            name_len = 0
            name_char = f.read(1)
            name = b""
            while name_char != b"\x00":
                name += name_char
                name_char = f.read(1)
                name_len += 1
            name = name.decode("utf-8")
            
            num_points = struct.unpack("Q", f.read(8))[0]
            f.read(24 * num_points)  # skip points
            
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3D_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin and return xyz and rgb arrays."""
    xyz = []
    rgb = []
    with open(path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_points):
            _pid = struct.unpack("Q", f.read(8))[0]
            x, y, z = struct.unpack("ddd", f.read(24))
            r, g, b = struct.unpack("BBB", f.read(3))
            _error = struct.unpack("d", f.read(8))[0]
            track_len = struct.unpack("Q", f.read(8))[0]
            f.read(8 * track_len)  # skip track data
            xyz.append([x, y, z])
            rgb.append([r, g, b])
    return np.array(xyz), np.array(rgb)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


class COLMAPDataset:
    """Load COLMAP reconstruction for training."""
    
    def __init__(self, colmap_dir: Path, image_dir: Path):
        self.colmap_dir = Path(colmap_dir)
        self.image_dir = Path(image_dir)
        
        # Read COLMAP data
        cameras = read_cameras_binary(self.colmap_dir / "cameras.bin")
        images = read_images_binary(self.colmap_dir / "images.bin")
        self.points, self.points_rgb = read_points3D_binary(self.colmap_dir / "points3D.bin")
        if len(self.points) > 0:
            self.points = (COLMAP_TO_OPENGL_3X3 @ self.points.T).T
        
        # Parse camera data
        self.image_names = []
        self.camtoworlds = []
        self.Ks = []
        self.image_sizes = []
        
        for img_id, img_data in sorted(images.items()):
            cam = cameras[img_data["camera_id"]]
            
            # Camera intrinsics (OPENCV model: fx, fy, cx, cy)
            fx, fy, cx, cy = cam["params"]
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # Camera extrinsics (world-to-camera)
            R = qvec2rotmat(img_data["qvec"])
            t = img_data["tvec"]
            
            # Convert to camera-to-world
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            
            c2w = COLMAP_TO_OPENGL @ c2w
            self.image_names.append(img_data["name"])
            self.camtoworlds.append(c2w)
            self.Ks.append(K)
            self.image_sizes.append((cam["width"], cam["height"]))
        
        self.camtoworlds = np.array(self.camtoworlds)
        self.Ks = np.array(self.Ks)
        
        # Compute scene scale (for normalization)
        self.scene_scale = 1.0
        if len(self.points) > 0:
            points_center = self.points.mean(axis=0)
            points_std = np.linalg.norm(self.points - points_center, axis=1).mean()
            self.scene_scale = points_std
    
    def __len__(self):
        return len(self.image_names)
