import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bimba3d_backend.worker.ai_input_modes import apply_initial_preset
from bimba3d_backend.worker.ai_input_modes.exif_only import EXIF_ONLY_FEATURE_KEYS
from bimba3d_backend.worker.ai_input_modes.exif_plus_flight_plan import FLIGHT_PLAN_FEATURE_KEYS
from bimba3d_backend.worker.ai_input_modes.exif_plus_flight_plan_plus_external import EXTERNAL_IMAGE_FEATURE_KEYS


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None


def _write_sample_image(path: Path) -> None:
    img = Image.new("RGB", (1600, 900), color=(60, 140, 70))
    img.save(path, format="JPEG")


class AiInputModesTests(unittest.TestCase):
    def test_legacy_mode_when_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {"feature_lr": 0.0025}
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "legacy")
            self.assertFalse(summary["applied"])
            self.assertEqual(params["feature_lr"], 0.0025)

    def test_exif_only_mode_applies_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_only",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "position_lr_final": 1.6e-6,
                "tune_interval": 100,
                "tune_min_improvement": 0.005,
                "densify_grad_threshold": 0.0002,
            }
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "exif_only")
            self.assertTrue(summary["applied"])
            self.assertIn("feature_lr", params)
            self.assertGreater(params["feature_lr"], 0.0)
            self.assertGreaterEqual(params["tune_interval"], 50)
            self.assertLessEqual(params["tune_interval"], 400)
            self.assertEqual(set(summary["features"].keys()), EXIF_ONLY_FEATURE_KEYS)

    def test_feature_summary_is_cached_per_project_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_only",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
            }
            first = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())
            second = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertFalse(first["cache_used"])
            self.assertTrue(second["cache_used"])
            self.assertEqual(first["features"], second["features"])
            self.assertEqual(first["heuristic_preset"], second["heuristic_preset"])

    def test_exif_plus_flight_plan_mode_has_exact_stacked_feature_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_plus_flight_plan",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "tune_interval": 100,
            }
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "exif_plus_flight_plan")
            self.assertTrue(summary["applied"])
            self.assertEqual(set(summary["features"].keys()), EXIF_ONLY_FEATURE_KEYS | FLIGHT_PLAN_FEATURE_KEYS)

    def test_plus_external_mode_computes_image_derived_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_plus_flight_plan_plus_external",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "tune_interval": 100,
                "colmap": {
                    "matching_type": "sequential",
                    "single_camera": True,
                    "guided_matching": True,
                    "run_image_registrator": True,
                },
            }
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "exif_plus_flight_plan_plus_external")
            self.assertTrue(summary["applied"])
            self.assertIn("vegetation_cover_percentage", summary["features"])
            self.assertIn("texture_density", summary["features"])
            self.assertIn("blur_motion_risk", summary["features"])
            self.assertEqual(
                set(summary["features"].keys()),
                EXIF_ONLY_FEATURE_KEYS | FLIGHT_PLAN_FEATURE_KEYS | EXTERNAL_IMAGE_FEATURE_KEYS,
            )


if __name__ == "__main__":
    unittest.main()
