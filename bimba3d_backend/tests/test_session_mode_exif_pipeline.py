import unittest

from bimba3d_backend.app.services.session_execution_mode import apply_session_execution_mode_overrides


class SessionModeExifPipelineTests(unittest.TestCase):
    def test_train_mode_keeps_exif_pipeline_fields(self):
        requested = {
            "session_execution_mode": "train",
        }
        payload = {
            "run_count": 1,
            "ai_input_mode": "exif_only",
            "ai_selector_strategy": "preset_bias",
            "baseline_session_id": "baseline_exif_001",
            "ai_preset_override": "conservative",
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "train")
        self.assertFalse(is_test)
        self.assertEqual(payload["ai_input_mode"], "exif_only")
        self.assertEqual(payload["ai_selector_strategy"], "preset_bias")
        self.assertEqual(payload["baseline_session_id"], "baseline_exif_001")
        self.assertEqual(payload["ai_preset_override"], "conservative")

    def test_test_mode_strips_exif_pipeline_fields(self):
        requested = {
            "session_execution_mode": "test",
        }
        payload = {
            "run_count": 4,
            "warmup_at_start": True,
            "ai_input_mode": "exif_only",
            "ai_selector_strategy": "preset_bias",
            "baseline_session_id": "baseline_exif_002",
            "ai_preset_override": "balanced",
            "run_jitter_mode": "fixed",
            "run_jitter_factor": 1.1,
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "test")
        self.assertTrue(is_test)
        self.assertEqual(payload["run_count"], 1)
        self.assertFalse(payload["warmup_at_start"])
        self.assertEqual(payload.get("ai_input_mode"), "exif_only")
        self.assertEqual(payload.get("ai_selector_strategy"), "preset_bias")
        self.assertNotIn("baseline_session_id", payload)
        self.assertNotIn("ai_preset_override", payload)
        self.assertNotIn("run_jitter_mode", payload)
        self.assertNotIn("run_jitter_factor", payload)


if __name__ == "__main__":
    unittest.main()
