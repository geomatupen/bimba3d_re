import unittest

from bimba3d_backend.app.services.session_execution_mode import apply_session_execution_mode_overrides


class SessionModeControllerPipelineTests(unittest.TestCase):
    def test_train_mode_keeps_controller_pipeline_fields(self):
        requested = {
            "session_execution_mode": "train",
        }
        payload = {
            "run_count": 2,
            "ai_input_mode": "exif_plus_flight_plan_plus_external",
            "ai_selector_strategy": "continuous_bandit_linear",
            "baseline_session_id": "baseline_run_001",
            "ai_preset_override": "balanced",
            "run_jitter_mode": "fixed",
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "train")
        self.assertFalse(is_test)
        self.assertEqual(payload["ai_input_mode"], "exif_plus_flight_plan_plus_external")
        self.assertEqual(payload["ai_selector_strategy"], "continuous_bandit_linear")
        self.assertEqual(payload["baseline_session_id"], "baseline_run_001")
        self.assertEqual(payload["run_jitter_mode"], "fixed")

    def test_test_mode_strips_controller_pipeline_fields(self):
        requested = {
            "session_execution_mode": "test",
        }
        payload = {
            "run_count": 3,
            "warmup_at_start": True,
            "run_jitter_mode": "random",
            "run_jitter_factor": 1.3,
            "ai_input_mode": "exif_plus_flight_plan_plus_external",
            "ai_selector_strategy": "continuous_bandit_linear",
            "baseline_session_id": "baseline_run_002",
            "ai_preset_override": "geometry_fast",
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "test")
        self.assertTrue(is_test)
        self.assertEqual(payload["run_count"], 1)
        self.assertFalse(payload["warmup_at_start"])
        self.assertNotIn("run_jitter_mode", payload)
        self.assertNotIn("run_jitter_factor", payload)
        self.assertEqual(payload.get("ai_input_mode"), "exif_plus_flight_plan_plus_external")
        self.assertEqual(payload.get("ai_selector_strategy"), "continuous_bandit_linear")
        self.assertNotIn("baseline_session_id", payload)
        self.assertNotIn("ai_preset_override", payload)


if __name__ == "__main__":
    unittest.main()
