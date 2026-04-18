import unittest

from fastapi import HTTPException

from bimba3d_backend.app.services.session_execution_mode import apply_session_execution_mode_overrides


class SessionExecutionModeTests(unittest.TestCase):
    def test_test_mode_removes_train_only_fields(self):
        requested = {
            "session_execution_mode": "test",
        }
        payload = {
            "run_count": 5,
            "warmup_at_start": True,
            "run_jitter_mode": "random",
            "run_jitter_factor": 1.2,
            "run_jitter_min": 0.5,
            "run_jitter_max": 1.5,
            "run_jitter_multiplier": 1.1,
            "continue_on_failure": True,
            "batch_connect_runs": True,
            "batch_plan_id": "batch_x",
            "batch_index": 2,
            "batch_total": 5,
            "batch_continue_on_failure": True,
            "batch_run_name_prefix": "demo",
            "warmup_phase": "phase_1",
            "ai_input_mode": "exif_only",
            "ai_selector_strategy": "preset_bias",
            "baseline_session_id": "base_run_1",
            "ai_preset_override": "balanced",
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "test")
        self.assertTrue(is_test)
        self.assertEqual(payload["session_execution_mode"], "test")
        self.assertEqual(payload["run_count"], 1)
        self.assertFalse(payload["warmup_at_start"])

        for key in (
            "run_jitter_mode",
            "run_jitter_factor",
            "run_jitter_min",
            "run_jitter_max",
            "run_jitter_multiplier",
            "continue_on_failure",
            "batch_connect_runs",
            "batch_plan_id",
            "batch_index",
            "batch_total",
            "batch_continue_on_failure",
            "batch_run_name_prefix",
            "warmup_phase",
            "baseline_session_id",
            "ai_preset_override",
        ):
            self.assertNotIn(key, payload)

        self.assertEqual(payload.get("ai_input_mode"), "exif_only")
        self.assertEqual(payload.get("ai_selector_strategy"), "preset_bias")

    def test_train_mode_keeps_train_fields(self):
        requested = {
            "session_execution_mode": "train",
        }
        payload = {
            "run_count": 3,
            "run_jitter_mode": "fixed",
            "ai_input_mode": "exif_plus_flight_plan",
            "baseline_session_id": "base_2",
        }

        mode, is_test = apply_session_execution_mode_overrides(requested, payload)

        self.assertEqual(mode, "train")
        self.assertFalse(is_test)
        self.assertEqual(payload["session_execution_mode"], "train")
        self.assertEqual(payload["run_count"], 3)
        self.assertEqual(payload["run_jitter_mode"], "fixed")
        self.assertEqual(payload["ai_input_mode"], "exif_plus_flight_plan")
        self.assertEqual(payload["baseline_session_id"], "base_2")

    def test_invalid_mode_raises(self):
        requested = {
            "session_execution_mode": "invalid",
        }
        payload = {}

        with self.assertRaises(HTTPException):
            apply_session_execution_mode_overrides(requested, payload)


if __name__ == "__main__":
    unittest.main()
