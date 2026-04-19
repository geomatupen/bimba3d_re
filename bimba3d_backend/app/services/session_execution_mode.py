from typing import Any, Mapping

from fastapi import HTTPException


def apply_session_execution_mode_overrides(
    requested_params: Mapping[str, Any],
    params_payload: dict[str, Any],
) -> tuple[str, bool]:
    """Validate session mode and apply mode-specific payload sanitization."""
    session_execution_mode = str(requested_params.get("session_execution_mode") or "train").strip().lower()
    if session_execution_mode not in {"train", "test"}:
        raise HTTPException(status_code=400, detail="session_execution_mode must be 'train' or 'test'")

    is_session_test = session_execution_mode == "test"
    params_payload["session_execution_mode"] = session_execution_mode

    if is_session_test:
        # Test sessions should run a single direct execution without train orchestration controls.
        params_payload["run_count"] = 1
        params_payload["warmup_at_start"] = False
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
        ):
            params_payload.pop(key, None)

        for key in (
            "ai_preset_override",
        ):
            params_payload.pop(key, None)

    return session_execution_mode, is_session_test
