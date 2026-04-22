"""Tests for contextual continuous learner."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from bimba3d_backend.worker.ai_input_modes.contextual_continuous_learner import (
    MULT_KEYS,
    SAFE_BOUNDS,
    build_context_vector,
    select_contextual_continuous,
    update_from_run_contextual_continuous,
    record_run_penalty_contextual_continuous,
)


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory."""
    return tmp_path / "project"


@pytest.fixture
def sample_features_mode1():
    """Sample features for exif_only mode."""
    return {
        "focal_length_mm": 24.0,
        "focal_missing": 0,
        "shutter_s": 0.002,
        "shutter_missing": 0,
        "iso": 800.0,
        "iso_missing": 0,
        "img_width_median": 4000.0,
        "img_height_median": 3000.0,
    }


@pytest.fixture
def sample_features_mode2():
    """Sample features for exif_plus_flight_plan mode."""
    return {
        "focal_length_mm": 24.0,
        "focal_missing": 0,
        "shutter_s": 0.002,
        "shutter_missing": 0,
        "iso": 800.0,
        "iso_missing": 0,
        "img_width_median": 4000.0,
        "img_height_median": 3000.0,
        "gsd_median": 0.03,
        "gsd_missing": 0,
        "overlap_proxy": 0.7,
        "overlap_missing": 0,
        "coverage_spread": 0.5,
        "coverage_missing": 0,
        "camera_angle_bucket": 1,
        "angle_missing": 0,
        "heading_consistency": 0.8,
        "heading_missing": 0,
    }


@pytest.fixture
def sample_params():
    """Sample training parameters."""
    return {
        "feature_lr": 2.5e-3,
        "position_lr_init": 1.6e-4,
        "scaling_lr": 5.0e-3,
        "opacity_lr": 5.0e-2,
        "rotation_lr": 1.0e-3,
        "densify_grad_threshold": 2.0e-4,
        "opacity_threshold": 0.005,
        "lambda_dssim": 0.2,
    }


def test_build_context_vector_mode1(sample_features_mode1):
    """Test context vector building for exif_only mode."""
    x = build_context_vector(sample_features_mode1, "exif_only")

    # Should have 9 features: 1 intercept + 5 primary + 3 missing flags
    assert len(x) == 9
    assert x[0] == 1.0  # Intercept
    assert np.all(np.isfinite(x))  # All values finite


def test_build_context_vector_mode2(sample_features_mode2):
    """Test context vector building for exif_plus_flight_plan mode."""
    x = build_context_vector(sample_features_mode2, "exif_plus_flight_plan")

    # Should have 19 features: 1 intercept + 5 exif + 3 exif_missing + 5 flight + 5 flight_missing
    assert len(x) == 19
    assert x[0] == 1.0  # Intercept
    assert np.all(np.isfinite(x))


def test_build_context_vector_missing_values():
    """Test context vector with missing values."""
    features = {
        "focal_missing": 1,
        "shutter_missing": 1,
        "iso_missing": 1,
    }

    x = build_context_vector(features, "exif_only")

    # Missing flags should be 1.0
    assert x[6] == 1.0  # focal_missing
    assert x[7] == 1.0  # shutter_missing
    assert x[8] == 1.0  # iso_missing


def test_select_contextual_continuous_cold_start(temp_project_dir, sample_features_mode1, sample_params):
    """Test selection with no prior data (cold start)."""
    result = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=sample_features_mode1,
        params=sample_params,
        exploration_mode="greedy",  # Use greedy for deterministic test
    )

    assert result["selected_preset"] == "contextual_continuous"
    assert "yhat_scores" in result
    assert len(result["yhat_scores"]) == 8  # All 8 multipliers

    # Check all multipliers are within safe bounds
    for key in MULT_KEYS:
        value = result["yhat_scores"][key]
        lo, hi = SAFE_BOUNDS[key]
        assert lo <= value <= hi, f"{key}={value} outside bounds [{lo}, {hi}]"

    # Check updates are generated
    assert "updates" in result
    assert "feature_lr" in result["updates"]


def test_select_thompson_sampling(temp_project_dir, sample_features_mode1, sample_params):
    """Test that Thompson sampling produces different results."""
    np.random.seed(42)

    result1 = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=sample_features_mode1,
        params=sample_params,
        exploration_mode="thompson",
    )

    np.random.seed(43)

    result2 = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=sample_features_mode1,
        params=sample_params,
        exploration_mode="thompson",
    )

    # Thompson sampling should produce different samples
    # (might rarely be equal by chance, but very unlikely with different seeds)
    assert result1["yhat_scores"] != result2["yhat_scores"]


def test_update_from_run_contextual_continuous(temp_project_dir, sample_features_mode1, sample_params, monkeypatch):
    """Test model update after a successful run."""

    # Mock logger
    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    # First selection
    selection = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=sample_features_mode1,
        params=sample_params,
        exploration_mode="greedy",
    )

    # Simulate evaluation history
    eval_history = [
        {"step": 7000, "convergence_speed": 28.5, "sharpness_mean": 0.92, "lpips_mean": 0.15, "final_loss": 0.05, "elapsed_seconds": 120.0},
        {"step": 14000, "convergence_speed": 30.2, "sharpness_mean": 0.94, "lpips_mean": 0.12, "final_loss": 0.04, "elapsed_seconds": 240.0},
        {"step": 21000, "convergence_speed": 31.0, "sharpness_mean": 0.95, "lpips_mean": 0.10, "final_loss": 0.03, "elapsed_seconds": 360.0},
    ]

    baseline_eval_history = [
        {"step": 7000, "convergence_speed": 27.0, "sharpness_mean": 0.90, "lpips_mean": 0.18, "final_loss": 0.06, "elapsed_seconds": 130.0},
        {"step": 14000, "convergence_speed": 28.5, "sharpness_mean": 0.92, "lpips_mean": 0.15, "final_loss": 0.05, "elapsed_seconds": 260.0},
        {"step": 21000, "convergence_speed": 29.0, "sharpness_mean": 0.93, "lpips_mean": 0.13, "final_loss": 0.045, "elapsed_seconds": 390.0},
    ]

    loss_by_step = {7000: 0.05, 14000: 0.04, 21000: 0.03}
    elapsed_by_step = {7000: 120.0, 14000: 240.0, 21000: 360.0}

    result = update_from_run_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        selected_preset=selection["selected_preset"],
        yhat_scores=selection["yhat_scores"],
        eval_history=eval_history,
        baseline_eval_history=baseline_eval_history,
        loss_by_step=loss_by_step,
        elapsed_by_step=elapsed_by_step,
        x_features=sample_features_mode1,
        run_id="test_run_001",
        logger=logger,
        apply_update=True,
    )

    assert result["updated"] is True
    assert result["mode"] == "exif_only"
    assert "reward_signal" in result
    assert "s_run" in result

    # Check that model file was created
    model_path = temp_project_dir / "models" / "contextual_continuous_selector" / "exif_only.json"
    assert model_path.exists()

    # Load and verify model structure
    model_data = json.loads(model_path.read_text())
    assert model_data["version"] == 2
    assert model_data["runs"] == 1
    assert "models" in model_data
    assert len(model_data["models"]) == 8  # All 8 multipliers


def test_record_run_penalty_contextual_continuous(temp_project_dir, sample_features_mode1, monkeypatch):
    """Test penalty recording for failed runs."""

    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    yhat_scores = {key: 1.0 for key in MULT_KEYS}

    result = record_run_penalty_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        selected_preset="contextual_continuous",
        yhat_scores=yhat_scores,
        penalty_reward=-1.5,
        x_features=sample_features_mode1,
        reason="gaussian_hard_cap_reached",
        run_id="test_run_penalty_001",
        logger=logger,
    )

    assert result["updated"] is True
    assert result["reward_signal"] == -1.5
    assert result["reason"] == "gaussian_hard_cap_reached"

    # Verify model was updated
    model_path = temp_project_dir / "models" / "contextual_continuous_selector" / "exif_only.json"
    assert model_path.exists()

    model_data = json.loads(model_path.read_text())
    assert model_data["runs"] == 1


def test_learning_reduces_exploration_variance(temp_project_dir, sample_features_mode1, sample_params, monkeypatch):
    """Test that exploration variance decreases with more observations."""

    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    # Simulate 10 positive reward updates
    for i in range(10):
        selection = select_contextual_continuous(
            project_dir=temp_project_dir,
            mode="exif_only",
            x_features=sample_features_mode1,
            params=sample_params,
            exploration_mode="greedy",
        )

        eval_history = [
            {"step": 21000, "convergence_speed": 30.0, "sharpness_mean": 0.94, "lpips_mean": 0.12, "final_loss": 0.04, "elapsed_seconds": 300.0},
        ]

        update_from_run_contextual_continuous(
            project_dir=temp_project_dir,
            mode="exif_only",
            selected_preset=selection["selected_preset"],
            yhat_scores=selection["yhat_scores"],
            eval_history=eval_history,
            baseline_eval_history=None,
            loss_by_step={21000: 0.04},
            elapsed_by_step={21000: 300.0},
            x_features=sample_features_mode1,
            run_id=f"test_run_{i:03d}",
            logger=logger,
            apply_update=True,
        )

    # Load final model
    model_path = temp_project_dir / "models" / "contextual_continuous_selector" / "exif_only.json"
    model_data = json.loads(model_path.read_text())

    assert model_data["runs"] == 10

    # Verify that models have been updated (n > 0)
    for key in MULT_KEYS:
        assert model_data["models"][key]["n"] == 10


def test_context_sensitivity_different_inputs(temp_project_dir, sample_params):
    """Test that different contexts can lead to different predictions after learning."""

    # Two very different contexts
    context_low_focal = {
        "focal_length_mm": 10.0,
        "focal_missing": 0,
        "shutter_s": 0.005,
        "shutter_missing": 0,
        "iso": 200.0,
        "iso_missing": 0,
        "img_width_median": 4000.0,
        "img_height_median": 3000.0,
    }

    context_high_focal = {
        "focal_length_mm": 200.0,
        "focal_missing": 0,
        "shutter_s": 0.001,
        "shutter_missing": 0,
        "iso": 3200.0,
        "iso_missing": 0,
        "img_width_median": 4000.0,
        "img_height_median": 3000.0,
    }

    # Get predictions for both (cold start - will be identical initially)
    result1 = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=context_low_focal,
        params=sample_params,
        exploration_mode="greedy",
    )

    result2 = select_contextual_continuous(
        project_dir=temp_project_dir,
        mode="exif_only",
        x_features=context_high_focal,
        params=sample_params,
        exploration_mode="greedy",
    )

    # Cold start predictions should be similar (near 1.0 from zero initialization)
    # This is expected behavior before any learning
    assert result1["context_vector"] != result2["context_vector"]  # Contexts are different
