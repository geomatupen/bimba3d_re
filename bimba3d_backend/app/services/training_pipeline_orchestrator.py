"""Training pipeline orchestrator - executes cross-project training with thermal management."""
from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.context_jitter import apply_context_jitter, apply_mild_jitter

logger = logging.getLogger(__name__)

# Global registry of running orchestrators
_running_orchestrators: dict[str, "PipelineOrchestrator"] = {}


class PipelineOrchestrator:
    """Orchestrates multi-project training pipeline execution with thermal management."""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.current_run_project_name: Optional[str] = None

    def start(self):
        """Start orchestrator in background thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"Pipeline {self.pipeline_id} orchestrator already running")
            return

        self.should_stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        _running_orchestrators[self.pipeline_id] = self
        logger.info(f"Started orchestrator for pipeline {self.pipeline_id}")

    def stop(self):
        """Signal orchestrator to stop."""
        self.should_stop = True
        logger.info(f"Stopping orchestrator for pipeline {self.pipeline_id}")

    def pause(self):
        """Pause execution (will wait for current run to complete)."""
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if pipeline:
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"status": "paused"})

    def resume(self):
        """Resume execution."""
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if pipeline:
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"status": "running"})

    def _run(self):
        """Main orchestrator loop."""
        try:
            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
            if not pipeline:
                logger.error(f"Pipeline {self.pipeline_id} not found")
                return

            config = pipeline["config"]
            phases = config["phases"]
            projects = config["projects"]
            thermal = config.get("thermal_management", {})

            # Execute each phase
            for phase_idx, phase in enumerate(phases):
                if self.should_stop:
                    break

                phase_num = phase["phase_number"]
                training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_phase": phase_num})

                logger.info(f"Pipeline {self.pipeline_id}: Starting Phase {phase_num} - {phase['name']}")

                # Execute passes within phase
                passes = phase.get("passes", 1)
                for pass_idx in range(passes):
                    if self.should_stop:
                        break

                    current_pass = pass_idx + 1
                    training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_pass": current_pass})

                    # Shuffle project order if requested
                    project_order = list(range(len(projects)))
                    if phase.get("shuffle_order", False):
                        random.shuffle(project_order)

                    # Execute each project in this pass
                    for proj_idx in project_order:
                        if self.should_stop:
                            break

                        # Check if paused
                        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                        while pipeline and pipeline["status"] == "paused":
                            logger.info(f"Pipeline {self.pipeline_id} paused, waiting...")
                            time.sleep(5)
                            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)

                        project = projects[proj_idx]
                        training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_project_index": proj_idx})

                        # Execute training run
                        self._execute_run(pipeline, project, phase, pass_idx + 1)

                        # Apply thermal management (cooldown)
                        if thermal.get("enabled", False):
                            self._apply_thermal_management(thermal)

            # Mark pipeline as completed
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })

            logger.info(f"Pipeline {self.pipeline_id} completed successfully")

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id} failed: {e}")
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "status": "failed",
                "last_error": str(e),
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })

        finally:
            _running_orchestrators.pop(self.pipeline_id, None)

    def _get_or_create_project_dir(self, pipeline: dict, project: dict) -> Path:
        """Get or create project directory within pipeline folder.

        Structure:
          {pipeline_folder}/
            ├── shared_models/  ← Shared models for cross-project learning
            │   └── contextual_continuous_selector/
            ├── {project1_name}/  ← Project directory (COLMAP, runs)
            ├── {project2_name}/
            └── ...

        Projects reference original data via config.json source_dir
        Shared model directory enables cross-project knowledge accumulation
        """
        config = pipeline["config"]
        pipeline_folder = Path(config["pipeline_folder"])
        project_name = project["name"]
        project_dir = pipeline_folder / project_name

        # Create shared model directory at pipeline level
        shared_model_dir = pipeline_folder / "shared_models"
        shared_model_dir.mkdir(parents=True, exist_ok=True)

        # Create project directory if it doesn't exist
        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)

            # Create config.json with reference to source data
            config_data = {
                "id": str(uuid.uuid4()),
                "name": project_name,
                "source_dir": project["dataset_path"],  # Points to read-only data folder
                "shared_model_dir": str(shared_model_dir),  # Shared models for cross-project learning
                "created_at": datetime.utcnow().isoformat() + "Z",
                "created_by": "training_pipeline",
                "pipeline_id": pipeline["id"],
                "pipeline_name": config.get("name"),
                **config.get("shared_config", {})
            }

            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Created project directory: {project_dir} with shared models at {shared_model_dir}")

        return project_dir

    def _execute_run(self, pipeline: dict, project: dict, phase: dict, pass_num: int):
        """Execute a single training run."""
        project_name = project["name"]
        self.current_run_project_name = project_name

        logger.info(f"Pipeline {self.pipeline_id}: Running {project_name}, phase {phase['phase_number']}, pass {pass_num}")

        try:
            # Get/create project directory in pipeline folder
            project_dir = self._get_or_create_project_dir(pipeline, project)

            # Build run configuration
            run_config = self._build_run_config(pipeline, project, phase, pass_num)

            # Execute training
            # NOTE: This is a placeholder - actual implementation would call gsplat_engine
            # For now, we'll simulate a run
            success, reward = self._simulate_training_run(run_config)

            # Record result
            run_result = {
                "project_name": project_name,
                "phase": phase["phase_number"],
                "pass": pass_num,
                "status": "success" if success else "failed",
                "reward": reward,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            training_pipeline_storage.add_run_result(self.pipeline_id, run_result)

            if success:
                logger.info(f"Pipeline {self.pipeline_id}: Run completed successfully, reward={reward}")
            else:
                logger.warning(f"Pipeline {self.pipeline_id}: Run failed")

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id}: Run execution failed: {e}")
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "failed_runs": pipeline.get("failed_runs", 0) + 1,
                "last_error": str(e)
            })

        finally:
            self.current_run_project_name = None

    def _build_run_config(self, pipeline: dict, project: dict, phase: dict, pass_num: int) -> dict:
        """Build configuration for a single training run."""
        config = pipeline["config"]
        shared_config = config["shared_config"]

        # Start with shared config
        run_config = shared_config.copy()

        # Apply phase-specific overrides
        if phase.get("strategy_override"):
            run_config["ai_selector_strategy"] = phase["strategy_override"]

        if phase.get("preset_override"):
            run_config["preset_override"] = phase["preset_override"]

        # Context jitter for multi-pass learning
        if phase.get("context_jitter", False):
            run_config["context_jitter_enabled"] = True
            # Jitter mode: "uniform" (sample from bounds), "mild" (±10%), or "gaussian" (±15%)
            run_config["context_jitter_mode"] = phase.get("context_jitter_mode", "uniform")

        # Session execution mode
        run_config["session_execution_mode"] = phase.get("session_execution_mode", "train")

        # Update model flag
        run_config["update_model"] = phase.get("update_model", True)

        # Baseline reference
        if project.get("baseline_run_id"):
            run_config["baseline_session_id"] = project["baseline_run_id"]

        # Run metadata
        run_config["pipeline_id"] = pipeline["id"]
        run_config["phase_number"] = phase["phase_number"]
        run_config["pass_number"] = pass_num

        return run_config

    def _simulate_training_run(self, run_config: dict) -> tuple[bool, Optional[float]]:
        """Simulate a training run (placeholder for actual implementation)."""
        # TODO: Integrate with actual gsplat_engine execution
        # This would call: bimba3d_backend.app.services.gsplat.run_training(project_dir, run_config)

        # Simulate training time
        time.sleep(2)  # Placeholder for actual training

        # Simulate success/failure and reward
        success = random.random() > 0.1  # 90% success rate
        reward = random.uniform(-0.2, 0.3) if success else None

        return success, reward

    def _apply_thermal_management(self, thermal_config: dict):
        """Apply thermal management strategy (cooldown period)."""
        strategy = thermal_config.get("strategy", "fixed_interval")

        training_pipeline_storage.update_pipeline(self.pipeline_id, {"cooldown_active": True})

        if strategy == "fixed_interval":
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            logger.info(f"Pipeline {self.pipeline_id}: Cooling down for {cooldown_minutes} minutes")

            # Calculate next run time
            next_run = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "next_run_scheduled_at": next_run.isoformat() + "Z"
            })

            # Wait
            time.sleep(cooldown_minutes * 60)

        elif strategy == "temperature_based":
            # TODO: Implement GPU temperature monitoring
            # For now, fall back to fixed interval
            logger.warning("Temperature-based cooldown not yet implemented, using fixed interval")
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            time.sleep(cooldown_minutes * 60)

        elif strategy == "time_scheduled":
            # TODO: Implement time-of-day scheduling
            logger.warning("Time-scheduled cooldown not yet implemented, using fixed interval")
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            time.sleep(cooldown_minutes * 60)

        training_pipeline_storage.update_pipeline(self.pipeline_id, {
            "cooldown_active": False,
            "next_run_scheduled_at": None,
            "last_run_ended_at": datetime.utcnow().isoformat() + "Z"
        })


# ========== Public API ==========

def start_pipeline_orchestrator(pipeline_id: str):
    """Start pipeline orchestrator in background."""
    orchestrator = PipelineOrchestrator(pipeline_id)
    orchestrator.start()
    return orchestrator


def stop_pipeline_orchestrator(pipeline_id: str):
    """Stop running orchestrator."""
    orchestrator = _running_orchestrators.get(pipeline_id)
    if orchestrator:
        orchestrator.stop()


def get_orchestrator_status(pipeline_id: str) -> Optional[dict]:
    """Get current orchestrator status."""
    orchestrator = _running_orchestrators.get(pipeline_id)
    if not orchestrator:
        return None

    return {
        "is_running": orchestrator.thread and orchestrator.thread.is_alive(),
        "current_project": orchestrator.current_run_project_name,
    }
