"""Training pipeline orchestrator - executes cross-project training with thermal management."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import status as project_status
from bimba3d_backend.app.services import training_pipeline_storage

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

                    # Execute each project in this pass and run repetitions.
                    for proj_idx in project_order:
                        if self.should_stop:
                            break

                        project = projects[proj_idx]
                        training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_project_index": proj_idx})

                        runs_per_project = max(int(phase.get("runs_per_project", 1) or 1), 1)
                        for run_idx in range(runs_per_project):
                            if self.should_stop:
                                break

                            # Reload pipeline to get latest config (e.g., baseline_run_id from previous phase)
                            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                            if pipeline:
                                # Refresh project dict to get any updates (baseline_run_id, etc.)
                                updated_projects = pipeline.get("config", {}).get("projects", [])
                                if proj_idx < len(updated_projects):
                                    project = updated_projects[proj_idx]

                            # Check if paused between runs.
                            while pipeline and pipeline["status"] == "paused" and not self.should_stop:
                                logger.info(f"Pipeline {self.pipeline_id} paused, waiting...")
                                time.sleep(5)
                                pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)

                            if self.should_stop:
                                break
                            if not pipeline:
                                logger.error(f"Pipeline {self.pipeline_id} disappeared during execution")
                                self.should_stop = True
                                break
                            if pipeline.get("status") in {"stopped", "failed"}:
                                self.should_stop = True
                                break

                            # Execute training run
                            run_outcome = self._execute_run(
                                pipeline,
                                project,
                                phase,
                                pass_idx + 1,
                                run_idx + 1,
                                runs_per_project,
                            )

                            # Skipped runs should not leave cooldown UI state active.
                            if run_outcome == "skipped":
                                training_pipeline_storage.update_pipeline(self.pipeline_id, {
                                    "cooldown_active": False,
                                    "next_run_scheduled_at": None,
                                })

                            # Apply thermal management (cooldown) between individual runs.
                            if (
                                thermal.get("enabled", False)
                                and not self.should_stop
                                and run_outcome != "skipped"
                            ):
                                self._apply_thermal_management(thermal)

            # Determine final status based on run results
            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
            failed_runs = pipeline.get("failed_runs", 0)
            completed_runs = pipeline.get("completed_runs", 0)
            total_runs = pipeline.get("total_runs", 0)

            # Determine status: completed only if no failures and all runs succeeded
            if self.should_stop:
                final_status = "stopped"
            elif failed_runs > 0 and completed_runs == 0:
                # All runs failed
                final_status = "failed"
            elif failed_runs > 0:
                # Some runs failed but some succeeded
                final_status = "completed_with_failures"
            else:
                # All runs succeeded
                final_status = "completed"

            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "status": final_status,
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })

            logger.info(f"Pipeline {self.pipeline_id} finished with status: {final_status} ({completed_runs}/{total_runs} successful)")

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

        If colmap_source_project_id is specified, copies COLMAP from that project
        """
        config = pipeline["config"]
        pipeline_folder = Path(config["pipeline_folder"])
        project_name = project["name"]
        # Sanitize project name for folder (replace spaces with underscores)
        sanitized_project_name = project_name.replace(" ", "_")
        project_dir = pipeline_folder / sanitized_project_name
        colmap_source_project_id = project.get("colmap_source_project_id")

        # Create shared model directory at pipeline level
        shared_model_dir = pipeline_folder / "shared_models"
        shared_model_dir.mkdir(parents=True, exist_ok=True)

        # Create project directory if it doesn't exist
        project_created = False
        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)
            project_created = True

            # Create symlink to source images directory
            source_path = Path(project["dataset_path"])
            images_link = project_dir / "images"

            if not images_link.exists():
                symlink_success = False
                try:
                    # Create symlink (Windows requires admin or developer mode)
                    import os
                    if os.name == 'nt':
                        # On Windows, try junction first (doesn't require admin)
                        try:
                            import subprocess
                            result = subprocess.run(['mklink', '/J', str(images_link), str(source_path)],
                                         shell=True, check=True, capture_output=True, text=True)
                            symlink_success = True
                            logger.info(f"Created images junction: {images_link} -> {source_path}")
                        except Exception as junction_err:
                            logger.warning(f"Junction creation failed: {junction_err}, trying symlink")
                            try:
                                images_link.symlink_to(source_path, target_is_directory=True)
                                symlink_success = True
                                logger.info(f"Created images symlink: {images_link} -> {source_path}")
                            except OSError as symlink_err:
                                logger.warning(f"Symlink creation also failed: {symlink_err}")
                    else:
                        # Unix/Linux: standard symlink
                        images_link.symlink_to(source_path, target_is_directory=True)
                        symlink_success = True
                        logger.info(f"Created images symlink: {images_link} -> {source_path}")

                except Exception as e:
                    logger.warning(f"Failed to create images symlink/junction: {e}")
                
                # If symlink/junction failed, copy images instead
                if not symlink_success:
                    logger.info(f"Falling back to copying images from {source_path} to {images_link}")
                    try:
                        import shutil
                        images_link.mkdir(parents=True, exist_ok=True)
                        
                        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                        copied_count = 0
                        
                        for img_file in source_path.iterdir():
                            if img_file.is_file() and img_file.suffix in image_extensions:
                                dest_file = images_link / img_file.name
                                if not dest_file.exists():
                                    shutil.copy2(img_file, dest_file)
                                    copied_count += 1
                        
                        logger.info(f"✓ Copied {copied_count} images to project folder")
                        
                        if copied_count == 0:
                            raise RuntimeError(f"No images found in source directory: {source_path}")
                    
                    except Exception as copy_err:
                        logger.error(f"Failed to copy images: {copy_err}")
                        # Create a note file for debugging
                        (project_dir / "images_source.txt").write_text(str(source_path))
                        raise RuntimeError(f"Failed to set up images directory: {copy_err}") from copy_err

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
                "pipeline_path": str(pipeline_folder),  # Direct path to pipeline folder for efficient lookup
                **config.get("shared_config", {})
            }

            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            # Store project_id in pipeline config for UI navigation
            self._update_project_id_in_pipeline(pipeline, project["name"], config_data["id"])

            logger.info(f"Created project directory: {project_dir} with shared models at {shared_model_dir}")

        # Copy COLMAP from source project if specified (do this even if project already exists)
        # This allows restarting pipelines with COLMAP copy enabled
        if colmap_source_project_id:
            target_colmap = project_dir / "outputs" / "sparse"
            if not target_colmap.exists():
                self._copy_colmap_from_source(project_dir, colmap_source_project_id, project_name)
            else:
                logger.info(f"COLMAP already exists for {project_name}, skipping copy")

        return project_dir

    def _update_project_id_in_pipeline(self, pipeline: dict, project_name: str, project_id: str):
        """Update project_id in pipeline config for UI navigation."""
        try:
            config = pipeline.get("config", {})
            projects = config.get("projects", [])
            for proj in projects:
                if proj.get("name") == project_name and not proj.get("project_id"):
                    proj["project_id"] = project_id
                    training_pipeline_storage.update_pipeline(pipeline["id"], {"config": config})
                    break
        except Exception as e:
            logger.warning(f"Failed to update project_id in pipeline config: {e}")

    def _copy_colmap_from_source(self, target_project_dir: Path, source_project_id: str, target_project_name: str):
        """Copy COLMAP sparse reconstruction and config from source project to target project.
        
        Uses the same project lookup logic as the rest of the system to find projects
        in DATA_DIR, pipeline folders, etc.
        """
        try:
            import shutil
            from bimba3d_backend.app.api.projects import _find_project_dir

            # Use the standard project finder which searches DATA_DIR and pipeline folders
            source_project_dir = _find_project_dir(source_project_id)
            
            if not source_project_dir:
                logger.warning(f"Source project {source_project_id} not found, skipping COLMAP copy")
                return
            
            logger.info(f"Found source project for COLMAP copy: {source_project_dir}")

            # Check if source has COLMAP outputs
            source_colmap = source_project_dir / "outputs" / "sparse"
            if not source_colmap.exists() or not (source_colmap / "0").exists():
                logger.warning(f"Source project {source_project_id} has no COLMAP outputs, skipping copy")
                return

            # Create target outputs directory
            target_outputs = target_project_dir / "outputs"
            target_outputs.mkdir(parents=True, exist_ok=True)

            # Copy COLMAP sparse directory
            target_colmap = target_outputs / "sparse"
            if target_colmap.exists():
                logger.info(f"Target already has COLMAP, skipping copy for {target_project_name}")
            else:
                shutil.copytree(source_colmap, target_colmap)
                logger.info(f"✓ Copied COLMAP from project {source_project_id} to {target_project_name} (saved ~15-30 minutes)")

            # Copy COLMAP-related config settings from source project
            source_config_path = source_project_dir / "config.json"
            target_config_path = target_project_dir / "config.json"
            
            if source_config_path.exists() and target_config_path.exists():
                try:
                    with open(source_config_path, "r") as f:
                        source_config = json.load(f)
                    
                    with open(target_config_path, "r") as f:
                        target_config = json.load(f)
                    
                    # Copy COLMAP-related settings
                    colmap_keys = [
                        "colmap_camera_model",
                        "colmap_camera_params",
                        "colmap_matcher",
                        "colmap_vocab_tree_path",
                        "colmap_gpu_index",
                        "colmap_num_threads",
                        "colmap_use_gpu",
                        "image_width",
                        "image_height",
                        "focal_length",
                        "camera_model",
                    ]
                    
                    updated = False
                    for key in colmap_keys:
                        if key in source_config and key not in target_config:
                            target_config[key] = source_config[key]
                            updated = True
                    
                    if updated:
                        with open(target_config_path, "w") as f:
                            json.dump(target_config, f, indent=2)
                        logger.info(f"✓ Copied COLMAP config settings to {target_project_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to copy COLMAP config settings: {e}")

        except Exception as e:
            logger.error(f"Failed to copy COLMAP from source project: {e}", exc_info=True)
            # Non-fatal: pipeline will run COLMAP if copy fails

    def _execute_run(
        self,
        pipeline: dict,
        project: dict,
        phase: dict,
        pass_num: int,
        run_num: int,
        runs_per_project: int,
    ) -> str:
        """Execute a single training run.

        Returns:
            "skipped" when no run was executed (already completed)
            "completed" when a run was attempted (success/failure)
        """
        project_name = project["name"]
        self.current_run_project_name = project_name
        run_id: Optional[str] = None

        logger.info(
            f"Pipeline {self.pipeline_id}: Running {project_name}, "
            f"phase {phase['phase_number']}, pass {pass_num}, run {run_num}/{runs_per_project}"
        )

        try:
            # Get/create project directory in pipeline folder
            project_dir = self._get_or_create_project_dir(pipeline, project)

            # Check if this run already completed successfully in pipeline history
            for existing in pipeline.get("runs", []):
                if (
                    str(existing.get("project_name") or "") == project_name
                    and str(existing.get("phase")) == str(phase["phase_number"])
                    and str(existing.get("pass")) == str(pass_num)
                    and str(existing.get("run", 1)) == str(run_num)
                    and str(existing.get("status") or "") == "success"
                ):
                    logger.info(
                        f"Pipeline {self.pipeline_id}: Run already completed for {project_name} "
                        f"phase {phase['phase_number']} pass {pass_num} run {run_num}, skipping"
                    )
                    return "skipped"

            # For phase 1 (baseline), also check if baseline run directory exists and succeeded
            if phase["phase_number"] == 1:
                runs_root = project_dir / "runs"
                if runs_root.exists() and runs_root.is_dir():
                    # Look for existing baseline run (first run alphabetically)
                    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
                    if run_dirs:
                        existing_baseline_dir = run_dirs[0]
                        # Check if it completed successfully
                        analytics_file = existing_baseline_dir / "analytics" / "run_analytics_v1.json"
                        if analytics_file.exists():
                            try:
                                with open(analytics_file, "r", encoding="utf-8") as fh:
                                    analytics = json.load(fh)
                                    summary = analytics.get("summary", {})
                                    status = str(summary.get("status", "")).lower()
                                    if status in {"completed", "success", "done"}:
                                        logger.info(
                                            f"Pipeline {self.pipeline_id}: Baseline run {existing_baseline_dir.name} "
                                            f"already completed successfully for {project_name}, skipping"
                                        )
                                        # Store baseline_run_id if not already stored
                                        if not project.get("baseline_run_id"):
                                            config = pipeline.get("config", {})
                                            projects = config.get("projects", [])
                                            for proj in projects:
                                                if proj.get("name") == project_name:
                                                    proj["baseline_run_id"] = existing_baseline_dir.name
                                                    training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
                                                    logger.info(f"Stored existing baseline_run_id={existing_baseline_dir.name} for project {project_name}")
                                                    break
                                        return "skipped"
                                    else:
                                        logger.info(
                                            f"Pipeline {self.pipeline_id}: Existing baseline run {existing_baseline_dir.name} "
                                            f"for {project_name} has status '{status}' - will re-run"
                                        )
                            except Exception as e:
                                logger.warning(f"Failed to read baseline analytics for {project_name}: {e}")

            # Build run configuration
            run_config = self._build_run_config(pipeline, project, phase, pass_num, run_num, runs_per_project)

            # Generate unique run ID with project name prefix
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            # Sanitize project name for use in run_id
            project_name_safe = project_name.lower().replace(" ", "_").replace("-", "_")
            run_id = f"{project_name_safe}_phase{phase['phase_number']}_pass{pass_num}_run{run_num}_{timestamp}"

            # Execute actual training
            logger.info(
                f"Pipeline {self.pipeline_id}: Starting training run {run_id} for {project_name} "
                f"phase {phase['phase_number']} pass {pass_num} run {run_num}/{runs_per_project}"
            )
            success, reward = self._execute_training_run(run_config, project_dir, run_id)

            # Generate run name
            phase_name = phase.get("name", f"Phase {phase['phase_number']}")
            run_name = f"{phase_name} - Phase {phase['phase_number']} Run {run_num}"

            # Record result
            run_result = {
                "project_name": project_name,
                "phase": phase["phase_number"],
                "pass": pass_num,
                "run": run_num,
                "run_id": run_id,
                "run_name": run_name,
                "status": "success" if success else "failed",
                "reward": reward,
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            training_pipeline_storage.add_run_result(self.pipeline_id, run_result)

            if success:
                logger.info(f"Pipeline {self.pipeline_id}: Run completed successfully, reward={reward}")

                # If this was a baseline run (phase 1), store run_id as baseline for future phases
                if phase["phase_number"] == 1:
                    # Update the project dict in pipeline config to include baseline_run_id
                    pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                    if pipeline:
                        config = pipeline.get("config", {})
                        projects = config.get("projects", [])
                        for proj in projects:
                            if proj.get("name") == project_name:
                                proj["baseline_run_id"] = run_id
                                training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
                                logger.info(f"Stored baseline_run_id={run_id} for project {project_name}")
                                break
            else:
                logger.warning(f"Pipeline {self.pipeline_id}: Run failed")

            return "completed"

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id}: Run execution failed: {e}")
            training_pipeline_storage.add_run_result(
                self.pipeline_id,
                {
                    "project_name": project_name,
                    "phase": phase.get("phase_number"),
                    "pass": pass_num,
                    "run": run_num,
                    "run_id": run_id,
                    "status": "failed",
                    "reward": None,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"last_error": str(e)})
            return "completed"

        finally:
            self.current_run_project_name = None

    def _build_run_config(
        self,
        pipeline: dict,
        project: dict,
        phase: dict,
        pass_num: int,
        run_num: int,
        runs_per_project: int,
    ) -> dict:
        """Build configuration for a single training run."""
        config = pipeline["config"]
        shared_config = config["shared_config"]

        # Start with shared config
        run_config = shared_config.copy()

        # Apply phase-specific overrides
        if phase.get("strategy_override"):
            run_config["ai_selector_strategy"] = phase["strategy_override"]

        # For non-baseline phases, ensure AI input mode is preserved
        # unless explicitly overridden by phase config
        if not phase.get("preset_override") and "ai_input_mode" in shared_config:
            run_config["ai_input_mode"] = shared_config["ai_input_mode"]

        # Pipeline AI phases must use the same Core AI optimization path as manual
        # project runs. Without this, the worker applies an initial AI preset but
        # never writes input_mode_learning_results.json or updates the learner model.
        if run_config.get("ai_input_mode") and phase.get("phase_number") != 1:
            run_config["tune_scope"] = "core_ai_optimization"

        if phase.get("preset_override"):
            run_config["preset_override"] = phase["preset_override"]

        # Context jitter for multi-pass learning (always use uniform mode for full exploration)
        if phase.get("context_jitter", False):
            run_config["context_jitter_enabled"] = True
            run_config["context_jitter_mode"] = "uniform"  # Fixed: uniform sampling from feature bounds

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
        run_config["phase_run"] = run_num
        run_config["phase_runs_total"] = runs_per_project

        return run_config

    def _execute_training_run(self, run_config: dict, project_dir: Path, run_id: str) -> tuple[bool, Optional[float]]:
        """Execute actual training run using the project pipeline system.

        This calls the same training system used by individual projects.
        """
        from bimba3d_backend.worker import pipeline

        phase_num = run_config.get("phase_number", 1)

        # Determine stage based on phase and whether COLMAP is complete and successful
        colmap_sparse_dir = project_dir / "outputs" / "sparse" / "0"
        colmap_complete = False

        if colmap_sparse_dir.exists():
            # Check if COLMAP completed successfully by looking for required files
            cameras_file = colmap_sparse_dir / "cameras.bin"
            images_file = colmap_sparse_dir / "images.bin"
            points_file = colmap_sparse_dir / "points3D.bin"

            if cameras_file.exists() and images_file.exists() and points_file.exists():
                # All required COLMAP files exist, consider it complete
                colmap_complete = True
                logger.info(f"✓ COLMAP outputs found and complete for {project_dir.name}")
            else:
                logger.warning(f"⚠ COLMAP directory exists but incomplete for {project_dir.name} - will re-run COLMAP")

        if phase_num == 1 and not colmap_complete:
            # Phase 1 and COLMAP not complete: Run full pipeline (COLMAP + training)
            stage = "full"
            logger.info(f"Phase 1: Running full pipeline (COLMAP + training) for {project_dir.name}")
        else:
            # Phase 2+ OR COLMAP already complete: Only run training
            stage = "train_only"
            if phase_num == 1:
                logger.info(f"Phase 1: Skipping COLMAP (already complete), running training only for {project_dir.name}")

        # Build params for the worker
        params = {
            "run_id": run_id,
            "stage": stage,
            "mode": "baseline" if phase_num == 1 else "modified",
            "max_steps": run_config.get("max_steps", 5000),
            "eval_interval": run_config.get("eval_interval", 1000),
            "log_interval": run_config.get("log_interval", 100),
            "densify_until_iter": run_config.get("densify_until_iter", 4000),
            "images_max_size": run_config.get("images_max_size"),
            "ai_input_mode": run_config.get("ai_input_mode"),
            "ai_selector_strategy": run_config.get("ai_selector_strategy"),
            "tune_scope": run_config.get("tune_scope"),
            "trend_scope": run_config.get("trend_scope", "run"),
            "session_execution_mode": run_config.get("session_execution_mode", "train"),
            "baseline_session_id": run_config.get("baseline_session_id"),
            "update_model": run_config.get("update_model", True),
            "context_jitter_enabled": run_config.get("context_jitter_enabled", False),
            "context_jitter_mode": "uniform",  # Always use uniform mode for consistent behavior
            "preset_override": run_config.get("preset_override"),
            "pipeline_id": run_config.get("pipeline_id"),
            "phase": phase_num,
            "pass": run_config.get("pass_number", 1),
            "phase_run": run_config.get("phase_run", 1),
            "phase_runs_total": run_config.get("phase_runs_total", 1),
            # Storage management
            "save_eval_images": run_config.get("save_eval_images", True),
            "replace_eval_images": run_config.get("replace_eval_images", False),
            "save_checkpoints": run_config.get("save_checkpoints", True),
            "replace_checkpoints": run_config.get("replace_checkpoints", False),
            "save_final_splat": run_config.get("save_final_splat", True),
            "save_best_splat": run_config.get("save_best_splat", run_config.get("save_final_splat", True)),
        }

        # Read project ID from config.json
        config_file = project_dir / "config.json"
        with open(config_file, "r") as f:
            project_config = json.load(f)

        project_id = project_config["id"]

        try:
            # Pipeline projects work directly from the user-specified pipeline folder
            # Images are symlinked during project creation, verify they exist
            images_dir = project_dir / "images"
            
            if not images_dir.exists():
                error_msg = f"CRITICAL: Images directory not found at {images_dir}. Symlink may have failed during project creation."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Verify images are accessible (either through symlink or actual files)
            image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")))
            if image_count == 0:
                error_msg = f"CRITICAL: No images found in {images_dir}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"✓ Found {image_count} images in project directory")

            # Run pipeline directly from the user-specified project directory
            # Pass the actual project directory as an override
            params["project_dir_override"] = str(project_dir)

            pipeline.run_full_pipeline(project_id, params)

            final_status = project_status.get_status(project_id)
            final_state = str(final_status.get("status") or "").strip().lower()
            if final_state in {"failed", "stopped"}:
                logger.warning(
                    "Training run %s ended with non-success project status=%s",
                    run_id,
                    final_state,
                )
                return False, None

            # Check for results file in the project run directory
            pipeline_run_dir = project_dir / "runs" / run_id
            results_file = pipeline_run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"

            reward: Optional[float] = None
            if results_file.exists():
                with open(results_file, "r") as f:
                    results = json.load(f)
                reward = results.get("reward")
            else:
                logger.info(
                    "Run %s completed without input_mode_learning_results.json; treating as success.",
                    run_id,
                )

            return True, reward

        except Exception as e:
            logger.error(f"Training execution failed: {e}", exc_info=True)
            return False, None

    def _old_simulate_training_run(self, run_config: dict) -> tuple[bool, Optional[float]]:
        """OLD SIMULATION - REPLACED BY REAL TRAINING.

        Kept for reference only. Remove after verification.
        """
        time.sleep(2)
        success = random.random() > 0.1

        # Baseline runs (phase 1) don't have rewards - they're just reference runs
        # Rewards only apply to AI-driven phases where we compare against baseline
        phase_num = run_config.get("phase_number", 1)
        if phase_num == 1:
            # Baseline phase - no reward calculation
            reward = None
        else:
            # AI learning phases - simulate reward (quality improvement vs baseline)
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
