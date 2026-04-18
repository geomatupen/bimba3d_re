from pydantic import BaseModel
from typing import Optional


class ProjectResponse(BaseModel):
    project_id: str
    name: Optional[str] = None
    created_at: Optional[str] = None


class ProjectListItem(BaseModel):
    project_id: str
    name: Optional[str] = None
    status: str
    progress: int
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    has_outputs: bool = False
    session_count: int = 0


class CreateProjectRequest(BaseModel):
    name: Optional[str] = None
    storage_root_id: Optional[str] = None
    storage_path: Optional[str] = None


class StorageRootResponse(BaseModel):
    id: str
    path: str
    label: str
    is_default: bool = False
    writable: bool = False


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None


class LastTuning(BaseModel):
    step: int
    action: str
    reason: str


class StatusResponse(BaseModel):
    project_id: str
    status: str  # "pending", "processing", "done", "failed"
    progress: int  # 0-100
    error: Optional[str] = None
    name: Optional[str] = None
    created_at: Optional[str] = None
    # NEW: Training mode and adaptive tuning info
    mode: Optional[str] = None  # "baseline" or "modified"
    tuning_active: Optional[bool] = None
    currentStep: Optional[int] = None
    maxSteps: Optional[int] = None
    current_loss: Optional[float] = None
    last_tuning: Optional[LastTuning] = None
    stop_requested: Optional[bool] = None
    stage: Optional[str] = None
    stage_progress: Optional[int] = None  # 0-100 progress for current stage
    message: Optional[str] = None
    device: Optional[str] = None
    can_resume: Optional[bool] = None  # Whether project has checkpoints/outputs to resume
    last_completed_step: Optional[int] = None
    engine: Optional[str] = None
    worker_mode: Optional[str] = None
    current_run_id: Optional[str] = None
    base_session_id: Optional[str] = None
    batch_total: Optional[int] = None
    batch_completed: Optional[int] = None
    batch_current_index: Optional[int] = None


class ProcessParams(BaseModel):
    # Training mode
    mode: Optional[str] = None  # "baseline" or "modified"
    # Worker runtime selection: "docker" (default) or "local"
    worker_mode: Optional[str] = None
    # Training engine selection: "gsplat" (default) or "litegs"
    engine: Optional[str] = None
    # Pipeline stage: "full" (default), "colmap_only", "train_only"
    stage: Optional[str] = None
    # Optional user-provided run session name (used as run folder id after sanitization)
    run_name: Optional[str] = None
    # Resume from last checkpoint if available
    resume: Optional[bool] = None
    # Force restart from scratch for the selected session (clear generated artifacts first)
    restart_fresh: Optional[bool] = None
    # Optional batch orchestration settings
    run_count: Optional[int] = None
    run_name_prefix: Optional[str] = None
    run_jitter_factor: Optional[float] = None
    run_jitter_mode: Optional[str] = None  # "fixed" | "random"
    run_jitter_min: Optional[float] = None
    run_jitter_max: Optional[float] = None
    run_jitter_multiplier: Optional[float] = None
    warmup_at_start: Optional[bool] = None
    continue_on_failure: Optional[bool] = None
    batch_connect_runs: Optional[bool] = None
    # Internal batch chain metadata (persisted in run configs)
    batch_plan_id: Optional[str] = None
    batch_index: Optional[int] = None
    batch_total: Optional[int] = None
    batch_continue_on_failure: Optional[bool] = None
    batch_run_name_prefix: Optional[str] = None
    # Core AI model reuse controls
    start_model_mode: Optional[str] = None  # "scratch" | "reuse"
    project_model_name: Optional[str] = None
    source_model_id: Optional[str] = None
    source_run_id: Optional[str] = None
    ai_input_mode: Optional[str] = None  # "exif_only" | "exif_plus_flight_plan" | "exif_plus_flight_plan_plus_external"
    baseline_session_id: Optional[str] = None  # completed baseline gsplat session for comparison
    ai_preset_override: Optional[str] = None  # optional forced preset for AI input mode selection
    ai_selector_strategy: Optional[str] = None  # "preset_bias" | "continuous_bandit_linear"
    # --- ORIGINAL KERBL PARAMETERS ---
    max_steps: Optional[int] = None  # [original]
    log_interval: Optional[int] = None  # [custom]
    batch_size: Optional[int] = None  # [original]
    eval_interval: Optional[int] = None  # [original]
    save_interval: Optional[int] = None  # [original]
    densify_from_iter: Optional[int] = None  # [original]
    densify_until_iter: Optional[int] = None  # [original]
    densification_interval: Optional[int] = None  # [original]
    densify_grad_threshold: Optional[float] = None  # [original]
    opacity_threshold: Optional[float] = None  # [original]
    lambda_dssim: Optional[float] = None  # [original]
    feature_lr: Optional[float] = None  # [original]
    opacity_lr: Optional[float] = None  # [original]
    scaling_lr: Optional[float] = None  # [original]
    rotation_lr: Optional[float] = None  # [original]
    percent_dense: Optional[float] = None  # [original]
    splat_export_interval: Optional[int] = None  # [original]
    best_splat_interval: Optional[int] = None  # [custom]
    best_splat_start_step: Optional[int] = None  # [custom]
    save_best_splat: Optional[bool] = None  # [custom]
    png_export_interval: Optional[int] = None  # [original]
    # --- CUSTOM PARAMETERS ---
    auto_early_stop: Optional[bool] = None  # [custom]
    early_stop_monitor_interval: Optional[int] = None  # [custom]
    early_stop_decision_points: Optional[int] = None  # [custom]
    early_stop_min_eval_points: Optional[int] = None  # [custom]
    early_stop_min_step_ratio: Optional[float] = None  # [custom]
    early_stop_monitor_min_relative_improvement: Optional[float] = None  # [custom]
    early_stop_eval_min_relative_improvement: Optional[float] = None  # [custom]
    early_stop_max_volatility_ratio: Optional[float] = None  # [custom]
    early_stop_ema_alpha: Optional[float] = None  # [custom]
    colmap: Optional[dict] = None  # [custom]
    gsplat_max_gaussians: Optional[int] = None  # [custom]
    amp: Optional[bool] = None  # [custom]
    pruning_enabled: Optional[bool] = None  # [custom]
    pruning_policy: Optional[str] = None  # [custom]
    pruning_weights: Optional[dict] = None  # [custom]
    tune_start_step: Optional[int] = None  # [custom]
    tune_min_improvement: Optional[float] = None  # [custom]
    tune_end_step: Optional[int] = None  # [custom]
    tune_interval: Optional[int] = None  # [custom]
    tune_scope: Optional[str] = None  # [custom] "core_individual" | "core_only" | "core_ai_optimization" | "core_individual_plus_strategy"
    trend_scope: Optional[str] = None  # [custom] "run" | "phase" (core_ai_optimization)
    session_execution_mode: Optional[str] = None  # [custom] "train" | "test"
    images_max_size: Optional[int] = None  # [custom]
    litegs_target_primitives: Optional[int] = None  # [custom]
    litegs_alpha_shrink: Optional[float] = None  # [custom]
    sparse_preference: Optional[str] = None  # [custom]
    sparse_merge_selection: Optional[list[str]] = None  # [custom]


class EvaluationMetrics(BaseModel):
    lpips_score: Optional[float] = None
    sharpness: Optional[float] = None
    convergence_speed: Optional[float] = None
    final_loss: Optional[float] = None
    gaussian_count: Optional[int] = None


class ComparisonRequest(BaseModel):
    name: Optional[str] = None
    max_steps: Optional[int] = 300
    batch_size: Optional[int] = 1


class ComparisonStatus(BaseModel):
    status: str
    baseline: Optional[dict] = None
    optimized: Optional[dict] = None
    baseline_project_id: Optional[str] = None
    optimized_project_id: Optional[str] = None


class SparseEditRequest(BaseModel):
    candidate: Optional[str] = None
    remove_point_ids: list[int]
    create_backup: Optional[bool] = True
    reoptimize: Optional[bool] = False


class SparseMergeRequest(BaseModel):
    selections: list[str]


class RenameRunRequest(BaseModel):
    run_name: str


class CreateRunRequest(BaseModel):
    run_name: Optional[str] = None
    resolved_params: Optional[dict] = None


class UpdateRunConfigRequest(BaseModel):
    requested_params: Optional[dict] = None
    resolved_params: Optional[dict] = None


class UpdateSharedConfigRequest(BaseModel):
    run_id: Optional[str] = None
    shared: dict


class ElevateModelRequest(BaseModel):
    model_name: Optional[str] = None


class RenameModelRequest(BaseModel):
    model_name: str


class ContinueBatchRequest(BaseModel):
    restart_current: Optional[bool] = True
