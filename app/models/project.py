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
    has_outputs: bool = False


class CreateProjectRequest(BaseModel):
    name: Optional[str] = None


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
    last_tuning: Optional[LastTuning] = None
    stop_requested: Optional[bool] = None
    stage: Optional[str] = None
    stage_progress: Optional[int] = None  # 0-100 progress for current stage
    message: Optional[str] = None
    device: Optional[str] = None
    can_resume: Optional[bool] = None  # Whether project has checkpoints/outputs to resume
    last_completed_step: Optional[int] = None
    engine: Optional[str] = None


class ProcessParams(BaseModel):
    # Training mode
    mode: Optional[str] = None  # "baseline" or "modified"
    # Training engine selection: "gsplat" (default) or "litegs"
    engine: Optional[str] = None
    # Pipeline stage: "full" (default), "colmap_only", "train_only"
    stage: Optional[str] = None
    # Resume from last checkpoint if available
    resume: Optional[bool] = None
    # Training parameters (gsplat)
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    eval_interval: Optional[int] = None
    save_interval: Optional[int] = None
    densify_from_iter: Optional[int] = None
    densify_until_iter: Optional[int] = None
    densification_interval: Optional[int] = None
    opacity_threshold: Optional[float] = None
    lambda_dssim: Optional[float] = None
    # Live export controls
    splat_export_interval: Optional[int] = None
    png_export_interval: Optional[int] = None
    auto_early_stop: Optional[bool] = None
    # COLMAP tuning options (passed through to COLMAP step)
    colmap: Optional[dict] = None
    # Trainer init limit: maximum number of Gaussians to initialize from COLMAP points
    gsplat_max_gaussians: Optional[int] = None
    # Hard cap enforcement and advanced trainer controls
    gsplat_hard_cap: Optional[int] = None
    amp: Optional[bool] = None
    pruning_enabled: Optional[bool] = None
    pruning_policy: Optional[str] = None
    pruning_weights: Optional[dict] = None
    images_max_size: Optional[int] = None
    # LiteGS-specific knobs
    litegs_target_primitives: Optional[int] = None
    litegs_alpha_shrink: Optional[float] = None
    # Sparse selection preference ("best" or specific folder name like "0")
    sparse_preference: Optional[str] = None


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
