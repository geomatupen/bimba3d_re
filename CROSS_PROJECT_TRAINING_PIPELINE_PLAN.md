# Cross-Project Training Pipeline - Implementation Plan

**Purpose:** Automated multi-project contextual learning pipeline with thermal management  
**Date:** 2026-04-22  
**Status:** 📋 Planning Phase (DO NOT CODE YET)

---

## 1. OVERVIEW

### Problem Statement
Currently, you can only train **one project at a time** with warmup/jitter within that project. For contextual continuous learning with **N=15 datasets**, we need:

1. ✅ **Cross-project training** - rotate between projects to learn diverse contexts
2. ✅ **Automated project creation** - batch import from folders
3. ✅ **Consistent configuration** - same settings across all projects
4. ✅ **Thermal management** - cooldown periods between runs
5. ✅ **Training phases** - baseline → exploration → refinement
6. ✅ **Progress tracking** - monitor rewards, runs, status

### Solution
**New "Training Pipeline" page** accessible from Dashboard for automated cross-project training orchestration.

---

## 2. USER WORKFLOW

### 2.1 Entry Point

**Dashboard (Homepage):**
```
┌─────────────────────────────────────────────────────────────┐
│  Bimba3D                                          [Settings] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Create Project]  [Training Pipeline] ← NEW BUTTON          │
│                                                               │
│  Recent Projects:                                            │
│  • Project A                                                 │
│  • Project B                                                 │
│  ...                                                          │
└─────────────────────────────────────────────────────────────┘
```

**Click "Training Pipeline" → Navigate to `/training-pipeline`**

---

### 2.2 Training Pipeline Page Structure

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Training Pipeline                                    [Back to Dashboard] │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─ STEP 1: Dataset Selection ────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  Base Directory: [Browse...] E:/Thesis/exp_new_method              │ │
│  │                                                                      │ │
│  │  Discovered Datasets (15):                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐│ │
│  │  │ [✓] podoli_oblique          Images: 45   Size: 120 MB         ││ │
│  │  │ [✓] bilovec_nadir           Images: 67   Size: 180 MB         ││ │
│  │  │ [✓] terrain_rough_mixed     Images: 52   Size: 145 MB         ││ │
│  │  │ [ ] old_dataset_skip        Images: 12   Size: 30 MB          ││ │
│  │  │ ...                                                             ││ │
│  │  └────────────────────────────────────────────────────────────────┘│ │
│  │                                                                      │ │
│  │  [Select All] [Deselect All] [Scan Directory]                      │ │
│  │                                                                      │ │
│  │  Project Creation:                                                  │ │
│  │  ( ) Create new projects for all selected                          │ │
│  │  (•) Use existing projects if available                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌─ STEP 2: Shared Training Configuration ─────────────────────────────┐ │
│  │                                                                      │ │
│  │  AI Input Mode:                                                     │ │
│  │  (•) exif_plus_flight_plan                                          │ │
│  │  ( ) exif_only                                                      │ │
│  │  ( ) exif_plus_flight_plan_plus_external                           │ │
│  │                                                                      │ │
│  │  Selector Strategy:                                                 │ │
│  │  (•) contextual_continuous                                          │ │
│  │  ( ) continuous_bandit_linear                                       │ │
│  │  ( ) preset_bias                                                    │ │
│  │                                                                      │ │
│  │  Training Parameters:                                               │ │
│  │  Max Steps:        [5000]         Log Interval:      [100]         │ │
│  │  Eval Interval:    [1000]         Densify Until:     [4000]        │ │
│  │  Images Max Size:  [1600]                                           │ │
│  │                                                                      │ │
│  │  [Advanced Settings ▼]                                              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌─ STEP 3: Training Schedule ──────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  Training Phases:                                                   │ │
│  │                                                                      │ │
│  │  Phase 1: Baseline Collection                                       │ │
│  │  ├─ Runs per project: [1]                                           │ │
│  │  ├─ Strategy: balanced preset (baseline)                            │ │
│  │  ├─ Update model: [ ] No (establish reference only)                │ │
│  │  └─ Total runs: 15 (1 × 15 projects)                               │ │
│  │                                                                      │ │
│  │  Phase 2: Initial Exploration (Pass 1)                              │ │
│  │  ├─ Runs per project: [1]                                           │ │
│  │  ├─ Strategy: contextual_continuous                                 │ │
│  │  ├─ Update model: [✓] Yes (start learning)                         │ │
│  │  ├─ Jitter: None (use predicted multipliers)                        │ │
│  │  └─ Total runs: 15 (1 × 15 projects)                               │ │
│  │                                                                      │ │
│  │  Phase 3: Multi-Pass Learning (Pass 2-6)                            │ │
│  │  ├─ Runs per project: [1]                                           │ │
│  │  ├─ Passes: [5]                                                     │ │
│  │  ├─ Strategy: contextual_continuous                                 │ │
│  │  ├─ Update model: [✓] Yes                                           │ │
│  │  ├─ Context jitter: [✓] Mild (±5%)                                  │ │
│  │  ├─ Shuffle order each pass: [✓] Yes                                │ │
│  │  └─ Total runs: 75 (5 × 15 projects)                               │ │
│  │                                                                      │ │
│  │  Grand Total: 105 runs (15 baseline + 15 pass1 + 75 multi-pass)    │ │
│  │                                                                      │ │
│  │  [Customize Phases]                                                 │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌─ STEP 4: Thermal Management ─────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  GPU Thermal Control:                                               │ │
│  │  [✓] Enable cooldown periods between runs                           │ │
│  │                                                                      │ │
│  │  Cooldown Strategy:                                                 │ │
│  │  (•) Fixed interval                                                 │ │
│  │      Wait time: [10] minutes after each run                         │ │
│  │                                                                      │ │
│  │  ( ) Temperature-based (requires GPU monitoring)                    │ │
│  │      Wait until GPU temp < [70]°C                                   │ │
│  │      Check interval: [30] seconds                                   │ │
│  │      Max wait time: [30] minutes (timeout)                          │ │
│  │                                                                      │ │
│  │  ( ) Time-of-day scheduling                                         │ │
│  │      Run only during: [22:00] to [06:00]                            │ │
│  │      Pause outside these hours                                      │ │
│  │                                                                      │ │
│  │  Estimated Total Time:                                              │ │
│  │  • Training time: 105 runs × 8 min ≈ 14 hours                      │ │
│  │  • Cooldown time: 105 × 10 min ≈ 17.5 hours                        │ │
│  │  • Total: ~31.5 hours (~1.3 days)                                   │ │
│  │                                                                      │ │
│  │  [Calculate Estimate]                                               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌─ STEP 5: Review & Launch ────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  Pipeline Summary:                                                  │ │
│  │  • Projects: 15                                                     │ │
│  │  • Total runs: 105                                                  │ │
│  │  • Strategy: contextual_continuous                                  │ │
│  │  • Estimated duration: 31.5 hours                                   │ │
│  │                                                                      │ │
│  │  Pipeline Name: [contextual_learning_batch_2026_04_22]             │ │
│  │                                                                      │ │
│  │  [< Back]  [Save Configuration]  [▶ Start Pipeline]                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Active Pipeline Monitoring

**Once started, show live monitoring dashboard:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Training Pipeline: contextual_learning_batch_2026_04_22                 │
│  Status: RUNNING  Started: 2026-04-22 10:30  Elapsed: 2h 15m            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Overall Progress: [████████░░░░░░░░░░░░░░░░░░] 32/105 runs (30%)       │
│                                                                            │
│  Current Phase: Phase 2 - Initial Exploration (Pass 1)                   │
│  Current Project: bilovec_nadir (Project 12/15)                          │
│  Current Run: train-contextual-bilovec-pass1                             │
│  Status: Training (Step 3200/5000)                                        │
│  ETA: 4 minutes                                                           │
│                                                                            │
│  ┌─ Next Up ────────────────────────────────────────────────────────────┐│
│  │  1. [COOLDOWN] Wait 10 minutes (thermal management)                 ││
│  │  2. Project: terrain_rough_mixed (13/15)                            ││
│  │  3. Project: podoli_oblique (14/15)                                 ││
│  │  4. Project: dataset_xyz (15/15)                                    ││
│  │  5. [PHASE COMPLETE] Start Phase 3 - Multi-Pass Learning            ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│  ┌─ Phase Breakdown ────────────────────────────────────────────────────┐│
│  │  Phase 1 (Baseline):          [████████████████] 15/15  ✓ Complete  ││
│  │  Phase 2 (Exploration Pass1): [████████░░░░░░░░] 12/15  ▶ Running   ││
│  │  Phase 3 (Multi-Pass 2-6):    [░░░░░░░░░░░░░░░░]  0/75  ⏸ Pending  ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│  ┌─ Recent Runs ────────────────────────────────────────────────────────┐│
│  │  Project              Run           Reward    S_Run   Status   Time  ││
│  │  ───────────────────────────────────────────────────────────────────││
│  │  podoli_oblique      pass1-run01    +0.12    0.78    ✓ Done    8m   ││
│  │  bilovec_nadir       pass1-run01    -0.05    0.65    ✓ Done    7m   ││
│  │  terrain_rough       pass1-run01    +0.08    0.72    ✓ Done    9m   ││
│  │  ...                                                                 ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│  ┌─ Learning Progress ──────────────────────────────────────────────────┐│
│  │  Mean Reward (Rolling 15 runs): +0.05                                ││
│  │  Success Rate (Reward > 0):     53% (17/32)                          ││
│  │  Best Reward:                   +0.18 (dataset_xyz, pass1)           ││
│  │  Model Updates:                 17/32 successful                     ││
│  │                                                                       ││
│  │  [View Detailed Analytics]                                           ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│  [⏸ Pause Pipeline]  [⏹ Stop Pipeline]  [View Logs]                     │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. BACKEND ARCHITECTURE

### 3.1 Data Model

**New Database Table: `training_pipelines`**

```sql
CREATE TABLE training_pipelines (
    id TEXT PRIMARY KEY,  -- e.g., "pipeline_uuid_xxxxx"
    name TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'pending', 'running', 'paused', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Configuration
    config JSON NOT NULL,  -- Full pipeline config (see below)
    
    -- Progress tracking
    current_phase INTEGER DEFAULT 1,
    current_pass INTEGER DEFAULT 1,
    current_project_index INTEGER DEFAULT 0,
    total_runs INTEGER,
    completed_runs INTEGER DEFAULT 0,
    failed_runs INTEGER DEFAULT 0,
    
    -- Statistics
    mean_reward REAL,
    success_rate REAL,
    best_reward REAL,
    
    -- Thermal management
    last_run_ended_at TIMESTAMP,
    next_run_scheduled_at TIMESTAMP,
    cooldown_active BOOLEAN DEFAULT FALSE,
    
    -- Error handling
    last_error TEXT,
    retry_count INTEGER DEFAULT 0
);
```

**Training Pipeline Config (JSON):**

```json
{
  "pipeline_id": "pipeline_20260422_103045",
  "name": "contextual_learning_batch_2026_04_22",
  "base_directory": "E:/Thesis/exp_new_method",
  
  "projects": [
    {
      "project_id": "uuid-1",
      "name": "podoli_oblique",
      "dataset_path": "E:/Thesis/exp_new_method/podoli_oblique",
      "baseline_run_id": null,  // Will be set after baseline run
      "image_count": 45,
      "created": false  // Whether project was created by pipeline
    },
    // ... 14 more projects
  ],
  
  "shared_config": {
    "ai_input_mode": "exif_plus_flight_plan",
    "ai_selector_strategy": "contextual_continuous",
    "max_steps": 5000,
    "log_interval": 100,
    "eval_interval": 1000,
    "images_max_size": 1600,
    "densify_until_iter": 4000,
    // ... other shared params from run_config.json
  },
  
  "phases": [
    {
      "phase_number": 1,
      "name": "Baseline Collection",
      "runs_per_project": 1,
      "strategy_override": "preset_bias",
      "preset_override": "balanced",
      "update_model": false,
      "context_jitter": false,
      "session_execution_mode": "test"
    },
    {
      "phase_number": 2,
      "name": "Initial Exploration",
      "runs_per_project": 1,
      "passes": 1,
      "update_model": true,
      "context_jitter": false,
      "shuffle_order": true,
      "session_execution_mode": "train"
    },
    {
      "phase_number": 3,
      "name": "Multi-Pass Learning",
      "runs_per_project": 1,
      "passes": 5,
      "update_model": true,
      "context_jitter": true,
      "context_jitter_percent": 5,
      "shuffle_order": true,
      "session_execution_mode": "train"
    }
  ],
  
  "thermal_management": {
    "enabled": true,
    "strategy": "fixed_interval",  // or "temperature_based", "time_scheduled"
    "cooldown_minutes": 10,
    "gpu_temp_threshold": 70,  // for temperature_based
    "check_interval_seconds": 30,
    "max_wait_minutes": 30,
    "allowed_hours": {  // for time_scheduled
      "start": "22:00",
      "end": "06:00"
    }
  },
  
  "failure_handling": {
    "continue_on_failure": true,
    "max_retries_per_run": 1,
    "skip_project_after_failures": 3
  }
}
```

---

### 3.2 Backend API Endpoints

**New endpoints in `bimba3d_backend/app/api/training_pipeline.py`:**

```python
# Discovery & Setup
POST   /api/training-pipeline/scan-directory
       Body: { "base_directory": "E:/path" }
       Returns: { "datasets": [...], "total": 15 }

POST   /api/training-pipeline/create
       Body: { config (see above) }
       Returns: { "pipeline_id": "...", "status": "pending" }

# Pipeline Control
POST   /api/training-pipeline/{id}/start
       Returns: { "status": "running", "next_run_scheduled_at": "..." }

POST   /api/training-pipeline/{id}/pause
       Returns: { "status": "paused" }

POST   /api/training-pipeline/{id}/resume
       Returns: { "status": "running" }

POST   /api/training-pipeline/{id}/stop
       Returns: { "status": "stopped" }

# Monitoring
GET    /api/training-pipeline/{id}
       Returns: Full pipeline status + progress

GET    /api/training-pipeline/{id}/runs
       Returns: List of all runs with rewards, status

GET    /api/training-pipeline/list
       Returns: All pipelines (recent first)

# Project Management
POST   /api/training-pipeline/batch-create-projects
       Body: { "datasets": [...], "shared_config": {...} }
       Returns: { "created": [...], "existing": [...], "failed": [...] }
```

---

### 3.3 Pipeline Orchestrator

**New file: `bimba3d_backend/worker/training_pipeline/orchestrator.py`**

```python
class TrainingPipelineOrchestrator:
    """
    Coordinates multi-project training pipeline execution.
    
    Responsibilities:
    - Schedule runs across projects
    - Manage thermal cooldown periods
    - Track progress and statistics
    - Handle failures and retries
    - Update contextual models
    """
    
    async def start_pipeline(pipeline_id: str):
        """Start executing pipeline runs."""
        # 1. Load pipeline config
        # 2. Validate all projects exist
        # 3. Enter main execution loop
        
    async def execute_next_run(pipeline: Pipeline):
        """Execute the next scheduled run."""
        # 1. Determine next project/phase/pass
        # 2. Check thermal cooldown
        # 3. Submit run to existing run queue
        # 4. Wait for completion
        # 5. Record results
        # 6. Schedule next run or cooldown
        
    async def thermal_cooldown(pipeline: Pipeline):
        """Handle thermal management between runs."""
        # Fixed interval: sleep for N minutes
        # Temperature-based: poll GPU temp until < threshold
        # Time-scheduled: wait until allowed time window
        
    async def handle_run_completion(pipeline: Pipeline, run_result):
        """Process completed run and update statistics."""
        # 1. Extract reward, s_run, status
        # 2. Update pipeline statistics
        # 3. Log progress
        # 4. Determine next action (next run vs cooldown vs phase complete)
        
    async def shuffle_project_order(pipeline: Pipeline):
        """Shuffle project order for next pass."""
        # Randomize project execution order while tracking progress
```

---

### 3.4 Thermal Management Module

**New file: `bimba3d_backend/worker/training_pipeline/thermal.py`**

```python
class ThermalManager:
    """Handles GPU/CPU thermal management during long training sessions."""
    
    async def wait_cooldown(strategy: str, config: dict):
        """Wait for cooldown period based on strategy."""
        
        if strategy == "fixed_interval":
            # Simple sleep for N minutes
            minutes = config["cooldown_minutes"]
            await asyncio.sleep(minutes * 60)
            
        elif strategy == "temperature_based":
            # Poll GPU temperature until < threshold
            threshold = config["gpu_temp_threshold"]
            check_interval = config["check_interval_seconds"]
            max_wait = config["max_wait_minutes"] * 60
            
            start = time.time()
            while True:
                temp = self.get_gpu_temperature()
                if temp < threshold:
                    break
                if time.time() - start > max_wait:
                    logger.warning("Cooldown timeout, proceeding anyway")
                    break
                await asyncio.sleep(check_interval)
                
        elif strategy == "time_scheduled":
            # Wait until allowed time window
            allowed_start = config["allowed_hours"]["start"]
            allowed_end = config["allowed_hours"]["end"]
            await self.wait_for_time_window(allowed_start, allowed_end)
    
    def get_gpu_temperature(self) -> float:
        """Get current GPU temperature (requires nvidia-smi or similar)."""
        # Use nvidia-smi or pynvml to query GPU temp
        # Return temperature in Celsius
```

---

## 4. EXECUTION FLOW

### 4.1 Pipeline Initialization

```
User clicks "Start Pipeline"
  ↓
Backend: POST /api/training-pipeline/{id}/start
  ↓
Orchestrator.start_pipeline(pipeline_id)
  ↓
Load pipeline config from DB
  ↓
Validate all projects exist
  ↓
Set status = "running"
  ↓
Schedule first run (Phase 1, Project 1, Baseline)
  ↓
Return to user: "Pipeline started"
```

---

### 4.2 Run Execution Loop

```
LOOP until all phases complete:
  ↓
1. Orchestrator.execute_next_run()
  ↓
  Determine next run:
  - Phase N, Pass M, Project P
  - Build run config (merge shared_config + phase overrides)
  - Generate run_name: "train-contextual-{project_name}-phase{N}-pass{M}"
  ↓
2. Submit run to existing API:
   POST /api/projects/{project_id}/run
   Body: { run config with ai_selector_strategy, baseline_session_id, etc. }
  ↓
3. WAIT for run completion
   - Poll: GET /api/projects/{project_id}/status
   - Track: step progress, estimated time
  ↓
4. Run completes → Extract results:
   - Read: runs/{run_id}/outputs/engines/gsplat/input_mode_learning_results.json
   - Get: reward_signal, s_run, yhat_scores
  ↓
5. Orchestrator.handle_run_completion()
   - Update pipeline statistics
   - Log: "Project {name} Phase {N} Pass {M}: Reward = {reward}"
   - Increment: completed_runs counter
  ↓
6. Check if phase/pass complete:
   - If end of pass → shuffle project order
   - If end of phase → log phase summary, move to next phase
   - If all phases done → mark pipeline as "completed"
  ↓
7. Thermal cooldown:
   ThermalManager.wait_cooldown(strategy, config)
   - Fixed: sleep 10 minutes
   - Temp-based: wait until GPU < 70°C
   - Time-scheduled: wait until 22:00
  ↓
8. Schedule next run
  ↓
REPEAT until completed
```

---

### 4.3 Error Handling

```
If run fails:
  ↓
Check retry_count < max_retries_per_run
  ↓
  YES: Retry same run
  NO:  ↓
       Mark run as failed
       ↓
       Check continue_on_failure flag
       ↓
         YES: Skip to next project/run
         NO:  Pause pipeline, notify user
```

---

## 5. FRONTEND COMPONENTS

### 5.1 New React Components

```
src/pages/TrainingPipeline.tsx
  - Main pipeline setup page
  - 5-step wizard UI

src/components/training-pipeline/
  ├── DatasetSelector.tsx           (Step 1: Scan & select datasets)
  ├── SharedConfigForm.tsx          (Step 2: Training parameters)
  ├── PhaseScheduler.tsx            (Step 3: Define phases)
  ├── ThermalSettings.tsx           (Step 4: Cooldown config)
  ├── PipelineSummary.tsx           (Step 5: Review & launch)
  ├── PipelineMonitor.tsx           (Active pipeline dashboard)
  ├── PhaseProgressBar.tsx          (Visual progress per phase)
  ├── RunHistoryTable.tsx           (Recent runs with rewards)
  └── LearningMetricsChart.tsx      (Reward trends over time)
```

---

### 5.2 State Management

**Use React Context or Zustand for pipeline state:**

```typescript
interface PipelineState {
  // Setup
  baseDirectory: string;
  discoveredDatasets: Dataset[];
  selectedDatasets: Dataset[];
  sharedConfig: SharedConfig;
  phases: Phase[];
  thermalConfig: ThermalConfig;
  
  // Runtime
  activePipeline: Pipeline | null;
  currentPhase: number;
  currentPass: number;
  currentProjectIndex: number;
  completedRuns: RunResult[];
  statistics: PipelineStatistics;
  
  // Actions
  scanDirectory: (path: string) => Promise<void>;
  createPipeline: (config: PipelineConfig) => Promise<string>;
  startPipeline: (id: string) => Promise<void>;
  pausePipeline: (id: string) => Promise<void>;
  stopPipeline: (id: string) => Promise<void>;
  fetchPipelineStatus: (id: string) => Promise<void>;
}
```

---

## 6. KEY FEATURES

### 6.1 Project Auto-Creation

**When user scans directory:**

```python
POST /api/training-pipeline/scan-directory
Body: { "base_directory": "E:/Thesis/exp_new_method" }

Backend Logic:
1. List subdirectories in base_directory
2. For each subdirectory:
   - Check if "images/" folder exists
   - Count images (*.jpg, *.png, etc.)
   - Estimate size
   - Check if project already exists in DB (by name or path)
3. Return dataset list with status (new vs existing)

Response:
{
  "datasets": [
    {
      "name": "podoli_oblique",
      "path": "E:/Thesis/exp_new_method/podoli_oblique",
      "image_count": 45,
      "size_mb": 120,
      "project_id": "uuid-1",  // if exists
      "status": "existing"     // or "new"
    },
    ...
  ],
  "total": 15,
  "new_count": 3,
  "existing_count": 12
}
```

**Batch project creation:**

```python
POST /api/training-pipeline/batch-create-projects

Backend Logic:
1. For each "new" dataset:
   - Create project in DB
   - Create directory structure
   - Symlink or copy images
   - Run COLMAP preprocessing
   - Create baseline run (if Phase 1 configured)
2. Track progress (async task)
3. Return results

Response:
{
  "created": ["uuid-1", "uuid-2", "uuid-3"],
  "existing": ["uuid-4", "uuid-5", ...],
  "failed": [],
  "colmap_jobs": ["job-1", "job-2", "job-3"]  // async preprocessing
}
```

---

### 6.2 Thermal Management Implementation

**Fixed Interval (Simplest):**

```python
async def wait_cooldown_fixed(minutes: int):
    logger.info(f"Thermal cooldown: waiting {minutes} minutes...")
    await asyncio.sleep(minutes * 60)
    logger.info("Cooldown complete, resuming pipeline")
```

**Temperature-Based (Advanced):**

```python
async def wait_cooldown_temperature(threshold: float, check_interval: int, max_wait: int):
    logger.info(f"Thermal cooldown: waiting for GPU < {threshold}°C...")
    
    start = time.time()
    while True:
        try:
            # Use nvidia-smi to get GPU temp
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            temp = float(result.stdout.strip())
            logger.info(f"Current GPU temp: {temp}°C")
            
            if temp < threshold:
                logger.info(f"GPU cooled to {temp}°C, resuming pipeline")
                break
                
        except Exception as e:
            logger.warning(f"Could not read GPU temp: {e}, using fixed wait")
            await asyncio.sleep(check_interval)
            
        if time.time() - start > max_wait * 60:
            logger.warning(f"Cooldown timeout after {max_wait} minutes, proceeding")
            break
            
        await asyncio.sleep(check_interval)
```

**Time-Scheduled (For overnight training):**

```python
async def wait_for_time_window(start_hour: str, end_hour: str):
    """Wait until current time is within allowed window."""
    logger.info(f"Waiting for time window {start_hour} - {end_hour}...")
    
    while True:
        now = datetime.now().time()
        start_time = datetime.strptime(start_hour, "%H:%M").time()
        end_time = datetime.strptime(end_hour, "%H:%M").time()
        
        # Handle overnight windows (e.g., 22:00 - 06:00)
        if start_time > end_time:
            in_window = now >= start_time or now < end_time
        else:
            in_window = start_time <= now < end_time
            
        if in_window:
            logger.info("Entered allowed time window, resuming pipeline")
            break
            
        # Check every 5 minutes
        await asyncio.sleep(300)
```

---

### 6.3 Context Jitter (Phase 3)

**Mild jitter on context features (±5%):**

```python
def apply_context_jitter(features: dict, jitter_percent: float = 5.0) -> dict:
    """Add small random noise to context features to increase diversity."""
    jittered = features.copy()
    
    # List of numeric features to jitter
    numeric_keys = [
        "focal_length_mm", "shutter_s", "iso",
        "img_width_median", "img_height_median",
        "gsd_median", "overlap_proxy", "coverage_spread",
        "heading_consistency", "vegetation_cover_percentage",
        "terrain_roughness_proxy", "texture_density", "blur_motion_risk"
    ]
    
    for key in numeric_keys:
        if key in jittered and not jittered.get(f"{key}_missing", 0):
            original = float(jittered[key])
            # Add ±jitter_percent Gaussian noise
            noise_factor = np.random.normal(1.0, jitter_percent / 100.0)
            jittered[key] = original * noise_factor
    
    return jittered
```

**Applied in feature extraction when `context_jitter` enabled in phase config.**

---

## 7. USER EXPERIENCE ENHANCEMENTS

### 7.1 Progress Notifications

**Desktop notifications (if browser supports):**

```javascript
// When phase completes
new Notification("Training Pipeline", {
  body: "Phase 2 completed! Starting Phase 3 - Multi-Pass Learning",
  icon: "/logo.png"
});

// When pipeline completes
new Notification("Training Pipeline", {
  body: "All 105 runs completed! Mean reward: +0.12",
  icon: "/logo.png"
});

// On errors
new Notification("Training Pipeline - Warning", {
  body: "Run failed: podoli_oblique pass3. Retrying...",
  icon: "/warning.png"
});
```

---

### 7.2 Pause/Resume Functionality

**Use cases:**
- User needs to use GPU for other work
- Unexpected system maintenance
- Want to review results before continuing

**Implementation:**
```
Pause button → 
  POST /api/training-pipeline/{id}/pause →
    Set status = "paused" →
    Current run continues to completion →
    No new runs scheduled →
    Cooldown timer paused

Resume button →
  POST /api/training-pipeline/{id}/resume →
    Set status = "running" →
    Resume cooldown if interrupted →
    Schedule next run
```

---

### 7.3 Analytics Dashboard

**After pipeline completion, show comprehensive analytics:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Pipeline Results: contextual_learning_batch_2026_04_22│
│  Status: COMPLETED  Duration: 31h 24m  Total Runs: 105          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Overall Performance:                                            │
│  ├─ Mean Reward:     +0.14 (baseline: 0.0)                      │
│  ├─ Success Rate:    62% (65/105 positive)                      │
│  ├─ Best Reward:     +0.38 (dataset_xyz, pass4)                 │
│  ├─ Worst Reward:    -0.22 (podoli_oblique, pass2)              │
│  └─ Improvement:     +14% over baseline                         │
│                                                                  │
│  Per-Phase Breakdown:                                            │
│  ├─ Phase 1 (Baseline):    15/15 runs, mean = 0.0 (by design)  │
│  ├─ Phase 2 (Pass 1):      15/15 runs, mean = +0.05 (45% pos)  │
│  └─ Phase 3 (Pass 2-6):    75/75 runs, mean = +0.16 (65% pos)  │
│                                                                  │
│  [View Detailed Report] [Export CSV] [Compare with Previous]    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. TESTING STRATEGY

### 8.1 Unit Tests

```python
# test_training_pipeline_orchestrator.py
def test_pipeline_creation():
    config = {...}
    pipeline_id = create_pipeline(config)
    assert pipeline_id is not None
    assert get_pipeline(pipeline_id).status == "pending"

def test_run_scheduling():
    pipeline = load_test_pipeline()
    next_run = orchestrator.get_next_run(pipeline)
    assert next_run.project_id == expected_project_id
    assert next_run.phase == 1

def test_thermal_cooldown_fixed():
    start = time.time()
    await thermal_manager.wait_cooldown("fixed_interval", {"cooldown_minutes": 0.1})
    elapsed = time.time() - start
    assert 5 < elapsed < 8  # ~6 seconds (0.1 min)

def test_project_order_shuffle():
    projects = ["A", "B", "C", "D", "E"]
    shuffled = orchestrator.shuffle_projects(projects)
    assert set(shuffled) == set(projects)  # Same projects
    assert shuffled != projects  # Different order (usually)
```

---

### 8.2 Integration Tests

```python
# test_training_pipeline_integration.py
async def test_mini_pipeline():
    """Run a minimal 3-project, 2-phase pipeline."""
    # Setup
    projects = create_test_projects(count=3)
    config = {
        "projects": projects,
        "phases": [
            {"phase_number": 1, "runs_per_project": 1, "update_model": False},
            {"phase_number": 2, "runs_per_project": 1, "passes": 2, "update_model": True}
        ],
        "thermal_management": {"enabled": True, "strategy": "fixed_interval", "cooldown_minutes": 0.1}
    }
    
    # Execute
    pipeline_id = create_pipeline(config)
    await start_pipeline(pipeline_id)
    
    # Wait for completion
    await wait_for_status(pipeline_id, "completed", timeout=600)
    
    # Verify
    pipeline = get_pipeline(pipeline_id)
    assert pipeline.completed_runs == 9  # 3 projects × (1 baseline + 2 passes)
    assert pipeline.status == "completed"
    assert pipeline.mean_reward is not None
```

---

## 9. DEPLOYMENT CONSIDERATIONS

### 9.1 Resource Requirements

**For 15 datasets × 6 runs each = 90 runs (plus 15 baseline = 105 total):**

- **Disk space:** ~50 GB per project × 15 = 750 GB (conservative)
- **RAM:** 32 GB minimum (peak during gsplat training)
- **GPU:** RTX 3060 or better (8+ GB VRAM)
- **CPU:** 8+ cores (COLMAP preprocessing benefits)
- **Time:** 30-40 hours with 10-minute cooldowns

---

### 9.2 Data Persistence

**Pipeline state must survive:**
- Backend restarts
- Browser refresh
- Network interruptions

**Solution:**
- Store pipeline state in SQLite DB (already used for projects)
- Background worker polls DB for active pipelines
- Resume from last completed run on restart

```python
# On backend startup
async def resume_active_pipelines():
    active = db.query(TrainingPipeline).filter(
        TrainingPipeline.status.in_(["running", "paused"])
    ).all()
    
    for pipeline in active:
        if pipeline.status == "running":
            logger.info(f"Resuming pipeline {pipeline.id}")
            asyncio.create_task(orchestrator.start_pipeline(pipeline.id))
        else:
            logger.info(f"Pipeline {pipeline.id} is paused, waiting for resume")
```

---

### 9.3 Multi-User Considerations

**What if multiple users want to run pipelines?**

- **Option 1 (Simple):** Only allow 1 active pipeline at a time
  - Check: No other pipeline is "running" before starting
  - Queue: If one is running, show "Pipeline already in progress"

- **Option 2 (Advanced):** Allow multiple pipelines with resource limits
  - GPU resource pool (only 1 run at a time uses GPU)
  - Queue system for GPU access
  - Complex scheduling (out of scope for v1)

**Recommendation:** Start with Option 1 (single active pipeline).

---

## 10. MIGRATION & ROLLOUT

### 10.1 Phased Rollout

**Phase 1 (Current Sprint):**
- ✅ Backend: Database schema, API endpoints
- ✅ Backend: Pipeline orchestrator (core logic)
- ✅ Backend: Thermal manager (fixed interval only)
- ✅ Frontend: Training pipeline page (5-step wizard)
- ✅ Frontend: Active pipeline monitor
- ✅ Tests: Unit tests for orchestrator

**Phase 2 (Next Sprint):**
- ✅ Backend: Temperature-based thermal management
- ✅ Backend: Time-scheduled training
- ✅ Frontend: Advanced analytics dashboard
- ✅ Frontend: Pipeline comparison view
- ✅ Tests: Integration tests (mini pipeline)

**Phase 3 (Future):**
- ✅ Multi-pipeline support (queue system)
- ✅ Email/Slack notifications
- ✅ Export reports (PDF, CSV)
- ✅ Pipeline templates (save/load configs)

---

### 10.2 Backward Compatibility

**Existing single-project workflow remains unchanged:**
- ProcessTab still works for individual projects
- Training pipeline is a **separate tool**, not a replacement
- Users can choose: single-project OR multi-project pipeline

---

## 11. OPEN QUESTIONS & DECISIONS NEEDED

### 11.1 Project Creation Strategy

**Q:** Should pipeline auto-create projects, or require manual creation first?

**Options:**
- **A) Auto-create:** Pipeline scans directory, creates all projects, runs COLMAP
  - ✅ Pro: Fully automated
  - ❌ Con: Slow startup (COLMAP for 15 datasets = hours)

- **B) Manual pre-create:** User creates projects first, pipeline uses existing
  - ✅ Pro: Fast pipeline start
  - ❌ Con: Extra manual work

- **C) Hybrid (RECOMMENDED):** Pipeline detects existing projects, creates only new ones
  - ✅ Pro: Flexible, works for both scenarios
  - ✅ Pro: Can start training while COLMAP runs for new projects

**Decision:** Use option C - hybrid approach with async COLMAP.

---

### 11.2 Baseline Run Timing

**Q:** When should baseline runs be created?

**Options:**
- **A) During pipeline setup:** Before starting Phase 1
  - ✅ Pro: Consistent baseline for all
  - ❌ Con: Adds time before training starts

- **B) As part of Phase 1:** First run of each project is baseline
  - ✅ Pro: Integrated into pipeline flow
  - ✅ Pro: Can start immediately (RECOMMENDED)

**Decision:** Use option B - baseline as Phase 1.

---

### 11.3 Model Sharing

**Q:** Should contextual models be shared across all projects in pipeline?

**Answer:** ✅ YES - this is the whole point of contextual learning!

**Implementation:**
- All projects in pipeline use the **same contextual model** stored at:
  - `bimba3d_backend/data/models/contextual_continuous_selector/{mode}.json`
- Each run updates the shared model (if `update_model: true`)
- Cross-project learning happens naturally through shared θ vectors

---

### 11.4 GPU Temperature Monitoring

**Q:** How to get GPU temperature on different systems?

**Options:**
- **nvidia-smi** (NVIDIA GPUs) - most common
- **rocm-smi** (AMD GPUs)
- **pynvml** Python library (NVIDIA)

**Recommendation:** Start with `nvidia-smi` (covers 90% of users), add AMD support later.

---

## 12. SUMMARY & NEXT STEPS

### 12.1 Plan Summary

✅ **New "Training Pipeline" page** accessible from Dashboard  
✅ **5-step wizard:** Dataset selection → Config → Schedule → Thermal → Launch  
✅ **Active monitoring dashboard:** Real-time progress, phase breakdown, learning metrics  
✅ **Cross-project orchestration:** Rotate between 15 datasets, 6 passes = 90 runs  
✅ **Thermal management:** Fixed interval (10 min), temperature-based, time-scheduled  
✅ **Context jitter:** Phase 3 adds ±5% noise to features  
✅ **Baseline integration:** Phase 1 creates baselines automatically  
✅ **Shared models:** All projects use same contextual_continuous model  
✅ **Progress persistence:** Survives restarts, browser refresh  
✅ **Error handling:** Continue on failure, retry logic, skip bad projects  

---

### 12.2 Estimated Development Time

| Component | Estimated Time |
|-----------|---------------|
| **Backend: Database schema & API** | 4 hours |
| **Backend: Pipeline orchestrator** | 8 hours |
| **Backend: Thermal manager** | 3 hours |
| **Frontend: Setup wizard (5 steps)** | 10 hours |
| **Frontend: Active monitor dashboard** | 6 hours |
| **Integration & testing** | 6 hours |
| **Documentation** | 2 hours |
| **TOTAL** | **~39 hours (~5 days)** |

---

### 12.3 Next Steps (Before Coding)

1. ✅ **Review this plan** - confirm all features align with your needs
2. ⏳ **Decide on open questions** (project creation, baseline timing)
3. ⏳ **Approve thermal strategy** (fixed interval vs temperature-based)
4. ⏳ **Confirm UI mockups** (5-step wizard flow)
5. ⏳ **Validate execution order** (baseline → pass1 → multi-pass with shuffle)
6. ✅ **Get approval to start coding**

---

### 12.4 Key Benefits

✅ **Automated training:** Set it and forget it - let it run overnight  
✅ **Cross-dataset learning:** Contextual model learns from diverse scenes  
✅ **Thermal safety:** Prevents GPU overheating during marathon sessions  
✅ **Progress tracking:** See which projects/phases performing best  
✅ **Time savings:** No manual run scheduling, automatic model updates  
✅ **Reproducible:** Save pipeline configs, repeat experiments  

---

**STATUS:** 📋 **PLAN COMPLETE - AWAITING APPROVAL TO CODE**

**Questions?** Please review and provide feedback on:
- UI/UX flow (5-step wizard)
- Thermal management strategy preference
- Project creation approach (auto vs manual vs hybrid)
- Any missing features or considerations
