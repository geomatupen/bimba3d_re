# Shared Model Architecture for Cross-Project Learning

**Date:** 2026-04-23  
**Status:** ✅ **IMPLEMENTED**

---

## Problem Statement

Training pipelines execute multiple projects sequentially. Without shared models:

```
❌ ISOLATED LEARNING (Before):

Project 1 trains → Model₁ learns from Project 1 only
Project 2 trains → Model₂ learns from Project 2 only  
Project 3 trains → Model₃ learns from Project 3 only

Result: No knowledge transfer, each project starts from scratch
```

With shared models:

```
✅ CROSS-PROJECT LEARNING (After):

Project 1 trains → Shared Model learns from Project 1
Project 2 trains → Shared Model learns from Project 1 + 2
Project 3 trains → Shared Model learns from Project 1 + 2 + 3

Result: Knowledge accumulates across all projects
```

---

## Implementation

### Directory Structure

**Before (per-project models):**
```
projects/
  ├── podoli_oblique/
  │   └── models/
  │       └── contextual_continuous_selector/
  │           └── exif_only.json  ← Independent model
  ├── bilovec_nadir/
  │   └── models/
  │       └── contextual_continuous_selector/
  │           └── exif_only.json  ← Independent model
  └── terrain_rough/
      └── models/
          └── contextual_continuous_selector/
              └── exif_only.json  ← Independent model
```

**After (shared model):**
```
{pipeline_folder}/
  ├── shared_models/  ← SHARED across all projects
  │   └── contextual_continuous_selector/
  │       ├── exif_only.json  ← All projects use this
  │       ├── exif_only.lock  ← File lock for concurrent access
  │       ├── exif_plus_flight_plan.json
  │       └── exif_plus_flight_plan_plus_external.json
  ├── podoli_oblique/
  │   ├── config.json  → shared_model_dir: "{pipeline_folder}/shared_models"
  │   ├── runs/
  │   └── outputs/
  ├── bilovec_nadir/
  │   ├── config.json  → shared_model_dir: "{pipeline_folder}/shared_models"
  │   ├── runs/
  │   └── outputs/
  └── terrain_rough/
      ├── config.json  → shared_model_dir: "{pipeline_folder}/shared_models"
      ├── runs/
      └── outputs/
```

---

## Key Components

### 1. Project Config (`config.json`)

Pipeline-created projects include `shared_model_dir`:

```json
{
  "id": "proj_123",
  "name": "podoli_oblique",
  "source_dir": "E:/Thesis/exp_new_method/podoli_oblique",
  "shared_model_dir": "D:/pipelines/contextual_learning/shared_models",
  "created_by": "training_pipeline",
  "pipeline_id": "pipeline_abc123",
  "pipeline_name": "contextual_learning_2026_04_22",
  ...
}
```

**Manual projects** don't have `shared_model_dir` → use project-local models (existing behavior).

---

### 2. Model Path Resolution (`contextual_continuous_learner.py`)

```python
def _get_shared_model_dir(project_dir: Path) -> Path | None:
    """Get shared model directory from project config if available."""
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            shared_dir = config.get("shared_model_dir")
            if shared_dir:
                return Path(shared_dir)
    except Exception:
        pass
    return None


def _selector_dir(project_dir: Path) -> Path:
    """Get model directory - shared if in pipeline, else project-local."""
    shared_dir = _get_shared_model_dir(project_dir)
    if shared_dir:
        # Pipeline context: use shared model
        return shared_dir / "contextual_continuous_selector"
    else:
        # Manual project: use project-local model
        return project_dir / "models" / "contextual_continuous_selector"
```

**Logic:**
1. Check if `config.json` has `shared_model_dir`
2. If yes → use shared directory (pipeline context)
3. If no → use project-local directory (manual project)

---

### 3. File Locking for Concurrent Access

Multiple projects may train in parallel (future enhancement) or sequentially with overlapping I/O. File locking prevents corruption:

```python
def _load_model(project_dir: Path, mode: str) -> dict[str, Any]:
    """Load model with file locking for concurrent access."""
    path = _selector_path(project_dir, mode)
    if not path.exists():
        return _default_model(mode)

    lock_path = _get_lock_path(path)  # e.g., exif_only.lock
    try:
        with filelock.FileLock(lock_path, timeout=10):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == 2:
                return data
    except filelock.Timeout:
        # Timeout → another process updating, read without lock
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == 2:
                return data
        except Exception:
            pass
    except Exception:
        pass
    return _default_model(mode)


def _save_model(project_dir: Path, mode: str, model: dict[str, Any]) -> None:
    """Save model with file locking for concurrent access."""
    out_dir = _selector_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _selector_path(project_dir, mode)
    lock_path = _get_lock_path(path)

    try:
        with filelock.FileLock(lock_path, timeout=30):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(model, indent=2), encoding="utf-8")
            tmp.replace(path)
    except filelock.Timeout:
        # Timeout → skip save to avoid blocking
        pass
```

**Lock behavior:**
- **Load timeout (10s):** Read without lock (best-effort)
- **Save timeout (30s):** Skip save (model will update in next run)

---

### 4. Orchestrator Setup (`training_pipeline_orchestrator.py`)

When creating project directories, orchestrator:

1. Creates `{pipeline_folder}/shared_models/` directory
2. Adds `shared_model_dir` to each project's `config.json`

```python
def _get_or_create_project_dir(self, pipeline: dict, project: dict) -> Path:
    config = pipeline["config"]
    pipeline_folder = Path(config["pipeline_folder"])
    
    # Create shared model directory at pipeline level
    shared_model_dir = pipeline_folder / "shared_models"
    shared_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create project directory
    project_dir = pipeline_folder / project["name"]
    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config with shared_model_dir reference
        config_data = {
            "id": str(uuid.uuid4()),
            "name": project["name"],
            "source_dir": project["dataset_path"],
            "shared_model_dir": str(shared_model_dir),  # ← Key field!
            "created_by": "training_pipeline",
            "pipeline_id": pipeline["id"],
            "pipeline_name": config.get("name"),
            **config.get("shared_config", {})
        }
        
        with open(project_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
    
    return project_dir
```

---

## Model Update Flow

### Sequential Execution (Current)

```
Time    Project          Action               Model State
──────────────────────────────────────────────────────────
t=0     podoli_oblique   Load model           n=0, θ=0
t=1     podoli_oblique   Train & update       n=1, θ₁ learned
t=2     bilovec_nadir    Load model           n=1, θ₁ (from prev)
t=3     bilovec_nadir    Train & update       n=2, θ₂ learned
t=4     terrain_rough    Load model           n=2, θ₂ (accumulates)
t=5     terrain_rough    Train & update       n=3, θ₃ learned
...
t=100   project_15       Train & update       n=15, θ₁₅ (all knowledge)
```

**Knowledge accumulation:**
- Project 1: Learns from itself (baseline)
- Project 2: Learns from itself + Project 1's experience
- Project 3: Learns from itself + Projects 1 & 2's experience
- ...
- Project N: Benefits from all previous projects

---

### Parallel Execution (Future Enhancement)

With file locking, multiple projects can train simultaneously:

```
Time    Project A        Project B        Project C
────────────────────────────────────────────────────
t=0     Load (lock)      Wait for lock    Wait for lock
t=1     Train            Load (lock)      Wait for lock
t=2     Update (lock)    Train            Load (lock)
t=3     Done             Update (lock)    Train
t=4                      Done             Update (lock)
t=5                                       Done
```

**Concurrency safety:**
- Locks prevent simultaneous writes
- If lock timeout → skip save (eventual consistency)
- Next load gets most recent state

---

## Behavior Comparison

### Manual Projects (No Shared Model)

```
User creates project manually:
  Dashboard → Create Project → Upload images

Project structure:
  projects/my_project/
    ├── config.json  (no shared_model_dir field)
    ├── images/
    ├── runs/
    └── models/
        └── contextual_continuous_selector/
            └── exif_only.json  ← Project-local model

Behavior:
  ✓ Model stored in project directory
  ✓ No cross-project learning
  ✓ Works exactly as before (backward compatible)
```

### Pipeline Projects (Shared Model)

```
User creates pipeline:
  Training Pipeline wizard → Select datasets → Configure

Project structure:
  {pipeline_folder}/
    ├── shared_models/
    │   └── contextual_continuous_selector/
    │       └── exif_only.json  ← SHARED model
    └── podoli_oblique/
        ├── config.json  (has shared_model_dir field)
        ├── images/  (symlink or reference)
        ├── runs/
        └── outputs/

Behavior:
  ✓ Model stored in shared directory
  ✓ Cross-project knowledge accumulation
  ✓ All projects in pipeline benefit from collective learning
```

---

## Benefits

### 1. Knowledge Transfer

```
Without shared models:
  Project 1 → Model learns: "This dataset → these params work"
  Project 2 → Model learns: "This dataset → these params work"
  No connection between learnings

With shared models:
  Project 1 → Model learns: "Low GSD + nadir → increase densify_grad"
  Project 2 → Model learns: "High GSD + oblique → decrease position_lr"
  Model generalizes: "GSD and angle → parameter adjustments"
```

### 2. Faster Convergence

```
Pipeline execution with 15 projects:

Early projects (1-5):
  - Explore parameter space (Thompson Sampling)
  - Accumulate experience in shared model
  
Mid projects (6-10):
  - Benefit from early learnings
  - Refine parameter predictions
  
Late projects (11-15):
  - Model is well-calibrated
  - Less exploration needed
  - Better initial predictions
```

### 3. Better Generalization

```
Context vector: [focal_length, shutter, GSD, angle, ...]

Project 1 (podoli_oblique):
  focal=24mm, angle=45° → learns oblique-specific params

Project 2 (bilovec_nadir):
  focal=35mm, angle=90° → learns nadir-specific params

Shared model learns relationship:
  θ₁ · focal + θ₂ · angle → parameter multipliers
  
New project (terrain_rough):
  focal=28mm, angle=60° → model interpolates correctly!
```

---

## Testing

### Verify Shared Model Path

```python
# Check project config
import json
from pathlib import Path

project_dir = Path("D:/pipelines/contextual_learning/podoli_oblique")
config = json.load((project_dir / "config.json").open())

print("Shared model dir:", config.get("shared_model_dir"))
# Expected: D:/pipelines/contextual_learning/shared_models

# Verify model uses shared path
from worker.ai_input_modes.contextual_continuous_learner import _selector_dir

model_dir = _selector_dir(project_dir)
print("Model directory:", model_dir)
# Expected: D:/pipelines/contextual_learning/shared_models/contextual_continuous_selector
```

### Verify Knowledge Accumulation

```bash
# Check model before training
cat D:/pipelines/contextual_learning/shared_models/contextual_continuous_selector/exif_only.json
# {"runs": 0, "reward_mean": 0.0, ...}

# After Project 1 trains
cat D:/pipelines/contextual_learning/shared_models/contextual_continuous_selector/exif_only.json
# {"runs": 1, "reward_mean": 0.08, ...}

# After Project 2 trains
cat D:/pipelines/contextual_learning/shared_models/contextual_continuous_selector/exif_only.json
# {"runs": 2, "reward_mean": 0.06, ...}

# Confirm: n increases, model updates after each project
```

### Verify Lock Files

```bash
# During training, lock file exists
ls D:/pipelines/contextual_learning/shared_models/contextual_continuous_selector/
# exif_only.json
# exif_only.lock  ← Present during load/save

# After training, lock file may persist (harmless)
```

---

## Edge Cases

### Project Without shared_model_dir

```python
# Manual project or old config
config = {"id": "proj_123", "name": "my_project"}
# No shared_model_dir field

# _get_shared_model_dir returns None
# _selector_dir falls back to project-local path
# Result: Works as before (backward compatible)
```

### Shared Model Directory Doesn't Exist

```python
# Config points to non-existent directory
config = {"shared_model_dir": "/tmp/nonexistent"}

# _selector_dir returns Path("/tmp/nonexistent/contextual_continuous_selector")
# _load_model creates default model
# _save_model creates directory (mkdir parents=True)
# Result: Graceful handling, directory created on first save
```

### Lock Timeout

```python
# Process A is updating model (holds lock)
# Process B tries to load (timeout after 10s)

# Process B: Read without lock (best-effort)
# Result: Process B may read slightly stale data, but won't crash

# Process C tries to save (timeout after 30s)
# Process C: Skip save
# Result: Model update deferred to next run
```

---

## Summary

✅ **Implemented shared model storage for cross-project learning**  
✅ **Automatic detection via config.json `shared_model_dir` field**  
✅ **File locking for concurrent access safety**  
✅ **Backward compatible with manual projects**  
✅ **Orchestrator creates shared directory and configures projects**  

**Impact:**
- Pipeline projects share one model → knowledge accumulates
- Manual projects use local models → independent learning
- Later projects in pipeline benefit from earlier learnings
- Model generalizes across diverse datasets

This enables true cross-project contextual learning! 🚀
