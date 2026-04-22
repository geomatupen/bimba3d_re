# Contextual Continuous Learner - Implementation Guide

## Overview

The **contextual continuous learner** is a context-aware parameter optimization strategy that uses contextual linear regression to predict optimal training parameter multipliers based on input features (EXIF data, flight plan info, terrain characteristics).

### Key Differences from Previous Strategies

| Feature | `preset_bias` | `continuous_bandit_linear` | `contextual_continuous` |
|---------|---------------|---------------------------|-------------------------|
| **Uses context (x)?** | ❌ No | ❌ No | ✅ Yes |
| **Predicts values?** | ❌ Discrete presets | ✅ Continuous multipliers | ✅ Continuous multipliers |
| **Adapts to input?** | ❌ Global | ❌ Global | ✅ Per-context |
| **Learning algorithm** | Preset bias | Gaussian exploration | Thompson Sampling + Ridge Regression |

---

## Architecture

### 1. Context Vector Building

Context features are normalized and stacked into a vector:

```python
x = [
    1.0,                    # Intercept (bias term)
    focal_norm,             # Normalized focal length
    shutter_norm,           # Log-scaled shutter speed
    iso_norm,               # Log-scaled ISO
    img_w_norm,             # Normalized image width
    img_h_norm,             # Normalized image height
    focal_missing,          # Binary flag
    shutter_missing,        # Binary flag
    iso_missing,            # Binary flag
    # ... Mode 2 & 3 features if enabled
]
```

**Mode-specific dimensions:**
- Mode 1 (exif_only): 9 features (1 intercept + 5 primary + 3 missing flags)
- Mode 2 (exif_plus_flight_plan): 19 features (Mode 1 + 5 flight primary + 5 flight missing flags)
- Mode 3 (exif_plus_flight_plan_plus_external): 29 features (Mode 2 + 5 external primary + 5 external missing flags)

**What is the intercept?**
The first element (1.0) is the intercept/bias term, allowing the model to predict non-zero multipliers when all features are at their normalized baseline values.

### 2. Linear Model Per Multiplier

Each of the 8 parameter multipliers has an independent linear model:

```
multiplier_value = θ^T · x
```

Where:
- `θ` = learned weight vector (d dimensions)
- `x` = context vector (d dimensions)

The model maintains:
- `A` = d×d matrix for ridge regression (sum of outer products)
- `b` = d vector for weighted targets (sum of reward-weighted contexts)
- `n` = number of observations

### 3. Thompson Sampling for Exploration

During selection:
1. Compute posterior mean: `θ_mean = A^{-1} · b`
2. Compute posterior covariance: `Σ = σ² · A^{-1}` where `σ² = 1/(n+1)`
3. Sample: `θ_sample ~ N(θ_mean, Σ)`
4. Predict: `multiplier = x^T · θ_sample`
5. Clamp to safe bounds

**Exploration automatically decreases** as `n` increases (variance shrinks).

### 4. Ridge Regression Update

After observing reward `r`:

```python
A += x ⊗ x  # Outer product
b += r · x
n += 1
```

Ridge regularization (λ=2.0) prevents overfitting with small datasets.

---

## Usage

### Configuration

Set the selector strategy in your parameters:

```python
params = {
    "ai_input_mode": "exif_plus_flight_plan",  # or exif_only, exif_plus_flight_plan_plus_external
    "ai_selector_strategy": "contextual_continuous",  # NEW strategy
    # ... other parameters
}
```

### Selection Flow

```python
from bimba3d_backend.worker.ai_input_modes.resolver import apply_initial_preset

result = apply_initial_preset(
    params,
    image_dir=Path("/path/to/images"),
    colmap_dir=Path("/path/to/colmap"),
    logger=logger,
)

# Result contains:
# - selected_preset: "contextual_continuous"
# - yhat_scores: {key: multiplier_value} for all 8 multipliers
# - updates: {key: final_parameter_value} after applying multipliers
# - context_vector: The normalized feature vector used
# - features: Raw extracted features
```

### Learning Flow

After a training run completes:

```python
from bimba3d_backend.worker.ai_input_modes.contextual_continuous_learner import (
    update_from_run_contextual_continuous
)

result = update_from_run_contextual_continuous(
    project_dir=project_dir,
    mode="exif_plus_flight_plan",
    selected_preset="contextual_continuous",
    yhat_scores=yhat_scores,  # From selection
    eval_history=eval_history,  # Training metrics
    baseline_eval_history=baseline_eval_history,  # Baseline comparison
    loss_by_step=loss_by_step,
    elapsed_by_step=elapsed_by_step,
    x_features=x_features,  # Raw features
    run_id=run_id,
    logger=logger,
    apply_update=True,
)

# Model is persisted to:
# project_dir/models/contextual_continuous_selector/{mode}.json
```

### Penalty Recording

For failed runs:

```python
from bimba3d_backend.worker.ai_input_modes.contextual_continuous_learner import (
    record_run_penalty_contextual_continuous
)

result = record_run_penalty_contextual_continuous(
    project_dir=project_dir,
    mode="exif_plus_flight_plan",
    selected_preset="contextual_continuous",
    yhat_scores=yhat_scores,
    penalty_reward=-1.5,
    x_features=x_features,
    reason="gaussian_hard_cap_reached",
    run_id=run_id,
    logger=logger,
)
```

---

## Training Protocol for N=15 Datasets

### Phase 1: Baseline Collection (Runs 1-30)

```bash
# Round 1: All datasets with balanced preset (baseline)
for dataset_id in {1..15}; do
    run_training \
        --dataset-id=$dataset_id \
        --seed=$((dataset_id * 1000)) \
        --strategy=preset_bias \
        --preset-override=balanced \
        --no-update  # Don't update model
done

# Round 2: All datasets with contextual_continuous (initial exploration)
for dataset_id in {1..15}; do
    run_training \
        --dataset-id=$dataset_id \
        --seed=$((dataset_id * 1000 + 1)) \
        --strategy=contextual_continuous \
        --baseline-run=balanced_${dataset_id} \
        --update  # Start learning
done
```

### Phase 2: Multi-Pass Learning (Runs 31-90)

Run each dataset 4 more times with shuffled order:

```python
for pass_num in range(3, 7):  # Passes 3-6
    dataset_order = list(range(1, 16))
    random.shuffle(dataset_order)
    
    for dataset_id in dataset_order:
        run_training(
            dataset_id=dataset_id,
            seed=dataset_id * 1000 + pass_num,
            strategy="contextual_continuous",
            baseline_run_id=f"balanced_{dataset_id}",
            update=True,
        )
```

### Expected Learning Curve

| Runs | Exploration Rate | % Better Than Baseline | Mean Reward |
|------|------------------|------------------------|-------------|
| 1-15 | High (~30%) | 50% (random) | ±0.05 |
| 16-30 | Medium (~20%) | 40% | +0.03 to +0.08 |
| 31-45 | Medium (~15%) | 55% | +0.06 to +0.12 |
| 46-60 | Low (~10%) | 60% | +0.08 to +0.15 |
| 61-90 | Low (~5%) | 65% | +0.10 to +0.18 |

---

## Model Persistence

Models are saved per mode:

```
project_dir/
  models/
    contextual_continuous_selector/
      exif_only.json
      exif_plus_flight_plan.json
      exif_plus_flight_plan_plus_external.json
```

### Model File Format

```json
{
  "version": 2,
  "mode": "exif_plus_flight_plan",
  "context_dim": 19,
  "lambda_ridge": 2.0,
  "runs": 42,
  "reward_mean": 0.08,
  "models": {
    "feature_lr_mult": {
      "A": [[...], ...],  // 19×19 matrix
      "b": [...],          // 19 vector
      "n": 42
    },
    // ... 7 more multipliers
  },
  "last": {
    "run_id": "run_20260422_142530",
    "selected_preset": "contextual_continuous",
    "reward_signal": 0.12,
    "context_vector": [1.0, -0.173, ...],  // 19 elements
    "theta_norms": {
      "feature_lr_mult": 0.234,
      // ...
    }
  }
}
```

---

## Observability

### Key Logs

1. **Selection Log:**
```
CONTEXTUAL_CONTINUOUS_SELECT mode=exif_plus_flight_plan 
  context_norm=1.234 
  multipliers={"feature_lr_mult": 1.05, ...}
```

2. **Update Log:**
```
CONTEXTUAL_CONTINUOUS_UPDATE mode=exif_plus_flight_plan 
  s_best=0.75 s_end=0.78 s_run=0.78 
  reward=0.12 context_norm=1.234
```

3. **Theta Norms (Model Drift):**
```
CONTEXTUAL_CONTINUOUS_THETA_NORMS mode=exif_plus_flight_plan 
  theta_norms={"feature_lr_mult": "0.2340", ...}
```

### Metrics to Monitor

- **Context norm:** Detects outlier contexts (should be < 5.0)
- **Reward signal:** Should converge to +0.10 to +0.20 range
- **Theta norms:** Should stabilize (not grow unbounded)
- **Exploration variance:** Should decrease over time

---

## Troubleshooting

### Issue: Multipliers always near 1.0 after 50 runs

**Cause:** Weak reward signal or near-zero rewards.

**Fix:**
- Verify baseline comparison is enabled (not None)
- Check that `baseline_eval_history` has data
- Increase reward magnitude if signals are too small

### Issue: Model file not created

**Cause:** Model only persists after `update_from_run`, not during selection.

**Fix:** This is expected. Model is created lazily on first update.

### Issue: Predictions vary wildly with similar contexts

**Cause:** Context features have different scales or contain NaNs.

**Fix:**
- Check context normalization in `build_context_vector`
- Verify all features are finite: `assert np.all(np.isfinite(x))`

### Issue: Poor generalization to new datasets

**Cause:** Overfitting to training set (N=15 too small).

**Fix:**
- Increase `lambda_ridge` from 2.0 to 3.0 or 5.0
- Collect more datasets (target: 25-30)
- Add mild context jitter (±5%) during training

---

## Rollback Plan

To revert to previous behavior:

```python
params["ai_selector_strategy"] = "continuous_bandit_linear"  # Context-free
# OR
params["ai_selector_strategy"] = "preset_bias"  # Discrete presets
```

Models are isolated by strategy, so reverting doesn't affect existing models.

---

## Future Enhancements

### Considered for v2:

1. **Neural Contextual Bandit:** Replace linear model with small NN for nonlinear patterns
2. **Multi-Task Learning:** Share θ vectors across similar multipliers
3. **Context Augmentation:** Add synthetic jitter during training
4. **Meta-Learning:** Transfer knowledge across projects
5. **Adaptive Ridge Parameter:** Tune λ based on cross-validation

---

## References

- **Algorithm:** Contextual Linear Regression with Thompson Sampling
- **Regularization:** Ridge regression (L2, λ=2.0)
- **Exploration:** Posterior sampling (variance = 1/(n+1))
- **Safe bounds:** All multipliers clamped to prevent extreme values

---

## Testing

Run tests:

```bash
# Unit tests
pytest bimba3d_backend/tests/test_contextual_continuous_learner.py -v

# Integration tests
pytest bimba3d_backend/tests/test_contextual_continuous_integration.py -v
```

All tests should pass before deployment.
