# Contextual Continuous Learner - Complete Feature List

## Overview

The contextual continuous learner uses ALL extracted features from the AI input modes. The context vector includes an **intercept term** (constant 1.0) plus all primary features and missing flags.

### What is the Intercept?

The **intercept** (also called bias term) is a constant `1.0` added as the first element of every context vector.

**Purpose:** Allows the model to predict non-zero multiplier values when all features are at their normalized baseline.

**Example:**
```
Without intercept: multiplier = θ₁·focal + θ₂·shutter + ...
  → If all normalized features = 0, prediction forced to 0

With intercept: multiplier = θ₀·(1.0) + θ₁·focal + θ₂·shutter + ...
  → θ₀ acts as baseline, prediction = θ₀ when features are 0
  → Allows neutral features to predict multiplier ≈ 1.0 (no adjustment)
```

---

## Mode 1: EXIF-Only (9 dimensions)

### Structure
```
[intercept, focal_norm, shutter_norm, iso_norm, img_w_norm, img_h_norm,
 focal_missing, shutter_missing, iso_missing]
```

### Features (8 extracted + 1 intercept)

| Index | Feature | Type | Normalization | Range | Source |
|-------|---------|------|---------------|-------|--------|
| 0 | **intercept** | constant | 1.0 | [1.0] | Added by learner |
| 1 | focal_length_mm | primary | (focal - 50.0) / 150.0 | [-0.28, 1.67] | EXIF FocalLength |
| 2 | shutter_s | primary | log10(shutter + 1e-6) / 3.0 | [-2.0, 0] | EXIF ExposureTime |
| 3 | iso | primary | (log10(iso) - 2.0) / 3.0 | [-0.43, 1.68] | EXIF ISOSpeedRatings |
| 4 | img_width_median | primary | (width - 4000.0) / 2000.0 | [-1.68, 2.0] | Image dimensions |
| 5 | img_height_median | primary | (height - 3000.0) / 1500.0 | [-1.68, 2.0] | Image dimensions |
| 6 | focal_missing | flag | 0 or 1 | [0, 1] | Missing focal data |
| 7 | shutter_missing | flag | 0 or 1 | [0, 1] | Missing exposure data |
| 8 | iso_missing | flag | 0 or 1 | [0, 1] | Missing ISO data |

**Total:** 1 intercept + 5 primary + 3 flags = **9 dimensions**

---

## Mode 2: EXIF + Flight Plan (19 dimensions)

### Structure
```
[Mode 1 features (9), gsd_norm, overlap, coverage, angle_norm, heading,
 gsd_missing, overlap_missing, coverage_missing, angle_missing, heading_missing]
```

### Additional Features (10 new features)

| Index | Feature | Type | Normalization | Range | Source |
|-------|---------|------|---------------|-------|--------|
| 9 | gsd_median | primary | gsd / 0.5 | [0.002, 1.0] | GPS altitude + focal |
| 10 | overlap_proxy | primary | 1.0 - (step/footprint) | [0, 1] | GPS point spacing |
| 11 | coverage_spread | primary | bbox_diag / 600.0 | [0, 1] | GPS bounding box |
| 12 | camera_angle_bucket | primary | bucket / 3.0 | [0, 0.33, 0.67, 1.0] | EXIF pitch angle |
| 13 | heading_consistency | primary | 1.0 - (turn_std / 90.0) | [0, 1] | GPS bearing variance |
| 14 | gsd_missing | flag | 0 or 1 | [0, 1] | Missing altitude/focal |
| 15 | overlap_missing | flag | 0 or 1 | [0, 1] | Missing GPS path |
| 16 | coverage_missing | flag | 0 or 1 | [0, 1] | Missing GPS coverage |
| 17 | angle_missing | flag | 0 or 1 | [0, 1] | Missing pitch data |
| 18 | heading_missing | flag | 0 or 1 | [0, 1] | Missing heading data |

**Total:** Mode 1 (9) + 5 primary + 5 flags = **19 dimensions**

---

## Mode 3: EXIF + Flight Plan + External (29 dimensions)

### Structure
```
[Mode 2 features (19), veg_cover, veg_complexity, terrain_rough, texture, blur_risk,
 green_missing, veg_complexity_missing, roughness_missing, texture_missing, blur_missing]
```

### Additional Features (10 new features)

| Index | Feature | Type | Normalization | Range | Source |
|-------|---------|------|---------------|-------|--------|
| 19 | vegetation_cover_percentage | primary | green_pixels / total | [0, 1] | Image RGB analysis |
| 20 | vegetation_complexity_score | primary | 0.55·cover + 0.45·texture | [0, 1] | Derived composite |
| 21 | terrain_roughness_proxy | primary | median(plane_residuals) / 10.0 | [0, 1] | COLMAP points / luma |
| 22 | texture_density | primary | sobel_edges / (W·H·255) | [0, 1] | Image gradient |
| 23 | blur_motion_risk | primary | 1.0 - (laplacian_sum / ...) | [0, 1] | Image sharpness |
| 24 | green_area_missing | flag | 0 or 1 | [0, 1] | Missing images |
| 25 | veg_complexity_missing | flag | 0 or 1 | [0, 1] | Missing images |
| 26 | roughness_missing | flag | 0 or 1 | [0, 1] | Missing COLMAP/images |
| 27 | texture_missing | flag | 0 or 1 | [0, 1] | Missing images |
| 28 | blur_missing | flag | 0 or 1 | [0, 1] | Missing images |

**Total:** Mode 2 (19) + 5 primary + 5 flags = **29 dimensions**

---

## Summary Table

| Mode | Dims | Intercept | EXIF Primary | EXIF Flags | Flight Primary | Flight Flags | External Primary | External Flags |
|------|------|-----------|--------------|------------|----------------|--------------|------------------|----------------|
| **exif_only** | 9 | 1 | 5 | 3 | - | - | - | - |
| **exif_plus_flight_plan** | 19 | 1 | 5 | 3 | 5 | 5 | - | - |
| **exif_plus_flight_plan_plus_external** | 29 | 1 | 5 | 3 | 5 | 5 | 5 | 5 |

---

## Feature Extraction Sources

### Feature Keys by Mode

**exif_only.py (8 keys):**
```python
EXIF_ONLY_FEATURE_KEYS = {
    "focal_length_mm", "focal_missing",
    "shutter_s", "shutter_missing",
    "iso", "iso_missing",
    "img_width_median", "img_height_median",
}
```

**exif_plus_flight_plan.py (+10 keys):**
```python
FLIGHT_PLAN_FEATURE_KEYS = {
    "gsd_median", "gsd_missing",
    "overlap_proxy", "overlap_missing",
    "coverage_spread", "coverage_missing",
    "camera_angle_bucket", "angle_missing",
    "heading_consistency", "heading_missing",
}
```

**exif_plus_flight_plan_plus_external.py (+10 keys):**
```python
EXTERNAL_IMAGE_FEATURE_KEYS = {
    "vegetation_cover_percentage", "green_area_missing",
    "vegetation_complexity_score", "veg_complexity_missing",
    "terrain_roughness_proxy", "roughness_missing",
    "texture_density", "texture_missing",
    "blur_motion_risk", "blur_missing",
}
```

**Total Feature Keys:** 8 + 10 + 10 = 28 extracted features
**Total Context Dimensions:** 1 intercept + 28 features = 29 (for Mode 3)

---

## Implementation Details

### Context Vector Builder

**Location:** [`contextual_continuous_learner.py:90-165`](bimba3d_backend/worker/ai_input_modes/contextual_continuous_learner.py#L90-L165)

**Process:**
1. Start with `x = [1.0]` (intercept)
2. Extract and normalize Mode 1 features (5 primary + 3 flags)
3. If Mode 2: Add flight plan features (5 primary + 5 flags)
4. If Mode 3: Add external features (5 primary + 5 flags)
5. Return numpy array with all features

### Model Persistence

Each mode maintains separate models:
```
project_dir/models/contextual_continuous_selector/
  ├── exif_only.json                            (9×9 matrices per multiplier)
  ├── exif_plus_flight_plan.json                (19×19 matrices per multiplier)
  └── exif_plus_flight_plan_plus_external.json  (29×29 matrices per multiplier)
```

Each model file stores:
- `context_dim`: 9, 19, or 29
- `models`: 8 multipliers, each with A (d×d), b (d), n (int)
- `lambda_ridge`: 2.0 (regularization strength)

---

## Verification

✅ **All features extracted by input modes are used by learner**
✅ **No feature mismatch between extraction and learning**
✅ **All tests passing with updated dimensions (9, 19, 29)**
✅ **Documentation synchronized across all files**

**Status:** Production ready with complete feature utilization.
