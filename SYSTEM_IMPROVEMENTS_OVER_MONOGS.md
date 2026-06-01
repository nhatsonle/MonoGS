# System Improvements Over Original MonoGS

This document summarizes the current changes in this repository relative to the
original MonoGS baseline.

## 1. Headless Evaluation Setup

The TUM monocular `fr3_office` config is configured for container/headless runs:

- GUI is disabled with `Results.use_gui: False`.
- Rendering evaluation is enabled with `Results.eval_rendering: True`.
- Matplotlib uses the non-interactive `Agg` backend during evaluation.
- The evo ATE plot path was patched to avoid the matplotlib colorbar error seen
  with the installed evo/matplotlib versions.

Impact:

- The system can run in a container without an X display.
- ATE plots and rendering metrics can be generated without GUI support.

Relevant files:

- `configs/mono/tum/fr3_office.yaml`
- `utils/eval_utils.py`

## 2. Limited-Frame Dataset Runs

TUM dataset loading now supports a config-level frame limit:

```yaml
Dataset:
  num_frames: 200
```

For TUM sequences, the loader slices RGB paths, depth paths, and poses to the
first `num_frames` frames. If `num_frames <= 0` or is omitted, the full sequence
is used.

Impact:

- Faster controlled experiments.
- Fairer ablations by keeping the same frame count across runs.

Relevant files:

- `utils/dataset.py`
- `configs/mono/tum/fr3_office.yaml`

## 3. Saved Render Visualizations

Rendering evaluation now saves image outputs under `viz/` in addition to metric
JSON files.

Output structure:

```text
results/.../<timestamp>/viz/before_opt/pred/
results/.../<timestamp>/viz/before_opt/gt/
results/.../<timestamp>/viz/before_opt/compare/
results/.../<timestamp>/viz/after_opt/pred/
results/.../<timestamp>/viz/after_opt/gt/
results/.../<timestamp>/viz/after_opt/compare/
```

`compare` images concatenate ground truth and rendered output horizontally.

Impact:

- Easier qualitative inspection of render degradation or improvement.
- Useful for diagnosing PSNR/SSIM/LPIPS changes.

Relevant file:

- `utils/eval_utils.py`

## 4. DUSt3R Integration

DUSt3R and its CroCo dependency were migrated into this repository:

- `dust3r/`
- `croco/`
- `utils/dust3r_utils.py`

The DUSt3R checkpoint is expected at:

```text
checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

The current workspace uses a symlink to the checkpoint from `opengs_SLAM` to
avoid duplicating the 2.2 GB file.

`slam.py` loads DUSt3R only when:

```yaml
Training:
  dust3r:
    enabled: True
```

Impact:

- DUSt3R can be switched on/off from config.
- When disabled, no model is loaded and no DUSt3R inference is performed.

Relevant files:

- `slam.py`
- `utils/dust3r_utils.py`
- `configs/mono/tum/fr3_office.yaml`

## 5. DUSt3R Pointmap-Based Gaussian Initialization

The Gaussian model now supports creating Gaussians directly from DUSt3R
pointmaps.

Pipeline:

1. Frontend detects a new keyframe.
2. A reference keyframe is selected using baseline constraints.
3. DUSt3R runs on the current/reference keyframe pair.
4. DUSt3R pointmap, image colors, confidence masks, scale divisor, and reference
   frame metadata are sent to the backend.
5. Backend transforms the pointmap into the map coordinate frame.
6. GaussianModel converts the filtered point cloud into Gaussian parameters.

Generated Gaussian attributes:

- position from DUSt3R pointmap
- color from DUSt3R-resized RGB image
- SH DC feature from RGB
- scale from nearest-neighbor distance
- identity rotation
- opacity initialized to 0.5

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 6. DUSt3R Pair-Based Map Initialization

DUSt3R can optionally be used before the normal monocular map initialization.
This mode stages one frame as an anchor, waits until a later frame satisfies the
configured baseline/search policy, and then initializes the map from the
DUSt3R pair result.

Config:

```yaml
Training:
  dust3r:
    enabled: True
    init:
      enabled: True
      max_search_frames: 10
      min_baseline: 0.50
      max_baseline: 20.0
      fallback_to_depth: True
      only: False
      prior_only: False
      depth_prior_weight: 0.05
      depth_prior_opacity_threshold: 0.2
      backproject_depth: True
      pcd_downsample: 4
      point_size_scale: 0.25
      sample_stride: 2
      gradient_extra_samples: True
      gradient_threshold: 0.08
      gradient_stride: 2
      max_points: 200000
      use_pixel_footprint_scale: True
      pixel_footprint_scale: 0.75
      use_confidence_mask: True
      fill_invalid_depth: True
      invalid_depth: 2.0
      invalid_depth_noise: 0.05
```

Initialization policy:

- The first suitable frame is staged as `dust3r_init_anchor_idx`.
- Later frames are tested against `min_baseline`, `max_baseline`, and
  `max_search_frames`.
- If DUSt3R succeeds, the backend receives an initialization payload with
  pointmaps, depth maps, masks, pose metadata, and scale divisors.
- If `prior_only=True`, DUSt3R is used as a depth regularization prior while the
  system still creates the initial Gaussians from the usual depth path.
- If `fallback_to_depth=True`, the system falls back to standard monocular depth
  initialization when DUSt3R initialization is unavailable.
- If `fallback_to_depth=False`, initialization waits instead of silently falling
  back.
- If `only=True`, DUSt3R is used only for the initialization pair and skipped
  for later keyframes after the pair initialization has succeeded.

Backprojection mode:

When `backproject_depth=True`, the backend initializes Gaussians by resizing the
DUSt3R depth map to the camera resolution and backprojecting it through the
current camera intrinsics/pose. This mode can:

- filter by DUSt3R confidence masks;
- fill invalid DUSt3R depths with a configurable fallback depth/noise;
- sample a regular grid using `sample_stride`;
- add extra samples on high-image-gradient pixels;
- cap the number of initialized points with `max_points`;
- initialize Gaussian scale from pixel footprint instead of nearest-neighbor
  distance.

Impact:

- Enables monocular map initialization from a non-degenerate DUSt3R image pair
  instead of only using the synthetic/random initial monocular depth.
- Provides a controlled fallback path for cases where DUSt3R fails or the
  baseline is not ready.
- Gives Waymo/KITTI-style outdoor sequences a denser, configurable initialization
  path based on DUSt3R depth backprojection.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`
- `configs/mono/waymo/base_config.yaml`
- `configs/mono/waymo/*_dust3r_init.yaml`

## 7. Safer DUSt3R Scheduling

The first implementation inserted DUSt3R pointmaps too aggressively and degraded
render quality. The current policy is more conservative.

Current safeguards:

- DUSt3R is not used for the first map initialization frame.
- DUSt3R can be delayed until SLAM is initialized:

```yaml
require_initialized: True
```

- Reference keyframe selection uses a baseline range:

```yaml
min_baseline: 0.10
max_baseline: 1.50
```

- If no reference keyframe satisfies the baseline range, DUSt3R is skipped for
  that keyframe.
- DUSt3R pointmaps can be rescaled from the ratio between DUSt3R pair
  translation and the current MonoGS keyframe baseline:

```yaml
scale:
  baseline_ratio: True
  pointmap_sync: True
scale_min: 0.05
scale_max: 20.0
```

`baseline_ratio` can be ablated independently from `pointmap_sync`. With
`pointmap_sync: False`, the system uses the baseline-ratio divisor directly for
all pointmaps in the pair. With both flags disabled, DUSt3R pointmaps are
inserted without this metric scale normalization.

Impact:

- Avoids degenerate `DUSt3R(img, img)` initialization.
- Avoids very small-baseline pairs where DUSt3R scale is unstable.
- Reduces risk of corrupting the early monocular map.
- Makes DUSt3R pointmap scale more consistent with the current MonoGS map scale
  before Gaussian insertion.

Relevant file:

- `utils/slam_frontend.py`

## 8. DUSt3R Pointmap Filtering

Before converting DUSt3R pointmaps into Gaussians, points are filtered by:

- finite XYZ
- positive depth
- configured depth range
- configured point radius
- DUSt3R confidence mask
- optional Open3D statistical outlier removal

Config:

```yaml
dust3r:
  depth_min: 0.05
  depth_max: 8.0
  max_point_radius: 10.0
  outlier_filter:
    enabled: True
    nb_neighbors: 20
    std_ratio: 2.0
```

Impact:

- Removes obvious outliers and invalid points before Gaussian insertion.
- Reduces floaters and noisy geometry from DUSt3R.

Relevant file:

- `gaussian_splatting/scene/gaussian_model.py`

## 9. DUSt3R-to-MonoGS Sim3 Alignment Gate

The backend can now align DUSt3R pointmaps to the current MonoGS map before
inserting Gaussians.

Pipeline:

1. Render the current Gaussian map from the new keyframe pose.
2. Keep rendered-depth correspondences only where opacity is high enough.
3. Pair DUSt3R pointmap pixels with rendered-depth 3D points from MonoGS.
4. Estimate a robust Sim3 correction with RANSAC + Umeyama.
5. Reject DUSt3R insertion if the alignment has too few inliers, high RMSE, or
   an excessive scale correction.
6. Apply the accepted Sim3 correction after the baseline scale normalization and
   before Gaussian insertion.

Config:

```yaml
Training:
  dust3r:
    alignment:
      enabled: True
      required: True
      min_points: 96
      sample_points: 4096
      ransac_iterations: 64
      inlier_threshold: 0.08
      max_rmse: 0.08
      opacity_threshold: 0.65
      min_scale_correction: 0.67
      max_scale_correction: 1.50
```

Expected logs:

```text
DUSt3R Sim3 alignment kf ...: rmse=..., scale=..., inliers=.../...
Rejecting DUSt3R Sim3 alignment ...
Skipping DUSt3R Sim3 alignment ...
```

Impact:

- Replaces pure baseline-ratio scaling with a map-aware Sim3 correction.
- Filters out DUSt3R pointmaps that do not agree with the existing MonoGS map.
- Reduces the chance of duplicate, blurred, or floating Gaussians from scale or
  pose mismatch.

Relevant files:

- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 10. Coverage/Residual-Guided DUSt3R Insertion

The backend can optionally render the current Gaussian map from the new keyframe
pose before inserting DUSt3R points.

It keeps DUSt3R points only where:

- current rendered opacity is low, or
- RGB residual against the keyframe image is high, or
- opacity is below a minimum floor.

Config:

```yaml
dust3r:
  insertion:
    enabled: True
    opacity_threshold: 0.35
    rgb_residual_threshold: 0.12
    min_opacity_floor: 0.08
    min_points: 128
```

Impact:

- Avoids blindly inserting the full DUSt3R pointmap.
- Focuses DUSt3R geometry on poorly reconstructed regions.
- Helps reduce render-quality degradation caused by duplicate or misaligned
  Gaussians.

Relevant file:

- `utils/slam_backend.py`

## 11. DUSt3R-Aware Optimization Scheduler

The backend can now use a successful DUSt3R insertion as a signal to reduce
optimization work for that keyframe.

Policy:

- If DUSt3R inserts enough Gaussians, the backend uses fewer mapping iterations
  for that keyframe.
- During monocular pre-initialization and initial BA, DUSt3R-supported keyframes
  can use separate reduced iteration budgets.
- After a sufficiently large DUSt3R insertion, densification can be skipped for
  a configurable number of scheduled densify events.
- DUSt3R inference can be throttled by a minimum keyframe gap to avoid paying
  the model cost on every keyframe.

Config:

```yaml
Training:
  dust3r:
    optimization:
      enabled: True
      min_keyframe_gap: 2
      mapping_iters_with_dust3r: 5
      mapping_iters_without_dust3r: 10
      preinit_mapping_iters_with_dust3r: 20
      initial_ba_iters_with_dust3r: 150
      skip_densify_after_insert: 2
      min_insert_points_for_fast_mapping: 256
```

Impact:

- Turns DUSt3R from an added geometry source into a scheduling signal.
- Targets runtime reduction by cutting optimization only when DUSt3R provided
  enough usable geometry.
- Avoids redundant densification immediately after dense DUSt3R insertion.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `configs/mono/tum/ablations/fr3_office_07_dust3r_adaptive_optimization.yaml`

## 12. Online Gaussian Lifecycle Controller

The system now includes an optional online lifecycle controller for Gaussians.
It manages each Gaussian as one of four states:

```text
newborn
stable
cold
bad
```

Policy:

- `newborn`: newly inserted Gaussians. They receive a grace period and are
  protected from early pruning.
- `stable`: sufficiently visible Gaussians that remain normally trainable.
- `cold`: old, visible, low-gradient Gaussians with sufficient opacity. These
  are frozen by zeroing gradients and Adam moments before optimizer steps.
- `bad`: Gaussians with persistently low opacity or low visibility after the
  grace period. These can be pruned after `bad_patience` lifecycle updates.

Config:

```yaml
Training:
  lifecycle:
    enabled: True
    newborn_grace: 3
    stable_min_visibility: 3
    cold_min_age: 10
    cold_grad_threshold: 0.00001
    cold_opacity_threshold: 0.4
    bad_opacity_threshold: 0.05
    bad_min_visibility: 1
    bad_patience: 3
    freeze_cold: True
    prune_bad: True
    log_interval: 10
```

Implementation details:

- Lifecycle metadata is stored per Gaussian and kept aligned through insertion,
  densification, clone/split, and pruning.
- Cold Gaussians still render, but their parameter gradients and Adam moments are
  zeroed before optimizer steps.
- Cold Gaussians are excluded from densification by zeroing their accumulated
  densification gradients.
- Newborn Gaussians are protected from opacity-based pruning during their grace
  period.
- Bad Gaussians are merged into the existing MonoGS prune mask when pruning is
  enabled.

Impact:

- Reduces unnecessary updates to converged Gaussians.
- Adds a principled grace period for newborn Gaussians.
- Adds persistence-based pruning for bad Gaussians.
- Provides logs such as:

```text
Lifecycle: newborn=... stable=... cold=... bad=... total=...
```

Relevant files:

- `gaussian_splatting/scene/gaussian_model.py`
- `utils/slam_backend.py`
- `configs/mono/tum/fr3_office.yaml`

## 13. Ablation Configs

Dedicated ablation configs were added under:

```text
configs/mono/tum/ablations/
```

Experiments:

```text
fr3_office_00_monogs.yaml
  MonoGS baseline, DUSt3R disabled, mapping_itr_num=80.

fr3_office_01_dust3r_same_iters_no_insertion.yaml
  DUSt3R enabled, same mapping iterations, no coverage insertion.

fr3_office_02_dust3r_reduced_mapping_no_insertion.yaml
  DUSt3R enabled, mapping_itr_num=40, no coverage insertion.

fr3_office_03_dust3r_reduced_mapping_insertion.yaml
  DUSt3R enabled, mapping_itr_num=40, coverage/residual insertion enabled.

fr3_office_04_dust3r_baseline_gated.yaml
  Same as 03 but with stricter min_baseline=0.25.

fr3_office_05_monogs_lifecycle.yaml
  MonoGS baseline plus lifecycle controller, DUSt3R disabled.

fr3_office_06_dust3r_baseline_gated_lifecycle.yaml
  DUSt3R baseline-gated config plus lifecycle controller.

fr3_office_07_dust3r_adaptive_optimization.yaml
  DUSt3R baseline-gated config plus Sim3 alignment gate, adaptive mapping
  iterations, keyframe-gap throttling, and post-insertion densify skipping.

fr3_office_08_dust3r_reduced_mapping_no_scale.yaml
  Same as 02 but disables both baseline-ratio scaling and pointmap scale sync.

fr3_office_09_dust3r_reduced_mapping_baseline_ratio_scale.yaml
  Same as 08 but enables baseline-ratio scaling while keeping pointmap scale
  sync disabled, isolating the baseline-ratio scale contribution.
```

A helper script runs all ablations and stores logs:

```bash
./scripts/run_fr3_office_ablations.sh
```

Relevant files:

- `configs/mono/tum/ablations/*.yaml`
- `scripts/run_fr3_office_ablations.sh`

## 14. Observed Behavior So Far

From the available 200-frame `fr3_office` runs:

MonoGS baseline:

```text
ATE   0.01408
Time  247.28s
FPS   0.8088
PSNR  26.168
SSIM  0.8384
LPIPS 0.2094
```

DUSt3R + reduced mapping, no insertion:

```text
ATE   0.00979
Time  259.79s
FPS   0.7698
PSNR  25.567
SSIM  0.8274
LPIPS 0.2296
DUSt3R calls 16
DUSt3R time 23.55s
```

Interpretation:

- DUSt3R improved trajectory accuracy in this run.
- DUSt3R overhead outweighed the mapping-iteration reduction.
- Rendering quality decreased when inserting pointmaps without coverage guidance.
- The next configs to evaluate are `03` and `04`, which test insertion gating and
  stricter baseline gating.

## 15. How To Run

Default current `fr3_office` config:

```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

Run all ablations:

```bash
./scripts/run_fr3_office_ablations.sh
```

Key metrics to compare:

```text
Eval: RMSE ATE
Eval: DUSt3R calls
Eval: DUSt3R total time
Eval: Total time
Eval: Total FPS
Eval: mean psnr
```

Also inspect saved renders under:

```text
results/.../<timestamp>/viz/
```
