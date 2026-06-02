# System Improvements Over Original MonoGS

This document summarizes the current changes in this repository relative to the
original MonoGS baseline, with emphasis on the DUSt3R integration, Gaussian
lifecycle controller, evaluation logging, and ablation configs.

## 1. Headless Evaluation Setup

The TUM monocular `fr3_office` config is set up for container/headless runs:

- `Results.use_gui: False`
- `Results.eval_rendering: True`
- evaluation paths use a non-interactive matplotlib backend
- rendering metrics and ATE can be generated without an X display

Relevant files:

- `configs/mono/tum/fr3_office.yaml`
- `utils/eval_utils.py`

## 2. Limited-Frame Dataset Runs

TUM dataset loading supports a config-level frame limit:

```yaml
Dataset:
  num_frames: 1500
```

For TUM sequences, RGB paths, depth paths, and poses are sliced to the first
`num_frames` frames. If `num_frames <= 0` or the key is omitted, the full
sequence is used.

Relevant files:

- `utils/dataset.py`
- `configs/mono/tum/fr3_office.yaml`

## 3. Saved Render Visualizations

Rendering evaluation saves predicted, ground-truth, and comparison images under:

```text
results/.../<timestamp>/viz/before_opt/pred/
results/.../<timestamp>/viz/before_opt/gt/
results/.../<timestamp>/viz/before_opt/compare/
results/.../<timestamp>/viz/after_opt/pred/
results/.../<timestamp>/viz/after_opt/gt/
results/.../<timestamp>/viz/after_opt/compare/
```

`compare` images concatenate ground truth and rendered output horizontally.

Relevant file:

- `utils/eval_utils.py`

## 4. Final Map And Memory Logging

The final evaluation log now includes Gaussian map size and GPU memory usage:

```text
Eval: Final Gaussian count ...
Eval: Final Gaussian model memory [MB] ...
Eval: Final Gaussian optimizer state memory [MB] ...
Eval: CUDA memory allocated [MB] ...
Eval: CUDA memory reserved [MB] ...
Eval: CUDA max memory allocated [MB] ...
Eval: CUDA max memory reserved [MB] ...
```

`Final Gaussian model memory [MB]` counts persistent Gaussian tensors such as
position, color features, opacity, scale, rotation, visibility buffers, and
lifecycle buffers.

`Final Gaussian optimizer state memory [MB]` counts Adam state tensors for the
Gaussian optimizer.

CUDA memory logs are process-level PyTorch CUDA allocator values. Peak counters
are reset at the beginning of each SLAM run.

Relevant file:

- `slam.py`

## 5. DUSt3R Integration

DUSt3R and CroCo are included in this repository:

- `dust3r/`
- `croco/`
- `utils/dust3r_utils.py`

The checkpoint is expected at:

```text
checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

`slam.py` loads DUSt3R only when:

```yaml
Training:
  dust3r:
    enabled: True
```

When disabled, no DUSt3R model is loaded and no DUSt3R inference is performed.

Relevant files:

- `slam.py`
- `utils/dust3r_utils.py`
- `utils/slam_frontend.py`
- `utils/slam_backend.py`

## 6. DUSt3R Pointmap Gaussian Insertion

The system can create Gaussians from DUSt3R pointmaps for new keyframes.

Pipeline:

1. Frontend detects a new keyframe.
2. A reference keyframe is selected using baseline constraints and optional
   quality-aware candidate scoring.
3. DUSt3R runs on the current/reference keyframe pair.
4. DUSt3R pointmaps, RGB images, confidence masks, raw confidence maps,
   reciprocal-match count, scale divisors, and reference-frame metadata are sent
   to the backend.
5. Backend optionally applies Sim3 alignment and coverage/residual masking.
6. `GaussianModel` converts the filtered point cloud into Gaussian parameters.

Generated Gaussian attributes:

- position from DUSt3R pointmap
- color from DUSt3R-resized RGB image
- SH DC feature from RGB
- scale from nearest-neighbor distance
- identity rotation
- opacity initialized to 0.5

Relevant files:

- `utils/dust3r_utils.py`
- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 7. DUSt3R Insertion Switches

There are two separate switches. They are intentionally distinct:

```yaml
Training:
  dust3r:
    pointmap_insert:
      enabled: True
    insertion:
      enabled: True
```

`dust3r.pointmap_insert.enabled` controls whether DUSt3R pointmaps are allowed
to create Gaussians at all.

`dust3r.insertion.enabled` controls only the coverage/residual mask applied
before an allowed pointmap insertion.

This split keeps the ablations clear:

- `pointmap_insert.enabled: False`: no DUSt3R pointmap Gaussians are inserted.
- `pointmap_insert.enabled: True` and `insertion.enabled: False`: pointmaps can
  still be inserted, but without coverage/residual gating.
- `pointmap_insert.enabled: True` and `insertion.enabled: True`: pointmaps are
  inserted only in poorly covered or high-residual regions.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `configs/mono/tum/fr3_office.yaml`
- `configs/mono/tum/ablations/*.yaml`

## 8. DUSt3R Pointmap Scale Handling

DUSt3R pointmaps have an unknown metric scale. The frontend can normalize them
against the current MonoGS map scale:

```yaml
Training:
  dust3r:
    scale:
      baseline_ratio: True
      pointmap_sync: True
    scale_min: 0.05
    scale_max: 20.0
```

`baseline_ratio` computes a scale divisor from the DUSt3R pair translation and
the current MonoGS keyframe baseline.

`pointmap_sync` estimates separate pointmap scale divisors from reciprocal
matches so both pointmaps in the pair are more consistent before insertion.

The scale ablations are:

- `08`: no baseline-ratio scale and no pointmap sync
- `09`: baseline-ratio scale only, no pointmap sync
- configs with default scale settings: both enabled

Relevant file:

- `utils/slam_frontend.py`

## 9. DUSt3R Quality-Aware Pair Selection

DUSt3R now returns raw confidence maps and reciprocal-match counts, not only
binary masks. The frontend can reject or rank DUSt3R payloads using:

- mean confidence over the confidence mask
- valid confidence-mask ratio
- reciprocal match count
- pointmap scale-ratio sanity check
- combined quality score

Config:

```yaml
Training:
  dust3r:
    selection:
      enabled: True
      candidate_pool: 4
      max_candidate_evals: 1
      target_baseline: 0.35
      min_pair_conf: 3.0
      min_valid_ratio: 0.08
      min_matches: 512
      min_score: 1.5
      max_pointmap_scale_ratio: 1.75
```

Expected logs:

```text
DUSt3R candidate quality: kf ..., ref ..., conf ..., valid ..., matches ..., scale_ratio ..., score ...
Rejecting DUSt3R payload: ...
Selected DUSt3R payload: ...
```

Runtime policy:

- `fr3_office_05_full_system.yaml` evaluates at most one regular-keyframe
  candidate and uses `optimization.min_keyframe_gap` to reduce DUSt3R call
  frequency.
- Initialization configs can try more candidates because this cost is paid only
  at startup.

Relevant files:

- `utils/dust3r_utils.py`
- `utils/slam_frontend.py`
- `configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml`
- `configs/mono/tum/ablations/fr3_office_05_full_system.yaml`

## 10. DUSt3R Bootstrap And Event Refresh

DUSt3R can replace the normal monocular pseudo-depth initialization while still
respecting the online SLAM data flow. In the event-refresh config, frame 0 is
initialized immediately from DUSt3R single-view depth by feeding the same image
as a pair. This creates the first Gaussian map without pseudo-depth and without
waiting for future frames.

After bootstrap, the system can call DUSt3R again only when it has evidence that
the current Gaussian map no longer supports the scene well enough. This keeps
DUSt3R as an event-triggered geometry refresh module rather than a per-keyframe
dependency.

Config:

```yaml
Training:
  dust3r:
    init:
      enabled: True
      mode: "single_view"
      fallback_to_depth: False
      only: True
      prior_only: False
      backproject_depth: True
      fill_invalid_depth: False
    refresh:
      enabled: True
      backproject_depth: True
      force_after_bootstrap: True
      min_frame_gap: 30
      min_keyframe_gap: 2
```

Initialization policy:

- Frame 0 is a keyframe immediately.
- Frame 0 pose is the SLAM world identity, not a ground-truth pose.
- DUSt3R is called on `(frame0, frame0)` to obtain monocular depth/pointmap.
- Backend backprojects the DUSt3R depth to create initial Gaussians.
- If `prior_only=True`, DUSt3R is used as a depth regularization prior while
  normal pseudo-depth Gaussians are still created.
- If `fallback_to_depth=True`, the system falls back to standard monocular
  pseudo-depth initialization.
- If `fallback_to_depth=False`, initialization waits instead of silently falling
  back.
- If `only=True`, normal periodic DUSt3R keyframe insertion is skipped; explicit
  event-triggered refresh can still request DUSt3R when map health degrades.

Refresh policy:

- `force_after_bootstrap=True` requests a DUSt3R multiview refresh when frame 1
  arrives, using frame 0 as the reference.
- Later refreshes are triggered only by map-health signals: tracking loss spike,
  low opacity coverage, low visible Gaussian ratio, or a large rendered-depth
  distribution shift.
- Refreshes obey frame/keyframe cooldowns so DUSt3R does not run continuously.
- Accepted refresh payloads are quality-gated and backprojected from DUSt3R
  depth only in high-confidence regions.

Backprojection mode:

When `backproject_depth=True`, the backend resizes DUSt3R depth to the camera
resolution and backprojects it through the current camera intrinsics and tracked
pose. This is used for both single-view bootstrap and event-triggered refresh.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`
- `configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml`
- `configs/mono/tum/ablations/fr3_office_05_full_system.yaml`

## 11. DUSt3R Pointmap Filtering

Before converting DUSt3R pointmaps into Gaussians, points are filtered by:

- finite XYZ
- positive depth
- configured depth range
- configured point radius
- DUSt3R confidence mask
- optional Open3D statistical outlier removal

Config:

```yaml
Training:
  dust3r:
    depth_min: 0.05
    depth_max: 8.0
    max_point_radius: 10.0
    outlier_filter:
      enabled: True
      nb_neighbors: 20
      std_ratio: 2.0
```

Relevant file:

- `gaussian_splatting/scene/gaussian_model.py`

## 12. DUSt3R-to-MonoGS Sim3 Alignment Gate

The backend can align DUSt3R pointmaps to the current MonoGS map before
inserting Gaussians.

Pipeline:

1. Render the current Gaussian map from the new keyframe pose.
2. Keep rendered-depth correspondences only where opacity is high enough.
3. Pair DUSt3R pointmap pixels with rendered-depth 3D points from MonoGS.
4. Estimate a robust Sim3 correction with RANSAC + Umeyama.
5. Reject DUSt3R insertion if the alignment has too few inliers, high RMSE, or
   an excessive scale correction.
6. Apply the accepted Sim3 correction after baseline scale normalization and
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

Relevant files:

- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 13. DUSt3R-Aware Optimization Scheduler

The backend can use successful DUSt3R insertion as a signal to reduce mapping
work for that keyframe.

Config:

```yaml
Training:
  dust3r:
    optimization:
      enabled: True
      min_keyframe_gap: 3
      mapping_iters_with_dust3r: 5
      mapping_iters_without_dust3r: 10
      preinit_mapping_iters_with_dust3r: 20
      initial_ba_iters_with_dust3r: 150
      skip_densify_after_insert: 2
      min_insert_points_for_fast_mapping: 256
```

Policy:

- If DUSt3R inserts enough Gaussians, fewer mapping iterations are used.
- During monocular pre-initialization and initial BA, DUSt3R-supported keyframes
  can use separate iteration budgets.
- After a sufficiently large insertion, densification can be skipped for a
  configured number of densify events.
- `min_keyframe_gap` throttles DUSt3R inference to reduce runtime overhead.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `configs/mono/tum/ablations/fr3_office_05_full_system.yaml`

## 14. Online Gaussian Lifecycle Controller

The lifecycle controller manages each Gaussian as one of four states:

```text
newborn
stable
cold
bad
```

Policy:

- `newborn`: newly inserted Gaussians protected from early pruning.
- `stable`: sufficiently visible Gaussians that remain normally trainable.
- `cold`: old, visible, low-gradient Gaussians with sufficient opacity. These
  still render but are frozen by zeroing gradients and Adam moments.
- `bad`: Gaussians with persistently low opacity or low visibility after the
  grace period. These can be pruned after `bad_patience` lifecycle updates.

Config:

```yaml
Training:
  lifecycle:
    enabled: True
    newborn_grace: 5
    stable_min_visibility: 5
    cold_min_age: 30
    cold_grad_threshold: 0.00001
    cold_opacity_threshold: 0.7
    bad_opacity_threshold: 0.02
    bad_min_visibility: 1
    bad_patience: 5
    freeze_cold: True
    prune_bad: True
    log_interval: 10
```

Expected logs:

```text
Lifecycle: newborn=... stable=... cold=... bad=... total=...
```

Note: `bad` Gaussians can be pruned before the lifecycle log is printed, so a
logged `bad=0` does not necessarily mean no bad candidates were ever detected.

Relevant files:

- `gaussian_splatting/scene/gaussian_model.py`
- `utils/slam_backend.py`

## 15. Ablation Configs

Dedicated TUM `fr3_office` ablation configs live under:

```text
configs/mono/tum/ablations/
```

Current configs:

```text
fr3_office_00_monogs.yaml
  MonoGS baseline, DUSt3R disabled, lifecycle disabled, mapping_itr_num=80.

fr3_office_01_monogs_lifecycle.yaml
  MonoGS baseline plus lifecycle controller, DUSt3R disabled.

fr3_office_02_dust3r_pointmap_no_scale.yaml
  DUSt3R pointmap Gaussian insertion, coverage/residual gating, reduced mapping
  iterations, and no DUSt3R scale correction.

fr3_office_03_dust3r_pointmap_scaled.yaml
  Config 02 plus baseline-ratio scaling and synchronized pointmap scaling.

fr3_office_04_dust3r_event_refresh.yaml
  DUSt3R single-view frame-0 bootstrap plus event-triggered multiview depth
  refresh. This is the focused test for replacing pseudo-depth while avoiding
  per-keyframe DUSt3R calls.

fr3_office_05_full_system.yaml
  Full integrated system: DUSt3R init, pointmap insertion, scale correction,
  quality gating, Sim3 alignment, adaptive mapping, and lifecycle controller.
```

The helper script runs all ablations and stores logs:

```bash
./scripts/run_fr3_office_ablations.sh
```

Relevant files:

- `configs/mono/tum/ablations/*.yaml`
- `scripts/run_fr3_office_ablations.sh`

## 16. Recommended Comparisons

For a compact comparison against the original MonoGS-style baseline:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_00_monogs.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_01_monogs_lifecycle.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_02_dust3r_pointmap_no_scale.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_03_dust3r_pointmap_scaled.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_05_full_system.yaml
```

Use `04` to test the main FPS hypothesis: DUSt3R pays a small startup cost for
frame-0 bootstrap, pays one early multiview refresh, and then runs only when map
health indicates a meaningful scene/depth change.

## 17. Key Metrics

Compare these logs:

```text
Eval: RMSE ATE
Eval: DUSt3R calls
Eval: DUSt3R total time
Eval: Total time
Eval: Total FPS
Eval: mean psnr
Eval: mean ssim
Eval: mean lpips
Eval: Final Gaussian count
Eval: Final Gaussian model memory [MB]
Eval: Final Gaussian optimizer state memory [MB]
Eval: CUDA max memory allocated [MB]
Eval: CUDA max memory reserved [MB]
```

Also inspect saved renders under:

```text
results/.../<timestamp>/viz/
```
