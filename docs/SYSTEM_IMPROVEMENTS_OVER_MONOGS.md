# System Improvements Used By Config 04

This document tracks only the improvements currently applied by:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

Config 04 inherits from the MonoGS baseline config and keeps the normal MonoGS
tracking, local mapping, keyframe window, and bundle-adjustment flow. The active
algorithmic changes over the original monocular MonoGS baseline are:

1. DUSt3R depth bootstrap and event refresh
2. DUSt3R pointmap scale synchronization
3. Online Gaussian lifecycle controller

### Config Layout (preset)

The shared, sequence-independent tuning for config 04 now lives in a named
preset (`event_refresh` in `utils/config_presets.py`). The YAML file itself only
opts in:

```yaml
inherit_from: "configs/mono/tum/ablations/fr3_office_00_monogs.yaml"
preset: event_refresh
```

The preset is merged below the file's own keys (so explicit YAML still wins) and
above the inherited chain. The YAML snippets shown below reflect the *effective*
values applied by the preset; opening the config file will only show the
`preset:` line plus any scene-specific overrides. The same preset is reused for
other TUM sequences (`fr1_desk_04_dust3r_event_refresh.yaml`,
`fr2_xyz_04_dust3r_event_refresh.yaml`), which differ only in dataset
path/calibration. All DUSt3R/lifecycle parameters are sequence-independent for
TUM because the scale-normalization mechanisms (Section 2) absorb per-scene
scale differences.

Evaluation logging for Gaussian count, model memory, optimizer memory, and CUDA
memory is also kept because it is needed to measure the lifecycle effect. It is
not an algorithmic change to SLAM behavior.

## 1. DUSt3R Depth Bootstrap And Event Refresh

Config 04 uses DUSt3R as a sparse online depth-prior source. It does not use
DUSt3R for tracking, and it does not directly initialize Gaussians from DUSt3R
XYZ pointmaps. Instead, it takes the z-coordinate/depth from DUSt3R pointmaps
and backprojects that depth with the current SLAM camera intrinsics.

Relevant config:

```yaml
Training:
  dust3r:
    enabled: True
    require_initialized: False
    depth_max: 20.0
    init:
      enabled: True
      mode: "single_view"
      fallback_to_depth: False
      only: True
      prior_only: False
      backproject_depth: True
      use_confidence_mask: False
      fill_invalid_depth: False
      depth_scale:
        enabled: True
        mode: "median"
        target_median: 2.0
        min_scale: 0.25
        max_scale: 4.0
    refresh:
      enabled: True
      backproject_depth: True
      force_after_bootstrap: True
      min_frame_gap: 50
      min_keyframe_gap: 3
      candidate_pool: 6
      min_baseline: 0.08
      max_baseline: 1.20
      target_baseline: 0.30
      min_opacity_coverage: 0.12
      opacity_threshold: 0.25
      max_tracking_loss_ratio: 2.2
      max_depth_change_ratio: 2.0
      min_visible_gaussian_ratio: 0.01
      ema_decay: 0.95
      max_calls: 3
```

### Frame-0 Bootstrap

The first frame is initialized immediately as the first keyframe. DUSt3R is
called in single-view mode by feeding frame 0 as both images in the pair. The
resulting DUSt3R pointmap provides depth for frame 0.

The backend then backprojects this depth through the SLAM camera intrinsics and
the frame-0 pose to create the initial Gaussian map. Because single-view DUSt3R
depth is not metric, config 04 normalizes the frame-0 depth median to 2.0 m.

This means config 04 does not use the original MonoGS monocular pseudo-depth
initialization for frame 0.

### Gaussian Count Control (downsample)

DUSt3R depth backprojection produces one candidate point per valid pixel
(hundreds of thousands at 640x480). To keep the initial Gaussian count
comparable to the MonoGS baseline (which downsamples its pseudo-depth point
cloud by `pcd_downsample_init`), config 04 sets:

```yaml
Training:
  dust3r:
    init:
      pcd_downsample: 32
      sample_stride: 1
      max_points: 200000
```

`create_pcd_from_dust3r_depth` applies `pcd_downsample` first (random keep of
`1/pcd_downsample` of the valid points), then uses `max_points` only as a final
safety cap. With `pcd_downsample: 32` the frame-0 map lands at roughly the same
order of magnitude as the MonoGS baseline (~10k Gaussians at 640x480) instead of
hitting the 200k cap. The same downsample path is used for refresh keyframes.

### Event Refresh

After initialization, DUSt3R is not called for every keyframe. The frontend
monitors simple map-health signals and requests a DUSt3R refresh only when the
current map appears weak for the current view.

Refresh triggers include:

- low opacity coverage
- low visible-Gaussian ratio
- tracking loss spike relative to the running average
- large rendered-depth distribution change
- one forced early multiview refresh after bootstrap

These four signals are fused into a single normalized "ill-health" score (the
legacy per-signal OR logic has been removed). Each signal is mapped to a severity
that is 0 while healthy and 1.0 at its threshold, then summed (weighted); a
refresh fires when the total reaches `threshold`. This lets several
sub-threshold-but-degraded signals accumulate and trigger together, which the
old discrete OR logic would miss. Config 04 applies it via the `event_refresh`
preset:

```yaml
Training:
  dust3r:
    refresh:
      health_score:
        threshold: 1.0          # < 1 = more sensitive, > 1 = more conservative
        weights:                # default 1.0 each
          opacity_coverage: 1.0
          visible_ratio: 1.0
          loss_ratio: 1.0
          depth_ratio: 1.0
```

With `threshold: 1.0` and unit weights, a single signal at its threshold scores
1.0 and triggers (matching the old OR boundary), while accumulated weak signals
can also cross the threshold together. The per-signal `min_*/max_*` values from
the refresh block are reused as the per-signal normalization points.

Accepted refresh payloads are inserted through the same depth-backprojection
path: use DUSt3R pointmap z as depth, scale it into the SLAM map scale, then
backproject with the SLAM camera intrinsics.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`
- `utils/dust3r_utils.py`

## 2. DUSt3R Pointmap Scale Synchronization

DUSt3R pointmaps have an arbitrary scale. Config 04 enables both a fallback
baseline-ratio scale and synchronized pointmap scaling:

```yaml
Training:
  dust3r:
    scale:
      baseline_ratio: True
      pointmap_sync: True
```

`baseline_ratio` estimates a fallback scale divisor from the DUSt3R pair
translation and the SLAM keyframe baseline:

```text
scale_divisor = ||t_dust3r|| / ||baseline_slam||
```

`pointmap_sync` estimates separate scale divisors for the current and reference
DUSt3R pointmaps using reciprocal matches. It solves for per-pointmap metric
scales that make matched DUSt3R 3D directions agree with the current SLAM
baseline:

```text
s_cur * vec_cur - s_ref * vec_ref ~= baseline_slam
scale_divisors = 1 / [s_cur, s_ref]
```

If pointmap sync fails, the system falls back to the baseline-ratio divisor. If
both mechanisms are disabled or unavailable, the payload scale is `1.0`.

In config 04, this scale synchronization affects multiview DUSt3R refreshes.
Even though config 04 uses depth backprojection instead of direct XYZ insertion,
the backend still divides DUSt3R depth by the selected pointmap divisor before
backprojection.

Relevant files:

- `utils/slam_frontend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 3. Online Gaussian Lifecycle Controller

Config 04 enables a conservative lifecycle controller for Gaussians:

```yaml
Training:
  lifecycle:
    enabled: True
    newborn_grace: 10
    stable_min_visibility: 5
    cold_min_age: 80
    cold_grad_threshold: 0.00001
    cold_opacity_threshold: 0.5
    bad_opacity_threshold: 0.02
    bad_min_visibility: 0
    bad_use_recent_visibility: False
    bad_patience: 5
    freeze_cold: False
    suppress_cold_densify: False
    prune_bad: True
    prune_bad_local_only: True
    protect_newborn_from_prune: False
    log_interval: 10
```

The controller tracks four states:

- `newborn`: recently inserted Gaussians inside the grace period
- `stable`: visible Gaussians that remain normally trainable
- `cold`: old, visible, low-gradient Gaussians
- `bad`: persistently low-opacity Gaussians after the grace period

The current config is intentionally conservative:

- cold Gaussians are not frozen
- cold Gaussians are not blocked from densification
- newborn Gaussians are not protected from MonoGS' normal opacity pruning
- bad pruning is restricted to MonoGS' local prune scope
- recent invisibility alone does not make an old Gaussian bad

This avoids the previous failure mode where valid old map Gaussians were pruned
after the camera turned away from them, which could later distort the trajectory
and move earlier good poses during backend optimization.

Expected log:

```text
Lifecycle: newborn=... stable=... cold=... bad=... total=...
```

Relevant files:

- `gaussian_splatting/scene/gaussian_model.py`
- `utils/slam_backend.py`

## 4. Evaluation Logging

Config 04 also benefits from extra evaluation logs used to compare map size and
runtime memory:

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
positions, SH features, opacity, scale, rotation, visibility buffers, and
lifecycle buffers.

`Final Gaussian optimizer state memory [MB]` counts Adam optimizer state tensors
owned by the Gaussian optimizer.

CUDA memory logs are process-level PyTorch CUDA allocator values. They are useful
for measuring practical memory pressure, but they include more than just the
Gaussian model tensors.

Relevant file:

- `slam.py`

## What Is Not Included Here

The following experimental branches are intentionally not documented as config
04 improvements:

- direct DUSt3R pointmap XYZ Gaussian initialization
- DUSt3R-to-MonoGS Sim3 alignment gate
- DUSt3R-aware mapping-iteration scheduler from config 05
- old pointmap insertion ablations
- headless evaluation, saved render images, and dataset frame limits

Those may still exist in the repository as utilities or ablation code, but they
are not part of the current config 04 system being compared against baseline
MonoGS.

## Summary

For config 04, the real improvements over baseline MonoGS are:

- DUSt3R replaces frame-0 pseudo-depth with DUSt3R-derived depth.
- DUSt3R is reused only as an event-triggered multiview depth refresh.
- Pointmap sync aligns DUSt3R multiview depth scale to the SLAM map scale.
- The lifecycle controller conservatively prunes persistently low-opacity local
  bad Gaussians and logs Gaussian states.
- Extra final logs report Gaussian count, model memory, optimizer memory, and
  CUDA memory for evaluation.
