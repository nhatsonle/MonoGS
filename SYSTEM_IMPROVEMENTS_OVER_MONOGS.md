# System Improvements Used By Config 04

This document tracks only the improvements currently applied by:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

Config 04 inherits from the MonoGS baseline config and keeps the normal MonoGS
tracking, local mapping, keyframe window, and bundle-adjustment flow. The active
algorithmic changes over the original monocular MonoGS baseline are:

1. DUSt3R depth bootstrap and event refresh
2. Baseline-ratio DUSt3R depth scaling
3. Online Gaussian lifecycle controller

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
    mode: adaptive
    require_initialized: False
    depth_max: 20.0
    scale:
      baseline_ratio: True
    init:
      enabled: True
      mode: "single_view"
      fallback_to_depth: False
      only: True
      backproject_depth: True
    refresh:
      enabled: True
      backproject_depth: True
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

### Map Evidence Refresh

After initialization, DUSt3R is not called for every keyframe. The frontend
computes a single map evidence loss and requests a DUSt3R refresh only when the
current Gaussian map no longer explains the incoming frame well enough:

```text
L_refresh =
    w_photo * L_photo
  + w_opacity * L_opacity
  + w_visibility * L_visibility
  + w_geometry * L_geometry
  + w_bootstrap * L_bootstrap
```

The former map-health signals are now normalized components of this one loss:

- `L_photo`: tracking loss increase relative to its running average
- `L_opacity`: lack of rendered opacity support in the current frame
- `L_visibility`: lack of visible Gaussian support
- `L_geometry`: log-depth distribution innovation since the last refresh
- `L_bootstrap`: temporary uncertainty after single-view bootstrap

In adaptive mode, the refresh threshold is estimated online from the running
median and MAD of `L_refresh`, rather than being fixed in the config:

```text
L_refresh >= median(L_refresh history) + k * MAD(L_refresh history)
```

Reference keyframes are selected by normalized parallax, so metric baseline
ranges do not need to be tuned per dataset. When the map-evidence trigger
fires, adaptive config 04 allows DUSt3R to run instead of requiring a fixed call
budget.

Accepted refresh payloads are inserted through the same depth-backprojection
path: use DUSt3R pointmap z as depth, scale it with the baseline-ratio divisor,
then backproject with the SLAM camera intrinsics.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `gaussian_splatting/scene/gaussian_model.py`
- `utils/dust3r_utils.py`

## 2. Baseline-Ratio DUSt3R Depth Scaling

DUSt3R pointmaps have an arbitrary scale. Config 04 uses a single
baseline-ratio scale divisor for multiview DUSt3R refreshes:

```yaml
Training:
  dust3r:
    scale:
      baseline_ratio: True
```

`baseline_ratio` estimates the scale divisor from the DUSt3R pair translation
and the SLAM keyframe baseline:

```text
scale_divisor = ||t_dust3r|| / ||baseline_slam||
```

The backend divides DUSt3R depth by this divisor before backprojection. This is
kept deliberately simple because config 04 uses depth backprojection rather
than direct XYZ pointmap insertion.

Relevant files:

- `utils/slam_frontend.py`
- `gaussian_splatting/scene/gaussian_model.py`

## 3. Online Gaussian Lifecycle Controller

Config 04 enables an adaptive lifecycle controller for Gaussians:

```yaml
Training:
  lifecycle:
    enabled: true
    mode: adaptive
    aggressiveness: 0.5
    local_only: true
    log_interval: 10
```

The controller tracks four states, but derives the internal thresholds from one
high-level parameter, `aggressiveness`:

- `newborn`: recently inserted Gaussians that are too young to judge
- `stable`: mature Gaussians without strong bad evidence
- `cold`: mature, supported, low-gradient Gaussians that appear converged
- `bad`: mature Gaussians with persistent low map evidence

Adaptive mode computes an EMA bad score from local rendering evidence:

```text
bad_score_ema <- decay * bad_score_ema + (1 - decay) * quality_loss
```

`quality_loss` combines low opacity relative to the current opacity
distribution, weak local-window support, and Gaussian maturity. Low visibility
alone is not enough to mark an old Gaussian bad; it only matters when the
Gaussian also has weak opacity evidence.

The controller modifies MonoGS' original map management in two conservative
ways:

- `bad` Gaussians can be added to MonoGS' local prune mask.
- `cold` or suspected-bad Gaussians are prevented from spawning extra
  clone/split children during densification.

Pruning remains restricted to MonoGS' local prune scope when `local_only: true`,
which avoids deleting old valid map regions merely because the camera has turned
away from them.

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
- DUSt3R is reused only when the map evidence loss requests multiview refresh.
- Baseline-ratio scaling aligns DUSt3R multiview depth to the SLAM map scale.
- The adaptive lifecycle controller uses one aggressiveness setting to prune
  persistent low-evidence local Gaussians and suppress redundant densification.
- Extra final logs report Gaussian count, model memory, optimizer memory, and
  CUDA memory for evaluation.
