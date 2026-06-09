# System Improvements Used By Config 04

This document tracks only the improvements currently applied by:

```bash
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

Config 04 inherits from the MonoGS baseline config and keeps the normal MonoGS
tracking, local mapping, keyframe window, and bundle-adjustment flow. The active
algorithmic changes over the original monocular MonoGS baseline are:

1. DUSt3R depth prior (bootstrap + event refresh insertion)
2. Loss/depth event-score selection for when to call DUSt3R
3. DUSt3R pointmap scale synchronization

These are the three contributions evaluated in the ablation study. The system no
longer uses a Gaussian lifecycle controller; that path is disabled
(`Training.lifecycle.enabled: False`) and is not part of the proposed method.

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
path/calibration. All DUSt3R parameters are sequence-independent for TUM because
the scale-normalization mechanisms (Section 3) absorb per-scene scale
differences.

Evaluation logging for Gaussian count, model memory, optimizer memory, and CUDA
memory is also kept because it is needed to measure map size and runtime memory.
It is not an algorithmic change to SLAM behavior.

## 1. DUSt3R Depth Prior (Bootstrap And Refresh Insertion)

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
      backproject_depth: True
      use_confidence_mask: False
      depth_scale:
        enabled: True
    refresh:
      enabled: True
      min_frame_gap: 50
      min_keyframe_gap: 3
      candidate_pool: 6
      min_baseline: 0.08
      max_baseline: 1.20
      target_baseline: 0.30
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

### Refresh Insertion

After initialization, DUSt3R is not called for every keyframe. The decision of
*when* to call it is the loss/depth event score of Section 2; this
section covers how an accepted refresh payload is inserted.

Accepted refresh payloads are inserted through the same depth-backprojection
path: use DUSt3R pointmap z as depth, scale it into the SLAM map scale
(Section 3), then backproject with the SLAM camera intrinsics.

Relevant files:

- `utils/slam_frontend.py`
- `utils/slam_backend.py`
- `utils/dust3r_utils.py`

## 2. Loss/Depth Event-Score Selection

DUSt3R inference is expensive (close to ~1 s per call with the large model).
Calling it on every keyframe would collapse FPS, so the frontend decides when a
DUSt3R refresh is worthwhile from cheap loss/depth event ratios computed from
the current render.

The refresh trigger now uses two event ratios:

- tracking loss spike relative to the running average
- large rendered-depth distribution change

Opacity coverage and visible-Gaussian ratio no longer participate in the refresh
score. The event trigger focuses on the two failure modes DUSt3R is meant to
repair: tracking loss spikes and depth distribution shifts.

### Loss/Depth Event Score

The two ratios are normalized with symmetric log-ratios and fused into one score:

```text
D_t = max(0, log(depth_ratio_t) / log(T_depth))
L_t = max(0, log(loss_ratio_t)  / log(T_loss))

score_t = max(D_t, L_t) + lambda_joint * min(D_t, L_t)
```

`D_t = 1` when the depth ratio reaches `T_depth`; `L_t = 1` when the tracking
loss ratio reaches `T_loss`. The `max(D_t, L_t)` term catches either individual
event, and `lambda_joint * min(D_t, L_t)` adds a small bonus when both tracking
and depth degrade moderately. A refresh fires when `score_t >= threshold`.

```yaml
Training:
  dust3r:
    refresh:
      health_score:
        threshold: 1.0
        loss_trigger_ratio: 2.2
        depth_trigger_ratio: 2.0
        joint_bonus: 0.25
```

In addition to the score, refreshes obey cooldown and budget limits so
DUSt3R is never called too densely:

```yaml
min_frame_gap: 50
min_keyframe_gap: 3
max_calls: 3
```

One forced early multiview refresh after bootstrap (`force_after_bootstrap: True`)
upgrades the single-view frame-0 depth with a higher-quality multiview pointmap
once a suitable reference frame exists.

Relevant files:

- `utils/slam_frontend.py`

## 3. DUSt3R Pointmap Scale Synchronization

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
positions, SH features, opacity, scale, rotation, and visibility buffers.

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

- the online Gaussian lifecycle controller (disabled; not part of the proposed
  method)
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

- DUSt3R replaces frame-0 pseudo-depth with DUSt3R-derived depth, and the same
  depth-prior path inserts geometry at event-triggered refreshes.
- A single loss/depth event score decides when DUSt3R is worth calling, instead
  of a fixed per-keyframe schedule or four separate health signals. It detects
  tracking-loss spikes, depth-distribution shifts, and moderate joint
  degradation.
- Pointmap sync aligns DUSt3R multiview depth scale to the SLAM map scale.
- Extra final logs report Gaussian count, model memory, optimizer memory, and
  CUDA memory for evaluation.
</content>
</invoke>
