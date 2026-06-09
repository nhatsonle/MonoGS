"""Reusable config presets.

A preset is a nested dict of config overrides that several scene configs share.
A YAML config opts in with a top-level ``preset: <name>`` key (or a list of
names). Presets are merged *below* the YAML file's own keys, so anything written
explicitly in the YAML still wins, and *above* the inherited (``inherit_from``)
chain.

This lets per-sequence configs shrink to just the dataset path/calibration plus
``preset: event_refresh`` while the shared DUSt3R / lifecycle tuning lives here
in one place. Only sequence-independent (unitless / structural) parameters
belong in a preset; anything that must change per scene stays in the YAML.
"""

# DUSt3R single-view init + event-driven refresh ("config 04").
# Every value here is sequence-independent for the TUM benchmark: quality
# thresholds (ratios/counts), refresh health ratios, lifecycle counters, and
# structural flags. Scale-sensitive knobs (baselines, depth ranges) are kept,
# but the auto scale-normalization (scale.* + init.depth_scale) lets the same
# metric values work across fr1/fr2/fr3.
EVENT_REFRESH = {
    "Training": {
        "dust3r": {
            # Master switch for the whole DUSt3R depth-prior path. False = pure
            # baseline MonoGS (pseudo-depth init, no refresh).
            "enabled": False,
            # If True, DUSt3R only runs once the SLAM map is already initialized.
            # We need DUSt3R *to* bootstrap frame 0, so this must stay False.
            "require_initialized": False,
            # Global minimum camera baseline (in SLAM map units) for any pair to
            # be considered a usable DUSt3R stereo pair. Used as the fallback
            # floor for the init/refresh reference search when their own
            # min_baseline is not set. Below this, two views are too close for
            # stable multiview geometry.
            "min_baseline": 0.25,
            # Hard cap on DUSt3R depth (meters in SLAM scale) kept when
            # backprojecting. Points further than this are dropped as unreliable.
            "depth_max": 20.0,
            # --- Selection-based path (prepare_best_keyframe_dust3r) ---
            # Quality gate + reference-search tuning for the generic
            # "pick the best keyframe pair and run DUSt3R" path. In config 04
            # the refresh path drives DUSt3R, so these mostly act as the shared
            # defaults that init.selection inherits from.
            "selection": {
                "enabled": True,
                # Preferred camera baseline; candidate pairs are ranked by how
                # close their baseline is to this value.
                "target_baseline": 0.35,
                # Reject a DUSt3R result if fewer than this fraction of pixels
                # produce valid (finite, in-range) depth.
                "min_valid_ratio": 0.08,
                # Reject if fewer than this many reciprocal feature matches were
                # found between the two views (too few = unreliable scale/geometry).
                "min_matches": 512,
                # Reject if the pair's quality score is below this. Higher = stricter.
                "min_score": 1.5,
                # Reject if the two pointmaps' estimated scales disagree by more
                # than this ratio (a sign the DUSt3R reconstruction is inconsistent).
                "max_pointmap_scale_ratio": 1.75,
            },
            # --- Frame-0 bootstrap (single-view init) ---
            "init": {
                "enabled": True,
                # "single_view": feed frame 0 as BOTH images of the pair so
                # DUSt3R produces a monocular depth prior for the very first frame.
                "mode": "single_view",
                # If DUSt3R init fails/low-quality: False = wait and retry rather
                # than silently falling back to MonoGS pseudo-depth (keeps the
                # ablation clean — frame 0 is always DUSt3R-initialized).
                "fallback_to_depth": False,
                # True = run DUSt3R init exactly once (at bootstrap) and never
                # re-run the init path again after the map exists.
                "only": True,
                # Use DUSt3R's z as depth and backproject through the SLAM
                # intrinsics (the proposed depth-prior method), instead of
                # inserting DUSt3R XYZ pointmaps directly.
                "backproject_depth": True,
                # Keep ~1/32 of the valid backprojected points so the frame-0
                # Gaussian count matches the MonoGS baseline (~10k at 640x480)
                # instead of hitting the per-pixel hundreds-of-thousands.
                "pcd_downsample": 32,
                # Multiplier on the per-Gaussian initial radius (footprint).
                # <1 = smaller, sharper initial splats.
                "point_size_scale": 0.25,
                # Pixel subsampling stride before downsample (1 = use every pixel).
                "sample_stride": 1,
                # Stride for the gradient/normal estimation used when sizing
                # splats (2 = coarser, cheaper).
                "gradient_stride": 2,
                # Single-view DUSt3R confidence is unreliable; masking it out
                # would discard valid geometry, so confidence masking is OFF here.
                "use_confidence_mask": False,
                # Normalize single-view (non-metric) depth so its median maps to
                # a fixed target (default target_median=2.0 m, scale clipped to
                # [0.25, 4.0]). Gives a stable, scene-independent initial scale.
                "depth_scale": {"enabled": True},
                # Reference/quality overrides specific to the bootstrap pair.
                # Inherits any unset key from the dust3r.selection block above.
                "selection": {
                    # Only need the single self-pair for single-view init.
                    "candidate_pool": 1,
                    "min_valid_ratio": 0.08,
                    # Looser score gate at bootstrap — we must get frame 0 in.
                    "min_score": 0.5,
                    "max_pointmap_scale_ratio": 1.75,
                },
            },
            # --- Event-driven refresh (the core of contribution 2) ---
            # When the loss/depth event score fires, insert fresh DUSt3R multiview
            # depth. These knobs bound HOW OFTEN that can happen and WHICH
            # reference frame is paired with the current one.
            "refresh": {
                "enabled": True,
                # Cooldown: at least this many frames since the last refresh.
                "min_frame_gap": 50,
                # Cooldown: at least this many new keyframes since the last refresh.
                "min_keyframe_gap": 3,
                # How many candidate reference keyframes to consider per refresh.
                "candidate_pool": 6,
                # Acceptable baseline window (SLAM units) for the refresh pair.
                # Too small = degenerate stereo; too large = matching breaks down.
                "min_baseline": 0.08,
                "max_baseline": 1.20,
                # Preferred baseline; candidates are ranked toward this value.
                "target_baseline": 0.30,
                # Tracking-loss spike (vs EMA) that reaches event severity 1.0.
                "max_tracking_loss_ratio": 2.2,
                # Rendered-depth distribution shift that reaches event severity 1.0.
                "max_depth_change_ratio": 2.0,
                # EMA decay for the tracking-loss running average.
                "ema_decay": 0.95,
                # Hard budget: at most this many DUSt3R refreshes for the whole run
                # (DUSt3R is ~1 s/call, so refreshes are rationed).
                "max_calls": 3,
                # Refresh is triggered by one event score using only:
                #   D = max(0, log(depth_ratio) / log(depth_trigger_ratio))
                #   L = max(0, log(loss_ratio) / log(loss_trigger_ratio))
                #   score = max(D, L) + joint_bonus * min(D, L)
                # A score of 1.0 means one signal hit its trigger ratio; the
                # joint bonus lets moderate depth and tracking changes combine.
                "health_score": {
                    "threshold": 1.0,
                    "loss_trigger_ratio": 2.2,
                    "depth_trigger_ratio": 2.0,
                    "joint_bonus": 0.25,
                },
            },
            # Legacy keyframe-gap-driven DUSt3R scheduling (separate from the
            # refresh cooldown above). Setting min_keyframe_gap absurdly high
            # disables that old fixed-schedule path so the loss/depth event score
            # is the only thing that triggers DUSt3R during the run.
            "optimization": {"min_keyframe_gap": 999999},
        },
        # --- Gaussian lifecycle controller ---
        # Tags each Gaussian as newborn/stable/cold/bad from its age, visibility,
        # opacity and gradient (see GaussianModel.update_lifecycle). NOTE: per the
        # proposed-method docs this controller is NOT part of config 04's
        # contributions; it remains here as instrumentation/ablation. The
        # classification thresholds are all unitless counters, hence
        # scene-independent.
        "lifecycle": {
            "enabled": True,
            # "bad" path: a Gaussian seen fewer than this many times recently is
            # a bad candidate. 0 disables the recency-based bad trigger (only the
            # low-opacity trigger remains).
            "bad_min_visibility": 0,
            # Opacity below this (past the newborn grace) marks a Gaussian as a
            # bad candidate (likely a floater/failed splat).
            "bad_opacity_threshold": 0.02,
            # Consecutive frames a Gaussian must stay a bad candidate before it
            # is actually labeled "bad" (debounce against transient occlusion).
            "bad_patience": 5,
            # "cold" path: minimum age (frames) before a well-behaved, settled
            # Gaussian can be considered cold (converged, low gradient).
            "cold_min_age": 80,
            # A cold Gaussian must have opacity at least this high (cold = solid
            # and converged, not faint).
            "cold_opacity_threshold": 0.5,
            # If True, cold Gaussians are frozen (excluded from optimization) to
            # save compute. False here = keep optimizing them (safer for quality).
            "freeze_cold": False,
            # Grace period (frames) after birth during which a Gaussian is never
            # judged bad/cold — gives new splats time to converge.
            "newborn_grace": 10,
            # Cumulative/recent visibility count required to be considered
            # "stable" (and a prerequisite for "cold").
            "stable_min_visibility": 5,
        },
    },
}

PRESETS = {
    "event_refresh": EVENT_REFRESH,
}


def get_preset(name):
    """Return the override dict for a named preset, or raise on unknown name."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown config preset {name!r}. Available presets: "
            f"{sorted(PRESETS)}"
        )
    return PRESETS[name]
