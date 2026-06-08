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
            "enabled": True,
            "require_initialized": False,
            "min_baseline": 0.25,
            "depth_max": 20.0,
            "selection": {
                "enabled": True,
                "target_baseline": 0.35,
                "min_valid_ratio": 0.08,
                "min_matches": 512,
                "min_score": 1.5,
                "max_pointmap_scale_ratio": 1.75,
            },
            "init": {
                "enabled": True,
                "mode": "single_view",
                "fallback_to_depth": False,
                "only": True,
                "backproject_depth": True,
                "pcd_downsample": 32,
                "point_size_scale": 0.25,
                "sample_stride": 1,
                "gradient_stride": 2,
                "use_confidence_mask": False,
                "depth_scale": {"enabled": True},
                "selection": {
                    "candidate_pool": 1,
                    "min_valid_ratio": 0.08,
                    "min_score": 0.5,
                    "max_pointmap_scale_ratio": 1.75,
                },
            },
            "refresh": {
                "enabled": True,
                "min_frame_gap": 50,
                "min_keyframe_gap": 3,
                "candidate_pool": 6,
                "min_baseline": 0.08,
                "max_baseline": 1.20,
                "target_baseline": 0.30,
                "min_opacity_coverage": 0.12,
                "max_tracking_loss_ratio": 2.2,
                "max_depth_change_ratio": 2.0,
                "ema_decay": 0.95,
                "max_calls": 3,
                # Refresh is triggered by the fused weighted health score (the
                # legacy per-signal OR logic has been removed). Each of the four
                # health signals is normalized to a severity (0 = healthy, 1.0 at
                # its threshold) and summed with these weights; a refresh fires
                # when the total reaches `threshold`.
                "health_score": {
                    "threshold": 1.0,
                    "weights": {
                        "opacity_coverage": 1.0,
                        "visible_ratio": 1.0,
                        "loss_ratio": 1.0,
                        "depth_ratio": 1.0,
                    },
                },
            },
            "optimization": {"min_keyframe_gap": 999999},
        },
        "lifecycle": {
            "enabled": True,
            "bad_min_visibility": 0,
            "bad_opacity_threshold": 0.02,
            "bad_patience": 5,
            "cold_min_age": 80,
            "cold_opacity_threshold": 0.5,
            "freeze_cold": False,
            "newborn_grace": 10,
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
