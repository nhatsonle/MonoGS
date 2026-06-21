# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**3RGS** is a dense monocular Gaussian-Splatting SLAM system built on top of MonoGS (Gaussian Splatting SLAM, CVPR 2024). It keeps MonoGS's tracking / local-mapping / keyframe-window / bundle-adjustment flow and replaces the weak RGB-only geometry initialization with a learned **DUSt3R depth prior**. DUSt3R is the source of new Gaussians at keyframes (frame-0 bootstrap + event-triggered refresh) instead of backprojecting RGB pseudo-depth.

The full proposed system is the **config-04 family** (e.g. `configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml`). The two design docs are authoritative for the method and should be consulted before changing DUSt3R logic:
- `docs/SYSTEM_IMPROVEMENTS_OVER_3RGS.md` — the three contributions over baseline MonoGS
- `docs/PROPOSED_METHOD_CONFIG04.md`

## Commands

```bash
# Run the proposed system (GUI pops up)
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml

# Headless eval: forces save_results, eval_rendering, use_wandb on; disables GUI.
# Writes ATE + render metrics to the save dir and run_metrics.json (FPS, DUSt3R
# cost, Gaussian count, memory).
python slam.py --config <cfg> --eval

# Deterministic / reproducible variant (no multiprocess randomness)
python slam.py --config configs/mono/tum/ablations/fr2_xyz_05_full_single_thread.yaml

# Baseline MonoGS (DUSt3R disabled)
python slam.py --config configs/mono/tum/fr3_office.yaml

# Datasets (download into ./datasets)
bash scripts/download_tum.sh        # also download_replica.sh, download_euroc.sh

# Leave-one-out ablations (proposed method with one improvement removed at a time)
bash scripts/run_fr2_ablation.sh    # also run_fr1_ablation.sh, run_fr3_office_ablations.sh

# Lint / format (ruff config in pyproject.toml: 88 cols, 4-space, double quotes)
ruff check .
ruff format .
```

There is **no automated test suite**. Validate a change by running the smallest relevant config and checking startup, tracking, mapping, and GUI/runtime for regressions. For CUDA/dependency changes also run the import checks in `README.md`.

## Hard environment constraints

- **Must run on NumPy 1.x (pinned `1.26.4`).** OpenCV and `numpy-quaternion` are compiled against the NumPy 1.x ABI and the data loaders use APIs removed in NumPy 2.0 (`np.unicode_`). NumPy 2.x crashes at runtime. The README install is deliberately split into small steps with a constraints file to stop any dependency from silently upgrading NumPy. Use `numpy==1.26.4` + `opencv-python>=4.9`.
- **DUSt3R checkpoint is required** (not optional): `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`. `slam.py` raises if `Training.dust3r.enabled` and the checkpoint is missing.
- CUDA kernels must be built before running: `croco/models/curope` (RoPE), plus the `submodules/simple-knn` and `submodules/diff-gaussian-rasterization` pip installs. `submodules/` and `croco/`, `dust3r/` are vendored — avoid editing unless the change is scoped there.

## Architecture

### Three-process design (entry point `slam.py`)
`SLAM.__init__` wires everything and spawns processes that communicate over `torch.multiprocessing` queues (start method is `spawn`):

- **FrontEnd** (`utils/slam_frontend.py`, runs in the main process via `self.frontend.run()`) — per-frame tracking (pose optimization against the current Gaussian render), keyframe selection, and **all DUSt3R decisions** (when to call it, reference-frame selection, scale estimation). Builds DUSt3R "payloads" and sends them to the backend.
- **BackEnd** (`utils/slam_backend.py`, separate `mp.Process`) — owns the `GaussianModel`, does local mapping / bundle adjustment, prunes, and inserts Gaussians (including from DUSt3R payloads). Syncs the updated map back to the frontend.
- **GUI** (`gui/slam_gui.py`, optional `mp.Process`) — Open3D/OpenGL viewer. Disabled with `use_gui=False` (forced off by `--eval`). `FakeQueue` (`utils/multiprocessing_utils.py`) stands in for the GUI queues when disabled.

### Queue protocol (the spine of the system)
- `backend_queue` (frontend→backend): messages `["init", ...]`, `["keyframe", ...]`, `["pause"]`/`["unpause"]`, `["color_refinement"]`, `["stop"]`. `init`/`keyframe` messages carry `cur_frame_idx, viewpoint, [window], depth_map, dust3r_payload`.
- `frontend_queue` (backend→frontend): `["sync_backend", gaussians, ...]`, tagged `"init"`/`"keyframe"` syncs that hand the updated Gaussian model back.
- `q_main2vis` / `q_vis2main`: GUI packets (`GaussianPacket`) and pause flags.

Both `run()` loops are the place to understand control flow: `FrontEnd.run` (`utils/slam_frontend.py:1345`) and `BackEnd.run` (`utils/slam_backend.py:1026`).

### The DUSt3R prior path (the actual contribution)
Three changes layered over baseline MonoGS — see the design docs for the math:

1. **Depth prior, not direct XYZ.** DUSt3R produces a pointmap; the system takes its z as depth, rescales it, and **backprojects through the SLAM intrinsics** (`backproject_depth: True`). It does *not* insert DUSt3R XYZ directly and does *not* use DUSt3R for tracking. Frame 0 is bootstrapped via single-view DUSt3R (frame 0 fed as both images of the pair); its non-metric depth median is normalized to ~2.0 m. Point-cloud creation: `GaussianModel.create_pcd_from_dust3r_depth` (`gaussian_splatting/scene/gaussian_model.py:521`); `pcd_downsample` keeps the Gaussian count comparable to the MonoGS baseline.
2. **Loss/depth event-score selection** decides *when* to call DUSt3R (it's ~1 s/call, too slow for every keyframe). Two self-normalizing ratios — tracking-loss spike vs. EMA, and rendered-depth distribution shift — are fused as `score = max(D, L)`. A **rising-edge trigger** fires once when `score >= threshold`, then disarms until the score settles below `rearm_threshold`. This replaces fixed cooldowns / per-run call budgets. Logic: `FrontEnd.dust3r_refresh_event_score` and `should_trigger_dust3r_refresh`.
3. **Pointmap scale synchronization** (`scale.baseline_ratio` + `scale.pointmap_sync`) absorbs DUSt3R's arbitrary scale so the same tuning works across scenes. `pointmap_sync` solves per-pointmap metric scales from reciprocal matches against the SLAM baseline; falls back to `baseline_ratio` (`||t_dust3r|| / ||baseline_slam||`), then to `1.0`. See `FrontEnd.estimate_dust3r_pointmap_scale_divisors`.

The Gaussian **lifecycle controller** (`GaussianModel.update_lifecycle`, `lifecycle.*` config) exists as instrumentation/ablation but is **disabled in config 04 and is NOT part of the proposed method** — don't treat it as load-bearing.

### Config system (read this before touching configs)
Configs use a two-axis composition resolved by `utils/config_utils.py::load_config`:

1. **`inherit_from`** chains a parent YAML (recursively). Layout is `configs/<sensor>/<dataset>/...` where `<sensor>` ∈ `mono | rgbd | stereo | live`. A typical chain: `fr3_office_04_*.yaml` → `fr3_office_00_3rgs.yaml` → `fr3_office.yaml` → `base_config.yaml`.
2. **`preset:`** pulls a named override block from `utils/config_presets.py` (currently only `event_refresh`, which holds *all* the shared, sequence-independent DUSt3R/lifecycle tuning).

Merge precedence (highest wins): **the file's own keys > preset > inherited chain**. So a per-sequence config-04 YAML shrinks to `inherit_from` + `preset: event_refresh` + a few scene-specific overrides (dataset path/calibration, `mapping_itr_num`). Only sequence-independent (unitless/structural) params belong in a preset; scale-sensitive per-scene knobs stay in YAML — though for TUM the scale-sync mechanisms let the same DUSt3R values work across fr1/fr2/fr3. `update_recursive` does a deep merge, so partial nested overrides are fine.

Datasets are dispatched by `Dataset.type` in `utils/dataset.py::load_dataset` (`tum`/`replica`/`euroc`/`realsense`/`kitti`/`waymo`/`stray_scanner`); sensor behavior keys off `Dataset.sensor_type` (`monocular`/`stereo`/`depth`).
