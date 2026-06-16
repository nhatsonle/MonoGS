<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">3RGS</h1>
  <h3 align="center">Monocular 3D Gaussian Splatting SLAM with a DUSt3R Depth Prior</h3>
</p>

<p align="center">
3RGS is a dense monocular SLAM system built on top of
<a href="https://arxiv.org/abs/2312.06741">Gaussian Splatting SLAM (MonoGS, CVPR 2024)</a>.
It keeps the MonoGS tracking, local mapping, keyframe window and bundle-adjustment
flow, and replaces the weak RGB-only geometry initialisation with a learned
<a href="https://github.com/naver/dust3r">DUSt3R</a> depth prior, so the map is
bootstrapped and refreshed from metric-consistent pointmaps instead of pseudo-depth.
</p>

# Method Overview

3RGS adds three changes on top of the monocular MonoGS baseline:

1. **DUSt3R depth prior** — frame-0 bootstrap plus event-triggered refresh
   insertion. DUSt3R is the source of new Gaussians at keyframes, instead of
   backprojecting RGB pseudo-depth.
2. **Loss/depth event-score selection** — a self-normalising score decides *when*
   to call DUSt3R, so it runs only when geometry actually needs a refresh rather
   than every frame.
3. **DUSt3R pointmap scale synchronization** — `baseline_ratio` + `pointmap_sync`
   absorb per-scene scale differences so the same tuning works across sequences.

The full system for a sequence is the config-04 family, e.g.
`configs/mono/tum/ablations/fr2_xyz_04_dust3r_event_refresh.yaml`. See
`docs/SYSTEM_IMPROVEMENTS_OVER_3RGS.md` and `docs/PROPOSED_METHOD_CONFIG04.md`
for details.

# Getting Started
## Installation
```bash
git clone https://github.com/nhatsonle/MonoGS.git --recursive
cd MonoGS
```

Set up the environment.

> **Tested with Python 3.10, CUDA 11.x/12.x.**
> This stack **must** run on NumPy 1.x. Several C-extensions (OpenCV, `numpy-quaternion`) are compiled against the NumPy 1.x ABI, and the data loaders use APIs removed in NumPy 2.0 (e.g. `np.unicode_`). NumPy 2.x will crash at runtime.

```bash
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libgles2-mesa \
    libosmesa6-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*
```

The install is split into separate steps **on purpose**. Running every package in one giant
`pip install` is fragile: if any single package fails to resolve, pip aborts the whole command and
**none** of the later packages get installed (this is why `imgviz` or `torchmetrics` can silently
go missing). A second package can also pull in NumPy 2.x and overwrite the pinned 1.26.4, leaving a
broken install with two `numpy-*.dist-info` folders. The steps below prevent both problems.

```bash
# Step 0: write a constraints file that hard-locks NumPy for EVERY later pip call.
# This is the key fix — it stops any dependency from silently upgrading NumPy to 2.x.
echo "numpy==1.26.4" > /tmp/3rgs-constraints.txt

# Step 1: install the pinned NumPy first.
pip install -c /tmp/3rgs-constraints.txt numpy==1.26.4

# Step 2: core dependencies, in small groups so a failure can't hide later packages.
#   -c .../constraints.txt           -> NumPy can never be upgraded past 1.26.4
#   --upgrade-strategy only-if-needed -> don't gratuitously bump already-satisfied deps
PIP="python -m pip install -c /tmp/3rgs-constraints.txt --upgrade-strategy only-if-needed"

$PIP scipy matplotlib pandas networkx tqdm pyyaml ninja imgviz munch rich ruff gdown
$PIP "opencv-python>=4.9" einops plyfile==0.8.1 trimesh roma
$PIP numpy-quaternion==2023.0.2
$PIP pycolmap evo lpips tensorboard wandb torchmetrics==1.4.0.post0
$PIP pyopengl pyrender glfw pyglm==2.7.1
$PIP huggingface-hub cvxpy
$PIP open3d

# Step 3: fix blinker conflict then reinstall open3d (still NumPy-locked)
pip install -c /tmp/3rgs-constraints.txt open3d --ignore-installed blinker

# Step 4: submodules
pip install -c /tmp/3rgs-constraints.txt submodules/simple-knn
pip install -c /tmp/3rgs-constraints.txt submodules/diff-gaussian-rasterization

# Step 5: VERIFY the environment is healthy before running anything.
# Expected: NumPy 1.26.4, exactly one numpy dist-info, and no missing modules.
python -c "import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__; print('numpy OK:', numpy.__version__)"
ls -d "$(python -c 'import site; print(site.getsitepackages()[0])')"/numpy-*.dist-info   # should print exactly ONE line
python -c "import imgviz, torchmetrics, pyrender, evo, lpips, open3d, cv2, quaternion; print('all imports OK')"
```

> **Notes**
> - `numpy-quaternion>=2024` requires `numpy>=1.25` and is incompatible with this stack — use `2023.0.2` as above.
> - If you ever see `AttributeError: np.unicode_ was removed` or a `ModuleNotFoundError`, your NumPy
>   was upgraded to 2.x (often leaving two `numpy-*.dist-info` folders). Fix it by uninstalling NumPy
>   **twice** (to clear both copies), then reinstalling the pin:
>   ```bash
>   pip uninstall -y numpy; pip uninstall -y numpy
>   pip install numpy==1.26.4
>   ```

Compile the CUDA kernels for RoPE (as in CroCo v2 and DUSt3R).
```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### DUSt3R checkpoint (required)

3RGS uses DUSt3R for the depth prior, so the checkpoint is **required** (not optional).
Download `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` into the `checkpoints/` folder:

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

Depending on your setup, change the pytorch/cudatoolkit versions in `environment.yml` by following [this document](https://pytorch.org/get-started/previous-versions/).

Our test setups were:
- Ubuntu 20.04: `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6`
- Ubuntu 18.04: `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3`

## Downloading Datasets
Running the following scripts will automatically download datasets to the `./datasets` folder.

### TUM-RGBD dataset
```bash
bash scripts/download_tum.sh
```

### Replica dataset
```bash
bash scripts/download_replica.sh
```

### EuRoC MAV dataset
```bash
bash scripts/download_euroc.sh
```

## Quick Demo
```bash
bash scripts/download_tum.sh
python slam.py --config configs/mono/tum/fr3_office.yaml
```
A GUI window will pop up.

## Run

### Proposed method (3RGS, monocular + DUSt3R prior)
```bash
python slam.py --config configs/mono/tum/ablations/fr2_xyz_04_dust3r_event_refresh.yaml
python slam.py --config configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml
```

Single-thread (deterministic, reproducible) variant:
```bash
python slam.py --config configs/mono/tum/ablations/fr2_xyz_05_full_single_thread.yaml
```

### Baseline / other modes
Plain monocular MonoGS baseline:
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

RGB-D:
```bash
python slam.py --config configs/rgbd/tum/fr3_office.yaml
python slam.py --config configs/rgbd/replica/office0.yaml
```

Stereo (experimental):
```bash
python slam.py --config configs/stereo/euroc/mh02.yaml
```

### Ablation studies
The leave-one-out ablation scripts run the proposed method with one improvement removed at a time:
```bash
bash scripts/run_fr2_ablation.sh
bash scripts/run_fr1_ablation.sh
```

## Live demo with Realsense
First install `pyrealsense2` inside the conda environment:
```bash
pip install pyrealsense2
```
Connect the Realsense camera to a **USB-3** port and run:
```bash
python slam.py --config configs/live/realsense.yaml
```
We tested with [Intel Realsense d455](https://www.mouser.co.uk/new/intel/intel-realsense-depth-camera-d455/). We recommend a global-shutter camera for robust tracking. Avoid aggressive camera motion, especially before the initial BA is performed.

# Evaluation
To evaluate in headless mode and log results (including rendering metrics), add `--eval`:
```bash
python slam.py --config configs/mono/tum/ablations/fr2_xyz_04_dust3r_event_refresh.yaml --eval
```
For benchmarking, disable the GUI (`use_gui=False`) to maximise GPU utilisation. To evaluate rendering quality, set `eval_rendering=True` in the config. Per-run metrics (FPS, DUSt3R cost, map size, memory) are written to `run_metrics.json` in the save directory.

# Reproducibility
Multi-process performance has some randomness due to GPU utilisation. For deterministic runs use the `single_thread` configs. We run all experiments on an RTX 4090; performance may differ on other GPUs.

# Troubleshooting

## `cameraMatrix is not a numpy array` (cv2.initUndistortRectifyMap)

**Cause:** `opencv-python<=4.8.x` is compiled against the NumPy 1.x ABI and fails silently with NumPy 2.x, reporting every array as an invalid argument.

**Fix:** upgrade OpenCV (do not downgrade NumPy — other packages may require NumPy 2.x):
```bash
pip install --upgrade "opencv-python>=4.9"
```

The safe combination for this repo is `numpy==1.26.4` + `opencv-python>=4.9`. Do **not** mix `opencv-python==4.8.x` with `numpy>=2.0`.

# Acknowledgement
This work builds directly on [Gaussian Splatting SLAM (MonoGS)](https://github.com/muskie82/MonoGS) and incorporates many open-source codes. We thank the authors of:
- [Gaussian Splatting SLAM (MonoGS)](https://github.com/muskie82/MonoGS)
- [DUSt3R](https://github.com/naver/dust3r) and [CroCo](https://github.com/naver/croco)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [SIBR_viewers](https://gitlab.inria.fr/sibr/sibr_core)
- [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)
- [Open3D](https://github.com/isl-org/Open3D)
- [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

# License
This project builds on MonoGS, which is released under **LICENSE.md**. For a list of code dependencies which are not property of the original authors, please check **Dependencies.md**.

# Citation
3RGS builds on Gaussian Splatting SLAM. If you use this code, please cite the original work:

```bibtex
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
