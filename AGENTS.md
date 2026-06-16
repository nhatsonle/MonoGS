# Repository Guidelines

## Project Structure & Module Organization

`slam.py` is the main entry point for MonoGS experiments. Core SLAM logic lives in `utils/`, GUI code in `gui/`, Gaussian Splatting code in `gaussian_splatting/`, and DUSt3R/CroCo integration code in `dust3r/` and `croco/`. YAML configs are under `configs/`, organized by sensor mode and dataset, for example `configs/mono/tum/fr3_office.yaml` and `configs/rgbd/replica/office0.yaml`. Dataset and experiment helpers live in `scripts/`. CUDA extension sources are in `submodules/`; avoid editing vendored code unless scoped there. Documentation belongs in `docs/`, while figures and screenshots belong in `media/`.

## Build, Test, and Development Commands

Set up dependencies from `README.md`; this project expects Python 3.10, CUDA, and NumPy 1.26.4. Build CUDA extensions after installing packages:

```bash
cd croco/models/curope && python setup.py build_ext --inplace
python -m pip install submodules/simple-knn
python -m pip install submodules/diff-gaussian-rasterization
```

Download sample data with `bash scripts/download_tum.sh`. Run a quick monocular demo with:

```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

Other common runs include `python slam.py --config configs/rgbd/replica/office0.yaml` and `python slam.py --config configs/stereo/euroc/mh02.yaml`.

## Coding Style & Naming Conventions

Python code uses 4-space indentation, 88-character lines, and double quotes as configured in `pyproject.toml`. Run `ruff check .` before submitting changes; use `ruff format .` for formatting. Prefer descriptive snake_case for functions, variables, and YAML keys. Keep config filenames tied to dataset sequence names or experiment purpose, such as `fr2_xyz.yaml` or `fr3_office_04_dust3r_event_refresh.yaml`.

## Testing Guidelines

There is no broad repository test suite at present. Validate changes by running the smallest relevant SLAM configuration and checking for startup, tracking, mapping, and GUI/runtime regressions. For config-only changes, run the exact affected config. For CUDA or dependency changes, also run the import checks in `README.md`. If adding tests, place them near the affected package or in a future `tests/` tree and name files `test_*.py`.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries such as `Refactor event selection mechanism in DUSt3R integration`. Keep commits focused on one topic. Pull requests should describe the behavioral change, list configs or scripts used for validation, link related issues or experiment notes, and include screenshots only for GUI or visualization changes. Mention required checkpoint, dataset, CUDA, or NumPy-version assumptions.

## Security & Configuration Tips

Do not commit datasets, checkpoints, generated logs, or local environment files. Keep machine-specific paths out of shared YAML configs; prefer documented defaults such as `./datasets` and `./checkpoints`.
