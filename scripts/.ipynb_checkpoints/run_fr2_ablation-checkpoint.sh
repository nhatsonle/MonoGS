#!/usr/bin/env bash
# fr2_xyz leave-one-out ablation for the three proposed improvements over MonoGS:
#   1. DUSt3R depth prior        (bootstrap + event-refresh insertion)
#   2. Weighted-score event selection (when to call DUSt3R)
#   3. DUSt3R pointmap scale synchronization
#
# Runs the FULL config (all three on) plus one config per improvement removed,
# then the single-thread variant (all three on). Each run uses --eval so it runs
# headless and writes plot/stats_final.json (RMSE ATE), which is summarized at
# the end.
#
#   04   (FULL)              -> all three improvements enabled (reference)
#   04 no_dust3r_depth       -> improvement 1 removed (back to MonoGS pseudo-depth)
#   04 no_event_selection    -> improvement 2 removed (bootstrap only, no refresh)
#   04 no_pointmap_scale     -> improvement 3 removed (scale divisor = 1.0)
#   05 full_single_thread    -> all three, serialized single-thread mode
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  "configs/mono/tum/ablations/fr2_xyz_04_ablate_no_event_selection.yaml"
  "configs/mono/tum/ablations/fr2_xyz_04_ablate_no_pointmap_scale.yaml"
  "configs/mono/tum/ablations/fr2_xyz_05_full_single_thread.yaml"
)

LOG_DIR="results/ablation_logs/fr2_xyz_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  log_path="${LOG_DIR}/${name}.log"
  echo "=================================================================="
  echo ">>> Running: ${name}"
  echo "=================================================================="
  python slam.py --config "${cfg}" --eval 2>&1 | tee "${log_path}"
done

echo ""
echo "Logs saved to ${LOG_DIR}"
echo ""
echo "=== Per-run summary (latest result dir each) ==="
python3 - "${#CONFIGS[@]}" <<'PY'
import glob, json, os, sys

n = int(sys.argv[1])
runs = sorted(
    glob.glob("results/tum_rgbd_dataset_freiburg2_xyz/*/plot/stats_final.json"),
    key=os.path.getmtime,
)
if not runs:
    print("  (no stats_final.json found under results/tum_rgbd_dataset_freiburg2_xyz)")
    sys.exit(0)


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def fmt(value, spec):
    if isinstance(value, (int, float)):
        return format(value, spec)
    width = "".join(c for c in spec.split(".")[0] if c.isdigit())  # column width
    return format("n/a", ">" + width) if width else "n/a"


header = (
    f"  {'run':<24} {'RMSE':>8} {'FPS':>7} {'d3r_s':>7} {'d3r#':>5} "
    f"{'gauss':>8} {'model_MB':>9} {'maxMem_MB':>10}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for f in runs[-n:]:
    run_dir = os.path.dirname(os.path.dirname(f))  # .../<timestamp>
    stats = load(f)
    m = load(os.path.join(run_dir, "run_metrics.json"))
    print(
        f"  {os.path.basename(run_dir):<24} "
        f"{fmt(stats.get('rmse'), '8.4f')} "
        f"{fmt(m.get('fps'), '7.3f')} "
        f"{fmt(m.get('dust3r_time_s'), '7.2f')} "
        f"{fmt(m.get('dust3r_calls'), '5d')} "
        f"{fmt(m.get('gaussian_count'), '8d')} "
        f"{fmt(m.get('model_size_mb'), '9.2f')} "
        f"{fmt(m.get('max_memory_usage_mb'), '10.1f')}"
    )
PY
