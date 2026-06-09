#!/usr/bin/env bash
# fr1_desk culprit-isolation ablation.
# Runs 4 configs in increasing order and prints final RMSE ATE for each.
# Expected diagnosis:
#   00 (MonoGS)       -> should reproduce ~0.05 m  (control / sanity)
#   04A (init-only)   -> if bad, the DUSt3R bootstrap map is the culprit
#   04B (refresh-only)-> if bad (esp. mid-sequence blowup), refresh is the culprit
#   04 (full system)  -> known ~0.70 m (the regression we are explaining)
set -e
cd "$(dirname "$0")/.."

CONFIGS=(
  "configs/mono/tum/ablations/fr1_desk_00_monogs.yaml"
  "configs/mono/tum/ablations/fr1_desk_04A_init_only.yaml"
  "configs/mono/tum/ablations/fr1_desk_04B_refresh_only.yaml"
  "configs/mono/tum/ablations/fr1_desk_04_dust3r_event_refresh.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=================================================================="
  echo ">>> Running: $cfg"
  echo "=================================================================="
  python slam.py --config "$cfg"
done

echo ""
echo "=== Final RMSE ATE per run (latest result dir each) ==="
python3 - <<'PY'
import glob, json, os
runs = sorted(glob.glob("results/tum_rgbd_dataset_freiburg1_desk/*/plot/stats_final.json"),
              key=os.path.getmtime)
for f in runs[-4:]:
    rmse = json.load(open(f))["rmse"]
    print(f"  {rmse:.4f}  <-  {f}")
PY
