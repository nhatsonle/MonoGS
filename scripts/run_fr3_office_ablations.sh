#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/mono/tum/ablations/fr3_office_00_monogs.yaml"
  "configs/mono/tum/ablations/fr3_office_01_monogs_lifecycle.yaml"
  "configs/mono/tum/ablations/fr3_office_02_dust3r_pointmap_no_scale.yaml"
  "configs/mono/tum/ablations/fr3_office_03_dust3r_pointmap_scaled.yaml"
  "configs/mono/tum/ablations/fr3_office_04_dust3r_event_refresh.yaml"
  "configs/mono/tum/ablations/fr3_office_05_full_system.yaml"
)

LOG_DIR="results/ablation_logs/fr3_office_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  log_path="${LOG_DIR}/${name}.log"
  echo "==> Running ${name}"
  python slam.py --config "${cfg}" 2>&1 | tee "${log_path}"
done

echo "Logs saved to ${LOG_DIR}"
