#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/mono/tum/ablations/fr3_office_00_monogs.yaml"
  "configs/mono/tum/ablations/fr3_office_01_dust3r_same_iters_no_insertion.yaml"
  "configs/mono/tum/ablations/fr3_office_02_dust3r_reduced_mapping_no_insertion.yaml"
  "configs/mono/tum/ablations/fr3_office_03_dust3r_reduced_mapping_insertion.yaml"
  "configs/mono/tum/ablations/fr3_office_04_dust3r_baseline_gated.yaml"
  "configs/mono/tum/ablations/fr3_office_05_monogs_lifecycle.yaml"
  "configs/mono/tum/ablations/fr3_office_06_dust3r_baseline_gated_lifecycle.yaml"
  "configs/mono/tum/ablations/fr3_office_07_dust3r_adaptive_optimization.yaml"
  "configs/mono/tum/ablations/fr3_office_08_dust3r_reduced_mapping_no_scale.yaml"
  "configs/mono/tum/ablations/fr3_office_09_dust3r_reduced_mapping_baseline_ratio_scale.yaml"
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
