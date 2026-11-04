#!/usr/bin/env bash
set -euo pipefail
# Run from the repository, after editing paths in resources.json if necessary.
ROOT="${1:-component-ablations-v1}"
GPUS="${2:-0,1,2,3}"
PER_GPU="${3:-1}"
MODE="${4:-jobs}"
GROUP_ARGS=()
case "$MODE" in group-seeds) GROUP_ARGS=(--group-seeds);; jobs) ;; *) echo "Mode must be jobs or group-seeds" >&2; exit 2;; esac
ABLATION_PYTHON="${ABLATION_PYTHON:-python}"
"$ABLATION_PYTHON" -m component_ablations audit --root "$ROOT" --seal-pretrained
"$ABLATION_PYTHON" -m component_ablations check --root "$ROOT"
"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase init --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"
"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase smoke --gpus "$GPUS" --jobs-per-gpu "$PER_GPU"
"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase train --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"
# Set ABLATION_DIAGNOSTICS=1 after binding both formal bank/probe manifests.
if [ "${ABLATION_DIAGNOSTICS:-0}" = "1" ]; then
  "$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase fixed --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"
  "$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase probe --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"
fi
"$ABLATION_PYTHON" -m component_ablations plot --root "$ROOT"
