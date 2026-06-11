#!/usr/bin/env bash
# Run connectome_analysis.py for every edge weight.
# Required env vars: GPICKLE_DIR, BASE_OUTPUT_DIR
# Optional env var:  BASE_CACHE_DIR (default: ./data/null_cache)
#
# Usage:
#   GPICKLE_DIR=./data/takahashi/na/cmp-v3.2.0/ \
#   BASE_OUTPUT_DIR=./results \
#   bash run_all_weights.sh [extra args passed to each run]
#
# Each weight gets its own --output-dir and --cache-dir to avoid collisions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

GPICKLE_DIR="${GPICKLE_DIR:-}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-./data/null_cache}"

if [[ -z "$GPICKLE_DIR" ]]; then
    echo "Error: GPICKLE_DIR is not set. Export it before running the script." >&2
    exit 1
fi
if [[ -z "$BASE_OUTPUT_DIR" ]]; then
    echo "Error: BASE_OUTPUT_DIR is not set. Export it before running the script." >&2
    exit 1
fi

WEIGHTS=(fiber_density fiber_number fiber_length FA)

for weight in "${WEIGHTS[@]}"; do
    echo "========================================"
    echo " Edge weight: ${weight}"
    echo "========================================"
    "$PYTHON" "$SCRIPT_DIR/code/connectome_analysis.py" \
        "$GPICKLE_DIR" \
        --edge-weight "$weight" \
        --output-dir "${BASE_OUTPUT_DIR}/${weight}" \
        --cache-dir  "${BASE_CACHE_DIR}/${weight}" \
        "$@"
    echo
done

echo "All weights complete."
