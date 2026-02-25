#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/run.sh <experiment_name>"
    echo "Examples:"
    echo "  bash scripts/run.sh 01_baseline"
    echo "  bash scripts/run.sh 02_rsa_basic"
    echo "  bash scripts/run.sh 03_rsa_verify"
    exit 1
fi

EXPERIMENT="$1"
SCRIPT="experiments/${EXPERIMENT}.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

echo "=== Running experiment: $EXPERIMENT ==="
python "$SCRIPT"
echo "=== Experiment complete ==="
