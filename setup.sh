#!/usr/bin/env bash
set -euo pipefail

echo "=== QED-RSA Setup ==="

# Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

# Download QED-Nano model weights
MODEL_DIR="/workspace/models/qed-nano"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "[2/3] Model already exists at $MODEL_DIR, skipping download."
else
    echo "[2/3] Downloading QED-Nano to $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download lm-provers/QED-Nano --local-dir "$MODEL_DIR"
fi

# Create results directory
echo "[3/3] Creating results directory..."
mkdir -p /workspace/results

echo "=== Setup complete ==="
echo "Start the server:  bash scripts/start_server.sh"
echo "Run experiments:    bash scripts/run.sh 01_baseline"
