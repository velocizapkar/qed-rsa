#!/usr/bin/env bash
set -euo pipefail

echo "=== QED-RSA Setup ==="

# Install Python dependencies — skip packages already present (e.g. RunPod
# images ship with torch + sglang pre-installed, and reinstalling torch
# can blow past the container disk limit).
echo "[1/3] Installing Python dependencies..."

install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install "$2"
}

install_if_missing sglang       "sglang[all]"
install_if_missing openai       openai
install_if_missing transformers transformers
install_if_missing huggingface_hub huggingface_hub

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
