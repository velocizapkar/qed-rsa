#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/qed-nano}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

echo "Starting SGLang server..."
echo "  Model:        $MODEL_PATH"
echo "  Port:         $PORT"
echo "  Mem fraction: $MEM_FRACTION"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp 1 \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION"
