#!/usr/bin/env bash
set -euo pipefail

# Self-heal: build the INT8 ONNX model on first start if it's not already there.
INT8_MODEL="data/onnx-bge-small-int8/onnx/model_qint8_avx512_vnni.onnx"
if [ ! -f "$INT8_MODEL" ]; then
    echo "[openrag] INT8 model not found — building once (~30s)..."
    python scripts/quantize_onnx.py
fi

# Honor PORT (Render/Railway/Fly inject this) but fall back to OPENRAG_PORT.
PORT="${PORT:-${OPENRAG_PORT:-8000}}"
HOST="${OPENRAG_HOST:-0.0.0.0}"

echo "[openrag] starting server on ${HOST}:${PORT}"
exec uvicorn openrag.app:app --host "${HOST}" --port "${PORT}"
