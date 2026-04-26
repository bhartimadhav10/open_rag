# --- Builder: install deps into an isolated layer ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# faiss-cpu wheels need libgomp at runtime; nothing else needed at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY openrag ./openrag

# Install runtime deps + the ONNX extras (optimum, onnxruntime, huggingface_hub)
RUN pip install --upgrade pip && \
    pip install -e ".[onnx]"


# --- Runtime: thin image, no compilers ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENRAG_HOST=0.0.0.0 \
    OPENRAG_PORT=8000 \
    HF_HOME=/app/.hf-cache

# faiss-cpu and onnxruntime need libgomp1 at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring over the installed Python packages + the source tree.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Source files that aren't part of the package
COPY scripts ./scripts
COPY static ./static
COPY examples ./examples
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Persistent state lives here (mount as a volume in prod):
#   - data/onnx-bge-small-int8/  (INT8 model, built on first start)
#   - data/index/                (your FAISS index)
#   - data/sessions.sqlite       (chat history)
RUN mkdir -p /app/data /app/.hf-cache
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
