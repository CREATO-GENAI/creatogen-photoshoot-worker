# syntax=docker/dockerfile:1.7

ARG CACHE_BUSTER=2025-09-05-1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000

# ---- system deps (single layer, cleaned) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps (use BuildKit cache for wheels) ----
# If you don't have CUDA on the target host, keep the +cu121 wheels as you do.
RUN python -m pip install --upgrade pip

# Install PyTorch + CUDA 12.1 runtime first (separate to maximize cache hits)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
      --index-url https://download.pytorch.org/whl/cu121

# Copy only requirements first for better layer caching
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---- app code ----
COPY app.py .

# (Optional) non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
  python - <<'PY' || exit 1
import socket,sys
s=socket.socket(); 
try:
    s.connect(("127.0.0.1", int(sys.argv[1] if len(sys.argv)>1 else 3000))); 
    s.close(); 
    print("ok")
except Exception as e:
    print(e); sys.exit(1)
PY

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "${UVICORN_HOST}", "--port", "${UVICORN_PORT}"]
