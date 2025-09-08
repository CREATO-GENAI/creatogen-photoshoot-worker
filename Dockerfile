# Dockerfile
ARG CACHE_BUSTER=2025-09-05-1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps (git fixes the provenance warning; ffmpeg/libgl for OpenCV/PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip once
RUN python -m pip install --upgrade pip

# PyTorch CUDA 12.1 wheels
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

# --- IMPORTANT ---
# Serverless containers must run the Python handler, not an HTTP server.
CMD ["python", "-u", "app.py"]
