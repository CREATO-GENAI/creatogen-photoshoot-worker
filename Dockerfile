# CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates \
    python3 python3-venv python3-pip \
    build-essential \
    libgl1 libglib2.0-0 ffmpeg \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
# Speed up HF downloads & resume
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=$HF_HOME
ENV HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1

# Do NOT bake HF tokens in the image; use RunPod Secrets instead.

WORKDIR /app

# ---- Install CUDA wheels first (most reliable on RunPod) ----
# Pin exactly matching cu121 builds from the NVIDIA index
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 && \
    pip install xformers==0.0.27.post2

# ---- App deps (everything except torch/torchvision/xformers) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app.py /app/app.py

# Serverless handler entry
CMD ["python", "app.py"]
