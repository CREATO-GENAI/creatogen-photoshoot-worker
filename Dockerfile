# CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-venv python3-pip libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=$HF_HOME
ENV HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1

# NOTE: do NOT bake secrets at build time. Set HUGGINGFACE_HUB_TOKEN via RunPod Secrets at runtime.

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    # ensure CUDA wheels are visible when installing -r
    pip config set global.extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app.py /app/app.py

# Serverless: run handler
CMD ["python", "app.py"]
