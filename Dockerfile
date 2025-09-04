# CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-venv python3-pip libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=$HF_HOME
ENV HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1

# Optional: pass your HF token at build time (or set in RunPod Secrets)
ARG HF_TOKEN=""
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# For serverless handler there’s no port to expose
CMD ["python", "app.py"]
