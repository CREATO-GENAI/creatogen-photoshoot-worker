# CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-venv python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Faster HF downloads across restarts
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=$HF_HOME
ENV HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1

# Optional: allow passing a token at deploy time
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Create app dir
WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app.py /app/app.py

# Expose FastAPI
EXPOSE 8000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
