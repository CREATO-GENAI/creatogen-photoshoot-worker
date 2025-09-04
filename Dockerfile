# Dockerfile
# Uses official PyTorch image with CUDA 12.1 and cuDNN 9, includes Python 3.10
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Minimal OS deps you likely need for image/vision + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (minus torch/torchvision/xformers/triton)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Your code
COPY app.py /app/app.py

# If this is a worker, just run your script.
# If you serve FastAPI with uvicorn, change the CMD accordingly (see below).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
