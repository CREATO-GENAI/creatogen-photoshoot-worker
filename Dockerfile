# Smaller base + GPU-compatible: install CUDA via PyTorch wheels, not via the base image
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps for opencv-headless, image libs, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First install torch/vision from the official CUDA 12.1 wheel index (smaller than pytorch/pytorch image)
# (Host driver must be >= CUDA 12.1, which is true on RunPod images)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# Install the rest (NO torch/torchvision/torchaudio/xformers here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py /app/app.py

# If this is a worker, just run your script.
# If you serve FastAPI with uvicorn, change the CMD accordingly (see below).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
