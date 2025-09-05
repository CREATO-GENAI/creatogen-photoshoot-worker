FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch + CUDA runtime (small, ~1.2 GB instead of 3 GB+)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of your deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app.py .

# If this is a worker, just run your script.
# If you serve FastAPI with uvicorn, change the CMD accordingly (see below).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
