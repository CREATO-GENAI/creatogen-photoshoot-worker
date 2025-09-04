# Dockerfile
FROM runpod/pytorch:2.1.2-py3.10-cuda12.1

WORKDIR /app
COPY requirements.txt .

# keep wheels out of the image layer + faster installs
ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

RUN pip install -U pip && \
    pip install -r requirements.txt

COPY app.py .

# run your serverless worker / app
CMD ["python", "-u", "app.py"]
