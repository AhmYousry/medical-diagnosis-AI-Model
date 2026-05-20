# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/root/.cache/torch \
    HF_HOME=/root/.cache/huggingface

WORKDIR /ai

# System deps (libgl needed by some image libraries; libgomp for torch)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./AI/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Pre-download model weights at build time so first request is instant
# (~30 MB, would otherwise be downloaded on first inference)
RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

COPY ./AI .

EXPOSE 5000

# Use gunicorn in production (1 worker — model is loaded into memory once,
# multiple workers would duplicate the ~150MB model)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "Model:app"]
