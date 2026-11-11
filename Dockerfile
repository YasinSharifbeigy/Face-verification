# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    FACE_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements-container.txt ./
# InsightFace also installs CPU ORT; remove it and restore GPU ORT's shared files.
# CUDA/cuDNN are supplied by the base image, not platform-specific local DLLs.
RUN pip install --no-cache-dir -r requirements-container.txt \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.23.2

COPY download_models.py ./
COPY app.py verify.py ./
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models && chown appuser:appuser /app/models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
