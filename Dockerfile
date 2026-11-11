ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    FACE_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

WORKDIR /app
COPY requirements-container.txt ./
# InsightFace pulls CPU ORT too; restore the GPU distribution's shared files.
RUN pip install --no-cache-dir -r requirements-container.txt \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.23.2

# Runtime libraries come from the official PyTorch CUDA 12.8 wheel index.
# No NVIDIA CUDA container image or full CUDA compiler toolkit is installed.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -c "import torch; assert torch.version.cuda == '12.8', torch.__version__"

COPY download_models.py ./
COPY app.py verify.py ./
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models && chown appuser:appuser /app/models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
