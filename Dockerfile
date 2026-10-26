# syntax=docker/dockerfile:1.7

# Jetson Orin Nano / JetPack 6.x (L4T r36.4.x, ARM64).
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-lohit-beng-assamese \
        fonts-noto-core \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        libgl1 libglib2.0-0 libgomp1 libopenblas0 \
        python3-opencv python3-pip python3-venv tini \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Preserve Ubuntu OpenCV's GStreamer support.
RUN python3 -m venv --system-site-packages /opt/ocr-venv
ENV PATH=/opt/ocr-venv/bin:${PATH} \
    PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /opt/assamese-ocr
COPY requirements-jetson.txt ./
RUN set -eux; \
    python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"; \
    python -m pip install --upgrade pip setuptools wheel; \
    python -m pip install -r requirements-jetson.txt; \
    python -m pip install "${ONNXRUNTIME_WHEEL_URL}"; \
    python -m pip check

COPY app.py detector.py inference_runtime.py web_app.py config.yaml det_asm.onnx model.onnx ./
COPY templates ./templates

# Actual GPU inference is checked at runtime, when the GPU is available.
RUN python -c "import app, web_app, cv2, onnxruntime as ort; print('OpenCV:', cv2.__version__); print('ONNX Runtime providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "web_app.py"]
