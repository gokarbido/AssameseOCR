#!/usr/bin/env bash
# Run inside the Python 3.12 application venv, on the Orin itself.
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo 'This build targets Jetson Orin (aarch64).' >&2
    exit 1
fi
python -c 'import sys; assert sys.prefix != sys.base_prefix, "Activate the application venv first"; assert sys.version_info[:2] == (3, 12), sys.version'

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_HOME="${CUDNN_HOME:-/usr}"
ORT_VERSION="${ORT_VERSION:-1.27.0}"
BUILD_JOBS="${BUILD_JOBS:-1}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Missing ${CUDA_HOME}/bin/nvcc. Install nvidia-jetpack or set CUDA_HOME to the toolkit directory." >&2
    exit 1
fi
if [[ ! -f "${CUDNN_HOME}/include/cudnn.h" && ! -f "${CUDNN_HOME}/include/aarch64-linux-gnu/cudnn.h" ]]; then
    echo 'Missing cuDNN headers. Install the JetPack cuDNN development package or set CUDNN_HOME.' >&2
    exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

python -m pip install 'cmake>=3.28,<4' ninja setuptools wheel packaging 'numpy==1.26.4'
# Keep build trees outside the OCR repo; mktemp never overwrites an existing build.
build_parent="${ORT_BUILD_ROOT:-${HOME}/.cache/assamese-ocr}"
mkdir -p "${build_parent}"
source_dir="$(mktemp -d "${build_parent}/ort-${ORT_VERSION}.XXXXXX")"
git clone --recursive --branch "v${ORT_VERSION}" --depth 1 \
    https://github.com/microsoft/onnxruntime.git "${source_dir}"
cd "${source_dir}"
./build.sh --config Release --update --build --build_wheel --skip_tests \
    --parallel "${BUILD_JOBS}" --nvcc_threads 1 \
    --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_USE_FLASH_ATTENTION=OFF \
    onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF

python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
python -m pip install build/Linux/Release/dist/*.whl
# Avoid importing from the ONNX Runtime source directory.
cd "${HOME}"
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
echo "Wheel retained in ${source_dir}/build/Linux/Release/dist; run app.py --smoke-test from the OCR repo next."
