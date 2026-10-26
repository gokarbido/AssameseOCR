# Assamese OCR — native Python on Jetson Orin Nano

Runs the included ONNX text detector (`det_asm.onnx`) and Assamese recognizer
(`model.onnx`) locally. Docker and PyTorch are not required. CUDA is the default
for both models; startup fails if CUDA cannot be activated. CPU execution must
be selected explicitly with `--device cpu`. ONNX Runtime can still place
unsupported or shape-related graph operations on CPU.

## Your device and compatibility

The reported `R39, REVISION: 2.1` identifies JetPack 7.2.1 / L4T 39.2.1.
Your installed `libcudnn9-cuda-13` is a CUDA 13 cuDNN runtime. The missing
`/usr/local/cuda/bin/nvcc` indicates an absent toolkit at that path; check for
versioned directories before building. Python 3.12.3 is suitable for the app.

The reference edge-agent Dockerfile targets a different environment: JetPack 6,
L4T 36.4, CUDA 12.6, Python 3.10. Its `jp6/cu126` / `cp310` wheel cannot be used
with your Python 3.12 / CUDA 13 setup. A GPU wheel must match Python (`cp312`),
Linux ARM64 (`aarch64`), CUDA/cuDNN, and Orin's GPU (`sm_87`). Merely listing
`CUDAExecutionProvider` does not guarantee the wheel contains working Orin kernels.

## 1. Install system dependencies

Run these commands on the Jetson, in the repository directory:

```bash
sudo apt update
sudo apt install -y python3.12-dev python3.12-venv python3-pip python3-opencv \
  libgl1 libglib2.0-0 libgomp1 libopenblas-dev \
  fonts-lohit-beng-assamese fonts-noto-core \
  build-essential git pkg-config ca-certificates \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav v4l-utils
```

Use your configured NVIDIA repositories for **R39.2.1** to install the JetPack
compute/development components, including CUDA compiler and cuDNN headers:

```bash
sudo apt install nvidia-jetpack
ls -d /usr/local/cuda*
/usr/local/cuda/bin/nvcc --version
dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|libcudnn'
```

If `nvidia-jetpack` has no installation candidate, fix the NVIDIA apt repository
configuration for the installed BSP using NVIDIA's guide below. Do not add an
R36 / JetPack 6 repository. If the compiler lives at a versioned path, use, for
example, `export CUDA_HOME=/usr/local/cuda-13.2` before building. Runtime cuDNN
alone does not provide development headers.

## 2. Create a Python 3.12 environment

```bash
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-jetson.txt
python -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
```

`--system-site-packages` exposes Ubuntu's Python 3.12 OpenCV, including its
GStreamer integration. Check for `GStreamer: YES` for CSI cameras. Do not install
PyPI OpenCV over it for that use. An apt OpenCV built for another Python minor
version cannot be reused by Python 3.12.

For a USB camera on a machine without usable system OpenCV, install
`python -m pip install -r requirements.txt` instead. That installs PyPI OpenCV
with desktop display support, but does not supply Jetson CSI/GStreamer support.
Neither requirements file installs ONNX Runtime: install exactly one compatible
runtime in the next step to avoid replacing the GPU build with a CPU package.

## 3. Install GPU ONNX Runtime

If you already have a trusted **Python 3.12 / ARM64 / CUDA 13 / cuDNN 9 / Orin**
wheel, install it inside the activated venv:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
python -m pip install /absolute/path/to/your-compatible-gpu-wheel.whl
```

Otherwise, build natively on the Jetson using the supplied script:

```bash
source .venv/bin/activate
bash scripts/build_onnxruntime_jetson.sh
```

The script defaults to ONNX Runtime 1.27.0 with CUDA support, Python 3.12, and
Orin architecture 87; TensorRT is not required. It uses one build job to limit
memory pressure. A native build can take hours and needs ample free disk and
RAM/swap. Build output is retained under `~/.cache/assamese-ocr`. Change
`BUILD_JOBS`, `ORT_BUILD_ROOT`, `CUDA_HOME`, or `CUDNN_HOME` through environment
variables if needed. This build recipe has not been executed on your hardware;
the smoke test below is required before treating the installation as working.

## 4. Verify and run

```bash
python -m pip check
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
python app.py --smoke-test
python app.py --camera 0
```

The smoke test executes **both models** on synthetic input without a camera,
GUI, or network request. Check that both session logs include
`CUDAExecutionProvider` and the test passes. Missing libraries, unavailable CUDA,
or kernel failures produce a nonzero exit code. To observe GPU activity while
running the camera, use `tegrastats` in another terminal.

The camera viewer needs a desktop session. Press `q` to quit, `s` to save a
frame, or `r` to reset the FPS counter. Model/config defaults resolve relative
to `app.py`, so it can also be launched from another working directory.

For a CSI camera, use GStreamer-enabled system OpenCV:

```bash
python app.py --gstreamer --camera 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
```

USB camera troubleshooting: `v4l2-ctl --list-devices`, then try `--camera 1`.
You can also supply a video filename or stream URL to `--camera`.

OCR works offline. Optional online English translation is enabled with:

```bash
python -m pip install 'deep-translator==1.11.4'
python app.py --camera 0 --translate
```

Translation sends recognized text to Google's translation service and can add
network latency. Assamese display uses the installed Lohit/Noto fonts.

For explicit CPU debugging only:

```bash
python -m pip install onnxruntime  # only in a separate CPU environment
python app.py --device cpu --smoke-test
python app.py --device cpu --camera 0
```

## References

- [NVIDIA JetPack SDK setup and R39.2.1 identification](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/latest/setup_jetpack.html)
- [JetPack 7.2.1 components](https://developer.nvidia.com/embedded/jetpack/downloads)
- [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime native CUDA/Jetson build instructions](https://onnxruntime.ai/docs/build/eps.html)
