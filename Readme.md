# Assamese OCR

Live Assamese text detection and recognition using the included ONNX models.
View the annotated camera feed and recognized text in a lightweight browser UI.
One inference worker shares the latest frame across viewers; no frontend build
or external assets are needed.

## Docker on Jetson Orin Nano

The Dockerfile uses the supplied **JetPack 6 / L4T r36.4.0 / Python 3.10** base
and GPU ONNX Runtime wheel. Use a compatible JetPack 6 host with Docker,
Compose v2, and the NVIDIA container runtime configured. This image does not
target JetPack 7 / CUDA 13: the base, Python, and GPU wheel must match.

Run on the Jetson from this directory, with a USB camera at `/dev/video0`:

```bash
docker compose config --quiet
docker compose build
docker compose run --rm assamese-ocr python app.py --smoke-test
docker compose up -d
```

Open **http://JETSON_IP:8000** in your browser, or
[localhost:8000](http://localhost:8000) on the Jetson itself. No desktop session
or X11 access is needed. The camera is attached to the Jetson, not the browser.
Pause/resume affects only the current browser view; inference continues.

```bash
docker compose logs -f
docker compose down
```

The smoke test executes both models without a camera window. Confirm that it
passes and both sessions include `CUDAExecutionProvider`. Build-time checks
verify imports and provider availability; GPU inference runs at runtime.
`/healthz` reports healthy only while frames are arriving. Check logs and
restart the service after fixing a camera or inference error.

Optional overrides in a local `.env` file:

```dotenv
CAMERA_DEVICE=/dev/video0
CAMERA_SOURCE=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
CAMERA_FOURCC=MJPG
WEB_PORT=8000
```

`CAMERA_DEVICE` is mapped to `/dev/video0` inside the container; keep
`CAMERA_SOURCE=0` when selecting another USB device. For a video file or RTSP
stream, set `CAMERA_SOURCE` to its container path or URL. Mount video files as
needed. Remove the Compose `devices` entry when no USB camera is attached,
including for a camera-free smoke test. CSI cameras additionally require host
Argus access and a GStreamer pipeline; the default Compose file targets USB.

Linux USB capture uses V4L2 directly and tests an actual frame before starting
OCR. It tries the configured mode, then YUYV 640×480 at 30 FPS, then device
defaults. Change the settings above to a mode supported by your camera:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
docker compose up -d --build --force-recreate
docker compose logs -f
```

If frames still time out, check the selected `/dev/video*` node, USB connection,
and other processes using the camera. A container health check does not
automatically restart an unhealthy service; restart it after fixing the device.
GStreamer errors mentioning `g_once_init_enter_pointer` or
`g_once_init_leave_pointer` indicate incompatible GLib/plugin libraries. Align
the host JetPack and container stack before using CSI/NVIDIA GStreamer plugins;
the V4L2 path bypasses these plugins for USB cameras.

The web viewer has no authentication. Use it on a trusted local network;
restrict access or add an authenticated proxy before exposing it externally.
Each live viewer holds a streaming connection; the server is sized for a few
simultaneous viewers.

## Native Python

Install system OpenCV and Assamese fonts, then use the system Python supported
by your JetPack installation:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv fonts-lohit-beng-assamese fonts-noto-core
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install /absolute/path/to/compatible-onnxruntime-gpu.whl
python -m pip check
python app.py --smoke-test
python web_app.py
```

The GPU wheel must match Python, ARM64, CUDA, cuDNN, and Orin. Neither requirements
file installs ONNX Runtime automatically. For native Python 3.12 / CUDA 13,
`scripts/build_onnxruntime_jetson.sh` provides an optional source build requiring
CUDA and cuDNN development headers; it has not been verified on target hardware.

Use `python web_app.py --help` for camera, port, CPU, and GStreamer options.
For a desktop camera window, run `python app.py --camera 0` instead; press `q`
to quit, `s` to save a frame, or `r` to reset FPS.

For CPU use in a separate environment, install `requirements.txt` and
`onnxruntime`, then run `python web_app.py --device cpu`.

Optional desktop translation requires `pip install deep-translator==1.11.4`
and `python app.py --translate`; it sends recognized text to Google.

References: [NVIDIA Docker setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_docker.html),
[ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[Flask streaming](https://flask.palletsprojects.com/en/stable/patterns/streaming/).
