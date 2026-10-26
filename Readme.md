# Assamese OCR

Live Assamese text detection and recognition with an annotated browser feed,
recognized text, FPS, and pause/resume. The camera is attached to the Jetson;
no desktop or X11 session is required by the viewer.

## CSI camera: use the working host camera stack

If `nvarguscamerasrc` works on the host but container plugins report GLib
symbol errors, use the host to capture frames and the container for OCR.

Stop the existing service:

```bash
docker compose down
```

In one host terminal, start the local JPEG sender and leave it running:

```bash
bash scripts/stream_csi_camera.sh
```

In another host terminal:

```bash
docker compose -f docker-compose.host-camera.yml up -d --build --force-recreate
docker compose -f docker-compose.host-camera.yml logs -f
```

Open **http://JETSON_IP:8000**. This mode uses host networking to receive JPEG
frames from `127.0.0.1:8554`. The container uses a Python TCP receiver and JPEG
decoding, without OpenCV VideoCapture, GStreamer, or the Argus socket. It keeps
the newest frame instead of queuing old frames during slower OCR. The host
needs GStreamer `jpegenc` and `tcpserversink`.
Stop other camera capture commands before starting the sender. If the sender
stops, restart it and then restart the OCR service.

Confirm the active capture path in logs: `Host JPEG camera connected ...;
capture does not use GStreamer`. If plugin-scanner warnings still appear,
check that this Compose file was selected and the image was rebuilt. Direct
CSI capture with the default Compose file still requires compatible plugins.

Optional settings: `SENSOR_ID=1 bash scripts/stream_csi_camera.sh` selects another
sensor. Set `CAMERA_STREAM_PORT` in both host terminals to change the sender
port; `WEB_PORT` selects the browser port. The sender captures 1920×1080 at
30 FPS and converts to JPEG before transmission; OCR throughput depends on
the device. Pause affects only the current browser view.

## Direct CSI capture in Docker

The default `docker-compose.yml` uses `nvarguscamerasrc sensor-id=0`, captures
1920×1080 at 30 FPS, converts BGRx to BGR, and reads frames through OpenCV's
GStreamer backend. It mounts `/tmp/argus_socket` to reach the host Argus daemon.
This mode requires compatible host and container camera libraries.

```bash
docker compose -f docker-compose.host-camera.yml down
docker compose config --quiet
docker compose up -d --build --force-recreate
docker compose logs -f
```

The host Argus daemon must already be running and its socket must exist.
Set `CAMERA_SOURCE` in `.env` to replace the complete pipeline. Keep
`CAMERA_GSTREAMER=1`. GStreamer errors mentioning `g_once_init_enter_pointer`
or `g_once_init_leave_pointer` indicate incompatible GLib/plugin libraries;
use the host-stream mode above or align the host and container stack.

The supplied image targets **JetPack 6 / L4T r36.4.0 / Python 3.10** with a
matching GPU ONNX Runtime wheel. It does not target JetPack 7 / CUDA 13.
The host-stream workaround avoids camera plugin dependencies; it does not
establish full GPU/driver compatibility across JetPack versions.

## Checks and other camera sources

```bash
docker compose run --rm assamese-ocr python app.py --smoke-test
```

This executes both models without a camera window. Confirm that it passes and
both sessions include `CUDAExecutionProvider`. `/healthz` is healthy only while
frames are arriving. A health failure does not automatically restart a service;
restart it after correcting the camera problem.

For a USB camera in Docker, replace the Argus `volumes` block with:

```yaml
devices:
  - /dev/video0:/dev/video0
```

Then set `CAMERA_GSTREAMER=0` and `CAMERA_SOURCE=0` in `.env` and recreate the
service. Linux USB capture uses V4L2, initially MJPG 640×480 at 30 FPS, with
fallback modes. Run `v4l2-ctl --device=/dev/video0 --list-formats-ext` to check
supported formats. Native USB settings can be selected with `--width`,
`--height`, `--fps`, and `--fourcc`. For a video file or RTSP URL, use
`CAMERA_GSTREAMER=0` and set `CAMERA_SOURCE` to its container path or URL.
Remove Argus mounts for these sources and mount video files as needed.

The viewer has no authentication. Use it on a trusted network or behind an
authenticated proxy. It is sized for a few simultaneous viewers, sharing one
camera/inference worker.

## Native Python

Use the system Python supported by your JetPack installation:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv fonts-lohit-beng-assamese fonts-noto-core
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install /absolute/path/to/compatible-onnxruntime-gpu.whl
python app.py --smoke-test
python web_app.py
```

The GPU wheel must match Python, ARM64, CUDA, cuDNN, and Orin. For native
Python 3.12 / CUDA 13, `scripts/build_onnxruntime_jetson.sh` provides an optional
source build requiring development headers; it is not verified on hardware.
For CPU use, install `requirements.txt` and `onnxruntime` in a separate
environment, then run `python web_app.py --device cpu`.

`python web_app.py --help` lists camera options. `python app.py --camera 0`
opens the desktop viewer instead. Optional desktop translation requires
`deep-translator==1.11.4` and `--translate`; it sends text to Google.

References: [NVIDIA CSI containers](https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera),
[NVIDIA Docker setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_docker.html).
