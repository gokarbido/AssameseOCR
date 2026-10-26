# Assamese OCR

Live Assamese text recognition from a Jetson CSI camera, with an annotated
browser feed, recognized text, FPS, and pause/resume.

## Run on the Jetson

The default Docker configuration receives CSI frames from a host-side sender.
The host uses its working NVIDIA camera stack; the container runs OCR without
loading GStreamer camera plugins.

Stop the old container first:

```bash
docker compose down
```

Terminal 1 — start the CSI camera sender and leave it running:

```bash
bash scripts/stream_csi_camera.sh
```

Terminal 2 — build and start OCR:

```bash
docker compose -f docker-compose.yml up -d --build --force-recreate
docker compose -f docker-compose.yml logs -f
```

Open **http://JETSON_IP:8000**. Expected logs:

```text
Capture mode: host JPEG; source: tcpjpeg://127.0.0.1:8554
Host JPEG camera connected ... capture does not use GStreamer
```

The sender uses `nvarguscamerasrc sensor-id=0` at 1920×1080, 30 FPS, converts
to JPEG, and serves on host loopback port 8554. The container uses host
networking and decodes JPEG directly, keeping the latest frame for OCR.
No X11 session or Argus mount is required by the container. The host needs
GStreamer `jpegenc` and `tcpserversink`. Stop other camera applications first.

If the sender stops, restart it, then run `docker compose restart`. Health
checks report missing frames but do not automatically restart the worker.
`docker-compose.host-camera.yml` remains an alias for the default setup.

Optional settings:

- `SENSOR_ID=1 bash scripts/stream_csi_camera.sh` selects another CSI sensor.
- Set `CAMERA_STREAM_PORT` in both terminals to change port 8554.
- Set `WEB_PORT` before starting Compose to change the browser port.

The image uses the supplied **JetPack 6 / L4T r36.4.0 / Python 3.10** base and
GPU ONNX Runtime wheel. Camera capture on the host avoids plugin mismatches;
it does not establish GPU compatibility with JetPack 7 / CUDA 13.

## Verify inference

```bash
docker compose run --rm assamese-ocr python app.py --smoke-test
```

Both models must pass with `CUDAExecutionProvider`. `/healthz` is healthy only
while frames are arriving. Pause affects only the current browser view.
The viewer is intended for a few viewers on a trusted network; add an
authenticated proxy before exposing it externally.

## Direct CSI capture (optional)

`docker-compose.csi.yml` contains the direct Argus/GStreamer configuration.
Use it only when host and container camera libraries match:

```bash
docker compose down
# Stop the host camera sender before using direct CSI capture.
docker compose -f docker-compose.csi.yml up -d --build
```

This mounts `/tmp/argus_socket` and uses OpenCV's GStreamer backend. Missing
`g_once_init_enter_pointer` / `g_once_init_leave_pointer` symbols indicate
incompatible plugin/GLib libraries. Use the default host sender setup in that
case. `CAMERA_SOURCE` can override the direct CSI pipeline.

## Native Python

Install system OpenCV and fonts, then use the Python version supported by
JetPack:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv fonts-lohit-beng-assamese fonts-noto-core
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-jetson.txt
python -m pip install /absolute/path/to/compatible-onnxruntime-gpu.whl
python app.py --smoke-test
python web_app.py --camera tcpjpeg://127.0.0.1:8554
```

The GPU wheel must match Python, ARM64, CUDA, cuDNN, and Orin. The optional
`scripts/build_onnxruntime_jetson.sh` targets native Python 3.12 / CUDA 13 and
requires development headers; it has not been verified on hardware.

For native USB capture, run `python web_app.py --camera 0`. For CPU use in a
separate environment, install `requirements.txt` and `onnxruntime`, then add
`--device cpu`. `python web_app.py --help` lists camera and stream options.
`python app.py --camera 0` opens the desktop viewer instead.

References: [NVIDIA CSI cameras](https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera),
[NVIDIA Docker setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_docker.html).
