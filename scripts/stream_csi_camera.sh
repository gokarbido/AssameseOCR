#!/usr/bin/env bash
# Run on the Jetson host; keep this terminal open while OCR is running.
set -euo pipefail

exec gst-launch-1.0 -e \
    nvarguscamerasrc sensor-id="${SENSOR_ID:-0}" ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv ! 'video/x-raw,format=BGRx' ! \
    videoconvert ! 'video/x-raw,format=I420' ! \
    jpegenc quality=85 ! \
    tcpserversink host=127.0.0.1 port="${CAMERA_STREAM_PORT:-8554}" sync=false
