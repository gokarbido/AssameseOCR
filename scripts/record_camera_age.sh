#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
output="${1:?New repository-relative output directory required}"
test ! -e "$output"
mkdir -p "$output"
running="$(docker compose ps --status running -q assamese-ocr)"
old="$(pgrep -f '^gst-launch-1.0.*nvarguscamerasrc.*port=8554' || true)"
sender=''
stopped_camera=0
name="assamese-age-$$"
cleanup() {
 docker stop "$name" >/dev/null 2>&1 || true
 if [[ -n "$sender" ]]; then kill "$sender" 2>/dev/null || true; wait "$sender" 2>/dev/null || true; fi
 if [[ "$stopped_camera" == 1 ]]; then nohup bash scripts/stream_csi_camera.sh > "$root/$output/restored-camera.log" 2>&1 < /dev/null & sleep 3; fi
 if [[ -n "$running" ]]; then docker compose start assamese-ocr; fi
}
trap cleanup EXIT
if [[ -n "$running" ]]; then docker compose stop assamese-ocr; fi
if [[ -n "$old" ]]; then
 test "$(printf '%s\n' "$old" | wc -l)" -eq 1
 kill -INT "$old"
 stopped_camera=1
 sleep 3
fi
python3 benchmark_age.py sender > "$output/sender.log" 2>&1 &
sender=$!
sleep 2
docker run --rm --name "$name" --runtime nvidia --network host \
 -e PYTHONDONTWRITEBYTECODE=1 -v "$root:/workspace:ro" -v "$root/$output:/results" \
 --entrypoint python assamese-ocr:jetpack6 /workspace/benchmark_age.py consumer --output /results/measurement --seconds 20
