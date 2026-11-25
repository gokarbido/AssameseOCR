#!/usr/bin/env bash
# Run from the repository root with sudo. Restores the previously running viewer.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
manifest="${1:?Pass a repository-relative manifest}"
output="${2:?Pass a new repository-relative output directory}"
test ! -e "$output"
mkdir -p "$output"
running="$(docker compose ps --status running -q assamese-ocr)"
telemetry_pid=''
name="assamese-full-pipeline-$$"
cleanup() {
  docker stop "$name" >/dev/null 2>&1 || true
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ -n "$running" ]]; then docker compose start assamese-ocr; fi
}
trap cleanup EXIT
if [[ -n "$running" ]]; then docker compose stop assamese-ocr; fi
date --iso-8601=seconds > "$output/host.txt"
nvpmodel -q >> "$output/host.txt"
jetson_clocks --show >> "$output/host.txt"
git -c safe.directory="$root" rev-parse HEAD >> "$output/host.txt"
tegrastats --interval 500 --logfile "$root/$output/tegrastats.log" &
telemetry_pid=$!
docker run --rm --name "$name" --runtime nvidia --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$root:/workspace:ro" -v "$root/$output:/results" \
  --entrypoint python assamese-ocr:jetpack6 -u /workspace/benchmark_translation.py \
  --manifest "/workspace/$manifest" --output /results/measurement --seconds 60
