#!/usr/bin/env python3
"""Run isolated Docker OCR comparisons and timestamped host telemetry on Jetson."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def command(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=20)
    return dict(command=args, returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def read(path):
    try:
        return Path(path).read_text().replace('\x00', '').strip()
    except OSError:
        return None


def parse_tegrastats(line):
    result = {}
    gpu = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    power = re.search(r'\bVDD_IN\s+(\d+)(?:mW)?/', line)
    temperatures = re.findall(r'(?:CPU|GPU|cpu|gpu)@(\d+(?:\.\d+)?)C', line)
    if gpu:
        result['gpu_percent'] = int(gpu[1])
    if ram:
        result['system_ram_mb'] = int(ram[1])
    if power:
        result['module_power_w'] = int(power[1]) / 1000
    if temperatures:
        result['max_cpu_gpu_c'] = max(map(float, temperatures))
    return result


def record_telemetry(process, path):
    with path.open('w', encoding='utf-8') as stream:
        for line in process.stdout:
            stream.write(json.dumps(dict(timestamp=utc(), raw=line.strip(), **parse_tegrastats(line)), ensure_ascii=False) + '\n')
            stream.flush()


def aggregate(output, configurations):
    telemetry_path = output / 'tegrastats.jsonl'
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
    rows = []
    for name in configurations:
        folder = output / name
        if not (folder / 'summary.json').exists():
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['status'] != 'complete':
            continue
        frames = list(csv.DictReader((folder / 'frames.csv').open(encoding='utf-8')))
        windows = [(r['start_utc'], r['end_utc']) for r in summary['repeats']]
        measured = [t for t in telemetry if any(start <= t['timestamp'] <= end for start, end in windows)]
        row = dict(condition=name, repeats=len(summary['repeats']), frames=len(frames),
                   throughput_mean_fps=summary['throughput_mean_fps'], throughput_stdev_fps=summary['throughput_sample_stdev_fps'],
                   mean_regions=statistics.fmean(int(f['regions']) for f in frames),
                   recognition_errors=sum(int(f['recognition_errors']) for f in frames),
                   process_peak_rss_mib=summary['peak_process_memory'].get('VmHWM_mib'),
                   accuracy_status=summary['accuracy_status'], telemetry_samples=len(measured))
        # Repeat-level percentiles remain in summary.json; this is their arithmetic mean.
        for key in ('p50', 'p95'):
            row[f'mean_repeat_processing_{key}_ms'] = statistics.fmean(r['stages_ms']['processing_ms'][key] for r in summary['repeats'])
        for metric in ('gpu_percent', 'system_ram_mb', 'module_power_w', 'max_cpu_gpu_c'):
            values = [t[metric] for t in measured if metric in t]
            row[metric + '_sample_mean'] = statistics.fmean(values) if values else None
            row[metric + '_sample_max'] = max(values) if values else None
        rows.append(row)
    with (output / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Jetson synthetic scene-text experiment', '',
             'These are measured resource/processing results on four AI-generated scenes. They do not establish accuracy on real photographs.', '',
             '| Condition | Frames | FPS mean ± SD | Processing p50 / p95 (ms)* | Peak process RSS (MiB) | Mean GPU (%) | Mean module power (W) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(value):
        return f'{value:.2f}' if value is not None else 'Unavailable'
    for r in rows:
        lines.append(f"| {r['condition']} | {r['frames']} | {number(r['throughput_mean_fps'])} ± {number(r['throughput_stdev_fps'])} | "
                     f"{number(r['mean_repeat_processing_p50_ms'])} / {number(r['mean_repeat_processing_p95_ms'])} | "
                     f"{number(r['process_peak_rss_mib'])} | {number(r['gpu_percent_sample_mean'])} | {number(r['module_power_w_sample_mean'])} |")
    lines += ['', '*Arithmetic mean of repeat-level percentiles; individual distributions and raw frames are exported.', '',
              '- Workload: unpaced offline image replay with OCR, annotation, and JPEG encoding; translation disabled.',
              '- Processing timings exclude capture, transport and browser delay. Throughput includes benchmark output overhead.',
              '- RSS includes benchmark buffers and model runtime; tegrastats RAM is system-wide.',
              '- VDD_IN is module power, not total carrier-board/input power. Missing sensors remain unavailable.',
              '- Repeated frames from four fixed images are correlated. Report repeat variability, not dataset confidence intervals.',
              '- Prompted text is not ground truth. Accuracy remains unscored until annotations are reviewed.',
              '- Available/session execution providers do not prove every operator ran on the GPU.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='assamese-ocr:jetpack6')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/synthetic_assamese_v1/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', default='cuda:1536,cuda:960,cpu:1536')
    parser.add_argument('--warmup-seconds', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--idle-seconds', type=float, default=15)
    parser.add_argument('--pause-service', action='store_true', help='Pause a running Compose OCR service and restore it in finally.')
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.error('Run with sudo: Docker access, power metadata and host telemetry require privileges.')
    if args.warmup_seconds < 0 or args.seconds <= 0 or args.repeats < 1 or args.idle_seconds < 0:
        parser.error('Invalid measurement intervals or repeat count.')
    conditions = []
    for condition in args.conditions.split(','):
        if not re.fullmatch(r'(cuda|cpu):\d+', condition):
            parser.error('Conditions must be comma-separated device:width pairs, e.g. cuda:1536,cpu:960.')
        device, width = condition.split(':')
        conditions.append((f'{device}-width{width}', device, width))
    if len({item[0] for item in conditions}) != len(conditions):
        parser.error('Conditions must be unique.')
    manifest = args.manifest.resolve()
    if ROOT not in manifest.parents:
        parser.error('Keep the manifest and its images within the repository for a read-only Docker mount.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(started_at=utc(), system=platform.platform(), device=read('/proc/device-tree/model'),
                    l4t=read('/etc/nv_tegra_release'), cuda_version=read('/usr/local/cuda/version.json'),
                    image=command(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image]),
                    source_revision=command(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD']),
                    source_status=command(['git', '-c', f'safe.directory={ROOT}', 'status', '--short']),
                    power_mode=command(['nvpmodel', '-q']), clocks=command(['jetson_clocks', '--show']),
                    host_memory=read('/proc/meminfo'), conditions=args.conditions, pause_service=args.pause_service)
    (output / 'host_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    telemetry, thread, paused, container_name = None, None, False, None
    configurations = [condition[0] for condition in conditions]
    try:
        running = command(['docker', 'compose', 'ps', '--status', 'running', '-q', 'assamese-ocr'])
        if running['stdout']:
            if not args.pause_service:
                raise RuntimeError('Live OCR is running. Use --pause-service for an isolated experiment.')
            subprocess.run(['docker', 'compose', 'stop', 'assamese-ocr'], cwd=ROOT, check=True)
            paused = True
        if shutil.which('tegrastats'):
            telemetry = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            thread = threading.Thread(target=record_telemetry, args=(telemetry, output / 'tegrastats.jsonl'), daemon=True)
            thread.start()
        else:
            metadata['telemetry_error'] = 'tegrastats is unavailable'
        metadata['idle_start_utc'] = utc()
        time.sleep(args.idle_seconds)
        metadata['idle_end_utc'] = utc()
        for name, device, width in conditions:
            print(f'START {name} at {utc()}', flush=True)
            container_name = 'assamese-benchmark-' + datetime.now().strftime('%Y%m%d%H%M%S')
            docker = ['docker', 'run', '--rm', '--name', container_name, '--runtime', 'nvidia', '--network', 'none',
                      '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', f'{ROOT}:/workspace:ro', '-v', f'{output}:/results',
                      '--entrypoint', 'python', args.image, '-u', '/workspace/benchmark.py', '--root', '/workspace',
                      '--manifest', '/workspace/' + manifest.relative_to(ROOT).as_posix(), '--output', '/results/' + name,
                      '--device', device, '--max-width', width, '--warmup-seconds', str(args.warmup_seconds),
                      '--seconds', str(args.seconds), '--repeats', str(args.repeats)]
            with (output / (name + '.log')).open('w') as log:
                result = subprocess.run(docker, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            container_name = None
            if result.returncode:
                raise RuntimeError(f'{name} failed; see its log. Exit code {result.returncode}')
            print(f'COMPLETE {name} at {utc()}', flush=True)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'], metadata['error'] = 'failed', str(error)
        raise
    finally:
        if container_name:
            command(['docker', 'stop', container_name])
        if telemetry:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait(timeout=5)
            thread.join(timeout=5)
        metadata['finished_at'] = utc()
        if paused:
            restored = command(['docker', 'compose', 'start', 'assamese-ocr'])
            metadata['service_restoration'] = restored
        (output / 'host_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        aggregate(output, configurations)
        # Exports stay readable by the calling user even when the runner used sudo.
        uid, gid = os.environ.get('SUDO_UID'), os.environ.get('SUDO_GID')
        if uid and gid:
            for path in [output, *output.rglob('*')]:
                os.chown(path, int(uid), int(gid))
    print(f'EXPORTED {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
