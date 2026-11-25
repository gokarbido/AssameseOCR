"""Record live OCR/English-translation resources without stopping the viewer.

Run on the Jetson host with sudo. Samples API status and Python-process RSS once
per second, plus tegrastats at 500 ms. No camera images or video are stored.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

from record_jetson_experiment import parse_tegrastats


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def python_worker(container):
    found = []
    for line in command('docker', 'top', container, '-eo', 'pid,comm,args').splitlines()[1:]:
        fields = line.split(None,2)
        if len(fields)==3 and fields[1].startswith('python') and 'web_app.py' in fields[2]:
            found.append(int(fields[0]))
    if len(found) != 1:
        raise RuntimeError(f'Expected one live Python OCR worker, found {found}')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--container', default='assameseocr-assamese-ocr-1')
    parser.add_argument('--url', default='http://127.0.0.1:8000/api/status')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    pid = python_worker(args.container)
    root = Path(__file__).resolve().parents[1]
    metadata = dict(start_utc=utc(), worker_pid=pid, source_revision=command('git', '-c', f'safe.directory={root}', 'rev-parse', 'HEAD'),
                    container=args.container, image=command('docker', 'inspect', '--format', '{{.Image}}', args.container),
                    power_mode=command('nvpmodel', '-q'), requested_seconds=args.seconds,
                    protocol='Already-running viewer; translation required and enabled, existing translation cache retained. No service restart, cold-start measurement or forced cache misses. Displayed-photo camera test; scene changes are not controlled.',
                    memory_scope='RSS of Python web_app.py worker, excluding tini; sampled maximum is within this window. VmHWM is lifetime high-water mark, not window peak.',
                    excludes='No camera images/video saved. API snapshots are not per-frame accuracy samples; no source timestamp or translation generation latency measured.')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    process = subprocess.Popen(['tegrastats','--interval','500'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    telemetry = []
    def collect():
        with (args.output/'tegrastats.jsonl').open('w') as stream:
            for line in process.stdout:
                row = dict(timestamp=utc(),monotonic=time.monotonic(),raw=line.strip(),**parse_tegrastats(line))
                telemetry.append(row)
                stream.write(json.dumps(row)+'\n'); stream.flush()
    thread = threading.Thread(target=collect)
    thread.start()
    rows, start = [], time.monotonic()
    try:
        with (args.output/'samples.jsonl').open('w',encoding='utf-8') as stream:
            while time.monotonic()-start < args.seconds:
                with urllib.request.urlopen(args.url,timeout=5) as response:
                    state = json.load(response)
                if state['state'] != 'live' or not state['translation_enabled']:
                    raise RuntimeError('Live viewer and required translation must remain enabled')
                memory = {}
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        key,value,_ = line.split()
                        memory[key.rstrip(':')+'_mib'] = int(value)/1024
                row = dict(timestamp=utc(),monotonic=time.monotonic(),elapsed_s=time.monotonic()-start,**memory,**state)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                time.sleep(min(1,max(0,args.seconds-(time.monotonic()-start))))
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        thread.join(timeout=5)
    if len(rows)<2 or rows[-1]['frames']<=rows[0]['frames']:
        raise RuntimeError('Insufficient advancing live frames')
    metric = lambda key: [t[key] for t in telemetry if key in t and rows[0]['monotonic']<=t['monotonic']<=rows[-1]['monotonic']]
    summary = dict(status='complete',samples=len(rows),observed_seconds=rows[-1]['monotonic']-rows[0]['monotonic'],
                   processed_frame_delta=rows[-1]['frames']-rows[0]['frames'],
                   throughput_fps=(rows[-1]['frames']-rows[0]['frames'])/(rows[-1]['monotonic']-rows[0]['monotonic']),
                   sampled_peak_worker_rss_mib=max(r['VmRSS_mib'] for r in rows),
                   worker_lifetime_hwm_mib=max(r['VmHWM_mib'] for r in rows),
                   text_present_snapshots=sum(bool(r['texts']) for r in rows),
                   translated_snapshots=sum(any(t for t in r['translations']) for r in rows),
                   unique_texts=sorted({t for r in rows for t in r['texts']}))
    for key in ('cpu_percent','gpu_percent','module_power_w','system_ram_mb'):
        values = metric(key)
        summary[key+'_sample_mean'] = statistics.fmean(values) if values else None
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
