"""Measure actual OCR output through the production asynchronous translator.

Translation latency is recorded separately from the OCR critical path. No
translation quality or recognition accuracy is inferred from unlabeled images.
"""
import argparse
import json
from pathlib import Path
import threading
import time

from benchmark import (MemorySampler, fingerprint, load_manifest, run_frame,
                       utc, write_json, distribution)
from app import LiveTextDetectionApp
from offline_translation import OfflineTranslator
from translation_service import TranslationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--max-width', type=int, default=1536)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('Duration must be positive.')
    root = Path(__file__).resolve().parent
    _, samples = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    model, events, rows, predictions = None, [], [], {}
    lock = threading.Lock()

    def translate(text):
        nonlocal model
        if model is None:
            started = time.perf_counter()
            model = OfflineTranslator()
            with lock:
                events.append(dict(kind='model_load', timestamp=utc(), elapsed_ms=(time.perf_counter()-started)*1000))
        started, timestamp = time.perf_counter(), utc()
        try:
            result = model.translate(text)
        except Exception as error:
            with lock:
                events.append(dict(kind='generation', timestamp=timestamp, source=text, error=str(error),
                                   elapsed_ms=(time.perf_counter()-started)*1000))
            raise
        with lock:
            events.append(dict(kind='generation', timestamp=timestamp, source=text, english=result,
                               elapsed_ms=(time.perf_counter()-started)*1000))
        return result

    service = TranslationService(enabled=True, provider=translate)
    with MemorySampler(args.output / 'process_memory.csv') as memory:
        pipeline = LiveTextDetectionApp(str(root/'det_asm.onnx'), str(root/'model.onnx'), str(root/'config.yaml'), device='cuda')
        service.start()
        memory.phase = 'combined_measurement'
        started, start_utc = time.perf_counter(), utc()
        try:
            while time.perf_counter()-started < args.seconds:
                for sample in samples:
                    timing, prediction, encoded = run_frame(pipeline, sample, args.max_width)
                    before = time.perf_counter()
                    service.submit(prediction['texts'])
                    english, error = service.snapshot(prediction['texts'])
                    timing['translation_submit_snapshot_ms'] = (time.perf_counter()-before)*1000
                    rows.append(dict(timestamp=utc(), sample_id=sample['id'], timings_ms=timing,
                                     translation_ready=sum(bool(t) for t in english), translation_error=error))
                    if sample['id'] not in predictions:
                        predictions[sample['id']] = dict(prediction, translations_at_publication=english)
                        (args.output/(sample['id']+'.jpg')).write_bytes(encoded.tobytes())
            elapsed, end_utc = time.perf_counter()-started, utc()
            memory.phase = 'translation_drain'
            # Drain each observed source separately; the live service deliberately
            # coalesces changing scenes and can skip unseen historical requests.
            sources = list(dict.fromkeys(t.strip() for p in predictions.values() for t in p['texts'] if t.strip()))
            for source in sources:
                service.submit([source])
                deadline = time.monotonic()+30
                while time.monotonic()<deadline:
                    translated, error = service.snapshot([source])
                    if translated[0] or error:
                        break
                    time.sleep(.05)
            memory.phase = 'cache_lookup'
            cache_times = []
            for _ in range(100):
                before = time.perf_counter()
                service.snapshot(sources)
                cache_times.append((time.perf_counter()-before)*1000)
        finally:
            service.stop()
            if service.thread:
                service.thread.join(timeout=60)
        with lock:
            write_json(args.output/'translation_events.json', events)
        write_json(args.output/'frames.json', rows)
        write_json(args.output/'predictions.json', predictions)
        write_json(args.output/'summary.json', dict(start_utc=start_utc,end_utc=end_utc,
            elapsed_s=elapsed, frames=len(rows), throughput_fps=len(rows)/elapsed,
            stages_ms={key:distribution([row['timings_ms'][key] for row in rows]) for key in rows[0]['timings_ms']},
            uncached_generation_ms=distribution([e['elapsed_ms'] for e in events if e['kind']=='generation' and 'error' not in e]),
            cache_snapshot_ms=distribution(cache_times), unique_sources=len(sources),
            translations=service.snapshot(sources), manifest=fingerprint(args.manifest),
            scope='CUDA OCR with production asynchronous CPU translation. Generation drain is outside measured OCR interval. Annotation is Assamese-only; English overlay cost is excluded.',
            accuracy='Unscored: no reviewed references. Cache timings measure lookup, not generation.'))
    print('EXPORTED',args.output,flush=True)


if __name__ == '__main__':
    main()
