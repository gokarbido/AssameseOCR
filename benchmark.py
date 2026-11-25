#!/usr/bin/env python3
"""Replay image scenes through the deployed OCR components and export evidence.

Processing timings exclude sensor acquisition and browser/network delay. AI prompt
text is never ground truth. Only explicitly reviewed annotations can be scored.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import threading
import time
import unicodedata

import cv2
import numpy as np


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def load_manifest(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    samples, seen = [], set()
    for sample in document['samples']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sample['id']):
            raise ValueError('Use letters, digits, underscores and hyphens in sample IDs.')
        if sample['id'] in seen:
            raise ValueError('Duplicate sample identifier.')
        seen.add(sample['id'])
        image = (path.parent / sample['image']).resolve()
        if path.parent not in image.parents or not image.is_file():
            raise ValueError('Image must exist within the manifest directory.')
        sample = dict(sample, path=image)
        if sample.get('annotation_status') == 'reviewed':
            if not sample.get('reviewer') or not isinstance(sample.get('regions'), list):
                raise ValueError('Reviewed annotations require a reviewer and regions list (empty for a negative image).')
            for region in sample['regions']:
                box = region['box']
                if len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
                    raise ValueError('Annotation boxes must contain four finite pixel coordinates.')
                if box[2] <= box[0] or box[3] <= box[1] or not isinstance(region['text'], str):
                    raise ValueError('Invalid reviewed region.')
        samples.append(sample)
    if not samples:
        raise ValueError('Manifest contains no samples.')
    return document, samples


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0


def match_boxes(reference, prediction, threshold=0.5):
    candidates = sorted(((box_iou(a, b), i, j) for i, a in enumerate(reference)
                         for j, b in enumerate(prediction)), reverse=True)
    used_reference, used_prediction, matches = set(), set(), []
    for overlap, i, j in candidates:
        if overlap >= threshold and i not in used_reference and j not in used_prediction:
            matches.append((i, j))
            used_reference.add(i)
            used_prediction.add(j)
    return matches


def distribution(values):
    if not values:
        return None
    return dict(n=len(values), mean=statistics.fmean(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), minimum=min(values), maximum=max(values))


def process_memory():
    values = {}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, *_ = line.split()
                values[key.rstrip(':') + '_mib'] = int(value) / 1024
    except OSError:
        pass
    return values


class MemorySampler:
    def __init__(self, output):
        self.output = output
        self.phase, self.repeat = 'initialization', 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.output.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['timestamp', 'phase', 'repeat', 'VmRSS_mib', 'VmHWM_mib'])
            writer.writeheader()
            while not self.stopped.is_set():
                writer.writerow(dict(timestamp=utc(), phase=self.phase, repeat=self.repeat, **process_memory()))
                stream.flush()
                self.stopped.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=3)


def load_frame(sample, max_width):
    frame = cv2.imread(str(sample['path']))
    if frame is None:
        raise ValueError(f"Cannot decode {sample['id']}")
    source_height, source_width = frame.shape[:2]
    if max_width and source_width > max_width:
        height = max(1, round(source_height * max_width / source_width))
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    return frame, (source_width, source_height)


def run_frame(pipeline, sample, max_width):
    from reading_order import sort_detections
    timings, failures = {}, []
    started = time.perf_counter()
    frame, source_size = load_frame(sample, max_width)
    after_decode = time.perf_counter()
    boxes, scores = pipeline.detector.detect_one(frame, fit_to_image=True)
    after_detect = time.perf_counter()
    boxes, scores = sort_detections(boxes, scores)
    after_order = time.perf_counter()
    texts, crop_times = [], []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        crop_start = time.perf_counter()
        try:
            if not crop.size:
                raise ValueError('Empty detected crop')
            texts.append(pipeline.recognizer.recognize(crop))
        except Exception as error:
            texts.append('')
            failures.append(str(error))
        crop_times.append((time.perf_counter() - crop_start) * 1000)
    after_recognize = time.perf_counter()
    annotated = pipeline.draw_detections(frame, boxes, scores, texts, [None] * len(texts))
    after_annotate = time.perf_counter()
    ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    finished = time.perf_counter()
    if not ok:
        raise RuntimeError('Output JPEG encoding failed.')
    timings.update(decode_resize_ms=(after_decode - started) * 1000,
                   detect_ms=(after_detect - after_decode) * 1000,
                   order_ms=(after_order - after_detect) * 1000,
                   recognize_ms=(after_recognize - after_order) * 1000,
                   annotate_ms=(after_annotate - after_recognize) * 1000,
                   encode_ms=(finished - after_annotate) * 1000,
                   processing_ms=(finished - started) * 1000)
    prediction = dict(sample_id=sample['id'], source_width=source_size[0], source_height=source_size[1],
                      input_width=frame.shape[1], input_height=frame.shape[0],
                      boxes=np.asarray(boxes).tolist(), scores=np.asarray(scores).tolist(), texts=texts,
                      recognition_errors=failures, crop_recognition_ms=crop_times)
    return timings, prediction, encoded


def evaluate_reviewed(pipeline, samples, predictions, max_width):
    reviewed = [s for s in samples if s.get('annotation_status') == 'reviewed']
    if not reviewed:
        return dict(status='not_scored', reason='No independently reviewed annotations. Prompted text is not ground truth.',
                    reviewed_images=0, unreviewed_images=len(samples), normalization='NFC and collapsed whitespace; Unicode code points')
    tp = fp = fn = spotting_tp = crop_exact = characters = edits = crop_count = 0
    for sample in reviewed:
        frame, (source_width, source_height) = load_frame(sample, max_width)
        xscale, yscale = frame.shape[1] / source_width, frame.shape[0] / source_height
        references = sample['regions']
        boxes = []
        for region in references:
            x1, y1, x2, y2 = region['box']
            if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
                raise ValueError('Reviewed annotation falls outside its image.')
            box = [round(x1 * xscale), round(y1 * yscale), round(x2 * xscale), round(y2 * yscale)]
            boxes.append(box)
            crop = frame[box[1]:box[3], box[0]:box[2]]
            if not crop.size:
                raise ValueError('Reviewed crop disappeared at the chosen input scale.')
            prediction = normalize(pipeline.recognizer.recognize(crop))
            reference = normalize(region['text'])
            crop_count += 1
            crop_exact += prediction == reference
            characters += len(reference)
            edits += edit_distance(reference, prediction)
        detected = predictions[sample['id']]
        matches = match_boxes(boxes, detected['boxes'])
        tp += len(matches)
        fp += len(detected['boxes']) - len(matches)
        fn += len(boxes) - len(matches)
        spotting_tp += sum(normalize(references[i]['text']) == normalize(detected['texts'][j]) for i, j in matches)
    def scores(correct, predicted, reference):
        precision = correct / predicted if predicted else 0.0
        recall = correct / reference if reference else 0.0
        return dict(precision=precision, recall=recall, hmean=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return dict(status='scored_reviewed_subset_only', reviewed_images=len(reviewed), unreviewed_images=len(samples) - len(reviewed),
                matching='Greedy descending IoU, one-to-one, IoU >= 0.5', normalization='NFC and collapsed whitespace; Unicode code points',
                detection=scores(tp, tp + fp, tp + fn), spotting=scores(spotting_tp, tp + fp, tp + fn),
                crop_exact_region_accuracy=crop_exact / crop_count if crop_count else None,
                crop_cer_percent=100 * edits / characters if characters else None,
                reference_regions=crop_count, reference_code_points=characters,
                note='Each reviewed source image scored once using its first measured prediction; repeated timing frames are not independent accuracy samples.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-width', type=int, default=0)
    parser.add_argument('--warmup-seconds', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_width < 0 or args.warmup_seconds < 0 or args.seconds <= 0 or args.repeats < 1:
        parser.error('Use nonnegative width/warmup, positive duration and at least one repeat.')
    args.root = args.root.resolve()
    manifest, samples = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'annotated').mkdir()
    import onnxruntime as ort
    from app import LiveTextDetectionApp
    metadata = dict(schema_version=1, started_at=utc(), device=args.device, max_width=args.max_width,
                    python=sys.version, platform=platform.platform(), opencv=cv2.__version__, onnxruntime=ort.__version__,
                    available_providers=ort.get_available_providers(), manifest_sha256=fingerprint(args.manifest),
                    samples=[dict(id=s['id'], sha256=fingerprint(s['path'])['sha256'],
                                  annotation_status=s.get('annotation_status', 'unreviewed')) for s in samples],
                    artifacts={name: fingerprint(args.root / name) for name in ('det_asm.onnx', 'model.onnx', 'config.yaml', 'benchmark.py')},
                    requested_protocol=dict(repeats=args.repeats, seconds=args.seconds, warmup_seconds=args.warmup_seconds),
                    workload='Offline unpaced image replay, detection + ordering + sequential recognition + annotation + JPEG; translation disabled',
                    timing_scope='Host wall-clock synchronous ONNX calls; includes decode/resize, excludes sensor acquisition and browser/network latency.',
                    generated_dataset=manifest.get('synthetic', False), limitations=manifest.get('limitations', []))
    write_json(args.output / 'metadata.json', metadata)
    with MemorySampler(args.output / 'process_memory.csv') as memory:
        load_start = time.perf_counter()
        pipeline = LiveTextDetectionApp(str(args.root / 'det_asm.onnx'), str(args.root / 'model.onnx'),
                                       str(args.root / 'config.yaml'), device=args.device)
        metadata['model_initialization_ms'] = (time.perf_counter() - load_start) * 1000
        metadata['session_providers'] = dict(detector=pipeline.detector._session.get_providers(), recognizer=pipeline.recognizer.session.get_providers())
        metadata['provider_note'] = 'Session registration is not proof of per-operator GPU placement.'
        write_json(args.output / 'metadata.json', metadata)
        columns = ['timestamp', 'repeat', 'sample_id', 'regions', 'recognition_errors', 'decode_resize_ms', 'detect_ms',
                   'order_ms', 'recognize_ms', 'annotate_ms', 'encode_ms', 'processing_ms']
        repeat_summaries, predictions = [], {}
        with (args.output / 'frames.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for repeat in range(1, args.repeats + 1):
                memory.phase, memory.repeat = 'warmup', repeat
                warmup_start, warmup_index = time.perf_counter(), 0
                while time.perf_counter() - warmup_start < args.warmup_seconds:
                    run_frame(pipeline, samples[warmup_index % len(samples)], args.max_width)
                    warmup_index += 1
                memory.phase = 'measurement'
                start_utc, started = utc(), time.perf_counter()
                rows, by_sample, frames = [], {s['id']: [] for s in samples}, 0
                # Complete full corpus cycles; record actual elapsed time, not the requested deadline.
                while time.perf_counter() - started < args.seconds or frames < len(samples):
                    for sample in samples:
                        timestamp = utc()
                        timings, prediction, encoded = run_frame(pipeline, sample, args.max_width)
                        writer.writerow(dict(timestamp=timestamp, repeat=repeat, sample_id=sample['id'], regions=len(prediction['boxes']),
                                             recognition_errors=len(prediction['recognition_errors']), **timings))
                        rows.append(timings)
                        by_sample[sample['id']].append(timings['processing_ms'])
                        if sample['id'] not in predictions:
                            predictions[sample['id']] = prediction
                            # Evidence writes are included in wall throughput, excluded from per-frame processing_ms.
                            (args.output / 'annotated' / (sample['id'] + '.jpg')).write_bytes(encoded.tobytes())
                        frames += 1
                    stream.flush()
                elapsed, end_utc = time.perf_counter() - started, utc()
                summary = dict(repeat=repeat, start_utc=start_utc, end_utc=end_utc, frames=frames, elapsed_s=elapsed,
                               throughput_fps=frames / elapsed,
                               stages_ms={key: distribution([row[key] for row in rows]) for key in rows[0]},
                               per_scene_processing_ms={key: distribution(values) for key, values in by_sample.items()},
                               memory_at_end=process_memory())
                repeat_summaries.append(summary)
                write_json(args.output / 'summary.json', dict(status='in_progress', repeats=repeat_summaries))
                print(f"MEASURED repeat={repeat} frames={frames} elapsed_s={elapsed:.3f} fps={frames / elapsed:.3f}", flush=True)
        memory.phase = 'accuracy'
        accuracy = evaluate_reviewed(pipeline, samples, predictions, args.max_width)
        write_json(args.output / 'accuracy.json', accuracy)
        write_json(args.output / 'predictions.json', predictions)
        throughput = [r['throughput_fps'] for r in repeat_summaries]
        write_json(args.output / 'summary.json', dict(status='complete', completed_at=utc(), device=args.device,
                   max_width=args.max_width, repeats=repeat_summaries, throughput_mean_fps=statistics.fmean(throughput),
                   throughput_sample_stdev_fps=statistics.stdev(throughput) if len(throughput) > 1 else None,
                   accuracy_status=accuracy['status'], peak_process_memory=process_memory(),
                   note='Repeat variability on four fixed synthetic scenes; not confidence intervals over natural-scene populations.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
