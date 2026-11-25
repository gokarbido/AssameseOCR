"""Export the completed November 25 pilot evidence without inventing references.

Run after copying the four completed experiment directories into experiment_results.
Only JSON, CSV, logs and resource plots are published; camera frames are not saved.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values)-1)*quantile
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('experiment_results'))
    parser.add_argument('--output', type=Path, default=Path('experiments/evidence-20261125'))
    args = parser.parse_args()
    for name in ('orin-pilot-20261125', 'real-all-stages-20261125',
                 'real-reviewed-20261125', 'camera-age-20261125'):
        for source in (args.input/name).rglob('*'):
            if source.is_file() and source.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt', '.svg', '.png'):
                target = args.output/name/source.relative_to(args.input/name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    load = lambda path: json.loads((args.output/path).read_text(encoding='utf-8'))
    accuracy = load(Path('real-reviewed-20261125/accuracy.json'))
    events = load(Path('real-all-stages-20261125/measurement/translation_events.json'))
    references = [('মোহনবাৰী', 'MOHANBARI'), ('টিবেট', 'TIBET'), ('কিচেন', 'KITCHEN')]
    rows = []
    for source, reference in references:
        matches = [e for e in events if e.get('kind') == 'generation' and e.get('source') == source]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one recorded generation for {source}')
        event = matches[0]
        rows.append(dict(source=source, reference=reference, output=event['english'],
                         exact_match=normalized(event['english']) == normalized(reference),
                         generation_ms=event['elapsed_ms']))
    translation = dict(status='visible_bilingual_reference_consistency_only', samples=rows,
                       exact_matches=sum(r['exact_match'] for r in rows), reference_count=len(rows),
                       reference_provenance='Visible paired English sign text in the two user-supplied photos; Assamese labels and three-word scoring scope confirmed by user in chat.',
                       normalization='NFC, collapsed whitespace and casefold',
                       limitations='Three proper-name/loanword outputs only; not human-rated semantic quality, BLEU, or a translation corpus. Spurious OCR source |ং is excluded from quality scoring, but included in four-generation latency.')
    with (args.output/'camera-age-20261125/measurement/frames.csv').open(newline='') as stream:
        frames = list(csv.DictReader(stream))
    ages = [float(row['age_ms']) for row in frames]
    if not ages:
        raise ValueError('Missing camera timing frames')
    timing = dict(frames=len(ages), mean_ms=statistics.mean(ages), p50_ms=percentile(ages,.5),
                  p95_ms=percentile(ages,.95), max_ms=max(ages),
                  regions=sum(int(row['regions']) for row in frames),
                  scope='Three consecutive 20 s camera repeats, no warmup; zero detected regions. Instrumented GStreamer source PTS to shared JPEG publication. Translation disabled; excludes calibrated exposure and browser display.')
    report = dict(accuracy=accuracy, translation=translation, camera_age=timing)
    (args.output/'completed_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'REPORT.md').write_text(f'''# Jetson pilot evidence — 25 November 2026

| Quantity | Measured evidence |
| --- | --- |
| Detector + recognizer storage | 64.77 MiB serialized ONNX files |
| Synthetic CUDA replay | 4.601 ± 0.011 frames/s, 4 unique synthetic scenes, 3 × 60 s repeats |
| CUDA peak process RSS / mean sampled module power | 1283.75 MiB / 8.95 W; translation disabled |
| Detection precision / recall / H-mean | 1.00 / 1.00 / 1.00 on 3 scoped words in 2 photos |
| Exact-text spotting precision / recall / H-mean | 1.00 / 1.00 / 1.00 on the same 3 words |
| Ground-truth-crop CER / exact word accuracy | 0% (18 NFC code points) / 3 of 3 words |
| Real-image replay with asynchronous translation | 198 frames / 60.535 s = 3.271 frames/s, single pilot |
| Translation model load | 6243.20 ms |
| Uncached translation generation | mean 138.57 ms, p95 187.34 ms, 4 unique sources |
| Translation bilingual reference consistency | {translation['exact_matches']}/3 outputs match visible English after casefold; no semantic-quality rating |
| Source-PTS-to-publication age | pooled p50 {timing['p50_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms; {len(ages)} frames, 3 × 20 s |

## Scope and limitations

The user confirmed the three Assamese/shared-script labels and exclusion of Latin and tiny/partial background text. Boxes and evaluation rectangles were manually proposed by the assistant. Predictions with at least half their area inside these rectangles are scored; IoU ≥ 0.5 and one-to-one greedy matching apply. This excludes a spurious prediction on Latin text from the scoped accuracy score. Results do not establish whole-image multilingual accuracy, script discrimination, independent held-out performance, or robustness. The training/test overlap is unknown.

Each photo is scored once. Repeated timing frames are correlated and are not additional accuracy samples. Ground-truth crop transcription and exact-text spotting are distinct measurements. The source photos and reviewed manifest are in `experiments/real_scenes_v1`.

Translation references are the visible English MOHANBARI and TIBET KITCHEN. These three proper-name/loanword matches do not establish general meaning preservation. The fourth latency source, a spurious OCR string, is retained in the raw event log. The translation pilot includes lazy loading and drains outstanding translations after replay; generation latency is not time from capture to English availability. English overlays were excluded.

Camera timing used an actual CSI stream with zero detected regions, no translation, and a timestamped experimental transport. It measures GStreamer source-buffer PTS to shared output JPEG publication, not calibrated physical exposure or sensor-to-display age. No camera images were stored. Pooled maximum age was {timing['max_ms']:.2f} ms; the first repeat includes startup. See repeat-level summaries. A text-bearing live camera test remains necessary.

The synthetic resource comparison is a separate controlled replay, not directly comparable with the real-photo translation pilot or empty-scene camera timing. VDD_IN is module power, not wall power. Provider checks do not establish per-operator GPU placement. Author/training provenance and a larger scene-disjoint evaluation remain required for publication.

## Reproduction and exports

Run the commands in `experiments/README.md`; use a fresh result directory for every device run. `scripts/finalize_pilot_report.py` copies completed raw exports and regenerates `completed_metrics.json` plus this report. JSON/CSV/raw telemetry and resource plots are published alongside this file. `SHA256SUMS.json` records hashes of the published evidence files. Regression validation is recorded separately in `regression-tests.log`.
''',encoding='utf-8')
    hashes = {str(p.relative_to(args.output)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(timing,indent=2))


if __name__ == '__main__':
    main()
