# Jetson pilot evidence — 25 November 2026

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
| Translation bilingual reference consistency | 3/3 outputs match visible English after casefold; no semantic-quality rating |
| Source-PTS-to-publication age | pooled p50 123.68 ms, p95 143.44 ms; 763 frames, 3 × 20 s |

## Scope and limitations

The user confirmed the three Assamese/shared-script labels and exclusion of Latin and tiny/partial background text. Boxes and evaluation rectangles were manually proposed by the assistant. Predictions with at least half their area inside these rectangles are scored; IoU ≥ 0.5 and one-to-one greedy matching apply. This excludes a spurious prediction on Latin text from the scoped accuracy score. Results do not establish whole-image multilingual accuracy, script discrimination, independent held-out performance, or robustness. The training/test overlap is unknown.

Each photo is scored once. Repeated timing frames are correlated and are not additional accuracy samples. Ground-truth crop transcription and exact-text spotting are distinct measurements. The source photos and reviewed manifest are in `experiments/real_scenes_v1`.

Translation references are the visible English MOHANBARI and TIBET KITCHEN. These three proper-name/loanword matches do not establish general meaning preservation. The fourth latency source, a spurious OCR string, is retained in the raw event log. The translation pilot includes lazy loading and drains outstanding translations after replay; generation latency is not time from capture to English availability. English overlays were excluded.

Camera timing used an actual CSI stream with zero detected regions, no translation, and a timestamped experimental transport. It measures GStreamer source-buffer PTS to shared output JPEG publication, not calibrated physical exposure or sensor-to-display age. No camera images were stored. Pooled maximum age was 932.85 ms; the first repeat includes startup. See repeat-level summaries. A text-bearing live camera test remains necessary.

The synthetic resource comparison is a separate controlled replay, not directly comparable with the real-photo translation pilot or empty-scene camera timing. VDD_IN is module power, not wall power. Provider checks do not establish per-operator GPU placement. Author/training provenance and a larger scene-disjoint evaluation remain required for publication.

## Reproduction and exports

Run the commands in `experiments/README.md`; use a fresh result directory for every device run. `scripts/finalize_pilot_report.py` copies completed raw exports and regenerates `completed_metrics.json` plus this report. JSON/CSV/raw telemetry and resource plots are published alongside this file. `SHA256SUMS.json` records hashes of the published evidence files. Regression validation is recorded separately in `regression-tests.log`.
