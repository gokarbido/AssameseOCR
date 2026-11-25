# Jetson experiment recording

This suite measures the deployed detector/recognizer and Jetson resources on four clearly labeled AI-generated natural-scene images. The photographs, exact generation prompts, source categories, and requested text are in `synthetic_assamese_v1/`. Requested text is deliberately kept separate from reviewed annotations. These samples support a reproducible pilot/stress experiment; they do not establish recognition accuracy on real photographs.

## Run on the Jetson

Use the existing CUDA-enabled application image. A rebuild is unnecessary because the runner mounts the current source and images read-only. From the repository root:

```bash
mkdir -p experiment_results
sudo python3 scripts/record_jetson_experiment.py \
  --output experiment_results/orin-pilot-$(date +%Y%m%d-%H%M%S) \
  --pause-service
```

The default study compares CUDA at widths 1536 and 960, and CPU at width 1536, preserving aspect ratio. Each condition uses 30 seconds of warmup before each of three 60-second measured repeats. Complete cycles through the four images may slightly exceed the requested duration; actual elapsed time is recorded. Power mode, clocks and cooling are observed, not changed. Translation is disabled to isolate scene-text processing.

`--pause-service` pauses the running Compose OCR service during the measurements and starts it again in `finally`, including ordinary failures and interruption. Host camera capture can remain running, so its load remains part of the recorded system environment. The script requires root for Docker, clock queries, and tegrastats. Output ownership is returned to the user invoking sudo.

For a short infrastructure check:

```bash
sudo python3 scripts/record_jetson_experiment.py \
  --output experiment_results/orin-smoke-$(date +%Y%m%d-%H%M%S) \
  --conditions cuda:1536 --warmup-seconds 2 --seconds 5 --repeats 1 \
  --idle-seconds 2 --pause-service
```

Do not combine smoke-test numbers with the main study.

## Exported evidence

- `host_metadata.json`: actual device, L4T, host CUDA description, image ID, source revision/status, power mode, clock settings, idle window, and service restoration outcome.
- `tegrastats.jsonl`: raw timestamped host telemetry every 500 ms, plus parsed GPU utilization, system memory, CPU/GPU temperatures and VDD_IN module power when available. Absent sensors are unavailable, not zero.
- `comparison.csv`, `comparison.json`, `REPORT.md`: condition comparison and an explicit statement of scope and limitations.
- Per condition: `metadata.json`, `frames.csv`, `process_memory.csv`, `summary.json`, `predictions.json`, `accuracy.json`, and four annotated JPEG examples. Raw predictions retain boxes, detector scores, strings, and recognition failures.

Processing time covers image decode/resize, detector, reading order, sequential crop recognition, annotation, and JPEG encoding. It excludes sensor exposure, stream buffering, and browser/network delay. Wall throughput also includes the runner's output overhead. Per-crop recognition times are preserved for the first saved prediction of each scene. Stage distributions are reported for each repeat; comparison-table percentiles are arithmetic means of repeat-level percentiles. Repeat variability is reported as sample standard deviation.

Process RSS/HWM includes the runner and model runtime; tegrastats RAM is system-wide. VDD_IN denotes total module power, not total carrier-board or wall-input power. Measurements are sampled and may miss short peaks. Available/session ONNX providers do not establish per-operator GPU placement. None of the output is camera-to-display latency.

The source images are fixed and correlated across repeated replay. Report four unique synthetic scenes, never thousands of independent test images. The CPU/GPU comparison uses the same serialized models and inputs, with ONNX Runtime's normal provider-specific execution.

## Add real images and reviewed accuracy labels

Create a separate manifest/corpus for actual camera photographs. Group images by physical text instance, scene and capture sequence before train/test splitting. Record provenance, permitted use, annotation granularity, Unicode policy, scene conditions and script mixture.

For each independently reviewed sample, set `annotation_status` to `reviewed`, identify `reviewer`, and provide `regions` as a list of `{ "box": [x1, y1, x2, y2], "text": "..." }` using original-image pixel coordinates. Use an empty list for a reviewed negative image. Annotate *all* in-scope text, and define a consistent word/line granularity. Do not copy the requested image-generation text into ground truth without inspecting the visible glyphs.

Only reviewed samples are scored. Ground-truth crops are recognized independently, and exact-region accuracy and code-point CER use NFC plus collapsed whitespace. Whole-image detection uses greedy descending-IoU one-to-one matching at IoU >= 0.5. Spotting requires a matched box and exact normalized transcription. Each unique image is scored once using its first measured prediction, avoiding duplicated accuracy counts from replay. Scores on a reviewed synthetic subset must still be reported separately from real-scene accuracy.

The primary command exports a directory. After the run completes, it can be archived with `tar -czf experiment-results.tar.gz experiment_results/RUN_DIRECTORY`. Large raw results are ignored by Git; selectively publish summaries/evidence when appropriate.

On an analysis workstation with matplotlib installed, run `python scripts/plot_experiment.py experiment_results/RUN_DIRECTORY` to export a 300-DPI PNG, a vector SVG, and an IEEE-style LaTeX results table in its `figures/` directory. The plotter reads the completed raw exports and preserves their values and metric definitions.

## Completed 25 November pilots

See `evidence-20261125/REPORT.md` and `completed_metrics.json` for the exported results, with raw JSON/CSV, telemetry and hash inventory. Accuracy uses the three words confirmed by the user in `real_scenes_v1/manifest.json`. Evaluation rectangles exclude Latin and tiny/partial background text; a prediction counts when at least 50% of its area lies inside a rectangle. This is scoped accuracy, not whole-image multilingual performance. Crop CER and exact-text spotting are separate measurements.

Run reviewed accuracy independently from timed resource comparisons:

```bash
sudo docker run --rm --runtime nvidia --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -v "$PWD/experiment_results:/results" --entrypoint python \
  assamese-ocr:jetpack6 /workspace/benchmark.py \
  --manifest /workspace/experiments/real_scenes_v1/manifest.json \
  --output /results/real-reviewed-NEW --device cuda --max-width 1536 \
  --warmup-seconds 0 --seconds 1 --repeats 1
```

The short run is for accuracy only; its timing is not a resource benchmark. For the real-image translation pilot and instrumented CSI timing, use fresh output paths:

```bash
sudo bash scripts/record_all_stages.sh experiments/real_scenes_v1/manifest.json experiment_results/real-all-stages-NEW
sudo bash scripts/record_camera_age.sh experiment_results/camera-age-NEW
```

The wrappers pause and restore the existing live service. Camera timing also temporarily replaces the host CSI sender. It records frame identifiers and source-buffer PTS mapped to the host monotonic clock; it saves no camera images. PTS is not calibrated exposure time, and shared-buffer publication excludes HTTP and browser display. The recorded pilot contained zero detected text regions. Test text-bearing scenes separately before claiming recognition-load camera freshness.

Copy completed result directories to the workstation, then run `python scripts/finalize_pilot_report.py` to regenerate the November report and machine-readable export. Visible paired English sign words provide only a three-output translation consistency check; general semantic quality remains unmeasured.

## Live resources with required English translation

English translation is a required project stage. Earlier translation-disabled measurements are OCR-only diagnostic ablations. To record resources while keeping the viewer available and the translator enabled, run on the Jetson host:

```bash
sudo python3 scripts/record_live_resources.py \
  --output experiment_results/live-translation-NEW --seconds 60
```

This recorder leaves the service and camera running. It verifies that the viewer stays live and translation stays enabled, records raw tegrastats every 500 ms, and polls status plus Python-worker RSS every second. The worker is selected from Docker's process inventory so the `tini` init process is excluded. The existing model/cache state is retained; this is not a cold-start or forced-generation benchmark. API text/translation presence counts describe correlated snapshots, not accuracy samples. Throughput uses the processed-frame counter difference over the first-to-last status observation. Peak sampled RSS applies to this observation window; VmHWM applies to the worker's lifetime. System RAM and module power are board-wide. No camera images/video are saved.

After exporting a completed directory, generate the publication chart with:

```bash
python scripts/plot_live_resources.py experiment_results/live-translation-NEW \
  --output paper/assamese-ocr-esl/figures/live_resources
```

The plotter exports vector PDF/SVG and a 300-DPI PNG. Report the observed text/translation presence and cache conditions with each chart; merely enabling translation does not establish sustained generation load.
