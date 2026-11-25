# Live camera resources with required English translation

The user displayed the Mohanbari test photograph in front of the Jetson CSI camera on 25 November 2026. The live viewer remained available throughout this resource recording. English translation is a required project output, runs asynchronously on CPU, and remained enabled. Existing translator/model/cache state was retained. No camera images or video were stored by the resource recorder.

## Text-bearing observation

Source: `live-mohanbari-resources-20261125-225625/`. Collection lasted 60 seconds, with 60 one-second status/RSS samples and raw 500 ms tegrastats. The first-to-last API observation span was 59.28088 seconds. The API frame counter advanced by 496, giving 8.36695 processed frames/s. This counter-based rate differs from the viewer's instantaneous reciprocal-loop-time display.

| Metric | Measured result |
| --- | --- |
| Sampled peak Python-worker RSS | 1651.92 MiB |
| Python-worker lifetime VmHWM | 1653.07 MiB; includes time before the observation |
| Mean CPU utilization across reported cores | 50.60% |
| Mean GPU utilization | 29.48% |
| Mean VDD_IN module power | 9.175 W |
| Mean reported system RAM | 4899.69 MB |
| Status snapshots containing text | 60 of 60 |
| Status snapshots containing an English output | 60 of 60 |
| Exact Mohanbari Assamese label snapshots | 59 of 60; one additional-vowel variant |
| English strings | `Mohanbari` in 59 snapshots; `mohanbari` in one |

Snapshot observations are correlated and do not establish independent frame or corpus accuracy. Translation-cache hits are legitimate normal operation, but this run does not measure sustained uncached generation, cold load, per-stage latency, or source-to-display age. All sampled English outputs match the sign's paired English name after casefolding; general semantic translation quality remains unestablished.

RSS is read from the Python `web_app.py` worker identified by Docker's process inventory; it excludes `tini`. System RAM, CPU/GPU utilization and module power describe the whole board environment, including host camera processing. Mean telemetry values include only records between the first and last status observations. The sampled RSS peak applies to that run window; VmHWM applies to the service lifetime. VDD_IN is module power, not wall-input power. Sampled utilization can miss activity between observations.

## Earlier unscripted observation

The separate `live-translation-resources-20261125-225432/` directory preserves the first observation, before the user repositioned the photo. Only 1 of 60 status snapshots contained text and none contained an English output. It is not used for the paper's text-bearing chart and is not a controlled comparator.

## Chart and reproduction

`scripts/record_live_resources.py` records resources without stopping the viewer. `scripts/plot_live_resources.py` exports the chart to `paper/assamese-ocr-esl/figures/live_resources.pdf`, `.svg`, and `.png`; accompanying `.metadata.json` records source hashes and measurement scope. The manuscript uses the vector PDF. `SHA256SUMS.json` hashes the records in this evidence directory. The OCR-only synthetic resource comparison remains a diagnostic ablation, distinct from this translation-enabled live observation.
