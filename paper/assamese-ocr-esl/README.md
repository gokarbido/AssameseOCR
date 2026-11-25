# Assamese natural-scene text recognition manuscript for IEEE Embedded Systems Letters

The primary task is locating and transcribing Assamese text in natural scenes. Signboards are one example; the scope also allows posters, storefronts, labels, and other text-bearing surfaces. These are intended evaluation categories, not claims about existing training-data coverage or demonstrated accuracy. Crop-level scene text recognition and end-to-end text spotting are evaluated separately. Offline English translation is optional. Object-level grouping, script classification, and perspective rectification are not implemented; robustness to complex scenes remains to be evaluated.

`main.tex` is the editable manuscript. `IEEEtran.cls` is copied unchanged from the IEEE template already supplied in the parent paper folder. References are embedded in the source. The architecture diagram is native TikZ in `figures/architecture.tex`, included by the main document. Compilation requires TikZ/PGF; BibTeX and external raster graphics are not required.

This is a working research draft, not a submission-ready empirical paper. Measured November 25 Jetson pilots now cover synthetic resource replay, three scoped real-image words, asynchronous translation generation, and source-PTS-to-publication camera age. Raw evidence and scope limits are published in `experiments/evidence-20261125`. General accuracy and semantic translation quality remain unestablished; the camera timing scene contained zero detected regions. Author details, training provenance, scene-disjoint evaluation and controlled buffering comparisons remain required.

## Build

Open this folder in Codex's LaTeX editor, or run from this directory:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

The second pass resolves references. The delivery PDF is `manuscript.pdf`; after a rebuild, copy `build/main.pdf` over it. Intermediate files and page previews stay in the ignored `build/` folder.

The delivery PDF is built with local pdfLaTeX. Rebuild twice to resolve references, then visually inspect every rendered page. If the app preview encounters MiKTeX settings/cache access errors, open `manuscript.pdf` directly or use a terminal with access to the normal MiKTeX settings/cache.

The November 25 delivery compiled to four pages with resolved references and no overfull-box warnings. Every rendered page was visually checked. The Jetson container passed 57 tests; the retained log is in the published evidence directory. The TikZ architecture diagram remains editable in the LaTeX source.

The journal's [author instructions](https://ieee-ceda.org/publication/esl/esl-author-instructions), checked October 29, 2026, specify at most four pages including figures, tables, and references. Recheck after adding author information and results. Keep standard IEEE type size and margins.

## Complete before submission

- Replace the author line and draft footnote with names, affiliations, correspondence, and any funding acknowledgement.
- Establish detector architecture, training data ownership/provenance, split sizes, training settings actually used, export procedure, and checkpoint selection. The configuration alone is insufficient evidence.
- Verify the recognizer ONNX graph against the SVTRv2-labelled configuration; distinguish any local architectural changes from published SVTRv2.
- Supply the Jetson SKU/RAM, camera, software versions, power mode, clocks, cooling, and operator-placement profile.
- Collect natural-scene text boxes and exact Assamese transcriptions. Group repeated views by physical text instance, scene, and capture sequence before splitting to prevent leakage. Record scene category, surface, perspective, small text, lighting, blur, occlusion, multilingual content, and multiple text-bearing objects. Declare the policy for non-Assamese regions.
- Report detection precision/recall/harmonic mean, ground-truth-crop STR accuracy, detected-crop accuracy, and end-to-end spotting results with fixed normalization and matching rules. Include an STR baseline on identical crops and splits, plus multiline reading-order evaluation.
- Run the proposed FIFO/latest-frame and synchronous/asynchronous comparisons. These baselines and capture-timestamp instrumentation still require implementation.
- Replace the pending evidence rows with measured accuracy, throughput, age percentiles, translation delay/quality, peak memory, and board power, including run variability.
- Replace the local test-environment limitation with the final regression outcome after installing project dependencies. Validate actual GPU inference and neural translation separately from mocks.
- Add representative real output images and failure cases if they strengthen the evidence; keep the complete letter within the page limit.
- Update abstract, results, limitations, and conclusion together when real measurements arrive. Check novelty, bibliography metadata, and journal submission requirements again.

## Evidence and source map

Source revision inspected: `d7cdbdb1eb304aa47c52994fe49a8e1d1d157ce1`.

| Manuscript detail | Repository source |
| --- | --- |
| Host CSI sender, 1080p/30, JPEG quality 85 | `scripts/stream_csi_camera.sh` |
| Latest complete JPEG slot, sequence IDs, framing guard | `jpeg_stream.py` |
| Detector thresholds, recognition preprocessing and greedy CTC | `app.py`, `detector.py` |
| Median-row grouping and aligned score ordering | `reading_order.py` |
| Current desired strings, cache caps, retry and inter-attempt delays | `translation_service.py` |
| Local CPU translator, token limits, normalization | `offline_translation.py` |
| Shared viewer, JPEG publication, FPS and health semantics | `web_app.py` |
| Requested provider checks; no CPU-only startup fallback | `inference_runtime.py` |
| Model-family labels and training configuration only | `config.yaml` |
| Translation revision and weight verification | `scripts/download_translation_model.py` |

Translation export revision pinned by the downloader: `a7ecdc0ba73148c6e1d051ca372332471d8aff4f`.

Local verification on October 29, 2026: `python -m unittest discover -s tests -v` passed 33 tests; one additional test-module import failed because `waitress` was unavailable. This was a Windows authoring-environment run, not a Jetson benchmark. Provider tests use mocks and worker tests inject translation providers.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `det_asm.onnx` | 10,583,484 | `fdb0fb9a1e145b92d557fa1deeb7d2e7158fecac424c784b8ba08b6bd96c64ef` |
| `model.onnx` | 57,329,274 | `0e8212f22233b45a1a31b7b3ebd8cbf9e24fe0001562aa97dd65c0ed43770718` |

Their combined serialized size is 64.77 MiB (1 MiB = 1,048,576 bytes); this excludes translation assets and runtime memory.
