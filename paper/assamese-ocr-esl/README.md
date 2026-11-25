# Embedded Assamese scene-text recognition and English translation on Jetson

This letter presents the embedded execution extension of **Design of a Natural Scene Assamese Text Recognition System**, by Monor Enghi, Anjan Kumar Talukdar, and Navajit Saikia. The authors confirmed the same author list and affiliations for this extension on 25 November 2026.

The preceding manuscript was supplied at `D:/GU_PHD_2024/Conference Paper/reviewed/Camera_Ready_Final/Manuscript.pdf`. According to the author, it was accepted at the 11th National Conference on Computer Vision, Pattern Recognition, Image Processing and Graphics, 22–24 July 2026, LNMIIT Jaipur, and its proceedings are not yet published. The letter cites it as an accepted manuscript, without a fabricated DOI, publisher, or page range. The unpublished source PDF is not copied into this repository.

## Relationship to the accepted work

| Aspect | Accepted conference manuscript | This embedded extension |
| --- | --- | --- |
| Main focus | Model design, training, recognition evaluation, and translation interface | Jetson execution, camera/inference coordination, publication age, memory and module power |
| Detection | YOLO11n | YOLO11n ONNX metadata verified; CUDA execution measured |
| Recognition | Described as twelve-layer VGG/transformer/CTC, reported 64 × 200 inputs, 159 classes | Deployed export inspected: eight blocks and 143 output symbols; runtime uses 128 × 32 crops |
| Translation | deep-translator interface, with gTTS speech output | Local CPU INT8 IndicTrans2 English translation, asynchronous worker and exact-source cache |
| Evidence | Conference model-development results | Fixed synthetic replay, three scoped real-image words, translation events, timestamped CSI test, translation-enabled live resource observation |

The letter makes an embedded integration contribution. It does not claim the original model design as a new contribution, transfer the preceding manuscript's accuracy scores to the deployed recognizer, or treat the four generated scenes as an accuracy dataset. The recognition-configuration distinction is explicit in the manuscript. `model_inspection.json` records model SHA-256 hashes, metadata, interface shapes, and backbone block indices inspected directly from the ONNX graphs.

## Manuscript and build

`main.tex` is the editable source. `IEEEtran.cls` is the unchanged class from the user's supplied IEEE template. The delivery PDF is `manuscript.pdf`. Run twice from this directory:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
Copy-Item build/main.pdf manuscript.pdf
```

Compilation uses TikZ/PGF and the supplied graphics; BibTeX is not needed. Build files and rendered previews stay in the ignored `build/` directory. The final letter must remain within four pages including references. Local pdfLaTeX uses the installed MiKTeX settings/cache; a failed app preview is separate from the saved delivery PDF.

## Figures

- Figure 1: native TikZ source `figures/architecture.tex`; the Loopback TCP label is offset from its arrow.
- Figure 2: `../../experiments/real_scenes_v1/images/mohanbari.jpg` and `figures/shop_signs_test1.jpg`. The second image is the user's supplied crop from `D:/Download/FCN8/test1.jpg`, copied unchanged. The experiment scores refer to the original full photographs and the three confirmed words, not a fresh evaluation of the presentation crop.
- Figure 3: `figures/live_resources.pdf`, with PNG/SVG exports and provenance JSON. Regenerate using `scripts/plot_live_resources.py` from the raw live observation. The CPU/GPU legend sits above the utilization panel.

## Measured evidence

The bibliography contains 13 references, including six additional primary sources verified on 25 November 2026. Citations are ordered by first appearance:

- SynthTIGER: training-data context for the accepted conference work; [authors' citation](https://github.com/clovaai/synthtiger#citation). The four new replay scenes were AI-generated, not SynthTIGER outputs.
- PARSeq: context-aware scene-text recognition; [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf) and [author-supplied metadata](https://github.com/baudm/parseq#citation).
- ICDAR 2015 Robust Reading Competition: separation of localization, crop recognition, and end-to-end spotting; [authors' paper](https://publication.imlab.jp/db/publication_data/1469/C02.pdf). The local three-word rectangle protocol is not presented as a full ICDAR evaluation.
- Jacob et al., CVPR 2018: background on efficient low-precision inference; [primary paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf). This does not claim that the translation export uses the paper's quantization-aware training procedure.
- Yates et al., IEEE JSAC 2021: information-freshness context for timestamped updates; [authors' paper](https://user.eng.umd.edu/~ulukus/papers/journal/aoi-survey.pdf). The receiver-slot bound and measured source-PTS boundary remain explicitly defined for this application.
- NVIDIA Tegrastats: telemetry definitions; [official guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html). Actual board/software versions and raw output are retained in the experiment records.

`experiments/evidence-20261125` preserves the controlled OCR replay, real-image functional check, translation events and source-PTS timing. `experiments/evidence-live-20261125` preserves the viewer-on live resource observations. Hash inventories accompany both records.

The principal translation-enabled observation processed 496 frames over a 59.28 s first-to-last status interval (8.367 frames/s), with 1651.92 MiB peak sampled Python-worker RSS and 9.18 W mean module power. All 60 status snapshots contained text and an English output; the existing cache was retained. These are correlated observations of a displayed photograph, not 60 independent accuracy examples or a sustained uncached translation-generation benchmark.

Controlled OCR replay used four synthetic images and three 60 s repeats per condition, following 30 s warmups. Full-size CUDA throughput was 4.601 ± 0.011 frames/s versus 0.904 ± 0.003 on CPU, giving a 5.09-fold throughput ratio. The live observation and synthetic replay differ in input, scene density, and cache conditions and are not a controlled speed comparison.

The Jetson container passed 57 regression tests. Raw status samples, stage records, predictions and telemetry provide the evidence boundaries. The letter retains a focused discussion of generalization, translation meaning preservation, and the horizontal-text assumption instead of a prospective experiment checklist. Before submission, confirm the export/checkpoint lineage and any funding acknowledgement, update the prior-work citation when proceedings metadata becomes available, and review the journal's current requirements for an extension of conference work.
