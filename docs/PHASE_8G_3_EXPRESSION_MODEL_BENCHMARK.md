# Phase 8G.3 — OpenCV Zoo expression-model benchmark

## Scope and provenance

Fetched `origin/qt-qml-export-phase8d` before editing; it matched local HEAD
`1adb92c80eeaef8f2e5395980f87d88988e4cd3a`. No keeper scoring, face
analysis, grouping, or production model path changed. This benchmark is local
and experimental. Source checkpoint files are pinned to OpenCV Zoo commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7` and are downloaded by
`app.tools.acquire_expression_models` into an ignored experimental directory.
The expected hashes and sizes come from the three pinned Git LFS pointers.

| Variant | Bytes | SHA-256 | ONNX input | Status |
| --- | ---: | --- | --- | --- |
| FP32 | 4,791,892 | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` | `data` [1,3,112,112] FP32 | Measured |
| INT8 | 1,364,007 | `f0d7093aff10e2638c734c5f18a6a7eabd2b9239b20bdb9b8090865a6f69a1ed` | `data` [1,3,112,112] FP32 | Measured |
| INT8-BQ | 1,376,702 | `0c3eaf9d0a7d442c0aa3beb3234243e1cdff9ad8871fb3cec346e90874caf57d` | `data` plus required `fc.weight` [7,512] FP32 | Unsupported unmodified input signature |

All output tensors are `label` [1,7] FP32. The FP32 graph uses ONNX opset 9
with Add, Conv, Flatten, Gemm, PRelu. Standard INT8 uses quantized Conv/MatMul
operators and ONNX opset 11 plus Microsoft domains. INT8-BQ uses opset 21,
DequantizeLinear and an externally exposed `fc.weight` input. The source is
the [official OpenCV Zoo directory](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition).
The directory [README](https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/README.md)
states Apache 2.0 covers **all files in the directory**, including these
distributed weights and preprocessing code. That is confirmed source wording;
training-data permissions and model contributor provenance for commercial
redistribution remain unresolved. OpenCV reports 88.27% on RAF-DB, which is
not a Room36 event-photo accuracy result. No checkpoint is bundled.

The block-quantized file cannot be compared fairly as published: it requires
an additional `fc.weight` feed absent from the official example. A separate
diagnostic feed of a zero tensor failed on CPU in ONNX Runtime's
`DequantizeLinear` block-quantization kernel; DirectML accepted that artificial
feed, but its output has no validated meaning. The benchmark therefore records
the original signature as unsupported and does not report BQ predictions or
latency. A repaired model would need its own provenance and numerical-parity
validation before comparison.

## Official preprocessing and sample

The benchmark follows [OpenCV Zoo's Python example](https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/facial_fer_model.py):
five SCRFD landmark points are mapped by nonreflective least-squares similarity
to the official 112×112 template; the BGR source is warped with OpenCV's
default `warpAffine`, converted to RGB, divided by 255, then normalized as
`(x-.5)/.5` and arranged NCHW. Class order is angry, disgust, fearful, happy,
neutral, sad, surprised. Outputs are treated as probabilities if nonnegative
and summing to one; otherwise stable softmax is applied. A deterministic
identity-transform test and shape/channel tests pass. There was no official
image-plus-probability fixture, so end-to-end numeric parity with OpenCV DNN
is not claimed.

The NSC audit CSV and read-only `image_face_analysis_faces` cache supplied
bboxes and five-point landmarks. SCRFD and ArcFace were **not** rerun. Seed
3603 selected 400 detections, at most two per photograph: 118 small, 119
angled, 44 low-light, and 119 other. All 400 aligned crops were usable on
this run. The categories are derived from stored area, yaw and luminance;
glasses and occlusion have no reliable labels, and some visual review crops
may be unrecognizable false detections. Contact sheets must be reviewed before
any accuracy claim. The source photos were read but never modified.

## Runtime and numerical results

Windows CPU and AMD DirectML providers were available. Each supported
model/provider ran sequentially on the same 400 crops with five warmups and
three timed repetitions (1,200 timed calls). Every ONNX model has fixed batch
dimension 1; batch sizes 4/8/16/32 were **not** claimed as supported. ORT
profiling recorded 1,206 FP32/INT8 inference nodes on DmlExecutionProvider
for each DirectML run; this does not measure GPU memory.

| Variant/provider | Init ms | First ms | Median ms | P95 ms | Faces/s | Session RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 CPU | 12.0 | 2.28 | 1.742 | 1.965 | 578 | 16.2 |
| FP32 DirectML | 136.3 | 2.83 | 0.280 | 0.323 | 3,491 | 62.2 |
| INT8 CPU | 14.7 | 2.08 | 1.440 | 1.702 | 673 | 1.7 |
| INT8 DirectML | 238.2 | 3.88 | 1.776 | 1.843 | 562 | 9.1 |
| INT8-BQ CPU/DirectML | Session loads | — | — | — | — | Unusable extra input |

CPU/DirectML peak **process** RSS ranged from 245 to 318 MiB across
sequential runs; these peaks are order-dependent and should not be read as
independent model memory costs. GPU memory could not be measured here.
Reading source images and preparing all 400 crops took 28.20 seconds,
separate from the inference table. There were 400 crop preparations and no
additional face-detector or embedding calls. Session creation and first-call
costs are reported separately from warm inference.

INT8 top-1 agreement with FP32 was 97.0% on CPU and 96.8% on DirectML;
top-2 set agreement was 79.0% and 79.2%. Mean probability absolute difference
was about 0.0165. On CPU, top-1 agreement was 98.3% for the 118 small-face
samples, 95.8% for 119 angled samples, 100% for 44 low-light samples, and
95.8% for 119 other samples. FP32 CPU/DirectML top-1 labels agreed on all
400 samples; INT8 CPU/DirectML agreed on 99.5%. These are **model agreement**
and provider consistency results, not expression accuracy. Twenty FP32 CPU
predictions had top probability under 0.5. No human expression labels exist.

## Integration estimate and recommendation

At the observed 4.88 detected faces per NSC photograph, inference-only time
estimated from measured mean throughput is approximately 1.7/0.28 seconds
for 200 photos, 8.4/1.40 seconds for 1,000, 42.2/7.0 seconds for 5,000,
and 14.6/2.4 seconds for 8,453 faces (FP32 CPU/DirectML respectively).
These exclude decoding, alignment, detection, session initialization and
cache I/O. A future implementation could reuse existing face crops and provider
configuration, cache outputs by model hash and crop version, and use bounded
batch-1 inference. None of that is installed in the production scan.

**Recommendation:** FP32 is the only variant worth considering for a later
controlled integration experiment on this machine. DirectML FP32 was much
faster than INT8 DirectML, and INT8 CPU saved little latency while changing
some predictions. INT8-BQ cannot be fairly inferred from the official file
without an additional weight tensor and separate validated transformation.
Do not integrate any variant into keeper scoring until commercial provenance
and in-domain human-labeled reliability are resolved. Expression labels are
not photographic-quality scores.

## Reproduce and review

```powershell
.\.venv\Scripts\python.exe -B -m app.tools.acquire_expression_models --output audit_results/expression_benchmark/models
.\.venv\Scripts\python.exe -B -m app.tools.benchmark_expression_models --sample-size 400 --seed 3603 --providers cpu directml --repetitions 3 --output audit_results/expression_benchmark/full
```

The local output contains `model_comparison.csv`, `prediction_comparison.csv`,
`latency_results.csv`, `memory_results.csv`, `expression_predictions.csv`,
`benchmark_summary.json`, eight review sheets for 64 faces, and a human-label
template. Review sheets, model binaries, per-face predictions and the labeling
template are ignored by Git because they may identify people. The template
supports visible expression, ambiguity, unusable crop and uncertainty; it has
no filled labels. Temporal consistency across near-duplicates was not computed
because reliable cross-photo subject matching was not established.
Lighting perturbation, crop-translation sensitivity, partial occlusion and
glasses-specific behavior also remain unmeasured because the cached audit has
no verified labels for those conditions. The review sheets expose disagreement
examples for manual inspection but cannot substitute for a labeled test set.
The local Windows CI-equivalent unittest command passed: 558 tests, 10 skipped,
no failures in 191.495 seconds. Hosted GitHub Actions has not run; publishing
this commit and audit artifacts requires explicit external-publication approval.
