# Phase 8G.3a — experimental expression insights

## Scope and workflow

Near Duplicates now offers **Expression insights** for the selected group. It is
off until a user clicks **Analyze expressions**. On first use, an explicit
dialog identifies the pinned OpenCV Zoo download, local processing, and the
experimental nature of the observations. The model is resolved through
`app_paths.model_path` in the same `onnx_models` resource directory as the
other Room36 ONNX models. An existing verified copy is used without download.
The existing acquisition utility checks the 4,791,892-byte FP32 checkpoint
against SHA-256
`4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9`.
Source commit: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
The OpenCV Zoo directory states Apache 2.0 covers its files; training-data
commercial permissions remain unresolved. This feature is experimental and
must not be packaged as a commercial model bundle without a separate review.

The user can review face-by-face estimated visible expressions in a scrollable
dialog. Predictions are not claims about emotional state. Scores and keeper
decisions are untouched. There is no photograph-level happiness score.

## Architecture

`face_processing/expression_insights.py` owns the versioned 5-landmark 112×112
alignment, RGB normalization, class order, verified model loading, fixed
batch-one inference, and result interpretation. The benchmark delegates those
preprocessing functions to the same implementation. ONNX Runtime prefers
DirectML when present, with sequential execution and memory patterns disabled;
CPU is used otherwise and after DirectML session-creation failure. The seven
classes are angry, disgust, fearful, happy, neutral, sad, surprised.

The selected-group controller starts one bounded background request. It uses
`ImageFaceAnalysisCache.get` with current file size and nanosecond modification
time; missing or stale detections remain unavailable. No SCRFD or ArcFace work
is initiated. Each photo is decoded at most once per analysis request, after
metadata-based memory screening. Groups over 64 photographs and decoded
images over 256 MiB are declined. Faces under 24 pixels in either bounding-box
dimension are marked unsuitable. These limits are provisional UI safeguards,
not validated recognition thresholds. Workers check cancellation between
faces and ignore results from previous sessions or groups. Ordinary duplicate
review is independent of expression failures.

Successful predictions are stored in a separate versioned SQLite table in
the unified image cache, keyed by source path, size, modification time,
bounding box, landmarks, face index, model SHA, and processing version.
Only class probabilities and status are stored; crops are discarded. Warm
analysis can avoid image decoding and model inference when all eligible faces
have valid cache entries. No photos or crops are uploaded. A model download
transfers only the pinned checkpoint from OpenCV Zoo.

## Confidence and limits

A top class probability below 0.5 is displayed as uncertain. This threshold
is a provisional presentation rule, not a calibrated accuracy statement.
Unusable or missing landmarks, stale caches, small faces, decode failures,
and inference failures have explicit unavailable states. The original 400-face
benchmark measured inference speed but had no human expression labels; consult
`PHASE_8G_3_EXPRESSION_MODEL_BENCHMARK.md` for those results. Predictions do
not feed V1, V2, duplicate grouping, removal suggestions, or export choices.

## Validation and remaining work

Focused tests cover hash rejection, output validation, seven classes, missing
landmarks, and benchmark preprocessing parity. Qt/QML review tests exercise
the modified page and existing keeper controls. With a private copy of the
NSC audit cache and the pinned local model, the service completed one photo
with one face in 0.364 s (one decode), a two-photo group with 41 detected
faces in 0.380 s (two decodes), and a 37-face group portrait in 0.308 s
(one decode). Repeating the portrait took 0.0039 s, with zero decodes and
37 cache hits. These are end-to-end worker measurements including cache
lookup, decoding, alignment, session initialization when needed, inference,
and serialization; they exclude Qt/QML rendering and are single observations,
not latency distributions. The local model loaded using DirectML.

The full Windows unittest run passed 562 tests with 10 skipped in 192.183 s after a QML
dialog sizing fix. Hosted CI, screenshots, Qt render timing, project-switch
latency, and a human-labeled evaluation are still outstanding. Interactive
network installation also needs a UI smoke test before release. These gaps
prevent a recognition-quality or production-readiness claim.
