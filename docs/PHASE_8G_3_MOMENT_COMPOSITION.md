# Phase 8G.3: moment and composition experiments

## Status and scope

Fetched `origin/qt-qml-export-phase8d` before editing. Both fetched and local
HEAD were `026c69b4e54a8a9d3a55a7a4c8a75dcb0e50ef9e`. V1 and default
shadow V2 are unchanged. This phase currently provides explicit M unavailability
and an audit-only, geometry-based C observation. It does **not** claim a
validated moment or aesthetic-quality score.

## Expression model decision

No model is bundled or run. The [OpenCV Zoo MobileFaceNet expression model](https://huggingface.co/opencv/facial_expression_recognition)
reports seven expression classes, Apache 2.0 licensing for files in that
directory, and 88.27% accuracy on RAF-DB. This does not establish event-photo
accuracy or resolve all training-data rights. The [OpenVINO retail model](https://docs.openvino.ai/2023.3/omz_models_model_emotions_recognition_retail_0003.html)
reports five classes, 64×64 BGR input, 70.20% accuracy on a selected AffectNet
validation subset, and expects near-frontal faces of at least 64 pixels.
Its suitability for small, occluded, or angled event faces is unproven. The
model file license, training data terms, CPU/DirectML runtime, memory, and
in-domain accuracy were not verified to the adoption gate. No model weights
were downloaded, so there is no model SHA-256 or measured inference benchmark.
Architecture or repository licenses alone do not resolve weight and dataset
rights. The remaining model benchmark is required before adoption.

| Candidate | Size and memory | Input/output | Published accuracy | Rights and event suitability | CPU/DirectML/batching |
| --- | --- | --- | --- | --- | --- |
| OpenCV Zoo MobileFaceNet | Not measured | Seven expression classes; preprocessing in linked model card | 88.27% RAF-DB | Directory says Apache 2.0; training-data permissions and event-domain accuracy not established | Not measured |
| OpenVINO retail 0003 | 2.483 million parameters; resident memory not measured | 64×64 BGR; five-class softmax | 70.20% on selected AffectNet subset | Model page specifies frontal and ≥64-pixel faces; redistribution and training-data terms require further verification | Not measured |

Neither candidate has a Room36 model hash, validated small-face behavior, or
measured CPU/DirectML results. These missing measurements are reasons for
rejection at this stage, not zero-valued benchmark results.

## M: moment

`moment_observation` returns an unavailable state with explicit reasons:
`expression_model_unverified` and `temporal_quality_unvalidated`. Capture
timestamps and existing embeddings cannot establish a peak action or a
preferred expression. No neutral, serious, or smiling face is assigned an
automatic quality reward or penalty. There are no extra decodes or model calls.
Expression coverage and temporal coverage are zero until a validated signal
exists. No psychological state, demographic attribute, or action is inferred.

## C: composition

The audit-only `composition_observation` measures a narrow framing fact:
the fraction of analyzed face boxes lying within one percent of the image
edge. A central or symmetric face receives no special bonus; a wide photo is
not penalized for small faces. This is a potential clipping diagnostic, not an
aesthetic rating. It uses cached geometry and image dimensions. It requires
equal detected-face counts across a duplicate group; otherwise the entire
group's C comparison is unavailable to avoid rewarding a missed detection.
Even equal counts do not prove face correspondence, so these values are not
validated for selection. Vibe composition descriptors remain grouping inputs
and are not reused as aesthetic scores.

## Component separation and variants

| Input | F | M | C | T |
| --- | --- | --- | --- | --- |
| Face sharpness, eye state, pose, exposure | Yes | No | No | No |
| Expression or temporal action | No | Unavailable | No | No |
| Face-box edge proximity | No | No | Experimental | No |
| Global blur raw metrics | No | No | No | Yes |

The audit exports baseline, `plus_m`, `plus_c`, and `plus_m_c` variants to
`experimental_variants.csv`. Active components use provisional weights
F=.40, M=.25, C=.20, T=.15 and group-consistent renormalization. Missing M/C
never becomes zero quality. In face-off configurations, every variant stays
technical-only. No experiment is written to the production V2 ranking cache.
The default V2 formula and all existing preferences remain unchanged.

## NSC Meeting replay

The cached replay used 1,732 photo records and eight confirmed duplicate
groups, with zero image decodes and zero inference calls. Baseline results
remain available for all eight groups. M and M+C are unavailable in all
face-enabled groups. C is calculable in four of eight groups with face quality
enabled; four fail the equal-detection-count or geometry gate. Those four
experimental C scores are only diagnostics. The output is in
`audit_results/phase8g3_experimental/`. There are no human preference labels,
so no accuracy comparison or justified experimental keeper winner.

## Human labeling and remaining work

The existing audit accepts `--labels-csv` with `image_a,image_b,preferred_image`;
`preferred_image` may be one candidate, `tie`, `equal`, or `uncertain`.
For this phase, reviewers should also be offered `neither acceptable` in a
separate annotation field before any M/C adoption; the existing
`review_annotations_template.csv` has per-candidate `unusable` and `uncertain`
columns that can record this review. Contact sheets and a
representative labeled event set are still needed. The experiment does not
measure visual correspondence, subject completion, gesture peaks, or artistic
composition. CPU/DirectML model latency, cold/warm scan overhead, GPU memory,
and process RSS cannot be claimed because no expression model was adopted.
The local Windows test command passed: 552 tests, 10 skipped, no failures in
200.484 seconds. Hosted GitHub Actions was not triggered; publishing the
commit and audit artifacts requires explicit external-publication approval.
