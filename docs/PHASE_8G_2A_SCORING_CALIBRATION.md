# Phase 8G.2a: V2 scoring correction

## Scope and provenance

Fetched `origin/qt-qml-export-phase8d` before editing. HEAD was
`ebb07eccd7f1a65464becefb191c678c50da4e56`. The checkout already had
unrelated generated-file changes; they were not part of this change. V1 keeper
selection, blur classification, face-model inference, and non-V2 caches are
unchanged. V2 remains a shadow result.

## Face evidence

`keeper_ranker_v2_adapter.extract_faces` is shared by live shadow ranking and
the CSV audit. A face requires analysis, a valid positive relative area, an
8-pixel bounding-box width and height, detector confidence at least 0.3, and
one usable eye-independent metric. There is no image-area floor. These are
provisional sanity gates, not a claim that every 8-pixel face is assessable.
Metric confidence remains separate; unavailable metrics are excluded at the
duplicate-group level through the existing common-feature policy.

Eligible areas are compared with the median detected face area. A face below
0.16 of that median is treated as background scale; other eligible faces are
primary for V2. Prominence combines bounded relative size, a modest position
factor, and detection confidence, then is bounded to 0.25–2.0. This allows
many similarly sized faces to contribute and limits one large face's weight.
The 20th-percentile and acceptable-face terms remain unweighted by design,
so an incidental detection that passes the background-scale rule can still
affect those terms. This requires human-labeled validation.

Eye-state and the eye-dependent reliability metric are excluded when the eye
preference is off. Eligibility and prominence never read those values or V1's
`global_selection_score`. With face quality off, V2 uses only technical data.

## Technical calibration

Version `raw-log-v1` uses the existing cached Laplacian, Sobel and local
contrast measurements. It applies `log1p` and divides by fixed log anchors
400, 60 and 50, respectively. Geometric Laplacian/Sobel agreement is reduced
by `exp(-abs(l-s))`; edge support is smoothly limited by contrast. The final
0–1 value combines 55% supported edge response and 45% contrast response.
There is no group min–max fitting and no clipping at the blur thresholds.
Hard clipping at the old thresholds discards sharp-photo differences; naive
per-group min–max would magnify small differences differently in every group.
Fixed-anchor logarithms retain differences while compressing extreme values.
Large raw edge energy alone cannot force a high score when contrast support
is low or the edge metrics disagree. The raw metrics, old blur `final_score`,
and `Sharp`/`Review`/`Blurry` status are preserved.

This is a provisional engineering calibration. It does not establish
perceptual accuracy. CPBD or regional sharpness would require a separate
labeled benchmark and runtime budget before production consideration.

## Score, ties, and cache

The existing 40/55 face and 15/55 technical weighting, equivalent to the
provisional 0.7273/0.2727 formula, is unchanged. Face aggregation remains
55% weighted mean, 25% lower 20th percentile, 20% acceptable fraction.
Component absence follows the existing group-level policy. `Score` now
reports exact ties, margins under 0.15 points as below measurement resolution,
and margins under 0.5 points as near ties. These are operational reporting
thresholds, not empirically validated visual thresholds. Filename ordering is
deterministic, and `quality_preference` is false below the resolution threshold.
The `keeper-v2.1-raw-log-v1` version changes only the V2 ranking cache key.

## NSC Meeting replay

The prior audit's 1,732 cached blur rows and face CSV were replayed without
image decoding or model execution. Exports are in
`audit_results/phase8g2a_replay/`: per-face eligibility/prominence and
exclusion reason, raw/calibrated technical data, and V1/original V2/corrected
V2 comparisons with margins and tie classifications. The replay includes all
four preference combinations. Seven of eight groups used both F and T with
face quality on; the remaining group used the common-feature fallback.
`IMG_2928` retained 23/23 detected faces as V2 primary, and `IMG_3439`
retained 28/28, all below the old 0.8% floor. The `IMG_3951`–`3953`,
`IMG_3955`–`3956`, and `IMG_4183`–`4184` candidates are included in the
comparison export. All eight groups had legacy Laplacian and Sobel saturation.
With both preferences on, corrected V2 changed three of eight original V2
winners; four top margins were distinct, two were near, and two were below
the reporting resolution. Four corrected winners differed from V1. The replay
took 1.47 seconds with a peak 84 MB of tracked
Python allocations on this Windows machine. This is a replay overhead
measurement, not an incremental full-scan benchmark. The source audit has no
human keeper labels, so changed winners and larger margins do not establish
better photographic accuracy.

## Validation limits

Focused tests cover the area-floor regression, group inclusion, background
scale, live/CSV parity, saturated technical metrics, noise-support behavior,
preference independence, ties and cache-version coupling. The full repository
suite uses the Windows CI command `python -B -m unittest discover -p
"test_*.py"`. Cached-versus-fresh model runs were not repeated; both paths
use the same serialized analysis fields, and direct live/CSV extraction parity
is tested. No inference or additional image decoding is added to ranking.
The Windows CI command completed locally: 542 tests, 10 skipped, zero
failures (216.742 seconds). The captured log is
`audit_results/phase8g2a_windows_suite.txt`. Hosted GitHub Actions was not
triggered because this checkout was not pushed.
