# Phase 8G cache stabilization

Baseline branch: `qt-qml-export-phase8d`, starting commit
`816cf043f02ba6c3d05381628e5ae5851e802949`.
Corrective implementation commit: `b319d5b6aca6453b2b98ca8986c6bfbfab9e6bff`.

## Confirmed cause

`default_face_analyzer.py::_eye_metric` rounds assessed eye probabilities and
derives a separate `MetricResult`. When neither eye is assessed it keeps raw and
quality values missing, but supplies a calibrated neutral ranking score.
Previously `face_processing/cache.py::_analysis_values` stored the eye state
and ranking score but omitted the independent metric's raw, quality and
confidence fields. `_analysis_from_row` rebuilt them from
`eye_combined_open_score` and discarded the ranking when that score was null.
The legacy JSON reader repeated the same substitution. This accounts for the
previous audit's missing neutral rankings and raw/quality mismatches; it does
not establish that every prior discrepancy had this sole cause.

## Correction and migration

Four nullable/defaulted columns now persist eye metric raw value, quality,
confidence and presence. New records round trip the actual metric, including
unknown eyes with a non-null neutral ranking. `ImageFaceAnalysisCache` version
2 makes earlier ambiguous entries a scoped cache miss. Old JSON without an
explicit independent `eyes` object migrates to an analysis-absent row; corrupt
JSON also becomes an analysis-absent row. Existing face-scan embeddings remain
stored, and a caller requiring analysis treats old face-scan rows as misses.
No project review, identity, known-person or source-photo tables are altered.

The existing duplicate selector now observes `prefer_open_eyes` and
`prioritize_primary_face_quality` by omitting only their corresponding
existing tuple criteria when disabled. The default ranking tuple is unchanged.
The four zero-valued image/pose fields remain placeholders: no equivalent
whole-image calibrated measurement is available to insert without designing
a new formula.

## Verification scope and remaining work

SQLite close/reopen tests cover independently valued eye metrics, partial and
unknown eye states, and scoped version misses. Focused keeper tests check both
preference switches. The previous 1,732-image audit has **not** been rerun;
its 5,091 and 3,074 mismatch counts are historical, not post-fix results.
An isolated 12-photo fresh/cached run compared 26 faces: `face_metrics.csv`
was byte-identical (zero field mismatches), all 12 photos hit the copied cache,
and `photo_metrics.csv` differed only in its expected `face_cache_hit` flag.
The subset had no duplicate comparison groups, so keeper-order parity was
not measured in this focused run. No record required recomputation in the
new cache; old version-1 records are deliberately recomputed when requested.
The local full suite passed 526 tests (10 skipped).
No human-labeled eye crops were supplied. The adapter's BGR 32×32 input,
normalization and `[closed, open]` class assumption were inspected, but the
class order cannot be independently validated without labeled inference.
The stage compatibility key still uses model file size and modification time;
content-identical metadata with changed model bytes is an outstanding
provenance risk. A future stage-specific key should address it without
invalidating embeddings for ranking-only changes.
