# Face quality V1/V2 audit export

`face_quality_v1_v2_all_values.csv` has one row for each duplicate-group
candidate under each of the four V2 preference combinations. It contains
68 rows (17 candidates × 4 combinations) and all recorded inputs used by the
V1 and V2 comparison, including per-face raw measurements, calibrated scores,
confidences, image-level blur measurements, weights, neutral priors, aggregate
scores, ranking positions, explanations, and missingness flags.

Columns are grouped by prefix:

- `v1_`: production V1 ranking tuple, position, keeper status and explanation.
- `v2_`: shadow V2 components, active weights, score, position and explanation.
- `comparison_`: winner agreement, margin, tie and uncertainty flags.
- `photo_`: photo-level assessments and the technical blur score.
- `technical_`: raw and normalized whole-image blur measurements from the
  isolated audit's SQLite cache.
- `weight_` and `neutral_prior_`: the V2 per-face calibration inputs.
- `face_00_`, `face_01_`, etc.: every recorded measurement for each detected
  face, plus whether geometry selected it as primary, its V2 prominence
  weight, and its effective per-face V2 score where applicable.

An empty cell means unavailable or inapplicable; it is not zero. The V2
preference IDs are `face_1_eyes_1`, `face_1_eyes_0`, `face_0_eyes_1`, and
`face_0_eyes_0`. `manifest.json` records the source dataset signature, commit,
row and column counts, and SHA-256 hashes of all source artifacts.

Recreate the export without rescanning photos:

```powershell
.\.venv\Scripts\python.exe -m app.tools.export_face_ql_audit_v2 `
  --source audit_results\phase8g2_release_20261103 `
  --output face_ql_audit_v2
```
