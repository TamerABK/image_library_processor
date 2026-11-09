# Room36 audit CLI commands

Run these PowerShell examples from the repository root. `main.py` starts the application; the audit commands are separate modules under `app.tools`. Use the project's configured Python environment (shown here as `.venv`). Keep each audit output directory outside the photo input directory.

```powershell
$py = '.\.venv\Scripts\python.exe'
$photos = '.\room36_test_set\NSC_Meeting'
```

## Fresh face-quality and keeper audit

Analyzes the photos, produces V1/V2 comparisons, and optionally computes the existing blur measurements needed by V2. This can take a while for the full NSC set.

```powershell
& $py -B -m app.tools.audit_face_quality --input $photos --output '.\audit_results\nsc_fresh' --mode full --provider automatic --include-blur --contact-sheets
```

For a quick smoke run, add `--max-images 25`. Without `--include-blur`, V2 uses only compatible blur results already in the copied cache.

## Audit from an existing analysis cache

The source database is copied into the isolated output directory; it is not edited.

```powershell
& $py -B -m app.tools.audit_face_quality --input $photos --output '.\audit_results\nsc_cached' --mode cached --cache-source '.\image_analysis_cache.sqlite3'
```

## Regenerate rankings from a verified checkpoint

Use a new output directory. The command checks the source manifest, dataset, and model fingerprints.

```powershell
& $py -B -m app.tools.audit_face_quality --input $photos --output '.\audit_results\nsc_checkpoint_replay' --mode cached --cache-source '.\image_analysis_cache.sqlite3' --checkpoint-source '.\audit_results\nsc_fresh\image_checkpoint.jsonl'
```

Rerun an interrupted audit with its original command and output path to resume its own checkpoint.

## Supply duplicate groups or human labels

`groups.csv` needs `group_id,image_path`. `labels.csv` needs `image_a,image_b,preferred_image`; the preference may be an image path or `tie`, `equal`, or `uncertain`.

```powershell
& $py -B -m app.tools.audit_face_quality --input $photos --output '.\audit_results\nsc_labeled' --mode cached --cache-source '.\image_analysis_cache.sqlite3' --groups-csv '.\groups.csv' --labels-csv '.\labels.csv'
```

## Replay V2 scoring from an existing audit

This reads the audit CSVs and its blur cache. It compares V1, original V2, Phase 8G.2a, and current cached V2 across all four preference combinations without image decoding.

```powershell
& $py -B -m app.tools.recalibrate_v2_audit '.\audit_results\phase8g2_full_20261103' '.\audit_results\nsc_v2_replay'
```

Add `--regional` to benchmark raw-log, regional, and registered regional scoring. Only confirmed duplicate candidates are decoded. Repeating the command with the same output directory exercises the regional cache.

The same replay also exports `experimental_variants.csv` for Phase 8G.3 baseline, moment, composition, and combined variants. Unavailable components have explicit reasons; these variants do not change the default V2 or V1 winner.

Generate Phase 8G.3 contact sheets after a replay (this loads only confirmed duplicate candidates for display):

```powershell
& $py -B -m app.tools.contact_sheet_8g3 '.\audit_results\nsc_v2_replay' $photos '.\audit_results\nsc_v2_replay\contact_sheets'
```

```powershell
& $py -B -m app.tools.recalibrate_v2_audit '.\audit_results\phase8g2_full_20261103' '.\audit_results\nsc_v2_regional' --regional
```

## Export a flattened face-quality comparison

This optional export reads a completed audit and writes a face-quality comparison file.

```powershell
& $py -B -m app.tools.export_face_ql_audit_v2 --source '.\audit_results\nsc_fresh' --output '.\audit_results\nsc_face_ql'
```

## Inspect results

The main audit writes `manifest.json`, `face_metrics.csv`, `photo_metrics.csv`, `ranking_decisions.csv`, `ranking_v1_v2_comparison.csv`, and V2 feature diagnostics. The V2 replay writes `ranking_comparison.csv`, `face_diagnostics.csv`, `technical_diagnostics.csv`, and `summary.json`; `--regional` also writes `regional_diagnostics.csv`.

```powershell
Import-Csv '.\audit_results\nsc_v2_regional\ranking_comparison.csv' | Where-Object { $_.position -eq '1' } | Select-Object group_id,preference_id,technical_method,corrected_v2_winner,tie_class,uncertainty_reasons
Get-Content '.\audit_results\nsc_v2_regional\summary.json'
```

See [docs/PHASE_8G_AUDIT.md](docs/PHASE_8G_AUDIT.md) for the audit's data contract and [docs/PHASE_8G_2B_SCORING_RELIABILITY.md](docs/PHASE_8G_2B_SCORING_RELIABILITY.md) for V2 limitations.
