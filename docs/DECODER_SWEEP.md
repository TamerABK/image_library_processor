# Decoder sweep

Run from the repository root using PyCharm's configured Python interpreter.

```powershell
& '.\.venv\Scripts\python.exe' -B -m app.tools.benchmark_decoder_sweep `
  --input '.\room36_test_set\NSC_Meeting' --mode smoke

& '.\.venv\Scripts\python.exe' -B -m app.tools.benchmark_decoder_sweep `
  --input '.\room36_test_set\NSC_Meeting' --mode comprehensive --plan-only

& '.\.venv\Scripts\python.exe' -B -m app.tools.benchmark_decoder_sweep `
  --input '.\room36_test_set\NSC_Meeting' --mode comprehensive `
  --output '.\benchmark_results\decoder_sweep_full' --allow-long-run
```

Resume an interrupted run with the same arguments, output path and `--resume`.
Use `--exhaustive` only when the full 140-pair Phase B matrix is desired.
Defaults: 100 stratified screening images, seed 36, two Phase A repetitions,
three Phase B repetitions and a 30-minute per-run timeout. The standard Phase A
matrix has 96 executions. The runner prints every Phase A configuration before
work; Phase B is selected from measured Phase A results and printed before it
starts. A full experiment can take many hours. The normal command stops before
processing unless `--allow-long-run` is supplied.

Each run starts in a new Python process and uses a fresh temporary analysis
cache, face database and review store. It never changes Room36 app preferences,
project reviews or source photos. The OS filesystem cache is left alone. The
benchmark-only `ROOM36_BENCHMARK_DECODE_LIMIT` is set only in child processes
to permit experimental worker counts beyond the production CPU cap. The child
clears `ROOM36_DECODE_WORKERS` and `ROOM36_DECODE_MEMORY_MIB` so requested
values reach the pool. Requested and effective values appear separately in
`runs.csv`; a missing pool is a failed run.

The screening subset is selected with a fixed seed across first-level folders
and file-size quartiles. Face-count strata cannot be known without first
analyzing the dataset, so the manifest records that limitation. Model hashes,
dataset metadata signature, version, machine facts, configuration and failures
are in `manifest.json`. Result signatures are compared within each dataset.
Rows with differing signatures or source metadata are marked failed. The full
JSON reports and traces stay under `raw/`; the ZIP includes CSVs, manifest and
report only.

CSV timing units are seconds and memory units are MiB. Substage inclusive and
exclusive times are summed across calls and can overlap on different threads;
they are **not** critical-path elapsed time. Unsupported telemetry is empty.
The decoder budget is a maximum reservation permission, not preallocated RAM.
Resource pressure can reduce the pool's active scheduling budget below its
configured ceiling; the current sampler does not expose that reduced value.
CPU and GPU utilization may be unavailable depending on Windows counters.
