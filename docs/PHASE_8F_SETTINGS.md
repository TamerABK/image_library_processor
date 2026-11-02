# Phase 8F settings

The durable app database stores version 1 preferences in `global_preferences` and
`project_preference_overrides`. Existing databases gain these tables through
`CREATE TABLE IF NOT EXISTS`; existing project rows and review decisions are
untouched. Overrides reference project IDs and cascade on project deletion.
Unknown and malformed values are ignored by the resolver.

The resolver uses built-in defaults, then saved global values, then project
overrides. A scan receives an immutable `AutomaticScanConfig` snapshot when it
is queued. Changes saved during processing take effect on the next scan.

Defaults: Normal processing, Keep All blur handling (the former Qt default),
balanced face quality, one duplicate keeper, open-eye preference on and
primary-face-quality priority on. With T available logical threads, Normal uses
`max(2, ceil(T * 0.33))` decoding workers. Performance uses
`max(4, ceil(T * 0.75))`. Performance is available only with at least four
known physical cores, six available logical threads and more workers than
Normal. Unknown physical topology or restricted affinity disables Performance.
A saved Performance preference remains stored but temporarily runs as Normal
when hardware eligibility is lost.

Both modes use a fixed 2,048 MiB decoder reservation ceiling. This is not
preallocated memory; the bounded pool can reduce its active budget under memory
pressure. Legacy `worker_count`, `memory_usage` and `decode_memory_budget_mib`
preference entries are ignored by the resolver and removed on the next save.
`ROOM36_DECODE_WORKERS` remains an explicit developer override. The memory
environment override applies only when `ROOM36_BENCHMARK_DECODE_LIMIT` is also
set by the isolated benchmark harness; ordinary scans use the fixed ceiling.

Face detection and embedding compatibility keys exclude performance, blur,
face-quality preset and duplicate keeper preferences. Ranking keys include
quality, blur and keeper preferences, while duplicate matching keys remain
independent. Reanalysis preserves durable manual reviews.

Settings save without restarting scans. For analysis changes, use the existing
project reanalysis control to refresh displayed results.
