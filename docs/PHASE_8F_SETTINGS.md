# Phase 8F settings

The durable app database stores version 1 preferences in `global_preferences` and
`project_preference_overrides`. Existing databases gain these tables through
`CREATE TABLE IF NOT EXISTS`; existing project rows and review decisions are
untouched. Overrides reference project IDs and cascade on project deletion.
Unknown and malformed values are ignored by the resolver.

The resolver uses built-in defaults, then saved global values, then project
overrides. A scan receives an immutable `AutomaticScanConfig` snapshot when it
is queued. Changes saved during processing take effect on the next scan.

Defaults: automatic decoding workers, automatic memory usage, Keep All blur
handling (the former Qt default), balanced face quality, one duplicate keeper,
open-eye preference on, primary-face-quality priority on. Automatic decoding
leaves two logical processors available where possible and is capped at four.
The warning uses physical core count from `psutil` capped by process affinity's
logical processor count. If unavailable, it explicitly falls back to available
logical processors. A worker count limits concurrent JPEG decode tasks; it is
not a CPU utilization target.

The decoding pool uses 512 MiB in automatic memory mode and 128 MiB in
conservative mode, still subject to its existing available-memory limit and
bounded prefetch queue. `ROOM36_DECODE_WORKERS` remains an explicit benchmark
override when set and takes precedence over saved worker preferences.
`ROOM36_DECODE_MEMORY_MIB` can lower the selected memory budget but cannot
raise it above the mode's cap.

Face detection and embedding compatibility keys exclude performance, blur,
face-quality preset and duplicate keeper preferences. Ranking keys include
quality, blur and keeper preferences, while duplicate matching keys remain
independent. Reanalysis preserves durable manual reviews.

Settings save without restarting scans. For analysis changes, use the existing
project reanalysis control to refresh displayed results.
