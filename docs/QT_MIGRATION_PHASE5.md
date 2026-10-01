# Phase 5 — projects, real photo browsing, and final stabilization

## Audited baseline and scope

The audit started on `qt-qml-photo-grid-phase5` at
`5711c06b0e2a30bf595241fffdc68042456521bc`. The tracked working tree was clean.
Existing untracked datasets, local configuration, runtime files, and build outputs
were left alone. The baseline explicit UI suite ran 81 tests successfully.

The real vertical slice is:

Home → New Project → native folder picker → recursive supported-photo discovery
→ project registration → PhotoListModel population → Library → real thumbnails
→ Home → persisted project card → reopen project.

Phase 5 provides folder browsing and generic selection. Phase 6 has not started.
Blurry-photo review, duplicates, people workflows, favorites, trash operations,
scan integration, authentication, subscription, settings, profile, file deletion,
and export remain outside this phase. Backend analysis algorithms and GPU/ONNX
configuration are unchanged. The current branch's visual design, including its
white Library background, is preserved.

## Ownership and project flow

`ui.qt.app.run_app()` creates a separate `ProjectController`, `PhotoListModel`,
and bounded `ThumbnailService`; the controller owns `ProjectListModel`. App
supplies the controller and photo model to `qml/App.qml`, starting on Home by
default. `QtPhotoCleanerBridge` retains its existing contracts and does not own
project logic. Basic remains the default customizable Qt Quick Controls style.
Startup explicitly requests a light Qt color scheme.

`ProjectController` owns the native `QFileDialog.getExistingDirectory()` boundary,
folder opening, project registration, active identity, discovery status, and
the lightweight JSON registry. The picker runs on the GUI thread; canceling
leaves current state untouched. Tests substitute the chooser boundary and feed
real folders and image files into production discovery.

Discovery validates that the root is an accessible directory, then recursively
calls the existing `find_supported_files()` helper with the loader's supported
extensions. Mixed-case extensions are accepted and unsupported files are omitted.
The worker creates lightweight `ResultItem` paths and filenames without reading
image metadata, decoding pixels, or running scans. An empty supported-photo set
is a valid project with count zero.

One named daemon discovery worker uses a one-entry task queue. Each request
advances a generation and replaces pending queued work with the latest request.
A second one-entry queue carries plain Python results. An active-work-only 20 ms
GUI timer drains them. A result applies only if its generation matches the latest
request and the controller is open. Obsolete successes and errors cannot overwrite
the latest project's model or status.

After successful discovery, the Qt owning thread:

1. Updates last-opened time and the actual discovered photo count.
2. Calls `PhotoListModel.replace_items(items, preserve_selection=False)`.
3. Registers or updates Home metadata, preserving project ID and creation time
   when reopening the same normalized folder.
4. Updates the active project and saves the registry.
5. Emits `projectOpened`; App navigates to Library with that ID and name.

Every project open clears selection, including reopening and overlapping parent
folders containing the same photos. This is deliberately separate from the
generic photo model's optional selection-preserving refresh. Project metadata
never moves into the bridge or thumbnail service.

Returning Home destroys the Library view and its thumbnail leases while retaining
project metadata. A card calls `openProject(projectId)`, which discovers the current
folder again. Restarting loads cards and saved counts without starting discovery.

## Home metadata, search, and sort semantics

`ProjectListModel` holds lightweight `Project` records with roles `projectId`,
`name`, `photoCount`, `thumbnailUrl`, and `lastOpened`. Project previews remain
placeholders: `thumbnailUrl` is empty. The existing card footer shows the saved
count alongside a valid last-opened label, retaining card geometry and typography.

Search uses case-insensitive substring matching on project names.

| Home option | Order | Tie breaker |
| --- | --- | --- |
| Newest (default) | Creation timestamp descending | Project ID descending |
| Oldest | Creation timestamp ascending | Project ID ascending |
| Name | Case-folded project name ascending | Project ID ascending |

Newest/Oldest mean **project creation**, not last use. Reopening updates
last-opened metadata without changing creation ordering. Tests cover conflicting
creation/opened times, equal timestamps, and equal case-folded names.

Last-opened labels use local time in `Last opened YYYY-MM-DD HH:MM` form.
Zero, missing legacy values, and invalid/uninitialized in-memory timestamps yield
an empty label rather than a misleading 1970 date. No localization infrastructure
is introduced. Malformed persisted timestamps are rejected per record.

| Production Home control | Behavior |
| --- | --- |
| New Project | Native picker; disabled while discovery is busy |
| Project card | Reopens the registered folder |
| Search | Filters project names |
| Sort by | Newest / Oldest / Name, as defined above |
| Home View sizing | Disabled; project-card size variants are unimplemented |
| Subscription, Settings, Profile | Disabled; future signal contracts only |
| Project-card menu | Disabled; its reserved pointer area cannot reopen the card |

`ROOM36_HOME_PREVIEW_DATA` remains an explicit development preview. Production Home
uses the controller's project model. Disabled controls retain future signal
contracts, but actual user mouse clicks emit no application actions. Lower-level
contract tests may emit signals directly; those emissions do not simulate clicks.

## Registry persistence and compatibility

The registry is `app_data_path("room36_projects.json")`. Current location behavior:

| Execution mode | Default location |
| --- | --- |
| Source checkout | Project root |
| Frozen application | Executable's parent directory |
| `IMAGE_DEDUPLICATOR_APP_DATA_ROOT` set | Expanded/resolved override directory |

Tests use temporary registries and an isolated application-data override.
Proper platform per-user data placement remains a packaging/pre-release task.
A future Windows build should use an appropriate user application-data directory
instead of the installation directory. Phase 5 does not refactor other storage.

The JSON envelope stays **version 1**. Each project persists exactly these fields:

| Field | Value |
| --- | --- |
| `project_id` | Stable generated string ID |
| `folder` | Normalized absolute folder path |
| `name` | Folder-derived project name |
| `created_at` | Creation timestamp |
| `last_opened` | Most recent successful open timestamp |
| `photo_count` | Integer >= 0, or -1 for unknown |

`photo_count` is an optional additive field; no schema bump is required.
Old version-1 records lacking it load with count -1. Booleans, floats, strings,
null, collections, and integers below -1 become -1 without discarding an otherwise
valid project. The next save writes the normalized count. Successful opening
refreshes and persists the actual discovered count. Empty projects persist zero.
Home displays a saved count immediately after restart, without enumeration.
Counts may become stale while folders are closed; reopening refreshes them.

Missing legacy `last_opened` defaults to zero. Invalid present timestamps,
non-finite/negative values, unsupported display ranges, invalid required strings,
and relative folder paths cause the record to be skipped safely. Duplicate IDs
and normalized folders are not registered twice. Missing folders remain as cards
so reopening can explain the error.

Writes use a sibling `.json.tmp` followed by replacement of the registry.
This is lightweight single-process metadata storage, without cross-process
locking, a database, photo pixels, thumbnail URLs, or persistent thumbnail caching.
Concurrent instances sharing a registry are not coordinated.

## Errors and discovery cancellation assessment

Missing/moved folders and inaccessible roots produce specific status messages.
Other discovery failures produce a generic message. Failed discovery neither
registers a new project nor replaces previous photo rows. Corrupt registry JSON
or invalid envelopes recover with an empty list and a message; malformed individual
records are skipped with a message. Registry write failure still permits opening
the project but clearly reports that metadata was not saved. Raw worker exceptions
and file paths are not exposed as error text.

The existing `find_supported_files()` uses `Path.rglob()` and provides no
cancellation callback. Cooperative cancellation would require changing the shared
backend traversal or duplicating it. Neither is part of this stabilization.

**Accepted Phase 5 limitation:** if discovery A is already running when project B
is requested, B waits for A's enumeration to finish. Generation checks guarantee
correctness but cannot interrupt active filesystem I/O. Nested-directory
accessibility follows the shared helper's existing best-effort traversal behavior.

`ProjectController.shutdown()` advances the generation, stops the timer, sets
the stop event, and waits up to 0.2 seconds. An already-blocked traversal can finish
later on its daemon worker; its result cannot apply after stop and it cannot
prevent process exit. Gated tests release their workers and verify termination.
There is no unsafe termination, duplicate traversal, or unbounded worker creation.

## Generic photo model and thumbnail service

`PhotoListModel` remains a generic QAbstractListModel adapting `ResultItem`.
Its ten roles are `photoId`, `sourcePath`, `filename`, `title`, `detail`,
`badgeText`, `personId`, `selected`, `thumbnailSource`, and `thumbnailState`.
Paths use lexical normpath/abspath/normcase without GUI-thread filesystem
resolution. Duplicate supplied paths collapse to one row; symlink/hardlink aliases
remain distinct. Thumbnail states are idle/loading/ready/error.

Selection is generic model-owned UI state. Inserts/removals use row notifications.
Filename sorting remaps persistent indexes with layout notifications. Replacement
resets the model and changes thumbnail revisions. Mutations enforce the Qt owning
thread. Model row removal never deletes a file. No favorite state, database ID,
or backend scan operation is added.

`ThumbnailService` remains a separate GUI-owned bounded scheduler. Workers reuse
ImageLoader's reduced JPEG decode, EXIF orientation, resize, RGB conversion, and
optional RAW-preview support. A fresh loader per request avoids retaining its
metadata cache. Decode and PNG/base64 encoding occur on workers. Plain queues,
events, and strings cross thread boundaries, never Qt objects.

Defaults are two workers, 256 outstanding subscriber tickets, and an LRU limited
by both 128 entries and 32 MiB of encoded ASCII payload. Sizes clamp to 32–512 px.
Identical keys coalesce; failures are cached. Oversized entries can be delivered
without retention. Ticket/identity/revision validation rejects stale thumbnails
after removal, refresh, or supersession. Saturation still returns ticket zero;
no polling or uncontrolled retry loop is added.

Each delegate owns an opaque lease. Last release cancels pending work and clears
its row's source. Qt Image loads bounded data URLs asynchronously with its global
URL cache disabled. Cache limits do not bound codec-internal working memory.
Formats without reduced decoding may require a full-size worker intermediate.
RAW support depends on rawpy and previews. No persistent thumbnail cache or file
watcher is introduced.

`PhotoListModel.close()` cancels requests; Qt-owner destruction cancels orphan work.
`ThumbnailService.shutdown()` cancels pending work, stops its timer, clears caches,
and waits up to 0.5 seconds. Native decodes cannot be forcibly interrupted.
Remaining daemon work holds no Qt objects, cannot publish after shutdown, and
cannot prevent exit. Bootstrap cleans up the controller, model, and service
through both `aboutToQuit` and `finally`.

## Library integration and saturation stabilization

`PhotoGrid` remains a virtualized GridView with a one-row cache buffer,
`reuseItems: false`, model-owned selection, and `currentIndex: -1`. Cards use real
thumbnail URLs and discovered filenames. Favorite/photo-menu controls stay hidden.

| Library control | Behavior |
| --- | --- |
| Sort by | Name A–Z / Name Z–A; case-folded filename + normalized path ordering |
| View | Small / Medium / Large; grid dimensions and thumbnail edge |
| Home button / sidebar Home | Returns to Home |
| Preferences | Disabled |
| Other sidebar routes | Honest centralized placeholders; no feature operations |

Small/Medium/Large use thumbnail edges 256/384/512 and the existing relative card
widths 0.75/1/1.25. Sort/View are disabled on placeholders. Blurry, duplicates,
favorites, known people, unknown people, and trash do not load scans or review
data. Leaving the grid releases its leases.

The saturation regression revealed that GridView can transiently create many
delegates during a large jump/relayout even though fewer than 100 survive.
Immediate acquisition could consume 256 tickets before temporary delegates were
destroyed. The scoped fix is a zero-interval **single-shot**, delegate-owned timer
that coalesces acquisition until synchronous layout settles. Destruction removes
that timer and releases existing work. Identity/model/edge changes schedule the
same acquisition. This is event deferral, not polling or retries. Neither the
photo model nor thumbnail service is redesigned.

The stress test holds decoding behind a gate while scrolling 6,000 rows and
switching all view sizes at 1280×720, 1440×900, and 1920×1080. It checks pending
requests at acquisition time, including transient peaks, asserts fewer than 100
live leases, and fails on saturation. Releasing the decoder verifies that current
delegates reach ready state.

## Pointer and logo regressions

Project-flow tests retain real `QTest.mouseClick()` coverage for New Project,
project-card reopen, Home return, and Library navigation. Actual disabled clicks
are checked for Subscription, Settings, Profile, Home View, Preferences, and
project menus. Controller coverage includes missing/inaccessible folders, stale
results, selection isolation, persistence, legacy counts/timestamps, and search/sort.
Qt model testers and warning handlers check model, Qt/QML, and shutdown behavior.

`logo_blue.svg`, `logo_white.svg`, and `logo_loading_white.svg` retain their
existing pure-vector branding and proportions. Tests reject embedded raster
images, base64, and clip constructs; check QSvgRenderer/QML loading; and sample
green aperture blades plus transparent openings for all three logos.

`app_icon.svg` still contains Illustrator-era embedded raster/clip constructs.
It is not referenced by Phase 5 production screens; cleanup remains a packaging
task. No branding assets are redesigned. The unrelated
`face_analyzer/test_classical_metrics.py` and `face_analyzer/test_face_analysis.py`
are not modified.

## Validation commands

Use the module's configured interpreter (Python 3.14.5 in this checkout). From
the project root:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:QML_DISABLE_DISK_CACHE = '1'
$env:QT_SHADER_CACHE_DISABLE = '1'
$env:QT_QPA_FONTDIR = "$PWD/assets/fonts"
Remove-Item Env:QT_QUICK_CONTROLS_STYLE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -B -m unittest -v ui.test_startup_routing ui.test_qt_bridge ui.test_qml_design_system ui.test_qml_pages ui.test_qml_library_shell ui.test_view_model ui.test_qt_photo_model ui.test_qt_thumbnails ui.test_qml_photo_grid ui.test_qt_projects ui.test_qml_project_flow ui.test_qml_logos
```

Leave the style override unset to validate production's Basic default. For isolated
validation, point `IMAGE_DEDUPLICATOR_APP_DATA_ROOT` at a temporary directory.
The executed validation harness additionally captures warnings outside per-test
handlers and checks for named discovery/thumbnail workers after the suite.

Broader discovery uses the same headless configuration:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v -s . -p 'test_*.py'
```

The pattern excludes the local `test.py` model-export script and `test2.py`
manual model probe. It does not run scripts that download/export models.

## Recorded final validation — 2026-10-01

| Run | Tests run | Passed | Failures | Errors | Skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit Phase 5/UI suite | 88 | 88 | 0 | 0 | 0 |
| Broader root unittest discovery | 181 | 169 | 0 | 12 | 0 |
| Isolated unchanged backend modules | 44 | 32 | 0 | 12 | 0 |

The explicit suite completed in 27.208 seconds. No Qt/QML, style, model-tester,
or worker-shutdown warnings were captured, including outside individual test
handlers. No named Room 36 discovery/thumbnail worker remained alive after either
the explicit suite or broader discovery.

Explicit-suite diagnostics: 6,000 rows populated in 23.7 ms; delegate samples
13/17/17/10; test LRU 16 entries / 1,996 encoded bytes. With decoders blocked,
the fast scroll/view-size stress peaked at 47 delegates, 47 leases, and 47/256
pending tickets, with zero saturated requests. Timings are observations, not FPS
or performance guarantees.

Broader discovery completed in 31.677 seconds. All 12 errors are Windows
`PermissionError: [WinError 32]` during temporary SQLite file cleanup, outside
the Phase 5 UI path. They reproduce in an isolated run of the unchanged
`face_processing.test_processing` and `grouping.test_vibe_grouping` modules
without loading any `ui.qt.*` module. Git comparison confirms the affected
backend/test files remain identical to the audited HEAD. They are not masked,
skipped, or repaired by changing unrelated backend storage in this pass.

Exact remaining broader-validation errors:

| Test class | Method |
| --- | --- |
| face_processing.test_processing.FaceProcessingTests | test_cache_requires_analysis_when_requested |
| face_processing.test_processing.FaceProcessingTests | test_cache_round_trips_analysis_payload |
| face_processing.test_processing.FaceProcessingTests | test_image_face_analysis_cache_round_trips_face_details |
| face_processing.test_processing.FaceProcessingTests | test_processor_backfills_image_analysis_cache_from_face_scan_cache |
| face_processing.test_processing.FaceProcessingTests | test_processor_persists_face_analysis_in_image_analysis_cache |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_algorithm_version_invalidates_result_cache_only |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_background_setting_invalidates_feature_cache |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_cancel_before_scan_raises |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_corrupted_file_is_reported_without_aborting |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_diagnostics_include_image_and_session_debug_data |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_prototype_change_reuses_cached_semantic_embeddings |
| grouping.test_vibe_grouping.VibeProcessorCacheTests | test_reuses_folder_result_cache_when_inputs_match |

The Phase 5-specific completion checks pass. A blanket zero-error claim for the
broader repository cannot be made until these independent Windows SQLite cleanup
errors are resolved. Accepted Phase 5 limitations remain active-discovery latency,
counts refreshed on reopen, native decode/traversal that cannot be forcibly
interrupted, and packaging tasks for per-user storage and app-icon cleanup.
