# Phase 5 — photo infrastructure

## Audit and architecture decision

`ui.models.ResultItem` carries path/title/detail/badge/person metadata, and the
legacy ViewModel keys selection by Path. There is no general favorite contract.
The Tk view synchronously uses Pillow draft/EXIF transpose/thumbnail and retains
ImageTk objects. These Tk-owned objects are unsuitable for Qt or bounded caching.
`image_loader.ImageLoader` already supports reduced JPEG decoding, resize, EXIF
orientation through the raster decoder, and optional RAW previews. Its public
`load_pil_for_hashing` method returns RGB PIL pixels and is reusable independently
of hashing. A fresh loader per thumbnail avoids retaining its metadata cache.
`image_cache_storage` persists analysis metadata, not thumbnail pixels.

Decision: a QAbstractListModel adapts ResultItem without changing backend types;
a separate service encodes small PNG data URLs on two daemon worker threads.
Qt Image loads only those bounded thumbnails asynchronously. No custom image
provider or competing persistent thumbnail database is needed. GridView owns a
bounded visible/near-visible set of PhotoCard delegates, with explicit thumbnail
leases and model-owned selection. Later routes remain placeholders.

## Model and adapter contract

`ui.qt.photo_model.PhotoListModel(thumbnails)` is a QAbstractListModel. Supply
already discovered `ResultItem` objects with `replace_items(items)` or
`append_items(items)`. It does not scan folders, access a project database, or
call analysis algorithms. Duplicate paths in supplied groups collapse to one row.
Production bootstrap creates an empty model and supplies it as `App.photoModel`.
An eventual project adapter must replace/supply that model when projects change;
the development Home preview has no connection to photo data.

Roles (Qt.UserRole + 1 through + 10, in this order):

| Role | Value |
| --- | --- |
| photoId | Normalized absolute path string, durable identity |
| sourcePath | Same normalized path; never bound to the grid Image source |
| filename | Basename |
| title | ResultItem.title |
| detail | ResultItem.detail |
| badgeText | ResultItem.badge_text |
| personId | Existing optional ResultItem.person_id; null if absent |
| selected | Model-owned boolean, initially false |
| thumbnailSource | Empty string or bounded PNG data URL |
| thumbnailState | idle, loading, ready, or error |

Path normalization uses normpath/abspath/normcase without filesystem resolution.
On Windows, case aliases normalize together. Symlink/hardlink aliases remain
distinct paths; renaming a file changes its identity. There is no database ID.

`setSelected(photoId, bool)` changes one role on one row, and `removePhoto(photoId)`
only removes a model row (never a file). Inserts/removals use row notifications.
`sortByFilename(descending)` reorders lightweight row references with layout
notifications and persistent-index remapping. Full replacement uses a reset,
preserves selection for surviving IDs by default, and invalidates thumbnail
revisions. Reinsertion also gets a new revision. No filtering proxy is added;
an adapter can supply a filtered item set, preserving surviving identities.
Selection is generic UI state and does not trigger legacy deletion/export state.
No favorite role/persistence is invented; the grid hides favorite/menu controls.

## Thumbnail request flow and memory bounds

Each delegate acquires an opaque lease with `acquireThumbnail(photoId, edge)`.
The model requests service work using path, maximum edge, and a revision. Multiple
leases for one row share work; the service also coalesces identical request keys.
Larger requested sizes supersede smaller in-flight work. Ticket + identity +
revision validation rejects late results after refresh, removal, or supersession.

The service has two workers by default, at most 256 outstanding subscriber tickets,
and a GUI-owned LRU bounded by BOTH 128 entries and 32 MiB of encoded ASCII payload.
Sizes clamp to 32-512 px; the grid requests 256/384/512 for Small/Medium/Large.
Failure entries are cached too. Oversized cache entries can be delivered without
being retained. A saturated request returns ticket 0, leaving an idle placeholder;
a later delegate acquisition can retry. These limits are constructor configurable.

Decoding, metadata reads, resize, RGB conversion, and PNG/base64 encoding happen
only on workers. Requests use a fresh ImageLoader to avoid its unbounded metadata
cache. No original pixels are retained in the cache. Encoded URLs have base64
overhead, but allow Qt to use standard Image delivery without a provider or disk
artifacts. Image.asynchronous is enabled and cache is disabled for grid photos,
so Qt's global URL cache does not duplicate the service's retention policy.

Delegate destruction releases its lease; the final lease cancels queued work and
clears the row's source. Thus rows do not accumulate thumbnail payloads after
scrolling. Consumers outside PhotoGrid must also release their leases. Decoded
textures are limited to instantiated cards, plus up to two worker intermediates;
the cache byte limit is not a bound on codec-internal working memory.

## Threading, errors, and shutdown

Plain Python queues/events/strings cross worker boundaries, never Qt objects.
An active-work-only 10 ms QTimer drains completions on the GUI thread. Model and
service mutators enforce owner-thread access. The task/result queues each hold
at most the worker count; canceled jobs without subscribers are removed before
dispatch. In-flight native decode cannot be safely interrupted, but its stale
result cannot update a model or delegate.

Missing, removed, unsupported, corrupt, or failed images return an empty source
and error state, leaving PhotoCard's normal placeholder visible. No exceptions
or paths are shown as UI error text, and no automatic retry loop runs. Refresh
changes revisions to retry failures or changed files; there is no file watcher.

`PhotoListModel.close()` cancels its requests; QObject destruction also cancels
orphans. `ThumbnailService.shutdown()` cancels pending work, clears caches, stops
the timer, and waits at most 0.5 seconds for workers. Remaining native decodes run
on daemon threads with plain Python state only, cannot publish after shutdown,
and cannot keep the process alive. Qt-owner destruction signals the stop event
as a fallback. Bootstrap connects close/shutdown to aboutToQuit and also executes
them in finally on normal exit or QML startup failure.

## Grid and Library integration

PhotoGrid is a GridView, not a Repeater. It keeps one extra row-height cacheBuffer
around the viewport. reuseItems is false: delegates are destroyed and reacquired,
which keeps lease lifecycle explicit. Stable model selection survives this churn.
GridView currentIndex is -1, independent of durable photo selection. Data-URL
sources belong to identity-checked rows; no row-index callback can paint a recycled
card. PhotoCard itself is reused with only asynchronous-image/cache controls added.

LibraryShell switches its content Loader between PhotoGrid for `library` and the
existing centralized placeholder for every other route. Sidebar/top bar lifetime
and navigation contracts are unchanged. App forwards photoModel to the shell.
View requests change generic grid dimensions only on the library route and are
still forwarded. Sort requests remain signals: the current Date/Type/orientation
options lack sufficient model metadata. Filename sorting is available to adapters
but is not mislabeled as one of those toolbar options.

## Validation and limitations

Tests use temporary tiny images for decode/aspect/EXIF/error checks and synthetic
data URLs for QML virtualization. The 6,000-row diagnostic observes population
time and live delegates without imposing machine-specific FPS thresholds. Model
tests include Qt's model tester, precise notifications, stable selection/sorting,
late results, destruction, and thread-affinity enforcement. Service tests cover
LRU bounds, coalescing, saturation, cancellation, and shutdown. QML tests cover
all three desktop sizes, view sizing, source identity, scrolling, refresh, model
replacement with null, and selection surviving delegate destruction. Qt/QML
warnings fail the new tests. The full prior UI suite is retained; its Library
route assertion now expects the grid instead of the Phase 4 placeholder.

The existing loader may return JPEGs smaller than the requested maximum because
it uses reduced decode factors. Formats without reduced decoding can require a
full-size temporary decode on a worker before resize. RAW behavior depends on
rawpy and available embedded previews; RAW/camera coverage is inherited, not
exhaustively tested here. There is no persistent thumbnail cache, automatic file
watching, color-management overhaul, project persistence, scan integration, or
feature-review UI. No backend algorithms or face/database semantics changed.

Validation command (headless, cache disabled, bundled font directory on Windows):

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:QML_DISABLE_DISK_CACHE = '1'
$env:QT_SHADER_CACHE_DISABLE = '1'
$env:QT_QPA_FONTDIR = "$PWD/assets/fonts"
.\.venv\Scripts\python.exe -B -m unittest -v ui.test_startup_routing ui.test_qt_bridge ui.test_qml_design_system ui.test_qml_pages ui.test_qml_library_shell ui.test_view_model ui.test_qt_photo_model ui.test_qt_thumbnails ui.test_qml_photo_grid
```

Leave QT_QUICK_CONTROLS_STYLE unset to validate the production Basic default.

Recorded Windows headless run: 63 passed, 0 failed, 0 skipped; no captured Qt/QML,
style, model-tester, or thread-shutdown warnings. The 6,000-row diagnostic populated
in 20.9 ms and sampled 13/17/17/10 live delegates while scrolling, with a test LRU
of 16 entries / 1,996 encoded bytes. These timings are observations, not guarantees.
