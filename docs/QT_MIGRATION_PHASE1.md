# Qt Migration Phase 1

## Scope

This phase adds a minimal PySide6 + Qt Quick/QML bootstrap beside the existing Tkinter application. The Python scan pipeline, detectors, face processing, grouping, and cache logic remain in the existing backend.

## Files Added

- `ui/qt/__init__.py`
- `ui/qt/app.py`
- `ui/qt/bridge.py`
- `qml/App.qml`
- `qml/theme/Theme.qml`
- `qml/theme/Typography.qml`
- `qml/theme/Metrics.qml`
- `ui/test_qt_bridge.py`
- `docs/QT_MIGRATION_PHASE1.md`

## Files Changed

- `main.py`
- `app_paths.py`
- `requirements.txt`
- `win_requirements.txt`

## Running Tkinter

Tkinter remains the default UI:

```bash
./.venv/bin/python main.py
```

Or explicitly:

```bash
ROOM36_UI=tk ./.venv/bin/python main.py
```

## Running Qt/QML

After installing dependencies, launch the Qt bootstrap with:

```bash
ROOM36_UI=qt ./.venv/bin/python main.py
```

## PySide6 Version

`PySide6==6.11.1` is pinned in `requirements.txt` and `win_requirements.txt`.

This version was selected against the current Python environment because the PyPI package metadata for `PySide6` lists support for Python `>=3.10,<3.15`, which includes this repository's Python 3.12 runtime, and publishes Linux and Windows wheels for that range.

Source:

- https://pypi.org/project/PySide6/

## UI Selection

`main.py` still performs the existing CUDA/NVIDIA preload work before importing either UI.

Selection is controlled by `ROOM36_UI`:

- unset or `ROOM36_UI=tk`: current Tkinter UI
- `ROOM36_UI=qt`: new Qt/QML bootstrap

## QML and Resource Paths

Resource paths are resolved from `app_paths.resource_root()`, not from the current working directory.

Helpers added in `app_paths.py`:

- `resource_path(...)`
- `qml_root()`
- `qml_path(...)`
- `assets_root()`
- `asset_path(...)`

`ui/qt/app.py` loads `qml/App.qml` through `qml_path("App.qml")`.

The typography asset references in `qml/theme/Typography.qml` use `Qt.resolvedUrl(...)`, so the font paths resolve relative to the QML file location rather than the shell working directory.

## Bridge Design

`ui/qt/bridge.py` provides a `QObject` facade around `PhotoCleanerViewModel`.

Exposed Qt properties:

- `folder`
- `status`
- `countText`
- `elapsedText`
- `isScanning`
- `canScan`
- `canCancel`
- `progressValue`
- `progressMaximum`
- `progressIndeterminate`

Exposed Qt slots:

- `setFolder(path)`
- `refreshFileTypes()`
- `startScan()`
- `cancelScan()`
- `chooseFolder()`

`chooseFolder()` keeps dialog handling out of `PhotoCleanerViewModel`.

## Bridge Polling

The existing backend worker still writes `BackgroundMessage` instances into the ViewModel queue.

The Qt bridge drains that queue with a `QTimer` every 50 ms:

1. `poll_background_message()`
2. dispatch to the existing ViewModel handler
3. compare the visible bridge snapshot
4. emit only the property signals whose values changed

Elapsed time is refreshed through a separate 1000 ms `QTimer` that calls the existing `refresh_elapsed()` method.

## Fonts

The QML typography layer attempts to load:

- `assets/fonts/Britanica Thin.ttf`
- `assets/fonts/Britanica Regular.ttf`
- `assets/fonts/Britanica Bold.ttf`

If those files are unavailable at runtime, the bootstrap falls back to the Qt application default font family.

## Known Limitations

- The QML surface is a development shell only; production Room 36 screens are not recreated here.
- The Qt bridge does not yet expose the full ViewModel surface.
- Unknown-face naming prompts are not implemented in Qt during this phase; unknown clusters still flow through the existing backend state.
- Packaging updates for bundling QML/assets into frozen builds are not included in this phase.
- Bridge tests are skipped automatically when PySide6 is not installed in the active environment.

## Recommended Next Step

Build the first real Qt screen on top of the bridge while expanding the bridge surface incrementally around existing ViewModel state instead of bypassing it.
