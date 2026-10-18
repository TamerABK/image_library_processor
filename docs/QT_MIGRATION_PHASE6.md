# Room 36 Phase 6: project processing and Blur review

Phase 6 starts project analysis after Phase 5 discovery. One processing worker
publishes bounded, project-scoped stage updates to the Qt controller. Opening a
different project cancels the old session and discards its late results.

The seven stages are Blur, duplicates, face detection, face quality, embeddings,
People matching, and duplicate representative ranking. The status strip displays
one progress bar per stage. A bar shows a photo count when the backend has one and
an active indicator during work without a counter.

`ui/qt/processing_controller.py` owns scheduling and review actions;
`ui/qt/blur_model.py` exposes Blur candidates. `qml/pages/BlurryPhotosPage.qml`
contains the review grid, and `qml/components/ProcessingStatus.qml` shows stage
status. `ui/qt/inspector_controller.py`, `ui/qt/previews.py`, and
`qml/components/PhotoInspector.qml` provide bounded photo previews. Review
decisions are persisted; Phase 6 does not execute deletion.

Per-photo analysis and project aggregates use the existing scan caches. A failed
People stage leaves completed Blur, duplicate, and face results available.
Unknown-face clustering still constructs an N × N similarity matrix, so People
matching fails with a safe message before clustering more than 2,000 unknown
faces. This limit does not stop earlier stages.

Run the Phase 5/6 UI and scan tests with the project's configured Python
interpreter:

```powershell
python -B -m unittest ui.test_qml_project_flow ui.test_qml_library_shell `
  ui.test_qt_processing ui.test_qml_blur_flow `
  automatic_scan.tests.test_project_pipeline
python -B -m unittest discover -p 'test_*.py'
```

The production app needs PySide6, the chosen single ONNX Runtime distribution,
and model files for face inference. Startup and QML loading do not require a
model session; face-model errors are reported by the relevant scan stage.
