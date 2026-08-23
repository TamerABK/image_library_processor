from __future__ import annotations

import importlib.util
import time
import unittest
from pathlib import Path

from ui.models import (
    ResultGroup,
    ResultItem,
    ScanErrorMessage,
    ScanProgressMessage,
    ScanResultMessage,
    UnknownFacesMessage,
)
from ui.view_model import PhotoCleanerViewModel

PYSIDE6_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE6_AVAILABLE:
    from PySide6.QtCore import QCoreApplication

    from ui.qt.bridge import QtPhotoCleanerBridge


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed")
class QtPhotoCleanerBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QCoreApplication.instance() or QCoreApplication([])

    def _make_bridge(self, view_model: PhotoCleanerViewModel | None = None) -> "QtPhotoCleanerBridge":
        bridge = QtPhotoCleanerBridge(view_model=view_model)
        self.addCleanup(bridge._background_timer.stop)
        self.addCleanup(bridge._elapsed_timer.stop)
        self.addCleanup(bridge.deleteLater)
        return bridge

    def test_bridge_creates_successfully(self) -> None:
        bridge = self._make_bridge()
        self.assertIsInstance(bridge, QtPhotoCleanerBridge)

    def test_initial_bridge_values_match_view_model_state(self) -> None:
        view_model = PhotoCleanerViewModel()
        bridge = self._make_bridge(view_model)

        self.assertEqual(bridge.folder, view_model.state.folder)
        self.assertEqual(bridge.status, view_model.state.status)
        self.assertEqual(bridge.countText, view_model.state.count_text)
        self.assertEqual(bridge.elapsedText, view_model.state.elapsed_text)
        self.assertEqual(bridge.canScan, view_model.state.can_scan)
        self.assertEqual(bridge.canCancel, view_model.state.can_cancel)
        self.assertEqual(bridge.progressValue, view_model.state.progress_value)
        self.assertEqual(bridge.progressMaximum, view_model.state.progress_max)
        self.assertEqual(
            bridge.progressIndeterminate,
            view_model.state.progress_mode == "indeterminate",
        )
        self.assertFalse(bridge.isScanning)

    def test_set_folder_updates_view_model_and_property(self) -> None:
        view_model = PhotoCleanerViewModel()
        bridge = self._make_bridge(view_model)

        bridge.setFolder("/tmp/room36")

        self.assertEqual(view_model.state.folder, "/tmp/room36")
        self.assertEqual(bridge.folder, "/tmp/room36")

    def test_scan_state_property_conversion_is_correct(self) -> None:
        view_model = PhotoCleanerViewModel()
        view_model.state.can_scan = False
        view_model.state.can_cancel = True
        view_model.state.progress_mode = "indeterminate"
        view_model.state.progress_value = 17
        view_model.state.progress_max = 40
        view_model._scan_start_time = time.monotonic()

        bridge = self._make_bridge(view_model)
        bridge._sync_visible_state()

        self.assertTrue(bridge.isScanning)
        self.assertFalse(bridge.canScan)
        self.assertTrue(bridge.canCancel)
        self.assertTrue(bridge.progressIndeterminate)
        self.assertEqual(bridge.progressValue, 17)
        self.assertEqual(bridge.progressMaximum, 40)

    def test_queue_drain_processes_progress_message(self) -> None:
        view_model = PhotoCleanerViewModel()
        view_model._scan_start_time = time.monotonic()
        bridge = self._make_bridge(view_model)

        view_model._queue.put(
            ScanProgressMessage(
                mode="duplicates",
                phase="indexing",
                done=3,
                total=10,
                known_people_only=False,
            )
        )

        bridge._drain_background_messages()

        self.assertIn("Scanning near duplicates", bridge.status)
        self.assertFalse(bridge.progressIndeterminate)
        self.assertGreater(bridge.progressValue, 0)
        self.assertEqual(bridge.progressMaximum, 100)

    def test_queue_drain_processes_unknown_faces_message(self) -> None:
        view_model = PhotoCleanerViewModel()
        bridge = self._make_bridge(view_model)
        face_result = object()

        view_model._queue.put(UnknownFacesMessage(face_result=face_result))

        bridge._drain_background_messages()

        self.assertIs(view_model._latest_face_result, face_result)

    def test_queue_drain_processes_result_message(self) -> None:
        view_model = PhotoCleanerViewModel()
        view_model.state.mode = "duplicates"
        view_model.state.can_scan = False
        view_model.state.can_cancel = True
        view_model._scan_start_time = time.monotonic()
        bridge = self._make_bridge(view_model)

        result_group = ResultGroup(
            title="Duplicate group 1",
            items=[
                ResultItem(
                    path=Path("/tmp/example.jpg"),
                    title="example.jpg",
                    detail="detail",
                    recommended_delete=True,
                )
            ],
        )
        view_model._queue.put(
            ScanResultMessage(
                mode="duplicates",
                results=[result_group],
                known_people_only=False,
                summary="Found duplicate content.",
            )
        )

        bridge._drain_background_messages()

        self.assertEqual(bridge.status, "Found duplicate content.")
        self.assertEqual(bridge.countText, "1 selected / 1 total")
        self.assertTrue(bridge.canScan)
        self.assertFalse(bridge.canCancel)
        self.assertFalse(bridge.isScanning)

    def test_queue_drain_processes_error_message(self) -> None:
        view_model = PhotoCleanerViewModel()
        view_model.state.can_scan = False
        view_model.state.can_cancel = True
        view_model._scan_start_time = time.monotonic()
        bridge = self._make_bridge(view_model)

        view_model._queue.put(ScanErrorMessage(message="Failed badly", canceled=False))

        bridge._drain_background_messages()

        self.assertEqual(bridge.status, "Failed badly")
        self.assertTrue(bridge.canScan)
        self.assertFalse(bridge.canCancel)
        self.assertFalse(bridge.isScanning)
