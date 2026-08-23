from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from ui.models import (
    BackgroundMessage,
    ScanErrorMessage,
    ScanProgressMessage,
    ScanResultMessage,
    UnknownFacesMessage,
)
from ui.view_model import PhotoCleanerViewModel


@dataclass(frozen=True, slots=True)
class _VisibleBridgeState:
    folder: str
    status: str
    count_text: str
    elapsed_text: str
    is_scanning: bool
    can_scan: bool
    can_cancel: bool
    progress_value: int
    progress_maximum: int
    progress_indeterminate: bool


class QtPhotoCleanerBridge(QObject):
    folderChanged = Signal()
    statusChanged = Signal()
    countTextChanged = Signal()
    elapsedTextChanged = Signal()
    isScanningChanged = Signal()
    canScanChanged = Signal()
    canCancelChanged = Signal()
    progressValueChanged = Signal()
    progressMaximumChanged = Signal()
    progressIndeterminateChanged = Signal()

    def __init__(
        self,
        view_model: PhotoCleanerViewModel | None = None,
        *,
        parent: QObject | None = None,
        background_poll_interval_ms: int = 50,
        elapsed_interval_ms: int = 1000,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model or PhotoCleanerViewModel()
        self._visible_state = self._capture_visible_state()

        self._background_timer = QTimer(self)
        self._background_timer.setInterval(background_poll_interval_ms)
        self._background_timer.timeout.connect(self._drain_background_messages)
        self._background_timer.start()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(elapsed_interval_ms)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed_time)
        self._sync_elapsed_timer()

    @Property(str, notify=folderChanged)
    def folder(self) -> str:
        return self._view_model.state.folder

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        return self._view_model.state.status

    @Property(str, notify=countTextChanged)
    def countText(self) -> str:
        return self._view_model.state.count_text

    @Property(str, notify=elapsedTextChanged)
    def elapsedText(self) -> str:
        return self._view_model.state.elapsed_text

    @Property(bool, notify=isScanningChanged)
    def isScanning(self) -> bool:
        return self._view_model.is_scanning()

    @Property(bool, notify=canScanChanged)
    def canScan(self) -> bool:
        return self._view_model.state.can_scan

    @Property(bool, notify=canCancelChanged)
    def canCancel(self) -> bool:
        return self._view_model.state.can_cancel

    @Property(int, notify=progressValueChanged)
    def progressValue(self) -> int:
        return self._view_model.state.progress_value

    @Property(int, notify=progressMaximumChanged)
    def progressMaximum(self) -> int:
        return self._view_model.state.progress_max

    @Property(bool, notify=progressIndeterminateChanged)
    def progressIndeterminate(self) -> bool:
        return self._view_model.state.progress_mode == "indeterminate"

    @Slot(str)
    def setFolder(self, path: str) -> None:
        normalized_path = str(Path(path).expanduser()) if path.strip() else ""
        self._view_model.set_folder(normalized_path)
        self._sync_visible_state()

    @Slot()
    def refreshFileTypes(self) -> None:
        self._view_model.refresh_file_types()
        self._sync_visible_state()

    @Slot(result=bool)
    def startScan(self) -> bool:
        error = self._view_model.start_scan()
        if error:
            self._view_model.state.status = error
        self._sync_visible_state()
        return error is None

    @Slot()
    def cancelScan(self) -> None:
        self._view_model.cancel_scan()
        self._sync_visible_state()

    @Slot(result=str)
    def chooseFolder(self) -> str:
        current_folder = self._view_model.state.folder.strip()
        start_directory = current_folder or str(Path.home())
        selected_folder = QFileDialog.getExistingDirectory(
            None,
            "Select photo folder",
            start_directory,
        )
        if selected_folder:
            self.setFolder(selected_folder)
            self.refreshFileTypes()
        return self._view_model.state.folder

    def _capture_visible_state(self) -> _VisibleBridgeState:
        state = self._view_model.state
        return _VisibleBridgeState(
            folder=state.folder,
            status=state.status,
            count_text=state.count_text,
            elapsed_text=state.elapsed_text,
            is_scanning=self._view_model.is_scanning(),
            can_scan=state.can_scan,
            can_cancel=state.can_cancel,
            progress_value=state.progress_value,
            progress_maximum=state.progress_max,
            progress_indeterminate=state.progress_mode == "indeterminate",
        )

    def _sync_visible_state(self) -> None:
        previous = self._visible_state
        current = self._capture_visible_state()
        self._visible_state = current

        if current.folder != previous.folder:
            self.folderChanged.emit()
        if current.status != previous.status:
            self.statusChanged.emit()
        if current.count_text != previous.count_text:
            self.countTextChanged.emit()
        if current.elapsed_text != previous.elapsed_text:
            self.elapsedTextChanged.emit()
        if current.is_scanning != previous.is_scanning:
            self.isScanningChanged.emit()
        if current.can_scan != previous.can_scan:
            self.canScanChanged.emit()
        if current.can_cancel != previous.can_cancel:
            self.canCancelChanged.emit()
        if current.progress_value != previous.progress_value:
            self.progressValueChanged.emit()
        if current.progress_maximum != previous.progress_maximum:
            self.progressMaximumChanged.emit()
        if current.progress_indeterminate != previous.progress_indeterminate:
            self.progressIndeterminateChanged.emit()

        self._sync_elapsed_timer()

    def _sync_elapsed_timer(self) -> None:
        if self._view_model.is_scanning():
            if not self._elapsed_timer.isActive():
                self._elapsed_timer.start()
            return

        if self._elapsed_timer.isActive():
            self._elapsed_timer.stop()

    def _refresh_elapsed_time(self) -> None:
        keep_refreshing = self._view_model.refresh_elapsed()
        self._sync_visible_state()
        if not keep_refreshing and self._elapsed_timer.isActive():
            self._elapsed_timer.stop()

    def _drain_background_messages(self) -> None:
        while True:
            message = self._view_model.poll_background_message()
            if message is None:
                break
            self._handle_background_message(message)
            self._sync_visible_state()

    def _handle_background_message(self, message: BackgroundMessage) -> None:
        if isinstance(message, ScanProgressMessage):
            self._view_model.handle_progress_message(message)
            return

        if isinstance(message, UnknownFacesMessage):
            self._view_model.handle_unknown_faces_message(message)
            return

        if isinstance(message, ScanResultMessage):
            self._view_model.handle_scan_result_message(message)
            return

        if isinstance(message, ScanErrorMessage):
            self._view_model.handle_scan_error_message(message)
