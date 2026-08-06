from __future__ import annotations

import threading


class ScanCancelledError(RuntimeError):
    """Raised when a background scan is canceled by the user."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_canceled(self) -> bool:
        return self._event.is_set()

    def raise_if_canceled(self) -> None:
        if self._event.is_set():
            raise ScanCancelledError("Scan canceled.")
