"""Diagnostic logging for automatic scans, independent of UI error messages."""

import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

from app_paths import app_data_path


LOG_FILENAME = "automatic_scan.log"
_SETUP_LOCK = Lock()


def get_scan_logger() -> logging.Logger:
    """Return a bounded local log; fall back to stderr if app data is unwritable."""
    try:
        path = Path(os.environ["ROOM36_SCAN_LOG_PATH"]) if os.environ.get("ROOM36_SCAN_LOG_PATH") else Path(app_data_path(LOG_FILENAME))
    except OSError:
        path = None
    name = "room36.automatic_scan." + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    with _SETUP_LOCK:
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                if path is None:
                    raise OSError("Application data directory is unavailable")
                handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024,
                                              backupCount=3, encoding="utf-8")
            except OSError:
                handler = logging.StreamHandler()
                handler.setLevel(logging.WARNING)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
            logger.addHandler(handler)
    return logger


def close_scan_logger_for_path(path: Path) -> None:
    """Release an isolated benchmark log before its temp directory is removed."""
    name = "room36.automatic_scan." + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    logger = logging.getLogger(name)
    with _SETUP_LOCK:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
