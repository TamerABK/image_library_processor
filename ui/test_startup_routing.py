from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from main import _ui_mode_from_environment


class StartupRoutingTests(unittest.TestCase):
    def test_qt_is_default_ui_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_ui_mode_from_environment(), "qt")

    def test_qt_ui_mode_override(self) -> None:
        with patch.dict(os.environ, {"ROOM36_UI": "qt"}, clear=False):
            self.assertEqual(_ui_mode_from_environment(), "qt")

    def test_ui_mode_is_normalized(self) -> None:
        for value, expected in ((" QT ", "qt"), (" TK ", "tk")):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"ROOM36_UI": value}, clear=False):
                    self.assertEqual(_ui_mode_from_environment(), expected)

    def test_tkinter_requires_explicit_opt_in(self) -> None:
        with patch.dict(os.environ, {"ROOM36_UI": "tk"}, clear=False):
            self.assertEqual(_ui_mode_from_environment(), "tk")

    def test_empty_or_unknown_mode_uses_qt(self) -> None:
        for value in ("", " ", "unsupported"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"ROOM36_UI": value}, clear=False):
                    self.assertEqual(_ui_mode_from_environment(), "qt")
