"""Mouse and keyboard contract of the shared toolbar popup."""
import unittest

from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from ui.qt.app import configure_quick_controls_style
from ui.test_qt_thumbnails import capture_qt_warnings


class ToolbarDropdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.engine = QQmlApplicationEngine()
        self.engine.load(qml_path("dev/ToolbarDropdownHarness.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.button = self.window.findChild(QObject, "standaloneDropdown")
        self.assertIsNotNone(self.button)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()

    def click(self, item):
        pos = item.mapToScene(QPointF(item.width()/2, item.height()/2)).toPoint()
        QTest.mouseClick(self.window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        self.app.processEvents()

    def item(self, name):
        def visit(parent):
            if parent.objectName() == name:
                return parent
            for child in parent.childItems():
                found = visit(child)
                if found is not None:
                    return found
            return None
        return visit(self.window.contentItem())

    def test_click_toggle_select_escape_and_outside(self):
        for _ in range(2):
            self.click(self.button)
            self.assertTrue(self.button.property("open"))
            self.click(self.button)
            self.assertFalse(self.button.property("open"))
        self.click(self.button)
        option = self.item("toolbarDropdownOption_Second")
        self.assertIsNotNone(option)
        self.click(option)
        self.assertFalse(self.button.property("open"))
        self.assertEqual(self.window.property("selectedOption"), "Second")
        self.click(self.button)
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(self.button.property("open"))
        self.click(self.button)
        QTest.mouseClick(self.window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPointF(400, 250).toPoint())
        self.app.processEvents()
        self.assertFalse(self.button.property("open"))


if __name__ == "__main__": unittest.main()
