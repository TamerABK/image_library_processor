import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from app_state_database import AppStateDatabase
from ui.qt.app import configure_quick_controls_style
from ui.qt.settings_controller import SettingsController
from ui.test_qt_thumbnails import capture_qt_warnings


class ProcessingStub(QObject):
    changed = Signal()
    def __init__(self):
        super().__init__()
        self.project = "A"
        self.state = "COMPLETE"
        self.retries = 0
    @Property(str, notify=changed)
    def activeProjectId(self): return self.project
    @Property(str, notify=changed)
    def overallState(self): return self.state
    @Slot()
    def retry(self): self.retries += 1


class SettingsQmlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.db = AppStateDatabase(Path(self.temp.name) / "state.sqlite3")
        with self.db.connect() as db:
            db.execute("INSERT INTO projects(project_id,folder,name,created_at) VALUES('A','A','Project A',0)")
        self.controller = SettingsController(self.db)
        self.topology = patch("ui.qt.settings_controller.available_cpu_topology",
                              return_value=(12, 6, "physical cores"))
        self.topology.start()
        self.processing = ProcessingStub()
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties(dict(startPage="settings", settingsProjectId="A",
            activeProjectName="Project A", settingsController=self.controller,
            processingController=self.processing))
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.page = self.window.findChild(QObject, "settingsPage")
        self.assertIsNotNone(self.page)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()
        self.topology.stop()
        self.temp.cleanup()

    def item(self, name):
        def visit(parent):
            if parent.objectName() == name: return parent
            for child in parent.childItems():
                found = visit(child)
                if found is not None: return found
            return None
        return visit(self.window.contentItem())

    def test_individual_override_save_and_refresh(self):
        self.page.choose("blur_policy", "automatic")
        self.page.choose("face_quality_preset", "strict")
        self.page.save()
        self.assertTrue(self.page.property("refreshAvailable"))
        self.assertTrue(self.page.property("canRefresh"))
        self.processing.state = "RUNNING"
        self.processing.changed.emit()
        self.assertFalse(self.page.property("canRefresh"))
        self.processing.state = "COMPLETE"
        self.processing.project = "B"
        self.processing.changed.emit()
        self.assertFalse(self.page.property("canRefresh"))
        self.processing.project = "A"
        self.processing.changed.emit()
        self.assertEqual(self.controller.resolve("A")["face_quality_preset"], "strict")
        self.page.inherit("blur_policy")
        self.page.save()
        self.assertNotIn("blur_policy", self.controller.pageData("A")["overrides"])
        self.assertEqual(self.controller.resolve("A")["face_quality_preset"], "strict")
        self.page.choose("performance_mode", "normal")
        self.page.save()
        self.assertFalse(self.page.property("refreshAvailable"))
        self.page.choose("keepers_per_group", 99)
        self.page.save()
        self.assertFalse(self.page.property("refreshAvailable"))
        self.assertIn("Could not save", self.page.property("message"))

    def test_performance_confirmation_and_disabled_fallback(self):
        selector = self.item("setting_performance_mode")
        dialog = self.page.findChild(QObject, "performanceConfirmation")
        self.assertIsNotNone(selector)
        self.assertIsNotNone(dialog)
        selector.activated.emit(1)
        self.assertTrue(dialog.property("visible"))
        dialog.reject()
        self.assertEqual(self.page.property("draft").toVariant()["performance_mode"], "normal")
        selector.activated.emit(1)
        dialog.accept()
        self.assertEqual(self.page.property("draft").toVariant()["performance_mode"], "performance")
        self.page.save()
        self.assertFalse(self.page.property("refreshAvailable"))
        with patch("ui.qt.settings_controller.available_cpu_topology", return_value=(12, None, "unknown")):
            self.page.reload()
            state = self.page.property("pageState")
            if hasattr(state, "toVariant"): state = state.toVariant()
            self.assertFalse(state["performanceAvailable"])
            self.assertEqual(state["effectiveMode"], "normal")
            self.assertEqual(state["values"]["performance_mode"], "performance")

    def test_unsaved_navigation_and_close_are_guarded(self):
        self.page.choose("blur_policy", "automatic")
        self.window.showPage("home")
        self.assertEqual(self.window.property("currentPage"), "settings")
        dialog = self.page.findChild(QObject, "unsavedSettingsConfirmation")
        self.assertTrue(dialog.property("visible"))
        dialog.reject()
        self.assertEqual(self.window.property("currentPage"), "settings")
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(dialog.property("visible"))
        dialog.reject()


if __name__ == "__main__": unittest.main()
