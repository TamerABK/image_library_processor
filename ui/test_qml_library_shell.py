"""Phase 4 navigation contracts; no scans or persistent project data."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QPoint, QPointF, Qt, QtMsgType, qInstallMessageHandler
from PySide6.QtQml import QQmlApplicationEngine, QQmlProperty
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path, qml_root
from ui.qt.app import configure_quick_controls_style


class LibraryShellTests(unittest.TestCase):
    routes = ("library", "blurry", "duplicates", "favorites", "knownPeople", "unknownPeople", "trash")

    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        self.previous_handler = qInstallMessageHandler(self.capture)
        self.engine = QQmlApplicationEngine()
        self.engine.addImportPath(str(qml_root()))
        self.engine.warnings.connect(lambda warnings: self.messages.extend(w.toString() for w in warnings))

    def capture(self, kind, context, message):
        if kind in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            self.messages.append(message)

    def tearDown(self):
        for root in self.engine.rootObjects():
            root.close()
        self.engine.deleteLater()
        self.app.processEvents()
        qInstallMessageHandler(self.previous_handler)
        self.assertEqual([], self.messages)

    def load(self, route="library", preview=False):
        self.engine.setInitialProperties({"startPage": route, "useHomePreviewData": preview})
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects(), self.messages)
        self.window = self.engine.rootObjects()[0]
        self.app.processEvents()
        return self.page()

    def page(self):
        return self.window.findChild(QObject, "pageLoader").property("item")

    def item(self, name):
        # Repeater delegates have a visual parent separate from QObject ownership.
        def find(item):
            if item.objectName() == name:
                return item
            for child in item.childItems():
                match = find(child)
                if match is not None:
                    return match
        found = find(self.window.contentItem())
        self.assertIsNotNone(found, name)
        return found

    def test_routes_selection_and_persistent_shell(self):
        shell = self.load()
        self.assertEqual("library", shell.property("currentRoute"))
        sidebar, topbar = self.item("librarySidebar"), self.item("libraryTopBar")
        shell.setProperty("projectId", "test-project")
        shell.setProperty("projectName", "Test project")
        for route in self.routes:
            self.item("libraryNav_" + route).clicked.emit()
            self.app.processEvents()
            self.assertEqual(route, shell.property("currentRoute"))
            if route == "library":
                self.assertIsNotNone(self.item("photoGrid"))
            else:
                self.assertEqual(route, self.item("libraryRoutePlaceholder").property("routeId"))
            self.assertIs(shell, self.page())
            self.assertIs(sidebar, self.item("librarySidebar"))
            self.assertIs(topbar, self.item("libraryTopBar"))
            self.assertEqual("test-project", shell.property("projectId"))
            self.assertEqual("Test project", topbar.property("projectName"))
            for candidate in self.routes:
                self.assertEqual(candidate == route, self.item("libraryNav_" + candidate).property("selected"))

    def test_invalid_routes_fall_back(self):
        self.load("invalid")
        self.assertEqual("home", self.window.property("currentPage"))
        self.window.setProperty("startPage", "library")
        shell = self.page()
        shell.setProperty("currentRoute", "invalid")
        self.assertEqual("library", shell.property("currentRoute"))
        self.assertTrue(self.item("libraryNav_library").property("selected"))

    def test_preview_project_activation_and_return_home(self):
        self.load("home", preview=True)
        for index in (0, 1):
            card = self.item("projectCard_" + str(index))
            expected_id, expected_name = card.property("projectIdentifier"), card.property("title")
            card.clicked.emit()
            self.app.processEvents()
            self.assertEqual("library", self.window.property("currentPage"))
            self.assertEqual(expected_id, self.page().property("projectId"))
            self.assertEqual(expected_name, self.page().property("projectName"))
            self.item("libraryHomeButton").clicked.emit()
            self.app.processEvents()
            self.assertEqual("home", self.window.property("currentPage"))

    def test_production_home_contract_and_invalid_project(self):
        home = self.load("home")
        home.projectActivated.emit(None, "Invalid")
        self.assertEqual("home", self.window.property("currentPage"))
        home.projectActivated.emit("real-adapter-id", "Adapter project")
        self.assertEqual("library", self.window.property("currentPage"))
        self.assertEqual("real-adapter-id", self.page().property("projectId"))
        self.item("librarySidebarHome").clicked.emit()
        self.assertEqual("home", self.window.property("currentPage"))

    def test_counts_are_optional_metadata(self):
        shell = self.load()
        for route in self.routes:
            self.assertEqual(-1, self.item("libraryNav_" + route).property("count"))
        shell.setProperty("routeCounts", {"library": 42, "blurry": 0, "duplicates": -3})
        self.assertEqual(42, self.item("libraryNav_library").property("count"))
        self.assertEqual(0, self.item("libraryNav_blurry").property("count"))
        self.assertEqual(-1, self.item("libraryNav_duplicates").property("count"))

    def test_toolbar_contracts_carry_context(self):
        self.load("home").projectActivated.emit("project-a", "Project A")
        self.item("libraryNav_duplicates").clicked.emit()
        events = []
        self.window.librarySortRequested.connect(lambda *args: events.append(args))
        self.window.libraryViewSizeRequested.connect(lambda *args: events.append(args))
        self.window.settingsRequested.connect(lambda: events.append(("settings",)))
        self.item("librarySortButton").optionSelected.emit("Date")
        self.item("libraryViewButton").optionSelected.emit("Medium")
        self.item("libraryPreferences").clicked.emit()
        self.assertEqual([("project-a", "duplicates", "Date"), ("project-a", "duplicates", "Medium"), ("settings",)], events)

    def test_keyboard_navigation(self):
        shell = self.load()
        button = self.item("libraryNav_blurry")
        button.forceActiveFocus()
        self.assertTrue(button.hasActiveFocus())
        QTest.keyClick(self.window, Qt.Key.Key_Space)
        self.assertEqual("blurry", shell.property("currentRoute"))

    def test_toolbar_state_overlays_preserve_library_base_fill(self):
        self.load()
        for name in ("librarySortButton", "libraryViewButton"):
            with self.subTest(button=name):
                button = self.item(name)
                base = button.findChild(QObject, "toolbarDropdownBase")
                overlay = button.findChild(QObject, "toolbarDropdownStateOverlay")
                border = button.findChild(QObject, "toolbarDropdownBorder")
                self.assertEqual("#d7f205", button.property("normalFill").name())
                self.assertIs(overlay.parent(), base)
                self.assertIs(border.parent(), base)

                def check_state(alpha, focused=False):
                    # Allow the existing 110 ms color animation to settle.
                    QTest.qWait(150)
                    self.assertEqual(button.property("normalFill"), base.property("color"))
                    self.assertEqual(255, base.property("color").alpha())
                    self.assertAlmostEqual(alpha, overlay.property("color").alphaF(), places=2)
                    self.assertEqual(2 if focused else 1, QQmlProperty.read(border, "border.width"))

                self.item("libraryHomeButton").forceActiveFocus()
                QTest.mouseMove(self.window, QPoint(1100, 600))
                check_state(0)
                center = button.mapToScene(QPointF(button.width() / 2, button.height() / 2)).toPoint()
                QTest.mouseMove(self.window, center)
                self.assertTrue(button.property("hovered"))
                check_state(0.08)
                QTest.mouseMove(self.window, QPoint(1100, 600))
                button.forceActiveFocus()
                check_state(0.08, focused=True)
                QTest.keyPress(self.window, Qt.Key.Key_Space)
                self.assertTrue(button.property("down"))
                check_state(0.16, focused=True)
                QTest.keyRelease(self.window, Qt.Key.Key_Space)
                QTest.qWait(200)
                self.assertTrue(button.property("open"))
                check_state(0.16, focused=True)
                QTest.keyClick(self.window, Qt.Key.Key_Escape)
                QTest.qWait(200)
                self.assertFalse(button.property("open"))
                self.item("libraryHomeButton").forceActiveFocus()
                button.setProperty("enabled", False)
                check_state(0.07)
                button.setProperty("enabled", True)
                check_state(0)

    def test_supported_desktop_geometry(self):
        shell = self.load()
        for width, height, sidebar_width in ((1920, 1080, 430), (1440, 900, 320), (1280, 720, 320)):
            with self.subTest(size=(width, height)):
                self.window.setWidth(width)
                self.window.setHeight(height)
                QTest.qWait(20)
                sidebar = self.item("librarySidebar")
                topbar = self.item("libraryTopBar")
                content = self.item("libraryContentLoader")
                self.assertEqual(sidebar_width, sidebar.width())
                self.assertEqual(height, sidebar.height())
                self.assertEqual(sidebar.width(), topbar.x())
                self.assertGreaterEqual(content.x(), sidebar.width())
                self.assertGreaterEqual(content.y(), topbar.height())
                for name in ("libraryTopBar", "libraryContentLoader", "libraryHomeButton", "librarySortButton", "libraryViewButton", "libraryPageTitle", "libraryProjectName"):
                    item = self.item(name)
                    point = item.mapToItem(shell, QPointF(0, 0))
                    self.assertGreater(item.width(), 0, name)
                    self.assertGreater(item.height(), 0, name)
                    self.assertGreaterEqual(point.x(), 0, name)
                    self.assertGreaterEqual(point.y(), 0, name)
                    self.assertLessEqual(point.x() + item.width(), width + 0.01, name)
                    self.assertLessEqual(point.y() + item.height(), height + 0.01, name)
                scroll = self.item("librarySidebarScroll")
                flickable = scroll.property("contentItem")
                flickable.setProperty("contentY", max(0, flickable.property("contentHeight") - flickable.height()))
                self.app.processEvents()
                last = self.item("libraryPreferences")
                point = last.mapToItem(shell, QPointF(0, 0))
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.y() + last.height(), height)
