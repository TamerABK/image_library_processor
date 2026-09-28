from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app_paths import asset_path, qml_path, qml_root

PYSIDE6_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE6_AVAILABLE:
    from PySide6.QtCore import QObject, QPointF, QUrl
    from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent
    from PySide6.QtQuick import QQuickItem
    from PySide6.QtWidgets import QApplication

    from ui.qt.bridge import QtPhotoCleanerBridge
    from ui.qt.app import (
        _home_preview_data_from_environment,
        _start_page_from_environment,
        configure_quick_controls_style,
    )


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed")
class QmlPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_quick_controls_style()
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._engines: list[QQmlApplicationEngine] = []
        self._components: list[QQmlComponent] = []

    def _make_engine(self) -> tuple["QQmlApplicationEngine", list[object]]:
        engine = QQmlApplicationEngine()
        engine.addImportPath(str(qml_root()))
        warnings: list[object] = []
        engine.warnings.connect(lambda items: warnings.extend(items))
        self.addCleanup(lambda: self.assertEqual([], [warning.toString() for warning in warnings]))
        return engine, warnings

    def _create_component(self, relative_path: str):
        engine, warnings = self._make_engine()
        self._engines.append(engine)
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_path(relative_path))))
        self._components.append(component)
        if component.isError():
            self.fail("\n".join(error.toString() for error in component.errors()))
        root = component.create()
        self.assertIsNotNone(root)
        self.assertEqual([], [warning.toString() for warning in warnings])
        return root

    def test_production_pages_instantiate(self) -> None:
        for page_path in (
            "pages/LoadingPage.qml",
            "pages/AuthPage.qml",
            "pages/HomePage.qml",
            "pages/LibraryShell.qml",
        ):
            with self.subTest(page_path=page_path):
                self._create_component(page_path)

    def test_production_startup_selects_style_before_loading_controls(self) -> None:
        # Qt locks its style once Controls are registered: use fresh processes.
        script = textwrap.dedent("""
            from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler
            from PySide6.QtQuickControls2 import QQuickStyle
            from PySide6.QtWidgets import QApplication
            from ui.qt.app import run_app
            messages = []
            def capture(kind, context, message):
                if kind in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                    messages.append(message)
            qInstallMessageHandler(capture)
            app = QApplication([])
            QTimer.singleShot(100, app.quit)
            assert run_app() == 0
            print(QQuickStyle.name())
            assert not messages, messages
        """)
        for override, expected, page in ((None, "Basic", "auth"), ("Fusion", "Fusion", "auth"),
                                         (None, "Basic", "library"), ("Fusion", "Fusion", "library")):
            with self.subTest(style=expected, page=page):
                environment = dict(os.environ)
                environment.pop("QT_QUICK_CONTROLS_STYLE", None)
                environment["ROOM36_START_PAGE"] = page
                environment["QT_QPA_PLATFORM"] = "offscreen"
                # The Windows offscreen plugin has no system font discovery.
                # Supply real bundled fonts rather than filtering its warnings.
                environment["QT_QPA_FONTDIR"] = str(asset_path("fonts"))
                environment["QML_DISABLE_DISK_CACHE"] = "1"
                if override:
                    environment["QT_QUICK_CONTROLS_STYLE"] = override
                result = subprocess.run(
                    [sys.executable, "-B", "-c", script],
                    cwd=qml_root().parent,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_loading_route_is_presentation_until_owner_changes_page(self) -> None:
        engine, warnings = self._make_engine()
        engine.setInitialProperties({"startPage": "loading"})
        engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(engine.rootObjects())
        window = engine.rootObjects()[0]
        loader = window.findChild(QObject, "pageLoader")
        self.application.processEvents()
        page = loader.property("item")
        self.assertEqual(window.property("currentPage"), "loading")
        self.assertEqual(page.metaObject().indexOfSignal("continueRequested()"), -1)
        window.setProperty("startPage", "home")
        self.application.processEvents()
        self.assertEqual(loader.property("item").property("objectName"), "homePage")
        self.assertEqual([], warnings)

    def test_auth_form_state_and_submission_survive_resizing(self) -> None:
        page = self._create_component("pages/AuthPage.qml")
        names = ("authFullName", "authEmail", "authPassword")
        fields = [page.findChild(QObject, name) for name in names]
        values = ("Test User", "test@example.invalid", "test-only-password")
        for field, value in zip(fields, values):
            self.assertIsNotNone(field)
            field.setProperty("text", value)
        submissions = []
        page.signUpRequested.connect(lambda *args: submissions.append(args))
        for width, height, compact in ((1920, 1080, False), (1440, 900, True),
                                       (1280, 720, True), (1920, 1080, False)):
            with self.subTest(size=(width, height)):
                page.setProperty("width", width)
                page.setProperty("height", height)
                self.application.processEvents()
                self.assertEqual(page.property("compactLayout"), compact)
                self.assertEqual([page.findChild(QObject, name) for name in names], fields)
                self.assertEqual(tuple(field.property("text") for field in fields), values)
                self.assertEqual(tuple(page.property(key) for key in ("fullName", "email", "password")), values)
                page.findChild(QObject, "authStart").clicked.emit()
                self.assertEqual(submissions[-1], values)
        fields[1].setProperty("text", "updated@example.invalid")
        self.assertEqual(page.property("email"), "updated@example.invalid")

    def test_auth_actions_fit_default_and_reference_sizes(self) -> None:
        page = self._create_component("pages/AuthPage.qml")
        viewport = page.findChild(QQuickItem, "authFormViewport")
        for width, height in ((1440, 900), (1280, 720), (1920, 1080)):
            with self.subTest(size=(width, height)):
                page.setProperty("width", width)
                page.setProperty("height", height)
                self.application.processEvents()
                self.assertFalse(viewport.property("interactive"))
                for name in ("authFullName", "authEmail", "authPassword", "authStart", "authLoginRow"):
                    item = page.findChild(QQuickItem, name)
                    top_left = item.mapToItem(page, QPointF(0, 0))
                    self.assertGreaterEqual(top_left.x(), 0, name)
                    self.assertGreaterEqual(top_left.y(), 0, name)
                    self.assertLessEqual(top_left.x() + item.width(), width, name)
                    self.assertLessEqual(top_left.y() + item.height(), height, name)

    def test_auth_scroll_reaches_actions_at_constrained_height(self) -> None:
        page = self._create_component("pages/AuthPage.qml")
        page.setProperty("width", 800)
        page.setProperty("height", 400)
        self.application.processEvents()
        viewport = page.findChild(QQuickItem, "authFormViewport")
        self.assertTrue(viewport.property("interactive"))
        viewport.setProperty("contentY", viewport.property("contentHeight") - viewport.height())
        self.application.processEvents()
        for name in ("authStart", "authLoginRow"):
            item = page.findChild(QQuickItem, name)
            top = item.mapToItem(page, QPointF(0, 0)).y()
            self.assertGreaterEqual(top, 0, name)
            self.assertLessEqual(top + item.height(), page.height(), name)

    def test_start_page_environment_routing(self) -> None:
        for page in ("loading", "auth", "home", "library", "gallery"):
            with self.subTest(page=page):
                with patch.dict(os.environ, {"ROOM36_START_PAGE": page}, clear=False):
                    self.assertEqual(_start_page_from_environment(), page)

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_start_page_from_environment(), "home")

        with patch.dict(os.environ, {"ROOM36_START_PAGE": "unsupported"}, clear=False):
            self.assertEqual(_start_page_from_environment(), "home")

    def test_app_router_loads_each_page_without_qml_warnings(self) -> None:
        expected_objects = {
            "loading": "loadingPage",
            "auth": "authPage",
            "home": "homePage",
            "library": "libraryShell",
            "gallery": "componentGallery",
        }

        for page, expected_object_name in expected_objects.items():
            with self.subTest(page=page):
                bridge = QtPhotoCleanerBridge(parent=self.application)
                self.addCleanup(bridge._background_timer.stop)
                self.addCleanup(bridge._elapsed_timer.stop)
                self.addCleanup(bridge.deleteLater)

                engine, warnings = self._make_engine()
                engine.setInitialProperties({"appBridge": bridge, "startPage": page})
                engine.load(qml_path("App.qml").as_uri())
                roots = engine.rootObjects()
                self.application.processEvents()

                self.assertTrue(roots)
                self.assertEqual(page, roots[0].property("currentPage"))
                loader = roots[0].findChild(QObject, "pageLoader")
                self.assertIsNotNone(loader)
                loaded_item = loader.property("item")
                self.assertEqual(loaded_item.property("objectName"), expected_object_name)
                self.assertEqual([], [warning.toString() for warning in warnings])

    def test_home_preview_data_is_dev_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_home_preview_data_from_environment())

        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"ROOM36_HOME_PREVIEW_DATA": value}, clear=False):
                    self.assertTrue(_home_preview_data_from_environment())

    def test_app_home_preview_route_is_explicit(self) -> None:
        bridge = QtPhotoCleanerBridge(parent=self.application)
        self.addCleanup(bridge._background_timer.stop)
        self.addCleanup(bridge._elapsed_timer.stop)
        self.addCleanup(bridge.deleteLater)

        engine, warnings = self._make_engine()
        engine.setInitialProperties(
            {
                "appBridge": bridge,
                "startPage": "home",
                "useHomePreviewData": True,
            }
        )
        engine.load(qml_path("App.qml").as_uri())
        roots = engine.rootObjects()
        self.application.processEvents()

        self.assertTrue(roots)
        loader = roots[0].findChild(QObject, "pageLoader")
        self.assertIsNotNone(loader)
        self.assertEqual(loader.property("item").property("objectName"), "screenPreview")
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_opaque_root_background_semantics(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_path("theme", "Theme.qml"))))
        if component.isError():
            self.fail("\n".join(error.toString() for error in component.errors()))
        theme = component.create()

        expected = {
            "appBackground": "#FFFFFF",
            "homeBackground": "#FFFFFF",
            "authBackground": "#FFFFFF",
            "libraryBackground": "#000000",
        }
        for property_name, expected_color in expected.items():
            color = theme.property(property_name)
            self.assertEqual(color.name().upper(), expected_color)
            self.assertEqual(color.alpha(), 255)

        self.assertEqual(theme.property("galleryBackground").alpha(), 255)
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_project_card_metrics_are_source_derived(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_path("theme", "Metrics.qml"))))
        if component.isError():
            self.fail("\n".join(error.toString() for error in component.errors()))
        metrics = component.create()

        self.assertAlmostEqual(float(metrics.property("projectCardReferenceWidth")), 418.26, places=2)
        self.assertAlmostEqual(float(metrics.property("projectCardImageHeight")), 278.83, places=2)
        self.assertAlmostEqual(float(metrics.property("projectCardFooterHeight")), 63.69, places=2)
        self.assertAlmostEqual(float(metrics.property("projectCardReferenceHeight")), 342.52, places=2)
        self.assertAlmostEqual(float(metrics.property("projectCardHorizontalGap")), 28.73, places=2)
        self.assertAlmostEqual(float(metrics.property("photoCardReferenceWidth")), 252.51, places=2)
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_auth_hero_asset_resolves(self) -> None:
        self.assertTrue(asset_path("backgrounds", "auth_hero.jpg").is_file())

        auth_page = self._create_component("pages/AuthPage.qml")
        hero_url = auth_page.property("heroSource")
        self.assertIsInstance(hero_url, QUrl)
        self.assertTrue(hero_url.toLocalFile().endswith("assets/backgrounds/auth_hero.jpg"))

    def test_auth_compact_layout_hides_hero_panel(self) -> None:
        auth_page = self._create_component("pages/AuthPage.qml")
        auth_page.setProperty("width", 1280)
        auth_page.setProperty("height", 720)
        self.application.processEvents()

        hero_panel = auth_page.findChild(QObject, "authHeroPanel")
        self.assertIsNotNone(hero_panel)
        self.assertFalse(hero_panel.property("visible"))

        auth_page.setProperty("width", 1920)
        auth_page.setProperty("height", 1080)
        self.application.processEvents()
        self.assertTrue(hero_panel.property("visible"))

    def test_primary_button_default_width_is_not_auth_width(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine)
        component.setData(
            b"""
import QtQuick
import "components" as Components

Components.PrimaryButton {
    text: "Start"
}
""",
            QUrl.fromLocalFile(str(qml_path("__primary_button_smoke.qml"))),
        )
        if component.isError():
            self.fail("\n".join(error.toString() for error in component.errors()))

        button = component.create()
        self.assertIsNotNone(button)
        self.assertLess(float(button.property("implicitWidth")), 445.98)
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_home_reference_grid_produces_four_columns_at_1920(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine)
        component.setData(
            b"""
import QtQuick
import "components" as Components

Components.ProjectGrid {
    width: 1767.54
    height: 740
}
""",
            QUrl.fromLocalFile(str(qml_path("__project_grid_smoke.qml"))),
        )
        if component.isError():
            self.fail("\n".join(error.toString() for error in component.errors()))

        grid = component.create()
        self.assertIsNotNone(grid)
        self.assertEqual(grid.property("columnCount"), 4)
        self.assertEqual([], [warning.toString() for warning in warnings])
