from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app_paths import asset_path, qml_path, qml_root

PYSIDE6_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE6_AVAILABLE:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent
    from PySide6.QtWidgets import QApplication

    from ui.qt.bridge import QtPhotoCleanerBridge
    from ui.qt.app import configure_quick_controls_style


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not installed")
class QmlDesignSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_quick_controls_style()
        cls.application = QApplication.instance() or QApplication([])

    def _make_engine(self) -> tuple["QQmlApplicationEngine", list[object]]:
        engine = QQmlApplicationEngine()
        engine.addImportPath(str(qml_root()))
        warnings: list[object] = []
        engine.warnings.connect(lambda items: warnings.extend(items))
        return engine, warnings

    def _assert_component_ready(self, component: "QQmlComponent") -> None:
        if component.isError():
            errors = "\n".join(error.toString() for error in component.errors())
            self.fail(errors)

    def test_asset_paths_resolve_and_raw_runtime_assets_are_removed(self) -> None:
        self.assertTrue(qml_path("App.qml").is_file())
        self.assertTrue(qml_path("components", "RoundedCornerMask.qml").is_file())
        self.assertTrue(asset_path("backgrounds", "auth_hero.jpg").is_file())
        self.assertTrue(asset_path("icons", "search.svg").is_file())
        self.assertTrue(asset_path("branding", "logo_blue.svg").is_file())
        self.assertTrue(asset_path("backgrounds", "loading_mesh.svg").is_file())
        self.assertFalse(asset_path("backgrounds", "loading screen - background mesh.svg").exists())
        self.assertFalse(
            asset_path("backgrounds", "page 1 - header -  export button rectangle.svg").exists()
        )

    def test_theme_palette_values_are_exact(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_path("theme", "Theme.qml"))))
        self._assert_component_ready(component)

        theme = component.create()

        expected = {
            "mainBlue": "#00008E",
            "midBlue": "#4B44E0",
            "lightBlue": "#A4A4FF",
            "neonGreen": "#D7F205",
            "white": "#FFFFFF",
            "black": "#000000",
        }
        for property_name, expected_value in expected.items():
            self.assertEqual(theme.property(property_name).name().upper(), expected_value)
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_britanica_font_files_expose_distinct_qt_family_names(self) -> None:
        expected = {
            "Britanica Thin.ttf": ["Britanica-Thin"],
            "Britanica Regular.ttf": ["Britanica-Regular"],
            "Britanica Bold.ttf": ["Britanica-Bold"],
        }

        for file_name, expected_families in expected.items():
            font_id = QFontDatabase.addApplicationFont(str(asset_path("fonts", file_name)))
            self.assertGreaterEqual(font_id, 0, file_name)
            self.assertEqual(QFontDatabase.applicationFontFamilies(font_id), expected_families)

    def test_qml_root_loads_without_qml_warnings(self) -> None:
        bridge = QtPhotoCleanerBridge(parent=self.application)
        self.addCleanup(bridge._background_timer.stop)
        self.addCleanup(bridge._elapsed_timer.stop)
        self.addCleanup(bridge.deleteLater)

        engine, warnings = self._make_engine()
        engine.setInitialProperties({"appBridge": bridge})
        engine.load(qml_path("App.qml").as_uri())

        self.assertTrue(engine.rootObjects())
        self.assertEqual([], [warning.toString() for warning in warnings])

    def test_critical_component_files_instantiate(self) -> None:
        engine, warnings = self._make_engine()
        component = QQmlComponent(engine)
        component.setData(
            b"""
import QtQuick
import "theme" as Room36Theme
import "components" as Components

Item {
    width: 1200
    height: 900

    Room36Theme.Theme { id: theme }

    Components.PrimaryButton { text: "Primary"; iconSource: theme.iconUrl("import.svg") }
    Components.PillButton { text: "Add"; variant: "filled"; y: 70 }
    Components.IconButton { source: theme.iconUrl("settings.svg"); accessibleLabel: "Settings"; y: 130 }
    Components.SearchField { placeholderText: "Search"; y: 190 }
    Components.AuthTextField { label: "Email"; y: 250 }
    Components.ToolbarDropdownButton { label: "Sort"; iconSource: theme.iconUrl("sort.svg"); y: 250 }
    Components.SidebarFilterItem { iconSource: theme.iconUrl("home.svg"); label: "Home"; selected: true; y: 310 }
    Components.SelectionCheckbox { checked: true; text: "Selected"; y: 370 }
    Components.ScoreBar { value: 0.75; y: 430 }
    Components.PersonAvatar { name: "Amina"; count: 4; showName: true; selected: true; y: 490 }
    Components.SocialLoginButton { iconSource: theme.iconUrl("google.svg"); provider: "google"; x: 520; y: 250 }
    Components.RoundedCornerMask { x: 620; y: 250; width: 80; height: 80; cornerRadius: 16; maskColor: theme.homeBackground }
    Components.NewProjectCard { x: 720; y: 80 }
    Components.ProjectGrid { x: 720; y: 450; width: 900; height: 360 }
    Components.PhotoCard {
        source: theme.backgroundUrl("loading_mesh.svg")
        filename: "IMG_2042.jpg"
        selected: true
        favorite: true
        badgeText: "Best"
        x: 140
        y: 490
    }
    Components.ProjectCard {
        source: theme.backgroundUrl("loading_mesh.svg")
        title: "Project"
        subtitle: "36 photos"
        selected: true
        x: 420
        y: 490
    }
}
""",
            QUrl.fromLocalFile(str(qml_path("__component_smoke.qml"))),
        )
        self._assert_component_ready(component)

        root = component.create()
        self.assertIsNotNone(root)
        self.assertEqual([], [warning.toString() for warning in warnings])
