"""Source and Qt loading contracts for the bundled Fluent subset."""
from __future__ import annotations

import os
import unittest
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage, QPainter
from PySide6.QtQml import QQmlComponent, QQmlEngine
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from app_paths import asset_path, qml_path
from scripts.sync_fluent_icons import ICONS


class FluentIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_every_declared_variant_is_bundled_and_valid_svg(self):
        for _group, (stem, sizes) in ICONS.items():
            for size in sizes:
                for style in ("regular", "filled"):
                    with self.subTest(stem=stem, size=size, style=style):
                        path = asset_path("icons", "fluent", f"{stem}_{size}_{style}.svg")
                        self.assertTrue(path.is_file(), path)
                        root = ET.parse(path).getroot()
                        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
                        self.assertEqual(root.attrib["viewBox"], f"0 0 {size} {size}")

    def test_component_loads_regular_and_selected_filled_in_qt(self):
        engine = QQmlEngine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(qml_path("components", "FluentIcon.qml"))))
        self.assertFalse(component.isError(), component.errors())
        icon = component.create()
        self.assertIsNotNone(icon, component.errors())
        icon.setProperty("name", "home")
        icon.setProperty("size", 20)
        for _ in range(10):
            self.app.processEvents()
        self.assertEqual(icon.property("effectiveVariant"), "regular")
        self.assertTrue(icon.property("ready"))
        icon.setProperty("color", "#ffffff")
        self.assertEqual(icon.property("colorSuffix"), "_white")
        for _ in range(10):
            self.app.processEvents()
        self.assertTrue(icon.property("ready"))
        icon.setProperty("selected", True)
        for _ in range(10):
            self.app.processEvents()
        self.assertEqual(icon.property("effectiveVariant"), "filled")
        self.assertTrue(icon.property("ready"))

    def test_theme_palette_variants_preserve_original_geometry(self):
        original = asset_path("icons", "fluent", "home_20_regular.svg").read_bytes()
        for name, fill in (("white", b"#FFFFFF"), ("blue", b"#00008E"),
                           ("midblue", b"#4B44E0")):
            with self.subTest(name=name):
                tinted = asset_path("icons", "fluent", f"home_20_regular_{name}.svg").read_bytes()
                self.assertEqual(tinted, original.replace(b"#212121", fill))
        warning = asset_path("icons", "fluent", "warning_20_regular.svg").read_bytes()
        for name, fill in (("red", b"#D92D20"), ("yellow", b"#B77900")):
            tinted = asset_path("icons", "fluent", f"warning_20_regular_{name}.svg").read_bytes()
            self.assertEqual(tinted, warning.replace(b"#212121", fill))

    def test_qt_svg_renderer_draws_at_standard_and_double_density(self):
        path = asset_path("icons", "fluent", "home_20_regular.svg")
        renderer = QSvgRenderer(str(path))
        self.assertTrue(renderer.isValid())
        for edge in (20, 40):
            with self.subTest(edge=edge):
                image = QImage(edge, edge, QImage.Format_ARGB32_Premultiplied)
                image.fill(0)
                painter = QPainter(image)
                renderer.render(painter)
                painter.end()
                self.assertFalse(image.isNull())
                self.assertTrue(any(image.pixelColor(x, y).alpha() > 0
                                    for x in range(edge) for y in range(edge)))


if __name__ == "__main__":
    unittest.main()
