import unittest
import xml.etree.ElementTree as ET

from PySide6.QtCore import QRectF, QUrl, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtQml import QQmlEngine, QQmlComponent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from app_paths import asset_path, qml_path
from ui.test_qt_thumbnails import capture_qt_warnings


class LogoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_production_logos_are_vector_and_load_in_qml(self):
        capture_qt_warnings(self)
        for name, viewbox in (("logo_blue.svg", "0 0 273.76 74.44"),
                              ("logo_white.svg", "0 0 273.76 74.44"),
                              ("logo_loading_white.svg", "0 0 1268.67 140.03")):
            with self.subTest(logo=name):
                path = asset_path("branding", name)
                text = path.read_text(encoding="utf-8")
                root = ET.fromstring(text)
                self.assertEqual(viewbox, root.get("viewBox"))
                self.assertNotIn("base64", text)
                self.assertNotIn("clip-path", text)
                self.assertFalse(any(node.tag.rsplit("}", 1)[-1] in ("image", "clipPath") for node in root.iter()))
                renderer = QSvgRenderer(str(path))
                self.assertTrue(renderer.isValid())
                expected_width, expected_height = map(float, viewbox.split()[2:])
                self.assertAlmostEqual(expected_width / expected_height,
                    renderer.viewBoxF().width() / renderer.viewBoxF().height())
                engine = QQmlEngine()
                component = QQmlComponent(engine)
                component.setData(b'import QtQuick\nImage { property url logoUrl; readonly property int loadStatus: status; source: logoUrl }', QUrl.fromLocalFile(str(qml_path("__logo_test.qml"))))
                image = component.createWithInitialProperties({"logoUrl": QUrl.fromLocalFile(str(path))})
                self.assertIsNotNone(image, [error.toString() for error in component.errors()])
                self.assertEqual(1, image.property("loadStatus"))  # Image.Ready
                self.assertAlmostEqual(renderer.defaultSize().width() / renderer.defaultSize().height(),
                                       image.property("implicitWidth") / image.property("implicitHeight"))

    def test_aperture_renders_green_blades_with_an_open_center(self):
        capture_qt_warnings(self)
        # Interior samples check the broken mark without a font/platform-sensitive
        # screenshot comparison. Coordinates are in the original SVG viewBox.
        for name, blade, opening in (("logo_blue.svg", (58, 39), (60, 24)),
                                      ("logo_white.svg", (58, 39), (60, 24)),
                                      ("logo_loading_white.svg", (189, 118), (195, 70))):
            with self.subTest(logo=name):
                renderer = QSvgRenderer(str(asset_path("branding", name)))
                viewbox = renderer.viewBoxF()
                pixels = QImage(round(viewbox.width() * 4), round(viewbox.height() * 4), QImage.Format.Format_ARGB32)
                pixels.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixels)
                renderer.render(painter, QRectF(0, 0, viewbox.width() * 4, viewbox.height() * 4))
                painter.end()
                self.assertEqual(QColor("#d7f205"), pixels.pixelColor(blade[0] * 4, blade[1] * 4))
                self.assertEqual(0, pixels.pixelColor(opening[0] * 4, opening[1] * 4).alpha())
