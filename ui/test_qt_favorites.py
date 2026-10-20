"""Phase 8B favorites stay project-scoped and independent of analysis reviews."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from app_state_database import AppStateDatabase
from automatic_scan.state_store import AutomaticStateStore, PersistedImageDecision
from ui.models import ResultItem
from ui.qt.app import configure_quick_controls_style
from ui.qt.blur_model import BlurCandidateModel
from ui.qt.favorites_controller import FavoritesController
from ui.qt import favorites_controller as favorites_module
from ui.qt.favorites_store import FavoritesStore
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.previews import PreviewService
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until
from ui import test_qml_project_flow as phase5


class FavoritesStoreTests(unittest.TestCase):
    def test_store_project_identity_restart_and_independent_review_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder_a, folder_b = root / "a", root / "b"
            folder_a.mkdir(); folder_b.mkdir()
            path = folder_a / "same.jpg"
            path.write_bytes(b"old")
            db_path = root / "app.sqlite3"
            store = FavoritesStore(db_path)
            automatic = AutomaticStateStore(database_path=db_path)
            state = automatic.load(folder_a)
            state.image_decisions[str(path)] = PersistedImageDecision(
                str(path), 3, 1, True, False, False, None,
                duplicate_group_id="group", is_duplicate_keeper=False, review_state="REMOVE")
            state.ignored_unknown_cluster_ids.add("cluster")
            automatic.save(state)
            automatic.save_duplicate_review(folder_a, "group", "signature", [str(path)],
                                            {str(path): "KEEP"}, True)
            before = automatic.load(folder_a)
            duplicate_before = automatic.duplicate_reviews(folder_a)
            store.set_favorite(folder_a, path, True)
            store.set_favorite(folder_a, path, True)
            self.assertEqual(store.favorite_paths(folder_a), {photo_identity(path)})
            self.assertFalse(store.is_favorite(folder_b, path))
            path.write_bytes(b"edited same path, different size")
            self.assertTrue(FavoritesStore(db_path).is_favorite(folder_a, path))
            self.assertEqual(automatic.load(folder_a).image_decisions, before.image_decisions)
            self.assertEqual(automatic.load(folder_a).ignored_unknown_cluster_ids,
                             before.ignored_unknown_cluster_ids)
            self.assertEqual(automatic.duplicate_reviews(folder_a), duplicate_before)
            store.set_favorite(folder_a, path, False)
            self.assertEqual(FavoritesStore(db_path).favorite_paths(folder_a), set())
            self.assertEqual(automatic.load(folder_a).image_decisions, before.image_decisions)
            self.assertEqual(automatic.duplicate_reviews(folder_a), duplicate_before)


class FavoritesFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder_a, self.folder_b = self.root / "project-a", self.root / "project-b"
        self.folder_a.mkdir(); self.folder_b.mkdir()
        for folder in (self.folder_a, self.folder_b):
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (64, 48), "blue").save(folder / name)
        self.db_path = self.root / "app.sqlite3"
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.projects = ProjectController(self.photos, registry_path=self.root / "projects.json",
                                          database_path=self.db_path)
        self.favorites = FavoritesController(self.projects, self.photos, self.thumbnails,
                                             database_path=self.db_path)
        self.blur = BlurCandidateModel(self.thumbnails)
        self.previews = PreviewService()
        self.inspector = PhotoInspectorController(self.blur, self.previews)
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties(dict(startPage="library", projectController=self.projects,
                                              photoModel=self.photos, favoritesController=self.favorites,
                                              inspectorController=self.inspector))
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.app.processEvents()
        self.open_folder(self.folder_a)

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.inspector.close()
        self.favorites.shutdown()
        self.projects.shutdown()
        self.blur.close()
        self.photos.close()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.favorites._worker.is_alive())
        self.temp.cleanup()

    items = phase5.ProjectFlowTests.items
    item = phase5.ProjectFlowTests.item
    click = phase5.ProjectFlowTests.click

    def click_object(self, item):
        QTest.qWait(50)
        position = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
        QTest.mouseClick(self.window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         position.toPoint())
        self.app.processEvents()

    def open_folder(self, folder):
        self.projects.openFolder(str(folder))
        wait_until(lambda: self.projects.activeProjectFolder == photo_identity(folder)
                   and self.favorites.loaded)
        self.app.processEvents()

    def test_library_heart_route_count_favorites_grid_sort_view_and_unfavorite(self):
        a = str(self.folder_a / "a.jpg")
        self.assertEqual(self.favorites.count, 0)
        self.click("libraryNav_favorites")
        self.assertIn("No favorite photos yet", self.item("favoritesGrid").property("emptyText"))
        self.click("libraryNav_library")
        self.click_object(self.items("photoFavoriteButton")[0])
        wait_until(lambda: self.favorites.count == 1)
        self.assertTrue(self.favorites.isFavorite(a))
        self.assertTrue(self.item("gridPhotoCard").property("favorite"))
        self.assertEqual(self.item("libraryNav_favorites").property("count"), 1)
        self.click("libraryNav_favorites")
        self.assertEqual(self.item("favoritesGrid").property("count"), 1)
        self.click("librarySortButton")
        self.click("toolbarDropdownOption_Name Z–A")
        self.click("libraryViewButton")
        self.click("toolbarDropdownOption_Large")
        self.assertEqual(self.item("favoritesPage").property("viewSize"), "Large")
        self.click("photoFavoriteButton")
        wait_until(lambda: self.favorites.count == 0)
        self.assertFalse(self.favorites.isFavorite(a))
        self.assertEqual(self.item("favoritesGrid").property("count"), 0)
        self.assertTrue((self.folder_a / "a.jpg").exists())
        self.click("libraryNav_library")
        self.assertFalse(self.item("gridPhotoCard").property("favorite"))

    def test_inspector_unfavorite_current_keeps_preview_and_navigation_safe(self):
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            self.assertTrue(self.favorites.setFavorite(str(self.folder_a / name), True))
            wait_until(lambda: not self.favorites.isPending(str(self.folder_a / name)))
        self.click("libraryNav_favorites")
        self.click("gridPhotoCard")
        self.assertTrue(self.inspector.opened)
        self.assertEqual(self.inspector.photo["filename"], "a.jpg")
        self.click("inspectorNext")
        self.assertEqual(self.inspector.photo["filename"], "b.jpg")
        self.click("inspectorFavoriteButton")
        wait_until(lambda: self.favorites.count == 2)
        self.assertTrue(self.inspector.opened)
        self.assertEqual(self.inspector.photo["filename"], "b.jpg")
        self.assertTrue(self.inspector.hasNext)
        self.click("inspectorNext")
        self.assertEqual(self.inspector.photo["filename"], "c.jpg")
        self.click("inspectorBack")
        self.assertFalse(self.inspector.opened)
        self.assertEqual(self.item("favoritesGrid").property("count"), 2)

    def test_favorites_sort_and_live_inspector_heart_state(self):
        for name in ("a.jpg", "b.jpg"):
            path = str(self.folder_a / name)
            self.assertTrue(self.favorites.setFavorite(path, True))
            wait_until(lambda: not self.favorites.isPending(path))
        self.click("libraryNav_favorites")
        self.click("librarySortButton")
        self.click("toolbarDropdownOption_Name Z–A")
        filename_role = self.favorites.photoModel.ROLE["filename"]
        self.assertEqual(self.favorites.photoModel.data(self.favorites.photoModel.index(0),
                                                        filename_role), "b.jpg")
        self.click("gridPhotoCard")
        self.assertEqual(self.inspector.photo["filename"], "b.jpg")
        self.assertIn("Remove favorite", self.item("inspectorFavoriteButton").property("accessibleLabel"))
        self.click("inspectorFavoriteButton")
        wait_until(lambda: self.favorites.count == 1)
        self.assertIn("Add favorite", self.item("inspectorFavoriteButton").property("accessibleLabel"))
        self.click("inspectorBack")
        self.assertEqual(self.favorites.photoModel.data(self.favorites.photoModel.index(0),
                                                        filename_role), "a.jpg")
        self.assertTrue(self.favorites.setFavorite(str(self.folder_a / "c.jpg"), True))
        wait_until(lambda: self.favorites.count == 2)
        self.assertEqual(self.favorites.photoModel.data(self.favorites.photoModel.index(0),
                                                        filename_role), "c.jpg")

    def test_project_switch_restart_missing_path_and_same_path_edit(self):
        a = self.folder_a / "a.jpg"
        self.assertTrue(self.favorites.setFavorite(str(a), True))
        wait_until(lambda: self.favorites.count == 1)
        self.projects.openFolder(str(self.folder_b))
        self.assertEqual(self.favorites.count, -1)
        self.assertFalse(self.favorites.isFavorite(str(a)))
        wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.folder_b)
                   and self.favorites.loaded)
        self.assertEqual(self.favorites.count, 0)
        self.open_folder(self.folder_a)
        self.assertEqual(self.favorites.count, 1)
        Image.new("RGB", (90, 70), "red").save(a)
        self.open_folder(self.folder_a)
        self.assertEqual(self.favorites.count, 1)
        a.rename(self.folder_a / "temporarily-away.jpg")
        self.open_folder(self.folder_a)
        self.assertEqual(self.favorites.count, 0)
        self.assertTrue(FavoritesStore(self.db_path).is_favorite(self.folder_a, a))
        (self.folder_a / "temporarily-away.jpg").rename(a)
        self.open_folder(self.folder_a)
        self.assertEqual(self.favorites.count, 1)
        reopened = FavoritesController(self.projects, self.photos, self.thumbnails,
                                       database_path=self.db_path)
        try:
            wait_until(lambda: reopened.loaded)
            self.assertEqual(reopened.count, 1)
        finally:
            reopened.shutdown()

    def test_pending_toggle_rejected_and_write_failure_keeps_truthful_state(self):
        a = str(self.folder_a / "a.jpg")
        entered, release = threading.Event(), threading.Event()
        original = FavoritesStore.set_favorite
        def delayed(store, folder, path, favorite):
            entered.set()
            release.wait(2)
            return original(store, folder, path, favorite)
        with patch.object(FavoritesStore, "set_favorite", delayed):
            self.assertTrue(self.favorites.setFavorite(a, True))
            wait_until(entered.is_set)
            self.assertFalse(self.favorites.setFavorite(a, False))
            self.assertFalse(self.favorites.isFavorite(a))
            release.set()
            wait_until(lambda: self.favorites.count == 1)
        with patch.object(FavoritesStore, "set_favorite", side_effect=sqlite3.OperationalError("locked")):
            with patch.object(favorites_module.log, "exception") as logged:
                self.assertTrue(self.favorites.setFavorite(a, False))
                wait_until(lambda: not self.favorites.isPending(a))
                logged.assert_called_once()
        self.assertEqual(self.favorites.count, 1)
        self.assertTrue(self.favorites.isFavorite(a))
        self.assertIn("could not be saved", self.favorites.error)
        self.assertEqual(self.item("libraryShell").property("currentRoute"), "library")
        self.assertTrue(self.item("favoritesErrorMessage").property("visible"))
        self.assertTrue(self.favorites.setFavorite(a, False))
        wait_until(lambda: self.favorites.count == 0)
        self.assertTrue(self.favorites.setFavorite(a, True))
        wait_until(lambda: self.favorites.count == 1)
        self.assertTrue(FavoritesStore(self.db_path).is_favorite(self.folder_a, a))

    def test_read_failure_is_local_and_reopen_recovers(self):
        with patch.object(FavoritesStore, "favorite_paths", side_effect=sqlite3.OperationalError("locked")):
            with patch.object(favorites_module.log, "exception") as logged:
                self.projects.openFolder(str(self.folder_a))
                wait_until(lambda: self.favorites.error.startswith("Favorites could not be loaded"))
                logged.assert_called_once()
        self.assertEqual(self.favorites.count, -1)
        self.assertEqual(self.photos.rowCount(), 3)
        self.assertEqual(self.item("libraryShell").property("currentRoute"), "library")
        self.click("libraryNav_favorites")
        self.assertIn("could not be loaded", self.item("favoritesMessage").property("text"))
        self.open_folder(self.folder_a)
        self.assertEqual(self.favorites.count, 0)

    def test_favorite_actions_never_call_project_processing(self):
        probe = Mock()
        self.projects._processing = probe
        path = str(self.folder_a / "a.jpg")
        self.assertTrue(self.favorites.setFavorite(path, True))
        wait_until(lambda: self.favorites.count == 1)
        self.assertTrue(self.favorites.setFavorite(path, False))
        wait_until(lambda: self.favorites.count == 0)
        probe.assert_not_called()
        self.assertEqual(probe.mock_calls, [])

    def test_old_project_write_finishes_without_changing_new_project(self):
        a = str(self.folder_a / "a.jpg")
        entered, release = threading.Event(), threading.Event()
        original = FavoritesStore.set_favorite
        def delayed(store, folder, path, favorite):
            entered.set()
            release.wait(2)
            return original(store, folder, path, favorite)
        with patch.object(FavoritesStore, "set_favorite", delayed):
            self.assertTrue(self.favorites.setFavorite(a, True))
            wait_until(entered.is_set)
            self.projects.openFolder(str(self.folder_b))
            self.assertEqual(self.favorites.count, -1)
            release.set()
            wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.folder_b)
                       and self.favorites.loaded)
        self.assertEqual(self.favorites.count, 0)
        self.assertFalse(self.favorites.isFavorite(a))
        self.assertTrue(FavoritesStore(self.db_path).is_favorite(self.folder_a, a))
        self.assertFalse(FavoritesStore(self.db_path).is_favorite(self.folder_b, a))


class FavoritesCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def test_six_thousand_favorites_use_bounded_delegates_and_thumbnails(self):
        capture_qt_warnings(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "large-project"
            folder.mkdir()
            paths = [folder / f"photo-{index:04}.jpg" for index in range(6000)]
            db_path = root / "app.sqlite3"
            database = AppStateDatabase(db_path)
            with database.connect() as db:
                db.execute("INSERT INTO folder_state(folder) VALUES (?)", (photo_identity(folder),))
                db.executemany("INSERT INTO favorite_photos(folder, path) VALUES (?, ?)",
                               ((photo_identity(folder), photo_identity(path)) for path in paths))
            thumbnails = ThumbnailService(decoder=lambda *_: "")
            photos = PhotoListModel(thumbnails)
            items = [ResultItem(path, path.name, "") for path in paths]
            projects = ProjectController(photos, registry_path=root / "projects.json",
                                         database_path=db_path, discover=lambda _: items)
            favorites = FavoritesController(projects, photos, thumbnails, database_path=db_path)
            engine = QQmlApplicationEngine()
            engine.setInitialProperties(dict(startPage="library", photoModel=photos,
                                             projectController=projects, favoritesController=favorites))
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            try:
                projects.openFolder(str(folder))
                wait_until(lambda: favorites.loaded and favorites.count == 6000, timeout=8)
                self.assertLess(thumbnails.decode_count, 100)
                shell = window.findChild(type(window.contentItem()), "libraryShell")
                # Use QML's existing route handler; the grid remains virtualized.
                shell.setProperty("currentRoute", "favorites")
                QTest.qWait(100)
                delegates = []
                def visit(item):
                    if item.objectName() == "photoGridDelegate":
                        delegates.append(item)
                    for child in item.childItems():
                        visit(child)
                visit(window.contentItem())
                self.assertLess(len(delegates), 100)
                self.assertLessEqual(thumbnails.pending_count, thumbnails.max_pending)
            finally:
                window.close()
                engine.deleteLater()
                self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                favorites.shutdown()
                projects.shutdown()
                photos.close()
                thumbnails.shutdown()
                self.assertFalse(favorites._worker.is_alive())
