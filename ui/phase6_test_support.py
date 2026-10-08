"""Deterministic backend seams: real Blur/hash/cache, no external ONNX models."""
from automatic_scan.config import AutomaticScanConfig
from automatic_scan.orchestrator import AutomaticScanOrchestrator
from automatic_scan.services import AutomaticScanServices


class NoFacesServices(AutomaticScanServices):
    def initialize_faces(self):
        pass

    def analyze_faces(self, record, **kwargs):
        self._analysis_cache.put(record.path, record.file_size, record.mtime_ns, [],
            width=record.width, height=record.height)
        return []


class FixtureRunner:
    def __init__(self, root, store, gate=None):
        self.root, self.store, self.gate = root, store, gate
        self.sessions = []
        self.services = []

    def process_project(self, folder, paths, **kwargs):
        self.sessions.append((folder, list(paths)))
        service = NoFacesServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        self.services.append(service)
        if self.gate is not None:
            def wait():
                self.gate.wait(10)
                kwargs["cancellation_token"].raise_if_canceled()
            service.initialize_faces = wait
        return AutomaticScanOrchestrator(state_store=self.store).process_project(
            folder, paths, services=service, **kwargs)
