from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


@dataclass(slots=True)
class ResultItem:
    path: Path
    title: str
    detail: str
    recommended_delete: bool = False
    person_id: int | None = None


@dataclass(slots=True)
class ResultGroup:
    title: str
    items: list[ResultItem]
    group_type: str = "default"


@dataclass(slots=True)
class ConfirmedUnknownPerson:
    person_id: int
    name: str


@dataclass(slots=True)
class PreviewItem:
    path: Path
    title: str
    detail: str
    group_title: str


@dataclass(slots=True)
class AppState:
    folder: str = ""
    file_type: str = "All supported"
    available_file_types: tuple[str, ...] = ("All supported",)
    orientation: str = "All pictures"
    available_orientations: tuple[str, ...] = (
        "All pictures",
        "Landscape only",
        "Portrait only",
    )
    known_people_only: bool = False
    auto_export_faces: bool = False
    mode: str = "duplicates"
    status: str = "Choose a folder to start."
    count_text: str = ""
    elapsed_text: str = "Elapsed: 00:00"
    page_label: str = ""
    face_group_label: str = ""
    face_group_labels: tuple[str, ...] = ()
    show_face_options: bool = False
    show_face_selector: bool = False
    show_pagination: bool = False
    can_show_previous_page: bool = False
    can_show_next_page: bool = False
    can_scan: bool = True
    can_delete: bool = False
    can_export: bool = False
    progress_mode: str = "determinate"
    progress_value: int = 0
    progress_max: int = 100


@dataclass(slots=True)
class ScanProgressMessage:
    mode: str
    phase: str
    done: int
    total: int | None
    known_people_only: bool


@dataclass(slots=True)
class UnknownFacesMessage:
    face_result: Any


@dataclass(slots=True)
class ScanResultMessage:
    mode: str
    results: list[ResultGroup]
    known_people_only: bool


@dataclass(slots=True)
class ScanErrorMessage:
    message: str


BackgroundMessage: TypeAlias = (
    ScanProgressMessage
    | UnknownFacesMessage
    | ScanResultMessage
    | ScanErrorMessage
)


@dataclass(slots=True)
class UnknownFacePrompt:
    cluster_id: int
    face_count: int
    representative_path: Path
    preview_array: Any
    suggestion_names: tuple[str, ...]
    cluster: Any


@dataclass(slots=True)
class DeleteResult:
    deleted_count: int
    errors: list[str]


@dataclass(slots=True)
class ExportResult:
    exported_count: int
    errors: list[str]
