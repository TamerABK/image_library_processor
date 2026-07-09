from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from blur_detector.blur_detector import BlurDetector
from duplicate_detector.duplicate_detector import DuplicateDetector
from duplicate_detector.models import DuplicateGroup


@dataclass(slots=True)
class ResultItem:
    path: Path
    title: str
    detail: str
    recommended_delete: bool = False


@dataclass(slots=True)
class ResultGroup:
    title: str
    items: list[ResultItem]


class PhotoCleanerApp:
    thumbnail_size = (220, 220)
    card_min_width = 260

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Image Deduplicator")
        self.root.geometry("1200x800")
        self.root.minsize(980, 640)

        self.folder_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="duplicates")
        self.status_var = tk.StringVar(value="Choose a folder to start.")
        self.count_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00")

        self._queue: Queue[tuple] = Queue()
        self._scan_thread: threading.Thread | None = None
        self._scan_start_time: float | None = None
        self._elapsed_after_id: str | None = None
        self._results: list[ResultGroup] = []
        self._selection_vars: dict[Path, tk.BooleanVar] = {}
        self._thumbnails: dict[Path, ImageTk.PhotoImage] = {}
        self._relayout_after_id: str | None = None
        self._last_progress_percent = -1

        self._build_ui()
        self.root.after(100, self._poll_queue)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Folder").grid(row=0, column=0, sticky="w")
        folder_entry = ttk.Entry(controls, textvariable=self.folder_var)
        folder_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        ttk.Button(controls, text="Browse", command=self._browse_folder).grid(
            row=0, column=2, sticky="ew"
        )

        mode_frame = ttk.Frame(controls)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(mode_frame, text="Scan mode").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Near duplicates",
            value="duplicates",
            variable=self.mode_var,
        ).grid(row=0, column=1, padx=(12, 0))
        ttk.Radiobutton(
            mode_frame,
            text="Blurry photos",
            value="blurry",
            variable=self.mode_var,
        ).grid(row=0, column=2, padx=(12, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        self.scan_button = ttk.Button(actions, text="Scan folder", command=self._start_scan)
        self.scan_button.grid(row=0, column=0, sticky="w")

        self.delete_button = ttk.Button(
            actions,
            text="Delete selected",
            command=self._delete_selected,
            state="disabled",
        )
        self.delete_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        status = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        status.grid(row=1, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        status.columnconfigure(1, weight=0)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.count_var).grid(row=0, column=1, sticky="e")
        ttk.Label(status, textvariable=self.elapsed_var).grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        results_outer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        results_outer.grid(row=2, column=0, sticky="nsew")
        results_outer.columnconfigure(0, weight=1)
        results_outer.rowconfigure(0, weight=1)

        self.results_canvas = tk.Canvas(results_outer, highlightthickness=0)
        self.results_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            results_outer,
            orient="vertical",
            command=self.results_canvas.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_frame.columnconfigure(0, weight=1)
        self._results_window = self.results_canvas.create_window(
            (0, 0),
            window=self.results_frame,
            anchor="nw",
        )

        self.results_frame.bind("<Configure>", self._sync_scroll_region)
        self.results_canvas.bind("<Configure>", self._resize_results_frame)

        self._render_empty_state("Choose a folder and scan to see photos.")

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def _start_scan(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return

        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid folder", "Choose a valid folder to scan.")
            return

        self._reset_results()
        self.scan_button.configure(state="disabled")
        self.delete_button.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._last_progress_percent = -1
        self._scan_start_time = time.monotonic()
        self._schedule_elapsed_update()
        mode = self.mode_var.get()
        self.status_var.set(f"Scanning {self._scan_target_label(mode)}...")
        self.count_var.set("")
        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(folder, mode),
            daemon=True,
        )
        self._scan_thread.start()

    def _scan_worker(self, folder: Path, mode: str) -> None:
        try:
            if mode == "duplicates":
                detector = DuplicateDetector()

                def progress_callback(phase, done, total):
                    self._queue.put(("progress", mode, phase.value, done, total))

                groups = detector.find_duplicates(folder, progress_callback)
                results = self._build_duplicate_results(groups)
            else:
                detector = BlurDetector()

                def progress_callback(done, total):
                    self._queue.put(("progress", mode, "scanning", done, total))

                blur_results = detector.scan_folder(folder, progress_callback)
                results = self._build_blurry_results(blur_results)

            self._queue.put(("result", mode, results))
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self._queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]

                if kind == "progress":
                    _, mode, phase, done, total = message
                    self._update_scan_progress(mode, phase, done, total)
                    continue

                if kind == "result":
                    _, mode, results = message
                    self._handle_scan_result(mode, results)
                    continue

                if kind == "error":
                    _, error_text = message
                    self._finish_scan(error_text, error=True)
        except Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _handle_scan_result(self, mode: str, results: list[ResultGroup]) -> None:
        self._results = results
        self._render_results()

        total_items = sum(len(group.items) for group in results)
        if total_items == 0:
            if mode == "duplicates":
                self._finish_scan("No near duplicates found.")
            else:
                self._finish_scan("No blurry photos found.")
            return

        if mode == "duplicates":
            self._finish_scan(f"Found {len(results)} duplicate groups with {total_items} photos.")
        else:
            self._finish_scan(f"Found {total_items} blurry photos.")

    def _finish_scan(self, status: str, error: bool = False) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._last_progress_percent = -1
        self.scan_button.configure(state="normal")
        self.status_var.set(status)
        self._update_elapsed_label(final=True)
        self._scan_start_time = None
        if self._elapsed_after_id is not None:
            self.root.after_cancel(self._elapsed_after_id)
            self._elapsed_after_id = None
        if error:
            messagebox.showerror("Scan failed", status)
        self._update_action_state()

    def _update_scan_progress(self, mode: str, phase: str, done: int, total: int | None) -> None:
        target = self._scan_target_label(mode)
        if not total or total <= 0:
            self.status_var.set(f"Scanning {target}...")
            return

        progress_percent = self._compute_progress_percent(mode, phase, done, total)
        progress_percent = max(self._last_progress_percent, progress_percent)

        if progress_percent <= self._last_progress_percent:
            return

        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=progress_percent)
        self._last_progress_percent = progress_percent

        if phase == "scanning":
            self.status_var.set(f"Scanning {target} {done}/{total}")
        else:
            self.status_var.set(f"Scanning {target}: {phase.title()} {done}/{total}")

    @staticmethod
    def _compute_progress_percent(mode: str, phase: str, done: int, total: int) -> int:
        clamped_done = min(max(done, 0), total)

        if mode != "duplicates":
            return int((clamped_done / total) * 100)

        if phase == "indexing":
            return int((clamped_done / total) * 50)

        if phase == "matching":
            return 50 + int((clamped_done / total) * 50)

        return int((clamped_done / total) * 100)

    def _build_duplicate_results(self, groups: list[DuplicateGroup]) -> list[ResultGroup]:
        results: list[ResultGroup] = []

        for index, group in enumerate(groups, start=1):
            items: list[ResultItem] = []
            best = group.best
            for photo in group.photos:
                recommended_delete = photo.path != best.path
                items.append(
                    ResultItem(
                        path=photo.path,
                        title=photo.filename,
                        detail=f"{photo.width}x{photo.height} • {photo.file_size // 1024} KB",
                        recommended_delete=recommended_delete,
                    )
                )

            results.append(
                ResultGroup(
                    title=f"Duplicate group {index} ({len(items)} photos)",
                    items=items,
                )
            )

        return results

    def _build_blurry_results(self, blur_results) -> list[ResultGroup]:
        items = [
            ResultItem(
                path=result.path,
                title=result.path.name,
                detail=f"Score {result.result.final_score:.3f} • {result.result.status}",
                recommended_delete=True,
            )
            for result in sorted(
                blur_results,
                key=lambda item: (item.result.final_score, item.path.name),
            )
        ]

        if not items:
            return []

        return [ResultGroup(title=f"Blurry photos ({len(items)})", items=items)]

    @staticmethod
    def _scan_target_label(mode: str) -> str:
        return "near duplicates" if mode == "duplicates" else "blurry photos"

    def _schedule_elapsed_update(self) -> None:
        if self._elapsed_after_id is not None:
            self.root.after_cancel(self._elapsed_after_id)

        self._update_elapsed_label()

    def _update_elapsed_label(self, final: bool = False) -> None:
        if self._scan_start_time is None:
            return

        elapsed_seconds = max(0, int(time.monotonic() - self._scan_start_time))
        self.elapsed_var.set(f"Elapsed: {self._format_elapsed(elapsed_seconds)}")

        if not final:
            self._elapsed_after_id = self.root.after(1000, self._update_elapsed_tick)
        else:
            self._elapsed_after_id = None

    def _update_elapsed_tick(self) -> None:
        self._elapsed_after_id = None
        self._update_elapsed_label()

    @staticmethod
    def _format_elapsed(seconds: int) -> str:
        minutes, remainder = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{remainder:02d}"

        return f"{minutes:02d}:{remainder:02d}"

    def _clear_results(self) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()

    def _reset_results(self) -> None:
        if self._relayout_after_id is not None:
            self.root.after_cancel(self._relayout_after_id)
            self._relayout_after_id = None

        self._clear_results()
        self._results = []
        self._selection_vars.clear()
        self._thumbnails.clear()

    def _render_results(self) -> None:
        existing_selection = {
            path: var.get()
            for path, var in self._selection_vars.items()
        }

        self._clear_results()
        self._selection_vars.clear()

        if not self._results:
            self._render_empty_state("No matching photos found.")
            return

        columns = self._columns_for_current_width()
        for row, group in enumerate(self._results):
            group_frame = ttk.LabelFrame(self.results_frame, text=group.title, padding=12)
            group_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
            group_frame.columnconfigure(0, weight=1)

            items_frame = ttk.Frame(group_frame)
            items_frame.grid(row=0, column=0, sticky="ew")

            group_columns = min(columns, len(group.items))
            for column in range(group_columns):
                items_frame.columnconfigure(column, weight=1)

            for index, item in enumerate(group.items):
                self._render_item_card(
                    items_frame,
                    item,
                    index,
                    group_columns,
                    existing_selection,
                )

        self._update_action_state()

    def _columns_for_current_width(self) -> int:
        width = max(self.results_canvas.winfo_width(), self.root.winfo_width() - 40, 1)
        return max(1, width // self.card_min_width)

    def _render_item_card(
        self,
        parent: ttk.Frame,
        item: ResultItem,
        index: int,
        columns: int,
        selection_defaults: dict[Path, bool],
    ) -> None:
        row = index // columns
        column = index % columns

        card = ttk.Frame(parent, padding=8, relief="ridge")
        card.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        card.columnconfigure(0, weight=1)

        thumbnail = self._load_thumbnail(item.path)

        if thumbnail is not None:
            image_label = ttk.Label(card, image=thumbnail)
        else:
            image_label = ttk.Label(card, text="Preview unavailable", anchor="center")
        image_label.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(card, text=item.title, wraplength=220, justify="center")
        title.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        detail = ttk.Label(card, text=item.detail, wraplength=220, justify="center")
        detail.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        var = tk.BooleanVar(value=selection_defaults.get(item.path, item.recommended_delete))
        var.trace_add("write", lambda *_: self._update_action_state())
        self._selection_vars[item.path] = var

        check = ttk.Checkbutton(card, text="Delete", variable=var)
        check.grid(row=3, column=0, pady=(8, 0))

        if item.recommended_delete:
            hint = ttk.Label(card, text="Recommended", foreground="#b45309")
            hint.grid(row=4, column=0, pady=(4, 0))

    def _load_thumbnail(self, path: Path) -> ImageTk.PhotoImage | None:
        if path in self._thumbnails:
            return self._thumbnails[path]

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                image = ImageOps.contain(image, self.thumbnail_size)
                thumbnail = ImageTk.PhotoImage(image)
                self._thumbnails[path] = thumbnail
                return thumbnail
        except Exception:
            return None

    def _render_empty_state(self, text: str) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()

        empty = ttk.Label(self.results_frame, text=text, anchor="center")
        empty.grid(row=0, column=0, sticky="nsew", pady=40)

    def _resize_results_frame(self, event: tk.Event) -> None:
        self.results_canvas.itemconfigure(self._results_window, width=event.width)
        self._schedule_results_relayout()

    def _schedule_results_relayout(self) -> None:
        if not self._results:
            return

        if self._relayout_after_id is not None:
            self.root.after_cancel(self._relayout_after_id)

        self._relayout_after_id = self.root.after(120, self._relayout_results)

    def _relayout_results(self) -> None:
        self._relayout_after_id = None
        if self._results:
            self._render_results()

    def _sync_scroll_region(self, _event: tk.Event) -> None:
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _selected_count(self) -> int:
        return sum(1 for var in self._selection_vars.values() if var.get())

    def _update_action_state(self) -> None:
        selected = self._selected_count()
        total = len(self._selection_vars)

        self.count_var.set(f"{selected} selected / {total} shown")
        self.delete_button.configure(
            state="normal" if selected and total else "disabled"
        )

    def _delete_selected(self) -> None:
        selected_paths = [
            path for path, var in self._selection_vars.items() if var.get()
        ]

        if not selected_paths:
            return

        if not messagebox.askyesno(
            "Confirm deletion",
            f"Delete {len(selected_paths)} selected photo(s)? This cannot be undone.",
        ):
            return

        errors: list[str] = []
        deleted_paths = set()

        for path in selected_paths:
            try:
                path.unlink()
                deleted_paths.add(path)
            except FileNotFoundError:
                errors.append(f"Missing: {path}")
            except OSError as exc:
                errors.append(f"{path}: {exc}")

        if deleted_paths:
            updated_results: list[ResultGroup] = []
            for group in self._results:
                items = [item for item in group.items if item.path not in deleted_paths]
                if items:
                    updated_results.append(ResultGroup(title=group.title, items=items))
            self._results = updated_results
            self._render_results()

        if errors:
            messagebox.showwarning(
                "Partial delete",
                "Some files could not be deleted:\n\n" + "\n".join(errors),
            )
        else:
            messagebox.showinfo(
                "Deleted",
                f"Deleted {len(deleted_paths)} photo(s).",
            )


def run_app() -> None:
    PhotoCleanerApp().run()
