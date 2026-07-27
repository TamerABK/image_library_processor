from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from .models import AppState, PreviewItem, ResultGroup, ResultItem, UnknownFacePrompt


@dataclass(slots=True)
class PhotoCleanerViewCallbacks:
    browse_folder: Callable[[], None]
    refresh_file_types: Callable[[], None]
    start_scan: Callable[[], None]
    mode_changed: Callable[[], None]
    face_group_selected: Callable[[], None]
    show_previous_page: Callable[[], None]
    show_next_page: Callable[[], None]
    item_selection_changed: Callable[[Path, bool], None]
    delete_selected: Callable[[], None]
    export_selected: Callable[[], None]


class PhotoCleanerView:
    thumbnail_size = (128, 128)
    card_min_width = 220
    full_image_max_size = (1600, 1200)
    confirmation_preview_size = (520, 520)
    single_preview_chrome_size = (40, 90)
    viewer_chrome_size = (80, 220)

    def __init__(self, callbacks: PhotoCleanerViewCallbacks) -> None:
        self._callbacks = callbacks
        self.root = tk.Tk()
        self.root.title("Image Deduplicator")
        self.root.geometry("1200x800")
        self.root.minsize(980, 640)

        self.folder_var = tk.StringVar()
        self.file_type_var = tk.StringVar(value="All supported")
        self.orientation_var = tk.StringVar(value="All pictures")
        self.known_people_only_var = tk.BooleanVar(value=False)
        self.auto_export_faces_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="duplicates")
        self.status_var = tk.StringVar(value="Choose a folder to start.")
        self.count_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00")
        self.face_group_var = tk.StringVar(value="")
        self.page_label_var = tk.StringVar(value="")

        self._is_applying_state = False
        self._is_applying_selection_update = False
        self._selection_vars: dict[Path, tk.BooleanVar] = {}
        self._selection_state: dict[Path, bool] = {}
        self._thumbnails: dict[Path, ImageTk.PhotoImage] = {}
        self._render_groups: list[ResultGroup] = []
        self._shown_preview_items: list[PreviewItem] = []
        self._relayout_after_id: str | None = None
        self._last_rendered_columns = 0
        self._progress_running = False

        self._viewer_dialog: tk.Toplevel | None = None
        self._viewer_index: int | None = None
        self._viewer_current_path: Path | None = None
        self._viewer_photo: ImageTk.PhotoImage | None = None
        self._viewer_image_label: ttk.Label | None = None
        self._viewer_title_var = tk.StringVar(value="")
        self._viewer_detail_var = tk.StringVar(value="")
        self._viewer_group_var = tk.StringVar(value="")
        self._viewer_position_var = tk.StringVar(value="")
        self._viewer_path_var = tk.StringVar(value="")
        self._viewer_selection_var = tk.BooleanVar(value=False)
        self._viewer_prev_button: ttk.Button | None = None
        self._viewer_next_button: ttk.Button | None = None

        self._build_ui()
        self._render_empty_state("Choose a folder and scan to see photos.")

    def run(self) -> None:
        self.root.mainloop()

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> str:
        return self.root.after(delay_ms, callback)

    def cancel_scheduled(self, after_id: str) -> None:
        self.root.after_cancel(after_id)

    def current_folder(self) -> str:
        return self.folder_var.get()

    def current_file_type(self) -> str:
        return self.file_type_var.get()

    def current_mode(self) -> str:
        return self.mode_var.get()

    def current_known_people_only(self) -> bool:
        return self.known_people_only_var.get()

    def current_auto_export_faces(self) -> bool:
        return self.auto_export_faces_var.get()

    def current_orientation(self) -> str:
        return self.orientation_var.get()

    def current_face_group_label(self) -> str:
        return self.face_group_var.get()

    def sync_state(self, state: AppState) -> None:
        self._is_applying_state = True
        try:
            self.folder_var.set(state.folder)
            self.file_type_var.set(state.file_type)
            self.orientation_var.set(state.orientation)
            self.known_people_only_var.set(state.known_people_only)
            self.auto_export_faces_var.set(state.auto_export_faces)
            self.mode_var.set(state.mode)
            self.status_var.set(state.status)
            self.count_var.set(state.count_text)
            self.elapsed_var.set(state.elapsed_text)
            self.face_group_var.set(state.face_group_label)
            self.page_label_var.set(state.page_label)
            self.file_type_combo.configure(values=state.available_file_types)
            self.orientation_combo.configure(values=state.available_orientations)
            self.face_group_combo.configure(values=state.face_group_labels)
        finally:
            self._is_applying_state = False

        if state.show_face_options:
            self.face_options_frame.grid()
        else:
            self.face_options_frame.grid_remove()

        if state.show_face_selector:
            self.face_selector_frame.grid()
        else:
            self.face_selector_frame.grid_remove()

        if state.show_pagination:
            self.pagination_frame.grid()
        else:
            self.pagination_frame.grid_remove()

        self.scan_button.configure(state="normal" if state.can_scan else "disabled")
        self.delete_button.configure(state="normal" if state.can_delete else "disabled")
        self.export_button.configure(state="normal" if state.can_export else "disabled")
        self.prev_page_button.configure(
            state="normal" if state.can_show_previous_page else "disabled"
        )
        self.next_page_button.configure(
            state="normal" if state.can_show_next_page else "disabled"
        )

        if state.progress_mode == "indeterminate":
            if not self._progress_running:
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
                self._progress_running = True
        else:
            if self._progress_running:
                self.progress.stop()
                self._progress_running = False
            self.progress.configure(
                mode="determinate",
                maximum=state.progress_max,
                value=state.progress_value,
            )

    def render_results(
        self,
        groups: list[ResultGroup],
        selection_state: dict[Path, bool],
    ) -> None:
        self._render_groups = groups
        self._selection_state = dict(selection_state)
        self._clear_results()
        self._selection_vars.clear()
        self._shown_preview_items = []

        if not groups:
            self._last_rendered_columns = 0
            self._sync_viewer_with_shown_items()
            self._render_empty_state("No matching photos found.")
            return

        columns = self._columns_for_current_width()
        self._last_rendered_columns = columns
        shown_item_index = 0

        for row, group in enumerate(groups):
            group_frame = ttk.LabelFrame(self.results_frame, text=group.title, padding=12)
            group_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
            group_frame.columnconfigure(0, weight=1)

            items_frame = ttk.Frame(group_frame)
            items_frame.grid(row=0, column=0, sticky="ew")

            group_columns = min(columns, len(group.items))
            for column in range(group_columns):
                items_frame.columnconfigure(column, weight=1)

            for index, item in enumerate(group.items):
                self._shown_preview_items.append(
                    PreviewItem(
                        path=item.path,
                        title=item.title,
                        detail=item.detail,
                        group_title=group.title,
                    )
                )
                self._render_item_card(
                    items_frame,
                    item,
                    index,
                    group_columns,
                    group.group_type,
                    shown_item_index,
                )
                shown_item_index += 1

        self._sync_viewer_with_shown_items()
        self._sync_scroll_region(None)

    def clear_results(self, text: str = "") -> None:
        self._render_groups = []
        self._selection_state = {}
        self._clear_results()
        self._selection_vars.clear()
        self._shown_preview_items = []
        self._last_rendered_columns = 0
        self._sync_viewer_with_shown_items()
        self._render_empty_state(text)

    def update_selection_state(self, selection_state: dict[Path, bool]) -> None:
        self._selection_state = dict(selection_state)
        self._is_applying_selection_update = True
        try:
            for path, var in self._selection_vars.items():
                selected = self._selection_state.get(path, False)
                if var.get() != selected:
                    var.set(selected)

            if (
                self._viewer_current_path is not None
                and self._viewer_current_path in self._selection_state
                and self._viewer_selection_var.get()
                != self._selection_state[self._viewer_current_path]
            ):
                self._viewer_selection_var.set(self._selection_state[self._viewer_current_path])
        finally:
            self._is_applying_selection_update = False

    def scroll_results_to_top(self) -> None:
        self.results_canvas.yview_moveto(0)

    def ask_directory(self, title: str | None = None) -> str:
        if title is None:
            return filedialog.askdirectory()
        return filedialog.askdirectory(title=title)

    def confirm_delete(self, count: int) -> bool:
        return messagebox.askyesno(
            "Confirm deletion",
            f"Delete {count} selected photo(s)? This cannot be undone.",
        )

    def prompt_export_type(self) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Export method")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        selection = {"value": None}
        if os.name == "nt":
            link_label = "Hard link"
            link_value = "hardlink"
        else:
            link_label = "Symbolic link"
            link_value = "symlink"

        frame = ttk.Frame(dialog, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            frame,
            text="Choose how to place the selected photos in the export folder.",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        def choose(value: str | None) -> None:
            selection["value"] = value
            dialog.destroy()

        ttk.Button(frame, text="Copy", command=lambda: choose("copy")).grid(
            row=1,
            column=0,
            padx=(0, 8),
            pady=(16, 0),
            sticky="ew",
        )
        ttk.Button(frame, text=link_label, command=lambda: choose(link_value)).grid(
            row=1,
            column=1,
            padx=(0, 8),
            pady=(16, 0),
            sticky="ew",
        )
        ttk.Button(frame, text="Cancel", command=lambda: choose(None)).grid(
            row=1,
            column=2,
            pady=(16, 0),
            sticky="ew",
        )

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose(None))
        dialog.update_idletasks()

        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        dialog.geometry(
            f"+{root_x + max((root_width - dialog_width) // 2, 0)}"
            f"+{root_y + max((root_height - dialog_height) // 2, 0)}"
        )

        self.root.wait_window(dialog)
        return selection["value"]

    def prompt_unknown_face(self, prompt: UnknownFacePrompt) -> str | None:
        preview_bgr = cv2.cvtColor(prompt.preview_array, cv2.COLOR_BGR2RGB)
        preview_pil = Image.fromarray(preview_bgr.astype("uint8"))
        preview_pil = ImageOps.contain(
            preview_pil,
            self.confirmation_preview_size,
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(preview_pil)

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Name Unknown Person {prompt.cluster_id}")
        dialog.geometry("920x820")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=24)
        frame.pack(fill="both", expand=True)

        label = ttk.Label(frame, image=photo, cursor="hand2")
        label.image = photo
        label.pack(padx=20, pady=(0, 20))
        label.bind(
            "<Button-1>",
            lambda _event, path=prompt.representative_path: self._open_single_image(path),
        )

        ttk.Label(
            frame,
            text=f"This person appears in {prompt.face_count} photo(s).\nEnter their name:",
            font=("TkDefaultFont", 14),
            justify="center",
        ).pack(pady=(0, 16))

        name_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=name_var, width=40, font=("TkDefaultFont", 14))
        entry.pack(pady=(0, 8), padx=20, fill="x")
        entry.focus()

        suggestion_var = tk.StringVar()
        ttk.Label(
            frame,
            textvariable=suggestion_var,
            foreground="#4b5563",
            anchor="w",
        ).pack(fill="x", padx=20)

        suggestions_box = tk.Listbox(frame, height=6, exportselection=False)
        suggestions_box.pack(fill="x", padx=20, pady=(8, 0))

        result = {"confirmed": False, "name": ""}
        suggestion_state = {"matches": []}

        def update_suggestions(*_args: object) -> None:
            typed = name_var.get().strip()
            if typed:
                matches = [
                    name
                    for name in prompt.suggestion_names
                    if name.lower().startswith(typed.lower())
                ]
            else:
                matches = list(prompt.suggestion_names)

            suggestion_state["matches"] = matches[:20]
            suggestions_box.delete(0, tk.END)
            for name in suggestion_state["matches"]:
                suggestions_box.insert(tk.END, name)

            if suggestion_state["matches"]:
                suggestions_box.selection_clear(0, tk.END)
                suggestions_box.selection_set(0)
                suggestions_box.activate(0)
                suggestion_var.set("Tab to accept suggestion")
            else:
                suggestion_var.set("")

        def apply_selected_suggestion() -> bool:
            selection = suggestions_box.curselection()
            if not selection:
                return False
            selected_name = suggestions_box.get(selection[0]).strip()
            if not selected_name:
                return False
            name_var.set(selected_name)
            entry.icursor(tk.END)
            return True

        def on_ok() -> None:
            name = name_var.get().strip()
            if name:
                result["confirmed"] = True
                result["name"] = name
                dialog.destroy()

        def on_skip() -> None:
            dialog.destroy()

        def on_key(event: tk.Event) -> str | None:
            if event.keysym == "Return":
                on_ok()
                return "break"
            if event.keysym == "Tab":
                if apply_selected_suggestion():
                    return "break"
            if event.keysym == "Down" and suggestions_box.size():
                current = suggestions_box.curselection()
                next_index = 0 if not current else min(current[0] + 1, suggestions_box.size() - 1)
                suggestions_box.selection_clear(0, tk.END)
                suggestions_box.selection_set(next_index)
                suggestions_box.activate(next_index)
                return "break"
            if event.keysym == "Up" and suggestions_box.size():
                current = suggestions_box.curselection()
                next_index = suggestions_box.size() - 1 if not current else max(current[0] - 1, 0)
                suggestions_box.selection_clear(0, tk.END)
                suggestions_box.selection_set(next_index)
                suggestions_box.activate(next_index)
                return "break"
            if event.keysym == "Escape":
                on_skip()
                return "break"
            return None

        def on_suggestion_activate(_event: tk.Event) -> str:
            apply_selected_suggestion()
            entry.focus_set()
            return "break"

        name_var.trace_add("write", update_suggestions)
        entry.bind("<Key>", on_key)
        suggestions_box.bind("<Double-Button-1>", on_suggestion_activate)
        suggestions_box.bind("<Return>", on_suggestion_activate)

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=24)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Skip", command=on_skip).pack(side="left", padx=10)

        update_suggestions()
        self.root.wait_window(dialog)
        if result["confirmed"]:
            return str(result["name"])
        return None

    def show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def show_warning(self, title: str, message: str) -> None:
        messagebox.showwarning(title, message)

    def show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)

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
        folder_entry.bind("<FocusOut>", lambda _event: self._callbacks.refresh_file_types())
        folder_entry.bind("<Return>", lambda _event: self._callbacks.refresh_file_types())

        ttk.Button(controls, text="Browse", command=self._callbacks.browse_folder).grid(
            row=0,
            column=2,
            sticky="ew",
        )

        ttk.Label(controls, text="File type").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.file_type_combo = ttk.Combobox(
            controls,
            textvariable=self.file_type_var,
            state="readonly",
            values=(self.file_type_var.get(),),
        )
        self.file_type_combo.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        ttk.Button(controls, text="Refresh", command=self._callbacks.refresh_file_types).grid(
            row=1,
            column=2,
            sticky="ew",
            pady=(10, 0),
        )

        ttk.Label(controls, text="Orientation").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.orientation_combo = ttk.Combobox(
            controls,
            textvariable=self.orientation_var,
            state="readonly",
            values=(self.orientation_var.get(),),
        )
        self.orientation_combo.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))

        mode_frame = ttk.Frame(controls)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
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
        ttk.Radiobutton(
            mode_frame,
            text="Facial recognition",
            value="faces",
            variable=self.mode_var,
        ).grid(row=0, column=3, padx=(12, 0))
        self.mode_var.trace_add("write", self._on_mode_changed)

        self.face_options_frame = ttk.Frame(controls)
        self.face_options_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            self.face_options_frame,
            text="Known people only",
            variable=self.known_people_only_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            self.face_options_frame,
            text="Auto export faces after scan",
            variable=self.auto_export_faces_var,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.face_options_frame.grid_remove()

        actions = ttk.Frame(controls)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        self.scan_button = ttk.Button(actions, text="Scan folder", command=self._callbacks.start_scan)
        self.scan_button.grid(row=0, column=0, sticky="w")

        self.delete_button = ttk.Button(
            actions,
            text="Delete selected",
            command=self._callbacks.delete_selected,
            state="disabled",
        )
        self.delete_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.export_button = ttk.Button(
            actions,
            text="Export selected",
            command=self._callbacks.export_selected,
            state="disabled",
        )
        self.export_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

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
        results_outer.rowconfigure(0, weight=0)
        results_outer.rowconfigure(1, weight=0)
        results_outer.rowconfigure(2, weight=1)

        self.face_selector_frame = ttk.Frame(results_outer)
        self.face_selector_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.face_selector_frame.columnconfigure(1, weight=1)

        ttk.Label(self.face_selector_frame, text="Person").grid(row=0, column=0, sticky="w")
        self.face_group_combo = ttk.Combobox(
            self.face_selector_frame,
            textvariable=self.face_group_var,
            state="readonly",
        )
        self.face_group_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.face_group_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._callbacks.face_group_selected(),
        )
        self.face_selector_frame.grid_remove()

        self.pagination_frame = ttk.Frame(results_outer)
        self.pagination_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.pagination_frame.columnconfigure(1, weight=1)

        self.prev_page_button = ttk.Button(
            self.pagination_frame,
            text="Previous",
            command=self._callbacks.show_previous_page,
        )
        self.prev_page_button.grid(row=0, column=0, sticky="w")

        ttk.Label(
            self.pagination_frame,
            textvariable=self.page_label_var,
            anchor="center",
        ).grid(row=0, column=1, sticky="ew")

        self.next_page_button = ttk.Button(
            self.pagination_frame,
            text="Next",
            command=self._callbacks.show_next_page,
        )
        self.next_page_button.grid(row=0, column=2, sticky="e")
        self.pagination_frame.grid_remove()

        self.results_canvas = tk.Canvas(results_outer, highlightthickness=0)
        self.results_canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(results_outer, orient="vertical", command=self.results_canvas.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
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

    def _on_mode_changed(self, *_args: object) -> None:
        if self._is_applying_state:
            return
        self._callbacks.mode_changed()

    def _clear_results(self) -> None:
        self._thumbnails.clear()
        for child in self.results_frame.winfo_children():
            child.destroy()

    def _render_item_card(
        self,
        parent: ttk.Frame,
        item: ResultItem,
        index: int,
        columns: int,
        group_type: str,
        shown_item_index: int,
    ) -> None:
        row = index // columns
        column = index % columns

        card = ttk.Frame(parent, padding=8, relief="ridge")
        card.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        card.columnconfigure(0, weight=1)

        thumbnail = self._load_thumbnail(item.path)
        if thumbnail is not None:
            image_label = ttk.Label(card, image=thumbnail, cursor="hand2")
        else:
            image_label = ttk.Label(card, text="Preview unavailable", anchor="center")
        image_label.grid(row=0, column=0, sticky="nsew")
        image_label.bind(
            "<Button-1>",
            lambda _event, preview_index=shown_item_index: self._open_full_image(preview_index),
        )

        ttk.Label(card, text=item.title, wraplength=180, justify="center").grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Label(card, text=item.detail, wraplength=180, justify="center").grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(2, 0),
        )

        default_selected = self._selection_state.get(
            item.path,
            True if group_type == "face" else item.recommended_delete,
        )
        var = tk.BooleanVar(value=default_selected)
        self._selection_state[item.path] = default_selected

        def on_selection_change(*_args: object) -> None:
            if self._is_applying_selection_update:
                return
            self._selection_state[item.path] = var.get()
            self._callbacks.item_selection_changed(item.path, var.get())

        var.trace_add("write", on_selection_change)
        self._selection_vars[item.path] = var

        ttk.Checkbutton(card, text="Selected", variable=var).grid(
            row=3,
            column=0,
            pady=(8, 0),
        )

        if item.recommended_delete and group_type != "face":
            ttk.Label(card, text="Recommended", foreground="#b45309").grid(
                row=4,
                column=0,
                pady=(4, 0),
            )

    def _load_thumbnail(self, path: Path) -> ImageTk.PhotoImage | None:
        if path in self._thumbnails:
            return self._thumbnails[path]

        try:
            with Image.open(path) as image:
                image.draft("RGB", self.thumbnail_size)
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                image.thumbnail(self.thumbnail_size, Image.Resampling.BILINEAR)
                thumbnail = ImageTk.PhotoImage(image)
                self._thumbnails[path] = thumbnail
                return thumbnail
        except Exception:
            return None

    def _open_full_image(self, preview_index: int) -> None:
        if not self._shown_preview_items:
            return

        preview_index = max(0, min(preview_index, len(self._shown_preview_items) - 1))

        if self._viewer_dialog is None or not self._viewer_dialog.winfo_exists():
            dialog = tk.Toplevel(self.root)
            dialog.transient(self.root)
            dialog.bind("<Left>", lambda _event: self._show_previous_preview())
            dialog.bind("<Right>", lambda _event: self._show_next_preview())
            dialog.bind("<Escape>", lambda _event: self._close_viewer())
            dialog.protocol("WM_DELETE_WINDOW", self._close_viewer)

            dialog.columnconfigure(0, weight=1)
            dialog.rowconfigure(1, weight=1)

            controls = ttk.Frame(dialog, padding=(12, 12, 12, 0))
            controls.grid(row=0, column=0, sticky="ew")
            controls.columnconfigure(1, weight=1)

            self._viewer_prev_button = ttk.Button(
                controls,
                text="< Previous",
                command=self._show_previous_preview,
            )
            self._viewer_prev_button.grid(row=0, column=0, sticky="w")

            ttk.Label(
                controls,
                textvariable=self._viewer_position_var,
                anchor="center",
            ).grid(row=0, column=1, sticky="ew", padx=12)

            self._viewer_next_button = ttk.Button(
                controls,
                text="Next >",
                command=self._show_next_preview,
            )
            self._viewer_next_button.grid(row=0, column=2, sticky="e")

            self._viewer_image_label = ttk.Label(dialog, anchor="center")
            self._viewer_image_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

            details = ttk.Frame(dialog, padding=(12, 0, 12, 12))
            details.grid(row=2, column=0, sticky="ew")
            details.columnconfigure(0, weight=1)

            ttk.Label(
                details,
                textvariable=self._viewer_title_var,
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, sticky="ew")
            ttk.Label(
                details,
                textvariable=self._viewer_detail_var,
                anchor="center",
                justify="center",
                wraplength=1000,
            ).grid(row=1, column=0, sticky="ew", pady=(4, 0))
            ttk.Label(
                details,
                textvariable=self._viewer_group_var,
                anchor="center",
                justify="center",
                wraplength=1000,
            ).grid(row=2, column=0, sticky="ew", pady=(4, 0))
            ttk.Checkbutton(
                details,
                text="Selected",
                variable=self._viewer_selection_var,
                command=self._toggle_viewer_selection,
            ).grid(row=3, column=0, pady=(8, 0))
            ttk.Label(
                details,
                textvariable=self._viewer_path_var,
                anchor="center",
                justify="center",
                wraplength=1000,
            ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

            self._viewer_dialog = dialog
            self._center_viewer_dialog()

        self._viewer_index = preview_index
        self._render_viewer_item()
        if self._viewer_dialog is not None and self._viewer_dialog.winfo_exists():
            self._viewer_dialog.deiconify()
            self._viewer_dialog.lift()
            self._viewer_dialog.focus_force()

    def _open_single_image(self, path: Path) -> None:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")

                display_image = image.copy()
                display_image.thumbnail(
                    self._preview_max_image_size(*self.single_preview_chrome_size),
                    Image.Resampling.LANCZOS,
                )
        except Exception as exc:
            messagebox.showerror("Preview failed", f"Could not open image:\n{path}\n\n{exc}")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(path.name)
        dialog.transient(self.root)

        photo = ImageTk.PhotoImage(display_image)
        image_label = ttk.Label(dialog, image=photo)
        image_label.image = photo
        image_label.grid(row=0, column=0, sticky="nsew")

        ttk.Label(dialog, text=str(path), anchor="center", wraplength=1000).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=12,
            pady=(0, 12),
        )

        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        dialog.update_idletasks()
        self._size_dialog_to_image(dialog, photo.width(), photo.height(), *self.single_preview_chrome_size)

    def _render_viewer_item(self) -> None:
        if self._viewer_dialog is None or not self._viewer_dialog.winfo_exists():
            return
        if self._viewer_index is None or not self._shown_preview_items:
            self._close_viewer()
            return

        self._viewer_index = max(0, min(self._viewer_index, len(self._shown_preview_items) - 1))
        preview_item = self._shown_preview_items[self._viewer_index]
        self._viewer_current_path = preview_item.path

        self._viewer_dialog.title(preview_item.path.name)
        self._viewer_position_var.set(
            f"Photo {self._viewer_index + 1} of {len(self._shown_preview_items)}"
        )
        self._viewer_title_var.set(preview_item.title)
        self._viewer_detail_var.set(preview_item.detail)
        self._viewer_group_var.set(f"Group: {preview_item.group_title}")
        self._viewer_path_var.set(str(preview_item.path))

        selected = self._selection_state.get(preview_item.path, False)
        self._is_applying_selection_update = True
        try:
            self._viewer_selection_var.set(selected)
        finally:
            self._is_applying_selection_update = False

        try:
            with Image.open(preview_item.path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")

                display_image = image.copy()
                display_image.thumbnail(
                    self._preview_max_image_size(*self.viewer_chrome_size),
                    Image.Resampling.LANCZOS,
                )
                self._viewer_photo = ImageTk.PhotoImage(display_image)
        except Exception as exc:
            self._viewer_photo = None
            self._viewer_detail_var.set(f"{preview_item.detail} | Preview unavailable: {exc}")

        if self._viewer_image_label is not None:
            if self._viewer_photo is None:
                self._viewer_image_label.configure(image="", text="Preview unavailable")
            else:
                self._viewer_image_label.configure(image=self._viewer_photo, text="")
            self._viewer_image_label.image = self._viewer_photo
            if self._viewer_photo is not None:
                self._size_dialog_to_image(
                    self._viewer_dialog,
                    self._viewer_photo.width(),
                    self._viewer_photo.height(),
                    *self.viewer_chrome_size,
                )

        if self._viewer_prev_button is not None:
            self._viewer_prev_button.configure(
                state="normal" if self._viewer_index > 0 else "disabled"
            )
        if self._viewer_next_button is not None:
            self._viewer_next_button.configure(
                state="normal"
                if self._viewer_index < len(self._shown_preview_items) - 1
                else "disabled"
            )

    def _toggle_viewer_selection(self) -> None:
        if (
            self._is_applying_selection_update
            or self._viewer_index is None
            or not self._shown_preview_items
        ):
            return

        preview_item = self._shown_preview_items[self._viewer_index]
        selected = self._viewer_selection_var.get()
        self._selection_state[preview_item.path] = selected
        self._callbacks.item_selection_changed(preview_item.path, selected)

    def _show_previous_preview(self) -> None:
        if self._viewer_index is None or self._viewer_index <= 0:
            return
        self._viewer_index -= 1
        self._render_viewer_item()

    def _show_next_preview(self) -> None:
        if self._viewer_index is None or self._viewer_index >= len(self._shown_preview_items) - 1:
            return
        self._viewer_index += 1
        self._render_viewer_item()

    def _sync_viewer_with_shown_items(self) -> None:
        if self._viewer_dialog is None or not self._viewer_dialog.winfo_exists():
            return
        if self._viewer_current_path is None or not self._shown_preview_items:
            self._close_viewer()
            return

        for index, preview_item in enumerate(self._shown_preview_items):
            if preview_item.path == self._viewer_current_path:
                self._viewer_index = index
                self._render_viewer_item()
                return

        self._close_viewer()

    def _close_viewer(self) -> None:
        if self._viewer_dialog is not None and self._viewer_dialog.winfo_exists():
            self._viewer_dialog.destroy()
        self._viewer_dialog = None
        self._viewer_index = None
        self._viewer_current_path = None
        self._viewer_photo = None
        self._viewer_image_label = None
        self._viewer_prev_button = None
        self._viewer_next_button = None

    def _center_viewer_dialog(self) -> None:
        if self._viewer_dialog is None or not self._viewer_dialog.winfo_exists():
            return
        self._size_dialog_to_image(
            self._viewer_dialog,
            self.full_image_max_size[0],
            self.full_image_max_size[1],
            *self.viewer_chrome_size,
        )

    def _preview_max_image_size(
        self,
        horizontal_chrome: int,
        vertical_chrome: int,
    ) -> tuple[int, int]:
        max_width = min(
            self.full_image_max_size[0],
            max(self.root.winfo_screenwidth() - horizontal_chrome, 200),
        )
        max_height = min(
            self.full_image_max_size[1],
            max(self.root.winfo_screenheight() - vertical_chrome, 200),
        )
        return max_width, max_height

    def _size_dialog_to_image(
        self,
        dialog: tk.Toplevel,
        image_width: int,
        image_height: int,
        horizontal_chrome: int,
        vertical_chrome: int,
    ) -> None:
        width = min(image_width + horizontal_chrome, self.root.winfo_screenwidth() - 40)
        height = min(image_height + vertical_chrome, self.root.winfo_screenheight() - 60)

        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog.geometry(
            f"{width}x{height}+"
            f"{root_x + max((root_width - width) // 2, 0)}+"
            f"{root_y + max((root_height - height) // 2, 0)}"
        )

    def _render_empty_state(self, text: str) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()
        ttk.Label(self.results_frame, text=text, anchor="center").grid(
            row=0,
            column=0,
            sticky="nsew",
            pady=40,
        )

    def _resize_results_frame(self, event: tk.Event) -> None:
        self.results_canvas.itemconfigure(self._results_window, width=event.width)
        self._schedule_results_relayout()

    def _schedule_results_relayout(self) -> None:
        if not self._render_groups:
            return

        current_columns = self._columns_for_current_width()
        if current_columns == self._last_rendered_columns:
            return

        if self._relayout_after_id is not None:
            self.root.after_cancel(self._relayout_after_id)
        self._relayout_after_id = self.root.after(120, self._relayout_results)

    def _relayout_results(self) -> None:
        self._relayout_after_id = None
        if self._render_groups:
            self.render_results(self._render_groups, self._selection_state)

    def _columns_for_current_width(self) -> int:
        width = max(self.results_canvas.winfo_width(), self.root.winfo_width() - 40, 1)
        return max(1, width // self.card_min_width)

    def _sync_scroll_region(self, _event: tk.Event | None) -> None:
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
