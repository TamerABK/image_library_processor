from __future__ import annotations

from pathlib import Path

from .models import ScanErrorMessage, ScanProgressMessage, ScanResultMessage, UnknownFacesMessage
from .view import PhotoCleanerView, PhotoCleanerViewCallbacks
from .view_model import PhotoCleanerViewModel


class PhotoCleanerApp:
    def __init__(self) -> None:
        self.view_model = PhotoCleanerViewModel()
        self.view = PhotoCleanerView(
            PhotoCleanerViewCallbacks(
                browse_folder=self._browse_folder,
                refresh_file_types=self._refresh_file_types,
                start_scan=self._start_scan,
                cancel_scan=self._cancel_scan,
                mode_changed=self._on_mode_changed,
                face_group_selected=self._on_face_group_selected,
                show_previous_page=self._show_previous_page,
                show_next_page=self._show_next_page,
                item_selection_changed=self._on_item_selection_changed,
                delete_selected=self._delete_selected,
                export_selected=self._export_selected,
                export_vibe_debug=self._export_vibe_debug,
            )
        )
        self._poll_after_id: str | None = None
        self._elapsed_after_id: str | None = None
        self._sync_view()
        self._schedule_poll()

    def run(self) -> None:
        self.view.run()

    def _browse_folder(self) -> None:
        folder = self.view.ask_directory()
        if not folder:
            return
        self.view_model.set_folder(folder)
        self.view_model.refresh_file_types(Path(folder))
        self._sync_view()

    def _refresh_file_types(self) -> None:
        self._sync_inputs_from_view()
        self.view_model.refresh_file_types()
        self._sync_view()

    def _on_mode_changed(self) -> None:
        self.view_model.set_mode(self.view.current_mode())
        self.view_model.set_known_people_only(self.view.current_known_people_only())
        self.view_model.set_auto_export_faces(self.view.current_auto_export_faces())
        self.view_model.set_orientation(self.view.current_orientation())
        self.view_model.set_vibe_preset(self.view.current_vibe_preset())
        self.view_model.set_vibe_include_people(self.view.current_vibe_include_people())
        self.view_model.set_vibe_include_color(self.view.current_vibe_include_color())
        self.view_model.set_vibe_include_composition(self.view.current_vibe_include_composition())
        self.view_model.set_vibe_show_advanced(self.view.current_vibe_show_advanced())
        self.view_model.set_vibe_session_gap_minutes(self.view.current_vibe_session_gap_minutes())
        self.view_model.set_vibe_minimum_similarity(self.view.current_vibe_minimum_similarity())
        self.view_model.set_vibe_minimum_cohesion(self.view.current_vibe_minimum_cohesion())
        self.view_model.set_vibe_maximum_group_size(self.view.current_vibe_maximum_group_size())
        self.view_model.set_vibe_batch_size(self.view.current_vibe_batch_size())
        self._sync_view()

    def _start_scan(self) -> None:
        self._sync_inputs_from_view()
        error = self.view_model.start_scan()
        if error:
            self.view.show_error("Invalid folder", error)
            return
        self._schedule_elapsed_update()
        self.view.clear_results()
        self._sync_view()

    def _cancel_scan(self) -> None:
        self.view_model.cancel_scan()
        self._sync_view()

    def _on_face_group_selected(self) -> None:
        self.view_model.show_face_group_by_label(self.view.current_face_group_label())
        self._render_results()

    def _show_previous_page(self) -> None:
        self.view_model.show_previous_page()
        self._render_results()
        self.view.scroll_results_to_top()

    def _show_next_page(self) -> None:
        self.view_model.show_next_page()
        self._render_results()
        self.view.scroll_results_to_top()

    def _on_item_selection_changed(self, path: Path, selected: bool) -> None:
        self.view_model.set_item_selected(path, selected)
        self.view.update_selection_state(self.view_model.selection_state_snapshot())
        self._sync_view()

    def _delete_selected(self) -> None:
        selected_count = self.view_model.selected_item_count()
        if not selected_count:
            return
        if not self.view.confirm_delete(selected_count):
            return

        result = self.view_model.delete_selected()
        self._render_results()
        if result.errors:
            self.view.show_warning(
                "Partial delete",
                "Some files could not be deleted:\n\n" + "\n".join(result.errors),
            )
        else:
            self.view.show_info("Deleted", f"Deleted {result.deleted_count} photo(s).")

    def _export_selected(self) -> None:
        if not self.view_model.selected_paths():
            return

        export_dir = self.view.ask_directory(title="Select export destination")
        if not export_dir:
            return

        export_type = self.view.prompt_export_type()
        if export_type is None:
            return

        result = self.view_model.export_selected(export_dir, export_type)
        if result.errors:
            self.view.show_warning(
                "Partial export",
                "Some files could not be exported:\n\n" + "\n".join(result.errors[:10]),
            )
        else:
            self.view.show_info(
                "Export complete",
                f"Exported {result.exported_count} photo(s) to {export_dir}",
            )

    def _export_vibe_debug(self) -> None:
        if not self.view_model.can_export_vibe_debug():
            return

        output_path = self.view.ask_save_path(
            title="Export vibe debug JSON",
            initialfile=self.view_model.suggest_vibe_debug_filename(),
        )
        if not output_path:
            return

        try:
            written_path = self.view_model.export_vibe_debug(output_path)
        except Exception as exc:
            self.view.show_error("Export failed", f"Could not export vibe debug JSON:\n\n{exc}")
            return

        self.view.show_info(
            "Export complete",
            f"Saved vibe debug JSON to {written_path}",
        )

    def _schedule_poll(self) -> None:
        self._poll_after_id = self.view.schedule(100, self._poll_background_messages)

    def _poll_background_messages(self) -> None:
        try:
            while True:
                message = self.view_model.poll_background_message()
                if message is None:
                    break

                if isinstance(message, ScanProgressMessage):
                    self.view_model.handle_progress_message(message)
                    self._sync_view()
                    continue

                if isinstance(message, UnknownFacesMessage):
                    self.view_model.handle_unknown_faces_message(message)
                    for cluster in message.face_result.unknown_clusters:
                        try:
                            prompt = self.view_model.build_unknown_face_prompt(cluster)
                            chosen_name = self.view.prompt_unknown_face(prompt)
                            if chosen_name:
                                self.view_model.apply_unknown_face_name(prompt, chosen_name)
                        except Exception as exc:
                            self.view.show_error("Error", f"Failed to process face for naming: {exc}")
                    continue

                if isinstance(message, ScanResultMessage):
                    self.view_model.handle_scan_result_message(message)
                    self._render_results()
                    if message.warning:
                        self.view.show_warning("Scan warning", message.warning)
                    if message.mode == "faces" and self.view_model.state.auto_export_faces:
                        self._auto_export_faces()
                    continue

                if isinstance(message, ScanErrorMessage):
                    self.view_model.handle_scan_error_message(message)
                    self._sync_view()
                    if not message.canceled:
                        self.view.show_error("Scan failed", message.message)
        finally:
            self._schedule_poll()

    def _schedule_elapsed_update(self) -> None:
        if self._elapsed_after_id is not None:
            self.view.cancel_scheduled(self._elapsed_after_id)
            self._elapsed_after_id = None

        if not self.view_model.refresh_elapsed():
            self._sync_view()
            return

        self._sync_view()
        self._elapsed_after_id = self.view.schedule(1000, self._schedule_elapsed_update)

    def _sync_inputs_from_view(self) -> None:
        self.view_model.set_folder(self.view.current_folder())
        self.view_model.set_file_type(self.view.current_file_type())
        self.view_model.set_orientation(self.view.current_orientation())
        self.view_model.set_mode(self.view.current_mode())
        self.view_model.set_known_people_only(self.view.current_known_people_only())
        self.view_model.set_auto_export_faces(self.view.current_auto_export_faces())
        self.view_model.set_vibe_preset(self.view.current_vibe_preset())
        self.view_model.set_vibe_include_people(self.view.current_vibe_include_people())
        self.view_model.set_vibe_include_color(self.view.current_vibe_include_color())
        self.view_model.set_vibe_include_composition(self.view.current_vibe_include_composition())
        self.view_model.set_vibe_show_advanced(self.view.current_vibe_show_advanced())
        self.view_model.set_vibe_session_gap_minutes(self.view.current_vibe_session_gap_minutes())
        self.view_model.set_vibe_minimum_similarity(self.view.current_vibe_minimum_similarity())
        self.view_model.set_vibe_minimum_cohesion(self.view.current_vibe_minimum_cohesion())
        self.view_model.set_vibe_maximum_group_size(self.view.current_vibe_maximum_group_size())
        self.view_model.set_vibe_batch_size(self.view.current_vibe_batch_size())

    def _sync_view(self) -> None:
        self.view.sync_state(self.view_model.state)
        self.view.update_selection_state(self.view_model.selection_state_snapshot())

    def _render_results(self) -> None:
        self.view.render_results(
            self.view_model.current_page_groups(),
            self.view_model.selection_state_snapshot(),
        )
        self._sync_view()

    def _auto_export_faces(self) -> None:
        if not self.view_model.current_page_groups():
            return

        export_dir = self.view.ask_directory(title="Select auto export destination")
        if not export_dir:
            return

        result = self.view_model.export_face_groups(export_dir)
        if result.errors:
            self.view.show_warning(
                "Partial auto export",
                "Some files could not be auto exported:\n\n" + "\n".join(result.errors[:10]),
            )
        else:
            self.view.show_info(
                "Auto export complete",
                f"Exported {result.exported_count} photo(s) to {export_dir}",
            )


def run_app() -> None:
    PhotoCleanerApp().run()
