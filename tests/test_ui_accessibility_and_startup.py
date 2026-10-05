"""Tests for UI accessibility metadata, DropArea, ThumbnailListWidget, and non-blocking headless startup."""

from __future__ import annotations

import os
from PySide6.QtWidgets import QApplication

from DeepSlice.gui.main_window import DeepSliceMainWindow, DropArea, ThumbnailListWidget


def test_drop_area_accessibility():
    app = QApplication.instance() or QApplication([])
    area = DropArea()
    try:
        assert area.accessibleName() == "Image and Folder Ingestion Drop Area"
        assert "Drag and drop" in area.toolTip()
    finally:
        area.deleteLater()


def test_thumbnail_list_widget_accessibility():
    app = QApplication.instance() or QApplication([])
    list_widget = ThumbnailListWidget()
    try:
        assert list_widget.accessibleName() == "Thumbnail Sections List"
        assert "thumbnails" in list_widget.toolTip().lower()
    finally:
        list_widget.deleteLater()


def test_main_window_status_accessibility_and_headless_startup():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        assert win.status_bar.accessibleName() == "Main Status Bar"
        assert win.global_progress.accessibleName() == "Global Task Progress"
        assert win.global_progress.toolTip() != ""
    finally:
        win.close()
        win.deleteLater()


def test_preferences_dialog_accessibility_and_reset_to_defaults():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        dlg = win._create_preferences_dialog()
        try:
            assert dlg.species_combo.accessibleName() == "Default species for new sessions"
            assert dlg.theme_combo.accessibleName() == "Application theme"
            assert dlg.output_dir_edit.accessibleName() == "Default output directory path"
            assert dlg.output_dir_browse.accessibleName() == "Browse default output directory"
            assert dlg.quicknii_edit.accessibleName() == "Default QuickNII executable path"
            assert dlg.quicknii_browse.accessibleName() == "Browse QuickNII executable"
            assert dlg.console_always_visible.accessibleName() == "Show runtime console by default"
            assert dlg.reset_btn is not None
            assert dlg.reset_btn.accessibleName() == "Reset preferences to factory defaults"
            assert "default values" in dlg.reset_btn.toolTip().lower()

            # Modify values and trigger reset
            dlg.species_combo.setCurrentText("rat")
            dlg.theme_combo.setCurrentText("light")
            dlg.output_dir_edit.setText("C:/custom_output")
            dlg.quicknii_edit.setText("C:/quicknii.exe")
            dlg.console_always_visible.setChecked(True)

            dlg.reset_btn.click()

            assert dlg.species_combo.currentText() == "mouse"
            assert dlg.theme_combo.currentText() == "dark"
            assert dlg.output_dir_edit.text() == ""
            assert dlg.quicknii_edit.text() == ""
            assert not dlg.console_always_visible.isChecked()
        finally:
            dlg.deleteLater()
    finally:
        win.close()
        win.deleteLater()


def test_console_and_prediction_controls_accessibility():
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        assert win.prediction_progress_bar.accessibleName() == "Prediction Task Progress"
        assert win.accept_predicted_thickness_button.accessibleName() == "Accept Predicted Section Thickness"
        assert "estimated section thickness" in win.accept_predicted_thickness_button.toolTip()
        assert win.accept_predicted_thickness_button.cursor().shape() == Qt.PointingHandCursor

        assert win.console_toggle.accessibleName() == "Toggle Runtime Console"
        assert win.console_toggle.toolTip() != ""
        assert win.console_toggle.cursor().shape() == Qt.PointingHandCursor

        assert win.console_autoscroll_toggle.accessibleName() == "Toggle Console Auto-scroll"
        assert win.console_autoscroll_toggle.toolTip() != ""
        assert win.console_autoscroll_toggle.cursor().shape() == Qt.PointingHandCursor

        assert win.clear_console_button.accessibleName() == "Clear Runtime Console"
        assert win.clear_console_button.toolTip() != ""
        assert win.clear_console_button.cursor().shape() == Qt.PointingHandCursor

        assert win.copy_console_button.accessibleName() == "Copy Runtime Console"
        assert win.copy_console_button.toolTip() != ""
        assert win.copy_console_button.cursor().shape() == Qt.PointingHandCursor

        assert win.console_output.accessibleName() == "Runtime Console Log Output"
        assert win.console_output.toolTip() != ""

        # Test copy console action
        win.console_output.setPlainText("sample deepslice log")
        win._copy_console()
        assert QApplication.clipboard().text() == "sample deepslice log"
        assert win.copy_console_button.text() == "✓ Copied!"
    finally:
        win.close()
        win.deleteLater()


def test_header_toolbar_accessibility_and_cursors():
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        assert win.hardware_button.accessibleName() == "Hardware Health and Acceleration Info"
        assert "diagnostics" in win.hardware_button.toolTip().lower()
        assert win.hardware_button.cursor().shape() == Qt.PointingHandCursor

        assert win.theme_toggle_button.accessibleName() == "Toggle Application Theme"
        assert win.theme_toggle_button.toolTip() != ""
        assert win.theme_toggle_button.cursor().shape() == Qt.PointingHandCursor

        assert win.new_session_button.accessibleName() == "Start New Session"
        assert "workspace" in win.new_session_button.toolTip().lower()
        assert win.new_session_button.cursor().shape() == Qt.PointingHandCursor

        assert win.save_session_button.accessibleName() == "Save Current Session"
        assert "save" in win.save_session_button.toolTip().lower()
        assert win.save_session_button.cursor().shape() == Qt.PointingHandCursor

        assert win.load_session_button.accessibleName() == "Load Session or QuickNII Project"
        assert "load" in win.load_session_button.toolTip().lower()
        assert win.load_session_button.cursor().shape() == Qt.PointingHandCursor

        assert win.session_io_spinner.accessibleName() == "Session I/O Activity Progress"
        assert win.session_io_spinner.toolTip() != ""

        assert win.shortcut_help_button.accessibleName() == "Keyboard Shortcuts Reference"
        assert "shortcut" in win.shortcut_help_button.toolTip().lower()
        assert win.shortcut_help_button.cursor().shape() == Qt.PointingHandCursor

        assert win.preferences_button.accessibleName() == "Application Preferences"
        assert "preferences" in win.preferences_button.toolTip().lower()
        assert win.preferences_button.cursor().shape() == Qt.PointingHandCursor

        assert win.about_button.accessibleName() == "About DeepSlice Desktop"
        assert "version" in win.about_button.toolTip().lower()
        assert win.about_button.cursor().shape() == Qt.PointingHandCursor

        assert win.error_menu_button.accessibleName() == "Runtime Error Center"
        assert "error" in win.error_menu_button.toolTip().lower()
        assert win.error_menu_button.cursor().shape() == Qt.PointingHandCursor
    finally:
        win.close()
        win.deleteLater()


def test_hardware_health_cpu_fallback(monkeypatch):
    import sys
    from unittest.mock import patch

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Simulate tensorflow import error
        with patch.dict(sys.modules, {"tensorflow": None}):
            with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                win._show_hardware_health()
                mock_info.assert_called_once()
                args, _ = mock_info.call_args
                assert "Hardware Health" in args
                assert "Mode: CPU (Lightweight)" in args[2]
                assert "CPU fallback" in args[2]
    finally:
        win.close()
        win.deleteLater()


def test_export_page_accessibility_and_copy_feedback():
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        assert win.output_dir_edit.accessibleName() == "Export Output Directory"
        assert ("directory" in win.output_dir_edit.toolTip().lower() or "folder" in win.output_dir_edit.toolTip().lower())

        assert win.browse_output_dir_button.accessibleName() == "Browse Export Output Directory"
        assert win.browse_output_dir_button.cursor().shape() == Qt.PointingHandCursor

        assert win.output_basename_edit.accessibleName() == "Export Base Filename"
        assert win.output_format_combo.accessibleName() == "Export Format"

        assert win.output_format_help_button.accessibleName() == "Export Format Documentation Help"
        assert win.output_format_help_button.cursor().shape() == Qt.PointingHandCursor

        assert win.export_size_estimate_label.accessibleName() == "Estimated Export File Size"

        assert win.export_button.accessibleName() == "Export Predictions Button"
        assert win.export_button.cursor().shape() == Qt.PointingHandCursor

        assert win.open_export_dir_button.accessibleName() == "Open Export Folder"
        assert win.open_export_dir_button.cursor().shape() == Qt.PointingHandCursor

        assert win.copy_export_path_button.accessibleName() == "Copy Export File Path"
        assert win.copy_export_path_button.cursor().shape() == Qt.PointingHandCursor

        assert win.report_button.accessibleName() == "Generate PDF Report Button"
        assert win.report_button.cursor().shape() == Qt.PointingHandCursor

        assert win.preview_report_button.accessibleName() == "Preview PDF Report Button"
        assert win.preview_report_button.cursor().shape() == Qt.PointingHandCursor

        assert win.pdf_content_group.accessibleName() == "PDF Report Contents Configuration"
        assert win.pdf_include_stats.accessibleName() == "Include Summary Stats in Report"
        assert win.pdf_include_stats.cursor().shape() == Qt.PointingHandCursor
        assert win.pdf_include_plot.accessibleName() == "Include Linearity Plot in Report"
        assert win.pdf_include_plot.cursor().shape() == Qt.PointingHandCursor
        assert win.pdf_include_images.accessibleName() == "Include Sample Images in Report"
        assert win.pdf_include_images.cursor().shape() == Qt.PointingHandCursor
        assert win.pdf_include_angles.accessibleName() == "Include Angle Metrics in Report"
        assert win.pdf_include_angles.cursor().shape() == Qt.PointingHandCursor

        assert win.quicknii_path_edit.accessibleName() == "QuickNII Executable Path"
        assert win.quicknii_browse_button.accessibleName() == "Browse QuickNII Executable"
        assert win.quicknii_browse_button.cursor().shape() == Qt.PointingHandCursor
        assert win.open_quicknii_button.accessibleName() == "Open in QuickNII Button"
        assert win.open_quicknii_button.cursor().shape() == Qt.PointingHandCursor

        assert win.summary_label.accessibleName() == "Export Processed Section Count Summary"
        assert win.deviation_label.accessibleName() == "Mean Angular Deviation Metric"
        assert win.markers_label.accessibleName() == "Export Validation Markers"
        assert win.export_notes.accessibleName() == "Export Format Instructions and Notes"

        # Test visual copy confirmation on copy_export_path
        win.last_export_basepath = "/fake/path/to/result"
        win._copy_export_path()
        assert win.copy_export_path_button.text() == "✓ Copied!"
        assert "/fake/path/to/result.json" in QApplication.clipboard().text()
    finally:
        win.close()
        win.deleteLater()


def test_helper_dialogs_accessibility_and_headless_safety():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # About dialog
        about_box = win._show_about_dialog()
        assert about_box.accessibleName() == "About DeepSlice Information Dialog"
        assert "DeepSlice Desktop" in about_box.text()

        # Shortcuts dialog
        shortcuts_box = win._show_shortcuts_help()
        assert shortcuts_box.accessibleName() == "Keyboard Shortcuts Help Dialog"
        assert "Keyboard Shortcuts" in shortcuts_box.text()

        # Naming helper dialog
        naming_box = win._show_naming_helper()
        assert naming_box.accessibleName() == "Naming Convention Help Dialog"
        assert "Naming Convention" in naming_box.text()

        # Orientation guide dialog
        orient_box = win._show_orientation_guide()
        assert orient_box.accessibleName() == "Orientation Guide Dialog"
        assert "Orientation Guide" in orient_box.text()

        # Direction guide dialog
        dir_box = win._show_direction_guide()
        assert dir_box.accessibleName() == "Direction Guide Dialog"
        assert "Direction Override" in dir_box.text()

        # Ensemble explanation dialog
        ens_box = win._show_ensemble_explanation()
        assert ens_box.accessibleName() == "Ensemble Prediction Help Dialog"
        assert "Ensemble Prediction" in ens_box.text()

        # Export format help dialog
        fmt_box = win._show_export_format_help()
        assert fmt_box.accessibleName() == "Export Format Documentation Dialog"
        assert "QuickNII" in fmt_box.text()
    finally:
        win.close()
        win.deleteLater()


def test_ingestion_and_configuration_pages_accessibility_and_cursors():
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Ingestion page controls
        assert win.add_folder_button.accessibleName() == "Add Folder"
        assert win.add_folder_button.cursor().shape() == Qt.PointingHandCursor
        assert win.add_files_button.accessibleName() == "Add Files"
        assert win.add_files_button.cursor().shape() == Qt.PointingHandCursor
        assert win.clear_images_button.accessibleName() == "Clear All Images"
        assert win.clear_images_button.cursor().shape() == Qt.PointingHandCursor
        assert win.naming_helper_button.accessibleName() == "Naming Helper"
        assert win.naming_helper_button.cursor().shape() == Qt.PointingHandCursor
        assert win.orientation_guide_button.accessibleName() == "Orientation Guide"
        assert win.orientation_guide_button.cursor().shape() == Qt.PointingHandCursor
        assert win.enable_section_numbers_checkbox.accessibleName() == "Detect Section Numbers"
        assert win.enable_section_numbers_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.legacy_parsing_checkbox.accessibleName() == "Legacy Parser Fallback"
        assert win.legacy_parsing_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.orientation_combo.accessibleName() == "Anatomical Orientation"
        assert win.thumbnail_sort_combo.accessibleName() == "Thumbnail Sort Order"
        assert win.thumbnail_filter_edit.accessibleName() == "Thumbnail Filter"
        assert win.index_table.accessibleName() == "Section Indices Table"
        assert win.slice_count_label.accessibleName() == "Slice Count Label"
        assert win.file_size_summary_label.accessibleName() == "File Size Summary Label"
        assert win.ingestion_summary_banner.accessibleName() == "Ingestion Summary Banner"
        assert win.ingestion_warning_label.accessibleName() == "Ingestion Warning Label"

        # Configuration page controls
        assert win.mouse_radio.accessibleName() == "Mouse CCFv3 Atlas"
        assert win.mouse_radio.cursor().shape() == Qt.PointingHandCursor
        assert win.rat_radio.accessibleName() == "Rat Waxholm Atlas"
        assert win.rat_radio.cursor().shape() == Qt.PointingHandCursor
        assert win.auto_thickness_checkbox.accessibleName() == "Auto-estimate Thickness"
        assert win.auto_thickness_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.suggest_thickness_button.accessibleName() == "Suggest Thickness"
        assert win.suggest_thickness_button.cursor().shape() == Qt.PointingHandCursor
        assert win.direction_guide_button.accessibleName() == "Direction Guide"
        assert win.direction_guide_button.cursor().shape() == Qt.PointingHandCursor
        assert win.ensemble_checkbox.accessibleName() == "Ensemble Prediction"
        assert win.ensemble_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.ensemble_help_button.accessibleName() == "Ensemble Help"
        assert win.ensemble_help_button.cursor().shape() == Qt.PointingHandCursor
        assert win.secondary_model_checkbox.accessibleName() == "Secondary Model Only"
        assert win.secondary_model_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.legacy_from_config_checkbox.accessibleName() == "Legacy Section Parser"
        assert win.legacy_from_config_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.preview_training_split_button.accessibleName() == "Preview Training Split"
        assert win.preview_training_split_button.cursor().shape() == Qt.PointingHandCursor
        assert win.save_training_split_button.accessibleName() == "Save Split Manifest"
        assert win.save_training_split_button.cursor().shape() == Qt.PointingHandCursor
        assert win.save_training_metadata_button.accessibleName() == "Save Training Metadata"
        assert win.save_training_metadata_button.cursor().shape() == Qt.PointingHandCursor
        assert win.validate_configuration_button.accessibleName() == "Validate Configuration"
        assert win.validate_configuration_button.cursor().shape() == Qt.PointingHandCursor
        assert win.tech_toggle.accessibleName() == "Technical Insights Toggle"
        assert win.tech_toggle.cursor().shape() == Qt.PointingHandCursor
    finally:
        win.close()
        win.deleteLater()


def test_curation_page_controls_accessibility_and_cursors():
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Curation list header & reordering controls
        assert win.curation_prev_button.accessibleName() == "Previous Section"
        assert win.curation_prev_button.cursor().shape() == Qt.PointingHandCursor
        assert win.curation_next_button.accessibleName() == "Next Section"
        assert win.curation_next_button.cursor().shape() == Qt.PointingHandCursor
        assert win.curation_select_all_btn.accessibleName() == "Select All Bad Section Flags"
        assert win.curation_select_all_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.curation_deselect_all_btn.accessibleName() == "Deselect All Bad Section Flags"
        assert win.curation_deselect_all_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.confidence_filter_combo.accessibleName() == "Filter Slices by Confidence"
        assert win.slice_flag_list.accessibleName() == "Curated Slices List"

        # Action and propagation buttons
        assert win.apply_bad_sections_button.accessibleName() == "Apply Bad Section Flags"
        assert win.apply_bad_sections_button.cursor().shape() == Qt.PointingHandCursor
        assert win.apply_manual_order_button.accessibleName() == "Apply Manual Reordering"
        assert win.apply_manual_order_button.cursor().shape() == Qt.PointingHandCursor
        assert win.move_slice_up_button.accessibleName() == "Move Selected Slice Up"
        assert win.move_slice_up_button.cursor().shape() == Qt.PointingHandCursor
        assert win.move_slice_down_button.accessibleName() == "Move Selected Slice Down"
        assert win.move_slice_down_button.cursor().shape() == Qt.PointingHandCursor
        assert win.detect_outliers_button.accessibleName() == "Detect Alignment Outliers"
        assert win.detect_outliers_button.cursor().shape() == Qt.PointingHandCursor
        assert win.reset_flags_button.accessibleName() == "Reset All Flags"
        assert win.reset_flags_button.cursor().shape() == Qt.PointingHandCursor
        assert win.auto_flag_low_conf_button.accessibleName() == "Auto-Flag Low Confidence Sections"
        assert win.auto_flag_low_conf_button.cursor().shape() == Qt.PointingHandCursor
        assert win.interpolate_bad_depth_button.accessibleName() == "Interpolate Flagged Depths"
        assert win.interpolate_bad_depth_button.cursor().shape() == Qt.PointingHandCursor
        assert win.toggle_current_flag_button.accessibleName() == "Toggle Current Section Flag"
        assert win.toggle_current_flag_button.cursor().shape() == Qt.PointingHandCursor
        assert win.slice_note_edit.accessibleName() == "Per-Slice Atlas Note"
        assert win.save_slice_note_button.accessibleName() == "Save Slice Note"
        assert win.save_slice_note_button.cursor().shape() == Qt.PointingHandCursor

        # Propagation controls
        assert win.normalize_angles_button.accessibleName() == "Normalize Angles"
        assert win.normalize_angles_button.cursor().shape() == Qt.PointingHandCursor
        assert win.enforce_order_button.accessibleName() == "Enforce Index Order"
        assert win.enforce_order_button.cursor().shape() == Qt.PointingHandCursor
        assert win.enforce_spacing_button.accessibleName() == "Enforce Index Spacing"
        assert win.enforce_spacing_button.cursor().shape() == Qt.PointingHandCursor
        assert win.ml_spin.accessibleName() == "Mediolateral (ML) Angle"
        assert win.dv_spin.accessibleName() == "Dorsoventral (DV) Angle"
        assert win.apply_manual_angles_button.accessibleName() == "Apply Manual Angles"
        assert win.apply_manual_angles_button.cursor().shape() == Qt.PointingHandCursor
        assert win.undo_button.accessibleName() == "Undo Curation Action"
        assert win.undo_button.cursor().shape() == Qt.PointingHandCursor
        assert win.redo_button.accessibleName() == "Redo Curation Action"
        assert win.redo_button.cursor().shape() == Qt.PointingHandCursor

        # Anchor controls
        assert win.anchor_depth_spin.accessibleName() == "Target AP Depth"
        assert win.set_anchor_button.accessibleName() == "Set or Update Anchor"
        assert win.set_anchor_button.cursor().shape() == Qt.PointingHandCursor
        assert win.remove_anchor_button.accessibleName() == "Remove Anchor"
        assert win.remove_anchor_button.cursor().shape() == Qt.PointingHandCursor
        assert win.apply_anchor_interpolation_button.accessibleName() == "Distribute Between Anchors"
        assert win.apply_anchor_interpolation_button.cursor().shape() == Qt.PointingHandCursor
        assert win.clear_anchor_button.accessibleName() == "Clear All Anchors"
        assert win.clear_anchor_button.cursor().shape() == Qt.PointingHandCursor
        assert win.anchor_list.accessibleName() == "Defined Anchors List"

        # Right panel controls
        assert win.zoom_fit_button.accessibleName() == "Fit Linearity Plot"
        assert win.zoom_fit_button.cursor().shape() == Qt.PointingHandCursor
        assert win.zoom_in_button.accessibleName() == "Zoom In Plot"
        assert win.zoom_in_button.cursor().shape() == Qt.PointingHandCursor
        assert win.zoom_out_button.accessibleName() == "Zoom Out Plot"
        assert win.zoom_out_button.cursor().shape() == Qt.PointingHandCursor
        assert win.enable_atlas_preview_checkbox.accessibleName() == "Enable Atlas Preview"
        assert win.enable_atlas_preview_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_volume_combo.accessibleName() == "Atlas Reference Volume"
        assert win.enable_blend_overlay_checkbox.accessibleName() == "Blend Atlas on Histology"
        assert win.enable_blend_overlay_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.blend_slider.accessibleName() == "Atlas Blend Opacity"
        assert win.before_after_toggle.accessibleName() == "Before/After Overlay"
        assert win.before_after_toggle.cursor().shape() == Qt.PointingHandCursor
        assert win.loupe_toggle.accessibleName() == "Magnification Loupe"
        assert win.loupe_toggle.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_flip_x_checkbox.accessibleName() == "Flip Atlas Horizontally"
        assert win.atlas_flip_x_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_flip_y_checkbox.accessibleName() == "Flip Atlas Vertically"
        assert win.atlas_flip_y_checkbox.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_rotate_combo.accessibleName() == "Rotate Atlas"
        assert win.atlas_rotate_combo.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_scale_slider.accessibleName() == "Atlas Scale Slider"
        assert win.atlas_offset_x_slider.accessibleName() == "Atlas Horizontal Offset"
        assert win.atlas_offset_y_slider.accessibleName() == "Atlas Vertical Offset"
        assert win.confidence_panel_toggle.accessibleName() == "Toggle Confidence Overlay Preview"
        assert win.confidence_panel_toggle.cursor().shape() == Qt.PointingHandCursor
    finally:
        win.close()
        win.deleteLater()






