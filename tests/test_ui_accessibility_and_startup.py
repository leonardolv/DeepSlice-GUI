"""Tests for UI accessibility metadata, DropArea, ThumbnailListWidget, and non-blocking headless startup."""

from __future__ import annotations

from PySide6.QtCore import Qt

from PySide6.QtWidgets import QApplication, QPushButton, QMessageBox

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
        assert win.objectName() == "deepSliceMainWindow"
        assert win.accessibleName() == "DeepSlice Desktop Main Window"
        assert win.accessibleDescription() == "Automated whole-brain histological section alignment workstation"
        assert win.current_page_index == 0
        assert win.current_page_title == win.STEP_GUIDE[0][0]
        assert win.loaded_image_count == 0
        assert win.selected_species in ("mouse", "rat")
        assert win.has_predictions is False
        assert win.is_dirty is False
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
            assert dlg.accessibleName() == "Preferences Dialog"
            assert dlg.form is not None
            assert dlg.species_label.objectName() == "prefSpeciesLabel"
            assert dlg.theme_label.objectName() == "prefThemeLabel"
            assert dlg.output_dir_label.objectName() == "prefOutputDirLabel"
            assert dlg.quicknii_label.objectName() == "prefQuickNiiLabel"
            assert dlg.console_label.objectName() == "prefConsoleLabel"
            vals = dlg.get_values()
            assert "default_species" in vals
            assert "theme" in vals
            dlg.set_values({"default_species": "rat", "theme": "light", "default_output_directory": "C:/custom_set", "console_always_visible": True})
            assert dlg.species_combo.currentText() == "rat"
            assert dlg.theme_combo.currentText() == "light"
            assert dlg.output_dir_edit.text() == "C:/custom_set"
            assert dlg.console_always_visible.isChecked()
            assert dlg.species_combo.accessibleName() == "Default species for new sessions"
            assert dlg.theme_combo.accessibleName() == "Application theme"
            assert dlg.output_dir_edit.accessibleName() == "Default output directory path"
            assert dlg.output_dir_browse.accessibleName() == "Browse default output directory"
            assert dlg.quicknii_edit.accessibleName() == "Default QuickNII executable path"
            assert dlg.quicknii_browse.accessibleName() == "Browse QuickNII executable"
            assert dlg.console_always_visible.accessibleName() == "Show runtime console by default"
            assert dlg.output_dir_browse.cursor().shape() == Qt.PointingHandCursor
            assert dlg.quicknii_browse.cursor().shape() == Qt.PointingHandCursor
            assert dlg.species_combo.cursor().shape() == Qt.PointingHandCursor
            assert dlg.theme_combo.cursor().shape() == Qt.PointingHandCursor
            assert dlg.console_always_visible.cursor().shape() == Qt.PointingHandCursor
            assert dlg.reset_btn is not None
            assert dlg.reset_btn.accessibleName() == "Reset preferences to factory defaults"
            assert "default values" in dlg.reset_btn.toolTip().lower()
            assert dlg.reset_btn.cursor().shape() == Qt.PointingHandCursor
            assert dlg.ok_btn is not None
            assert dlg.ok_btn.accessibleName() == "Save preferences"
            assert dlg.ok_btn.cursor().shape() == Qt.PointingHandCursor
            assert dlg.cancel_btn is not None
            assert dlg.cancel_btn.accessibleName() == "Cancel preferences"
            assert dlg.cancel_btn.cursor().shape() == Qt.PointingHandCursor

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

        # Verify headless safe execution and signal wiring of _open_preferences_dialog
        opened_dlg = win._open_preferences_dialog()
        assert opened_dlg is not None
        opened_dlg.species_combo.setCurrentText("rat")
        opened_dlg.theme_combo.setCurrentText("light")
        opened_dlg.output_dir_edit.setText("C:/test_output")
        opened_dlg.accept()
        assert str(win._settings.value("default_species")) == "rat"
        assert str(win._settings.value("theme")) == "light"
        assert str(win._settings.value("default_output_directory")) == "C:/test_output"
        assert win.state.species == "rat"
        opened_dlg.deleteLater()
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
            dlg = win._show_hardware_health()
            assert dlg is not None
            assert dlg.windowTitle() == "Hardware Health"
            assert "Mode: CPU (Lightweight)" in dlg.text()
            assert "CPU fallback" in dlg.text()
            assert dlg.accessibleName() == "Hardware Health Diagnostics Dialog"
            assert dlg.objectName() == "hardwareHealthDialog"
            btn = dlg.findChild(QPushButton, "hardwareHealthOkBtn")
            assert btn is not None
            assert btn.cursor().shape() == Qt.PointingHandCursor
            assert "Hardware" in btn.accessibleName()
            assert btn.toolTip() != ""
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
        assert about_box.objectName() == "aboutDialog"
        assert about_box.accessibleName() == "About DeepSlice Information Dialog"
        assert "DeepSlice Desktop" in about_box.text()
        for btn in about_box.buttons():
            assert btn.cursor().shape() == Qt.PointingHandCursor

        # Shortcuts dialog
        shortcuts_box = win._show_shortcuts_help()
        assert shortcuts_box.objectName() == "shortcutsDialog"
        assert shortcuts_box.accessibleName() == "Keyboard Shortcuts Help Dialog"
        assert "Keyboard Shortcuts" in shortcuts_box.text()
        for btn in shortcuts_box.buttons():
            assert btn.cursor().shape() == Qt.PointingHandCursor

        # Naming helper dialog
        naming_box = win._show_naming_helper()
        assert naming_box.objectName() == "namingHelperDialog"
        assert naming_box.accessibleName() == "Naming Convention Help Dialog"
        assert "Naming Convention" in naming_box.text()
        assert len(naming_box.buttons()) == 1
        btn = naming_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "namingHelperOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Orientation guide dialog
        orient_box = win._show_orientation_guide()
        assert orient_box.objectName() == "orientationGuideDialog"
        assert orient_box.accessibleName() == "Orientation Guide Dialog"
        assert "Orientation Guide" in orient_box.text()
        assert len(orient_box.buttons()) == 1
        btn = orient_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "orientationGuideOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Direction guide dialog
        dir_box = win._show_direction_guide()
        assert dir_box.objectName() == "directionGuideDialog"
        assert dir_box.accessibleName() == "Direction Guide Dialog"
        assert "Direction Override" in dir_box.text()
        assert len(dir_box.buttons()) == 1
        btn = dir_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "directionGuideOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Ensemble explanation dialog
        ens_box = win._show_ensemble_explanation()
        assert ens_box.objectName() == "ensembleExplanationDialog"
        assert ens_box.accessibleName() == "Ensemble Prediction Help Dialog"
        assert "Ensemble Prediction" in ens_box.text()
        assert len(ens_box.buttons()) == 1
        btn = ens_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "ensembleExplanationOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Export format help dialog
        fmt_box = win._show_export_format_help()
        assert fmt_box.objectName() == "exportFormatHelpDialog"
        assert fmt_box.accessibleName() == "Export Format Documentation Dialog"
        assert "QuickNII" in fmt_box.text()
        assert len(fmt_box.buttons()) == 1
        btn = fmt_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "exportFormatHelpOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Onboarding dialog
        onboarding_box = win._show_onboarding_dialog()
        assert onboarding_box.objectName() == "onboardingDialog"
        assert onboarding_box.accessibleName() == "Onboarding Walkthrough Dialog"
        assert "Welcome to DeepSlice" in onboarding_box.windowTitle()
        assert len(onboarding_box.buttons()) == 1
        btn = onboarding_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "onboardingOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # What's New dialog
        whats_new_box = win._show_whats_new_dialog("1.0.0")
        assert whats_new_box.objectName() == "whatsNewDialog"
        assert whats_new_box.accessibleName() == "What's New Highlights Dialog"
        assert "What's New" in whats_new_box.windowTitle()
        assert len(whats_new_box.buttons()) == 1
        btn = whats_new_box.button(QMessageBox.StandardButton.Ok)
        assert btn.objectName() == "whatsNewOkBtn"
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert len(btn.toolTip()) > 0
        assert len(btn.accessibleName()) > 0

        # Hardware health dialog
        health_box = win._show_hardware_health()
        assert health_box.objectName() == "hardwareHealthDialog"
        assert health_box.accessibleName() == "Hardware Health Diagnostics Dialog"
        assert "Hardware Health" in health_box.windowTitle()
        for btn in health_box.buttons():
            assert btn.cursor().shape() == Qt.PointingHandCursor

        # Context help headless safety
        win._open_context_help()
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


def test_shortcuts_and_linearity_zoom_tooltips():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Check tooltips reflect shortcut hints
        assert "Ctrl+," in win.preferences_button.toolTip()
        assert "Ctrl+0" in win.zoom_fit_button.toolTip()
        assert "Ctrl++" in win.zoom_in_button.toolTip()
        assert "Ctrl+-" in win.zoom_out_button.toolTip()

        # Check shortcut bindings exist
        seq_strings = [s.key().toString() for s in win._shortcuts]
        assert "Ctrl+," in seq_strings
        assert "Ctrl+0" in seq_strings
        assert "Ctrl++" in seq_strings
        assert "Ctrl+-" in seq_strings

        # Check shortcuts help dialog text
        dialog = win._show_shortcuts_help()
        dialog_text = dialog.text()
        assert "Ctrl+," in dialog_text
        assert "Ctrl+0" in dialog_text
        assert "Ctrl++ / Ctrl+-" in dialog_text
        assert dialog.objectName() == "shortcutsDialog"
        assert dialog.accessibleName() == "Keyboard Shortcuts Help Dialog"
        btn = dialog.findChild(QPushButton, "shortcutsOkBtn")
        assert btn is not None
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert "Shortcuts" in btn.accessibleName()
        assert btn.toolTip() != ""
    finally:
        win.close()
        win.deleteLater()









def test_polish_pass_layout_and_theme_rules():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Pages are scrollable so the Review page cannot force a huge minimum window size.
        for i in range(win.stack.count()):
            assert win.stack.widget(i).objectName() == "PageScroll"
            assert win.stack.widget(i).minimumSizeHint().width() < 400
        # Secondary buttons are marked; primary actions are left alone.
        assert win.hardware_button.property("variant") == "secondary"
        assert win.detect_outliers_button.property("variant") == "secondary"
        assert win.export_button.property("variant") != "secondary"
        assert win.run_alignment_button.property("variant") != "secondary"
        # Labels no longer paint opaque bars over panels; tooltips are themed.
        sheet = win.styleSheet()
        assert "QLabel, QCheckBox, QRadioButton" in sheet and "background: transparent" in sheet
        assert 'QPushButton[variant="secondary"]' in sheet and "QToolTip" in sheet
        # An unknown version is not shown as "vUnknown".
        assert "vUnknown" not in win.project_label.text()
        assert win.collapse_sidebar_button.accessibleName()
        # Narrow windows move actions to their own row.
        win._place_top_bar_actions(wide=False)
        assert win._top_row_actions.count() > 0
        win._place_top_bar_actions(wide=True)
        assert win._top_row_actions.count() == 0
    finally:
        win.close()
        win.deleteLater()


def test_fullscreen_preview_and_logged_error_accessibility_and_cursors():
    from PySide6.QtWidgets import QListWidgetItem, QPushButton, QMessageBox
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        item = QListWidgetItem("test_slice.png")
        item.setData(Qt.UserRole, "test_slice.png")
        preview_dlg = win._open_fullscreen_thumbnail_preview(item)
        try:
            assert preview_dlg.objectName() == "fullscreenThumbnailPreviewDialog"
            close_btn = preview_dlg.findChild(QPushButton, "previewCloseBtn")
            assert close_btn is not None
            assert close_btn.cursor().shape() == Qt.PointingHandCursor
            assert close_btn.accessibleName() == "Close fullscreen preview"
        finally:
            preview_dlg.deleteLater()

        win._show_logged_error("Test Error", "Test Context", "Test Error Text")
        msg_box = win.findChild(QMessageBox, "loggedErrorDialog")
        assert msg_box is not None
        assert "Test Error" in msg_box.accessibleName()
        for btn in msg_box.buttons():
            assert btn.cursor().shape() == Qt.PointingHandCursor
    finally:
        win.close()
        win.deleteLater()

def test_training_actions_and_split_dialog_accessibility_and_cursors(tmp_path):
    from PySide6.QtWidgets import QMessageBox
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Check widget object names, accessible names, cursors, and tooltips
        assert win.preview_training_split_button.objectName() == "previewTrainingSplitBtn"
        assert win.preview_training_split_button.cursor().shape() == Qt.PointingHandCursor
        assert win.preview_training_split_button.toolTip() != ""

        assert win.save_training_split_button.objectName() == "saveTrainingSplitBtn"
        assert win.save_training_split_button.cursor().shape() == Qt.PointingHandCursor
        assert win.save_training_split_button.toolTip() != ""

        assert win.save_training_metadata_button.objectName() == "saveTrainingMetadataBtn"
        assert win.save_training_metadata_button.cursor().shape() == Qt.PointingHandCursor
        assert win.save_training_metadata_button.toolTip() != ""

        assert win.training_split_summary_label.objectName() == "trainingSplitSummaryLabel"
        assert win.training_split_summary_label.accessibleName() == "Training dataset split summary"

        # Non-blocking headless toast when images empty
        res = win._preview_training_split()
        assert res is None
        assert win._save_training_split_manifest() is None
        assert win._save_training_metadata_manifest() is None

        # When images exist, verify dialog construction and properties
        img_file = tmp_path / "slice_s001.png"
        img_file.write_bytes(b"dummy")
        win.state.image_paths = [str(img_file)]
        win.state.section_numbers = [1]
        win.state.section_depth = [10.0]

        dlg = win._preview_training_split()
        assert dlg is not None
        try:
            assert dlg.objectName() == "trainingSplitDialog"
            assert dlg.accessibleName() == "Training Split Preview Dialog"
            ok_btn = dlg.button(QMessageBox.StandardButton.Ok)
            assert ok_btn is not None
            assert ok_btn.objectName() == "trainingSplitOkBtn"
            assert ok_btn.accessibleName() == "Dismiss training split preview dialog"
            for btn in dlg.buttons():
                assert btn.cursor().shape() == Qt.PointingHandCursor
        finally:
            dlg.deleteLater()
    finally:
        win.close()
        win.deleteLater()



def test_about_dialog_accessibility_and_button_cursors():
    from PySide6.QtWidgets import QPushButton
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        dlg = win._show_about_dialog()
        assert dlg is not None
        assert dlg.objectName() == "aboutDialog"
        assert dlg.windowTitle() == "About DeepSlice"
        assert dlg.accessibleName() == "About DeepSlice Information Dialog"
        btn = dlg.findChild(QPushButton, "aboutOkBtn")
        assert btn is not None
        assert btn.cursor().shape() == Qt.PointingHandCursor
        assert "About" in btn.accessibleName()
        assert btn.toolTip() != ""
    finally:
        win.close()
        win.deleteLater()


def test_actionable_warning_headless_and_accessibility():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        box = win._actionable_warning(
            title="Alignment Warning",
            message="Section spacing discrepancy detected.",
            actions=[("Ignore and Proceed", lambda: None)],
        )
        assert box is not None
        assert box.objectName() == "actionableWarningDialog"
        assert "Alignment Warning" in box.accessibleName()
        close_btn = box.findChild(QPushButton, "actionableWarningCloseBtn")
        assert close_btn is not None
        assert close_btn.cursor().shape() == Qt.PointingHandCursor
        action_btn = box.findChild(QPushButton, "actionableWarningActionBtn0")
        assert action_btn is not None
        assert action_btn.cursor().shape() == Qt.PointingHandCursor
        assert action_btn.accessibleName() == "Action: Ignore and Proceed"
    finally:
        win.close()
        win.deleteLater()


def test_thumbnail_and_curation_context_menus_accessibility_and_actions():
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtCore import QPoint
        item = QListWidgetItem("dummy_slice_01.png")
        item.setData(Qt.UserRole, "/path/to/dummy_slice_01.png")
        win.thumbnail_list.addItem(item)

        menu = win._show_thumbnail_context_menu(QPoint(10, 10))
        assert menu is not None
        assert menu.objectName() == "thumbnailContextMenu"
        assert menu.accessibleName() == "Thumbnail Item Context Menu"

        actions = {a.objectName(): a for a in menu.actions()}
        assert "thumbnailPreviewAction" in actions
        assert "thumbnailCopyPathAction" in actions
        assert "thumbnailRemoveAction" in actions
        assert actions["thumbnailPreviewAction"].text() == "Preview Fullscreen"
        assert actions["thumbnailCopyPathAction"].text() == "Copy Image Path"
        assert actions["thumbnailRemoveAction"].text() == "Remove this image"

        cur_item = QListWidgetItem("Section 01")
        cur_item.setData(Qt.UserRole, 0)
        win.slice_flag_list.addItem(cur_item)
        cur_menu = win._show_curation_context_menu(QPoint(10, 10))
        assert cur_menu is not None
        assert cur_menu.objectName() == "curationContextMenu"
        assert cur_menu.accessibleName() == "Curation Slice Context Menu"
        cur_actions = {a.objectName(): a for a in cur_menu.actions()}
        assert "curationToggleFlagAction" in cur_actions
        assert "curationJumpViewerAction" in cur_actions
        assert "curationSetAnchorAction" in cur_actions
        assert "curationRemoveAnchorAction" in cur_actions
        assert "curationEditNoteAction" in cur_actions
        assert "curationCopyPathAction" in cur_actions
        assert "curationShowDetailsAction" in cur_actions
    finally:
        win.close()
        win.deleteLater()


def test_export_and_report_headless_actionable_warnings():
    """Verify _export_predictions, _generate_report, and _preview_report trigger actionable warnings headlessly."""
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        win.state.predictions = None
        warning_calls = []

        orig_actionable = win._actionable_warning
        def mock_warning(*args, **kwargs):
            warning_calls.append((args, kwargs))
            return orig_actionable(*args, **kwargs)

        win._actionable_warning = mock_warning

        # 1. Export predictions without predictions
        win._export_predictions()
        assert len(warning_calls) == 1
        args, kwargs = warning_calls[-1]
        assert kwargs.get("title") == "Export"
        assert "No predictions available" in kwargs.get("message")
        actions = kwargs.get("actions") or []
        assert any("Run Prediction" in a[0] for a in actions)

        # 2. Generate report without predictions
        win._generate_report()
        assert len(warning_calls) == 2
        args, kwargs = warning_calls[-1]
        assert kwargs.get("title") == "Report"
        assert "No predictions available" in kwargs.get("message")

        # 3. Preview report without predictions
        win._preview_report()
        assert len(warning_calls) == 3
        args, kwargs = warning_calls[-1]
        assert kwargs.get("title") == "Report Preview"
        assert "No predictions available" in kwargs.get("message")
    finally:
        win.close()
        win.deleteLater()


def test_alignment_and_anchor_headless_actionable_warnings():
    from DeepSlice.gui.main_window import DeepSliceMainWindow
    import pandas as pd
    import numpy as np

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        warning_calls = []
        orig_actionable = win._actionable_warning

        def mock_warning(*args, **kwargs):
            warning_calls.append((args, kwargs))
            return orig_actionable(*args, **kwargs)

        win._actionable_warning = mock_warning

        # 1. Start prediction worker without images loaded
        win.state.image_paths = []
        win._run_alignment()
        assert len(warning_calls) == 1
        args, kwargs = warning_calls[-1]
        assert kwargs.get("title") == "Cannot Run Alignment"
        assert "No images loaded" in kwargs.get("message")
        actions = kwargs.get("actions") or []
        assert any("Ingestion Page" in a[0] for a in actions)

        # 2. Anchor alignment with depth payload mismatch
        win.state.predictions = pd.DataFrame({
            "Filenames": ["s1.png", "s2.png"],
            "ox": [0.0, 0.0],
            "oy": [100.0, 200.0],
            "oz": [0.0, 0.0],
            "bad": [False, False],
        })
        win._anchor_depth_targets = {0: 105.0, 1: 195.0}
        # mock linearity_payload to return mismatched length
        orig_lin = win.state.linearity_payload
        win.state.linearity_payload = lambda: {"y": np.array([100.0])}
        win._apply_anchor_interpolation()
        assert len(warning_calls) == 2
        args, kwargs = warning_calls[-1]
        assert kwargs.get("title") == "Anchor Alignment"
        assert "Depth payload does not match" in kwargs.get("message")
        win.state.linearity_payload = orig_lin
    finally:
        win.close()
        win.deleteLater()


def test_headless_session_reset_and_quality_gate_confirmations(qapp):
    win = DeepSliceMainWindow()
    win.show()
    try:
        # 1. Session reset when dirty does not block headlessly
        win.state.is_dirty = True
        win._reset_session()
        assert not win.state.is_dirty

        # 2. Quality gate failure does not block headlessly
        worker_started = []
        win._start_prediction_worker = lambda: worker_started.append(True)
        win._on_quality_gate_error("Sample failure")
        assert len(worker_started) == 1

        # 3. Quality gate warnings do not block headlessly
        win._build_quality_gate_warnings = lambda rep: ["Low resolution detected", "Blur detected"]
        win._on_quality_gate_finished({"has_issues": True})
        assert len(worker_started) == 2

        # 4. closeEvent accepts when dirty under headless runner
        win.state.is_dirty = True
        class MockCloseEvent:
            def __init__(self):
                self.accepted = False
            def accept(self):
                self.accepted = True
            def ignore(self):
                self.accepted = False
        event = MockCloseEvent()
        win.closeEvent(event)
        assert event.accepted
    finally:
        win.close()
        win.deleteLater()


def test_main_window_extended_navigation_and_helper_dialogs(qapp):
    win = DeepSliceMainWindow()
    win.show()
    try:
        # Navigation properties
        assert win.page_count >= 5
        assert win.can_navigate_backward is False
        assert win.can_navigate_forward is True
        assert win.has_images is False
        assert win.has_anchors is False
        assert win.active_theme in ("dark", "light")

        # Programmatic page navigation
        assert win.go_to_page(1) is True
        assert win.current_page_index == 1
        assert win.can_navigate_backward is True

        assert win.go_to_page(0) is True
        assert win.current_page_index == 0

        # Out-of-bounds navigation is safely refused
        assert win.go_to_page(999) is False
        assert win.go_to_page(-1) is False

        # Headless helper dialog assertions
        shortcuts_box = win._show_shortcuts_help()
        assert shortcuts_box.objectName() == "shortcutsDialog"
        assert shortcuts_box.accessibleName() == "Keyboard Shortcuts Help Dialog"

        onboarding_box = win._show_onboarding_dialog()
        assert onboarding_box.objectName() == "onboardingDialog"
        assert onboarding_box.accessibleName() == "Onboarding Walkthrough Dialog"

        whats_new_box = win._show_whats_new_dialog("1.0.0")
        assert whats_new_box.objectName() == "whatsNewDialog"
        assert whats_new_box.accessibleName() == "What's New Highlights Dialog"

        naming_box = win._show_naming_helper()
        assert naming_box.objectName() == "namingHelperDialog"
        assert naming_box.accessibleName() == "Naming Convention Help Dialog"

        orientation_box = win._show_orientation_guide()
        assert orientation_box.objectName() == "orientationGuideDialog"
        assert orientation_box.accessibleName() == "Orientation Guide Dialog"

        direction_box = win._show_direction_guide()
        assert direction_box.objectName() == "directionGuideDialog"
        assert direction_box.accessibleName() == "Direction Guide Dialog"

        ensemble_box = win._show_ensemble_explanation()
        assert ensemble_box.objectName() == "ensembleExplanationDialog"
        assert ensemble_box.accessibleName() == "Ensemble Prediction Help Dialog"

        export_help_box = win._show_export_format_help()
        assert export_help_box.objectName() == "exportFormatHelpDialog"
        assert export_help_box.accessibleName() == "Export Format Documentation Dialog"
    finally:
        win.close()
        win.deleteLater()


def test_main_window_extended_ergonomics_and_species_selection():
    """Assert next_page, previous_page, bad section queries, export properties, and species selection."""
    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # Step forward and backward
        assert win.current_page_index == 0
        assert win.can_navigate_backward is False
        assert win.can_navigate_forward is True

        adv = win.next_page()
        assert adv is True
        assert win.current_page_index == 1

        prev = win.previous_page()
        assert prev is True
        assert win.current_page_index == 0

        # Species selection helper
        assert win.select_species("rat") is True
        assert win.selected_species == "rat"
        assert win.rat_radio.isChecked()

        assert win.select_species("mouse") is True
        assert win.selected_species == "mouse"
        assert win.mouse_radio.isChecked()

        assert win.select_species("invalid_species") is False

        # Bad section queries
        assert win.bad_section_count == 0
        assert win.has_bad_sections is False
        win.state.bad_sections = ["sec1.png", "sec2.png"]
        assert win.bad_section_count == 2
        assert win.has_bad_sections is True

        # Export properties
        assert isinstance(win.export_format, str)
        assert isinstance(win.export_basename, str)
    finally:
        win.close()
        win.deleteLater()


def test_drop_area_toast_overlay_and_slice_view_ergonomics():
    """Assert DropArea, ToastOverlay, and SliceGraphicsView accessibility and ergonomics helpers."""
    from DeepSlice.gui.main_window import DropArea, ToastOverlay, SliceGraphicsView
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    # 1. DropArea
    drop = DropArea()
    try:
        assert drop.objectName() == "DropArea"
        assert drop.accessibleName() == "Image and Folder Ingestion Drop Area"
        assert len(drop.accessibleDescription()) > 0
        assert "Drag and Drop" in drop.title_text()
        assert "Supports JPG" in drop.subtitle_text()
        assert drop.title_label is not None
        assert drop.subtitle_label is not None
        assert drop.icon_label is not None
    finally:
        drop.deleteLater()

    # 2. ToastOverlay
    toast = ToastOverlay()
    try:
        assert toast.objectName() == "ToastOverlay"
        assert toast.accessibleName() == "Notification Toast"
        assert len(toast.accessibleDescription()) > 0
        assert toast.message_label.objectName() == "ToastMessageLabel"
        assert toast.message_label.accessibleName() == "Toast Message"
        assert toast.message_text() == ""
        toast.show_message("Predictions loaded successfully", timeout_ms=2000, level="success")
        assert toast.message_text() == "Predictions loaded successfully"
    finally:
        toast.deleteLater()

    # 3. SliceGraphicsView
    view = SliceGraphicsView()
    try:
        assert view.objectName() == "sliceGraphicsView"
        assert view.accessibleName() == "Slice Interactive Canvas"
        assert len(view.accessibleDescription()) > 0
        assert view.zoom_level() == 0
        assert view.is_loupe_enabled() is False
        assert view.has_sync_partner() is False
        assert view.is_mouse_tracking_active() is True

        view.set_loupe_enabled(True)
        assert view.is_loupe_enabled() is True
    finally:
        view.deleteLater()


def test_thumbnail_and_flag_list_widget_and_main_window_curation_ergonomics():
    """Assert ThumbnailListWidget and FlagListWidget accessibility and DeepSliceMainWindow curation triggers."""
    from PySide6.QtWidgets import QApplication, QListWidgetItem
    from PySide6.QtCore import Qt
    from DeepSlice.gui.main_window import DeepSliceMainWindow, ThumbnailListWidget, FlagListWidget

    app = QApplication.instance() or QApplication([])

    # 1. ThumbnailListWidget
    thumb = ThumbnailListWidget()
    try:
        assert thumb.objectName() == "thumbnailListWidget"
        assert thumb.accessibleName() == "Thumbnail Sections List"
        assert "thumbnail list" in thumb.accessibleDescription().lower()
        assert thumb.item_count() == 0
        assert thumb.has_items() is False
        assert thumb.selected_index() == -1
        assert thumb.selected_text() == ""

        thumb.addItem("slide_001.png")
        thumb.addItem("slide_002.png")
        assert thumb.item_count() == 2
        assert thumb.has_items() is True
        assert thumb.item_texts() == ["slide_001.png", "slide_002.png"]

        assert thumb.select_index(1) is True
        assert thumb.selected_index() == 1
        assert thumb.selected_text() == "slide_002.png"
        assert thumb.select_index(999) is False
    finally:
        thumb.deleteLater()

    # 2. FlagListWidget
    flags = FlagListWidget()
    try:
        assert flags.objectName() == "flagListWidget"
        assert flags.accessibleName() == "Flag Sections List"
        assert "checklist" in flags.accessibleDescription().lower()
        assert flags.item_count() == 0
        assert flags.has_items() is False
        assert flags.checked_count() == 0
        assert flags.unchecked_count() == 0

        it1 = QListWidgetItem("sec_01.png")
        it1.setFlags(it1.flags() | Qt.ItemIsUserCheckable)
        it1.setCheckState(Qt.Unchecked)
        flags.addItem(it1)

        it2 = QListWidgetItem("sec_02.png")
        it2.setFlags(it2.flags() | Qt.ItemIsUserCheckable)
        it2.setCheckState(Qt.Checked)
        flags.addItem(it2)

        assert flags.item_count() == 2
        assert flags.has_items() is True
        assert flags.checked_count() == 1
        assert flags.unchecked_count() == 1
        assert flags.is_item_checked(0) is False
        assert flags.is_item_checked(1) is True
        assert flags.checked_texts() == ["sec_02.png"]

        assert flags.set_item_checked(0, True) is True
        assert flags.is_item_checked(0) is True
        assert flags.checked_count() == 2

        assert flags.toggle_item(1) is True
        assert flags.is_item_checked(1) is False
        assert flags.checked_count() == 1

        assert flags.select_index(0) is True
        assert flags.selected_index() == 0
        assert flags.trigger_toggle_current() is True
        assert flags.is_item_checked(0) is False
    finally:
        flags.deleteLater()

    # 3. DeepSliceMainWindow curation queries and triggers
    win = DeepSliceMainWindow()
    try:
        assert win.thumbnail_count == 0
        assert win.curation_slice_count == 0
        assert win.curation_flagged_count == 0

        win.thumbnail_list.addItem("test_section.png")
        assert win.thumbnail_count == 1

        item = QListWidgetItem("test_slice.png")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        win.slice_flag_list.addItem(item)
        assert win.curation_slice_count == 1
        assert win.curation_flagged_count == 0

        # Triggers
        assert isinstance(win.trigger_curation_select_all(), bool)
        assert isinstance(win.trigger_curation_deselect_all(), bool)
        assert isinstance(win.trigger_apply_bad_section_flags(), bool)
        assert isinstance(win.trigger_apply_manual_order(), bool)
    finally:
        win.close()
        win.deleteLater()


def test_configuration_page_accessibility_properties_and_ergonomics():
    """Assert Configuration Page controls accessibility metadata, pointers, properties, and ergonomics helpers."""
    from DeepSlice.gui.main_window import DeepSliceMainWindow
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        # 1. Object names and accessible metadata
        controls = [
            (win.mouse_radio, "mouseRadio", "Mouse CCFv3 Atlas"),
            (win.rat_radio, "ratRadio", "Rat Waxholm Atlas"),
            (win.auto_thickness_checkbox, "autoThicknessCheck", "Auto-estimate Thickness"),
            (win.thickness_spin, "thicknessSpin", "Section Thickness SpinBox"),
            (win.suggest_thickness_button, "suggestThicknessBtn", "Suggest Thickness"),
            (win.direction_override_combo, "directionOverrideCombo", "Direction Override"),
            (win.direction_guide_button, "directionGuideBtn", "Direction Guide"),
            (win.ensemble_checkbox, "ensembleCheck", "Ensemble Prediction"),
            (win.ensemble_help_button, "ensembleHelpBtn", "Ensemble Help"),
            (win.secondary_model_checkbox, "secondaryModelCheck", "Secondary Model Only"),
            (win.legacy_from_config_checkbox, "legacyParserCheck", "Legacy Section Parser"),
            (win.outlier_sigma_spin, "outlierSigmaSpin", "Outlier Sigma Threshold"),
            (win.confidence_high_spin, "confidenceHighSpin", "High Confidence Threshold"),
            (win.confidence_medium_spin, "confidenceMediumSpin", "Medium Confidence Threshold"),
            (win.inference_batch_spin, "inferenceBatchSpin", "Inference Batch Size"),
            (win.quality_gate_checkbox, "qualityGateCheck", "Input Quality Gate"),
            (win.tissue_crop_checkbox, "tissueCropCheck", "Tissue ROI Crop"),
            (win.clahe_checkbox, "claheCheck", "CLAHE Contrast Normalization"),
            (win.stain_norm_combo, "stainNormCombo", "Stain Normalization Mode"),
            (win.preserve_color_checkbox, "preserveColorCheck", "Preserve Color Channels"),
            (win.percentile_norm_checkbox, "percentileNormCheck", "Percentile Intensity Normalization"),
            (win.bilateral_checkbox, "bilateralCheck", "Bilateral Denoise"),
            (win.gamma_spin, "gammaSpin", "Gamma Correction"),
            (win.min_resolution_spin, "minResolutionSpin", "Minimum Resolution in Pixels"),
            (win.blur_threshold_spin, "blurThresholdSpin", "Blur Variance Threshold"),
            (win.tta_checkbox, "ttaCheck", "Test-Time Augmentation"),
            (win.multiscale_checkbox, "multiscaleCheck", "Multi-Scale Inference"),
            (win.fast_fp16_checkbox, "fastFp16Check", "Fast Mode Float16"),
            (win.section_dropout_spin, "sectionDropoutSpin", "Section Dropout Passes"),
            (win.conf_weighted_ensemble_checkbox, "confWeightedEnsembleCheck", "Confidence Weighted Ensemble"),
            (win.training_group_mode_combo, "trainingGroupModeCombo", "Training Group Mode"),
            (win.training_seed_spin, "trainingSeedSpin", "Training Random Seed"),
            (win.training_train_fraction_spin, "trainingTrainFractionSpin", "Training Train Split Fraction"),
            (win.training_val_fraction_spin, "trainingValFractionSpin", "Training Validation Split Fraction"),
            (win.training_patience_spin, "trainingPatienceSpin", "Training Early Stopping Patience"),
            (win.training_lr_factor_spin, "trainingLrFactorSpin", "Training Learning Rate Decay Factor"),
            (win.training_min_lr_spin, "trainingMinLrSpin", "Training Minimum Learning Rate"),
            (win.training_mixed_precision_checkbox, "trainingMixedPrecisionCheck", "Training Mixed Precision"),
        ]

        for widget, obj_name, acc_name in controls:
            assert widget.objectName() == obj_name, f"Expected objectName {obj_name}, got {widget.objectName()}"
            assert widget.accessibleName() == acc_name, f"Expected accessibleName {acc_name}, got {widget.accessibleName()}"
            assert len(widget.toolTip()) > 0, f"Expected non-empty tooltip for {obj_name}"

        # 2. PointingHandCursor affordances on interactive controls
        cursor_controls = [
            win.mouse_radio,
            win.rat_radio,
            win.auto_thickness_checkbox,
            win.suggest_thickness_button,
            win.direction_guide_button,
            win.advanced_options_toggle,
            win.ensemble_checkbox,
            win.ensemble_help_button,
            win.secondary_model_checkbox,
            win.legacy_from_config_checkbox,
            win.quality_gate_checkbox,
            win.tissue_crop_checkbox,
            win.clahe_checkbox,
            win.preserve_color_checkbox,
            win.percentile_norm_checkbox,
            win.bilateral_checkbox,
            win.tta_checkbox,
            win.multiscale_checkbox,
            win.fast_fp16_checkbox,
            win.conf_weighted_ensemble_checkbox,
            win.preview_training_split_button,
            win.save_training_split_button,
            win.save_training_metadata_button,
        ]
        for ctrl in cursor_controls:
            assert ctrl.cursor().shape() == Qt.CursorShape.PointingHandCursor, f"Expected PointingHandCursor for {ctrl.objectName()}"

        # 3. MainWindow property aliases
        assert win.mouse_atlas_radio is win.mouse_radio
        assert win.rat_atlas_radio is win.rat_radio
        assert win.auto_thickness_check is win.auto_thickness_checkbox
        assert win.thickness_spin_box is win.thickness_spin
        assert win.suggest_thickness_btn is win.suggest_thickness_button
        assert win.direction_override_combobox is win.direction_override_combo
        assert win.direction_guide_btn is win.direction_guide_button
        assert win.advanced_options_toggle_btn is win.advanced_options_toggle
        assert win.ensemble_check is win.ensemble_checkbox
        assert win.ensemble_help_btn is win.ensemble_help_button
        assert win.secondary_model_check is win.secondary_model_checkbox
        assert win.legacy_parser_check is win.legacy_from_config_checkbox
        assert win.outlier_sigma_spin_box is win.outlier_sigma_spin
        assert win.confidence_high_spin_box is win.confidence_high_spin
        assert win.confidence_medium_spin_box is win.confidence_medium_spin
        assert win.inference_batch_spin_box is win.inference_batch_spin
        assert win.quality_gate_check is win.quality_gate_checkbox
        assert win.tissue_crop_check is win.tissue_crop_checkbox
        assert win.clahe_check is win.clahe_checkbox
        assert win.stain_norm_combobox is win.stain_norm_combo
        assert win.preserve_color_check is win.preserve_color_checkbox
        assert win.percentile_norm_check is win.percentile_norm_checkbox
        assert win.bilateral_check is win.bilateral_checkbox
        assert win.gamma_spin_box is win.gamma_spin
        assert win.min_resolution_spin_box is win.min_resolution_spin
        assert win.blur_threshold_spin_box is win.blur_threshold_spin
        assert win.tta_check is win.tta_checkbox
        assert win.multiscale_check is win.multiscale_checkbox
        assert win.fast_fp16_check is win.fast_fp16_checkbox
        assert win.section_dropout_spin_box is win.section_dropout_spin
        assert win.conf_weighted_ensemble_check is win.conf_weighted_ensemble_checkbox
        assert win.training_group_mode_combobox is win.training_group_mode_combo
        assert win.preview_training_split_btn is win.preview_training_split_button
        assert win.save_training_split_btn is win.save_training_split_button
        assert win.save_training_metadata_btn is win.save_training_metadata_button

        # 4. Ergonomic query and action helpers
        assert win.is_auto_thickness_enabled() is True
        win.set_auto_thickness_enabled(False)
        assert win.is_auto_thickness_enabled() is False
        win.set_auto_thickness_enabled(True)
        assert win.is_auto_thickness_enabled() is True

        win.set_section_thickness_um(40.5)
        assert win.section_thickness_um() == 40.5

        assert win.direction_override_text() in ("Auto", "rostro-caudal", "caudal-rostro")
        win.set_direction_override("caudal-rostro")
        assert win.direction_override_text() == "caudal-rostro"

        assert win.is_advanced_options_expanded() is False
        win.set_advanced_options_expanded(True)
        assert win.is_advanced_options_expanded() is True
        win.set_advanced_options_expanded(False)
        assert win.is_advanced_options_expanded() is False

        assert win.is_ensemble_enabled() is True
        win.set_ensemble_enabled(False)
        assert win.is_ensemble_enabled() is False
        win.set_ensemble_enabled(True)
        assert win.is_ensemble_enabled() is True

        win.set_quality_gate_enabled(True)
        assert win.is_quality_gate_enabled() is True

        win.set_tissue_crop_enabled(True)
        assert win.is_tissue_crop_enabled() is True

        win.set_clahe_enabled(True)
        assert win.is_clahe_enabled() is True

        # Triggers
        assert isinstance(win.trigger_suggest_thickness(), bool)
        assert isinstance(win.trigger_direction_guide(), bool)
        assert isinstance(win.trigger_ensemble_help(), bool)
        assert isinstance(win.trigger_preview_training_split(), bool)
    finally:
        win.close()
        win.deleteLater()


def test_export_page_accessibility_properties_and_ergonomics(qtbot, tmp_path):
    """Verify Export Page control object names, pointer affordances, public properties, and ergonomic helpers."""
    win = DeepSliceMainWindow()
    qtbot.addWidget(win)
    try:
        win.go_to_page(4)

        # Explicit object names
        assert win.output_dir_line_edit.objectName() == "outputDirEdit"
        assert win.browse_output_dir_btn.objectName() == "browseOutputDirBtn"
        assert win.output_basename_line_edit.objectName() == "outputBasenameEdit"
        assert win.output_format_combobox.objectName() == "outputFormatCombo"
        assert win.output_format_help_btn.objectName() == "outputFormatHelpBtn"
        assert win.export_size_estimate_lbl.objectName() == "exportSizeEstimateLabel"
        assert win.export_btn.objectName() == "exportBtn"
        assert win.open_export_dir_btn.objectName() == "openExportDirBtn"
        assert win.copy_export_path_btn.objectName() == "copyExportPathBtn"
        assert win.report_btn.objectName() == "reportBtn"
        assert win.preview_report_btn.objectName() == "previewReportBtn"
        assert win.pdf_content_group_box.objectName() == "pdfContentGroup"
        assert win.pdf_include_stats_check.objectName() == "pdfIncludeStatsCheck"
        assert win.pdf_include_plot_check.objectName() == "pdfIncludePlotCheck"
        assert win.pdf_include_images_check.objectName() == "pdfIncludeImagesCheck"
        assert win.pdf_include_angles_check.objectName() == "pdfIncludeAnglesCheck"
        assert win.quicknii_path_line_edit.objectName() == "quickniiPathEdit"
        assert win.quicknii_browse_btn.objectName() == "quickniiBrowseBtn"
        assert win.open_quicknii_btn.objectName() == "openQuickniiBtn"
        assert win.export_summary_label.objectName() == "exportSummaryLabel"
        assert win.export_deviation_label.objectName() == "exportDeviationLabel"

        # Cursor affordances
        assert win.browse_output_dir_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.output_format_help_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.export_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.open_export_dir_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.copy_export_path_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.report_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.preview_report_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.quicknii_browse_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.open_quicknii_btn.cursor().shape() == Qt.PointingHandCursor

        # Ergonomic queries and setters
        win.set_export_output_dir(str(tmp_path))
        assert win.export_output_dir() == str(tmp_path)

        win.set_export_base_filename("CohortStudy_Export")
        assert win.export_base_filename() == "CohortStudy_Export"

        win.set_export_format_name("Legacy XML")
        assert "Legacy XML" in win.export_format_name()

        win.set_pdf_stats_included(False)
        assert win.is_pdf_stats_included() is False
        win.set_pdf_stats_included(True)
        assert win.is_pdf_stats_included() is True

        win.set_pdf_plot_included(False)
        assert win.is_pdf_plot_included() is False

        win.set_pdf_images_included(True)
        assert win.is_pdf_images_included() is True

        win.set_pdf_angles_included(False)
        assert win.is_pdf_angles_included() is False

        assert "Processed" in win.export_summary_text()
        assert "angular deviation" in win.export_deviation_text()

        # Action triggers
        assert isinstance(win.trigger_export_format_help(), bool)
        assert isinstance(win.trigger_copy_export_path(), bool)
        assert isinstance(win.trigger_preview_report(), bool)
    finally:
        win.close()
        win.deleteLater()


def test_prediction_page_accessibility_properties_and_ergonomics(qtbot):
    """Verify accessibility metadata, cursor shapes, properties, and ergonomic helpers on Prediction Page."""
    win = DeepSliceMainWindow()
    qtbot.addWidget(win)
    win.show()
    try:
        # Navigate to Prediction Page (Step index 2)
        win.go_to_page(2)

        # Explicit object names and accessible names
        assert win.run_alignment_btn.objectName() == "runAlignmentBtn"
        assert win.run_alignment_btn.accessibleName() == "Run Alignment"
        assert win.cancel_alignment_btn.objectName() == "cancelAlignmentBtn"
        assert win.cancel_alignment_btn.accessibleName() == "Cancel Alignment"

        assert win.prediction_phase_lbl.objectName() == "predictionPhaseLabel"
        assert win.prediction_phase_lbl.accessibleName() == "Prediction Phase Status"
        assert win.prediction_progress_lbl.objectName() == "predictionProgressLabel"
        assert win.prediction_progress_lbl.accessibleName() == "Prediction Progress Status"
        assert win.prediction_elapsed_lbl.objectName() == "predictionElapsedLabel"
        assert win.prediction_elapsed_lbl.accessibleName() == "Elapsed Alignment Time"
        assert win.prediction_eta_lbl.objectName() == "predictionEtaLabel"
        assert win.prediction_eta_lbl.accessibleName() == "Estimated Time Remaining"

        assert win.prediction_progress_bar_widget.objectName() == "predictionProgressBar"
        assert win.predicted_thickness_lbl.objectName() == "predictedThicknessLabel"
        assert win.predicted_thickness_lbl.accessibleName() == "Estimated Section Thickness"
        assert win.accept_predicted_thickness_btn.objectName() == "acceptPredictedThicknessBtn"
        assert win.prediction_direction_lbl.objectName() == "predictionDirectionLabel"
        assert win.prediction_direction_lbl.accessibleName() == "Detected Indexing Direction"

        assert win.console_toggle_btn.objectName() == "consoleToggle"
        assert win.console_autoscroll_toggle_btn.objectName() == "consoleAutoscrollToggle"
        assert win.clear_console_btn.objectName() == "clearConsoleButton"
        assert win.copy_console_btn.objectName() == "copyConsoleButton"
        assert win.console_output_widget.objectName() == "consoleOutput"
        assert win.run_blocker_lbl.objectName() == "WarningText"

        assert win.prediction_slice_selector_combo.objectName() == "predictionSliceSelector"
        assert win.prediction_compare_check.objectName() == "predictionCompareCheck"
        assert win.prediction_atlas_info_lbl.objectName() == "predictionAtlasInfoLabel"
        assert win.preprocessing_preview_info_lbl.objectName() == "preprocessingPreviewInfoLabel"
        assert win.preprocessing_preview_lbl.objectName() == "preprocessingPreviewLabel"
        assert win.prediction_slice_viewer.objectName() == "predictionSliceViewer"
        assert win.prediction_atlas_viewer_widget.objectName() == "predictionAtlasViewer"

        # Cursor affordances
        assert win.run_alignment_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.cancel_alignment_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.accept_predicted_thickness_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.console_toggle_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.console_autoscroll_toggle_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.clear_console_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.copy_console_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.prediction_slice_selector_combo.cursor().shape() == Qt.PointingHandCursor
        assert win.prediction_compare_check.cursor().shape() == Qt.PointingHandCursor

        # State readouts
        assert "idle" in win.prediction_phase_text().lower()
        assert "0" in win.prediction_progress_text()
        assert "00:00" in win.prediction_elapsed_text()
        assert "--:--" in win.prediction_eta_text()
        assert win.prediction_progress_value() == 0
        assert "-" in win.predicted_thickness_text()
        assert "-" in win.prediction_direction_text()
        assert isinstance(win.is_run_alignment_enabled(), bool)
        assert win.is_cancel_alignment_enabled() is False
        assert win.is_accept_predicted_thickness_enabled() is False

        # Console controls and ergonomics
        assert win.is_console_visible() is False
        win.set_console_visible(True)
        assert win.is_console_visible() is True
        win.set_console_visible(False)
        assert win.is_console_visible() is False

        assert win.is_console_autoscroll_enabled() is True
        win.set_console_autoscroll_enabled(False)
        assert win.is_console_autoscroll_enabled() is False
        win.set_console_autoscroll_enabled(True)
        assert win.is_console_autoscroll_enabled() is True

        assert isinstance(win.console_text(), str)
        assert win.trigger_clear_console() is True
        assert win.trigger_copy_console() is True

        # Preview controls
        assert win.prediction_slice_count() >= 0
        assert isinstance(win.prediction_slice_texts(), list)
        assert win.is_atlas_comparison_checked() is True
        assert isinstance(win.trigger_accept_predicted_thickness(), bool)
    finally:
        win.close()
        win.deleteLater()

def test_ingestion_page_accessibility_properties_and_ergonomics(qtbot):
    """Verify accessibility metadata, cursor shapes, properties, and ergonomic helpers on Ingestion Page."""
    win = DeepSliceMainWindow()
    qtbot.addWidget(win)
    win.show()
    try:
        win.go_to_page(0)
        assert win.current_page_index == 0

        # Object names and accessible names
        assert win.drop_area_widget.objectName() == "ingestionDropArea"
        assert win.drop_area_widget.accessibleName() == "Image Drop Area"
        assert "Drag and drop" in win.drop_area_widget.accessibleDescription()

        assert win.add_folder_btn.objectName() == "addFolderBtn"
        assert win.add_folder_btn.accessibleName() == "Add Folder"
        assert win.add_folder_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.add_files_btn.objectName() == "addFilesBtn"
        assert win.add_files_btn.accessibleName() == "Add Files"
        assert win.add_files_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.clear_images_btn.objectName() == "clearImagesBtn"
        assert win.clear_images_btn.accessibleName() == "Clear All Images"
        assert win.clear_images_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.enable_section_numbers_check.objectName() == "enableSectionNumbersCheck"
        assert win.enable_section_numbers_check.accessibleName() == "Detect Section Numbers"
        assert win.enable_section_numbers_check.cursor().shape() == Qt.PointingHandCursor

        assert win.naming_helper_btn.objectName() == "namingHelperBtn"
        assert win.naming_helper_btn.accessibleName() == "Naming Helper"
        assert win.naming_helper_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.legacy_parsing_check.objectName() == "legacyParsingCheck"
        assert win.legacy_parsing_check.accessibleName() == "Legacy Parser Fallback"
        assert win.legacy_parsing_check.cursor().shape() == Qt.PointingHandCursor

        assert win.orientation_combobox.objectName() == "orientationCombo"
        assert win.orientation_combobox.accessibleName() == "Anatomical Orientation"
        assert win.orientation_combobox.cursor().shape() == Qt.PointingHandCursor

        assert win.orientation_guide_btn.objectName() == "orientationGuideBtn"
        assert win.orientation_guide_btn.accessibleName() == "Orientation Guide"
        assert win.orientation_guide_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.slice_count_lbl.objectName() == "sliceCountLabel"
        assert win.file_size_summary_lbl.objectName() == "fileSizeSummaryLabel"
        assert win.ingestion_summary_banner_lbl.objectName() == "SummaryBanner"
        assert win.ingestion_warning_lbl.objectName() == "WarningText"

        assert win.index_table_widget.objectName() == "indexTable"
        assert win.index_table_widget.accessibleName() == "Section Indices Table"

        assert win.thumbnail_sort_combobox.objectName() == "thumbnailSortCombo"
        assert win.thumbnail_sort_combobox.cursor().shape() == Qt.PointingHandCursor

        assert win.thumbnail_filter_input.objectName() == "thumbnailFilterEdit"
        assert win.thumbnail_list_widget.objectName() == "ingestionThumbnailList"
        assert win.thumbnail_progress_bar.objectName() == "thumbnailProgress"
        assert win.ingestion_preview_view.objectName() == "ingestionPreview"

        # Ergonomic state queries and setters
        assert "Slices: 0" in win.slice_count_text()
        assert "0 files loaded" in win.file_size_summary_text()
        assert "No files loaded" in win.ingestion_summary_banner_text()
        assert win.index_table_row_count() == 0

        assert win.is_section_number_detection_enabled() is True
        win.set_section_number_detection_enabled(False)
        assert win.is_section_number_detection_enabled() is False
        win.set_section_number_detection_enabled(True)
        assert win.is_section_number_detection_enabled() is True

        assert win.is_legacy_parsing_enabled() is False
        win.set_legacy_parsing_enabled(True)
        assert win.is_legacy_parsing_enabled() is True
        win.set_legacy_parsing_enabled(False)
        assert win.is_legacy_parsing_enabled() is False

        assert "Coronal" in win.cutting_plane_orientation()
        win.set_cutting_plane_orientation("Sagittal")
        assert "Sagittal" in win.cutting_plane_orientation()
        win.set_cutting_plane_orientation("Coronal")
        assert "Coronal" in win.cutting_plane_orientation()

        assert "Filename" in win.thumbnail_sort_order()
        win.set_thumbnail_sort_order("Detected Index")
        assert "Detected Index" in win.thumbnail_sort_order()
        win.set_thumbnail_sort_order("Filename")

        win.set_thumbnail_filter_query("brain_s001")
        assert win.thumbnail_filter_query() == "brain_s001"
        win.clear_thumbnail_filter()
        assert win.thumbnail_filter_query() == ""

        # Programmatic dialog triggers (headless safe)
        helper_dlg = win.trigger_naming_helper()
        assert helper_dlg is not None
        guide_dlg = win.trigger_orientation_guide()
        assert guide_dlg is not None

        win.trigger_clear_images()
    finally:
        win.close()
        win.deleteLater()



def test_curation_page_accessibility_properties_and_ergonomics(qtbot):
    """Verify accessibility metadata, cursor shapes, properties, and ergonomic helpers on Curation Page."""
    win = DeepSliceMainWindow()
    qtbot.addWidget(win)
    win.show()
    try:
        win.go_to_page(3)
        assert win.current_page_index == 3

        # Object names and accessible names on left panel
        assert win.curation_prev_btn.objectName() == "curationPrevBtn"
        assert win.curation_prev_btn.accessibleName() == "Previous Section"
        assert "previous histological section" in win.curation_prev_btn.accessibleDescription()
        assert win.curation_prev_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.curation_next_btn.objectName() == "curationNextBtn"
        assert win.curation_next_btn.accessibleName() == "Next Section"
        assert "next histological section" in win.curation_next_btn.accessibleDescription()
        assert win.curation_next_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.curation_select_all_button.objectName() == "curationSelectAllBtn"
        assert win.curation_select_all_button.accessibleName() == "Select All Bad Section Flags"
        assert win.curation_select_all_button.cursor().shape() == Qt.PointingHandCursor

        assert win.curation_deselect_all_button.objectName() == "curationDeselectAllBtn"
        assert win.curation_deselect_all_button.accessibleName() == "Deselect All Bad Section Flags"
        assert win.curation_deselect_all_button.cursor().shape() == Qt.PointingHandCursor

        assert win.confidence_filter_combobox.objectName() == "confidenceFilterCombo"
        assert win.confidence_filter_combobox.accessibleName() == "Filter Slices by Confidence"
        assert win.confidence_filter_combobox.cursor().shape() == Qt.PointingHandCursor

        assert win.slice_flag_list_widget.objectName() == "sliceFlagList"
        assert win.slice_flag_list_widget.accessibleName() == "Curated Slices List"

        assert win.apply_bad_sections_btn.objectName() == "applyBadSectionsBtn"
        assert win.apply_bad_sections_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.apply_manual_order_btn.objectName() == "applyManualOrderBtn"
        assert win.apply_manual_order_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.move_slice_up_btn.objectName() == "moveSliceUpBtn"
        assert win.move_slice_up_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.move_slice_down_btn.objectName() == "moveSliceDownBtn"
        assert win.move_slice_down_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.detect_outliers_btn.objectName() == "detectOutliersBtn"
        assert win.detect_outliers_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.reset_flags_btn.objectName() == "resetFlagsBtn"
        assert win.reset_flags_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.auto_flag_low_conf_btn.objectName() == "autoFlagLowConfBtn"
        assert win.auto_flag_low_conf_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.interpolate_bad_depth_btn.objectName() == "interpolateBadDepthBtn"
        assert win.interpolate_bad_depth_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.toggle_current_flag_btn.objectName() == "toggleCurrentFlagBtn"
        assert win.toggle_current_flag_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.slice_note_line_edit.objectName() == "sliceNoteEdit"
        assert win.save_slice_note_btn.objectName() == "saveSliceNoteBtn"
        assert win.save_slice_note_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.curation_queue_lbl.objectName() == "curationQueueLabel"

        assert win.controls_group_box.objectName() == "controlsGroupBox"
        assert win.normalize_angles_btn.objectName() == "normalizeAnglesBtn"
        assert win.normalize_angles_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.enforce_order_btn.objectName() == "enforceOrderBtn"
        assert win.enforce_order_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.enforce_spacing_btn.objectName() == "enforceSpacingBtn"
        assert win.enforce_spacing_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.ml_angle_spin_box.objectName() == "mlSpin"
        assert win.dv_angle_spin_box.objectName() == "dvSpin"
        assert win.apply_manual_angles_btn.objectName() == "applyManualAnglesBtn"
        assert win.apply_manual_angles_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.undo_curation_btn.objectName() == "undoBtn"
        assert win.undo_curation_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.redo_curation_btn.objectName() == "redoBtn"
        assert win.redo_curation_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.anchor_group_box.objectName() == "anchorGroupBox"
        assert win.anchor_summary_lbl.objectName() == "anchorSummaryLabel"
        assert win.anchor_depth_spin_box.objectName() == "anchorDepthSpin"
        assert win.set_anchor_btn.objectName() == "setAnchorBtn"
        assert win.set_anchor_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.remove_anchor_btn.objectName() == "removeAnchorBtn"
        assert win.remove_anchor_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.apply_anchor_interpolation_btn.objectName() == "applyAnchorInterpolationBtn"
        assert win.apply_anchor_interpolation_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.clear_anchor_btn.objectName() == "clearAnchorBtn"
        assert win.clear_anchor_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.anchor_list_widget.objectName() == "anchorList"

        # Right panel controls
        assert win.linearity_figure_canvas.objectName() == "linearityCanvas"
        assert win.zoom_fit_plot_btn.objectName() == "zoomFitBtn"
        assert win.zoom_fit_plot_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.zoom_in_plot_btn.objectName() == "zoomInBtn"
        assert win.zoom_in_plot_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.zoom_out_plot_btn.objectName() == "zoomOutBtn"
        assert win.zoom_out_plot_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.hist_figure_canvas.objectName() == "histCanvas"

        assert win.enable_atlas_preview_check.objectName() == "enableAtlasPreviewCheck"
        assert win.enable_atlas_preview_check.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_volume_combobox.objectName() == "atlasVolumeCombo"
        assert win.enable_blend_overlay_check.objectName() == "enableBlendOverlayCheck"
        assert win.enable_blend_overlay_check.cursor().shape() == Qt.PointingHandCursor
        assert win.blend_opacity_slider.objectName() == "blendSlider"
        assert win.blend_percent_lbl.objectName() == "blendPercentLabel"
        assert win.atlas_slice_info_lbl.objectName() == "atlasSliceInfoLabel"
        assert win.before_after_check.objectName() == "beforeAfterToggle"
        assert win.before_after_check.cursor().shape() == Qt.PointingHandCursor
        assert win.loupe_check.objectName() == "loupeToggle"
        assert win.loupe_check.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_coords_lbl.objectName() == "atlasCoordsLabel"

        assert win.atlas_flip_x_check.objectName() == "atlasFlipXCheck"
        assert win.atlas_flip_x_check.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_flip_y_check.objectName() == "atlasFlipYCheck"
        assert win.atlas_flip_y_check.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_rotate_combobox.objectName() == "atlasRotateCombo"
        assert win.atlas_rotate_combobox.cursor().shape() == Qt.PointingHandCursor
        assert win.atlas_scale_slider_widget.objectName() == "atlasScaleSlider"
        assert win.atlas_scale_lbl.objectName() == "atlasScaleLabel"
        assert win.atlas_offset_x_slider_widget.objectName() == "atlasOffsetXSlider"
        assert win.atlas_offset_x_lbl.objectName() == "atlasOffsetXLabel"
        assert win.atlas_offset_y_slider_widget.objectName() == "atlasOffsetYSlider"
        assert win.atlas_offset_y_lbl.objectName() == "atlasOffsetYLabel"

        assert win.confidence_panel_toggle_btn.objectName() == "confidencePanelToggle"
        assert win.confidence_panel_toggle_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.confidence_panel_frame.objectName() == "confidencePanel"
        assert win.confidence_overlay_slice_viewer.objectName() == "confidenceOverlayViewer"
        assert win.curation_slice_viewer.objectName() == "curationViewer"
        assert win.atlas_slice_viewer.objectName() == "atlasViewer"

        # Ergonomic state queries and setters
        assert "All Confidences" in win.curation_confidence_filter()
        win.set_curation_confidence_filter("High Only")
        assert "High Only" in win.curation_confidence_filter()
        win.set_curation_confidence_filter("All Confidences")

        win.set_slice_note_text("Lesion area noted")
        assert win.slice_note_text() == "Lesion area noted"

        win.set_manual_ml_angle(1.5)
        assert win.manual_ml_angle() == 1.5

        win.set_manual_dv_angle(-2.0)
        assert win.manual_dv_angle() == -2.0

        win.set_anchor_depth(120.5)
        assert win.anchor_depth() == 120.5

        assert win.anchor_count() == 0

        assert win.is_atlas_preview_enabled() is False
        win.set_atlas_preview_enabled(True)
        assert win.is_atlas_preview_enabled() is True
        win.set_atlas_preview_enabled(False)

        assert win.is_blend_overlay_enabled() is False
        win.set_blend_overlay_enabled(True)
        assert win.is_blend_overlay_enabled() is True
        win.set_blend_overlay_enabled(False)

        win.set_blend_opacity(50)
        assert win.blend_opacity() == 50

        assert win.is_before_after_overlay_enabled() is False
        win.set_before_after_overlay_enabled(True)
        assert win.is_before_after_overlay_enabled() is True
        win.set_before_after_overlay_enabled(False)

        assert win.is_loupe_enabled() is False
        win.set_loupe_enabled(True)
        assert win.is_loupe_enabled() is True
        win.set_loupe_enabled(False)

        assert win.is_atlas_flip_x_enabled() is False
        win.set_atlas_flip_x_enabled(True)
        assert win.is_atlas_flip_x_enabled() is True
        win.set_atlas_flip_x_enabled(False)

        assert win.is_atlas_flip_y_enabled() is False
        win.set_atlas_flip_y_enabled(True)
        assert win.is_atlas_flip_y_enabled() is True
        win.set_atlas_flip_y_enabled(False)

        win.set_atlas_scale_percent(120)
        assert win.atlas_scale_percent() == 120

        win.set_atlas_offset_x(10)
        assert win.atlas_offset_x() == 10

        win.set_atlas_offset_y(-5)
        assert win.atlas_offset_y() == -5

        assert win.is_confidence_panel_expanded() is False
        win.set_confidence_panel_expanded(True)
        assert win.is_confidence_panel_expanded() is True
        win.set_confidence_panel_expanded(False)

        # Triggers
        assert isinstance(win.trigger_curation_prev_slice(), bool)
        assert isinstance(win.trigger_curation_next_slice(), bool)
        assert isinstance(win.trigger_move_slice_up(), bool)
        assert isinstance(win.trigger_move_slice_down(), bool)
        assert isinstance(win.trigger_detect_outliers(), bool)
        assert isinstance(win.trigger_reset_flags(), bool)
        assert isinstance(win.trigger_auto_flag_low_conf(), bool)
        assert isinstance(win.trigger_interpolate_bad_depth(), bool)
        assert isinstance(win.trigger_toggle_current_flag(), bool)
        assert isinstance(win.trigger_save_slice_note(), bool)
        assert isinstance(win.trigger_normalize_angles(), bool)
        assert isinstance(win.trigger_enforce_order(), bool)
        assert isinstance(win.trigger_enforce_spacing(), bool)
        assert isinstance(win.trigger_apply_manual_angles(), bool)
        assert isinstance(win.trigger_undo(), bool)
        assert isinstance(win.trigger_redo(), bool)
        assert isinstance(win.trigger_set_anchor(), bool)
        assert isinstance(win.trigger_remove_anchor(), bool)
        assert isinstance(win.trigger_apply_anchor_interpolation(), bool)
        assert isinstance(win.trigger_clear_anchors(), bool)
        assert isinstance(win.trigger_plot_zoom_fit(), bool)
        assert isinstance(win.trigger_plot_zoom_in(), bool)
        assert isinstance(win.trigger_plot_zoom_out(), bool)
    finally:
        win.close()
        win.deleteLater()


def test_top_bar_sidebar_guide_banner_and_status_bar_accessibility_and_ergonomics(qtbot):
    win = DeepSliceMainWindow()
    qtbot.addWidget(win)
    win.show()
    try:
        # TopBar metadata and properties
        assert win.top_bar_widget.objectName() == "TopBar"
        assert win.project_title_label.objectName() == "ProjectLabel"
        assert win.session_status_lbl.objectName() == "SessionLabel"
        assert win.hardware_mode_lbl.objectName() == "HardwareLabel"

        assert win.hardware_health_btn.objectName() == "hardwareHealthBtn"
        assert win.hardware_health_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.theme_toggle_btn.objectName() == "themeToggleBtn"
        assert win.theme_toggle_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.new_session_btn.objectName() == "newSessionBtn"
        assert win.new_session_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.save_session_btn.objectName() == "saveSessionBtn"
        assert win.save_session_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.load_session_btn.objectName() == "loadSessionBtn"
        assert win.load_session_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.session_io_spinner_widget.objectName() == "sessionIoSpinner"

        assert win.shortcut_help_btn.objectName() == "shortcutHelpBtn"
        assert win.shortcut_help_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.preferences_btn.objectName() == "preferencesBtn"
        assert win.preferences_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.about_btn.objectName() == "aboutBtn"
        assert win.about_btn.cursor().shape() == Qt.PointingHandCursor

        assert win.error_menu_btn.objectName() == "ErrorMenuButton"
        assert win.error_menu_btn.cursor().shape() == Qt.PointingHandCursor

        # Sidebar metadata and properties
        assert win.sidebar_container_widget.objectName() == "sidebarContainer"
        assert win.sidebar_container_widget.accessibleName() == "Step Navigation Sidebar"
        assert win.completion_lbl.objectName() == "CompletionLabel"
        assert win.collapse_sidebar_btn.objectName() == "CollapseButton"
        assert win.collapse_sidebar_btn.cursor().shape() == Qt.PointingHandCursor
        assert win.step_list_widget.objectName() == "StepNavigator"

        # GuideBanner metadata and properties
        assert win.guide_banner_frame.objectName() == "GuideBanner"
        assert win.guide_title_lbl.objectName() == "GuideTitle"
        assert win.guide_text_lbl.objectName() == "GuideText"
        assert win.guide_next_btn.objectName() == "GuideNextButton"
        assert win.guide_next_btn.cursor().shape() == Qt.PointingHandCursor

        # StatusBar metadata and properties
        assert win.status_bar_widget.objectName() == "mainStatusBar"
        assert win.status_bar_widget.accessibleName() == "Main Status Bar"
        assert win.global_progress_bar.objectName() == "globalProgressBar"
        assert win.global_progress_bar.accessibleName() == "Global Task Progress"

        # Ergonomic queries and setters
        assert "DeepSlice" in win.project_title_text()
        assert "Session:" in win.session_status_text()
        assert "Runs on:" in win.hardware_mode_text()

        assert win.is_sidebar_collapsed() is False
        win.set_sidebar_collapsed(True)
        assert win.is_sidebar_collapsed() is True
        win.set_sidebar_collapsed(False)
        assert win.is_sidebar_collapsed() is False

        assert "complete" in win.completion_percentage_text().lower()
        assert isinstance(win.guide_title_text(), str)
        assert isinstance(win.guide_body_text(), str)
        assert isinstance(win.is_guide_next_visible(), bool)
        assert win.status_bar_message() == "Ready"
        assert win.is_global_progress_visible() is False

        # Programmatic triggers
        health_dialog = win.trigger_hardware_health()
        assert health_dialog is not None

        shortcuts_dialog = win.trigger_shortcuts_help()
        assert shortcuts_dialog is not None

        about_dialog = win.trigger_about_dialog()
        assert about_dialog is not None

        pref_dialog = win.trigger_open_preferences()
        assert pref_dialog is not None
        pref_dialog.close()

        win.trigger_toggle_sidebar()
        assert win.is_sidebar_collapsed() is True
        win.trigger_toggle_sidebar()
        assert win.is_sidebar_collapsed() is False

        # Guide next trigger (locked when 0 images, advances when images present)
        assert win.trigger_guide_next() is False
        assert win.step_list_widget.currentRow() == 0

        win.state.image_paths = ["dummy.png"]
        assert win.trigger_guide_next() is True
        assert win.step_list_widget.currentRow() == 1
    finally:
        win.close()
        win.deleteLater()

