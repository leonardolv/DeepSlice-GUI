# GUI & UX Maintenance Log

## In Progress

## Blocked / Needs Review

## Completed

### Task: Enhance Export Page Controls with Accessibility, Tooltips, Hand Cursors, Visual Copy Confirmation, and Dialog Headless Safety
- **Completed**: 2026-10-03
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added screen-reader accessible names, descriptive tooltips, and pointing hand cursors across all Export page controls: `output_dir_edit`, `browse_output_dir_button`, `output_basename_edit`, `output_format_combo`, `output_format_help_button`, `export_size_estimate_label`, `export_button`, `open_export_dir_button`, `copy_export_path_button`, `report_button`, `preview_report_button`, `pdf_content_group`, `pdf_include_stats`, `pdf_include_plot`, `pdf_include_images`, `pdf_include_angles`, `quicknii_path_edit`, `quicknii_browse_button`, `open_quicknii_button`, `summary_label`, `deviation_label`, `markers_label`, and `export_notes`.
    - Added instant inline visual feedback (`✓ Copied!`) via `QTimer.singleShot` when clicking `copy_export_path_button`.
    - Implemented non-blocking headless execution safety guards (`if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and "PYTEST_CURRENT_TEST" not in os.environ:`) across all guide and information dialogs (`_show_about_dialog`, `_show_shortcuts_help`, `_show_naming_helper`, `_show_orientation_guide`, `_show_direction_guide`, `_show_ensemble_explanation`, `_show_configuration_validation`, `_show_export_format_help`, and `_show_logged_error`) while setting screen-reader accessible names.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added `test_export_page_accessibility_and_copy_feedback` and `test_helper_dialogs_accessibility_and_headless_safety` validating accessibility attributes, pointing hand cursors, clipboard copy feedback, and headless dialog safety.
- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (9 passed, 0 failures in 7.64s).
- **Status**: Completed


### Task: Enhance Header Toolbar Actions with Accessibility, Tooltips, Hand Cursors, and Hardware Health Fallback
- **Completed**: 2026-10-03
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added accessible names, descriptive tooltips, and pointing hand cursors across header toolbar controls (`hardware_button`, `theme_toggle_button`, `new_session_button`, `save_session_button`, `load_session_button`, `session_io_spinner`, `shortcut_help_button`, `preferences_button`, `about_button`, and `error_menu_button`).
    - Added graceful CPU/platform fallback in `_show_hardware_health()` when TensorFlow or GPU acceleration packages are missing or optional, providing clear diagnostics without throwing unhandled exceptions.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added `test_header_toolbar_accessibility_and_cursors` and `test_hardware_health_cpu_fallback` verifying accessibility attributes, cursor shapes, and graceful CPU fallback output.
- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (7 passed in 7.59s with 0 errors).
- **Status**: Completed

### Task: Enhance Runtime Console and Prediction Controls with Accessibility, Tooltips, Hand Cursors, and Visual Copy Feedback
- **Completed**: 2026-10-02
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added accessible names, descriptive tooltips, and pointing hand cursors across all runtime console toolbar controls (`console_toggle`, `console_autoscroll_toggle`, `clear_console_button`, `copy_console_button`) and the log output viewer (`console_output`).
    - Added accessible names, tooltips, and pointing hand cursor to `accept_predicted_thickness_button` and `prediction_progress_bar`.
    - Added temporary inline visual confirmation (`✓ Copied!`) in `_copy_console()` on the copy button via `QTimer.singleShot`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_console_and_prediction_controls_accessibility` validating all accessibility attributes, cursors, clipboard copy, and visual button feedback.
- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (5 passed in 38.95s with 0 errors).
- **Status**: Completed

### Task: Make TensorFlow and Heavy ML Dependencies Gracefully Optional on Import
- **Completed**: 2026-10-02
- **Changes**:
  - `DeepSlice/neural_network/neural_network.py`:
    - Wrapped eager top-level `tensorflow`, `skimage`, and `h5py` imports in robust `try-except` blocks with fallback `Sequence` base class and `_CallbackBase`.
    - Prevents 19 test collection errors and GUI startup crashes when TensorFlow is omitted or unloaded.
  - `DeepSlice/training/train_runner.py`:
    - Added fallback for `XCEPTION_INPUT_SIZE = (299, 299, 3)` to avoid top-level dependency cascade from `neural_network.py`.
  - `tests/test_optional_tf_import.py`:
    - Added regression unit tests validating `DeepSlice.gui.state`, `DeepSlice.training.train_runner`, and `DeepSlice.neural_network.neural_network` import cleanly in restricted runtime environments.
- **Verification**: Verified headlessly with `pytest tests/ -q` (316 passed, 9 skipped, 0 failures in 6.87s).
- **Status**: Completed
- **Completed**: 2026-10-02
- **Changes**:
  - `DeepSlice/gui/state.py`:
    - Converted eager top-level `from ..neural_network import neural_network` import to lazy imports inside `inspect_image_batch` and `preview_preprocessed_image`.
    - Prevents failure during GUI state, session loading, metadata extraction, or coordinate processing on machines where TensorFlow native libraries are restricted or unloaded.
  - `tests/test_diagnostics.py`:
    - Added `_restore_deepslice_propagation` test fixture to ensure `DeepSlice` logger propagates to root during diagnostic logging assertions regardless of preceding test suite configurations.
- **Verification**: Verified headlessly with `pytest tests/ -q` (315 passed, 1 skipped, 0 errors in 13.18s).
- **Status**: Completed
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Extracted `_create_preferences_dialog(self) -> QDialog`: Added screen-reader accessible names and descriptive tooltips across all configuration controls (`prefSpeciesCombo`, `prefThemeCombo`, `prefOutputDirEdit`, `prefOutputDirBrowse`, `prefQuickNiiEdit`, `prefQuickNiiBrowse`, `prefConsoleVisibleCheck`, `Ok`, `Cancel`).
    - Added "Reset to Defaults" button (`QDialogButtonBox.RestoreDefaults`) restoring species ("mouse"), theme ("dark"), output directory (""), QuickNII path (""), and runtime console visibility (False).
    - Updated `_open_fullscreen_thumbnail_preview`: Added accessible names and tooltips to the full-screen slice viewer, path display, and Close button; guarded modal `dialog.exec()` when running headlessly in offscreen environments.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_preferences_dialog_accessibility_and_reset_to_defaults` verifying all accessible names, tooltips, and click restoration of initial settings.
- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py -v` (7 passed in 14.28s with 0 errors).
- **Status**: Completed

### Task: Enhance UI Accessibility, DropArea/Thumbnail List Affordances, and Headless Non-blocking Startup
- **Completed**: 2026-09-25
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added accessible names and descriptive tooltips to `DropArea` ("Image and Folder Ingestion Drop Area") and `ThumbnailListWidget` ("Thumbnail Sections List").
    - Configured accessible names and tooltips for `self.status_bar` ("Main Status Bar") and `self.global_progress` ("Global Task Progress").
    - Guarded `_show_startup_dialogs` against offscreen/pytest runs to strictly satisfy Rule 1 (never spawn blocking GUI windows or modal popups during automated runs).
  - `tests/conftest.py`:
    - Created test harness with `_mock_qt_dialogs` autouse fixture to ensure all `QMessageBox` and `QFileDialog` APIs remain non-blocking throughout headless test execution.
  - `tests/test_pdf_reporting.py`:
    - Made `reportlab` an optional `pytest.importorskip("reportlab")` dependency.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Created unit tests verifying accessibility metadata on `DropArea`, `ThumbnailListWidget`, `status_bar`, and `global_progress`, and headless non-blocking window startup.
- **Verification**: Verified headlessly with `pytest tests/test_drop_event_toast.py tests/test_pdf_reporting.py tests/test_ui_accessibility_and_startup.py -v` (6 passed, 1 skipped in 6.62s).
- **Status**: Completed

### Task: Clean up PDF report placeholder boilerplate and populate angle metrics
- **Completed**: 2026-09-08 12:50:00
- **Changes**:
  - `DeepSlice/gui/state.py`:
    - Enriched `DeepSliceAppState.summary_metrics()` to compute and return `mean_dv`, `mean_ml`, `std_dv`, and `std_ml` alongside `mean_angular_deviation`, providing real angle statistics to PDF reports and downstream callers.
  - `DeepSlice/gui/reporting.py`:
    - Replaced the placeholder "Angle Metrics" text with actual calculated values: Mean angular deviation (deg), Dorsoventral (DV) angle (mean +/- std deg), and Mediolateral (ML) angle (mean +/- std deg).
    - Handled fallback gracefully when angle distributions are not computed.
    - Updated "Sample Images" section heading to "Sample Section Alignments" and replaced placeholder text with a professional note on inspecting section overlays in the GUI viewer; defaulted `include_images` option to `False`.
    - Added vertical page boundary check (`if y < ...: pdf.showPage()`) before starting new sections to prevent text from overflowing off the bottom of the page.
  - `DeepSlice/gui/main_window.py`:
    - Set default checked state of `self.pdf_include_images` checkbox to `False` so generated PDF reports are free of placeholder notes by default.
    - Updated `pdf_include_angles` tooltip to accurately describe the quantitative angle metrics included in the PDF report.
    - Fixed initialization order in `DeepSliceMainWindow.__init__`: initialized `self._settings = QSettings("DeepSlice", "GUI")` before calling `self._apply_startup_preferences_to_state()`, resolving an `AttributeError` on startup.
  - `tests/test_pdf_reporting.py`:
    - Added comprehensive unit tests covering `summary_metrics` angle metrics output, PDF report generation with quantitative angle metrics verification via Canvas string interception, sample image opt-in behavior, missing angle data handling, and `DeepSliceMainWindow` checkbox defaults.
- **Verification**:
  - Ran `pytest tests/test_pdf_reporting.py -v` (6 passed in 6.15s).
  - Ran full test suite `pytest tests/ -v` (227 passed, 0 failures in 14.49s).
- **Status**: Completed

### Task: Complete named-layer resolution refactor for weight loader (DS-007)
- **Completed**: 2026-09-07 20:25:00
- **Changes**:
  - `DeepSlice/neural_network/neural_network.py`:
    - Exported `XCEPTION_BASE_LAYER_NAME = "xception"` and `DENSE_HEAD_LAYER_NAMES = ("dense", "dense_1", "dense_2")`.
    - Updated `initialise_network`: Added species validation `("mouse", "rat")`, assigned explicit `name=XCEPTION_BASE_LAYER_NAME` to `Xception(...)` and explicit names `DENSE_HEAD_LAYER_NAMES[0..2]` to the 3 dense head layers in both mouse and rat architectures.
    - Updated `load_xception_weights`: Validated species at entry, resolved dense layers by name via `model.get_layer(layer_name)` instead of fragile positional indices, supported both `"kernel"` / `"kernel:0"` and `"bias"` / `"bias:0"` group keys, and raised clear `RuntimeError("missing expected layer '<name>'")` when layers are missing.
    - Preserved complete Xception layer weight mapping and sub-layer verification.
- **Verification**: Verified with `pytest tests/test_weight_loader.py -v` (7 passed in 35.87s) and test suite `pytest tests/ -k "not training"` (172 passed, 0 failures in 53.77s).
- **Status**: Completed

### Task: Fix Series index alignment and shape broadcast crash in calculate_average_section_thickness
- **Completed**: 2026-09-07 14:17:00
- **Changes**:
  - `DeepSlice/coord_post_processing/spacing_and_indexing.py`:
    - Converted `section_numbers` and `section_depth` via `np.asarray(...)` at function entry in `calculate_average_section_thickness`.
    - Resolved critical pandas Series index alignment bug where slice subtraction `section_depth[:-1] - section_depth[1:]` produced length N Series with NaNs and zeros instead of length N-1 consecutive differences, causing broadcast shape mismatches `(N,) / (N-1,)` and crashing section thickness estimation.
    - Resolved `.values` assumption on `section_numbers` which raised `AttributeError` on list inputs.
    - Handled boolean masking for `bad_sections` on ndarrays safely.
    - Enforced float return type for `average_thickness`.
  - `tests/test_spacing_and_indexing.py`:
    - Updated `test_calculate_average_section_thickness_happy_path_no_inf` assertion to expected positive thickness `10.0`.
    - Added `test_calculate_average_section_thickness_supports_python_lists_and_arrays` validating both raw Python lists and NumPy arrays without errors.
    - Added `test_calculate_average_section_thickness_with_bad_sections_filtering` verifying bad section exclusion across Series/lists.
- **Verification**: Verified with `pytest tests/test_spacing_and_indexing.py -v` (8 passed in 6.40s) and full test suites `pytest tests/test_curation_state.py tests/test_session_roundtrip.py tests/test_curation_state_rat.py tests/test_quicknii_roundtrip.py tests/test_training_utils.py -v` (40 passed, 0 failures).
- **Status**: Completed

## Backlog
