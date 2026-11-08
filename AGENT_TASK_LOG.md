# Agent Task Log — Continuous Improvement

Shared continuity file for automated maintenance runs. Multiple agents may

work this file; always append, never overwrite another agent's entries.



This file is the coordination point for the recurring maintenance task only.

`CHANGELOG.md` remains the repo's own release log — record new maintenance work

here.



## In Progress

## Completed

### Task: Curation Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across Curation Page controls: `curationPrevBtn`, `curationNextBtn`, `curationSelectAllBtn`, `curationDeselectAllBtn`, `confidenceFilterCombo`, `sliceFlagList`, `applyBadSectionsBtn`, `applyManualOrderBtn`, `moveSliceUpBtn`, `moveSliceDownBtn`, `detectOutliersBtn`, `resetFlagsBtn`, `autoFlagLowConfBtn`, `interpolateBadDepthBtn`, `toggleCurrentFlagBtn`, `sliceNoteEdit`, `saveSliceNoteBtn`, `curationQueueLabel`, `controlsGroupBox`, `normalizeAnglesBtn`, `enforceOrderBtn`, `enforceSpacingBtn`, `mlSpin`, `dvSpin`, `applyManualAnglesBtn`, `undoBtn`, `redoBtn`, `anchorGroupBox`, `anchorSummaryLabel`, `anchorDepthSpin`, `setAnchorBtn`, `removeAnchorBtn`, `applyAnchorInterpolationBtn`, `clearAnchorBtn`, `anchorList`, `linearityCanvas`, `zoomFitBtn`, `zoomInBtn`, `zoomOutBtn`, `histCanvas`, `enableAtlasPreviewCheck`, `atlasVolumeCombo`, `enableBlendOverlayCheck`, `blendSlider`, `blendPercentLabel`, `atlasSliceInfoLabel`, `beforeAfterToggle`, `loupeToggle`, `atlasCoordsLabel`, `atlasFlipXCheck`, `atlasFlipYCheck`, `atlasRotateCombo`, `atlasScaleSlider`, `atlasScaleLabel`, `atlasOffsetXSlider`, `atlasOffsetXLabel`, `atlasOffsetYSlider`, `atlasOffsetYLabel`, `confidencePanelToggle`, `confidencePanel`, `confidenceOverlayViewer`, `curationViewer`, `atlasViewer`.
    - Standardized accessible names, descriptions, tooltips, and pointing hand cursor affordances across left panel list navigation, outlier detection, angle propagation, anchor alignment, linearity plot zoom toolbar, volume selection, blend overlays, and synchronized slice/atlas viewers.
    - Exposed 65 public widget properties on `DeepSliceMainWindow` and added 46 ergonomic state queries, setters, and action triggers.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_curation_page_accessibility_properties_and_ergonomics` verifying object names, accessible metadata, cursor affordances, properties, and ergonomic setters, getters, and action triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (41 passed in 10.73s, 0 errors).
- **Status**: Completed

### 2026-11-08 - Atomic QUINT JSON / QuickNII XML export (Claude Sonnet 5.5)
Status: **done** (PR left open for review, not merged, per coordinator)
- Root cause: `write_QUINT_JSON` opened the target with `"w"` and streamed `json.dump`, so a serialisation error (reproduced with an ndarray in `markers`) left truncated invalid JSON over the user's previous alignment; `write_QuickNII_XML` had the same truncate-first shape.
- Fix: new `DeepSlice/read_and_write/atomic_io.py` (`write_json_atomic` moved there, re-exported from `gui/state.py`; new `write_file_atomic`); both exporters use it; ndarray markers are converted with `tolist()`.
- Validation: `tests/test_quint_export_atomic.py` (3 tests, all red before the fix); full suite 564 passed, 9 skipped; ruff clean on new files.
- PR: https://github.com/leonardolv/DeepSlice-GUI/pull/64 (draft, not merged)

### Task: Ingestion Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across Ingestion Page controls: `drop_area` (`ingestionDropArea`), `add_folder_button` (`addFolderBtn`), `add_files_button` (`addFilesBtn`), `clear_images_button` (`clearImagesBtn`), `enable_section_numbers_checkbox` (`enableSectionNumbersCheck`), `naming_helper_button` (`namingHelperBtn`), `legacy_parsing_checkbox` (`legacyParsingCheck`), `orientation_combo` (`orientationCombo`), `orientation_guide_button` (`orientationGuideBtn`), `slice_count_label` (`sliceCountLabel`), `file_size_summary_label` (`fileSizeSummaryLabel`), `index_table` (`indexTable`), `thumbnail_sort_combo` (`thumbnailSortCombo`), `thumbnail_filter_edit` (`thumbnailFilterEdit`), `thumbnail_list` (`ingestionThumbnailList`), `thumbnail_progress` (`thumbnailProgress`), `ingestion_preview` (`ingestionPreview`).
    - Standardized accessible names, descriptions, tooltips, and pointing hand cursor affordances across file addition, option configuration, sorting, filtering, and preview controls.
    - Exposed 19 public widget properties on `DeepSliceMainWindow`: `drop_area_widget`, `add_folder_btn`, `add_files_btn`, `clear_images_btn`, `enable_section_numbers_check`, `naming_helper_btn`, `legacy_parsing_check`, `orientation_combobox`, `orientation_guide_btn`, `slice_count_lbl`, `file_size_summary_lbl`, `ingestion_summary_banner_lbl`, `ingestion_warning_lbl`, `index_table_widget`, `thumbnail_sort_combobox`, `thumbnail_filter_input`, `thumbnail_list_widget`, `thumbnail_progress_bar`, `ingestion_preview_view`.
    - Added 19 ergonomic state queries and triggers: `slice_count_text()`, `file_size_summary_text()`, `ingestion_summary_banner_text()`, `ingestion_warning_text()`, `is_section_number_detection_enabled()`, `set_section_number_detection_enabled()`, `is_legacy_parsing_enabled()`, `set_legacy_parsing_enabled()`, `cutting_plane_orientation()`, `set_cutting_plane_orientation()`, `thumbnail_sort_order()`, `set_thumbnail_sort_order()`, `thumbnail_filter_query()`, `set_thumbnail_filter_query()`, `clear_thumbnail_filter()`, `index_table_row_count()`, `trigger_clear_images()`, `trigger_naming_helper()`, `trigger_orientation_guide()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_ingestion_page_accessibility_properties_and_ergonomics` verifying object names, accessible metadata, cursor affordances, properties, and ergonomic setters, getters, and action triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (40 passed in 11.89s, 0 errors).
- **Status**: Completed


### 2026-11-08 - Export Page Accessibility, Pointer Affordances, Object Names, Properties, and Ergonomics Helpers
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Standardized explicit object names across Export Page controls: `output_dir_edit` (`outputDirEdit`), `browse_output_dir_button` (`browseOutputDirBtn`), `output_basename_edit` (`outputBasenameEdit`), `output_format_combo` (`outputFormatCombo`), `output_format_help_button` (`outputFormatHelpBtn`), `export_size_estimate_label` (`exportSizeEstimateLabel`), `export_button` (`exportBtn`), `open_export_dir_button` (`openExportDirBtn`), `copy_export_path_button` (`copyExportPathBtn`), `report_button` (`reportBtn`), `preview_report_button` (`previewReportBtn`), `pdf_content_group` (`pdfContentGroup`), `pdf_include_stats` (`pdfIncludeStatsCheck`), `pdf_include_plot` (`pdfIncludePlotCheck`), `pdf_include_images` (`pdfIncludeImagesCheck`), `pdf_include_angles` (`pdfIncludeAnglesCheck`), `quicknii_path_edit` (`quickniiPathEdit`), `quicknii_browse_button` (`quickniiBrowseBtn`), `open_quicknii_button` (`openQuickniiBtn`), `summary_label` (`exportSummaryLabel`), `deviation_label` (`exportDeviationLabel`). Exposed public widget properties on `DeepSliceMainWindow` (`output_dir_line_edit`, `browse_output_dir_btn`, `output_basename_line_edit`, `output_format_combobox`, `output_format_help_btn`, `export_size_estimate_lbl`, `export_btn`, `open_export_dir_btn`, `copy_export_path_btn`, `report_btn`, `preview_report_btn`, `pdf_content_group_box`, `pdf_include_stats_check`, `pdf_include_plot_check`, `pdf_include_images_check`, `pdf_include_angles_check`, `quicknii_path_line_edit`, `quicknii_browse_btn`, `open_quicknii_btn`, `export_summary_label`, `export_deviation_label`). Added ergonomic state queries and triggers (`export_output_dir()`, `set_export_output_dir(path)`, `export_base_filename()`, `set_export_base_filename(name)`, `export_format_name()`, `set_export_format_name(format_name)`, `is_pdf_stats_included()`, `set_pdf_stats_included(enabled)`, `is_pdf_plot_included()`, `set_pdf_plot_included(enabled)`, `is_pdf_images_included()`, `set_pdf_images_included(enabled)`, `is_pdf_angles_included()`, `set_pdf_angles_included(enabled)`, `export_summary_text()`, `export_deviation_text()`, `trigger_export_format_help()`, `trigger_copy_export_path()`, `trigger_preview_report()`).
- `tests/test_ui_accessibility_and_startup.py`: Added unit test `test_export_page_accessibility_properties_and_ergonomics` validating object names, cursor affordances, properties, and ergonomic setters/getters/triggers.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (38 passed in 15.09s, 0 errors).

### 2026-11-08 - Configuration Page Accessibility, Pointer Affordances, Object Names, Properties, and Ergonomics Helpers
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Configured explicit object names, accessible descriptions, tooltips, and pointing hand cursors across Configuration Page controls: `mouse_radio` (`mouseRadio`), `rat_radio` (`ratRadio`), `auto_thickness_checkbox` (`autoThicknessCheck`), `thickness_spin` (`thicknessSpin`), `suggest_thickness_button` (`suggestThicknessBtn`), `direction_override_combo` (`directionOverrideCombo`), `direction_guide_button` (`directionGuideBtn`), `advanced_options_toggle` (`advancedOptionsToggle`), `ensemble_checkbox` (`ensembleCheck`), `ensemble_help_button` (`ensembleHelpBtn`), `secondary_model_checkbox` (`secondaryModelCheck`), `legacy_from_config_checkbox` (`legacyParserCheck`), `outlier_sigma_spin` (`outlierSigmaSpin`), `confidence_high_spin` (`confidenceHighSpin`), `confidence_medium_spin` (`confidenceMediumSpin`), `inference_batch_spin` (`inferenceBatchSpin`), `quality_gate_checkbox` (`qualityGateCheck`), `tissue_crop_checkbox` (`tissueCropCheck`), `clahe_checkbox` (`claheCheck`), `stain_norm_combo` (`stainNormCombo`), `preserve_color_checkbox` (`preserveColorCheck`), `percentile_norm_checkbox` (`percentileNormCheck`), `bilateral_checkbox` (`bilateralCheck`), `gamma_spin` (`gammaSpin`), `min_resolution_spin` (`minResolutionSpin`), `blur_threshold_spin` (`blurThresholdSpin`), `tta_checkbox` (`ttaCheck`), `multiscale_checkbox` (`multiscaleCheck`), `fast_fp16_checkbox` (`fastFp16Check`), `section_dropout_spin` (`sectionDropoutSpin`), `conf_weighted_ensemble_checkbox` (`confWeightedEnsembleCheck`), `training_group_mode_combo` (`trainingGroupModeCombo`), `training_seed_spin` (`trainingSeedSpin`), `training_train_fraction_spin` (`trainingTrainFractionSpin`), `training_val_fraction_spin` (`trainingValFractionSpin`), `training_patience_spin` (`trainingPatienceSpin`), `training_lr_factor_spin` (`trainingLrFactorSpin`), `training_min_lr_spin` (`trainingMinLrSpin`), `training_mixed_precision_checkbox` (`trainingMixedPrecisionCheck`). Exposed public widget properties on `DeepSliceMainWindow` (`mouse_atlas_radio`, `rat_atlas_radio`, `auto_thickness_check`, `thickness_spin_box`, `suggest_thickness_btn`, `direction_override_combobox`, `direction_guide_btn`, `advanced_options_toggle_btn`, `ensemble_check`, `ensemble_help_btn`, `secondary_model_check`, `legacy_parser_check`, `outlier_sigma_spin_box`, `confidence_high_spin_box`, `confidence_medium_spin_box`, `inference_batch_spin_box`, `quality_gate_check`, `tissue_crop_check`, `clahe_check`, `stain_norm_combobox`, `preserve_color_check`, `percentile_norm_check`, `bilateral_check`, `gamma_spin_box`, `min_resolution_spin_box`, `blur_threshold_spin_box`, `tta_check`, `multiscale_check`, `fast_fp16_check`, `section_dropout_spin_box`, `conf_weighted_ensemble_check`, `training_group_mode_combobox`, `preview_training_split_btn`, `save_training_split_btn`, `save_training_metadata_btn`). Added ergonomic state queries and triggers (`is_auto_thickness_enabled()`, `set_auto_thickness_enabled()`, `section_thickness_um()`, `set_section_thickness_um()`, `direction_override_text()`, `set_direction_override()`, `is_advanced_options_expanded()`, `set_advanced_options_expanded()`, `is_ensemble_enabled()`, `set_ensemble_enabled()`, `is_quality_gate_enabled()`, `set_quality_gate_enabled()`, `is_tissue_crop_enabled()`, `set_tissue_crop_enabled()`, `is_clahe_enabled()`, `set_clahe_enabled()`, `trigger_suggest_thickness()`, `trigger_direction_guide()`, `trigger_ensemble_help()`, `trigger_preview_training_split()`).
- `tests/test_ui_accessibility_and_startup.py`: Added unit test `test_configuration_page_accessibility_properties_and_ergonomics` validating accessibility metadata, pointer cursors, property aliases, and ergonomic queries/triggers.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (37 passed in 11.18s, 0 errors).

### 2026-11-07 - ThumbnailListWidget and FlagListWidget Accessibility and Curation Ergonomics Helpers
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added explicit object name (`thumbnailListWidget`), accessible description, and tooltip to `ThumbnailListWidget`; added public query and selection helpers `item_count()`, `has_items()`, `selected_index()`, `select_index(index)`, `item_texts()`, and `selected_text()`. Added explicit `__init__` on `FlagListWidget` configuring object name (`flagListWidget`), accessible name, description, and spacebar tooltip; added checklist queries and item manipulation helpers `item_count()`, `has_items()`, `checked_count()`, `unchecked_count()`, `is_item_checked(row)`, `set_item_checked(row, checked)`, `toggle_item(row)`, `checked_texts()`, `selected_index()`, `select_index(row)`, and `trigger_toggle_current()`. Added public count properties `thumbnail_count`, `curation_slice_count`, `curation_flagged_count` and action triggers `trigger_apply_bad_section_flags()`, `trigger_apply_manual_order()`, `trigger_curation_select_all()`, `trigger_curation_deselect_all()` to `DeepSliceMainWindow`.
- `tests/test_ui_accessibility_and_startup.py`: Added unit test `test_thumbnail_and_flag_list_widget_and_main_window_curation_ergonomics` validating accessibility metadata, item text queries, row selection, check state toggles, and MainWindow curation count/trigger properties.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (36 passed in 14.04s, 0 errors).

### 2026-11-07 - DropArea, ToastOverlay, and SliceGraphicsView Accessibility and Ergonomics Helpers
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added accessible descriptions and explicit child label attributes (`icon_label`, `title_label`, `subtitle_label`) to `DropArea`; added text query helpers `title_text()` and `subtitle_text()`; configured accessible name, description, and child label metadata (`ToastMessageLabel`) on `ToastOverlay` with `message_text()` and `is_active()` query helpers; configured accessibility metadata on `SliceGraphicsView` (`setObjectName("sliceGraphicsView")`, `setAccessibleName("Slice Interactive Canvas")`, description) and added query helpers `zoom_level()`, `is_loupe_enabled()`, `has_sync_partner()`, `is_mouse_tracking_active()`.
- `tests/test_ui_accessibility_and_startup.py`: Added unit test `test_drop_area_toast_overlay_and_slice_view_ergonomics` validating accessibility metadata, label object names, text readouts, loupe toggling, and query helper state.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (35 passed in 24.92s, 0 errors).


### 2026-11-05 - DeepSliceMainWindow Page Step Helpers, Bad Section Queries, and Species Selection
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added public ergonomic workflow helpers (`next_page`, `previous_page`, `select_species`) and status/export properties (`bad_section_count`, `has_bad_sections`, `export_format`, `export_basename`).
- `tests/test_ui_accessibility_and_startup.py`: Added unit test coverage `test_main_window_extended_ergonomics_and_species_selection` asserting forward/backward step execution, programmatic species selection with radio button syncing, bad section queries, and export properties.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (34 passed in 12.28s, 0 errors).

### 2026-11-05 UTC - DeepSliceMainWindow Page Navigation Properties, Workflow State Queries, and Helper Dialog Coverage
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added public ergonomic properties and navigation helpers: `page_count`, `can_navigate_forward`, `can_navigate_backward`, `has_images`, `has_anchors`, `active_theme`, and `go_to_page(page_index)` for programmatic testing and inspection.
- `tests/test_ui_accessibility_and_startup.py`: Added unit test `test_main_window_extended_navigation_and_helper_dialogs` asserting navigation properties, step boundaries, programmatic transitions, and headless execution across all helper dialogs (shortcuts, onboarding, what-is-new, naming, orientation, direction, ensemble, and export format).
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (33 passed in 10.95s, 0 errors).

### 2026-11-05 UTC - Guard Confirmation Modals with Offscreen Headless Safety in Session Lifecycle and Quality Gates
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added headless offscreen guards around `QMessageBox.question` confirmation prompts in `_reset_session`, `closeEvent`, `_on_quality_gate_error`, `_on_quality_gate_finished`, `_on_prediction_failed` partial recovery, and `_export_predictions` low-confidence warning to prevent blocking modal popups during automated headless runs.
- `tests/test_ui_accessibility_and_startup.py`: Added `test_headless_session_reset_and_quality_gate_confirmations` verifying non-blocking execution across dirty session resets, quality gate errors, warning handling, and window close events.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (32 passed in 11.28s, 0 errors).

### 2026-11-05 UTC - Replace Naked QMessageBox in Alignment and Anchor Workflows with Accessible Actionable Warnings
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Replaced naked modal `QMessageBox.warning` in `_run_alignment` and `_apply_anchor_interpolation` with `self._actionable_warning`; added actionable `("Go to Ingestion Page", ...)` guidance button when prediction is triggered with zero images; added offscreen headless checks for `QMessageBox.question` confirmation prompts; updated curation confidence details dialog to use accessible non-blocking warning.
- `tests/test_ui_accessibility_and_startup.py`: Added `test_alignment_and_anchor_headless_actionable_warnings` verifying actionable warnings and navigation callbacks under headless offscreen execution.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (31 passed in 8.52s, 0 errors).

### 2026-11-04 UTC - DeepSliceMainWindow Root Accessibility Metadata, Object Names, and Public Ergonomic Property Helpers
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Configured explicit `setObjectName("deepSliceMainWindow")`, `setAccessibleName("DeepSlice Desktop Main Window")`, and `setAccessibleDescription(...)` on root window; added public ergonomic property helpers: `current_page_index`, `current_page_title`, `loaded_image_count`, `selected_species`, `has_predictions`, and `is_dirty`.
- `tests/test_ui_accessibility_and_startup.py`: Added unit test assertions in `test_main_window_status_accessibility_and_headless_startup` verifying root window object name, accessibility name and description, and initial state of public property accessors.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (19 passed in 19.43s, 0 errors).

### 2026-11-04 UTC - Replace Naked QMessageBox.warning with Accessible _actionable_warning and Action Callbacks
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Replaced naked modal `QMessageBox.warning` in `_export_predictions`, `_generate_report`, and `_preview_report` with `self._actionable_warning`; added actionable `actions=[("Run Prediction", self._start_prediction_worker)]` callbacks guiding users directly into running predictions, and `actions=[("Choose Directory...", self._browse_output_directory)]` for unwritable directories; ensured non-blocking headless execution under offscreen test runners.
- `tests/test_ui_accessibility_and_startup.py`: Added `test_export_and_report_headless_actionable_warnings` asserting non-blocking behavior, warning titles, messages, and actionable prediction callbacks.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (30 passed in 9.92s, 0 errors).
### 2026-10-30 UTC - A user-specified section thickness reversed caudal-to-rostral series
Status: **done** (Claude)

**Root cause.** `space_according_to_index(section_thickness=...)` used the positive user value directly, but the inferred thickness is signed (depth falls with section number when a series is cut caudal-to-rostral). With a given thickness the ideal spacing therefore ran the opposite way and the sections were laid out in reverse order (verified: oy 130..100 became 105..135). Separately, `ideal_spacing` with every section flagged bad raised a bare `ZeroDivisionError` from `np.average`.

**Fix.** A given thickness is taken as a magnitude and signed by `determine_direction_of_indexing`; `ideal_spacing` raises a clear `ValueError` when all sections are bad.

**Validation.** New `tests/test_user_thickness_direction.py` (first and third tests fail before the fix); full suite 538 passed, 10 skipped. PR: see GitHub branch `claude/bold-cray-nvlnre`.
### 2026-11-04 UTC - Spacing with every section marked bad raised a bare ZeroDivisionError
Status: **done** (Claude)

**Root cause.** `ideal_spacing` weights depths with `np.average`; when every section is flagged bad (reachable with a user-specified `section_thickness`, which skips the thickness check) all weights are 0 and numpy raised an unhelpful `ZeroDivisionError: Weights sum to zero`.

**Fix.** Explicit `ValueError` telling the user to unmark at least one section.

**Validation.** New `tests/test_ideal_spacing_all_bad.py` (all-bad case fails before the fix); related pure-logic tests pass. The Qt/GUI test modules cannot be collected in this sandbox (no PySide6), so the full suite was not run. PR: see GitHub (branch `claude/bold-cray-1dilav`).
### 2026-10-30 UTC - Bad-section flagging by exact filename was rejected when the name was a substring of another
Status: **done** (Claude)

**Root cause.** The GUI flags bad sections by passing full filenames to `set_bad_sections_util`, which does a substring match and raises "Multiple sections match" if more than one file matches. With files like `1.png`, `11.png`, `21.png`, flagging `1.png` always failed with an error dialog.

**Fix.** New `_match_bad_section`: an identifier equal to exactly one filename (or basename) selects only that file; otherwise the old literal-substring behaviour and uniqueness check apply.

**Validation.** New `tests/test_bad_sections_exact_filename.py` (first test fails before the fix). Suite: 541 passed, 9 skipped; 3 pre-existing failures in `test_neural_network_utils.py` (also fail on clean main in this sandbox).

### 2026-10-28 UTC - Auto bad-section detection flagged every section of a perfectly linear series
Status: **done** (Claude)

**Root cause.** `set_bad_sections_util(auto=True)` flagged sections whose depth residual exceeded `1.5 * std(residuals)`. For an exactly evenly spaced series the residuals are pure floating-point noise (~1e-14), so the cutoff is noise too and every section was marked bad (verified: 12 of 12), after which angle propagation fails with "no usable sections".

**Fix.** The cutoff is floored at `1e-9 * max(1, ptp(depths))`; real outliers are still flagged.

**Validation.** New `tests/test_auto_bad_sections_noise_floor.py` (linear-series test fails before the fix); full suite 534 passed, 10 skipped. PR: see GitHub (branch `claude/bold-cray-9v6ioy`).

### 2026-10-28 UTC - Bad-section identifiers were interpreted as regular expressions
Status: **done** (Claude)

**Root cause.** `set_bad_sections_util` used `Series.str.contains(bad_section)`, whose default is `regex=True`. Identifiers such as `a.1` matched unrelated files (`aX1`), and `s[1]+` or an unbalanced `(` silently matched the wrong sections or raised `re.error`.

**Fix.** Literal substring match (`regex=False`, identifier coerced to `str`); the "multiple sections match" check is unchanged.

**Validation.** New `tests/test_bad_sections_literal_match.py` (2 of 4 fail before the fix); full suite 532 passed, 10 skipped. PR: see GitHub branch `claude/bold-cray-4f1lyt`.

### 2026-10-27 UTC - Session load dropped all three quality thresholds when one was invalid
Status: **done** (Claude)

**Root cause.** `load_session_dict` passed outlier sigma, medium and high confidence to `set_quality_controls` (which validates them jointly) inside `try/except Exception: pass`. One out-of-range or malformed value silently discarded the other valid ones too, with no log line.

**Fix.** `DeepSliceAppState._apply_session_quality_controls`: tries the joint apply first; on failure, applies each value individually on top of the current settings, keeps valid ones, and logs a warning naming each rejected value.

**Validation.** New `tests/test_session_quality_controls.py` (5 of 6 fail before the fix); full suite 534 passed, 9 skipped (scikit-image, matplotlib, PySide6 had to be installed in the sandbox). PR: see GitHub branch `claude/wonderful-euler-5agjho`.


### 2026-10-26 UTC - PreferencesDialog get_values/set_values helpers, form layout bindings, and label object names
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added explicit object names on preferences form labels (`prefSpeciesLabel`, `prefThemeLabel`, `prefOutputDirLabel`, `prefQuickNiiLabel`, `prefConsoleLabel`); added `dialog.get_values()` and `dialog.set_values()` helpers; exposed `dialog.form`; updated `_save_and_apply_preferences` to consume `dialog.get_values()`.
- `tests/test_ui_accessibility_and_startup.py`: Added assertions testing form layout, label object names, and programmatic get/set values.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (18 passed in 9.09s, 0 errors).

### 2026-10-26 UTC - PreferencesDialog Combobox Cursors, Automatic Acceptance Persistence, and Unit Tests
Status: **done** (Antigravity)
- `DeepSlice/gui/main_window.py`: Added pointing hand cursor affordances to `species_combo` and `theme_combo` in `_create_preferences_dialog`; refactored `_save_and_apply_preferences(dialog)` and wired `dialog.accepted` signal so settings are safely and uniformly persisted across interactive and headless test invocations.
- `tests/test_ui_accessibility_and_startup.py`: Expanded `test_preferences_dialog_accessibility_and_reset_to_defaults` to test combobox cursor shapes, headless dialog acceptance, and settings persistence.
- Validation: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (18 passed in 5.41s, 0 errors).

