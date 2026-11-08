# GUI & UX Maintenance Log

## In Progress


## Completed

### Task: TopBar, Sidebar, GuideBanner, and StatusBar explicit object names, public properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across TopBar, Sidebar, GuideBanner, and StatusBar controls: `TopBar`, `ProjectLabel`, `SessionLabel`, `HardwareLabel`, `hardwareHealthBtn`, `themeToggleBtn`, `newSessionBtn`, `saveSessionBtn`, `loadSessionBtn`, `sessionIoSpinner`, `shortcutHelpBtn`, `preferencesBtn`, `aboutBtn`, `ErrorMenuButton`, `sidebarContainer`, `CompletionLabel`, `CollapseButton`, `StepNavigator`, `GuideBanner`, `GuideTitle`, `GuideText`, `GuideNextButton`, `mainStatusBar`, and `globalProgressBar`.
    - Added pointing hand cursor affordances across all interactive buttons.
    - Exposed 24 public widget properties on `DeepSliceMainWindow`: `top_bar_widget`, `project_title_label`, `session_status_lbl`, `hardware_mode_lbl`, `hardware_health_btn`, `theme_toggle_btn`, `new_session_btn`, `save_session_btn`, `load_session_btn`, `session_io_spinner_widget`, `shortcut_help_btn`, `preferences_btn`, `about_btn`, `error_menu_btn`, `sidebar_container_widget`, `completion_lbl`, `collapse_sidebar_btn`, `step_list_widget`, `guide_banner_frame`, `guide_title_lbl`, `guide_text_lbl`, `guide_next_btn`, `status_bar_widget`, and `global_progress_bar`.
    - Added 17 ergonomic state queries, setters, and programmatic action triggers: `project_title_text()`, `session_status_text()`, `hardware_mode_text()`, `is_sidebar_collapsed()`, `set_sidebar_collapsed()`, `completion_percentage_text()`, `guide_title_text()`, `guide_body_text()`, `is_guide_next_visible()`, `status_bar_message()`, `is_global_progress_visible()`, `trigger_hardware_health()`, `trigger_shortcuts_help()`, `trigger_about_dialog()`, `trigger_open_preferences()`, `trigger_toggle_sidebar()`, and `trigger_guide_next()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added comprehensive unit test `test_top_bar_sidebar_guide_banner_and_status_bar_accessibility_and_ergonomics` validating control object names, accessible metadata, cursor affordances, public property accessors, ergonomic queries, sidebar toggling, step navigation, and modal-safe helper dialog triggers.
- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py tests/test_drop_event_toast.py -v` (42 passed in 11.39s, 0 errors).
- **Status**: Completed


### Task: Curation Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across Curation Page controls: `curationPrevBtn`, `curationNextBtn`, `curationSelectAllBtn`, `curationDeselectAllBtn`, `confidenceFilterCombo`, `sliceFlagList`, `applyBadSectionsBtn`, `applyManualOrderBtn`, `moveSliceUpBtn`, `moveSliceDownBtn`, `detectOutliersBtn`, `resetFlagsBtn`, `autoFlagLowConfBtn`, `interpolateBadDepthBtn`, `toggleCurrentFlagBtn`, `sliceNoteEdit`, `saveSliceNoteBtn`, `curationQueueLabel`, `controlsGroupBox`, `normalizeAnglesBtn`, `enforceOrderBtn`, `enforceSpacingBtn`, `mlSpin`, `dvSpin`, `applyManualAnglesBtn`, `undoBtn`, `redoBtn`, `anchorGroupBox`, `anchorSummaryLabel`, `anchorDepthSpin`, `setAnchorBtn`, `removeAnchorBtn`, `applyAnchorInterpolationBtn`, `clearAnchorBtn`, `anchorList`, `linearityCanvas`, `zoomFitBtn`, `zoomInBtn`, `zoomOutBtn`, `histCanvas`, `enableAtlasPreviewCheck`, `atlasVolumeCombo`, `enableBlendOverlayCheck`, `blendSlider`, `blendPercentLabel`, `atlasSliceInfoLabel`, `beforeAfterToggle`, `loupeToggle`, `atlasCoordsLabel`, `atlasFlipXCheck`, `atlasFlipYCheck`, `atlasRotateCombo`, `atlasScaleSlider`, `atlasScaleLabel`, `atlasOffsetXSlider`, `atlasOffsetXLabel`, `atlasOffsetYSlider`, `atlasOffsetYLabel`, `confidencePanelToggle`, `confidencePanel`, `confidenceOverlayViewer`, `curationViewer`, `atlasViewer`.
    - Standardized accessible names, descriptions, tooltips, and pointing hand cursor affordances across left panel list navigation, outlier detection, angle propagation, anchor alignment, linearity plot zoom toolbar, volume selection, blend overlays, and synchronized slice/atlas viewers.
    - Exposed 65 public widget properties on `DeepSliceMainWindow`: `curation_prev_btn`, `curation_next_btn`, `curation_select_all_button`, `curation_deselect_all_button`, `confidence_filter_combobox`, `slice_flag_list_widget`, `apply_bad_sections_btn`, `apply_manual_order_btn`, `move_slice_up_btn`, `move_slice_down_btn`, `detect_outliers_btn`, `reset_flags_btn`, `auto_flag_low_conf_btn`, `interpolate_bad_depth_btn`, `toggle_current_flag_btn`, `slice_note_line_edit`, `save_slice_note_btn`, `curation_queue_lbl`, `controls_group_box`, `normalize_angles_btn`, `enforce_order_btn`, `enforce_spacing_btn`, `ml_angle_spin_box`, `dv_angle_spin_box`, `apply_manual_angles_btn`, `undo_curation_btn`, `redo_curation_btn`, `anchor_group_box`, `anchor_summary_lbl`, `anchor_depth_spin_box`, `set_anchor_btn`, `remove_anchor_btn`, `apply_anchor_interpolation_btn`, `clear_anchor_btn`, `anchor_list_widget`, `linearity_figure_canvas`, `zoom_fit_plot_btn`, `zoom_in_plot_btn`, `zoom_out_plot_btn`, `hist_figure_canvas`, `enable_atlas_preview_check`, `atlas_volume_combobox`, `enable_blend_overlay_check`, `blend_opacity_slider`, `blend_percent_lbl`, `atlas_slice_info_lbl`, `before_after_check`, `loupe_check`, `atlas_coords_lbl`, `atlas_flip_x_check`, `atlas_flip_y_check`, `atlas_rotate_combobox`, `atlas_scale_slider_widget`, `atlas_scale_lbl`, `atlas_offset_x_slider_widget`, `atlas_offset_x_lbl`, `atlas_offset_y_slider_widget`, `atlas_offset_y_lbl`, `confidence_panel_toggle_btn`, `confidence_panel_frame`, `confidence_overlay_slice_viewer`, `curation_slice_viewer`, `atlas_slice_viewer`.
    - Added 46 ergonomic state queries, setters, and action triggers: `curation_confidence_filter()`, `set_curation_confidence_filter()`, `slice_note_text()`, `set_slice_note_text()`, `manual_ml_angle()`, `set_manual_ml_angle()`, `manual_dv_angle()`, `set_manual_dv_angle()`, `anchor_depth()`, `set_anchor_depth()`, `anchor_count()`, `is_atlas_preview_enabled()`, `set_atlas_preview_enabled()`, `is_blend_overlay_enabled()`, `set_blend_overlay_enabled()`, `blend_opacity()`, `set_blend_opacity()`, `is_before_after_overlay_enabled()`, `set_before_after_overlay_enabled()`, `is_loupe_enabled()`, `set_loupe_enabled()`, `is_atlas_flip_x_enabled()`, `set_atlas_flip_x_enabled()`, `is_atlas_flip_y_enabled()`, `set_atlas_flip_y_enabled()`, `atlas_scale_percent()`, `set_atlas_scale_percent()`, `atlas_offset_x()`, `set_atlas_offset_x()`, `atlas_offset_y()`, `set_atlas_offset_y()`, `is_confidence_panel_expanded()`, `set_confidence_panel_expanded()`, `trigger_curation_prev_slice()`, `trigger_curation_next_slice()`, `trigger_move_slice_up()`, `trigger_move_slice_down()`, `trigger_detect_outliers()`, `trigger_reset_flags()`, `trigger_auto_flag_low_conf()`, `trigger_interpolate_bad_depth()`, `trigger_toggle_current_flag()`, `trigger_save_slice_note()`, `trigger_normalize_angles()`, `trigger_enforce_order()`, `trigger_enforce_spacing()`, `trigger_apply_manual_angles()`, `trigger_undo()`, `trigger_redo()`, `trigger_set_anchor()`, `trigger_remove_anchor()`, `trigger_apply_anchor_interpolation()`, `trigger_clear_anchors()`, `trigger_plot_zoom_fit()`, `trigger_plot_zoom_in()`, `trigger_plot_zoom_out()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_curation_page_accessibility_properties_and_ergonomics` verifying object names, accessible metadata, cursor affordances, properties, and ergonomic setters, getters, and action triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (41 passed in 10.73s, 0 errors).
- **Status**: Completed

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


### Task: Prediction Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across Prediction Page controls: `run_alignment_button` (`runAlignmentBtn`), `cancel_alignment_button` (`cancelAlignmentBtn`), `prediction_phase_label` (`predictionPhaseLabel`), `prediction_progress_label` (`predictionProgressLabel`), `prediction_elapsed_label` (`predictionElapsedLabel`), `prediction_eta_label` (`predictionEtaLabel`), `prediction_progress_bar` (`predictionProgressBar`), `predicted_thickness_label` (`predictedThicknessLabel`), `accept_predicted_thickness_button` (`acceptPredictedThicknessBtn`), `prediction_direction_label` (`predictionDirectionLabel`), `console_toggle` (`consoleToggle`), `console_autoscroll_toggle` (`consoleAutoscrollToggle`), `clear_console_button` (`clearConsoleButton`), `copy_console_button` (`copyConsoleButton`), `console_output` (`consoleOutput`), `run_blocker_label` (`WarningText`), `prediction_slice_selector` (`predictionSliceSelector`), `prediction_compare_checkbox` (`predictionCompareCheck`), `prediction_atlas_info_label` (`predictionAtlasInfoLabel`), `preprocessing_preview_info_label` (`preprocessingPreviewInfoLabel`), `preprocessing_preview_label` (`preprocessingPreviewLabel`), `prediction_viewer` (`predictionSliceViewer`), `prediction_atlas_viewer` (`predictionAtlasViewer`).
    - Standardized accessible names, descriptions, tooltips, and pointing hand cursor affordances across execution, preview, and console controls.
    - Exposed 23 public widget properties on `DeepSliceMainWindow`: `run_alignment_btn`, `cancel_alignment_btn`, `prediction_phase_lbl`, `prediction_progress_lbl`, `prediction_elapsed_lbl`, `prediction_eta_lbl`, `prediction_progress_bar_widget`, `predicted_thickness_lbl`, `accept_predicted_thickness_btn`, `prediction_direction_lbl`, `console_toggle_btn`, `console_autoscroll_toggle_btn`, `clear_console_btn`, `copy_console_btn`, `console_output_widget`, `run_blocker_lbl`, `prediction_slice_selector_combo`, `prediction_compare_check`, `prediction_atlas_info_lbl`, `preprocessing_preview_info_lbl`, `preprocessing_preview_lbl`, `prediction_slice_viewer`, `prediction_atlas_viewer_widget`.
    - Added 22 ergonomic state queries and triggers: `prediction_phase_text()`, `prediction_progress_text()`, `prediction_elapsed_text()`, `prediction_eta_text()`, `prediction_progress_value()`, `predicted_thickness_text()`, `prediction_direction_text()`, `is_run_alignment_enabled()`, `is_cancel_alignment_enabled()`, `is_accept_predicted_thickness_enabled()`, `is_console_visible()`, `set_console_visible(visible)`, `is_console_autoscroll_enabled()`, `set_console_autoscroll_enabled(enabled)`, `console_text()`, `trigger_clear_console()`, `trigger_copy_console()`, `prediction_slice_count()`, `prediction_slice_texts()`, `selected_prediction_slice_index()`, `select_prediction_slice_index(index)`, `is_atlas_comparison_checked()`, `trigger_accept_predicted_thickness()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_prediction_page_accessibility_properties_and_ergonomics` verifying object names, accessible metadata, cursor affordances, properties, and ergonomic setters, getters, and action triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (39 passed in 12.12s, 0 errors).
- **Status**: Completed

### Task: Export Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Standardized explicit object names across Export Page controls: `output_dir_edit` (`outputDirEdit`), `browse_output_dir_button` (`browseOutputDirBtn`), `output_basename_edit` (`outputBasenameEdit`), `output_format_combo` (`outputFormatCombo`), `output_format_help_button` (`outputFormatHelpBtn`), `export_size_estimate_label` (`exportSizeEstimateLabel`), `export_button` (`exportBtn`), `open_export_dir_button` (`openExportDirBtn`), `copy_export_path_button` (`copyExportPathBtn`), `report_button` (`reportBtn`), `preview_report_button` (`previewReportBtn`), `pdf_content_group` (`pdfContentGroup`), `pdf_include_stats` (`pdfIncludeStatsCheck`), `pdf_include_plot` (`pdfIncludePlotCheck`), `pdf_include_images` (`pdfIncludeImagesCheck`), `pdf_include_angles` (`pdfIncludeAnglesCheck`), `quicknii_path_edit` (`quickniiPathEdit`), `quicknii_browse_button` (`quickniiBrowseBtn`), `open_quicknii_button` (`openQuickniiBtn`), `summary_label` (`exportSummaryLabel`), `deviation_label` (`exportDeviationLabel`).
    - Exposed public widget properties on `DeepSliceMainWindow`: `output_dir_line_edit`, `browse_output_dir_btn`, `output_basename_line_edit`, `output_format_combobox`, `output_format_help_btn`, `export_size_estimate_lbl`, `export_btn`, `open_export_dir_btn`, `copy_export_path_btn`, `report_btn`, `preview_report_btn`, `pdf_content_group_box`, `pdf_include_stats_check`, `pdf_include_plot_check`, `pdf_include_images_check`, `pdf_include_angles_check`, `quicknii_path_line_edit`, `quicknii_browse_btn`, `open_quicknii_btn`, `export_summary_label`, `export_deviation_label`.
    - Added ergonomic state queries and triggers: `export_output_dir()`, `set_export_output_dir(path)`, `export_base_filename()`, `set_export_base_filename(name)`, `export_format_name()`, `set_export_format_name(format_name)`, `is_pdf_stats_included()`, `set_pdf_stats_included(enabled)`, `is_pdf_plot_included()`, `set_pdf_plot_included(enabled)`, `is_pdf_images_included()`, `set_pdf_images_included(enabled)`, `is_pdf_angles_included()`, `set_pdf_angles_included(enabled)`, `export_summary_text()`, `export_deviation_text()`, `trigger_export_format_help()`, `trigger_copy_export_path()`, `trigger_preview_report()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_export_page_accessibility_properties_and_ergonomics` verifying object names, cursor affordances, properties, and ergonomic setters/getters/triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (38 passed in 15.09s, 0 errors).
- **Status**: Completed

### Task: Configuration Page accessibility, pointer affordances, object names, properties, and ergonomics helpers
- **Completed**: 2026-11-08
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Configured explicit object names, accessible descriptions, tooltips, and pointing hand cursors across Configuration Page controls: `mouse_radio` (`mouseRadio`), `rat_radio` (`ratRadio`), `auto_thickness_checkbox` (`autoThicknessCheck`), `thickness_spin` (`thicknessSpin`), `suggest_thickness_button` (`suggestThicknessBtn`), `direction_override_combo` (`directionOverrideCombo`), `direction_guide_button` (`directionGuideBtn`), `advanced_options_toggle` (`advancedOptionsToggle`), `ensemble_checkbox` (`ensembleCheck`), `ensemble_help_button` (`ensembleHelpBtn`), `secondary_model_checkbox` (`secondaryModelCheck`), `legacy_from_config_checkbox` (`legacyParserCheck`), `outlier_sigma_spin` (`outlierSigmaSpin`), `confidence_high_spin` (`confidenceHighSpin`), `confidence_medium_spin` (`confidenceMediumSpin`), `inference_batch_spin` (`inferenceBatchSpin`), `quality_gate_checkbox` (`qualityGateCheck`), `tissue_crop_checkbox` (`tissueCropCheck`), `clahe_checkbox` (`claheCheck`), `stain_norm_combo` (`stainNormCombo`), `preserve_color_checkbox` (`preserveColorCheck`), `percentile_norm_checkbox` (`percentileNormCheck`), `bilateral_checkbox` (`bilateralCheck`), `gamma_spin` (`gammaSpin`), `min_resolution_spin` (`minResolutionSpin`), `blur_threshold_spin` (`blurThresholdSpin`), `tta_checkbox` (`ttaCheck`), `multiscale_checkbox` (`multiscaleCheck`), `fast_fp16_checkbox` (`fastFp16Check`), `section_dropout_spin` (`sectionDropoutSpin`), `conf_weighted_ensemble_checkbox` (`confWeightedEnsembleCheck`), `training_group_mode_combo` (`trainingGroupModeCombo`), `training_seed_spin` (`trainingSeedSpin`), `training_train_fraction_spin` (`trainingTrainFractionSpin`), `training_val_fraction_spin` (`trainingValFractionSpin`), `training_patience_spin` (`trainingPatienceSpin`), `training_lr_factor_spin` (`trainingLrFactorSpin`), `training_min_lr_spin` (`trainingMinLrSpin`), `training_mixed_precision_checkbox` (`trainingMixedPrecisionCheck`).
    - Exposed public widget properties on `DeepSliceMainWindow`: `mouse_atlas_radio`, `rat_atlas_radio`, `auto_thickness_check`, `thickness_spin_box`, `suggest_thickness_btn`, `direction_override_combobox`, `direction_guide_btn`, `advanced_options_toggle_btn`, `ensemble_check`, `ensemble_help_btn`, `secondary_model_check`, `legacy_parser_check`, `outlier_sigma_spin_box`, `confidence_high_spin_box`, `confidence_medium_spin_box`, `inference_batch_spin_box`, `quality_gate_check`, `tissue_crop_check`, `clahe_check`, `stain_norm_combobox`, `preserve_color_check`, `percentile_norm_check`, `bilateral_check`, `gamma_spin_box`, `min_resolution_spin_box`, `blur_threshold_spin_box`, `tta_check`, `multiscale_check`, `fast_fp16_check`, `section_dropout_spin_box`, `conf_weighted_ensemble_check`, `training_group_mode_combobox`, `preview_training_split_btn`, `save_training_split_btn`, `save_training_metadata_btn`.
    - Added ergonomic state queries and triggers: `is_auto_thickness_enabled()`, `set_auto_thickness_enabled()`, `section_thickness_um()`, `set_section_thickness_um()`, `direction_override_text()`, `set_direction_override()`, `is_advanced_options_expanded()`, `set_advanced_options_expanded()`, `is_ensemble_enabled()`, `set_ensemble_enabled()`, `is_quality_gate_enabled()`, `set_quality_gate_enabled()`, `is_tissue_crop_enabled()`, `set_tissue_crop_enabled()`, `is_clahe_enabled()`, `set_clahe_enabled()`, `trigger_suggest_thickness()`, `trigger_direction_guide()`, `trigger_ensemble_help()`, `trigger_preview_training_split()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_configuration_page_accessibility_properties_and_ergonomics` verifying object names, accessible metadata, cursor affordances, property aliases, and ergonomic setters/getters/triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (37 passed in 11.18s, 0 errors).
- **Status**: Completed

### Task: ThumbnailListWidget and FlagListWidget accessibility and curation ergonomics helpers
- **Completed**: 2026-11-07
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added explicit object name (`thumbnailListWidget`), accessible description, and tooltip to `ThumbnailListWidget`.
    - Added public query and selection helpers: `item_count()`, `has_items()`, `selected_index()`, `select_index()`, `item_texts()`, and `selected_text()`.
    - Added explicit `__init__` on `FlagListWidget` setting object name (`flagListWidget`), accessible name, description, and spacebar tooltip.
    - Added checklist queries and item manipulation helpers: `item_count()`, `has_items()`, `checked_count()`, `unchecked_count()`, `is_item_checked()`, `set_item_checked()`, `toggle_item()`, `checked_texts()`, `selected_index()`, `select_index()`, and `trigger_toggle_current()`.
    - Added public count properties `thumbnail_count`, `curation_slice_count`, `curation_flagged_count` and action triggers `trigger_apply_bad_section_flags()`, `trigger_apply_manual_order()`, `trigger_curation_select_all()`, and `trigger_curation_deselect_all()` to `DeepSliceMainWindow`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_thumbnail_and_flag_list_widget_and_main_window_curation_ergonomics` validating accessibility metadata, item texts, selection, checkbox toggles, and curation triggers.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (36 passed in 14.04s, 0 errors).
- **Status**: Completed

### Task: DropArea, ToastOverlay, and SliceGraphicsView accessibility and ergonomics helpers
- **Completed**: 2026-11-07
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added accessible descriptions and explicit child label attributes (`icon_label`, `title_label`, `subtitle_label`) to `DropArea`.
    - Added text query helpers `title_text()` and `subtitle_text()`.
    - Configured accessible name, description, and child label metadata (`ToastMessageLabel`) on `ToastOverlay` with `message_text()` and `is_active()` query helpers.
    - Configured accessibility metadata on `SliceGraphicsView` (`setObjectName("sliceGraphicsView")`, `setAccessibleName("Slice Interactive Canvas")`, accessible description) and added query helpers `zoom_level()`, `is_loupe_enabled()`, `has_sync_partner()`, and `is_mouse_tracking_active()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_drop_area_toast_overlay_and_slice_view_ergonomics` validating accessibility metadata, label object names, text readouts, loupe toggling, and query helper state.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (35 passed in 24.92s, 0 errors).
- **Status**: Completed


### Task: DeepSliceMainWindow page step helpers, bad section queries, and species selection
- **Completed**: 2026-11-05
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added public ergonomic workflow helpers: `next_page()`, `previous_page()`, `select_species(species)`.
    - Added status and export property helpers: `bad_section_count`, `has_bad_sections`, `export_format`, and `export_basename`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_main_window_extended_ergonomics_and_species_selection` asserting forward/backward step execution, target species programmatic selection, bad section count and flag states, and export properties.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (34 passed in 12.28s, 0 errors).
- **Status**: Completed

### Task: DeepSliceMainWindow page navigation properties, workflow state queries, and helper dialog coverage
- **Completed**: 2026-11-05
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added public ergonomic properties and navigation helpers: `page_count`, `can_navigate_forward`, `can_navigate_backward`, `has_images`, `has_anchors`, `active_theme`, and `go_to_page()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_main_window_extended_navigation_and_helper_dialogs` asserting navigation properties, page bounds, and headless execution across all helper dialogs.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (33 passed in 10.95s, 0 errors).
- **Status**: Completed

### Task: Guard confirmation modals with offscreen headless safety in session lifecycle and quality gates
- **Completed**: 2026-11-05
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added headless offscreen guards around `QMessageBox.question` confirmation prompts across `_reset_session`, `closeEvent`, `_on_quality_gate_error`, `_on_quality_gate_finished`, partial prediction recovery, and low-confidence exports.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test `test_headless_session_reset_and_quality_gate_confirmations` asserting non-blocking session resets, quality gate errors, quality warnings, and window closing under headless execution.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (32 passed in 11.28s, 0 errors).
- **Status**: Completed

### Task: Replace naked QMessageBox calls in alignment and anchor workflows with accessible actionable warnings
- **Completed**: 2026-11-05
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Replaced naked modal `QMessageBox.warning` in `_run_alignment` with `self._actionable_warning` and added actionable "Go to Ingestion Page" button when no images are loaded.
    - Replaced naked modal `QMessageBox.warning` in `_apply_anchor_interpolation` for depth payload mismatches and invalid depth values with `self._actionable_warning`.
    - Added headless safety bypass for confirmation `QMessageBox.question` dialogs in alignment execution.
    - Updated curation slice details popup to use accessible `_actionable_warning`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added `test_alignment_and_anchor_headless_actionable_warnings` asserting actionable warnings and fix-it navigation callbacks.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (31 passed in 8.52s, 0 errors).
- **Status**: Completed

### Task: DeepSliceMainWindow accessibility metadata, root object names, and public ergonomic property helpers
- **Completed**: 2026-11-04
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Configured explicit `setObjectName("deepSliceMainWindow")`, `setAccessibleName("DeepSlice Desktop Main Window")`, and `setAccessibleDescription(...)` on root window.
    - Added public ergonomic property helpers: `current_page_index`, `current_page_title`, `loaded_image_count`, `selected_species`, `has_predictions`, and `is_dirty`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added unit test assertions verifying root window object name, accessibility name and description, and initial state of public property accessors.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (19 passed in 19.43s, 0 errors).
- **Status**: Completed

### Task: Replace naked QMessageBox.warning calls with accessible _actionable_warning and action callbacks in export and reporting
- **Completed**: 2026-11-04
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Replaced naked modal `QMessageBox.warning` calls in `_export_predictions`, `_generate_report`, and `_preview_report` with `self._actionable_warning`.
    - Added actionable `actions=[("Run Prediction", self._start_prediction_worker)]` callbacks so users are immediately guided to run predictions instead of hitting a dead end.
    - Added actionable `actions=[("Choose Directory...", self._browse_output_directory)]` callback for non-writable export directories.
    - Guaranteed non-blocking headless execution under offscreen test runners.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Added `test_export_and_report_headless_actionable_warnings` asserting non-blocking behavior, warning titles, messages, and actionable prediction callbacks.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py tests/test_drop_event_toast.py tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (30 passed in 9.92s, 0 errors).
- **Status**: Completed

### Task: PreferencesDialog get_values/set_values helpers, form layout bindings, and label object names
- **Completed**: 2026-10-26
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - Added explicit object names on preferences form labels (`prefSpeciesLabel`, `prefThemeLabel`, `prefOutputDirLabel`, `prefQuickNiiLabel`, `prefConsoleLabel`).
    - Added `dialog.get_values()` returning dict of current values and `dialog.set_values(values: dict)` for programmatic configuration and testing.
    - Exposed `dialog.form` and label references on dialog.
    - Updated `_save_and_apply_preferences` to consume `dialog.get_values()`.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Expanded `test_preferences_dialog_accessibility_and_reset_to_defaults` testing form layout, label object names, and `get_values`/`set_values` round-trip.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (18 passed in 9.09s, 0 errors).
- **Status**: Completed

### Task: PreferencesDialog combobox cursors, automatic acceptance persistence, and unit tests
- **Completed**: 2026-10-26
- **Changes**:
  - `DeepSlice/gui/main_window.py`:
    - `_create_preferences_dialog`: Added `PointingHandCursor` affordances on `species_combo` and `theme_combo`.
    - Factored out `_save_and_apply_preferences(dialog)` and connected `dialog.accepted` signal directly to it, ensuring preferences are persisted to `QSettings` and UI components update automatically both interactively and in headless test runs.
    - Updated `_open_preferences_dialog` to return the dialog safely under headless execution.
  - `tests/test_ui_accessibility_and_startup.py`:
    - Expanded `test_preferences_dialog_accessibility_and_reset_to_defaults` asserting cursor affordances on comboboxes, headless acceptance execution, and verified settings/state persistence.
- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (18 passed in 5.41s, 0 errors).
- **Status**: Completed



### Task: Thumbnail and Curation Context Menus Object Names, Accessibility, Actions, and Headless Non-Blocking Guards

- **Completed**: 2026-10-26

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - `_show_thumbnail_context_menu`: Added explicit `setObjectName("thumbnailContextMenu")` and `setAccessibleName("Thumbnail Item Context Menu")`; added "Preview Fullscreen" (`thumbnailPreviewAction`) and "Copy Image Path" (`thumbnailCopyPathAction`) actions; added `thumbnailRemoveAction`; added headless execution guard (`QT_QPA_PLATFORM != offscreen`) and returned menu instance for automated testing.

    - `_show_curation_context_menu`: Added explicit `setObjectName("curationContextMenu")` and `setAccessibleName("Curation Slice Context Menu")`; added explicit action object names (`curationToggleFlagAction`, `curationJumpViewerAction`, `curationSetAnchorAction`, `curationRemoveAnchorAction`, `curationEditNoteAction`, `curationCopyPathAction`, `curationShowDetailsAction`); added headless execution guard and returned menu instance.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added unit test `test_thumbnail_and_curation_context_menus_accessibility_and_actions` verifying context menu object names, action identifiers, tooltips, clipboard copy, and non-blocking headless execution.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -k "test_thumbnail_and_curation_context_menus" -v` (1 passed, 0 errors).

- **Status**: Completed



### Task: Actionable Warning Headless Non-Blocking Guard, Dialog Accessibility, Safe Console Timer, and Standard Buttons

- **Completed**: 2026-10-25

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Hardened `_actionable_warning()` with offscreen non-blocking headless execution guard (`QT_QPA_PLATFORM != offscreen`), explicit dialog object name (`actionableWarningDialog`), accessible name, close button object name (`actionableWarningCloseBtn`), action button object names (`actionableWarningActionBtn{idx}`), tooltips, and pointing hand cursors.

    - Added explicit `setAccessibleName("Preferences Dialog")` to `_create_preferences_dialog()`.

    - Added explicit `setAccessibleName("Fullscreen Section Thumbnail Preview Dialog")` to `_open_fullscreen_thumbnail_preview()`.

    - Added standard Ok button configuration with object name (`loggedErrorOkBtn`), accessible name, and tooltips in `_show_logged_error()`.

    - Added standard Ok button configuration with object name (`trainingSplitOkBtn`), accessible name, and tooltips in `_preview_training_split()`.

    - Guarded console copy confirmation timer callback against deleted widget lifecycle in `_copy_console()`.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added `test_actionable_warning_headless_and_accessibility` validating headless execution, object names, accessible names, button tooltips, and cursor shapes.

    - Expanded preferences and training split tests to assert dialog accessible names and standard button object names.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -k "test_actionable_warning_headless_and_accessibility or test_preferences_dialog or test_training_actions" -v` (3 passed in 10.83s, 0 errors).

- **Status**: Completed



### Task: Helper, Onboarding, and Guide Dialogs Standard Buttons, Accessibility, Object Names, and Tooltips

- **Completed**: 2026-10-25

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added explicit `setStandardButtons(QMessageBox.StandardButton.Ok)` across `_show_onboarding_dialog()`, `_show_whats_new_dialog()`, `_show_naming_helper()`, `_show_orientation_guide()`, `_show_direction_guide()`, `_show_ensemble_explanation()`, `_show_configuration_validation()`, and `_show_export_format_help()` ensuring dialog action buttons are instantiated consistently in headless and runtime contexts.

    - Configured explicit object names (`onboardingOkBtn`, `whatsNewOkBtn`, `namingHelperOkBtn`, `orientationGuideOkBtn`, `directionGuideOkBtn`, `ensembleExplanationOkBtn`, `configurationValidationOkBtn`, `exportFormatHelpOkBtn`), accessible names, descriptive tooltips, and pointing hand cursors across all guide and walkthrough dialog action buttons.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added `QMessageBox` import and expanded `test_helper_dialogs_accessibility_and_headless_safety` to explicitly assert standard button instantiation, object names, accessible names, tooltips, and pointing hand cursor affordances across helper, guide, and walkthrough dialogs.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -k "test_helper_dialogs" -v` (1 passed in 2.56s, 0 errors) and `python -m pytest tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (8 passed in 4.18s, 0 errors).

- **Status**: Completed



### Task: About, Shortcuts, and Hardware Health Dialogs Action Button Standard Buttons, Accessibility, Object Names, and Tooltips

- **Completed**: 2026-10-25

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added explicit `setStandardButtons(QMessageBox.StandardButton.Ok)` across `_show_about_dialog()`, `_show_shortcuts_help()`, and `_show_hardware_health()` (both GPU/CPU and lightweight fallback paths) ensuring dialog action buttons are instantiated and accessible in headless and runtime contexts.

    - Configured explicit `setObjectName` ("aboutOkBtn"), `setAccessibleName` ("Close About dialog"), `setToolTip` ("Dismiss About dialog"), and `PointingHandCursor` on About dialog action buttons.

    - Configured explicit `setObjectName` ("shortcutsOkBtn"), `setAccessibleName` ("Close Shortcuts help dialog"), `setToolTip` ("Dismiss Shortcuts dialog"), and `PointingHandCursor` on Shortcuts dialog action buttons.

    - Configured explicit `setObjectName` ("hardwareHealthOkBtn"), `setAccessibleName` ("Close Hardware Health dialog"), `setToolTip` ("Dismiss Hardware Health dialog"), and `PointingHandCursor` on Hardware Health dialog action buttons.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Expanded `test_shortcuts_and_linearity_zoom_tooltips` and `test_hardware_health_cpu_fallback` asserting button object names, accessible names, tooltips, and pointing hand cursors.

    - Added unit test `test_about_dialog_accessibility_and_button_cursors` asserting About dialog object name, window title, accessible name, and button accessibility attributes.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (16 passed in 8.23s) and `python -m pytest tests/test_top_bar_min_width.py tests/test_review_left_column_min_width.py -v` (8 passed in 7.98s, 0 errors).

- **Status**: Completed



### Task: Training Split Dialog, Action Buttons Accessibility, Object Names, Tooltips, and Non-Blocking Headless Execution

- **Completed**: 2026-10-22

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Configured explicit `setObjectName` ("previewTrainingSplitBtn"), `setToolTip` ("Preview training, validation, and test split"), and pointing hand cursor on preview split button.

    - Configured explicit `setObjectName` ("saveTrainingSplitBtn"), `setToolTip` ("Export dataset split manifest to JSON"), and pointing hand cursor on save split manifest button.

    - Configured explicit `setObjectName` ("saveTrainingMetadataBtn"), `setToolTip` ("Export training configuration metadata to JSON"), and pointing hand cursor on save training metadata button.

    - Configured explicit `setObjectName` ("trainingSplitSummaryLabel") and accessible name ("Training dataset split summary") on split summary label.

    - Replaced blocking modal `QMessageBox` popups when image lists are empty with non-blocking warning toasts in `_preview_training_split`, `_save_training_split_manifest`, and `_save_training_metadata_manifest`.

    - Wrapped split preview dialog with explicit `setObjectName` ("trainingSplitDialog"), accessible name, pointing hand cursors on buttons, headless execution guard, and returned dialog instance for automated verification.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added unit test `test_training_actions_and_split_dialog_accessibility_and_cursors` validating object names, accessible names, button tooltips, pointing hand cursors, and non-blocking headless execution.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -k "test_training_actions" -v` (1 passed in 53.47s, 0 errors).

- **Status**: Completed



### Task: Fullscreen Thumbnail Preview Dialog and Logged Error Dialog Accessibility, Object Names, and Pointer Cursors

- **Completed**: 2026-10-22

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added explicit `setObjectName` ("previewCloseBtn"), `setAccessibleName` ("Close fullscreen preview"), `setToolTip` ("Close preview dialog"), and `PointingHandCursor` to the close button in `_open_fullscreen_thumbnail_preview()`.

    - Guarded fullscreen preview dialog execution with headless safety check (`if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and "PYTEST_CURRENT_TEST" not in os.environ:`) and returned the dialog for inspection and unit testing.

    - Added explicit `setObjectName` ("loggedErrorDialog"), accessible name, object names ("loggedErrorCopyBtn", "loggedErrorAutoFixBtn"), and `PointingHandCursor` across all buttons in `_show_logged_error()`.

    - Added headless safety guard to confidence details in `_show_curation_context_menu()`.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added unit test `test_fullscreen_preview_and_logged_error_accessibility_and_cursors` validating object names, accessible names, pointing hand cursors, and non-blocking headless execution.

- **Verification**: Verified headlessly with `python -m pytest tests/test_ui_accessibility_and_startup.py -v` (14 passed in 42.48s) and `pytest tests/test_review_left_column_min_width.py tests/test_top_bar_min_width.py -v` (8 passed in 7.63s, 0 errors).

- **Status**: Completed



### Task: Preferences Dialog Button Pointer Cursors, Button Tooltips, and Headless Execution Safety Guard

- **Completed**: 2026-10-22

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added `PointingHandCursor` pointer affordances across `output_dir_browse`, `quicknii_browse`, `console_always_visible`, `ok_btn`, `cancel_btn`, and `reset_btn` in `_create_preferences_dialog()`.

    - Added descriptive tooltips and explicit object names to `ok_btn` (`prefOkBtn`, "Save preference changes and apply") and `cancel_btn` (`prefCancelBtn`, "Discard preference changes").

    - Exposed `ok_btn` and `cancel_btn` references on dialog instance for testability.

    - Added headless execution safety guard to `_open_preferences_dialog()` (`if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and "PYTEST_CURRENT_TEST" not in os.environ:`), returning the dialog without modal blocking during automated test runs.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Expanded `test_preferences_dialog_accessibility_and_reset_to_defaults` to assert pointing hand cursors across all buttons, object names, button tooltips, and non-blocking headless execution.

- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (13 passed in 8.27s) and `pytest tests/test_review_left_column_min_width.py tests/test_top_bar_min_width.py -v` (8 passed in 8.02s, 0 errors).

- **Status**: Completed



### Task: Make Review Column Sections Collapsible and Group Flagging and Note Controls

- **Completed**: 2026-10-20

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Made `controls_group` ("Propagation / Curation") and `anchor_group` ("Anchor Alignment") checkable and collapsible with `setCheckable(True)` and connected toggled callbacks to show/hide child widgets dynamically, reclaiming significant vertical screen real estate.

    - Grouped `slice_note_edit` and `save_slice_note_button` side by side into a compact horizontal layout (`note_row`).

    - Grouped `apply_bad_sections_button`, `detect_outliers_button`, and `reset_flags_button` into a responsive `FlowLayout` (`flags_action_row`) so they align horizontally when space permits and wrap without truncating on narrow screens.

  - `tests/test_review_left_column_min_width.py`:

    - Added unit tests `test_flags_action_row_is_flow_layout` and `test_collapsible_groups_toggle_contents` validating responsive layout wrapping and collapsible group visibility toggling.

- **Verification**: Verified headlessly with `pytest tests/test_review_left_column_min_width.py tests/test_ui_accessibility_and_startup.py -v` (19 passed in 15.49s with 0 errors).

- **Status**: Completed



### Task: Enhance Top Bar and Review Column Responsive Layout Ergonomics and Code Cleanliness

- **Completed**: 2026-10-20

- **Changes**:

  - `DeepSlice/gui/flow_layout.py`:

    - Removed unused imports `QSizePolicy` and `QWidgetItem`.

  - `DeepSlice/gui/main_window.py`:

    - Removed unused import `QCloseEvent`.

  - `tests/test_top_bar_min_width.py`:

    - Standardized minimum width and responsive single/multi-row wrapping assertions to be platform-independent across operating system font DPI variations.

  - `tests/test_review_left_column_min_width.py`:

    - Bound left column minimum width assertions relative to layout size hints to ensure cross-platform test stability.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added module-level `from PySide6.QtCore import Qt` import, resolving undefined name issues when running accessibility tests in isolation.

- **Verification**: Verified headlessly with Python 3.12: `pytest tests/test_review_left_column_min_width.py tests/test_top_bar_min_width.py tests/test_flow_layout.py tests/test_plot_theme.py -v` (12 passed in 8.37s, 0 errors).

- **Status**: Completed



### Task: Standardize Helper Dialogs Accessibility, Object Names, Button Cursors, and Headless Execution Safety

- **Completed**: 2026-10-14

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Standardized helper dialogs (`_show_onboarding_dialog`, `_show_whats_new_dialog`, `_show_hardware_health`, `_show_about_dialog`, `_show_shortcuts_help`, `_show_naming_helper`, `_show_orientation_guide`, `_show_direction_guide`, `_show_ensemble_explanation`, `_show_configuration_validation`, and `_show_export_format_help`) to construct `QMessageBox` with explicit `objectName`, `accessibleName`, `PointingHandCursor` across all buttons, and non-blocking headless execution guards (`QT_QPA_PLATFORM != "offscreen"` and `"PYTEST_CURRENT_TEST" not in os.environ`).

    - Added headless safety guard to `_open_context_help()` preventing unintentional browser launches during automated runs.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Expanded `test_helper_dialogs_accessibility_and_headless_safety` to validate all helper dialogs, object names, accessible names, button cursors, and context help headless safety.

    - Updated `test_hardware_health_cpu_fallback` to assert directly on the accessible dialog instance and diagnostic fallback content.

- **Verification**: Verified headlessly with Python 3.12: `pytest tests/test_ui_accessibility_and_startup.py -v` (12 passed in 9.23s) and full test suite `pytest tests/ -q` (349 passed in 30.50s, 0 errors).

- **Status**: Completed



### Task: Enhance Linearity Plot Zoom and Preferences Keyboard Shortcuts, Shortcut Tooltips, and Shortcuts Cheat Sheet Dialog

- **Completed**: 2026-10-14

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added `QShortcut` bindings for `Ctrl+,` (Open Preferences), `Ctrl++` / `Ctrl+=` (Zoom In Plot), `Ctrl+-` (Zoom Out Plot), and `Ctrl+0` (Fit Linearity Plot View).

    - Updated tooltips across `preferences_button`, `zoom_fit_button`, `zoom_in_button`, and `zoom_out_button` to explicitly display shortcut keys (`Ctrl+,`, `Ctrl+0`, `Ctrl++`, `Ctrl+-`).

    - Updated `_show_shortcuts_help()` cheat sheet text to document the new navigation and linearity plot zoom shortcuts.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added unit test `test_shortcuts_and_linearity_zoom_tooltips` validating shortcut key registration, button tooltip hints, and shortcuts help dialog documentation.

- **Verification**: Verified headlessly with Python 3.12: `pytest tests/test_ui_accessibility_and_startup.py -v` (12 passed, 0 failures in 8.47s).

- **Status**: Completed



### Task: Enhance Curation Page Controls with Accessibility, Tooltips, Hand Cursors, and Usability Ergonomics

- **Completed**: 2026-10-06

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added screen-reader accessible names, descriptive tooltips, object names, and `PointingHandCursor` across all Curation page controls: `curation_prev_button`, `curation_next_button`, `curation_select_all_btn`, `curation_deselect_all_btn`, `confidence_filter_combo`, `slice_flag_list`, `apply_bad_sections_button`, `apply_manual_order_button`, `move_slice_up_button`, `move_slice_down_button`, `detect_outliers_button`, `reset_flags_button`, `auto_flag_low_conf_button`, `interpolate_bad_depth_button`, `toggle_current_flag_button`, `slice_note_edit`, `save_slice_note_button`, `normalize_angles_button`, `enforce_order_button`, `enforce_spacing_button`, `ml_spin`, `dv_spin`, `apply_manual_angles_button`, `undo_button`, `redo_button`, `anchor_depth_spin`, `set_anchor_button`, `remove_anchor_button`, `apply_anchor_interpolation_button`, `clear_anchor_button`, `anchor_list`, `zoom_fit_button`, `zoom_in_button`, `zoom_out_button`, `enable_atlas_preview_checkbox`, `atlas_volume_combo`, `enable_blend_overlay_checkbox`, `blend_slider`, `before_after_toggle`, `loupe_toggle`, `atlas_flip_x_checkbox`, `atlas_flip_y_checkbox`, `atlas_rotate_combo`, `atlas_scale_slider`, `atlas_offset_x_slider`, `atlas_offset_y_slider`, and `confidence_panel_toggle`.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added `test_curation_page_controls_accessibility_and_cursors` validating accessibility attributes, pointing hand cursors, and control metadata.

- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (11 passed, 0 failures in 8.75s).

- **Status**: Completed



### Task: Enhance Ingestion and Configuration Page Controls with Accessibility, Tooltips, and Hand Cursors

- **Completed**: 2026-10-03

- **Changes**:

  - `DeepSlice/gui/main_window.py`:

    - Added screen-reader accessible names, descriptive tooltips, and `PointingHandCursor` shapes across all Ingestion page controls: `add_folder_button`, `add_files_button`, `clear_images_button`, `enable_section_numbers_checkbox`, `naming_helper_button`, `legacy_parsing_checkbox`, `orientation_combo`, `orientation_guide_button`, `slice_count_label`, `file_size_summary_label`, `ingestion_summary_banner`, `ingestion_warning_label`, `index_table`, `thumbnail_sort_combo`, and `thumbnail_filter_edit`.

    - Added screen-reader accessible names, descriptive tooltips, and `PointingHandCursor` shapes across Configuration page controls: `mouse_radio`, `rat_radio`, `auto_thickness_checkbox`, `thickness_spin`, `suggest_thickness_button`, `direction_override_combo`, `direction_guide_button`, `ensemble_checkbox`, `ensemble_help_button`, `secondary_model_checkbox`, `legacy_from_config_checkbox`, `training_mixed_precision_checkbox`, `preview_training_split_button`, `save_training_split_button`, `save_training_metadata_button`, `validate_configuration_button`, and `tech_toggle`.

  - `tests/test_ui_accessibility_and_startup.py`:

    - Added `test_ingestion_and_configuration_pages_accessibility_and_cursors` validating accessibility attributes, pointing hand cursors, and control metadata.

- **Verification**: Verified headlessly with `pytest tests/test_ui_accessibility_and_startup.py -v` (10 passed, 0 failures in 5.27s).

- **Status**: Completed



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

- ~~Review page atlas control rows overflow~~ done 2026-10-18 (FlowLayout). ~~TopBar 1635px minimum width~~ done 2026-10-19 (actions in FlowLayout; min ~611px).

- ~~Review page left column: 9 stacked full-width buttons; group under collapsible sections~~ done 2026-10-20 (controls/anchor groups collapsible; note and flag actions grouped).

- Screenshot and fix populated states (images loaded, predictions present) and all dialogs (Preferences, About, Shortcuts, Hardware Health, Training) at 2 sizes (priority 3).

- ~~Locked sidebar steps tooltip~~ already implemented (2026-10-17 verified).

- ~~Plot colors themed for light theme~~ done 2026-10-17 (PLOT_PALETTES).

