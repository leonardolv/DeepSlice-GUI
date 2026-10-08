"""Headless tests for the second usability pass (hierarchy, sizing, errors, export)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox

from DeepSlice.gui.main_window import DeepSliceMainWindow


@pytest.fixture()
def win():
    QApplication.instance() or QApplication([])
    w = DeepSliceMainWindow()
    # The window persists some paths to the real QSettings; put them back afterwards.
    saved = {k: w._settings.value(k, "") for k in ("quicknii_path", "export_directory")}
    yield w
    for key, value in saved.items():
        w._settings.setValue(key, value)
    w.close()
    w.deleteLater()


def _predictions(names):
    n = len(names)
    return pd.DataFrame(
        {
            "Filenames": names,
            "ox": [300.0] * n, "oy": [200.0 + 30 * i for i in range(n)], "oz": [250.0] * n,
            "ux": [400.0] * n, "uy": [0.0] * n, "uz": [0.0] * n,
            "vx": [0.0] * n, "vy": [0.0] * n, "vz": [-300.0] * n,
            "width": [256] * n, "height": [256] * n,
        }
    )


def test_sidebar_and_banner_follow_a_programmatic_page_change(win):
    win.state.image_paths = ["a_s001.png", "a_s002.png"]
    win.state.predictions = _predictions(["a_s001.png", "a_s002.png"])
    win._refresh_step_states()
    win.stack.setCurrentIndex(3)
    assert win.step_list.currentRow() == 3
    assert "Step 4 of 5" in win.guide_title_label.text()


def test_banner_next_button_keeps_the_ampersand(win):
    win.state.image_paths = ["a_s001.png"]
    win.state.predictions = _predictions(["a_s001.png"])
    win._refresh_step_states()
    win.step_list.setCurrentRow(2)
    # "&" alone would be swallowed as a mnemonic marker ("Review _fix").
    assert "&&" in win.guide_next_button.text()


def test_alt_arrows_move_between_unlocked_steps(win):
    win.state.image_paths = ["a_s001.png"]
    win._refresh_step_states()
    win._go_to_next_step()
    assert win.stack.currentIndex() == 1
    win._go_to_previous_step()
    assert win.stack.currentIndex() == 0
    win._go_to_previous_step()  # already first: no-op
    assert win.stack.currentIndex() == 0


def test_window_minimum_size_fits_a_laptop_screen(win):
    hint = win.minimumSizeHint()
    assert hint.width() <= 1366 and hint.height() <= 768
    assert win.minimumWidth() <= 1000 and win.minimumHeight() <= 650
    # Every page scrolls instead of forcing the window larger.
    assert len(win.page_scroll_areas) == win.stack.count() == 5


def test_top_bar_collapses_to_icons_when_narrow(win):
    win.resize(1000, 700)
    win.show()
    QApplication.processEvents()
    win._update_top_bar_compact()
    assert win._top_bar_stage == 2
    assert win.save_session_button.text() == ""
    assert win.save_session_button.toolTip()  # still discoverable
    win.resize(1900, 900)
    QApplication.processEvents()
    win._update_top_bar_compact()
    assert win._top_bar_stage == 0
    assert win.save_session_button.text() == "Save Session"


def test_curation_has_one_primary_per_group_and_danger_reset(win):
    assert win.reset_flags_button.property("role") == "danger"
    for name in ("detect_outliers_button", "auto_flag_low_conf_button", "undo_button"):
        assert getattr(win, name).property("role") == "secondary"
    # primary actions carry no role -> they keep the filled blue style
    assert win.apply_bad_sections_button.property("role") is None
    assert win.normalize_angles_button.property("role") is None
    assert "Ctrl+Enter" in win.apply_bad_sections_button.text()
    assert "(F)" in win.toggle_current_flag_button.text()
    assert win.curation_select_all_btn.text() == "Flag all"
    assert win.curation_prev_button.toolButtonStyle() == Qt.ToolButtonTextBesideIcon


def test_checkboxes_get_a_visible_indicator_style(win):
    sheet = win.styleSheet()
    assert "QCheckBox::indicator" in sheet
    assert 'QPushButton[role="secondary"]' in sheet
    assert win.findChildren(QCheckBox)


def test_index_table_cells_are_not_scrambled_by_sorting(win, tmp_path):
    from PIL import Image

    paths = []
    for i in (3, 1, 2):
        p = tmp_path / f"brain_s{i:03d}.png"
        Image.new("RGB", (16, 16)).save(p)
        paths.append(str(p))
    win.state.add_images(paths)
    win._refresh_all_views()
    table = win.index_table
    assert table.rowCount() == 3
    for row in range(3):
        name = table.item(row, 0).text()
        idx = table.item(row, 1)
        status = table.item(row, 2)
        assert idx is not None and status is not None
        assert str(idx.data(Qt.DisplayRole)) == str(int(name.split("_s")[1][:3]))
        assert status.text() == "OK"
    assert table.isSortingEnabled()


def test_ingestion_summary_pluralises_and_flags_warnings(win, tmp_path):
    from PIL import Image

    p = tmp_path / "only_s001.png"
    Image.new("RGB", (16, 16)).save(p)
    win.state.add_images([str(p)])
    win._refresh_all_views()
    text = win.ingestion_summary_banner.text()
    assert text.startswith("1 file -")
    assert "2 warnings" in text  # too few sections for spacing + angle propagation
    assert "1 warnings" not in text


def test_drop_area_is_clickable_and_keyboard_operable(win, monkeypatch):
    # The real slot opens a file dialog; never let a test block on one.
    from DeepSlice.gui import main_window as mw

    opened = []
    monkeypatch.setattr(
        mw.QFileDialog, "getOpenFileNames", lambda *a, **k: (opened.append(1) or ([], ""))
    )
    hits = []
    win.drop_area.clicked.connect(lambda: hits.append(1))
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    win.drop_area.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert hits == [1]
    assert opened == [1]  # clicking the drop area opens the same picker as "Add Files"
    assert win.drop_area.focusPolicy() == Qt.StrongFocus


def test_prediction_headline_tracks_the_run(win):
    assert "Nothing to align" in win.prediction_status_headline.text()
    win.state.image_paths = ["a_s001.png", "a_s002.png"]
    win._update_run_button_state()
    assert "Ready to align 2 images" in win.prediction_status_headline.text()
    win._on_prediction_progress(1, 2, "primary")
    assert "section 1 of 2" in win.prediction_status_headline.text()
    assert win.prediction_progress_label.text() == "Sections done: 1 of 2"
    assert "Phase" not in win.prediction_phase_label.text()
    assert win.prediction_compare_checkbox.isHidden()


def test_console_toggle_text_follows_state(win):
    assert win.console_toggle.text() == "Show detailed log"
    win.console_toggle.setChecked(True)
    assert win.console_toggle.text() == "Hide detailed log"


def test_plain_error_message_leads_with_reason_and_next_steps():
    text = DeepSliceMainWindow._plain_error_message(
        "Alignment prediction task failed",
        "Traceback (most recent call last):\n  File x\nMemoryError: out of memory",
        {"recommendations": ["Copy the generated error report."]},
        "/tmp/log.txt",
    )
    assert "What went wrong: MemoryError: out of memory" in text
    assert "What you can try" in text and "lower the batch size" in text
    assert "/tmp/log.txt" in text
    assert "Traceback" not in text


def test_plain_error_message_has_a_fallback_tip():
    text = DeepSliceMainWindow._plain_error_message("Export failed", "", {}, "/x")
    assert "No further details" in text
    assert "Copy Last Error" in text


def test_quicknii_open_is_gated_on_a_json_export_and_explains_why(win, tmp_path):
    win.quicknii_path_edit.setText("")
    win._update_quicknii_controls()
    assert not win.open_quicknii_button.isEnabled()
    assert "Export a JSON file first" in win.quicknii_status_label.text()

    missing = tmp_path / "nope" / "QuickNII"
    win.quicknii_path_edit.setText(str(missing))
    assert "File not found" in win.quicknii_status_label.text()

    program = tmp_path / "QuickNII"
    program.write_text("x")
    base = tmp_path / "out"
    (tmp_path / "out.json").write_text("{}")
    win.last_export_basepath = str(base)
    win.quicknii_path_edit.setText(str(program))
    assert "QuickNII found" in win.quicknii_status_label.text()
    assert win.open_quicknii_button.isEnabled()


def test_export_shows_a_persistent_result_line(win, tmp_path):
    win._show_export_result(str(tmp_path / "Res"), "json", 2)
    assert not win.export_result_label.isHidden()
    assert "Res.json" in win.export_result_label.text()
    assert "Res.csv" in win.export_result_label.text()


def test_new_session_forgets_the_last_export_and_leaves_locked_pages(win, monkeypatch, tmp_path):
    win.state.image_paths = ["a_s001.png"]
    win.state.predictions = _predictions(["a_s001.png"])
    win._refresh_step_states()
    win.last_export_basepath = str(tmp_path / "x")
    win._show_export_result(str(tmp_path / "x"), "json", 2)
    win.step_list.setCurrentRow(4)
    win.state.is_dirty = False
    win._reset_session()
    assert win.last_export_basepath is None
    assert win.export_result_label.isHidden()
    assert win.stack.currentIndex() == 0
    assert win.step_list.currentRow() == 0


def test_atlas_adjustments_sit_behind_a_toggle(win):
    assert win.atlas_adjust_container.isHidden()
    win.atlas_adjust_toggle.setChecked(True)
    assert not win.atlas_adjust_container.isHidden()
    # the controls still exist and work while collapsed
    win.atlas_adjust_toggle.setChecked(False)
    win.atlas_flip_x_checkbox.setChecked(True)
    assert win.atlas_flip_x_checkbox.isChecked()
