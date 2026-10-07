"""Headless tests for the first-time-user guidance added to the main window."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from DeepSlice.gui.main_window import DeepSliceMainWindow


@pytest.fixture()
def win():
    QApplication.instance() or QApplication([])
    w = DeepSliceMainWindow()
    yield w
    w.close()
    w.deleteLater()


def test_step_names_are_plain_language_and_numbered(win):
    assert win.STEP_LABELS[0].endswith("Add images")
    assert all(label[0].isdigit() for label in win.STEP_LABELS)
    assert len(win.STEP_GUIDE) == len(win.STEP_LABELS)


def test_banner_explains_first_step_and_locks_next(win):
    assert "Step 1 of 5" in win.guide_title_label.text()
    assert "Nothing loaded yet" in win.guide_text_label.text()
    assert not win.guide_next_button.isEnabled()
    assert "image" in win.guide_next_button.toolTip().lower()
    # Locked steps say why in their tooltip.
    assert win.step_list.item(3).toolTip().startswith("Locked")


def test_banner_follows_step_and_next_button_unlocks(win, tmp_path):
    win.state.image_paths = [str(tmp_path / "a_s001.png")]
    win._refresh_step_states()
    assert win.guide_next_button.isEnabled()
    assert "Settings" in win.guide_next_button.text()
    assert "1 image loaded" in win.guide_text_label.text()
    win.guide_next_button.click()
    assert win.stack.currentIndex() == 1
    assert "Step 2 of 5" in win.guide_title_label.text()


def test_last_step_hides_next_button(win):
    win.stack.setCurrentIndex(4)
    assert win.guide_next_button.isHidden()


def test_advanced_options_hidden_until_toggled(win):
    assert win.advanced_options_container.isHidden()
    assert not win.advanced_options_toggle.isChecked()
    win.advanced_options_toggle.setChecked(True)
    assert not win.advanced_options_container.isHidden()
    assert win.advanced_options_toggle.text().startswith("v ")


def test_run_alignment_says_why_it_is_disabled(win):
    win._update_run_button_state()
    assert not win.run_alignment_button.isEnabled()
    assert not win.run_blocker_label.isHidden()
    assert "unavailable" in win.run_blocker_label.text()
    assert win.run_alignment_button.toolTip() == win.run_blocker_label.text()


def test_top_bar_utility_buttons_show_their_names(win):
    from PySide6.QtCore import Qt

    for btn in (win.shortcut_help_button, win.preferences_button, win.about_button):
        assert btn.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
        assert btn.text()
    assert win.hardware_mode_label.text().startswith("Runs on:")
