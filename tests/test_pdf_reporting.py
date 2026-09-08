"""Tests for PDF report generation, angle metrics rendering, and UI defaults."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from reportlab.pdfgen import canvas

from DeepSlice.gui.reporting import generate_pdf_report
from DeepSlice.gui.state import DeepSliceAppState


def _sample_predictions() -> pd.DataFrame:
    rows = []
    for idx in range(6):
        rows.append(
            {
                "Filenames": f"brain_s{idx + 1:03d}.png",
                "nr": (idx + 1) * 5,
                "height": 640,
                "width": 1024,
                "ox": 480.0 + idx,
                "oy": 320.0 - (idx * 8.0),
                "oz": 332.0 + (idx * 0.5),
                "ux": -505.0 + (idx * 0.2),
                "uy": 0.72 + (idx * 0.01),
                "uz": 8.5 + (idx * 0.1),
                "vx": -8.0 - (idx * 0.1),
                "vy": 1.30 + (idx * 0.01),
                "vz": -380.0 - (idx * 0.3),
                "bad_section": idx == 5,
            }
        )
    return pd.DataFrame(rows)


class TestSummaryMetrics:
    def test_summary_metrics_when_predictions_none(self):
        state = DeepSliceAppState()
        state.predictions = None

        metrics = state.summary_metrics()
        assert metrics["processed"] == 0
        assert metrics["excluded"] == 0
        assert metrics["slice_count"] == 0
        assert metrics["mean_angular_deviation"] == 0.0
        assert metrics["mean_dv"] == 0.0
        assert metrics["mean_ml"] == 0.0
        assert metrics["std_dv"] == 0.0
        assert metrics["std_ml"] == 0.0

    def test_summary_metrics_with_predictions(self):
        state = DeepSliceAppState(species="mouse")
        state.predictions = _sample_predictions()

        metrics = state.summary_metrics()
        assert metrics["slice_count"] == 6
        assert metrics["excluded"] == 1
        assert metrics["processed"] == 5
        assert isinstance(metrics["mean_angular_deviation"], float)
        assert isinstance(metrics["mean_dv"], float)
        assert isinstance(metrics["mean_ml"], float)
        assert isinstance(metrics["std_dv"], float)
        assert isinstance(metrics["std_ml"], float)


class TestPdfReporting:
    def test_generate_pdf_report_renders_quantitative_angle_metrics(self, tmp_path: Path):
        pdf_path = tmp_path / "test_report.pdf"
        summary = {
            "slice_count": 12,
            "processed": 11,
            "excluded": 1,
            "mean_angular_deviation": 1.234,
            "mean_dv": -2.50,
            "mean_ml": 3.75,
            "std_dv": 0.85,
            "std_ml": 1.12,
        }
        options = {
            "include_stats": True,
            "include_plot": False,
            "include_images": False,
            "include_angles": True,
        }

        captured_strings: list[str] = []
        original_canvas = canvas.Canvas

        class SpyingCanvas(original_canvas):
            def drawString(self, x, y, text):
                captured_strings.append(text)
                super().drawString(x, y, text)

        with patch("reportlab.pdfgen.canvas.Canvas", SpyingCanvas):
            generate_pdf_report(str(pdf_path), summary, options)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 500

        # Verify angle metrics were rendered quantitatively
        assert "Angle Metrics" in captured_strings
        assert "Mean angular deviation: 1.234 deg" in captured_strings
        assert "Dorsoventral (DV) angle: -2.50 deg (+/- 0.85 deg)" in captured_strings
        assert "Mediolateral (ML) angle: 3.75 deg (+/- 1.12 deg)" in captured_strings

        # Verify boilerplate placeholders are absent
        for s in captured_strings:
            assert "Placeholder" not in s
            assert "future versions" not in s

        # Sample images was False by default
        assert "Sample Section Alignments" not in captured_strings

    def test_generate_pdf_report_sample_images_opt_in(self, tmp_path: Path):
        pdf_path = tmp_path / "test_report_images.pdf"
        summary = {
            "slice_count": 5,
            "processed": 5,
            "excluded": 0,
            "mean_angular_deviation": 0.5,
            "mean_dv": 0.0,
            "mean_ml": 0.0,
            "std_dv": 0.0,
            "std_ml": 0.0,
        }
        options = {
            "include_stats": True,
            "include_plot": False,
            "include_images": True,
            "include_angles": False,
        }

        captured_strings: list[str] = []
        original_canvas = canvas.Canvas

        class SpyingCanvas(original_canvas):
            def drawString(self, x, y, text):
                captured_strings.append(text)
                super().drawString(x, y, text)

        with patch("reportlab.pdfgen.canvas.Canvas", SpyingCanvas):
            generate_pdf_report(str(pdf_path), summary, options)

        assert pdf_path.exists()
        assert "Sample Section Alignments" in captured_strings
        assert "Section alignment previews are not embedded in this report format." in captured_strings
        assert "Angle Metrics" not in captured_strings

        # Confirm no "(Placeholder)" text
        for s in captured_strings:
            assert "(Placeholder)" not in s

    def test_generate_pdf_report_without_angle_data(self, tmp_path: Path):
        pdf_path = tmp_path / "test_report_no_angles.pdf"
        summary = {
            "slice_count": 0,
            "processed": 0,
            "excluded": 0,
        }
        options = {
            "include_stats": False,
            "include_plot": False,
            "include_images": False,
            "include_angles": True,
        }

        captured_strings: list[str] = []
        original_canvas = canvas.Canvas

        class SpyingCanvas(original_canvas):
            def drawString(self, x, y, text):
                captured_strings.append(text)
                super().drawString(x, y, text)

        with patch("reportlab.pdfgen.canvas.Canvas", SpyingCanvas):
            generate_pdf_report(str(pdf_path), summary, options)

        assert pdf_path.exists()
        assert "Angle Metrics" in captured_strings
        assert "Angle distributions were not computed for this session." in captured_strings


class TestMainWindowPdfDefaults:
    def test_pdf_checkboxes_defaults_in_main_window(self):
        from PySide6.QtWidgets import QApplication
        from DeepSlice.gui.main_window import DeepSliceMainWindow

        app = QApplication.instance() or QApplication([])
        win = DeepSliceMainWindow()
        try:
            assert win.pdf_include_stats.isChecked() is True
            assert win.pdf_include_plot.isChecked() is True
            assert win.pdf_include_images.isChecked() is False
            assert win.pdf_include_angles.isChecked() is True

            # Verify updated tooltip on pdf_include_angles
            assert "angle metrics" in win.pdf_include_angles.toolTip().lower()
        finally:
            win.close()
