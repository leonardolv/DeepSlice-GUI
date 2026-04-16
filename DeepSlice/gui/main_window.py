from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent, QColor, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QMenu,
    QMessageBox,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QRadioButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)
from PIL import Image

from ..error_auto_fix import ErrorAutoFixer
from ..error_logging import (
    build_error_report,
    configure_error_logging,
    get_logger,
    log_error_text,
    read_error_log_tail,
)
from . import reporting
from .state import DeepSliceAppState, SUPPORTED_IMAGE_FORMATS
from .workers import FunctionWorker


class DropArea(QFrame):
    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DropArea")
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        title = QLabel("Drag and Drop Images or Folders")
        title.setObjectName("DropTitle")
        subtitle = QLabel("Supports JPG, PNG, TIFF. Folder drops recurse into subfolders.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("DropSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        dropped_paths = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if local_path:
                dropped_paths.append(local_path)
        self.pathsDropped.emit(dropped_paths)
        event.acceptProposedAction()


class SliceGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor("#11161D"))
        self._zoom = 0
        self._scale_bar_item = None
        self._pixel_spacing_um = None

    def clear_with_text(self, message: str):
        self._scene.clear()
        self._scale_bar_item = None
        text_item = QGraphicsTextItem(message)
        text_item.setDefaultTextColor(QColor("#C4CBD3"))
        text_item.setFont(QFont("Segoe UI", 10))
        self._scene.addItem(text_item)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())

    def _draw_scale_bar(self):
        if self._scale_bar_item is not None:
            self._scene.removeItem(self._scale_bar_item)
            self._scale_bar_item = None
            
        if self._pixel_spacing_um is None:
            return
            
        rect = self.sceneRect()
        width = rect.width()
        
        target_um = max(100.0, 10 ** np.floor(np.log10(width * self._pixel_spacing_um * 0.2)))
        bar_width_px = target_um / self._pixel_spacing_um
        
        if bar_width_px > width * 0.5:
            return
            
        from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItemGroup
        
        group = QGraphicsItemGroup()
        
        bar = QGraphicsRectItem(0, 0, bar_width_px, 4)
        bar.setBrush(QColor("#F7FBFF"))
        bar.setPen(Qt.NoPen)
        group.addToGroup(bar)
        
        text = QGraphicsTextItem(f"{int(target_um)} µm")
        text.setDefaultTextColor(QColor("#F7FBFF"))
        text.setFont(QFont("Segoe UI", 10, QFont.Bold))
        text_rect = text.boundingRect()
        text.setPos(bar_width_px / 2 - text_rect.width() / 2, -text_rect.height())
        group.addToGroup(text)
        
        margin = 20
        group.setPos(rect.right() - bar_width_px - margin, rect.bottom() - margin)
        
        self._scale_bar_item = group
        self._scene.addItem(self._scale_bar_item)

    def set_image(
        self,
        image_path: Optional[str],
        overlay_lines: Optional[List[str]] = None,
        border_color: Optional[QColor] = None,
        pixel_spacing_um: Optional[float] = None,
    ):
        self._scene.clear()
        self._scale_bar_item = None
        self._zoom = 0
        self._pixel_spacing_um = pixel_spacing_um

        if image_path is None or not os.path.exists(image_path):
            self.clear_with_text("Image preview unavailable")
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.clear_with_text("Unable to load image preview")
            return

        pix_item = self._scene.addPixmap(pixmap)

        if border_color is not None:
            pen_width = 4
            self._scene.addRect(
                pix_item.boundingRect(),
                border_color,
            )

        if overlay_lines:
            overlay_text = "\n".join(overlay_lines)
            text_item = QGraphicsTextItem(overlay_text)
            text_item.setDefaultTextColor(QColor("#52E5FF"))
            text_item.setFont(QFont("Segoe UI", 10))
            text_item.setPos(12, 12)
            self._scene.addItem(text_item)

        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._draw_scale_bar()

    def set_array_image(
        self,
        image_array: np.ndarray,
        overlay_lines: Optional[List[str]] = None,
        border_color: Optional[QColor] = None,
    ):
        self._scene.clear()
        self._zoom = 0

        if image_array is None or image_array.size == 0:
            self.clear_with_text("Atlas preview unavailable")
            return

        data = np.asarray(image_array)
        if data.ndim not in {2, 3}:
            self.clear_with_text("Array preview expects a 2D or RGB image")
            return

        if data.ndim == 2:
            normalized = np.nan_to_num(
                data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            if normalized.dtype != np.uint8:
                min_value = float(np.min(normalized))
                max_value = float(np.max(normalized))
                if max_value <= min_value:
                    normalized = np.zeros_like(normalized, dtype=np.uint8)
                else:
                    normalized = (
                        (normalized - min_value) / (max_value - min_value) * 255.0
                    ).astype(np.uint8)

            normalized = np.ascontiguousarray(normalized)
            height, width = normalized.shape
            image = QImage(
                normalized.data,
                width,
                height,
                normalized.strides[0],
                QImage.Format_Grayscale8,
            ).copy()
        else:
            normalized = np.nan_to_num(
                data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            if normalized.shape[2] != 3:
                self.clear_with_text("RGB array preview expects exactly 3 channels")
                return
            if normalized.dtype != np.uint8:
                min_value = float(np.min(normalized))
                max_value = float(np.max(normalized))
                if max_value <= min_value:
                    normalized = np.zeros_like(normalized, dtype=np.uint8)
                else:
                    normalized = (
                        (normalized - min_value) / (max_value - min_value) * 255.0
                    ).astype(np.uint8)
            normalized = np.ascontiguousarray(normalized)
            height, width, _ = normalized.shape
            image = QImage(
                normalized.data,
                width,
                height,
                normalized.strides[0],
                QImage.Format_RGB888,
            ).copy()

        pixmap = QPixmap.fromImage(image)
        pix_item = self._scene.addPixmap(pixmap)

        if border_color is not None:
            self._scene.addRect(pix_item.boundingRect(), border_color)

        if overlay_lines:
            overlay_text = "\n".join(overlay_lines)
            text_item = QGraphicsTextItem(overlay_text)
            text_item.setDefaultTextColor(QColor("#52E5FF"))
            text_item.setFont(QFont("Segoe UI", 10))
            text_item.setPos(12, 12)
            self._scene.addItem(text_item)

        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15
            self._zoom += 1
        else:
            zoom_factor = 1.0 / 1.15
            self._zoom -= 1

        if self._zoom < -5:
            self._zoom = -5
            return
        self.scale(zoom_factor, zoom_factor)


class DeepSliceMainWindow(QMainWindow):
    STEP_LABELS = [
        "Ingestion",
        "Configuration",
        "Prediction",
        "Curation",
        "Export",
    ]

    def __init__(self):
        super().__init__()
        self.error_log_path = configure_error_logging()
        self._logger = get_logger("gui.main_window")
        self._error_autofixer = ErrorAutoFixer()
        self._last_error_report = ""
        self._last_error_context = ""
        self._last_error_text = ""
        self._last_error_analysis = None

        self.state = DeepSliceAppState()
        self._session_base_text = "Session: New"
        self._setup_shortcuts()
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers = []
        self.last_export_basepath: Optional[str] = None
        self._linearity_payload = None
        self._atlas_request_token = 0
        self._latest_atlas_slice: Optional[np.ndarray] = None
        self._latest_atlas_meta: Optional[dict] = None

        self.setWindowTitle("DeepSlice Desktop")
        self.resize(1600, 980)

        self._build_ui()
        self._apply_theme()
        self._update_hardware_mode_label()
        self._refresh_all_views()
        self._logger.info("DeepSlice GUI initialized. Error log: %s", self.error_log_path)

    def _track_worker(self, worker: FunctionWorker):
        self.active_workers.append(worker)
        worker.signals.finished.connect(
            lambda _result, tracked_worker=worker: self._release_worker(tracked_worker)
        )
        worker.signals.error.connect(
            lambda _error, tracked_worker=worker: self._release_worker(tracked_worker)
        )

    def _release_worker(self, worker: FunctionWorker):
        try:
            self.active_workers.remove(worker)
        except ValueError:
            pass

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._build_top_bar())

        body_split = QSplitter(Qt.Horizontal)
        body_split.setHandleWidth(6)

        self.sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sidebar_header = QHBoxLayout()
        self.completion_label = QLabel("0% Complete")
        self.completion_label.setObjectName("CompletionLabel")
        self.collapse_sidebar_button = QPushButton("<<")
        self.collapse_sidebar_button.setFixedWidth(30)
        self.collapse_sidebar_button.clicked.connect(self._toggle_sidebar)
        self.sidebar_header.addWidget(self.completion_label)
        self.sidebar_header.addStretch()
        self.sidebar_header.addWidget(self.collapse_sidebar_button)
        
        sidebar_layout.addLayout(self.sidebar_header)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepNavigator")
        self.step_list.setFixedWidth(230)
        for step in self.STEP_LABELS:
            self.step_list.addItem(QListWidgetItem(step))
            
        sidebar_layout.addWidget(self.step_list, stretch=1)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ingestion_page())
        self.stack.addWidget(self._build_configuration_page())
        self.stack.addWidget(self._build_prediction_page())
        self.stack.addWidget(self._build_curation_page())
        self.stack.addWidget(self._build_export_page())

        # Connect navigation after stack exists because setCurrentRow emits currentRowChanged.
        self.step_list.currentRowChanged.connect(self._on_step_changed)
        self.step_list.setCurrentRow(0)

        body_split.addWidget(self.sidebar_container)
        body_split.addWidget(self.stack)
        body_split.setStretchFactor(0, 0)
        body_split.setStretchFactor(1, 1)

        root.addWidget(body_split, stretch=1)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self.global_progress = QProgressBar()
        self.global_progress.setMaximumWidth(200)
        self.global_progress.setMaximumHeight(14)
        self.global_progress.setTextVisible(False)
        self.global_progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.global_progress)

    def _toggle_sidebar(self):
        is_visible = self.step_list.isVisible()
        self.step_list.setVisible(not is_visible)
        self.completion_label.setVisible(not is_visible)
        if not is_visible:
            self.collapse_sidebar_button.setText("<<")
            self.sidebar_container.setFixedWidth(230)
        else:
            self.collapse_sidebar_button.setText(">>")
            self.sidebar_container.setFixedWidth(30)

    def _build_top_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TopBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        try:
            from importlib.metadata import version
            app_version = version("DeepSlice")
        except Exception:
            app_version = "Unknown"

        self.project_label = QLabel(f"DeepSlice Desktop v{app_version}")
        self.project_label.setObjectName("ProjectLabel")

        self.session_status_label = QLabel("Session: New")
        self.session_status_label.setObjectName("SessionLabel")

        self.hardware_mode_label = QLabel("Mode: Detecting")
        self.hardware_mode_label.setObjectName("HardwareLabel")

        self.hardware_button = QPushButton("Hardware Health")
        self.hardware_button.clicked.connect(self._show_hardware_health)

        self.new_session_button = QPushButton("New Session")
        self.new_session_button.clicked.connect(self._reset_session)

        self.save_session_button = QPushButton("Save Session")
        self.save_session_button.clicked.connect(self._save_session)

        self.load_session_button = QPushButton("Load Session / QuickNII")
        self.load_session_menu = QMenu(self.load_session_button)
        self.load_session_action = self.load_session_menu.addAction("Browse...")
        self.load_session_action.triggered.connect(self._load_session_or_quint)
        self.load_session_menu.addSeparator()
        self.recent_sessions_actions = []
        for i in range(5):
            action = self.load_session_menu.addAction(f"Recent {i+1}")
            action.setVisible(False)
            self.recent_sessions_actions.append(action)
        self.load_session_button.setMenu(self.load_session_menu)
        self._update_recent_sessions_menu()

        self.error_menu_button = QPushButton("Errors")
        self.error_menu = QMenu(self.error_menu_button)
        
        self._unread_error_count = 0
        
        self.open_log_action = self.error_menu.addAction("Open Error Log")
        self.open_log_action.triggered.connect(self._open_error_log)
        
        self.copy_error_action = self.error_menu.addAction("Copy Last Error")
        self.copy_error_action.triggered.connect(self._copy_last_error_report)
        self.copy_error_action.setEnabled(False)
        
        self.auto_fix_action = self.error_menu.addAction("Try Auto-Fix Last Error")
        self.auto_fix_action.triggered.connect(self._try_auto_fix_last_error)
        self.auto_fix_action.setEnabled(False)

        self.error_menu_button.setMenu(self.error_menu)
        self.error_menu_button.setObjectName("ErrorMenuButton")

        self.copy_error_button = self.copy_error_action
        self.auto_fix_button = self.auto_fix_action

        layout.addWidget(self.project_label)
        layout.addWidget(self.session_status_label)
        layout.addStretch(1)
        layout.addWidget(self.hardware_mode_label)
        layout.addWidget(self.hardware_button)
        layout.addWidget(self.new_session_button)
        layout.addWidget(self.save_session_button)
        layout.addWidget(self.load_session_button)
        layout.addWidget(self.error_menu_button)
        return frame

    def _update_recent_sessions_menu(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        recent = settings.value("recent_sessions", [])
        if not isinstance(recent, list):
            recent = []
        for i, action in enumerate(self.recent_sessions_actions):
            if i < len(recent):
                action.setText(recent[i])
                action.setVisible(True)
                try: action.triggered.disconnect()
                except Exception: pass
                # Lambda with default arg to capture the current path
                action.triggered.connect(lambda checked=False, path=recent[i]: self._load_session_file(path))
            else:
                action.setVisible(False)

    def _add_recent_session(self, path: str):
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        recent = settings.value("recent_sessions", [])
        if not isinstance(recent, list):
            recent = []
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:5]
        settings.setValue("recent_sessions", recent)
        self._update_recent_sessions_menu()


    def _setup_shortcuts(self):
        for i in range(5):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i+1}"), self)
            shortcut.activated.connect(lambda index=i: self.stacked_widget.setCurrentIndex(index))
            
        help_shortcut = QShortcut(QKeySequence("Ctrl+?"), self)
        help_shortcut.activated.connect(self._show_shortcuts_help)
        f1_shortcut = QShortcut(QKeySequence("F1"), self)
        f1_shortcut.activated.connect(self._show_shortcuts_help)
        
        QShortcut(QKeySequence.Undo, self).activated.connect(self._undo)
        QShortcut(QKeySequence.Redo, self).activated.connect(self._redo)
        
    def _show_shortcuts_help(self):
        text = (
            "Keyboard Shortcuts:\n\n"
            "Ctrl+1 to Ctrl+5 : Navigate between pages\n"
            "Ctrl+Z : Undo curation changes\n"
            "Ctrl+Y : Redo curation changes\n"
            "Ctrl+? / F1 : Show this help"
        )
        QMessageBox.information(self, "Shortcuts", text)
        
    def _reset_session(self):
        if getattr(self.state, "is_dirty", False):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Start a new session anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        self.state = DeepSliceAppState()
        self._session_base_text = "Session: New"
        self._update_session_status()
        self._refresh_all_views()
        
    def _update_session_status(self):
        text = self._session_base_text
        is_dirty = getattr(self.state, "is_dirty", False)
        if is_dirty:
            text = f"*{text}"
        self.session_status_label.setText(text)
        self.setWindowTitle(f"DeepSlice Desktop{' *' if is_dirty else ''}")

    def closeEvent(self, event):
        if getattr(self.state, "is_dirty", False):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            
    def _record_error(self, context: str, error_text: str):
        clean_context = str(context).strip() or "DeepSlice error"
        clean_text = str(error_text).strip() or "No additional error details were provided."
        analysis = self._error_autofixer.analyze_error(clean_context, clean_text)
        analysis_text = self._error_autofixer.format_analysis(analysis)

        report_body = clean_text
        if analysis_text:
            report_body = f"{clean_text}\n\nAuto Analysis:\n{analysis_text}"

        if hasattr(self, "console_output"):
            self._append_console_log(f"[ERROR] {clean_context}")
            self._append_console_log(clean_text)
            if analysis.get("summary"):
                self._append_console_log(f"[ANALYSIS] {analysis['summary']}")

        self._last_error_report = build_error_report(
            context=clean_context,
            error_text=report_body,
            log_path=self.error_log_path,
        )
        self._last_error_context = clean_context
        self._last_error_text = clean_text
        self._last_error_analysis = analysis

        if hasattr(self, "copy_error_button"):
            self.copy_error_button.setEnabled(True)
        if hasattr(self, "auto_fix_button"):
            self.auto_fix_button.setEnabled(bool(analysis.get("auto_fix_available", False)))
            
        if hasattr(self, "_unread_error_count") and hasattr(self, "error_menu_button"):
            self._unread_error_count += 1
            self.error_menu_button.setText(f"Errors ({self._unread_error_count})")
            self.error_menu_button.setStyleSheet("QPushButton { background-color: #A43344; }")

        log_error_text(clean_context, report_body)

    def _copy_last_error_report(self, show_message: bool = True):
        report = self._last_error_report.strip()
        if not report:
            tail_text = read_error_log_tail()
            if not tail_text.strip():
                tail_text = "No logged errors found yet."
            report = build_error_report(
                context="DeepSlice log tail",
                error_text=tail_text,
                log_path=self.error_log_path,
            )

        QApplication.clipboard().setText(report)

        if show_message:
            QMessageBox.information(
                self,
                "Error Report Copied",
                "Copied an error report to clipboard. You can paste it into chat or an issue report.",
            )

    def _try_auto_fix_last_error(self):
        if not self._last_error_text:
            QMessageBox.information(
                self,
                "Auto-Fix",
                "No previous error is available for auto-fix.",
            )
            return

        self._start_auto_fix(
            context=self._last_error_context or "Last recorded error",
            error_text=self._last_error_text,
        )

    def _start_auto_fix(self, context: str, error_text: str):
        analysis = self._error_autofixer.analyze_error(context, error_text)
        if not analysis.get("auto_fix_available", False):
            QMessageBox.information(
                self,
                "Auto-Fix",
                self._error_autofixer.format_analysis(analysis),
            )
            return

        self._append_console_log("Starting automatic error-fix attempt...")
        self.auto_fix_button.setEnabled(False)

        worker = FunctionWorker(self._auto_fix_task, context, error_text)
        worker.signals.finished.connect(self._on_auto_fix_finished)
        worker.signals.error.connect(self._on_auto_fix_error)
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _auto_fix_task(self, context: str, error_text: str):
        return self._error_autofixer.try_auto_fix(context, error_text)

    def _on_auto_fix_finished(self, result: dict):
        analysis = result.get("analysis", {}) or {}
        self.auto_fix_button.setEnabled(bool(analysis.get("auto_fix_available", False)))

        summary = str(result.get("summary", "Automatic fix finished.")).strip()
        details = str(result.get("details", "")).strip()
        combined = summary if not details else f"{summary}\n\n{details}"

        if bool(result.get("succeeded", False)):
            self._append_console_log(f"[AUTO-FIX] {summary}")
            QMessageBox.information(self, "Auto-Fix Succeeded", combined)
            return

        if bool(result.get("attempted", False)):
            self._show_logged_error(
                title="Auto-Fix Failed",
                context="Automatic fix attempt failed",
                error_text=combined,
                icon=QMessageBox.Warning,
            )
            return

        QMessageBox.information(self, "Auto-Fix", combined)

    def _on_auto_fix_error(self, error_text: str):
        self.auto_fix_button.setEnabled(bool(self._last_error_analysis and self._last_error_analysis.get("auto_fix_available", False)))
        self._show_logged_error(
            title="Auto-Fix Error",
            context="Automatic fix process crashed",
            error_text=error_text,
            icon=QMessageBox.Warning,
        )

    def _open_error_log(self):
        if hasattr(self, "_unread_error_count"):
            self._unread_error_count = 0
            self.error_menu_button.setText("Errors")
            self.error_menu_button.setStyleSheet("")

        if not os.path.exists(self.error_log_path):
            QMessageBox.information(
                self,
                "Open Error Log",
                f"No error log file exists yet:\n{self.error_log_path}",
            )
            return

        try:
            if os.name == "nt":
                os.startfile(self.error_log_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", self.error_log_path])
        except Exception as exc:
            self._show_logged_exception(
                title="Open Error Log",
                context="Unable to open the error log file",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _show_logged_error(
        self,
        title: str,
        context: str,
        error_text: str,
        icon=QMessageBox.Critical,
    ):
        self._record_error(context, error_text)
        analysis = self._last_error_analysis or {}
        analysis_text = self._error_autofixer.format_analysis(analysis)
        can_auto_fix = bool(analysis.get("auto_fix_available", False))

        message_box = QMessageBox(self)
        message_box.setWindowTitle(title)
        message_box.setIcon(icon)
        message_box.setText(
            (
                f"{context}\n\n"
                f"Details have been written to:\n{self.error_log_path}\n\n"
                f"Analysis: {analysis.get('summary', 'No automatic pattern match found.')}"
            )
        )
        detail_blocks = [str(error_text)]
        if analysis_text:
            detail_blocks.append("Auto Analysis\n" + analysis_text)
        message_box.setDetailedText("\n\n".join(detail_blocks))
        message_box.setStandardButtons(QMessageBox.Ok)
        copy_button = message_box.addButton("Copy Error Report", QMessageBox.ActionRole)
        auto_fix_button = None
        if can_auto_fix:
            auto_fix_button = message_box.addButton("Try Auto-Fix", QMessageBox.ActionRole)
        message_box.exec()

        if message_box.clickedButton() == copy_button:
            self._copy_last_error_report(show_message=False)
        elif auto_fix_button is not None and message_box.clickedButton() == auto_fix_button:
            self._start_auto_fix(context=context, error_text=error_text)

    def _show_logged_exception(
        self,
        title: str,
        context: str,
        exc: BaseException,
        icon=QMessageBox.Critical,
    ):
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._show_logged_error(title, context, error_text, icon=icon)

    def _build_ingestion_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)

        self.drop_area = DropArea()
        self.drop_area.pathsDropped.connect(self._handle_dropped_paths)
        left_layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self._add_folder)
        self.add_files_button = QPushButton("Add Files")
        self.add_files_button.clicked.connect(self._add_files)
        self.clear_images_button = QPushButton("Clear All")
        self.clear_images_button.clicked.connect(self._clear_images)
        button_row.addWidget(self.add_folder_button)
        button_row.addWidget(self.add_files_button)
        button_row.addWidget(self.clear_images_button)
        left_layout.addLayout(button_row)

        options_group = QGroupBox("Pre-flight Options")
        options_layout = QVBoxLayout(options_group)
        
        section_number_layout = QHBoxLayout()
        self.enable_section_numbers_checkbox = QCheckBox(
            "Detect section numbers from filename (_sXXX)"
        )
        self.enable_section_numbers_checkbox.setChecked(True)
        self.enable_section_numbers_checkbox.toggled.connect(self._update_run_button_state)
        self.naming_helper_button = QToolButton()
        self.naming_helper_button.setText("?")
        self.naming_helper_button.setToolTip("Help with naming conventions")
        self.naming_helper_button.clicked.connect(self._show_naming_helper)
        section_number_layout.addWidget(self.enable_section_numbers_checkbox)
        section_number_layout.addWidget(self.naming_helper_button)
        section_number_layout.addStretch(1)

        self.legacy_parsing_checkbox = QCheckBox(
            "Legacy parser fallback (last 3 digits)"
        )
        self.legacy_parsing_checkbox.toggled.connect(self._refresh_ingestion_views)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(
            [
                "Coronal (supported)",
                "Sagittal (coming soon)",
                "Horizontal (coming soon)",
            ]
        )
        self.orientation_combo.currentIndexChanged.connect(self._update_run_button_state)
        options_layout.addLayout(section_number_layout)
        options_layout.addWidget(self.legacy_parsing_checkbox)
        options_layout.addWidget(self.orientation_combo)
        left_layout.addWidget(options_group)

        self.slice_count_label = QLabel("Slices: 0")
        self.ingestion_warning_label = QLabel("")
        self.ingestion_warning_label.setWordWrap(True)
        self.ingestion_warning_label.setObjectName("WarningText")
        left_layout.addWidget(self.slice_count_label)
        left_layout.addWidget(self.ingestion_warning_label)

        self.index_table = QTableWidget(0, 3)
        self.index_table.setHorizontalHeaderLabels(["Filename", "Detected Index", "Status"])
        self.index_table.horizontalHeader().setStretchLastSection(True)
        self.index_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.index_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.index_table.setSortingEnabled(True)
        left_layout.addWidget(self.index_table, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        
        thumbnail_header = QHBoxLayout()
        thumbnail_header.addWidget(QLabel("Thumbnail Grid (cached previews)"))
        thumbnail_header.addStretch(1)
        self.thumbnail_filter_edit = QLineEdit()
        self.thumbnail_filter_edit.setPlaceholderText("Filter by filename...")
        self.thumbnail_filter_edit.setFixedWidth(200)
        self.thumbnail_filter_edit.textChanged.connect(self._filter_thumbnails)
        thumbnail_header.addWidget(self.thumbnail_filter_edit)
        
        right_layout.addLayout(thumbnail_header)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        self.thumbnail_list.setIconSize(QSize(256, 256))
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setSpacing(8)
        self.thumbnail_list.itemSelectionChanged.connect(self._on_thumbnail_selection_changed)
        self.thumbnail_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thumbnail_list.customContextMenuRequested.connect(self._show_thumbnail_context_menu)
        right_layout.addWidget(self.thumbnail_list, stretch=1)

        self.ingestion_preview = SliceGraphicsView()
        right_layout.addWidget(self.ingestion_preview, stretch=2)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _show_naming_helper(self):
        text = (
            "Naming Convention Help:\n\n"
            "DeepSlice can automatically detect the section index from filenames if they follow the pattern `_sXXX`.\n"
            "For example:\n"
            "  - `brain1_s001.png` -> Index 1\n"
            "  - `mouse_A_s142_fluoro.tif` -> Index 142\n\n"
            "If this fails, you can try the 'Legacy parser fallback' which looks at the last 3 digits in the filename."
        )
        QMessageBox.information(self, "Naming Convention", text)

    def _filter_thumbnails(self, text: str):
        query = text.lower()
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            filename = item.text().lower()
            item.setHidden(query not in filename)

    def _show_thumbnail_context_menu(self, pos):
        item = self.thumbnail_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove this image")
        action = menu.exec(self.thumbnail_list.viewport().mapToGlobal(pos))
        if action == remove_action:
            image_path = item.data(Qt.UserRole)
            self.state.remove_image(image_path)
            self._refresh_all_views()

    def _build_configuration_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        species_group = QGroupBox("Species / Atlas")
        species_layout = QVBoxLayout(species_group)
        self.mouse_radio = QRadioButton("Mouse (Allen CCFv3)")
        self.rat_radio = QRadioButton("Rat (Waxholm Rat Atlas)")
        self.mouse_radio.setChecked(True)
        self.mouse_radio.toggled.connect(self._on_species_changed)
        self.rat_radio.toggled.connect(self._on_species_changed)
        species_layout.addWidget(self.mouse_radio)
        species_layout.addWidget(self.rat_radio)
        left_layout.addWidget(species_group)

        geometry_group = QGroupBox("Physical Geometry")
        geometry_layout = QFormLayout(geometry_group)
        self.auto_thickness_checkbox = QCheckBox("Auto-estimate thickness")
        self.auto_thickness_checkbox.setChecked(True)
        self.auto_thickness_checkbox.toggled.connect(self._on_auto_thickness_toggled)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.0, 1000.0)
        self.thickness_spin.setDecimals(2)
        self.thickness_spin.setSuffix(" um")
        self.thickness_spin.setEnabled(False)
        self.suggest_thickness_button = QPushButton("Suggest Thickness")
        self.suggest_thickness_button.clicked.connect(self._suggest_thickness)
        self.detected_direction_label = QLabel("Detected direction: unknown")
        self.direction_override_combo = QComboBox()
        self.direction_override_combo.addItems(
            ["Auto", "rostro-caudal", "caudal-rostro"]
        )
        self.direction_override_combo.currentTextChanged.connect(
            self._on_direction_override_changed
        )
        geometry_layout.addRow(self.auto_thickness_checkbox)
        geometry_layout.addRow("Section Thickness", self.thickness_spin)
        geometry_layout.addRow(self.suggest_thickness_button)
        geometry_layout.addRow(self.detected_direction_label)
        geometry_layout.addRow("Direction Override", self.direction_override_combo)
        left_layout.addWidget(geometry_group)

        prediction_group = QGroupBox("Prediction Modes")
        prediction_layout = QVBoxLayout(prediction_group)
        self.ensemble_checkbox = QCheckBox("Ensemble prediction (if available)")
        self.ensemble_checkbox.setChecked(True)
        self.secondary_model_checkbox = QCheckBox(
            "Use secondary model only (for comparison)"
        )
        self.secondary_model_checkbox.setChecked(False)
        self.secondary_model_checkbox.setToolTip(
            "Runs only the secondary model weights. Do not use together with ensemble."
        )
        self.legacy_from_config_checkbox = QCheckBox(
            "Legacy section-number parser"
        )
        self.legacy_from_config_checkbox.toggled.connect(self._sync_legacy_checkbox)
        prediction_layout.addWidget(self.ensemble_checkbox)
        prediction_layout.addWidget(self.secondary_model_checkbox)
        prediction_layout.addWidget(self.legacy_from_config_checkbox)
        left_layout.addWidget(prediction_group)

        self.config_validation_label = QLabel("")
        self.config_validation_label.setObjectName("WarningText")
        self.config_validation_label.setWordWrap(True)
        left_layout.addWidget(self.config_validation_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.tech_toggle = QToolButton()
        self.tech_toggle.setText("Technical Insights")
        self.tech_toggle.setCheckable(True)
        self.tech_toggle.setChecked(True)
        self.tech_toggle.toggled.connect(self._toggle_tech_insights)

        self.tech_insights = QPlainTextEdit()
        self.tech_insights.setReadOnly(True)
        self.tech_insights.setPlainText(
            "O/U/V vectors define each section plane in atlas space.\n\n"
            "Angle normalization is not a simple vector average. DeepSlice calculates\n"
            "DV and ML angle per section, computes a Gaussian-weighted mean around\n"
            "atlas center depth, then rotates each section plane toward those means.\n"
            "The process runs twice because adjusting one plane perturbs the other.\n\n"
            "Thickness suggestion is estimated from section-number spacing relative to\n"
            "predicted depth spacing, then weighted by center-proximal Gaussian scores."
        )

        right_layout.addWidget(self.tech_toggle)
        right_layout.addWidget(self.tech_insights, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _build_prediction_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        run_layout = QHBoxLayout()
        self.run_alignment_button = QPushButton("Run Alignment")
        self.run_alignment_button.clicked.connect(self._run_alignment)
        self.run_alignment_button.setMinimumHeight(44)
        
        self.cancel_alignment_button = QPushButton("Cancel")
        self.cancel_alignment_button.clicked.connect(self._cancel_alignment)
        self.cancel_alignment_button.setMinimumHeight(44)
        self.cancel_alignment_button.setEnabled(False)
        
        run_layout.addWidget(self.run_alignment_button, stretch=3)
        run_layout.addWidget(self.cancel_alignment_button, stretch=1)

        self.prediction_phase_label = QLabel("Phase: idle")
        self.prediction_progress_label = QLabel("Progress: 0 / 0")
        self.prediction_progress_bar = QProgressBar()
        self.prediction_progress_bar.setRange(0, 100)
        self.prediction_progress_bar.setValue(0)

        self.predicted_thickness_label = QLabel("Estimated thickness: -")
        self.accept_predicted_thickness_button = QPushButton("Use Predicted Thickness")
        self.accept_predicted_thickness_button.clicked.connect(
            self._accept_predicted_thickness
        )
        self.accept_predicted_thickness_button.setEnabled(False)

        self.prediction_direction_label = QLabel("Detected indexing direction: -")

        console_tools = QHBoxLayout()
        self.console_toggle = QToolButton()
        self.console_toggle.setCheckable(True)
        self.console_toggle.setText("Show Runtime Console")
        self.console_toggle.toggled.connect(self._toggle_console)
        
        self.clear_console_button = QToolButton()
        self.clear_console_button.setText("Clear")
        self.clear_console_button.clicked.connect(lambda: self.console_output.clear())
        self.clear_console_button.setVisible(False)
        
        self.copy_console_button = QToolButton()
        self.copy_console_button.setText("Copy")
        self.copy_console_button.clicked.connect(lambda: QApplication.clipboard().setText(self.console_output.toPlainText()))
        self.copy_console_button.setVisible(False)
        
        self.console_toggle.toggled.connect(lambda v: self.clear_console_button.setVisible(v))
        self.console_toggle.toggled.connect(lambda v: self.copy_console_button.setVisible(v))
        
        console_tools.addWidget(self.console_toggle)
        console_tools.addStretch(1)
        console_tools.addWidget(self.clear_console_button)
        console_tools.addWidget(self.copy_console_button)

        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setVisible(False)

        left_layout.addLayout(run_layout)
        left_layout.addWidget(self.prediction_phase_label)
        left_layout.addWidget(self.prediction_progress_label)
        left_layout.addWidget(self.prediction_progress_bar)
        left_layout.addWidget(self.predicted_thickness_label)
        left_layout.addWidget(self.accept_predicted_thickness_button)
        left_layout.addWidget(self.prediction_direction_label)
        left_layout.addLayout(console_tools)
        left_layout.addWidget(self.console_output, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.prediction_slice_selector = QComboBox()
        self.prediction_slice_selector.currentIndexChanged.connect(
            self._refresh_prediction_preview
        )
        right_layout.addWidget(self.prediction_slice_selector)

        self.prediction_viewer = SliceGraphicsView()
        right_layout.addWidget(self.prediction_viewer, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _build_curation_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        list_header_layout = QHBoxLayout()
        self.curation_select_all_btn = QToolButton()
        self.curation_select_all_btn.setText("Select All")
        self.curation_select_all_btn.clicked.connect(lambda: self._set_all_flags(Qt.Checked))
        
        self.curation_deselect_all_btn = QToolButton()
        self.curation_deselect_all_btn.setText("Deselect All")
        self.curation_deselect_all_btn.clicked.connect(lambda: self._set_all_flags(Qt.Unchecked))
        
        self.confidence_filter_combo = QComboBox()
        self.confidence_filter_combo.addItems(["All Confidences", "High Only", "Medium Only", "Low Only"])
        self.confidence_filter_combo.currentIndexChanged.connect(self._filter_curation_list)
        
        list_header_layout.addWidget(self.curation_select_all_btn)
        list_header_layout.addWidget(self.curation_deselect_all_btn)
        list_header_layout.addStretch()
        list_header_layout.addWidget(QLabel("Filter:"))
        list_header_layout.addWidget(self.confidence_filter_combo)
        
        left_layout.addLayout(list_header_layout)

        self.slice_flag_list = QListWidget()
        self.slice_flag_list.currentRowChanged.connect(self._on_curation_slice_selected)
        self.slice_flag_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.slice_flag_list.setDefaultDropAction(Qt.MoveAction)
        self.slice_flag_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.slice_flag_list.customContextMenuRequested.connect(self._show_curation_context_menu)

        self.apply_bad_sections_button = QPushButton("Apply Bad Section Flags")
        self.apply_bad_sections_button.clicked.connect(self._apply_bad_section_flags)
        self.apply_manual_order_button = QPushButton("Apply Manual Reordering")
        self.apply_manual_order_button.clicked.connect(self._apply_manual_order)
        self.detect_outliers_button = QPushButton("Detect Outliers")
        self.detect_outliers_button.clicked.connect(self._detect_outliers)

        controls_group = QGroupBox("Propagation / Curation")
        controls_layout = QGridLayout(controls_group)

        self.normalize_angles_button = QPushButton("Normalize Angles")
        self.normalize_angles_button.setToolTip(
            "Applies Gaussian-weighted DV/ML mean-angle propagation (two-pass adjustment)."
        )
        self.normalize_angles_button.clicked.connect(self._normalize_angles)

        self.enforce_order_button = QPushButton("Enforce Index Order")
        self.enforce_order_button.setToolTip(
            "Reorders Oy values to match index ordering while preserving measured spacing."
        )
        self.enforce_order_button.clicked.connect(self._enforce_index_order)

        self.enforce_spacing_button = QPushButton("Enforce Index Spacing")
        self.enforce_spacing_button.setToolTip(
            "Recalculates Oy values to be evenly spaced based on section thickness."
        )
        self.enforce_spacing_button.clicked.connect(self._enforce_index_spacing)

        self.ml_spin = QDoubleSpinBox()
        self.ml_spin.setRange(-90.0, 90.0)
        self.ml_spin.setDecimals(2)
        self.ml_spin.setSuffix(" deg")

        self.dv_spin = QDoubleSpinBox()
        self.dv_spin.setRange(-90.0, 90.0)
        self.dv_spin.setDecimals(2)
        self.dv_spin.setSuffix(" deg")

        self.apply_manual_angles_button = QPushButton("Apply Manual Angles")
        self.apply_manual_angles_button.setToolTip(
            "Directly overrides dataset ML and DV angles in degrees."
        )
        self.apply_manual_angles_button.clicked.connect(self._apply_manual_angles)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)

        controls_layout.addWidget(self.normalize_angles_button, 0, 0, 1, 2)
        controls_layout.addWidget(self.enforce_order_button, 1, 0, 1, 2)
        controls_layout.addWidget(self.enforce_spacing_button, 2, 0, 1, 2)
        controls_layout.addWidget(QLabel("ML Angle"), 3, 0)
        controls_layout.addWidget(self.ml_spin, 3, 1)
        controls_layout.addWidget(QLabel("DV Angle"), 4, 0)
        controls_layout.addWidget(self.dv_spin, 4, 1)
        controls_layout.addWidget(self.apply_manual_angles_button, 5, 0, 1, 2)
        controls_layout.addWidget(self.undo_button, 6, 0)
        controls_layout.addWidget(self.redo_button, 6, 1)

        left_layout.addWidget(self.slice_flag_list, stretch=2)
        left_layout.addWidget(self.apply_manual_order_button)
        left_layout.addWidget(self.apply_bad_sections_button)
        left_layout.addWidget(self.detect_outliers_button)
        left_layout.addWidget(controls_group, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.linearity_figure = Figure(figsize=(6, 4), facecolor="#1A1F26")
        self.linearity_canvas = FigureCanvas(self.linearity_figure)
        self.linearity_axis = self.linearity_figure.add_subplot(111)
        self.linearity_canvas.mpl_connect("button_press_event", self._on_linearity_click)

        atlas_controls = QHBoxLayout()
        self.enable_atlas_preview_checkbox = QCheckBox("Atlas volume preview")
        self.enable_atlas_preview_checkbox.toggled.connect(self._on_atlas_preview_toggled)
        self.atlas_volume_combo = QComboBox()
        self.atlas_volume_combo.currentTextChanged.connect(self._on_atlas_volume_changed)
        self.enable_blend_overlay_checkbox = QCheckBox("Blend atlas on histology")
        self.enable_blend_overlay_checkbox.toggled.connect(self._on_blend_overlay_toggled)
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(5, 95)
        self.blend_slider.setValue(35)
        self.blend_slider.valueChanged.connect(self._on_blend_slider_changed)
        self.blend_percent_label = QLabel("Blend: 35%")
        self.atlas_slice_info_label = QLabel("Atlas: disabled")
        atlas_controls.addWidget(self.enable_atlas_preview_checkbox)
        atlas_controls.addWidget(QLabel("Volume"))
        atlas_controls.addWidget(self.atlas_volume_combo)
        atlas_controls.addWidget(self.enable_blend_overlay_checkbox)
        atlas_controls.addWidget(self.blend_slider)
        atlas_controls.addWidget(self.blend_percent_label)
        atlas_controls.addWidget(self.atlas_slice_info_label, stretch=1)

        self.curation_viewer = SliceGraphicsView()
        self.atlas_viewer = SliceGraphicsView()
        self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")

        viewer_split = QSplitter(Qt.Horizontal)
        viewer_split.addWidget(self.curation_viewer)
        viewer_split.addWidget(self.atlas_viewer)
        viewer_split.setStretchFactor(0, 1)
        viewer_split.setStretchFactor(1, 1)

        right_layout.addWidget(self.linearity_canvas, stretch=2)
        right_layout.addLayout(atlas_controls)
        right_layout.addWidget(viewer_split, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout(page)
        root.addWidget(split)
        self._refresh_atlas_volume_options()
        return page

    def _build_export_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        output_group = QGroupBox("Export Configuration")
        output_layout = QFormLayout(output_group)

        output_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self._get_persisted_export_path())
        self.output_dir_edit.textChanged.connect(self._persist_export_path)
        self.browse_output_dir_button = QPushButton("Browse")
        self.browse_output_dir_button.clicked.connect(self._browse_output_directory)
        output_dir_row.addWidget(self.output_dir_edit)
        output_dir_row.addWidget(self.browse_output_dir_button)

        self.output_basename_edit = QLineEdit("DeepSliceResults")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            [
                "JSON (QuickNII/VisuAlign)",
                "Legacy XML",
            ]
        )
        
        self.export_size_estimate_label = QLabel("~0 MB")
        self.output_format_combo.currentIndexChanged.connect(self._update_export_size_estimate)

        output_layout.addRow("Output Directory", output_dir_row)
        output_layout.addRow("Base Filename", self.output_basename_edit)
        output_layout.addRow("Primary Export", self.output_format_combo)
        output_layout.addRow("Estimated Size", self.export_size_estimate_label)

        export_actions_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Predictions")
        self.export_button.clicked.connect(self._export_predictions)
        
        self.open_export_dir_button = QToolButton()
        self.open_export_dir_button.setText("Open Folder")
        self.open_export_dir_button.clicked.connect(self._open_export_directory)
        
        self.copy_export_path_button = QToolButton()
        self.copy_export_path_button.setText("Copy Path")
        self.copy_export_path_button.clicked.connect(self._copy_export_path)
        
        export_actions_layout.addWidget(self.export_button, stretch=1)
        export_actions_layout.addWidget(self.open_export_dir_button)
        export_actions_layout.addWidget(self.copy_export_path_button)

        self.report_button = QPushButton("Generate Report (PDF)")
        self.report_button.clicked.connect(self._generate_report)
        
        self.preview_report_button = QToolButton()
        self.preview_report_button.setText("Preview Report")
        self.preview_report_button.clicked.connect(self._preview_report)
        
        report_layout = QHBoxLayout()
        report_layout.addWidget(self.report_button, stretch=3)
        report_layout.addWidget(self.preview_report_button, stretch=1)

        self.pdf_content_group = QGroupBox("PDF Contents")
        pdf_content_layout = QHBoxLayout(self.pdf_content_group)
        self.pdf_include_stats = QCheckBox("Summary Stats")
        self.pdf_include_stats.setChecked(True)
        self.pdf_include_plot = QCheckBox("Linearity Plot")
        self.pdf_include_plot.setChecked(True)
        self.pdf_include_images = QCheckBox("Sample Images")
        self.pdf_include_images.setChecked(True)
        pdf_content_layout.addWidget(self.pdf_include_stats)
        pdf_content_layout.addWidget(self.pdf_include_plot)
        pdf_content_layout.addWidget(self.pdf_include_images)

        quicknii_row = QHBoxLayout()
        self.quicknii_path_edit = QLineEdit()
        self.quicknii_path_edit.setPlaceholderText("Optional path to QuickNII executable")
        self.quicknii_path_edit.setText(self._get_persisted_quicknii_path())
        self.quicknii_path_edit.textChanged.connect(self._persist_quicknii_path)
        self.quicknii_browse_button = QPushButton("Browse")
        self.quicknii_browse_button.clicked.connect(self._browse_quicknii_path)
        quicknii_row.addWidget(self.quicknii_path_edit)
        quicknii_row.addWidget(self.quicknii_browse_button)

        self.open_quicknii_button = QPushButton("Open in QuickNII")
        self.open_quicknii_button.clicked.connect(self._open_in_quicknii)

        self.summary_label = QLabel("Processed: 0 | Excluded: 0")
        self.deviation_label = QLabel("Mean angular deviation: 0.00 deg")
        self.markers_label = QLabel("")
        self.markers_label.setWordWrap(True)
        self.markers_label.setObjectName("WarningText")

        left_layout.addWidget(output_group)
        left_layout.addLayout(export_actions_layout)
        left_layout.addWidget(self.pdf_content_group)
        left_layout.addLayout(report_layout)
        left_layout.addLayout(quicknii_row)
        left_layout.addWidget(self.open_quicknii_button)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.deviation_label)
        left_layout.addWidget(self.markers_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.export_notes = QPlainTextEdit()
        self.export_notes.setReadOnly(True)
        self.export_notes.setPlainText(
            "Export details:\n\n"
            "1) CSV is always exported alongside JSON/XML.\n"
            "2) JSON is QuickNII/VisuAlign-compatible and preserves markers.\n"
            "3) Legacy XML is provided for older workflows.\n"
            "4) Load Session can re-open previous QuickNII JSON/XML for re-curation."
        )

        right_layout.addWidget(self.export_notes)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #0B0F14;
                color: #DEE6EF;
                font-family: Inter, Roboto, 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
            #TopBar {
                background: #1A1F26;
                border-radius: 10px;
            }
            #ProjectLabel {
                font-size: 13pt;
                font-weight: 600;
                color: #F5FAFF;
            }
            #SessionLabel, #HardwareLabel {
                color: #9CB0C7;
            }
            #WarningText {
                color: #F2B544;
            }
            QGroupBox {
                border: 1px solid #27303A;
                border-radius: 10px;
                margin-top: 10px;
                padding: 8px;
                background: #1A1F26;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #8FB7FF;
            }
            QPushButton {
                background: #1E6FFF;
                border: 1px solid #2A79FF;
                border-radius: 10px;
                padding: 7px 12px;
                color: #F7FBFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #2A79FF;
            }
            QPushButton:disabled {
                background: #2A2F39;
                border-color: #2A2F39;
                color: #7A8492;
            }
            QLineEdit, QPlainTextEdit, QListWidget, QTableWidget, QComboBox, QDoubleSpinBox {
                background: #11161D;
                border: 1px solid #2A313B;
                border-radius: 9px;
                padding: 5px;
            }
            QHeaderView::section {
                background: #1A1F26;
                border: 1px solid #2A313B;
                padding: 4px;
                color: #9CB0C7;
            }
            #DropArea {
                border: 1px dashed #416EAE;
                border-radius: 12px;
                background: #131A23;
            }
            #DropTitle {
                color: #CFE0FF;
                font-weight: 600;
            }
            #DropSubtitle {
                color: #9CB0C7;
                font-size: 9pt;
            }
            #StepNavigator {
                background: #1A1F26;
                border: 1px solid #27303A;
                border-radius: 12px;
                padding: 6px;
            }
            #StepNavigator::item {
                border-radius: 8px;
                padding: 8px;
                margin: 3px;
            }
            #StepNavigator::item:selected {
                background: #164DB4;
                color: #F7FBFF;
            }
            QToolButton {
                background: #1A1F26;
                border: 1px solid #2A313B;
                border-radius: 9px;
                padding: 6px 10px;
            }
            """
        )

    def _on_step_changed(self, index: int):
        if index < 0:
            return
        if index > self._max_unlocked_step():
            self.step_list.setCurrentRow(self._max_unlocked_step())
            return
        self.stack.setCurrentIndex(index)
        self._refresh_step_states()

    def _max_unlocked_step(self) -> int:
        if len(self.state.image_paths) == 0:
            return 0
        if self.state.predictions is None:
            return 2
        return 4

    def _refresh_step_states(self):
        max_unlocked = self._max_unlocked_step()

        for idx in range(self.step_list.count()):
            item = self.step_list.item(idx)
            label = self.STEP_LABELS[idx]
            if idx <= max_unlocked:
                item.setFlags(item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

            prefix = ""
            if idx < max_unlocked:
                prefix = "✓ "
            elif idx == self.stack.currentIndex():
                prefix = "● "
            elif idx > max_unlocked:
                prefix = "◌ "
            item.setText(prefix + label)

        percent = int(((max_unlocked + 1) / len(self.STEP_LABELS)) * 100)
        if hasattr(self, "completion_label"):
            self.completion_label.setText(f"{percent}% Complete")

    def _collect_supported_files_from_paths(self, dropped_paths: List[str]) -> List[str]:
        image_paths = []
        for path in dropped_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        extension = os.path.splitext(full_path)[1].lower()
                        if extension in SUPPORTED_IMAGE_FORMATS:
                            image_paths.append(full_path)
            elif os.path.isfile(path):
                extension = os.path.splitext(path)[1].lower()
                if extension in SUPPORTED_IMAGE_FORMATS:
                    image_paths.append(path)
        return image_paths

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        image_paths = self._collect_supported_files_from_paths([folder])
        self.state.add_images(image_paths)
        self._refresh_all_views()

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Histology Images",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if not files:
            return
        self.state.add_images(files)
        self._refresh_all_views()

    def _handle_dropped_paths(self, paths: List[str]):
        image_paths = self._collect_supported_files_from_paths(paths)
        self.state.add_images(image_paths)
        self._refresh_all_views()

    def _clear_images(self):
        self.state.clear_images()
        self._refresh_all_views()

    def _on_thumbnail_selection_changed(self):
        selected = self.thumbnail_list.selectedItems()
        if len(selected) == 0:
            self.ingestion_preview.clear_with_text("Select a thumbnail to preview")
            return
        image_path = selected[0].data(Qt.UserRole)
        self.ingestion_preview.set_image(image_path)

    def _refresh_ingestion_views(self):
        self.slice_count_label.setText(f"Slices: {len(self.state.image_paths)}")

        report = self.state.image_format_report()
        warnings = []
        if len(report["unsupported"]) > 0:
            warnings.append(
                f"Unsupported files excluded: {len(report['unsupported'])}. Accepted: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
            )
        if len(report["supported"]) < 2 and len(report["supported"]) > 0:
            warnings.append("At least 2 sections are required for spacing and curation tools.")
        if 0 < len(report["supported"]) < 10:
            warnings.append("Fewer than 10 slices may reduce angle-propagation reliability.")
        if self.orientation_combo.currentIndex() != 0:
            warnings.append("Non-coronal orientation is currently not supported in this release.")

        index_report = self.state.build_index_report(
            legacy_section_numbers=self.legacy_parsing_checkbox.isChecked()
        )
        if self.enable_section_numbers_checkbox.isChecked():
            if index_report["parse_error"]:
                warnings.append(index_report["parse_error"])
            if len(index_report["duplicate_indices"]) > 0:
                warnings.append(
                    "Duplicate indices detected: "
                    + ", ".join([str(x) for x in index_report["duplicate_indices"]])
                )
            if len(index_report["missing_indices"]) > 0:
                warnings.append(
                    "Missing indices: "
                    + ", ".join([str(x) for x in index_report["missing_indices"][:20]])
                )

        self.ingestion_warning_label.setText("\n".join(warnings))

        self.index_table.setRowCount(len(index_report["rows"]))
        for row_idx, row in enumerate(index_report["rows"]):
            filename_item = QTableWidgetItem(row["filename"])
            index_item = QTableWidgetItem(str(row["detected_index"]))
            status_item = QTableWidgetItem(row["status"])

            if row["status"] != "OK":
                amber = QColor("#A26B1D")
                filename_item.setBackground(amber)
                index_item.setBackground(amber)
                status_item.setBackground(amber)

            self.index_table.setItem(row_idx, 0, filename_item)
            self.index_table.setItem(row_idx, 1, index_item)
            self.index_table.setItem(row_idx, 2, status_item)

        self.thumbnail_list.clear()
        for image_path in self.state.image_paths:
            icon = QIcon(
                QPixmap(image_path).scaled(
                    256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            item = QListWidgetItem(icon, os.path.basename(image_path))
            item.setData(Qt.UserRole, image_path)
            self.thumbnail_list.addItem(item)

        if self.thumbnail_list.count() == 0:
            self.ingestion_preview.clear_with_text("No images loaded")

    def _on_species_changed(self):
        species = "mouse" if self.mouse_radio.isChecked() else "rat"
        self.state.set_species(species)
        self._refresh_atlas_volume_options()
        self._update_run_button_state()

    def _on_auto_thickness_toggled(self, checked: bool):
        self.thickness_spin.setEnabled(not checked)
        self._update_run_button_state()

    def _suggest_thickness(self):
        try:
            value = self.state.estimate_section_thickness_um()
        except Exception as exc:
            self._show_logged_exception(
                title="Thickness Suggestion",
                context="Unable to estimate thickness yet",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self.thickness_spin.setValue(float(value))
        self.auto_thickness_checkbox.setChecked(False)

    def _sync_legacy_checkbox(self, checked: bool):
        self.legacy_parsing_checkbox.setChecked(checked)
        self._refresh_ingestion_views()

    def _toggle_tech_insights(self, visible: bool):
        self.tech_insights.setVisible(visible)

    def _toggle_console(self, visible: bool):
        self.console_output.setVisible(visible)

    def _on_direction_override_changed(self, value: str):
        if value == "Auto":
            self.state.selected_indexing_direction = self.state.detected_indexing_direction
        else:
            self.state.selected_indexing_direction = value

    def _validate_before_prediction(self) -> (List[str], List[str]):
        errors: List[str] = []
        warnings: List[str] = []

        if len(self.state.image_paths) == 0:
            errors.append("No images loaded")

        if len(self.state.image_paths) < 2 and len(self.state.image_paths) > 0:
            errors.append("At least 2 sections are required")

        if len(self.state.image_paths) < 10 and len(self.state.image_paths) > 0:
            warnings.append("Dataset has fewer than 10 slices")

        if self.orientation_combo.currentIndex() != 0:
            errors.append("Only coronal orientation is currently supported")

        if self.enable_section_numbers_checkbox.isChecked():
            index_report = self.state.build_index_report(
                legacy_section_numbers=self.legacy_parsing_checkbox.isChecked()
            )
            if index_report["parse_error"]:
                errors.append(index_report["parse_error"])
            if len(index_report["duplicate_indices"]) > 0:
                errors.append("Duplicate section numbers are not allowed")

        if not self.auto_thickness_checkbox.isChecked() and self.thickness_spin.value() <= 0:
            errors.append("Section thickness must be greater than zero")

        return errors, warnings

    def _update_run_button_state(self):
        errors, _ = self._validate_before_prediction()
        self.run_alignment_button.setEnabled(len(errors) == 0)
        if len(errors) == 0:
            self.config_validation_label.setText("Validation: ready")
        else:
            self.config_validation_label.setText("Validation: " + " | ".join(errors[:3]))

    def _run_alignment(self):
        errors, warnings = self._validate_before_prediction()
        if len(errors) > 0:
            QMessageBox.warning(self, "Cannot Run Alignment", "\n".join(errors))
            return

        if len(warnings) > 0:
            answer = QMessageBox.question(
                self,
                "Proceed With Warnings",
                "\n".join(warnings) + "\n\nContinue anyway?",
            )
            if answer != QMessageBox.Yes:
                return

        self.state.section_numbers = self.enable_section_numbers_checkbox.isChecked()
        self.state.legacy_section_numbers = self.legacy_parsing_checkbox.isChecked()
        self.state.ensemble = self.ensemble_checkbox.isChecked()
        self.state.use_secondary_model = self.secondary_model_checkbox.isChecked()

        self.run_alignment_button.setEnabled(False)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(True)
        self.prediction_progress_bar.setValue(0)
        self.console_output.clear()
        self.prediction_phase_label.setText("Phase: initializing")

        options = {
            "section_numbers": self.state.section_numbers,
            "legacy_section_numbers": self.state.legacy_section_numbers,
            "ensemble": self.state.ensemble,
            "use_secondary_model": self.state.use_secondary_model,
        }

        self._current_prediction_worker = FunctionWorker(
            self._run_prediction_task,
            options,
            inject_callbacks=True,
        )
        self._current_prediction_worker.signals.progress.connect(self._on_prediction_progress)
        self._current_prediction_worker.signals.log.connect(self._append_console_log)
        self._current_prediction_worker.signals.error.connect(self._on_prediction_error)
        self._current_prediction_worker.signals.finished.connect(self._on_prediction_finished)
        self._track_worker(self._current_prediction_worker)
        self.thread_pool.start(self._current_prediction_worker)

    def _cancel_alignment(self):
        self._append_console_log("[SYSTEM] Cancellation requested... This may take a moment to stop underlying processes.")
        self.cancel_alignment_button.setEnabled(False)
        self.run_alignment_button.setEnabled(True)
        if hasattr(self, "_current_prediction_worker"):
            # A full kill requires more advanced threading or process management.
            pass

    def _run_prediction_task(self, options: dict, progress_callback=None, log_callback=None):
        return self.state.run_prediction(
            section_numbers=options["section_numbers"],
            legacy_section_numbers=options["legacy_section_numbers"],
            ensemble=options["ensemble"],
            use_secondary_model=options["use_secondary_model"],
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    def _on_prediction_progress(self, completed: int, total: int, phase: str):
        self.prediction_phase_label.setText(f"Phase: {phase}")
        self.prediction_progress_label.setText(f"Progress: {completed} / {total}")
        self.prediction_progress_bar.setRange(0, max(total, 1))
        self.prediction_progress_bar.setValue(min(completed, total))

        if phase == "primary" and 0 < completed <= len(self.state.image_paths):
            preview_path = self.state.image_paths[completed - 1]
            self.prediction_viewer.set_image(preview_path, [f"Inferring slice {completed}/{total}"])

    def _append_console_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_output.appendPlainText(f"[{timestamp}] {message}")

    def _on_prediction_error(self, error_text: str):
        self.run_alignment_button.setEnabled(True)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(False)
        self._show_logged_error(
            title="Prediction Failed",
            context="Alignment prediction task failed",
            error_text=error_text,
            icon=QMessageBox.Critical,
        )

    def _on_prediction_finished(self, result: dict):
        self.run_alignment_button.setEnabled(True)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(False)
        self._session_base_text = f"Session: Predicted {result['slice_count']} slices"
        self._update_session_status()

        direction = result.get("direction")
        if direction:
            self.detected_direction_label.setText(f"Detected direction: {direction}")
            self.prediction_direction_label.setText(
                f"Detected indexing direction: {direction}"
            )

        predicted_thickness = result.get("predicted_thickness_um")
        if predicted_thickness is not None:
            self.predicted_thickness_label.setText(
                f"Estimated thickness: {predicted_thickness:.2f} um"
            )
            self.accept_predicted_thickness_button.setEnabled(True)
        else:
            self.predicted_thickness_label.setText("Estimated thickness: unavailable")
            self.accept_predicted_thickness_button.setEnabled(False)

        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()
        self._refresh_step_states()
        self._update_run_button_state()

    def _accept_predicted_thickness(self):
        label_text = self.predicted_thickness_label.text()
        try:
            value_text = label_text.split(":", 1)[1].replace("um", "").strip()
            value = float(value_text)
        except Exception as exc:
            self._show_logged_exception(
                title="Thickness",
                context="No predicted thickness value is available to apply",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self.auto_thickness_checkbox.setChecked(False)
        self.thickness_spin.setValue(value)

    def _refresh_prediction_selector(self):
        self.prediction_slice_selector.blockSignals(True)
        self.prediction_slice_selector.clear()

        if self.state.predictions is not None and len(self.state.predictions) > 0:
            for _, row in self.state.predictions.iterrows():
                nr = row["nr"] if "nr" in row else "-"
                self.prediction_slice_selector.addItem(f"{nr} | {row['Filenames']}")

        self.prediction_slice_selector.blockSignals(False)
        self._refresh_prediction_preview()

    def _refresh_prediction_preview(self):
        if self.state.predictions is None or len(self.state.predictions) == 0:
            self.prediction_viewer.clear_with_text("No predictions to preview")
            return

        row_index = self.prediction_slice_selector.currentIndex()
        if row_index < 0:
            row_index = 0
        row = self.state.predictions.iloc[row_index]

        image_path = self._resolve_image_path_for_filename(row["Filenames"])
        overlay = [
            f"Filename: {row['Filenames']}",
            f"O=({row['ox']:.2f}, {row['oy']:.2f}, {row['oz']:.2f})",
            f"U=({row['ux']:.2f}, {row['uy']:.2f}, {row['uz']:.2f})",
            f"V=({row['vx']:.2f}, {row['vy']:.2f}, {row['vz']:.2f})",
        ]
        self.prediction_viewer.set_image(image_path, overlay)

    def _resolve_image_path_for_filename(self, filename: str) -> Optional[str]:
        filename = os.path.basename(filename)
        for image_path in self.state.image_paths:
            if os.path.basename(image_path) == filename:
                return image_path
        return None

    def _refresh_atlas_volume_options(self):
        if not hasattr(self, "atlas_volume_combo"):
            return
        options = self.state.atlas_volume_options()
        current = self.atlas_volume_combo.currentText().strip().lower()
        normalized_options = [option.lower() for option in options]

        self.atlas_volume_combo.blockSignals(True)
        self.atlas_volume_combo.clear()
        for option in options:
            self.atlas_volume_combo.addItem(option)

        target = current if current in normalized_options else self.state.default_atlas_volume().lower()
        index = -1
        for option_index in range(self.atlas_volume_combo.count()):
            if self.atlas_volume_combo.itemText(option_index).strip().lower() == target:
                index = option_index
                break
        if index >= 0:
            self.atlas_volume_combo.setCurrentIndex(index)
        elif self.atlas_volume_combo.count() > 0:
            self.atlas_volume_combo.setCurrentIndex(0)
        self.atlas_volume_combo.blockSignals(False)

    def _on_atlas_preview_toggled(self, enabled: bool):
        if not enabled:
            self.atlas_slice_info_label.setText("Atlas: disabled")
            self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")
            self._latest_atlas_slice = None
            self._latest_atlas_meta = None
            row = self.slice_flag_list.currentRow()
            self._render_histology_preview(row)
            return
        row = self.slice_flag_list.currentRow()
        self._request_atlas_preview(row)

    def _on_atlas_volume_changed(self, _value: str):
        if not self.enable_atlas_preview_checkbox.isChecked():
            return
        row = self.slice_flag_list.currentRow()
        self._request_atlas_preview(row)

    def _on_blend_overlay_toggled(self, enabled: bool):
        self.blend_slider.setEnabled(enabled)
        row = self.slice_flag_list.currentRow()
        if enabled and self.enable_atlas_preview_checkbox.isChecked() and self._latest_atlas_slice is None:
            self._request_atlas_preview(row)
        else:
            self._render_histology_preview(row)

    def _on_blend_slider_changed(self, value: int):
        self.blend_percent_label.setText(f"Blend: {value}%")
        if not self.enable_blend_overlay_checkbox.isChecked():
            return
        row = self.slice_flag_list.currentRow()
        self._render_histology_preview(row)

    def _build_blended_overlay_image(
        self,
        histology_path: Optional[str],
        atlas_slice: Optional[np.ndarray],
        alpha: float,
    ) -> Optional[np.ndarray]:
        if histology_path is None or atlas_slice is None:
            return None
        if not os.path.exists(histology_path):
            return None

        try:
            with Image.open(histology_path) as image:
                image = image.convert("RGB")
                hist_array = np.array(image, dtype=np.float32)
        except Exception:
            return None

        atlas_array = np.asarray(atlas_slice, dtype=np.float32)
        if atlas_array.ndim != 2:
            return None

        atlas_image = Image.fromarray(np.uint8(np.clip(atlas_array, 0, 255)), mode="L")
        atlas_image = atlas_image.resize((hist_array.shape[1], hist_array.shape[0]), Image.BILINEAR)
        atlas_resized = np.array(atlas_image, dtype=np.float32) / 255.0

        # Tint atlas signal as cyan heat overlay.
        overlay = np.stack(
            [
                atlas_resized * 35.0,
                atlas_resized * 210.0,
                atlas_resized * 245.0,
            ],
            axis=2,
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blended = ((1.0 - alpha) * hist_array) + (alpha * overlay)
        return np.uint8(np.clip(blended, 0, 255))

    def _render_histology_preview(self, row_index: int):
        if self.state.predictions is None:
            self.curation_viewer.clear_with_text("No predictions to curate")
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        row = self.state.predictions.iloc[row_index]
        filename = row["Filenames"]
        image_path = self._resolve_image_path_for_filename(filename)

        confidence = 1.0
        confidence_level = "high"
        if self._linearity_payload is not None:
            confidence = float(self._linearity_payload["confidence"][row_index])
            confidence_level = str(self._linearity_payload["confidence_level"][row_index])

        if confidence_level == "high":
            border_color = QColor("#2CC784")
        elif confidence_level == "medium":
            border_color = QColor("#E3A33B")
        else:
            border_color = QColor("#D33E56")

        overlay = [
            f"Slice: {filename}",
            f"Composite confidence: {confidence:.2f} ({confidence_level})",
            f"Oy: {row['oy']:.2f}",
        ]

        if self.enable_blend_overlay_checkbox.isChecked():
            alpha = self.blend_slider.value() / 100.0
            blended = self._build_blended_overlay_image(image_path, self._latest_atlas_slice, alpha)
            if blended is not None:
                overlay.append(f"Atlas blend alpha: {alpha:.2f}")
                self.curation_viewer.set_array_image(
                    blended,
                    overlay_lines=overlay,
                    border_color=border_color,
                )
                return

        self.curation_viewer.set_image(image_path, overlay, border_color=border_color)

    def _request_atlas_preview(self, row_index: int):
        if not hasattr(self, "enable_atlas_preview_checkbox"):
            return
        if not self.enable_atlas_preview_checkbox.isChecked():
            return
        if self.state.predictions is None or len(self.state.predictions) == 0:
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        self._atlas_request_token += 1
        request_token = self._atlas_request_token

        depth_value = None
        if self._linearity_payload is not None and "y" in self._linearity_payload:
            depth_value = float(self._linearity_payload["y"][row_index])
        volume_key = self.atlas_volume_combo.currentText().strip().lower()
        self.atlas_slice_info_label.setText("Atlas: loading...")

        worker = FunctionWorker(
            self._atlas_preview_task,
            depth_value,
            volume_key,
            request_token,
            inject_callbacks=True,
        )
        worker.signals.progress.connect(self._on_atlas_progress)
        worker.signals.log.connect(self._append_console_log)
        worker.signals.error.connect(self._on_atlas_error)
        worker.signals.finished.connect(self._on_atlas_ready)
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _atlas_preview_task(
        self,
        depth_value: Optional[float],
        volume_key: str,
        request_token: int,
        progress_callback=None,
        log_callback=None,
    ):
        result = self.state.get_atlas_slice(
            depth_value=depth_value,
            volume_key=volume_key,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        result["request_token"] = request_token
        return result

    def _on_atlas_progress(self, completed: int, total: int, phase: str):
        if phase == "atlas-download":
            if total > 0:
                percent = (completed / total) * 100.0
                self.atlas_slice_info_label.setText(f"Atlas download: {percent:.1f}%")
            else:
                self.atlas_slice_info_label.setText(f"Atlas download: {completed} bytes")
        elif phase == "atlas-ready":
            self.atlas_slice_info_label.setText("Atlas: rendering")

    def _on_atlas_error(self, error_text: str):
        self._record_error("Atlas preview task failed", error_text)
        self.atlas_slice_info_label.setText("Atlas: failed")
        self._latest_atlas_slice = None
        self._latest_atlas_meta = None
        self.atlas_viewer.clear_with_text("Atlas preview failed. Check console for details.")

    def _on_atlas_ready(self, result: dict):
        if result.get("request_token") != self._atlas_request_token:
            return

        image = result["image"]
        slice_index = result["slice_index"]
        shape = result["shape"]
        volume_label = str(result["volume_label"])
        self.atlas_slice_info_label.setText(
            f"Atlas: {volume_label} | y={slice_index}/{shape[1] - 1}"
        )
        self._latest_atlas_slice = image
        self._latest_atlas_meta = result

        overlay = [
            f"Volume: {volume_label}",
            f"Coronal index (y): {slice_index}",
            f"Shape: {shape}",
        ]
        self.atlas_viewer.set_array_image(image, overlay_lines=overlay)

        if self.enable_blend_overlay_checkbox.isChecked():
            row = self.slice_flag_list.currentRow()
            self._render_histology_preview(row)

    def _refresh_curation_views(self):
        self.slice_flag_list.clear()
        self.linearity_axis.clear()
        self._linearity_payload = None

        if self.state.predictions is None or len(self.state.predictions) == 0:
            self.curation_viewer.clear_with_text("No predictions to curate")
            self.atlas_slice_info_label.setText("Atlas: disabled")
            self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")
            self.linearity_canvas.draw_idle()
            return

        payload = self.state.linearity_payload()
        self._linearity_payload = payload

        for idx, row in self.state.predictions.iterrows():
            nr = row["nr"] if "nr" in row else idx + 1
            filename = row["Filenames"]
            item = QListWidgetItem(f"{nr} | {filename}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)

            is_bad = False
            if "bad_section" in self.state.predictions.columns:
                is_bad = bool(self.state.predictions.iloc[idx]["bad_section"])
            item.setCheckState(Qt.Checked if is_bad else Qt.Unchecked)

            confidence = float(payload["confidence"][idx])
            confidence_level = str(payload["confidence_level"][idx])
            if confidence_level == "high":
                item.setBackground(QColor(44, 129, 79, 120))
            elif confidence_level == "medium":
                item.setBackground(QColor(150, 111, 36, 120))
            else:
                item.setBackground(QColor(164, 51, 68, 120))

            components = payload["confidence_components"]
            item.setToolTip(
                "\n".join(
                    [
                        f"Score: {confidence:.3f} ({confidence_level})",
                        f"Residual: {payload['residuals'][idx]:.3f}",
                        f"Angle deviation: {payload['angle_deviation'][idx]:.3f}",
                        f"Spacing deviation: {payload['spacing_deviation'][idx]:.3f}",
                        f"Gaussian weight: {payload['weights'][idx]:.3f}",
                        f"Components: residual={components['residual'][idx]:.2f}, angle={components['angle'][idx]:.2f}, spacing={components['spacing'][idx]:.2f}, center={components['center_weight'][idx]:.2f}",
                    ]
                )
            )
            item.setData(Qt.UserRole, idx)
            self.slice_flag_list.addItem(item)

        x = payload["x"]
        y = payload["y"]
        trend = payload["trend"]
        weights = payload["weights"]
        outliers = payload["outliers"]

        self.linearity_axis.set_facecolor("#11161D")
        self.linearity_axis.scatter(
            x,
            y,
            c=payload["confidence"],
            cmap="RdYlGn",
            edgecolors="white",
            linewidths=0.5,
            s=36,
            label="Sections",
        )
        self.linearity_axis.plot(x, trend, color="#1E6FFF", linewidth=2.0, label="Linear fit")

        if len(y) > 0:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            if y_max == y_min:
                y_max = y_min + 1.0
            scaled_weights = y_min + (weights * (y_max - y_min))
            self.linearity_axis.plot(
                x,
                scaled_weights,
                color="#52E5FF",
                linestyle="--",
                linewidth=1.6,
                label="Gaussian weighting (scaled)",
            )

        if np.any(outliers):
            self.linearity_axis.scatter(
                x[outliers],
                y[outliers],
                facecolors="none",
                edgecolors="#E03A4F",
                linewidths=1.6,
                s=80,
                label="Outliers",
            )

        self.linearity_axis.set_xlabel("Section Index")
        self.linearity_axis.set_ylabel("Predicted AP Position")
        self.linearity_axis.tick_params(colors="#B9C7D8")
        self.linearity_axis.xaxis.label.set_color("#B9C7D8")
        self.linearity_axis.yaxis.label.set_color("#B9C7D8")
        self.linearity_axis.spines["bottom"].set_color("#3B4655")
        self.linearity_axis.spines["left"].set_color("#3B4655")
        self.linearity_axis.spines["top"].set_color("#3B4655")
        self.linearity_axis.spines["right"].set_color("#3B4655")
        self.linearity_axis.legend(loc="best", facecolor="#1A1F26", edgecolor="#2A313B")

        self.linearity_canvas.draw_idle()

        self.slice_flag_list.setCurrentRow(0)
        self._refresh_export_views()

    def _on_linearity_click(self, event):
        if self._linearity_payload is None or event.xdata is None:
            return
        x_values = self._linearity_payload["x"]
        nearest = int(np.argmin(np.abs(x_values - event.xdata)))
        self.slice_flag_list.setCurrentRow(nearest)
        self._on_curation_slice_selected(nearest)

    def _set_all_flags(self, check_state: Qt.CheckState):
        if self.state.predictions is None:
            return
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            if not item.isHidden():
                item.setCheckState(check_state)
                
    def _filter_curation_list(self, filter_index: int):
        if self.state.predictions is None or self._linearity_payload is None:
            return
        
        levels = self._linearity_payload["confidence_level"]
        
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            source_index = item.data(Qt.UserRole)
            level = levels[source_index]
            
            if filter_index == 0:  # All Confidences
                item.setHidden(False)
            elif filter_index == 1:  # High Only
                item.setHidden(level != "high")
            elif filter_index == 2:  # Medium Only
                item.setHidden(level != "medium")
            elif filter_index == 3:  # Low Only
                item.setHidden(level != "low")

    def _show_curation_context_menu(self, pos):
        item = self.slice_flag_list.itemAt(pos)
        if item is None:
            return
            
        menu = QMenu(self)
        toggle_flag_action = menu.addAction("Toggle Bad Section Flag")
        show_details_action = menu.addAction("Show Confidence Details")
        
        action = menu.exec(self.slice_flag_list.viewport().mapToGlobal(pos))
        
        if action == toggle_flag_action:
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
        elif action == show_details_action:
            QMessageBox.information(self, "Confidence Details", item.toolTip())

    def _on_curation_slice_selected(self, row_index: int):
        if self.state.predictions is None:
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        self._render_histology_preview(row_index)
        self._request_atlas_preview(row_index)

    def _apply_bad_section_flags(self):
        if self.state.predictions is None:
            return

        bad_sections = []
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            if item.checkState() == Qt.Checked:
                source_index = item.data(Qt.UserRole)
                source_index = idx if source_index is None else int(source_index)
                filename = self.state.predictions.iloc[source_index]["Filenames"]
                bad_sections.append(str(filename))

        try:
            self.state.set_bad_sections(bad_sections, auto=False)
        except Exception as exc:
            self._show_logged_exception(
                title="Bad Section Flagging",
                context="Unable to apply bad section flags",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return

        self._refresh_curation_views()

    def _apply_manual_order(self):
        if self.state.predictions is None:
            return

        ordered_indices = []
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            source_index = item.data(Qt.UserRole)
            ordered_indices.append(int(source_index))

        try:
            self.state.apply_manual_order(ordered_indices)
        except Exception as exc:
            self._show_logged_exception(
                title="Manual Reordering",
                context="Unable to apply manual section reordering",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return

        self._refresh_curation_views()

    def _detect_outliers(self):
        if self.state.predictions is None:
            return
        try:
            self.state.set_bad_sections([], auto=True)
        except Exception as exc:
            self._show_logged_exception(
                title="Outlier Detection",
                context="Unable to detect outlier sections",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._refresh_curation_views()

    def _normalize_angles(self):
        if self.state.predictions is None:
            return
        try:
            self.state.propagate_angles()
        except Exception as exc:
            self._show_logged_exception(
                title="Normalize Angles",
                context="Unable to normalize section angles",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._refresh_curation_views()

    def _apply_manual_angles(self):
        if self.state.predictions is None:
            return
        try:
            self.state.adjust_angles(
                ml_angle=float(self.ml_spin.value()),
                dv_angle=float(self.dv_spin.value()),
            )
        except Exception as exc:
            self._show_logged_exception(
                title="Manual Angle Override",
                context="Unable to apply manual angle override",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._refresh_curation_views()

    def _enforce_index_order(self):
        if self.state.predictions is None:
            return
        try:
            self.state.enforce_index_order()
        except Exception as exc:
            self._show_logged_exception(
                title="Enforce Index Order",
                context="Unable to enforce index order",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._refresh_curation_views()

    def _enforce_index_spacing(self):
        if self.state.predictions is None:
            return
        section_thickness = None
        if not self.auto_thickness_checkbox.isChecked():
            section_thickness = float(self.thickness_spin.value())
        try:
            self.state.enforce_index_spacing(section_thickness_um=section_thickness)
        except Exception as exc:
            self._show_logged_exception(
                title="Enforce Index Spacing",
                context="Unable to enforce index spacing",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._refresh_curation_views()

    def _undo(self):
        try:
            self.state.undo()
        except Exception as exc:
            self._show_logged_exception(
                title="Undo",
                context="Undo operation failed",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self._refresh_curation_views()

    def _redo(self):
        try:
            self.state.redo()
        except Exception as exc:
            self._show_logged_exception(
                title="Redo",
                context="Redo operation failed",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self._refresh_curation_views()

    def _browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_quicknii_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Locate QuickNII Executable",
            "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self.quicknii_path_edit.setText(path)

    def _get_persisted_export_path(self) -> str:
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        path = settings.value("export_directory", "")
        return path if path else os.getcwd()

    def _persist_export_path(self, path: str):
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("export_directory", path)

    def _get_persisted_quicknii_path(self) -> str:
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        return settings.value("quicknii_path", "")

    def _persist_quicknii_path(self, path: str):
        from PySide6.QtCore import QSettings
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("quicknii_path", path)

    def _update_export_size_estimate(self, index: int):
        if self.state.predictions is None:
            self.export_size_estimate_label.setText("~0 MB")
            return
            
        num_slices = len(self.state.predictions)
        # Very rough estimates: JSON ~ 50KB/slice, XML ~ 80KB/slice
        kb_per_slice = 50 if index == 0 else 80
        total_mb = (num_slices * kb_per_slice) / 1024
        
        # Add sidecar CSV estimate
        total_mb += (num_slices * 2) / 1024
        
        if total_mb < 0.1:
            self.export_size_estimate_label.setText("< 0.1 MB")
        else:
            self.export_size_estimate_label.setText(f"~{total_mb:.1f} MB")

    def _open_export_directory(self):
        output_dir = self.output_dir_edit.text().strip()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "Export Folder", "The export folder does not exist yet.")
            return
            
        try:
            if os.name == "nt":
                os.startfile(output_dir)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", output_dir])
        except Exception as exc:
            self._show_logged_exception(
                title="Open Export Folder",
                context="Unable to open the export directory",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _copy_export_path(self):
        if self.last_export_basepath is None:
            QMessageBox.information(
                self,
                "Copy Path",
                "Export predictions first to generate an output path.",
            )
            return

        output_format = "json" if self.output_format_combo.currentIndex() == 0 else "xml"
        target_file = self.last_export_basepath + f".{output_format}"
        
        QApplication.clipboard().setText(target_file)
        # Use status bar for non-blocking toast
        self.status_bar.showMessage(f"Copied export path: {target_file}", 3000)

    def _export_predictions(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Export", "No predictions available")
            return

        output_dir = self.output_dir_edit.text().strip()
        base_name = self.output_basename_edit.text().strip()
        if not output_dir or not base_name:
            QMessageBox.warning(
                self,
                "Export",
                "Output directory and base filename are required",
            )
            return

        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, base_name)
        output_format = "json" if self.output_format_combo.currentIndex() == 0 else "xml"

        try:
            self.state.save_predictions(base_path, output_format=output_format)
        except Exception as exc:
            self._show_logged_exception(
                title="Export Failed",
                context="Prediction export failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        self.last_export_basepath = base_path
        self._session_base_text = "Session: Export complete"
        self._update_session_status()
        QMessageBox.information(
            self,
            "Export Complete",
            (
                f"Saved {output_format.upper()} export and CSV sidecar:\n"
                f"{base_path}.{output_format}\n"
                f"{base_path}.csv"
            ),
        )

    def _generate_report(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Report", "No predictions available")
            return

        output_dir = self.output_dir_edit.text().strip()
        base_name = self.output_basename_edit.text().strip() or "DeepSliceResults"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, base_name + "_report.pdf")

        summary = self.state.summary_metrics()
        options = {
            "species": self.state.species,
            "ensemble": bool(self.ensemble_checkbox.isChecked()),
            "use_secondary_model": bool(self.secondary_model_checkbox.isChecked()),
            "section_numbers": bool(self.enable_section_numbers_checkbox.isChecked()),
            "legacy_section_numbers": bool(self.legacy_parsing_checkbox.isChecked()),
            "direction": self.state.selected_indexing_direction,
            "thickness_um": None
            if self.auto_thickness_checkbox.isChecked()
            else float(self.thickness_spin.value()),
            "include_stats": self.pdf_include_stats.isChecked(),
            "include_plot": self.pdf_include_plot.isChecked(),
            "include_images": self.pdf_include_images.isChecked(),
        }

        try:
            reporting.generate_pdf_report(
                output_path=report_path,
                summary=summary,
                options=options,
            )
        except Exception as exc:
            self._show_logged_exception(
                title="Report Failed",
                context="Report generation failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        QMessageBox.information(self, "Report", f"Report created:\n{report_path}")

    def _preview_report(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Report Preview", "No predictions available")
            return
            
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_report_path = os.path.join(temp_dir, "DeepSlice_Report_Preview.pdf")
        
        summary = self.state.summary_metrics()
        options = {
            "species": self.state.species,
            "ensemble": bool(self.ensemble_checkbox.isChecked()),
            "use_secondary_model": bool(self.secondary_model_checkbox.isChecked()),
            "section_numbers": bool(self.enable_section_numbers_checkbox.isChecked()),
            "legacy_section_numbers": bool(self.legacy_parsing_checkbox.isChecked()),
            "direction": self.state.selected_indexing_direction,
            "thickness_um": None
            if self.auto_thickness_checkbox.isChecked()
            else float(self.thickness_spin.value()),
            "include_stats": self.pdf_include_stats.isChecked(),
            "include_plot": self.pdf_include_plot.isChecked(),
            "include_images": self.pdf_include_images.isChecked(),
        }

        self.status_bar.showMessage("Generating report preview...", 5000)
        QApplication.processEvents()

        try:
            reporting.generate_pdf_report(
                output_path=temp_report_path,
                summary=summary,
                options=options,
            )
            
            if os.name == "nt":
                os.startfile(temp_report_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", temp_report_path])
                
        except Exception as exc:
            self._show_logged_exception(
                title="Preview Failed",
                context="Report preview generation failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

    def _open_in_quicknii(self):
        if self.last_export_basepath is None:
            QMessageBox.information(
                self,
                "QuickNII",
                "Export predictions first to generate a file for QuickNII.",
            )
            return

        target_file = self.last_export_basepath + ".json"
        if not os.path.exists(target_file):
            QMessageBox.information(
                self,
                "QuickNII",
                "QuickNII launch expects a JSON export. Export JSON first.",
            )
            return

        quicknii_path = self.quicknii_path_edit.text().strip()
        if not quicknii_path:
            candidates = [
                os.path.join(os.environ.get("ProgramFiles", ""), "QuickNII", "QuickNII.exe"),
                os.path.join(
                    os.environ.get("ProgramFiles(x86)", ""),
                    "QuickNII",
                    "QuickNII.exe",
                ),
            ]
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    quicknii_path = candidate
                    break

        if not quicknii_path or not os.path.exists(quicknii_path):
            QMessageBox.warning(
                self,
                "QuickNII",
                "QuickNII executable not found. Set path in the export panel.",
            )
            return

        try:
            subprocess.Popen([quicknii_path, target_file])
        except Exception as exc:
            self._show_logged_exception(
                title="QuickNII",
                context="Failed to launch QuickNII",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

    def _refresh_export_views(self):
        summary = self.state.summary_metrics()
        self.summary_label.setText(
            f"Processed: {summary['processed']} | Excluded: {summary['excluded']} | Total: {summary['slice_count']}"
        )
        self.deviation_label.setText(
            f"Mean angular deviation: {summary['mean_angular_deviation']:.2f} deg"
        )

        if self.state.predictions is not None and len(self.state.predictions) > 0:
            payload = self.state.linearity_payload()
            levels = payload["confidence_level"]
            high = int(np.sum(levels == "high"))
            medium = int(np.sum(levels == "medium"))
            low = int(np.sum(levels == "low"))
            self.deviation_label.setText(
                f"Mean angular deviation: {summary['mean_angular_deviation']:.2f} deg | Confidence H/M/L: {high}/{medium}/{low}"
            )

        if self.state.predictions is not None and "markers" in self.state.predictions.columns:
            count = int(
                np.sum(
                    [
                        isinstance(marker, (list, tuple)) and len(marker) > 0
                        for marker in self.state.predictions["markers"]
                    ]
                )
            )
            self.markers_label.setText(
                f"Loaded marker annotations found in {count} slices and preserved on JSON export."
            )
        else:
            self.markers_label.setText("")

    def _save_session(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save DeepSlice GUI Session",
            "",
            "DeepSlice Session (*.deepslice-session.json)",
        )
        if not filename:
            return

        if not filename.endswith(".deepslice-session.json"):
            filename = filename + ".deepslice-session.json"

        self.save_session_button.setText("Saving...")
        self.save_session_button.setEnabled(False)
        QApplication.processEvents()

        try:
            payload = self.state.to_session_dict()
            with open(filename, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
        except Exception as exc:
            self.save_session_button.setText("Save Session")
            self.save_session_button.setEnabled(True)
            self._show_logged_exception(
                title="Save Session",
                context="Failed to save DeepSlice session",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        self.save_session_button.setText("Save Session")
        self.save_session_button.setEnabled(True)
        self._session_base_text = f"Session: Saved {os.path.basename(filename)}"
        self._update_session_status()

    def _load_session_or_quint(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session or QuickNII",
            "",
            "Session/QuickNII (*.json *.xml *.deepslice-session.json)",
        )
        if not filename:
            return
        self._load_session_file(filename)

    def _load_session_file(self, filename: str):
        if filename.lower().endswith(".deepslice-session.json"):
            try:
                with open(filename, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
                self.state.load_session_dict(payload)
                self._apply_state_to_widgets()
                self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                self._update_session_status()
                self._refresh_all_views()
                self._add_recent_session(filename)
            except Exception as exc:
                self._show_logged_exception(
                    title="Load Session",
                    context="Failed to load DeepSlice session",
                    exc=exc,
                    icon=QMessageBox.Critical,
                )
            return

        if filename.lower().endswith(".json"):
            try:
                with open(filename, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
                if payload.get("session_format") == "deepslice_gui_v1":
                    self.state.load_session_dict(payload)
                    self._apply_state_to_widgets()
                    self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                    self._update_session_status()
                    self._refresh_all_views()
                    self._add_recent_session(filename)
                    return
            except Exception:
                pass

        worker = FunctionWorker(self._load_quint_task, filename, inject_callbacks=True)
        worker.signals.log.connect(self._append_console_log)
        worker.signals.error.connect(self._on_prediction_error)
        worker.signals.finished.connect(lambda res: self._on_load_quint_finished(res, filename))
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _load_quint_task(self, filename: str, progress_callback=None, log_callback=None):
        return self.state.load_quint(filename, log_callback=log_callback)

    def _on_load_quint_finished(self, result: dict, filename: str = None):
        if self.state.species == "mouse":
            self.mouse_radio.setChecked(True)
        else:
            self.rat_radio.setChecked(True)

        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()
        self._refresh_step_states()

        marker_note = ""
        if result.get("marker_count", 0) > 0:
            marker_note = f"\nMarkers preserved: {result['marker_count']}"

        self._session_base_text = f"Session: Loaded {result['slice_count']} slices ({result['species']})"
        self._update_session_status()
        if filename:
            self._add_recent_session(filename)
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {result['slice_count']} slices as {result['species']}.{marker_note}",
        )

    def _apply_state_to_widgets(self):
        self.mouse_radio.setChecked(self.state.species == "mouse")
        self.rat_radio.setChecked(self.state.species == "rat")
        self._refresh_atlas_volume_options()
        self.enable_section_numbers_checkbox.setChecked(self.state.section_numbers)
        self.legacy_parsing_checkbox.setChecked(self.state.legacy_section_numbers)
        self.legacy_from_config_checkbox.setChecked(self.state.legacy_section_numbers)

        if self.state.selected_indexing_direction in {"rostro-caudal", "caudal-rostro"}:
            self.direction_override_combo.setCurrentText(self.state.selected_indexing_direction)
        else:
            self.direction_override_combo.setCurrentText("Auto")

    def _show_hardware_health(self):
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get("cuda_version", "unknown")
            cudnn_version = build_info.get("cudnn_version", "unknown")

            mode = "GPU" if len(gpus) > 0 else "CPU"
            self.hardware_mode_label.setText(f"Mode: {mode}")

            lines = [
                f"Mode: {mode}",
                f"CUDA: {cuda_version}",
                f"cuDNN: {cudnn_version}",
                f"Detected GPUs: {len(gpus)}",
            ]

            for gpu_idx, gpu in enumerate(gpus):
                lines.append(f"GPU {gpu_idx}: {gpu.name}")
                try:
                    memory_info = tf.config.experimental.get_memory_info(f"GPU:{gpu_idx}")
                    current_mb = memory_info.get("current", 0) / (1024 * 1024)
                    peak_mb = memory_info.get("peak", 0) / (1024 * 1024)
                    lines.append(f"  VRAM current: {current_mb:.1f} MB, peak: {peak_mb:.1f} MB")
                except Exception:
                    lines.append("  VRAM usage unavailable in this TensorFlow build")

            QMessageBox.information(self, "Hardware Health", "\n".join(lines))
        except Exception as exc:
            self._show_logged_exception(
                title="Hardware Health",
                context="Unable to query TensorFlow hardware details",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _update_hardware_mode_label(self):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            mode = "GPU" if len(gpus) > 0 else "CPU"
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get("cuda_version", "unknown")
            cudnn_version = build_info.get("cudnn_version", "unknown")
            self.hardware_mode_label.setToolTip(f"Mode: {mode}\nCUDA: {cuda_version}\ncuDNN: {cudnn_version}")
        except Exception:
            mode = "CPU"
            self.hardware_mode_label.setToolTip("Hardware info not available")
        self.hardware_mode_label.setText(f"Mode: {mode}")

    def _refresh_all_views(self):
        self._update_session_status()
        self._refresh_ingestion_views()
        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()

        if self.state.detected_indexing_direction:
            self.detected_direction_label.setText(
                f"Detected direction: {self.state.detected_indexing_direction}"
            )
            self.prediction_direction_label.setText(
                f"Detected indexing direction: {self.state.detected_indexing_direction}"
            )

        self._refresh_step_states()
        self._update_run_button_state()


def launch_gui():
    app = QApplication.instance() or QApplication([])
    window = DeepSliceMainWindow()
    window.show()
    return app.exec()
