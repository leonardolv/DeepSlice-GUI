import os

file_path = "DeepSlice/gui/main_window.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add imports
if "QShortcut" not in content:
    content = content.replace("from PySide6.QtGui import ", "from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent, ")
if "QMenu" not in content:
    content = content.replace("from PySide6.QtWidgets import (", "from PySide6.QtWidgets import (\n    QMenu,\n    QMessageBox,")

# 2. Add init setup
content = content.replace(
    "self.state = DeepSliceAppState()",
    "self.state = DeepSliceAppState()\n        self._session_base_text = \"Session: New\"\n        self._setup_shortcuts()"
)

# 3. Replace _build_top_bar
old_top_bar_start = "    def _build_top_bar(self) -> QWidget:"
old_top_bar_end = "        return frame"

start_idx = content.find(old_top_bar_start)
end_idx = content.find(old_top_bar_end, start_idx) + len(old_top_bar_end)

new_top_bar = """    def _build_top_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TopBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.project_label = QLabel("DeepSlice Desktop")
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
        self.load_session_button.clicked.connect(self._load_session_or_quint)

        self.error_menu_button = QPushButton("Errors")
        self.error_menu = QMenu(self.error_menu_button)
        
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
        return frame"""

content = content[:start_idx] + new_top_bar + content[end_idx:]

# 4. Add new methods at the end of class (before _record_error or similar)
new_methods = """
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
            "Keyboard Shortcuts:\\n\\n"
            "Ctrl+1 to Ctrl+5 : Navigate between pages\\n"
            "Ctrl+Z : Undo curation changes\\n"
            "Ctrl+Y : Redo curation changes\\n"
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
            
    def _record_error"""

content = content.replace("    def _record_error", new_methods)

# 5. Fix setTexts for session label (explicit replacements)
reps = [
    (
        '''self.session_status_label.setText(
            f"Session: Predicted {result['slice_count']} slices"
        )''',
        '''self._session_base_text = f"Session: Predicted {result['slice_count']} slices"
        self._update_session_status()'''
    ),
    (
        '''self.session_status_label.setText("Session: Export complete")''',
        '''self._session_base_text = "Session: Export complete"
        self._update_session_status()'''
    ),
    (
        '''self.session_status_label.setText(f"Session: Saved {os.path.basename(filename)}")''',
        '''self._session_base_text = f"Session: Saved {os.path.basename(filename)}"
        self._update_session_status()'''
    ),
    (
        '''self.session_status_label.setText(
                    f"Session: Loaded {os.path.basename(filename)}"
                )''',
        '''self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                self._update_session_status()'''
    ),
    (
        '''self.session_status_label.setText(
                        f"Session: Loaded {os.path.basename(filename)}"
                    )''',
        '''self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                    self._update_session_status()'''
    ),
    (
        '''self.session_status_label.setText(
            f"Session: Loaded {result['slice_count']} slices ({result['species']})"
        )''',
        '''self._session_base_text = f"Session: Loaded {result['slice_count']} slices ({result['species']})"
        self._update_session_status()'''
    )
]

for old_str, new_str in reps:
    content = content.replace(old_str, new_str)


# 6. Update hardware tooltip
content = content.replace(
    "def _update_hardware_mode_label(self):\n        try:\n            import tensorflow as tf\n\n            mode = \"GPU\" if len(tf.config.list_physical_devices(\"GPU\")) > 0 else \"CPU\"\n        except Exception:\n            mode = \"CPU\"\n        self.hardware_mode_label.setText(f\"Mode: {mode}\")",
    """def _update_hardware_mode_label(self):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            mode = "GPU" if len(gpus) > 0 else "CPU"
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get("cuda_version", "unknown")
            cudnn_version = build_info.get("cudnn_version", "unknown")
            self.hardware_mode_label.setToolTip(f"Mode: {mode}\\nCUDA: {cuda_version}\\ncuDNN: {cudnn_version}")
        except Exception:
            mode = "CPU"
            self.hardware_mode_label.setToolTip("Hardware info not available")
        self.hardware_mode_label.setText(f"Mode: {mode}")"""
)

# Call update status on refresh_all_views so it picks up the asterisk
content = content.replace(
    "def _refresh_all_views(self):",
    "def _refresh_all_views(self):\n        self._update_session_status()"
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Main Window Patch Applied.")
