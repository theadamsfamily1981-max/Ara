#!/usr/bin/env python3
"""
main_window.py

Main GUI window for FPGA ML Bitstream Analysis Toolkit.

Features:
- Multi-tab interface for all system functions
- Integrated logging and progress tracking
- Dark/light theme support
- Real-time visualization
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QLabel, QStatusBar, QAction, QMenuBar,
    QFileDialog, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.preprocessing_tab import PreprocessingTab
from gui.training_tab import TrainingTab
from gui.inference_tab import InferenceTab
from gui.sidechannel_tab import SideChannelTab
from gui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPGA ML Bitstream Analysis Toolkit")
        self.setGeometry(100, 100, 1400, 900)

        # Load settings
        self.settings = {
            'theme': 'dark',
            'quartus_path': '',
            'default_output_dir': str(Path.home() / 'fpga_analysis'),
        }

        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create menu bar
        self._create_menu_bar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Add tabs
        self.preprocessing_tab = PreprocessingTab(self)
        self.training_tab = TrainingTab(self)
        self.inference_tab = InferenceTab(self)
        self.sidechannel_tab = SideChannelTab(self)

        self.tabs.addTab(self.preprocessing_tab, "📊 Preprocessing")
        self.tabs.addTab(self.training_tab, "🤖 Training")
        self.tabs.addTab(self.inference_tab, "🔍 Inference")
        self.tabs.addTab(self.sidechannel_tab, "⚡ Side-Channel")

        # Create splitter for main content and log viewer
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)

        # Create log viewer
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_label = QLabel("📋 System Log")
        log_label.setFont(QFont("Arial", 10, QFont.Bold))
        log_layout.addWidget(log_label)

        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(200)
        log_layout.addWidget(self.log_viewer)

        splitter.addWidget(log_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Bitstream...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_bitstream)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_theme_action = QAction("Toggle &Theme", self)
        toggle_theme_action.setShortcut("Ctrl+T")
        toggle_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme_action)

        clear_log_action = QAction("&Clear Log", self)
        clear_log_action.triggered.connect(self.log_viewer.clear)
        view_menu.addAction(clear_log_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._show_settings)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self._show_docs)
        help_menu.addAction(docs_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_theme(self):
        """Apply the selected theme."""
        if self.settings['theme'] == 'dark':
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

    def _apply_dark_theme(self):
        """Apply dark theme stylesheet."""
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #252525;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #d4d4d4;
            padding: 8px 20px;
            border: 1px solid #3c3c3c;
        }
        QTabBar::tab:selected {
            background-color: #007acc;
        }
        QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            padding: 5px;
        }
        QPushButton {
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #007acc;
        }
        QLabel {
            color: #d4d4d4;
        }
        QGroupBox {
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def _apply_light_theme(self):
        """Apply light theme (default Qt style)."""
        self.setStyleSheet("")

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.settings['theme'] == 'dark':
            self.settings['theme'] = 'light'
        else:
            self.settings['theme'] = 'dark'
        self._apply_theme()

    def _open_bitstream(self):
        """Open a bitstream file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Bitstream File",
            str(Path.home()),
            "FPGA Bitstreams (*.sof *.rbf *.bit);;All Files (*)"
        )
        if file_path:
            self.log(f"Opened: {file_path}")
            # Set the file in preprocessing tab
            self.preprocessing_tab.set_input_file(file_path)

    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_():
            self.settings = dialog.get_settings()
            self._apply_theme()
            self.log("Settings updated")

    def _show_docs(self):
        """Show documentation."""
        docs_path = Path(__file__).parent.parent / "README.md"
        if docs_path.exists():
            QMessageBox.information(
                self,
                "Documentation",
                f"Documentation available at:\n{docs_path}\n\n"
                "See also:\n"
                "- QUICKSTART.md\n"
                "- docs/paper_draft.md\n"
                "- docs/hardware_notes_k10_a10.md"
            )
        else:
            QMessageBox.information(
                self,
                "Documentation",
                "Documentation not found. Check the repository root for README.md"
            )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FPGA ML Bitstream Analysis",
            "<h2>FPGA ML Bitstream Analysis Toolkit</h2>"
            "<p>Version 1.0</p>"
            "<p>ML-assisted analysis of Intel FPGA bitstreams for "
            "Hardware Trojan detection.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Static bitstream forensics using CNNs</li>"
            "<li>Automated Trojan dataset generation</li>"
            "<li>Side-channel power trace analysis</li>"
            "<li>Support for Arria 10 and Stratix 10</li>"
            "</ul>"
            "<p><b>Target Hardware:</b></p>"
            "<ul>"
            "<li>BittWare A10PED (Arria 10)</li>"
            "<li>Superscalar K10 / ColEngine P2 (Stratix 10)</li>"
            "</ul>"
        )

    def log(self, message: str):
        """Add a message to the log viewer."""
        self.log_viewer.append(f"[{self._get_timestamp()}] {message}")
        # Auto-scroll to bottom
        self.log_viewer.verticalScrollBar().setValue(
            self.log_viewer.verticalScrollBar().maximum()
        )

    def _get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def update_status(self, message: str):
        """Update the status bar."""
        self.status_bar.showMessage(message)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("FPGA ML Bitstream Analysis")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
