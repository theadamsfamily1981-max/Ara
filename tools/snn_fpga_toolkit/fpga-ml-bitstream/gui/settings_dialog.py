"""
settings_dialog.py

Settings dialog for the application.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QComboBox, QFileDialog,
    QDialogButtonBox, QGroupBox
)


class SettingsDialog(QDialog):
    """Settings dialog."""

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.settings = current_settings.copy()
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.settings['theme'])
        appearance_layout.addRow("Theme:", self.theme_combo)

        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)

        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QFormLayout()

        self.quartus_line = QLineEdit(self.settings['quartus_path'])
        quartus_browse = QPushButton("Browse...")
        quartus_browse.clicked.connect(self._browse_quartus)
        quartus_row = QHBoxLayout()
        quartus_row.addWidget(self.quartus_line)
        quartus_row.addWidget(quartus_browse)
        paths_layout.addRow("Quartus path:", quartus_row)

        self.output_line = QLineEdit(self.settings['default_output_dir'])
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_line)
        output_row.addWidget(output_browse)
        paths_layout.addRow("Default output:", output_row)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_quartus(self):
        """Browse for Quartus installation."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Quartus Installation Directory",
            self.quartus_line.text() or "/opt/intel/quartus"
        )
        if dir_path:
            self.quartus_line.setText(dir_path)

    def _browse_output(self):
        """Browse for default output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Default Output Directory",
            self.output_line.text()
        )
        if dir_path:
            self.output_line.setText(dir_path)

    def get_settings(self):
        """Get the updated settings."""
        return {
            'theme': self.theme_combo.currentText(),
            'quartus_path': self.quartus_line.text(),
            'default_output_dir': self.output_line.text(),
        }
