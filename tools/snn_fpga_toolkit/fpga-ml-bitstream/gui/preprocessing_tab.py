"""
preprocessing_tab.py

GUI tab for bitstream preprocessing operations.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QSpinBox, QCheckBox, QComboBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class PreprocessWorker(QThread):
    """Worker thread for preprocessing operations."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_file, output_dir, options):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.options = options

    def run(self):
        """Execute preprocessing pipeline."""
        try:
            from preprocess.sof_to_bits import bytes_to_bits
            from preprocess.extract_sof_sections import extract_design_section
            from preprocess.guess_width_autocorr import guess_width, bits_to_image
            import numpy as np
            from skimage.io import imsave

            self.status.emit("Loading bitstream...")
            self.progress.emit(10)

            # Step 1: Load and extract
            raw = Path(self.input_file).read_bytes()
            design = extract_design_section(raw)
            bits = bytes_to_bits(design)

            self.status.emit("Converting to bits...")
            self.progress.emit(30)

            # Save bits
            bits_path = Path(self.output_dir) / f"{Path(self.input_file).stem}_bits.npz"
            np.savez_compressed(bits_path, bits=bits, meta={'num_bits': len(bits)})

            self.status.emit("Detecting optimal width...")
            self.progress.emit(50)

            # Step 2: Width detection
            if self.options['auto_width']:
                width = guess_width(
                    bits,
                    min_width=self.options['min_width'],
                    max_width=self.options['max_width']
                )
            else:
                width = self.options['manual_width']

            self.status.emit(f"Using width: {width} bits")
            self.progress.emit(70)

            # Step 3: Generate image
            img = bits_to_image(bits, width)
            img_path = Path(self.output_dir) / f"{Path(self.input_file).stem}_image.png"
            imsave(img_path, img)

            self.status.emit("Computing entropy map...")
            self.progress.emit(85)

            # Step 4: Entropy map (if requested)
            if self.options['compute_entropy']:
                from preprocess.visualize_entropy import compute_local_entropy
                entropy_map = compute_local_entropy(img, self.options['window_size'])
                entropy_path = Path(self.output_dir) / f"{Path(self.input_file).stem}_entropy.png"
                imsave(entropy_path, entropy_map)

            self.progress.emit(100)
            self.status.emit("Preprocessing complete!")

            # Return results
            results = {
                'bits_path': str(bits_path),
                'image_path': str(img_path),
                'width': width,
                'num_bits': len(bits),
            }

            if self.options['compute_entropy']:
                results['entropy_path'] = str(entropy_path)

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class PreprocessingTab(QWidget):
    """Tab for bitstream preprocessing."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Input section
        input_group = QGroupBox("Input Bitstream")
        input_layout = QFormLayout()

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Select a .sof or .rbf file...")
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self._browse_input)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_line)
        input_row.addWidget(input_browse_btn)
        input_layout.addRow("File:", input_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Output section
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        self.output_line = QLineEdit()
        self.output_line.setText(str(Path.home() / "fpga_analysis"))
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_line)
        output_row.addWidget(output_browse_btn)
        output_layout.addRow("Directory:", output_row)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Width detection section
        width_group = QGroupBox("Width Detection")
        width_layout = QFormLayout()

        self.auto_width_check = QCheckBox("Automatic width detection")
        self.auto_width_check.setChecked(True)
        self.auto_width_check.toggled.connect(self._toggle_width_mode)
        width_layout.addRow(self.auto_width_check)

        self.manual_width_spin = QSpinBox()
        self.manual_width_spin.setRange(128, 8192)
        self.manual_width_spin.setValue(1024)
        self.manual_width_spin.setSingleStep(64)
        self.manual_width_spin.setEnabled(False)
        width_layout.addRow("Manual width:", self.manual_width_spin)

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(128, 4096)
        self.min_width_spin.setValue(128)
        self.min_width_spin.setSingleStep(64)
        width_layout.addRow("Min width:", self.min_width_spin)

        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(256, 8192)
        self.max_width_spin.setValue(4096)
        self.max_width_spin.setSingleStep(64)
        width_layout.addRow("Max width:", self.max_width_spin)

        width_group.setLayout(width_layout)
        layout.addWidget(width_group)

        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()

        self.entropy_check = QCheckBox("Compute entropy map")
        self.entropy_check.setChecked(True)
        advanced_layout.addRow(self.entropy_check)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(4, 64)
        self.window_size_spin.setValue(16)
        advanced_layout.addRow("Entropy window size:", self.window_size_spin)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Progress section
        progress_layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        layout.addLayout(progress_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.process_btn = QPushButton("🚀 Start Preprocessing")
        self.process_btn.clicked.connect(self._start_preprocessing)
        btn_layout.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_preprocessing)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addStretch()

    def _browse_input(self):
        """Browse for input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Bitstream File",
            str(Path.home()),
            "FPGA Bitstreams (*.sof *.rbf *.bit);;All Files (*)"
        )
        if file_path:
            self.input_line.setText(file_path)

    def _browse_output(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_line.text()
        )
        if dir_path:
            self.output_line.setText(dir_path)

    def _toggle_width_mode(self, auto):
        """Toggle between auto and manual width modes."""
        self.manual_width_spin.setEnabled(not auto)
        self.min_width_spin.setEnabled(auto)
        self.max_width_spin.setEnabled(auto)

    def _start_preprocessing(self):
        """Start the preprocessing operation."""
        # Validate inputs
        input_file = self.input_line.text()
        if not input_file or not Path(input_file).exists():
            self.main_window.log("❌ Error: Please select a valid input file")
            return

        output_dir = self.output_line.text()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Gather options
        options = {
            'auto_width': self.auto_width_check.isChecked(),
            'manual_width': self.manual_width_spin.value(),
            'min_width': self.min_width_spin.value(),
            'max_width': self.max_width_spin.value(),
            'compute_entropy': self.entropy_check.isChecked(),
            'window_size': self.window_size_spin.value(),
        }

        self.main_window.log(f"🔄 Starting preprocessing: {Path(input_file).name}")

        # Create worker thread
        self.worker = PreprocessWorker(input_file, output_dir, options)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.status.connect(self.main_window.log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Start processing
        self.worker.start()

    def _cancel_preprocessing(self):
        """Cancel the preprocessing operation."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.main_window.log("⏹ Preprocessing cancelled")
            self._reset_ui()

    def _on_finished(self, results):
        """Handle preprocessing completion."""
        self.main_window.log("✅ Preprocessing complete!")
        self.main_window.log(f"   Bits: {results['bits_path']}")
        self.main_window.log(f"   Image: {results['image_path']}")
        self.main_window.log(f"   Width: {results['width']} bits")

        if 'entropy_path' in results:
            self.main_window.log(f"   Entropy: {results['entropy_path']}")

        self._reset_ui()

    def _on_error(self, error_msg):
        """Handle preprocessing error."""
        self.main_window.log(f"❌ Preprocessing error: {error_msg}")
        self._reset_ui()

    def _reset_ui(self):
        """Reset UI to ready state."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def set_input_file(self, file_path: str):
        """Set the input file from external source."""
        self.input_line.setText(file_path)
