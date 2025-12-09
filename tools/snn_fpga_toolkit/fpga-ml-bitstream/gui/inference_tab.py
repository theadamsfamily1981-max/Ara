"""
inference_tab.py

GUI tab for running inference on bitstreams.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QDoubleSpinBox, QFormLayout, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class InferenceWorker(QThread):
    """Worker thread for inference."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_path, image_path, threshold):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.threshold = threshold

    def run(self):
        """Run inference."""
        try:
            self.status.emit("Loading model...")
            self.progress.emit(25)

            # Simulate inference
            import time
            time.sleep(0.5)

            self.status.emit("Running inference...")
            self.progress.emit(75)

            # Fake result
            import random
            probability = random.random()
            prediction = 1 if probability > self.threshold else 0

            result = {
                'file': Path(self.image_path).name,
                'prediction': 'Trojan' if prediction == 1 else 'Clean',
                'probability': probability,
                'confidence': abs(probability - 0.5) * 2,
            }

            self.progress.emit(100)
            self.result.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class InferenceTab(QWidget):
    """Tab for inference operations."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Model section
        model_group = QGroupBox("Model")
        model_layout = QFormLayout()

        self.model_line = QLineEdit()
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_line)
        model_row.addWidget(model_browse)
        model_layout.addRow("Model file:", model_row)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        model_layout.addRow("Threshold:", self.threshold_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Input section
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()

        self.image_line = QLineEdit()
        image_browse = QPushButton("Browse...")
        image_browse.clicked.connect(self._browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_line)
        image_row.addWidget(image_browse)
        input_layout.addRow("Image file:", image_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["File", "Prediction", "Probability", "Confidence"])
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 Analyze")
        self.analyze_btn.clicked.connect(self._start_inference)
        btn_layout.addWidget(self.analyze_btn)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.results_table.clearContents)
        btn_layout.addWidget(self.clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_line.setText(file_path)

    def _browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg)")
        if file_path:
            self.image_line.setText(file_path)

    def _start_inference(self):
        """Start inference."""
        model_path = self.model_line.text()
        image_path = self.image_line.text()

        if not model_path or not image_path:
            self.main_window.log("❌ Please select model and image")
            return

        self.main_window.log(f"🔍 Analyzing: {Path(image_path).name}")

        self.worker = InferenceWorker(model_path, image_path, self.threshold_spin.value())
        self.worker.status.connect(self.main_window.log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result.connect(self._add_result)
        self.worker.error.connect(lambda e: self.main_window.log(f"❌ Error: {e}"))

        self.analyze_btn.setEnabled(False)
        self.worker.finished.connect(lambda: self.analyze_btn.setEnabled(True))
        self.worker.start()

    def _add_result(self, result):
        """Add result to table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 0, QTableWidgetItem(result['file']))

        pred_item = QTableWidgetItem(result['prediction'])
        if result['prediction'] == 'Trojan':
            pred_item.setBackground(QColor(255, 200, 200))
        else:
            pred_item.setBackground(QColor(200, 255, 200))
        self.results_table.setItem(row, 1, pred_item)

        self.results_table.setItem(row, 2, QTableWidgetItem(f"{result['probability']:.4f}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{result['confidence']:.2%}"))

        self.results_table.resizeColumnsToContents()
        self.main_window.log(f"✅ Result: {result['prediction']} (p={result['probability']:.4f})")
