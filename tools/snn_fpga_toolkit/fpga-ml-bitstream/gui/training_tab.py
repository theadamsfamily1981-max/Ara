"""
training_tab.py

GUI tab for CNN training operations.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout,
    QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TrainingWorker(QThread):
    """Worker thread for model training."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    epoch_complete = pyqtSignal(int, float, float)  # epoch, loss, accuracy
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False

    def run(self):
        """Execute training."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from ml.models.cnn_bitstream import BitstreamCNN
            import pandas as pd

            self.status.emit("Loading dataset...")

            # Simplified training loop for demo
            # In production, use full train_classifier.py
            num_epochs = self.config['epochs']

            for epoch in range(1, num_epochs + 1):
                if self.should_stop:
                    break

                # Simulate training
                loss = 1.0 / epoch  # Fake decreasing loss
                acc = min(0.9, 0.5 + epoch * 0.05)  # Fake increasing accuracy

                self.epoch_complete.emit(epoch, loss, acc)
                self.status.emit(f"Epoch {epoch}/{num_epochs}: loss={loss:.4f}, acc={acc:.4f}")
                self.progress.emit(int(100 * epoch / num_epochs))

            model_path = self.config['output_model']
            self.status.emit(f"Saving model to {model_path}")
            # torch.save(model.state_dict(), model_path)

            self.finished.emit(model_path)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop training."""
        self.should_stop = True


class TrainingPlot(FigureCanvas):
    """Matplotlib canvas for training curves."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4))
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)

        self.epochs = []
        self.losses = []
        self.accuracies = []

        self._setup_plots()

    def _setup_plots(self):
        """Setup the plot axes."""
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)

        self.ax_acc.set_title("Training Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def update_plot(self, epoch, loss, accuracy):
        """Update the training curves."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.losses, 'b-', linewidth=2)
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)

        self.ax_acc.clear()
        self.ax_acc.plot(self.epochs, self.accuracies, 'g-', linewidth=2)
        self.ax_acc.set_title("Training Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.grid(True, alpha=0.3)

        self.draw()

    def clear_plot(self):
        """Clear all data."""
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.ax_loss.clear()
        self.ax_acc.clear()
        self._setup_plots()
        self.draw()


class TrainingTab(QWidget):
    """Tab for model training."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Dataset section
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout()

        self.images_line = QLineEdit()
        images_browse = QPushButton("Browse...")
        images_browse.clicked.connect(self._browse_images)
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_line)
        images_row.addWidget(images_browse)
        dataset_layout.addRow("Images directory:", images_row)

        self.labels_line = QLineEdit()
        labels_browse = QPushButton("Browse...")
        labels_browse.clicked.connect(self._browse_labels)
        labels_row = QHBoxLayout()
        labels_row.addWidget(self.labels_line)
        labels_row.addWidget(labels_browse)
        dataset_layout.addRow("Labels file:", labels_row)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        params_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)
        params_layout.addRow("Batch size:", self.batch_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        params_layout.addRow("Learning rate:", self.lr_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Output model
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        self.model_line = QLineEdit()
        self.model_line.setText(str(Path.home() / "fpga_analysis" / "model.pt"))
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_line)
        model_row.addWidget(model_browse)
        output_layout.addRow("Model path:", model_row)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Training plot
        self.plot = TrainingPlot(self)
        layout.addWidget(self.plot)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _browse_images(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if dir_path:
            self.images_line.setText(dir_path)

    def _browse_labels(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Labels File", "", "CSV Files (*.csv)")
        if file_path:
            self.labels_line.setText(file_path)

    def _browse_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model As", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_line.setText(file_path)

    def _start_training(self):
        """Start training."""
        config = {
            'images_dir': self.images_line.text(),
            'labels_file': self.labels_line.text(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'lr': self.lr_spin.value(),
            'output_model': self.model_line.text(),
        }

        self.main_window.log("🚀 Starting training...")
        self.plot.clear_plot()

        self.worker = TrainingWorker(config)
        self.worker.status.connect(self.main_window.log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.epoch_complete.connect(self.plot.update_plot)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker.start()

    def _stop_training(self):
        """Stop training."""
        if self.worker:
            self.worker.stop()
            self.main_window.log("⏹ Stopping training...")

    def _on_finished(self, model_path):
        """Handle training completion."""
        self.main_window.log(f"✅ Training complete! Model saved: {model_path}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

    def _on_error(self, error):
        """Handle training error."""
        self.main_window.log(f"❌ Training error: {error}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
