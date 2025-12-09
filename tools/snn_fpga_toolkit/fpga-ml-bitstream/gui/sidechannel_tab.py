"""
sidechannel_tab.py

GUI tab for side-channel power trace analysis.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QSpinBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TracePlot(FigureCanvas):
    """Matplotlib canvas for power traces."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Power Trace")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Power (a.u.)")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

    def plot_trace(self, trace):
        """Plot a power trace."""
        self.ax.clear()
        self.ax.plot(trace, linewidth=0.5)
        self.ax.set_title("Power Trace")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Power (a.u.)")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class SideChannelTab(QWidget):
    """Tab for side-channel analysis."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Capture section
        capture_group = QGroupBox("Trace Capture")
        capture_layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Synthetic (Demo)", "Oscilloscope (TBD)", "Custom ADC (TBD)"])
        capture_layout.addRow("Device:", self.device_combo)

        self.num_traces_spin = QSpinBox()
        self.num_traces_spin.setRange(1, 10000)
        self.num_traces_spin.setValue(100)
        capture_layout.addRow("Number of traces:", self.num_traces_spin)

        self.trace_len_spin = QSpinBox()
        self.trace_len_spin.setRange(100, 100000)
        self.trace_len_spin.setValue(2000)
        capture_layout.addRow("Trace length:", self.trace_len_spin)

        self.output_traces_line = QLineEdit()
        self.output_traces_line.setText(str(Path.home() / "fpga_analysis" / "traces"))
        traces_browse = QPushButton("Browse...")
        traces_browse.clicked.connect(self._browse_traces_output)
        traces_row = QHBoxLayout()
        traces_row.addWidget(self.output_traces_line)
        traces_row.addWidget(traces_browse)
        capture_layout.addRow("Output directory:", traces_row)

        self.capture_btn = QPushButton("📡 Capture Traces")
        self.capture_btn.clicked.connect(self._capture_traces)
        capture_layout.addRow(self.capture_btn)

        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)

        # Visualization
        self.trace_plot = TracePlot(self)
        layout.addWidget(self.trace_plot)

        # Analysis section
        analysis_group = QGroupBox("Trace Analysis")
        analysis_layout = QFormLayout()

        self.traces_dir_line = QLineEdit()
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_traces_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.traces_dir_line)
        dir_row.addWidget(dir_browse)
        analysis_layout.addRow("Traces directory:", dir_row)

        self.model_line = QLineEdit()
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_line)
        model_row.addWidget(model_browse)
        analysis_layout.addRow("Model file:", model_row)

        btn_layout = QHBoxLayout()

        self.train_traces_btn = QPushButton("🚀 Train Model")
        self.train_traces_btn.clicked.connect(self._train_model)
        btn_layout.addWidget(self.train_traces_btn)

        self.analyze_traces_btn = QPushButton("🔍 Analyze Traces")
        self.analyze_traces_btn.clicked.connect(self._analyze_traces)
        btn_layout.addWidget(self.analyze_traces_btn)

        analysis_layout.addRow(btn_layout)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

    def _browse_traces_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_traces_line.setText(dir_path)

    def _browse_traces_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Traces Directory")
        if dir_path:
            self.traces_dir_line.setText(dir_path)

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_line.setText(file_path)

    def _capture_traces(self):
        """Capture power traces."""
        self.main_window.log("📡 Capturing power traces (synthetic demo)...")

        try:
            from sidechannel.capture_stub import capture_traces
            import numpy as np

            output_dir = Path(self.output_traces_line.text())
            num_traces = self.num_traces_spin.value()
            trace_len = self.trace_len_spin.value()

            # Capture synthetic traces
            capture_traces(num_traces, "demo", output_dir, trace_len)

            # Load and display one trace
            trace_file = list(output_dir.glob("*.npy"))[0]
            trace = np.load(trace_file)
            self.trace_plot.plot_trace(trace)

            self.main_window.log(f"✅ Captured {num_traces} traces to {output_dir}")

        except Exception as e:
            self.main_window.log(f"❌ Capture error: {e}")

    def _train_model(self):
        """Train trace classifier."""
        self.main_window.log("🚀 Training trace classifier...")
        self.main_window.log("⚠️ Feature not yet implemented - see sidechannel/train_trace_classifier.py")

    def _analyze_traces(self):
        """Analyze captured traces."""
        self.main_window.log("🔍 Analyzing traces...")
        self.main_window.log("⚠️ Feature not yet implemented - see sidechannel/train_trace_classifier.py")
