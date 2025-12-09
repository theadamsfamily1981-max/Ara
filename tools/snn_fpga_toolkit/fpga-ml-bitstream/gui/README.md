# FPGA ML Bitstream Analysis - GUI

Modern PyQt5-based graphical interface for the FPGA ML Bitstream Analysis Toolkit.

## Features

### 📊 Preprocessing Tab
- Load Intel .sof/.rbf bitstream files
- Automatic or manual frame width detection
- Entropy map generation with configurable window size
- Real-time progress tracking
- Visual output browser

### 🤖 Training Tab
- Configure dataset paths (images + labels CSV)
- Adjustable hyperparameters (epochs, batch size, learning rate)
- Real-time training curves (loss and accuracy)
- Model checkpoint management
- Stop/resume training

### 🔍 Inference Tab
- Load trained PyTorch models
- Analyze bitstream images
- Adjustable detection threshold
- Results table with color-coded predictions
- Batch analysis support

### ⚡ Side-Channel Tab
- Synthetic power trace generation (for testing)
- Real-time trace visualization
- Trace classifier training
- Oscilloscope integration (placeholder)

### 🎨 User Interface
- **Dark/Light themes** (Ctrl+T to toggle)
- **Integrated log viewer** for all operations
- **Status bar** with real-time updates
- **Multi-threaded** processing (non-blocking UI)
- **Settings dialog** for paths and preferences

---

## Installation

### 1. Install Dependencies

```bash
# From repository root
pip install -r requirements.txt
```

**Key GUI dependencies:**
- PyQt5 >= 5.15.0
- matplotlib >= 3.4.0
- numpy, torch, scikit-image

### 2. Verify Installation

```bash
python run_gui.py
```

If successful, the GUI will launch with a welcome screen.

---

## Quick Start

### Launch the GUI

```bash
# From repository root
python run_gui.py

# Or if executable:
./run_gui.py
```

### Basic Workflow

#### 1. Preprocess a Bitstream

1. Go to **Preprocessing** tab
2. Click **Browse** to select a `.sof` or `.rbf` file
3. Configure options:
   - ✅ **Automatic width detection** (recommended)
   - Min/max width range: 128-4096 bits
   - ✅ **Compute entropy map** (for CNN input)
   - Window size: 16 (default)
4. Click **🚀 Start Preprocessing**
5. Monitor progress in the status bar and log
6. Outputs saved to specified directory:
   - `*_bits.npz` (raw bit vector)
   - `*_image.png` (2D image)
   - `*_entropy.png` (entropy map)

#### 2. Train a Model

1. Go to **Training** tab
2. Set **Images directory** (e.g., `dataset/images/`)
3. Set **Labels file** (CSV with `id,label` columns)
4. Configure parameters:
   - Epochs: 20 (for quick test)
   - Batch size: 16
   - Learning rate: 0.001
5. Set output model path (e.g., `models/my_model.pt`)
6. Click **🚀 Start Training**
7. Watch real-time training curves
8. Model saved automatically when complete

#### 3. Run Inference

1. Go to **Inference** tab
2. Select **Model file** (trained `.pt` checkpoint)
3. Select **Image file** (preprocessed entropy map)
4. Set **Threshold** (0.5 = balanced, 0.7 = higher confidence)
5. Click **🔍 Analyze**
6. Results appear in table:
   - 🟢 **Green** = Clean
   - 🔴 **Red** = Trojan detected
   - Probability and confidence scores

#### 4. Side-Channel Analysis

1. Go to **Side-Channel** tab
2. Configure trace capture:
   - Device: Synthetic (for testing)
   - Number of traces: 100
   - Trace length: 2000 samples
3. Click **📡 Capture Traces**
4. View captured trace in plot
5. Use **🚀 Train Model** to train trace classifier
6. Use **🔍 Analyze Traces** for classification

---

## GUI Architecture

### Main Components

```
gui/
├── main_window.py          - Main application window
├── preprocessing_tab.py    - Bitstream preprocessing UI
├── training_tab.py         - CNN training UI
├── inference_tab.py        - Inference UI
├── sidechannel_tab.py      - Side-channel analysis UI
├── settings_dialog.py      - Settings dialog
└── __init__.py
```

### Worker Threads

All long-running operations use `QThread` workers to keep the UI responsive:

- **PreprocessWorker**: Runs bitstream → image pipeline
- **TrainingWorker**: Executes CNN training loop
- **InferenceWorker**: Performs model inference

### Signals and Slots

The GUI uses Qt's signal/slot mechanism for thread-safe communication:

```python
worker.progress.connect(progress_bar.setValue)
worker.status.connect(log_viewer.append)
worker.finished.connect(on_complete_handler)
worker.error.connect(on_error_handler)
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open bitstream file |
| `Ctrl+T` | Toggle dark/light theme |
| `Ctrl+,` | Open settings |
| `Ctrl+Q` | Quit application |
| `F1` | Show documentation |

---

## Customization

### Themes

The GUI supports dark and light themes. Toggle with:
- **Menu**: View → Toggle Theme
- **Keyboard**: `Ctrl+T`
- **Settings**: File → Settings → Appearance

### Settings

Access via `File → Settings` (Ctrl+,):

- **Theme**: dark/light
- **Quartus path**: Path to Intel Quartus installation
- **Default output directory**: Where to save processed files

Settings are stored in memory for the current session. Future versions will persist to config file.

---

## Troubleshooting

### GUI won't launch

**Error**: `ModuleNotFoundError: No module named 'PyQt5'`

**Solution**:
```bash
pip install PyQt5>=5.15.0
```

### Plots not displaying

**Error**: Matplotlib backend issues

**Solution**:
```bash
pip install --upgrade matplotlib
# Or try: export MPLBACKEND=Qt5Agg
```

### Preprocessing fails

**Error**: `No module named 'skimage'`

**Solution**:
```bash
pip install scikit-image
```

### Training doesn't start

**Check**:
1. Images directory exists and contains `.png` files
2. Labels CSV has correct format: `id,label`
3. PyTorch is installed: `pip install torch torchvision`

---

## Advanced Usage

### Batch Processing

While the GUI processes one file at a time in the Preprocessing tab, you can:

1. Use CLI tools for batch processing:
   ```bash
   python cli/preprocess_dataset.py --input dataset/raw_sof/ --output dataset/images/
   ```

2. Then use GUI for training/inference

### Integration with Quartus

The GUI doesn't directly call Quartus (use Tcl scripts for that), but you can:

1. Generate bitstreams with `quartus_tcl/generate_clean_designs.tcl`
2. Use GUI to preprocess them
3. Train models in GUI
4. Deploy via CLI or GUI inference

### Custom Models

To use a custom PyTorch model architecture:

1. Modify `ml/models/cnn_bitstream.py`
2. Restart GUI
3. Train tab will use new architecture

---

## Screenshots

*(Screenshots would go here in production)*

### Preprocessing Tab
- File browser, width detection controls, entropy options
- Progress bar and real-time log

### Training Tab
- Hyperparameter configuration
- Live training curves (loss and accuracy)

### Inference Tab
- Model selection, threshold adjustment
- Color-coded results table

### Side-Channel Tab
- Trace capture configuration
- Real-time power trace visualization

---

## Future Enhancements

Planned features for future versions:

- [ ] Persistent settings (save to config file)
- [ ] Batch inference with file glob patterns
- [ ] Export results to CSV/JSON
- [ ] Integrated visualization (bitstream + entropy side-by-side)
- [ ] Model architecture viewer
- [ ] Confusion matrix and ROC curve plots
- [ ] Dataset statistics dashboard
- [ ] Quartus integration (direct Tcl execution)
- [ ] Oscilloscope drivers for real side-channel capture
- [ ] Plugin system for custom preprocessing/analysis

---

## API for Developers

### Adding a New Tab

```python
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class MyCustomTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        # Add widgets here
```

Then in `main_window.py`:

```python
from gui.my_custom_tab import MyCustomTab

# In MainWindow._init_ui():
self.my_tab = MyCustomTab(self)
self.tabs.addTab(self.my_tab, "🔧 My Tab")
```

### Logging Messages

```python
self.main_window.log("Message to log viewer")
self.main_window.update_status("Status bar message")
```

### Creating Worker Threads

```python
from PyQt5.QtCore import QThread, pyqtSignal

class MyWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def run(self):
        # Do work
        self.progress.emit(50)
        # More work
        self.finished.emit({'result': 'data'})
```

---

## License

Same as main project (MIT License).

---

## Support

For GUI-specific issues:
1. Check this README
2. Review error messages in the log viewer
3. Try CLI tools to isolate the issue
4. Check PyQt5 and matplotlib versions

For general toolkit issues, see main `README.md`.
