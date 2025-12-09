# FPGA ML Bitstream - Quickstart Guide

Get the system running in 15 minutes.

---

## 🎨 GUI Available!

**NEW**: A modern graphical interface is now available for easy-to-use operation!

```bash
# Launch the GUI
python run_gui.py
```

Features:
- 📊 **Preprocessing tab** with automatic width detection
- 🤖 **Training tab** with real-time curves
- 🔍 **Inference tab** with visual results
- ⚡ **Side-channel tab** for power trace analysis
- 🌙 **Dark/light themes** (Ctrl+T to toggle)

See `gui/README.md` for full GUI documentation.

---

## Prerequisites

- **Python 3.8+** with pip
- **Intel Quartus Prime Pro** (for dataset generation only)
- **BittWare A10PED** or similar Arria 10/Stratix 10 board (optional, for real data)

---

## 1. Setup Environment (2 minutes)

```bash
# Clone or navigate to the repository
cd fpga-ml-bitstream

# Run automated setup
./setup.sh

# OR manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Test Preprocessing Pipeline (5 minutes)

### Option A: Using Real .SOF Files

If you have a `.sof` file from Quartus:

```bash
# Step 1: Extract bits from .sof
python preprocess/sof_to_bits.py \
    --input /path/to/your/design.sof \
    --output dataset/parsed_npz/design_bits.npz

# Step 2: Detect optimal width and generate image
python preprocess/guess_width_autocorr.py \
    --input dataset/parsed_npz/design_bits.npz \
    --output dataset/images/design.png \
    --width-out design_width.txt

# Step 3: Visualize entropy
python preprocess/visualize_entropy.py \
    --input dataset/images/design.png \
    --output dataset/images/design_entropy.png \
    --window-size 16
```

### Option B: Using Synthetic Data

Test without Quartus:

```bash
# Generate synthetic bitstream (random data as proof-of-concept)
python3 -c "
import numpy as np
np.savez('dataset/parsed_npz/synthetic_bits.npz',
         bits=np.random.randint(0, 2, size=1000000, dtype=np.uint8))
"

# Process it
python preprocess/guess_width_autocorr.py \
    --input dataset/parsed_npz/synthetic_bits.npz \
    --output dataset/images/synthetic.png
```

---

## 3. Train a Simple CNN (5 minutes)

Run the proof-of-concept notebook:

```bash
jupyter notebook notebooks/03_cnn_proof_of_concept.ipynb
```

This trains a CNN on synthetic data to verify the ML pipeline works.

---

## 4. Generate Real Dataset with Quartus (Longer)

### Generate Clean Designs

```bash
# From Quartus shell
quartus_sh -t quartus_tcl/generate_clean_designs.tcl \
    my_project \
    my_revision \
    10AX115S2F45I1SG \
    dataset/raw_sof/clean
```

This compiles your design with 10 different fitter seeds.

### Inject Trojans (Experimental)

```bash
quartus_sh -t quartus_tcl/inject_trojan_eco.tcl \
    my_project \
    base_revision \
    trojan_revision \
    dataset/raw_sof/trojan
```

**Note**: The ECO Trojan injection is a scaffold. You'll need to implement device-specific logic (see TODOs in the script).

### Preprocess All Bitstreams

```bash
# Process all clean designs
for sof in dataset/raw_sof/clean/*.sof; do
    base=$(basename "$sof" .sof)
    python preprocess/sof_to_bits.py --input "$sof" --output "dataset/parsed_npz/${base}_bits.npz"
    python preprocess/guess_width_autocorr.py --input "dataset/parsed_npz/${base}_bits.npz" --output "dataset/images/${base}.png"
done

# Process all trojan designs
for sof in dataset/raw_sof/trojan/*.sof; do
    base=$(basename "$sof" .sof)
    python preprocess/sof_to_bits.py --input "$sof" --output "dataset/parsed_npz/${base}_bits.npz"
    python preprocess/guess_width_autocorr.py --input "dataset/parsed_npz/${base}_bits.npz" --output "dataset/images/${base}.png"
done
```

### Create Labels File

```bash
cat > dataset/labels.csv << EOF
id,label
clean_seed1,clean
clean_seed2,clean
trojan_001,trojan
trojan_002,trojan
EOF
```

### Train CNN

```bash
python ml/train_classifier.py \
    --images-root dataset/images \
    --labels-file dataset/labels.csv \
    --output models/bitstream_cnn.pt \
    --epochs 20 \
    --batch-size 16
```

---

## 5. Run Inference on Unknown Bitstream

```bash
python ml/eval_on_bitstreams.py \
    --model models/bitstream_cnn.pt \
    --images "dataset/images/unknown_*.png" \
    --output results.csv

# View results
cat results.csv
```

---

## 6. Side-Channel Analysis (For Encrypted Designs)

If you have a Stratix 10 with encrypted bitstreams (e.g., K10 miner):

```bash
# Generate synthetic traces for testing
python sidechannel/capture_stub.py

# Preprocess traces
python sidechannel/preprocess_traces.py \
    --input-dir test_traces/clean \
    --output-dir processed_traces

# Train trace classifier
python sidechannel/train_trace_classifier.py \
    --traces-dir processed_traces \
    --output models/trace_cnn.pt \
    --epochs 20
```

**Note**: `capture_stub.py` generates synthetic traces. For real traces, you'll need oscilloscope integration (see TODOs in the file).

---

## Common Issues

### "ModuleNotFoundError: No module named 'torch'"

Make sure you activated the virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "No such file or directory: .sof"

The `.sof` file path must be absolute or relative to the current directory. Check with:

```bash
ls dataset/raw_sof/
```

### Width detection gives poor results

Try different width ranges:

```bash
python preprocess/guess_width_autocorr.py \
    --input design_bits.npz \
    --output design.png \
    --min-width 512 \
    --max-width 4096
```

### Quartus Tcl scripts fail

- Make sure you're in a Quartus shell: `quartus_sh --shell`
- Verify project exists: `quartus_sh --project_list`
- Check device name: `quartus_sh --help=device`

---

## Next Steps

1. **Explore notebooks**: `jupyter notebook notebooks/`
2. **Read docs**: `docs/paper_draft.md` for technical details
3. **Refine TODOs**: Many scripts have `# TODO` comments for improvements
4. **Test on K10**: Extract firmware from SD card and analyze

---

## Repository Structure

```
fpga-ml-bitstream/
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
├── setup.sh                  # Automated setup
├── requirements.txt          # Python dependencies
├── quartus_tcl/              # Quartus automation
│   ├── generate_clean_designs.tcl
│   ├── inject_trojan_eco.tcl
│   └── batch_compile.tcl
├── preprocess/               # Bitstream preprocessing
│   ├── extract_sof_sections.py
│   ├── sof_to_bits.py
│   ├── guess_width_autocorr.py
│   ├── bits_to_image.py
│   └── visualize_entropy.py
├── ml/                       # CNN training/inference
│   ├── models/
│   │   └── cnn_bitstream.py
│   ├── train_classifier.py
│   └── eval_on_bitstreams.py
├── sidechannel/              # Power trace analysis
│   ├── capture_stub.py
│   ├── preprocess_traces.py
│   └── train_trace_classifier.py
├── notebooks/                # Jupyter experiments
│   ├── 01_width_detection_experiments.ipynb
│   ├── 02_entropy_and_visuals.ipynb
│   └── 03_cnn_proof_of_concept.ipynb
├── docs/                     # Documentation
│   ├── paper_draft.md
│   └── hardware_notes_k10_a10.md
└── dataset/                  # Data (git-ignored)
    ├── raw_sof/
    ├── parsed_npz/
    └── images/
```

---

## Support

For issues or questions:

1. Check `docs/paper_draft.md` for technical background
2. Read inline comments and TODOs in the code
3. Review notebooks for examples

Happy hacking! 🔓
