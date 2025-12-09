# fpga-ml-bitstream

ML-assisted analysis of Intel FPGA bitstreams (Arria 10 / Stratix 10), with a focus on:

* **Static bitstream forensics** on `.sof` / PR bitstreams using CNNs.
* **Automated Trojan injection** via Quartus ECO flow to generate labeled datasets.
* **Side-channel analysis** (power traces) for fully encrypted designs.
* Real-world targets: BittWare **A10PED** (Arria 10) and **Superscalar K10 / ColEngine P2** (Stratix 10).

The repo implements the system described in:

> "ML-Assisted Intel FPGA Bitstream Analysis"
> (internal draft PDF, e.g. `/mnt/data/ML-Assisted Intel FPGA Bitstream Analysis.pdf`)

---

## High-level architecture

There are three main loops:

1. **Dataset generation (Quartus side)**
   *Use Intel Quartus Pro + Tcl scripts.*

   - `quartus_tcl/generate_clean_designs.tcl`
     Compiles many variants of a base design (different seeds, options).

   - `quartus_tcl/inject_trojan_eco.tcl`
     Uses the ECO flow (`::quartus::eco`) to automatically insert small Trojans
     (extra LUTs, altered connections) and recompile.

   - Outputs:
     - Clean `.sof` / `.rbf` files.
     - Trojan-infected `.sof` / `.rbf` files.
     - A CSV or JSON manifest with labels.

2. **Static bitstream → image → CNN**

   Python scripts in `preprocess/`:

   - `extract_sof_sections.py`
     Strips headers/firmware sections from `.sof`, keeps configuration region.

   - `sof_to_bits.py`
     Converts the configuration payload to a flat `bits` array (`0/1`, `np.uint8`) and stores it in `.npz`.

   - `guess_width_autocorr.py`
     Heuristically guesses a good 2D width using autocorrelation + entropy minimization so that LAB/ALM columns align vertically.

   - `bits_to_image.py`
     Reshapes the bit vector into a 2D image with the chosen width and writes PNGs for CNNs.

   ML code in `ml/`:

   - `ml/models/cnn_bitstream.py`
     A small PyTorch CNN for binary classification (clean vs Trojan) or multi-class (Trojan type).

   - `ml/train_classifier.py`
     Trains CNNs on the generated bitstream images.

   - `ml/eval_on_bitstreams.py`
     Runs trained models on unknown `.sof` / `.rbf` (e.g. K10 board firmware).

3. **Side-channel branch (for encrypted Stratix 10)**

   For boards where bitstreams are fully encrypted (e.g. Superscalar K10):

   - `sidechannel/capture_stub.py`
     Placeholder API for attaching to an oscilloscope / ADC and recording power traces during configuration or runtime.

   - `sidechannel/preprocess_traces.py`
     Cleans, normalizes, and segments traces.

   - `sidechannel/train_trace_classifier.py`
     Trains ML models (e.g. 1D CNN) to classify "known good" vs "anomalous/Trojan" traces.

---

## Hardware targets

### 1. Arria 10 – BittWare A10PED

Used as a **lab board** to generate large, labeled datasets:

- We control the design and Quartus flow.
- We can compile both clean and Trojan-infected designs.
- We can emit `.sof` (SRAM Object File) and uncompressed PR bitstreams.
- Ideal for validating the full static pipeline.

### 2. Stratix 10 – Superscalar K10 / ColEngine P2 miner

Used as a **black-box test case**:

- Commercial miner with Stratix 10 FPGA(s) and Secure Device Manager (SDM).
- Bitstreams are likely:
  - Signed & encrypted.
  - Loaded from an SD card / internal flash.

Two modes:

1. **If we can extract meaningful `.sof` / `.rbf` files**
   → run through the static pipeline.

2. **If everything is encrypted noise**
   → use only the side-channel branch.

---

## Quickstart (for developers)

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install numpy scipy scikit-image matplotlib torch torchvision tqdm click
```

Optional: `scikit-learn`, `pandas`, `ipywidgets` for notebooks.

### Example static pipeline run

Assuming you have at least one `.sof` in `dataset/raw_sof/`:

```bash
# 1. Extract configuration bits
python preprocess/sof_to_bits.py \
    --input dataset/raw_sof/example.sof \
    --output dataset/parsed_npz/example_bits.npz

# 2. Guess a good 2D width and create an image
python preprocess/guess_width_autocorr.py \
    --input dataset/parsed_npz/example_bits.npz \
    --output dataset/images/example.png

# 3. Train a simple classifier on a dataset
python ml/train_classifier.py \
    --images-root dataset/images \
    --labels-file dataset/labels.csv \
    --output models/cnn_bitstream_example.pt
```

(Initially, images + labels can be synthetic; later they come from the Tcl-generated data.)

---

## Quartus side (Arria 10 / Stratix 10)

*Requires Intel Quartus Prime Pro and access to the appropriate device family.*

From a Quartus shell:

```bash
quartus_sh -t quartus_tcl/generate_clean_designs.tcl
quartus_sh -t quartus_tcl/inject_trojan_eco.tcl
quartus_sh -t quartus_tcl/batch_compile.tcl
```

These Tcl scripts are written to be parameterized via environment variables or command-line arguments (design name, device, number of variants, etc.).

---

## Docs

* `docs/paper_draft.md`
  – narrative version of the system (copied/cleaned from the PDF + notes).

* `docs/hardware_notes_k10_a10.md`
  – pinouts, JTAG/SD card notes, safety / power information, photos.

---

## Status

This repository is a **scaffold**. Many scripts are stubs with clear TODOs and docstrings.
The immediate goal is to get:

1. `sof_to_bits.py` and `guess_width_autocorr.py` working on sample `.sof` files.
2. A tiny CNN running end-to-end on synthetic bitstream images.
3. The Quartus Tcl scripts generating a few dozen labeled examples.

Once those three milestones are reached, we can scale up the dataset and refine the ML models.
