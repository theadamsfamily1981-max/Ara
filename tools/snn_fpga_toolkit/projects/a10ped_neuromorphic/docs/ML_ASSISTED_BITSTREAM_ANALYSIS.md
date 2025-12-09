# ML-Assisted Bitstream Analysis for Legacy Intel Accelerator Cards

**Paper-Ready Section for A10PED + Azure Stratix-V Neuromorphic Tiles**

This document contains a complete paper section exploring machine-learning-assisted bitstream analysis as a research tool for understanding and repurposing decommissioned Intel-based accelerator cards.

---

## 3.x ML-Assisted Bitstream Analysis for Arria 10 and Stratix V Neuromorphic Tiles

In addition to conventional tool-supported development, we explore **machine-learning-assisted bitstream analysis** as a way to better understand and repurpose decommissioned Intel-based accelerator cards—specifically, the **BittWare A10PED** (dual Arria 10 GX) and the **Azure X930613-001** ("cat board," Stratix V GS). Both boards expose powerful FPGA devices and memory systems (DDR4 and Hybrid Memory Cube on A10PED; large Stratix V fabric and PCIe on the Azure card), but lack open documentation at the bitstream level. ML-assisted analysis is not used to circumvent vendor security features; instead, it serves as a **pattern-accelerated microscope** for studying unencrypted configuration images, guiding manual and scripted reverse-engineering (RE) towards the neuromorphic tile shells we need.

### 3.x.1 Problem Setting: Unknown Shells on Known Architectures

Our goal is not full recovery of third-party designs, but to answer concrete, architectural questions such as:

* *Where in the bitstream are the PCIe hard IP blocks and DDR/EMIF controllers configured?*
* *Can we reliably distinguish "fabric-only" regions from IO/transceiver-heavy regions?*
* *Do vendor- or cloud-specific boards (A10PED, Azure Stratix V) use unusual configuration options that might constrain our neuromorphic tile layouts?*

For supported platforms (Arria 10 dev kits, Stratix V reference designs), Quartus can still generate bitstreams, and open projects such as Mistral/nextpnr for Cyclone V provide a fully open reference pipeline for Intel ALM-based devices. We assume that for A10PED and Azure boards we have access to **unencrypted configuration images** (e.g., raw `.rbf` or `.pof` extracted via JTAG or flash programming), but not to full architecture databases as in Mistral.

Under these conditions, ML-assisted analysis is used as a **front-end to conventional RE**: it suggests where functionality is likely located and how configuration frames co-vary across design changes.

### 3.x.2 Data Generation: Paired Design–Bitstream Corpora

The first ingredient is a corpus of **paired designs and bitstreams** on devices where we *do* control the full flow:

1. **Reference platforms.**
   We use standard Arria 10 and Stratix V development boards, for which Quartus supports:

   * PCIe hard IP variations (endpoint/root, lane counts),
   * DDR3/DDR4 EMIF configurations,
   * Fabric-only test designs.

2. **Systematic design sweeps.**
   For each reference platform we generate families of small designs that:

   * Move a known "probe" module (e.g., a tiny SNN tile, a register file, a DSP chain) across different locations in the fabric.
   * Toggle configuration options (PCIe lane widths, BAR sizes, DDR timing parameters).
   * Enable/disable specific peripherals (e.g., one EMIF instance vs two).

3. **Bitstream collection and labeling.**
   For every design we retain:

   * The full configuration image (e.g., `.rbf`),
   * Higher-level metadata:

     * Which features are present (PCIe yes/no, DDR yes/no),
     * The coordinates or regions used by the probe module (when known from Quartus reports),
     * Device type and tool version.

This yields a dataset where each bitstream is annotated with **coarse functional labels** (e.g., "contains PCIe x8 endpoint," "contains DDR4 EMIF at interface A," "fabric-only test"), and a subset has **spatial hints** from the vendor tools.

### 3.x.3 ML Models: From Pattern Discovery to Region Proposals

We then apply a small set of ML techniques, each targeting a different subtask:

* **CNN-based region classification.**
  Bitstreams (or their per-frame encodings) are reshaped into 2-D arrays or 1-D sequences of feature vectors. A CNN is trained to classify windows of configuration data according to coarse labels such as `{fabric-only, PCIe-related, EMIF-related, transceiver-heavy region}`. Prior work on FPGA bitstream malware detection reports high accuracy (75–90%) for similar classification tasks, suggesting that CNNs can differentiate configuration "textures" associated with different resource types once enough labeled examples are available.

* **Unsupervised clustering of configuration frames.**
  In parallel, we treat each configuration frame or column as a feature vector (e.g., raw bits, histograms, or learned embeddings) and apply clustering (k-means, Gaussian mixtures, or autoencoders). By comparing clusters across design sweeps—e.g., "frames that change only when PCIe settings change"—we obtain candidate sets of frames associated with particular subsystems. This mirrors the clustering stage in comprehensive RE pipelines that use differential experiments to recover a large fraction of bitstream semantics.

* **Graph-based reasoning for interconnect.**
  Once we have a provisional partition of frames into "likely fabric," "likely PCIe," and "likely memory/IO," we can build a coarse **connectivity graph** between these regions (e.g., based on known probe modules and I/O columns). Graph-based models, including graph neural networks, are then candidates for learning routing or connectivity patterns, though in this work we treat them as future work rather than a production-ready component.

Importantly, all ML models are trained and validated on **supported dev boards**, never on proprietary cloud designs, so we always have a ground truth for evaluation.

### 3.x.4 Applying the Models to A10PED and Azure Stratix V

With trained models in hand, we can analyze bitstreams extracted from the **BittWare A10PED** and the **Azure Stratix V card**:

1. **Region proposals for unknown boards.**

   * The CNN classifier scans the A10PED and Azure bitstreams and outputs heatmaps over configuration regions indicating "high probability of PCIe configuration," "high probability of DDR/EMIF," and "fabric-like" areas.
   * These heatmaps do *not* give us native coordinates (we lack the full Arria 10 / Stratix V DB), but they provide **strong priors** on where PCIe and memory subsystems are likely configured.

2. **Differential experiments with known designs.**
   When we compile and load our own test designs on these boards (e.g., a minimal PCIe endpoint with a dummy fabric, or a fabric-only blinker), we can:

   * Compare their bitstreams to any existing images (if available),
   * Use the clustering model to group frames that change in plausible ways when:

     * We alter BAR sizes,
     * We enable an additional EMIF instance,
     * We add or move a neuromorphic core.

   This helps narrow down which configuration frames are tied to **host interfaces and memory**, which is exactly where our neuromorphic tile shell must integrate.

3. **Guidance for shell design and validation.**
   The ML-derived region proposals and frame clusters are not used to auto-generate new bitstreams, but they inform:

   * Which resource configurations and locations appear to be used by vendor designs,
   * Whether our own shells (built with standard Quartus flows) appear to occupy similar "zones" (sanity-checking against known-good images on reference boards),
   * Where to focus more precise RE or timing analysis if we observe unexpected behavior.

In this sense, ML acts as a **prior and triage mechanism** for human and scripted RE, rather than as an automatic bitstream decompiler.

### 3.x.5 Limitations and Ethical Scope

Our use of ML-assisted bitstream analysis is bounded by two constraints:

* **Technical visibility.**
  If A10PED or Azure cards are configured with **encrypted bitstreams**, we cannot inspect the underlying configuration bits; ML techniques are only applicable to unencrypted images or to designs we compile ourselves on supported boards. Encryption and authentication features (e.g., AES-256 bitstream encryption, secure boot) explicitly prevent such analysis, as documented in Intel's design security guides.

* **Ethical and legal considerations.**
  Our aim is to enable educational and research use of decommissioned boards, not to recover or redistribute proprietary IP. We therefore restrict ML-RE experiments to:

  * Designs we generate ourselves, or
  * Legacy images where usage clearly falls within analysis, interoperability, or preservation contexts.

Under these constraints, ML-assisted analysis provides a **practical and ethically bounded** way to accelerate the understanding of legacy Intel accelerator cards, complementing conventional EDA flows and RE tools as we build neuromorphic tiles atop the A10PED and Azure Stratix V platforms.

---

## Figure X: ML-Assisted Bitstream RE Loop for A10PED / Stratix V

**Figure Description (for diagram generation):**

The figure is a left-to-right block diagram with three main stages and two hardware "branches."

1. **Left: Reference Design Pipeline**
   - Box labeled **"Reference designs (Arria-10 / Stratix-V dev kits)"**
   - Shows small icons of FPGA boards
   - Text bullets: "probe modules," "PCIe/DDR sweeps," "fabric-only tests"
   - Arrows feed into: **"Paired data: HDL + bitstreams + metadata"**

2. **Center: ML Analysis Stage**
   - Large **"ML-assisted analysis"** box split into three sub-boxes:
     - **Top: "CNN classifier"**
       - Arrows pointing to heatmap icon
       - Label: "PCIe / EMIF / fabric region scores"
     - **Middle: "Unsupervised clustering"**
       - Icons of grouped dots
       - Caption: "frames grouped by co-variation across experiments"
     - **Bottom: "Graph-based modeling (future)"**
       - Small node–edge graph icon
       - Label: "coarse routing / interconnect structure"

3. **Right: Target Hardware Branches**
   - **Upper right: "BittWare A10PED (Arria 10 GX ×2)"**
     - Card icon
     - Bullets: "PCIe Gen3 x8," "DDR4/HMC"
     - Incoming arrows labeled: "region proposals," "frame clusters"

   - **Lower right: "Azure Stratix V card"**
     - Card icon
     - Bullets: "Stratix V GS," "dual PCIe HIP"
     - Incoming arrows labeled: "region proposals," "frame clusters"

4. **Bottom: Integration Stage**
   - Box: **"Neuromorphic tile shell design & validation"**
   - Collects arrows from both hardware boxes and ML block
   - Caption: "informs shell placement," "guides RE focus," "checks for anomalies"

5. **Feedback Loop**
   - Dashed arrow from validation box back to leftmost "Reference designs" box
   - Label: **"generate new experiments / refine models"**

**Visual Style:**
- Clean, minimal design
- Distinct colors for three central ML submodules
- Clear captions emphasizing ML provides **hints and priors**, not direct decryption or automatic RTL recovery
- Professional academic figure quality

---

## Integration with YAML-Driven Architecture

This ML-assisted approach complements the YAML-driven "Quartus-as-a-service" architecture documented in `README_ARCH.md`:

* **YAML specs** define the *target* neuromorphic tile architecture (CSR, memory, PCIe)
* **ML bitstream analysis** helps understand *existing* configurations on legacy boards
* **Together**, they enable informed design decisions when porting tiles to new platforms

The ML analysis can inform:
- Which board YAML configurations are most compatible with observed vendor layouts
- Where to place custom SNN cores relative to discovered PCIe/EMIF regions
- Validation that generated bitstreams follow expected patterns

This creates a **bidirectional workflow**:
1. Forward: YAML → Quartus → bitstream
2. Reverse: Legacy bitstream → ML analysis → YAML refinement

---

## Related Work and Citations

Key references for this section:

1. **Bitstream RE fundamentals:**
   - Symbiflow/nextpnr documentation on open bitstream formats
   - Mistral project (Cyclone V open bitstream DB)

2. **ML for FPGA security:**
   - CNN-based bitstream malware detection (75-90% accuracy reported)
   - Autoencoder-based bitstream anomaly detection

3. **Differential analysis:**
   - Academic work on clustering configuration frames by co-variation
   - Systematic design sweeps for bitstream reverse engineering

4. **Intel security documentation:**
   - Design Security User Guides (Arria 10, Stratix V)
   - AES-256 bitstream encryption specifications
   - Secure device manager (SDM) documentation

---

## Future Work

1. **Expand training corpus:**
   - More Arria 10 and Stratix V reference designs
   - Broader sweep of PCIe/DDR configurations
   - Include transceiver and HMC configurations

2. **Graph neural networks:**
   - Develop GNN models for routing/interconnect inference
   - Test on Mistral-supported devices (Cyclone V) with ground truth

3. **Azure Stratix V characterization:**
   - Obtain decommissioned Azure cards
   - Extract and analyze bitstreams
   - Map to neuromorphic tile architecture

4. **Automated shell generation:**
   - Use ML insights to suggest optimal tile placements
   - Generate candidate YAML configurations
   - Validate against timing/resource constraints

5. **Open dataset release:**
   - Publish paired HDL/bitstream corpus (for educational use)
   - Release trained models and analysis scripts
   - Enable reproducible research on legacy FPGA repurposing
