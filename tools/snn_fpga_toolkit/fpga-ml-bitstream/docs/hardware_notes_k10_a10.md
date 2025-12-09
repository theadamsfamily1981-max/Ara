# Hardware Notes – BittWare A10PED & Superscalar K10

This repo is designed around two main Intel FPGA platforms:

## 1. BittWare A10PED (Arria 10)

- Dual Arria 10 GX FPGAs.
- Up to 16–32 GB DDR4 + optional HMC.
- PCIe Gen3 x8 per FPGA.
- Lab usage:
  - Full control via Quartus Pro.
  - Used to generate clean + Trojan datasets.
  - Can emit `.sof` and uncompressed PR bitstreams.

### Key Specifications

- **Device**: 10AX115S2F45I1SG (or similar Arria 10 GX variant)
- **Logic Elements**: ~1.15M LEs
- **Memory**: 32 Mb M20K blocks
- **DSP**: 1,518 variable-precision DSP blocks
- **Transceivers**: Up to 48 transceivers (10G/25G)

### Configuration Options

- JTAG programming via USB Blaster II
- Active Serial x4 configuration from flash
- Passive Serial configuration from host
- Partial Reconfiguration support

### Usage in This Project

The A10PED is our **primary dataset generation platform**:

1. **Clean bitstream generation**: Use `quartus_tcl/generate_clean_designs.tcl` with various seeds
2. **Trojan injection**: Use `quartus_tcl/inject_trojan_eco.tcl` to insert synthetic Trojans
3. **Configuration dumps**: Extract `.sof` and `.rbf` files for analysis
4. **PR bitstream analysis**: Test partial reconfiguration bitstream forensics

---

## 2. Superscalar K10 / ColEngine P2 (Stratix 10 Miner)

- Commercial FPGA miner using Intel Stratix 10.
- Configuration via SD card (firmware / algo files).
- Bitstreams likely:
  - Signed & encrypted (SDM).
  - Compressed.

### Key Specifications

- **Device**: Stratix 10 GX/SX (exact part TBD from markings)
- **Configuration**: Secure Device Manager (SDM) with AES encryption
- **Boot source**: SD card or QSPI flash
- **Power**: High-power design (~200-400W typical)

### SDM Security Features

- AES-256 bitstream encryption
- ECDSA-384 authentication
- Anti-tamper protection
- Secure key storage in battery-backed RAM

### Usage in This Project

The K10 is our **black-box target** for testing the system on real-world encrypted designs:

1. **If we can extract meaningful `.sof` / `.rbf` files from SD card**
   - Run through static pipeline
   - Test width detection algorithms
   - Attempt entropy analysis (limited by encryption)

2. **If bitstreams are fully encrypted (expected)**
   - Use only side-channel analysis branch
   - Capture power traces during configuration
   - Train ML models on trace patterns
   - Attempt to fingerprint different firmware versions

### Safety Notes

⚠️ **IMPORTANT SAFETY WARNINGS**:

- High power device: Ensure adequate cooling and power supply
- SDM security: Do not attempt to extract keys (violates device security model)
- Card may have tamper detection: Work carefully to avoid bricking
- Check for firmware updates that may patch vulnerabilities

### Access Points

- **JTAG**: May be available via test pads or connector
- **SD Card**: Contains firmware, potentially extractable
- **UART/Serial**: May have debug console (check for unpopulated headers)
- **Power rails**: Accessible for side-channel monitoring

---

## Comparison

| Feature | BittWare A10PED | Superscalar K10 |
|---------|-----------------|-----------------|
| **Device** | Arria 10 GX | Stratix 10 GX/SX |
| **Control** | Full (we own it) | Black-box |
| **Bitstreams** | Unencrypted available | Encrypted (SDM) |
| **Use Case** | Dataset generation | Real-world test |
| **Analysis** | Static + Side-channel | Side-channel only |

---

## Additional Resources

For full theoretical background and system design, see:

- `ML-Assisted Intel FPGA Bitstream Analysis.pdf` (local copy, e.g. `/mnt/data/ML-Assisted Intel FPGA Bitstream Analysis.pdf`)
- Intel Arria 10 Configuration User Guide
- Intel Stratix 10 Configuration User Guide
- Intel Secure Device Manager (SDM) documentation
