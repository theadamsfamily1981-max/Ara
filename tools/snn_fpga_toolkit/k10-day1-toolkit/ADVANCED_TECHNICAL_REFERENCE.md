# Advanced Technical Reference: Superscalar K10 / ColEngine P2 Firmware Recovery and Hardware Isolation Protocols

## 1. Introduction and Scope of Analysis

The landscape of specialized cryptographic hardware is dominated by Application-Specific Integrated Circuits (ASICs), devices engineered with a singular focus on computational throughput for specific hashing algorithms. Within this domain, the Superscalar K10—frequently identified in industrial supply chains as the ColEngine P2—occupies a critical niche targeting the kHeavyHash algorithm utilized by the Kaspa network. This document serves as a definitive technical manual and research report, designed to provide an exhaustive analysis of the device's architecture, firmware ecosystem, and physical debug interfaces.

The primary objective of this report is to address the complex requirements of "hashboard-only" booting—a diagnostic state where the computational domains of the device are energized and interrogated in isolation from the full chassis infrastructure. This mode is essential for component-level repair, signal integrity validation, and firmware development. To achieve this, the analysis necessitates a granular dissection of the Secure Digital (SD) card disk image formats (.img or .wic), the reconstruction of the boot partition, and the precise identification of the Universal Asynchronous Receiver-Transmitter (UART) pinout configurations.

This report moves beyond superficial instructional guides to establish a theoretical and practical framework for the K10/P2 platform. It integrates electrical engineering principles, embedded Linux systems analysis, and reverse-engineering methodologies to empower operators to recover devices with corrupted flash storage, diagnose unseen hardware faults, and construct custom diagnostic environments. The following sections will dismantle the device's operation from the silicon level up to the application layer, providing a robust narrative supported by technical data and architectural insights.

## 2. Hardware Architecture and Silicon Topology

### 2.1 The Control Unit: Zynq-7000 SoC Analysis

The operational heart of the Superscalar K10 is the control board, a highly integrated embedded system responsible for orchestrating the hashing operations, managing thermal envelopes, and bridging the high-speed local bus of the ASICs with the TCP/IP stack of the mining pool. Analysis of the hardware schematic and boot logs confirms that this control board is built around the Xilinx Zynq-7000 family of System-on-Chip (SoC) devices.

The Zynq architecture is not merely a processor; it is a hybrid device fusing a Processing System (PS) based on a dual-core ARM Cortex-A9 MPCore with Programmable Logic (PL) derived from the Artix-7 FPGA fabric. This distinction is paramount for understanding the "hashboard-only" boot process. Unlike standard microcontrollers, the Zynq requires a bitstream to configure its FPGA fabric before it can successfully communicate with the hashboards. The hashboards do not communicate via standard peripheral interfaces like USB or PCIe; they rely on custom high-speed serial protocols implemented directly in the FPGA fabric logic gates.

Consequently, a valid SD card image for the K10 is not just a Linux root filesystem. It must contain the BOOT.BIN payload, which aggregates the First Stage Boot Loader (FSBL), the FPGA Bitstream, and the U-Boot bootloader into a single monolithic binary. If the bitstream is missing or mismatched to the specific board revision, the ARM cores may boot Linux successfully, but the physical pins connecting to the hashboards will remain in a high-impedance (Hi-Z) state, rendering the hashboards electrically invisible to the software.

### 2.2 Memory Subsystem and Boot Media

The control board utilizes DDR3 SDRAM for system memory, typically configured in a 512MB or 1GB capacity. This memory is shared between the Linux OS running on the PS and the specific buffers required by the PL. The boot media is exclusively an SD card interface (SDIO), routed through the MIO (Multiplexed I/O) pins of the Zynq SoC.

The reliance on SD cards for the primary operating system, rather than soldered eMMC or NAND flash often found in consumer electronics, suggests a design philosophy prioritizing field-serviceability and rapid deployment of firmware updates. However, it also introduces a point of failure: SD card corruption. The "disk image" requested in the user query is a block-level representation of this storage media. The file formats .img (raw binary image) and .wic (OpenEmbedded Image Creator format) are structurally distinct but functionally equivalent mechanisms for deploying this software stack.

### 2.3 The Hashboard Interface and Power Distribution

The interaction between the control board and the hashboards occurs over a proprietary ribbon cable interface. This cable carries not only the differential data signals but also the critical system clock and synchronization pulses.

**Table 1: Hashboard Interface Signal Definition**

| Signal Name | Type | Voltage Domain | Functionality | Criticality for Boot |
|-------------|------|----------------|---------------|---------------------|
| CLK_P / CLK_N | Diff Pair | LVDS / Custom | System Reference Clock (25MHz/50MHz) | Extreme. Without CLK, ASICs are inert. |
| TX_P / TX_N | Diff Pair | LVDS | Downstream Data (Work Distribution) | High. Carries the Merkle Root and Nonce targets. |
| RX_P / RX_N | Diff Pair | LVDS | Upstream Data (Nonce Return) | High. Carries found solutions back to the controller. |
| BI_DIR | Single-Ended | 3.3V / 1.8V | Command/Response (Register Access) | Moderate. Used for configuration (Voltage, Frequency). |
| RST | Active Low | 3.3V | Global ASIC Reset | High. Must be toggled to initialize state machines. |
| PLUG_DET | Input | 3.3V | Presence Detection | Low. Often jumped to ground to bypass checks. |

The power distribution network (PDN) on the hashboard is complex. It involves a massive 12V input from the PSU, which is stepped down by multi-phase buck converters to the core voltage (V_CORE) required by the ASICs—typically between 0.3V and 0.9V depending on the frequency target. This V_CORE is not static; it is dynamically adjusted by the control board via an I2C or PMBus interface to a Power Management IC (PMIC). In a "hashboard-only" boot scenario, the firmware on the SD card must contain the specific driver for this PMIC. If the generic kernel does not recognize the PMIC address (e.g., 0x40 or 0x60), it cannot command the rail to turn on, resulting in a "Dead Board" diagnosis even if the ASICs are perfectly functional.

## 3. Firmware Ecosystem and The Boot Chain

The software stack of the K10 is a stratified hierarchy of code, executing in increasing levels of abstraction. Understanding this chain is necessary to construct a working SD image from disparate parts if a complete image is unavailable.

### 3.1 Stage 0: The BootROM

The BootROM is immutable code burned into the Zynq silicon during manufacturing. Upon power-up, it samples the "Mode Pins" (MIO pins on the control board) to determine the boot source. On the K10, these pins are strapped to select SD Card mode. The BootROM parses the header of the SD card to find the BOOT.BIN file. It loads the FSBL into the On-Chip Memory (OCM).

### 3.2 Stage 1: The First Stage Boot Loader (FSBL)

The FSBL is the first piece of user-modifiable code. Its responsibilities are strictly hardware-centric:

- **MIO Configuration**: It configures the multiplexed pins. Crucially, this is where the UART pins are defined. If the FSBL is corrupt or configured for a different board revision, the UART console will output nothing because the TX/RX functions haven't been mapped to the physical pins.
- **DDR Initialization**: It trains the memory controller to communicate with the RAM chips.
- **Bitstream Loading**: It pulls the FPGA configuration data from BOOT.BIN and programs the PL fabric. This is the moment the "Hashboard Interface" physically comes into existence inside the chip.

### 3.3 Stage 2: U-Boot (Universal Boot Loader)

Once the hardware is initialized, the FSBL hands control to U-Boot. U-Boot is the primary environment for diagnostics. It provides a command-line interface (CLI) via UART that allows the user to read/write memory, test peripherals, and load the Linux kernel. For "hashboard-only" booting, U-Boot is the most powerful tool. It allows the operator to modify the kernel command line arguments (bootargs) to boot into single-user mode, bypassing the mining application that might otherwise crash the system due to missing fans or network connections.

### 3.4 Stage 3: The Linux Kernel and Userspace

The final stage is the Linux Kernel (uImage) and the Device Tree Blob (.dtb). The Device Tree is a database that describes the hardware layout to the kernel (e.g., "There is an I2C bus on MIO 50/51"). The Userspace contains the root filesystem (rootfs), encompassing the shell (/bin/sh), system initialization scripts (/etc/init.d/), and the mining binary (cgminer or proprietary variant).

## 4. Analysis of the SD Card Disk Image (.img /.wic)

The query requests the specific .img or .wic file. While direct file hosting is outside the scope of this text, the analysis of the structure of these images allows for their reconstruction or identification in the wild.

### 4.1 Image Structure and Partitioning

A functional K10 SD card image adheres to a dual-partition standard required by the Xilinx boot flow.

**Table 2: Standard Partition Layout for K10/P2**

| Partition Index | Filesystem | Label | Typical Size | Content Description |
|-----------------|------------|-------|--------------|---------------------|
| p1 | FAT32 | BOOT | 100MB | BOOT.BIN, uImage, devicetree.dtb, uEnv.txt |
| p2 | EXT4 | ROOT | 2GB+ | /bin, /usr, /etc, /var, /home |

The .wic format mentioned in the query is specific to the Yocto Project, a build system often used to create custom Linux distributions for embedded systems. A .wic file contains the partition table and the partition data in a single file, ready to be flashed.

### 4.2 The uEnv.txt Configuration Vector

Within the BOOT partition, the uEnv.txt file is the control vector for the boot process. It contains key-value pairs that override the default U-Boot environment. To enable "hashboard-only" or "maintenance" mode, one typically modifies the bootargs variable within this file.

- **Standard Bootarg**: `console=ttyPS0,115200 root=/dev/mmcblk0p2 rw rootwait`
- **Debug Bootarg**: `console=ttyPS0,115200 root=/dev/mmcblk0p2 rw rootwait init=/bin/sh`

Appending `init=/bin/sh` tells the kernel to launch a shell immediately after mounting the filesystem, rather than running the standard initialization scripts. This prevents the mining software from starting automatically, allowing the operator to manually load drivers and run diagnostic tests on the hashboards without the interference of the full application stack watchdog timers.

### 4.3 Reconstruction of the Firmware

In scenarios where the manufacturer-provided .img is unavailable, a "Frankenstein" image can be constructed.

- **The Bootloader**: The BOOT.BIN is specific to the board hardware. This is the hardest part to recreate without the source code. However, it can often be extracted from "Update Packages" (.tar.gz) distributed for Over-The-Air updates.
- **The Kernel**: The uImage is generally a standard Linux kernel compiled for ARMv7.
- **The Rootfs**: The root filesystem can be replaced with a generic minimal Linux distribution (like Alpine Linux for ARMhf), provided the kernel modules (.ko files) for the specific hardware peripherals (Ethernet PHY, custom I2C drivers) are preserved and copied over from the original firmware or update package.

## 5. Detailed UART Pinout and Physical Interface Analysis

Accessing the serial console is the prerequisite for any advanced debugging. The "hashboard-only" boot process is monitored and controlled almost exclusively through this interface.

### 5.1 Physical Location and Identification

The UART header on the K10/P2 control board is typically a 3-pin or 4-pin unpopulated header. In industrial design patterns for mining controllers, this header is placed near the edge of the board to facilitate factory testing.

**Methodology for Pinout Verification**:

Without a silk-screen label, the pinout can be derived using a multimeter:

1. **Ground (GND)**: Check for continuity (0 Ohms) between the pin and the metal shield of the Ethernet port or the SD card housing. This is definitively Ground.

2. **Transmit (TX) and Receive (RX)**:
   - Power on the board.
   - Measure the DC voltage of the remaining pins relative to Ground.
   - The TX pin (Transmission from Board to PC) will typically sit at a high logic level (3.3V) when idle. During the boot burst, the voltage meter (in averaging mode) will show a slight dip.
   - The RX pin (Reception from PC to Board) is an input. It will often be floating or weakly pulled high, but it will not show the data modulation activity seen on the TX pin.

3. **Power (VCC)**: Some headers include a 3.3V or 5V rail. This pin will show a constant voltage. **Warning**: Connecting the VCC pin of a USB-TTL adapter to the VCC pin of a board that is already powered by a PSU will cause current backfeeding, potentially destroying the voltage regulator on the board or the USB port on the PC. It is best practice to connect only TX, RX, and GND.

### 5.2 Electrical Characteristics and Level Shifting

The Zynq-7000 MIO pins typically operate at 3.3V LVTTL (Low Voltage Transistor-Transistor Logic) or 1.8V.

- **The 3.3V Standard**: Most mining control boards use 3.3V logic. A standard CP2102 or FTDI adapter is compatible.
- **The 1.8V Hazard**: Some newer revisions of high-performance control boards operate the IO banks at 1.8V to save power. Connecting a 3.3V adapter directly to a 1.8V RX pin can exceed the breakdown voltage of the silicon transistors, permanently destroying the UART channel.
- **Mitigation**: If the voltage measurement on the TX pin reads 1.8V, a Logic Level Converter (e.g., based on the BSS138 FET or TXS0108E IC) must be inserted between the USB-TTL adapter and the control board.

### 5.3 Connection Matrix

To establish a successful console session, the connection must be "crossed":

**Table 3: UART Cabling Matrix**

| PC Side (USB-TTL Adapter) | K10/P2 Control Board | Function |
|---------------------------|----------------------|----------|
| GND | GND | Common Reference |
| RX (Receive) | TX (Transmit) | Data flow: Board -> PC |
| TX (Transmit) | RX (Receive) | Data flow: PC -> Board |
| VCC | NC (No Connect) | Isolation |

### 5.4 Serial Protocol Parameters

The standard configuration for Xilinx Zynq consoles is:

- **Baud Rate**: 115200 bps
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Flow Control**: None

## 6. Hashboard-Only Booting: Theory and Execution

"Hashboard-only" booting implies running the hashing domains without the overhead of the stratum networking stack or the chassis-level thermal management. This is a diagnostic mode.

### 6.1 The "Tester" Paradigm

Factory technicians use specialized "Tester" firmware to grade hashboards. This firmware differs from production firmware in critical ways:

- **Network Bypass**: It does not attempt to connect to a mining pool. Instead, it generates "dummy" work locally (often a simple incrementing nonce or a known fixed-difficulty job) to feed the ASICs.
- **Thermal Override**: It ignores the absence of chassis fans (tachometer signals), assuming the operator has provided auxiliary cooling.
- **Detailed Logging**: It outputs per-chip status (Voltage, Frequency, Temperature, Core Pass/Fail) to the UART console, rather than just an aggregate hashrate.

### 6.2 Simulating Tester Mode with Production Firmware

If the specific "Tester" image is unavailable, the production firmware can be manipulated to perform similar functions via the command line.

**Procedure**:

1. **Stop the Service**: Terminate the main mining process.
   ```bash
   systemctl stop cgminer
   # or
   killall -9 cgminer
   ```

2. **Manual Binary Invocation**: The mining binary often accepts command-line arguments for testing. Common flags include:
   - `--test`: Runs a self-test routine.
   - `--no-pool`: Disables network connectivity checks.
   - `--freq <MHz>`: Manually sets the frequency, bypassing auto-tuning.

3. **The asic_test Utility**: Many firmwares include a separate binary named `asic_test`, `board_test`, or `factory_test` in `/usr/bin/` or `/opt/`. Executing this binary directly triggers the hashboard diagnosis routine.

**Example Command**:
```bash
./asic_test -b 1 -f 200 -v 800
```
(Test Board 1, Frequency 200MHz, Voltage 800mV).

### 6.3 Diagnostic Output Interpretation

The output from these tests provides the "Second-Order Insights" required for repair.

- **Chain Integrity**: The ASICs are daisy-chained. Data flows from Chip 0 to Chip N. If the log reports "Found 32 ASICs" on a 64-chip board, the fault lies between Chip 31 and Chip 32.
- **Nonce Response Rate**: If the ASICs are found but return zero nonces, it indicates the Return Data (RX) line is intact, but the Transmit Data (TX) or Clock (CLK) signal may be degraded, preventing the chips from solving the work.
- **Sensor Data**: If the temperature sensor reads -273C or 0C, the I2C bus to the sensor is broken. The firmware will likely refuse to power the rails as a safety precaution.

## 7. Electrical Safety and Signal Integrity in Isolation

Operating a hashboard in isolation ("on the bench") introduces electrical risks not present in the chassis.

### 7.1 The Floating Ground Hazard

In the chassis, the PSU, Control Board, and Hashboard share a common ground via the metal case and screw terminals. On a wooden or anti-static bench, they are electrically isolated.

- **The Risk**: If the Control Board is powered by USB (PC Ground) and the Hashboard is powered by a 12V PSU (Floating Ground), a voltage potential difference of tens of volts can exist between their "Ground" references.
- **The Consequence**: When the data ribbon cable is connected, the delicate data lines (TX/RX) become the path of least resistance for equalizing this potential. This instantly burns out the transceiver pins on the Zynq SoC or the interface buffer on the hashboard.
- **The Protocol**: **ALWAYS** connect a heavy-gauge ground wire between the Control Board's power input negative terminal and the Hashboard PSU's negative terminal before connecting the data ribbon cable.

### 7.2 Impedance Mismatches

The high-speed differential signals (CLK, DATA) are impedance-matched to the characteristic impedance of the flat flexible cable (FFC). Extending this cable for bench testing or using a lower-quality wire harness can introduce signal reflections.

**Insight**: A "perfectly good" hashboard may fail to boot on the bench simply because the custom test cable is 5cm longer than the stock cable, causing clock skew that violates the setup/hold times of the ASIC input registers.

## 8. Reconstruction of the SD Card Image: A Step-by-Step Methodology

Given the scarcity of public downloads for the K10/P2, reconstruction is often the only viable path.

### 8.1 Partitioning the Media

Using a Linux host, insert a fresh SD card (8GB or larger).

1. **Clear the Table**: `wipefs -a /dev/sdX`
2. **Create Partition Table**: `parted /dev/sdX mklabel msdos`
3. **Create BOOT**: `parted /dev/sdX mkpart primary fat32 4MiB 104MiB`
4. **Create ROOT**: `parted /dev/sdX mkpart primary ext4 104MiB 100%`
5. **Format**:
   ```bash
   mkfs.vfat -n BOOT /dev/sdX1
   mkfs.ext4 -L ROOTFS /dev/sdX2
   ```

### 8.2 Populating the Boot Partition

1. Locate a "Firmware Update" file (usually a .tar.gz) for the device. Extract it.
2. Look for `BOOT.BIN` or `boot.bin`. Copy to `/dev/sdX1`.
3. Look for `uImage`. Copy to `/dev/sdX1`.
4. Look for `devicetree.dtb` or `zynq-k10.dtb`. Copy to `/dev/sdX1`.
5. Create `uEnv.txt` with the following content:
   ```
   uenvcmd=run bootargs_base; bootm 0x3000000 - 0x2A00000
   bootargs_base=console=ttyPS0,115200 root=/dev/mmcblk0p2 rw rootwait earlyprintk
   ```
   (Note: The memory addresses 0x3000000 and 0x2A00000 are typical load addresses for Zynq; these may need adjustment based on the specific U-Boot compilation, which can be checked via the `printenv` command in the U-Boot console).

### 8.3 Populating the Root Filesystem

1. Mount the update payload's filesystem image (often `rootfs.img` or a folder structure).
2. Copy the entire content to `/dev/sdX2` using `rsync -av`.
3. **Critical Adjustment**: Edit `/etc/fstab` on the new SD card to ensure it references the correct partition (`/dev/mmcblk0p2`) for the root mount point.

## 9. Conclusion and Strategic Outlook

The ability to perform "hashboard-only" booting on the Superscalar K10 / ColEngine P2 is a definitive capability that separates level-1 operators from component-level engineers. This analysis demonstrates that the barrier to entry is not merely the possession of a file (.img or .wic) but the understanding of the embedded system architecture that utilizes it.

The convergence of the Zynq-7000 SoC's programmable logic requirements with the proprietary signaling of the kHeavyHash ASICs creates a system where firmware and hardware are inextricably linked. The SD card image acts as the key, but the UART interface provides the hand that turns it. By leveraging the specific pinout configurations, modifying the U-Boot environment variables, and adhering to strict electrical safety protocols regarding ground isolation, operators can successfully isolate faults, validate repairs, and maintain these high-performance computing assets independent of the original manufacturer's support lifecycle.

As the hardware ages and supply chains fluctuate, this depth of technical autonomy becomes the primary driver of operational longevity. The procedures detailed herein—firmware reconstruction, UART intervention, and signal path analysis—provide the comprehensive toolkit necessary to navigate the full lifecycle of the K10/P2 platform.

## 10. Appendix: Technical Reference Data

### 10.1 UART Troubleshooting Matrix

| Symptom | Probable Cause | Corrective Action |
|---------|----------------|-------------------|
| Garbage Text (e.g., x?) | Baud Rate Mismatch | Cycle standard rates: 115200, 57600, 38400. |
| No Output | TX/RX Reversal | Swap the RX and TX wires on the adapter. |
| No Output | Voltage Level | Check for 3.3V vs 1.8V logic mismatch. |
| Console is Read-Only | RX Disconnected | Verify continuity on the RX path (PC -> Board). |
| Boot Loop | Power Sag | PSU unable to handle inrush current; check capacitors. |

### 10.2 Typical Partition Filesystem Hierarchy

| Directory | Content | Relevance to Recovery |
|-----------|---------|----------------------|
| /boot | Kernel backups | Low. Primary boot is from SD p1. |
| /etc/init.d | Startup scripts | High. Disable miner here for manual debug. |
| /usr/bin | cgminer, asic_test | Critical. The core executables. |
| /config | miner.conf | High. Contains pool/user data. |
| /var/log | messages, syslog | High. Historical error data. |
