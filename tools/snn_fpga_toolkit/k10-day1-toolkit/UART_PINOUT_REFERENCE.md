# K10/P2 UART Pinout Identification Reference

**Quick visual guide for locating UART console pins on Stratix 10 SoC hashboards.**

---

## Standard Intel Stratix 10 UART Headers

### Typical Locations

```
┌─────────────────────────────────────────┐
│  Power Connector         SD Card Slot   │
│      │                       │           │
│      ▼                       ▼           │
│   ┌───┐                  ┌─────┐        │
│   │ ▓ │                  │     │        │
│   └───┘                  └─────┘        │
│                                          │
│    ┌──────────────┐                     │
│    │  Stratix 10  │    ┌───┐ ← UART    │
│    │  SoC (BGA)   │    │ ▓ │   Header  │
│    │              │    └───┘            │
│    └──────────────┘                     │
│                                          │
│  Ethernet PHY                            │
│   (may be absent)                        │
└─────────────────────────────────────────┘
```

**Common Names**:
- J1, J2, J3 (generic)
- CON_UART, UART0, UART_DEBUG
- HPS_UART (Hard Processor System UART)
- CONSOLE

---

## Pin Configuration

### 4-Pin Header (Most Common)

```
 ┌─┬─┬─┬─┐
 │1│2│3│4│  Pin Numbers
 └─┴─┴─┴─┘
  │ │ │ │
  │ │ │ └── VCC (3.3V or 1.8V) - DO NOT CONNECT
  │ │ └──── TX  (HPS transmits, you receive)
  │ └────── RX  (HPS receives, you transmit)
  └──────── GND (Ground reference)
```

### 3-Pin Header (Minimal)

```
 ┌─┬─┬─┐
 │1│2│3│  Pin Numbers
 └─┴─┴─┘
  │ │ │
  │ │ └── TX
  │ └──── RX
  └────── GND
```

---

## Connection Diagram

### USB-to-TTL Adapter → Hashboard

```
USB-TTL Adapter               K10 Hashboard UART
┌──────────────┐             ┌──────────────┐
│              │             │              │
│  GND (Black) ├─────────────┤ Pin 1 (GND)  │
│              │             │              │
│  TX  (Green) ├─────────────┤ Pin 2 (RX)   │ ← Crossed!
│              │             │              │
│  RX  (White) ├─────────────┤ Pin 3 (TX)   │ ← Crossed!
│              │             │              │
│  VCC (Red)   │   X (NC)    │ Pin 4 (VCC)  │ ← Not connected!
│              │             │              │
└──────────────┘             └──────────────┘

NC = Not Connected
X  = Do not connect (prevent damage)
```

**Critical**: TX and RX are **crossed** (adapter TX → board RX, adapter RX → board TX).

---

## Voltage Identification

### Measuring with Multimeter

**Setup**:
1. Board powered OFF
2. Multimeter in continuity mode (beep)
3. Black probe on known GND (power connector negative pin)
4. Red probe on each UART pin

**Expected Results**:

| Pin Function | Continuity to GND | Voltage (Powered ON) |
|--------------|-------------------|----------------------|
| GND          | ✅ Beep (0Ω)     | 0.0V                 |
| TX           | ❌ No beep       | 1.8V - 3.3V (idle)   |
| RX           | ❌ No beep       | 1.8V - 3.3V (idle)   |
| VCC          | ❌ No beep       | 1.8V or 3.3V         |

---

## Visual PCB Markers

### Silkscreen Indicators

**Look for these labels near headers**:

```
Text on PCB:        Meaning:
─────────────────────────────
J1 / J2 / J3        Generic connector number
UART / UART0        UART interface
CONSOLE / DEBUG     Debug console
HPS_UART            Hard Processor System UART
TX / RX / GND       Explicit pin labels
1  2  3  4          Pin numbers
▼                   Pin 1 indicator (triangle)
●                   Pin 1 indicator (dot)
```

**Pin 1 Indicator**: Usually marked with:
- Square pad (other pins are round)
- White triangle (▼) or dot (●)
- "1" printed on silkscreen

---

## Component-Based Identification

### Method: Follow PCB Traces

**Strategy**: UART pins connect to level shifter chip near UART header.

**Common Level Shifters**:
- SN74LVC2T45 (Texas Instruments)
- NVT2008 (ON Semi)
- SN74AVC1T45 (TI)

**Procedure**:
1. Locate Stratix 10 SoC (large BGA chip)
2. Find level shifter chip nearby (usually TSSOP-8 or VSSOP-8 package)
3. Trace PCB from level shifter to header pins
4. Level shifter pins:
   - Pin 1, 2: I/O direction control
   - Pin 3, 4: TX/RX data lines
   - Pin 5, 6: Power (VCC/GND)

---

## Testing UART Connection

### Loopback Test (Before Connecting to Board)

**Test USB-TTL Adapter**:

1. **Short TX to RX** on adapter (jumper wire)
2. Open serial terminal (minicom, PuTTY)
3. Type characters
4. **Expected**: Characters echo back
5. **If not**: Adapter is faulty or wrong driver

**Linux**:
```bash
# Connect TX to RX with jumper
sudo minicom -D /dev/ttyUSB0 -b 115200
# Type: test
# Should see: test (echoed back)
```

---

### Live Test (Connected to Board)

**Procedure**:

1. Connect UART (GND, TX, RX)
2. Power on board
3. Watch serial terminal

**Expected Immediate Output** (within 5 seconds):
```
U-Boot SPL 2021.10 (Jan 01 2022 - 12:00:00)
SDRAM: Initializing DDR4
```

**If Nothing Appears**:
- Try swapping TX/RX
- Try different baud rates (9600, 57600, 115200)
- Check GND connection
- Verify power is on (measure 12V at connector)

---

## Common Mistakes

### ❌ Wrong Voltage Level

**Problem**: 5V adapter connected to 3.3V board.

**Symptom**: Board unresponsive, TX/RX pins measure 5V.

**Damage**: Permanent damage to HPS UART pins.

**Prevention**: Always use 3.3V logic level adapters.

---

### ❌ TX/RX Not Crossed

**Problem**: TX→TX, RX→RX (parallel connection).

**Symptom**: Can send but can't receive, or vice versa.

**Fix**: Swap TX and RX connections.

**Visual Aid**:
```
WRONG:
Adapter TX ────→ Board TX  ❌
Adapter RX ────→ Board RX  ❌

CORRECT:
Adapter TX ────→ Board RX  ✅ (crossed)
Adapter RX ────→ Board TX  ✅ (crossed)
```

---

### ❌ GND Not Connected

**Problem**: TX/RX connected, but GND floating.

**Symptom**: Garbled characters, random noise.

**Fix**: Always connect GND first.

**Why**: UART is single-ended signaling - requires common ground reference.

---

### ❌ Wrong Baud Rate

**Problem**: Terminal set to 9600, board uses 115200.

**Symptom**: Garbled characters like: `����������������`

**Fix**: Try standard baud rates: 9600, 19200, 57600, **115200** (most common), 230400.

**Test Sequence**:
```bash
# Try each baud rate (with board powered and booting):
sudo minicom -D /dev/ttyUSB0 -b 9600
sudo minicom -D /dev/ttyUSB0 -b 57600
sudo minicom -D /dev/ttyUSB0 -b 115200  ← Most likely
```

---

## Alternative: Logic Analyzer Method

**For Difficult Identifications**:

**Hardware**: 8-channel USB logic analyzer ($10-20)

**Procedure**:

1. Connect logic analyzer probes to suspected UART pins
2. Power on board
3. Capture signals during boot
4. Analyze in PulseView or Saleae Logic

**What to Look For**:

**TX Pin** (HPS transmits):
- Active during boot (first 5-30 seconds)
- Idle high (3.3V), pulses low
- Baud rate: typically 115200 (8.68 µs bit period)

**RX Pin** (HPS receives):
- Usually idle high
- May toggle if board expects commands

**Capture Example**:
```
TX Pin:  ‾‾‾\___|‾\___|‾‾‾‾\___|‾\___|‾‾‾‾\___|‾‾‾  (data bits)
         Idle  Start   D0-D7     Parity  Stop  Idle
         High  Low     Data      Bit     High  High

Timing: Each bit = 8.68 µs (115200 baud)
```

---

## Pinout Database (Common Boards)

### Intel Stratix 10 SoC Development Kit (Official)

**Board**: DK-SOC-1SSX-H-D

**UART Header**: J37 (4-pin, 2.54mm pitch)

```
Pin 1: GND
Pin 2: RX  (HPS receives)
Pin 3: TX  (HPS transmits)
Pin 4: VCC (3.3V)
```

**Location**: Near USB ports, top-left area.

---

### Terasic DE10-Pro (Similar Architecture)

**Board**: Stratix 10 GX Development Kit

**UART Header**: JP1 (4-pin)

```
Pin 1: GND
Pin 2: TX  (Board transmits)
Pin 3: RX  (Board receives)
Pin 4: NC  (Not connected)
```

---

### K10 Hashboard (Estimated)

**Based on**: Intel reference design patterns

**Likely Header**: J1 or J2 (near Stratix 10 SoC)

**Pin Configuration** (To be verified):
```
Pin 1: GND
Pin 2: RX  (HPS receives - connect to adapter TX)
Pin 3: TX  (HPS transmits - connect to adapter RX)
Pin 4: VCC (3.3V - do not connect)
```

**Note**: This is an educated guess. Verify with multimeter first!

---

## Emergency: No UART Header

**If board has NO accessible UART header**:

### Option 1: Solder Wires Directly

1. Locate TX/RX test points on PCB
2. Solder thin wires (30 AWG) to pads
3. Connect to USB-TTL adapter
4. Use hot glue for strain relief

**Skills Required**: Soldering experience with small components.

---

### Option 2: JTAG Interface

**If UART is truly inaccessible**, use JTAG:

**Hardware**: Intel USB Blaster II ($100-300)

**Procedure**:
1. Locate JTAG header (10-pin or 20-pin)
2. Connect USB Blaster
3. Use Intel Quartus Programmer
4. Program FPGA and load software via JTAG

**Downside**: Slower, more expensive, requires Quartus software.

---

### Option 3: Network Interface (Unlikely)

**If board has Ethernet PHY**:

1. Check for RJ-45 jack (rare on hashboard alone)
2. If present, boot and scan network for DHCP lease
3. SSH to board: `ssh root@<discovered_ip>`

**Probability**: Low (Ethernet usually on control board).

---

## Quick Diagnostic Checklist

Before assuming UART is broken, verify:

- [ ] USB-TTL adapter tested with loopback (TX→RX shorted)
- [ ] Correct driver installed (check `dmesg | grep tty` on Linux)
- [ ] GND connected between adapter and board
- [ ] TX/RX crossed (adapter TX → board RX, adapter RX → board TX)
- [ ] Baud rate = 115200, 8N1 (no parity, 1 stop bit)
- [ ] Flow control disabled (no RTS/CTS)
- [ ] Board powered (measure 12V at connector)
- [ ] Tried multiple baud rates (9600, 57600, 115200)
- [ ] VCC NOT connected (prevents damage)

---

## Visual Troubleshooting

### Symptom: Garbled Characters

**Example Output**:
```
����������������������������
���������������������������
```

**Diagnosis**: Baud rate mismatch or noise.

**Fix**:
1. Try 115200 baud first
2. Check GND connection
3. Shorten cables (<15cm ideal)
4. Use shielded cables if interference suspected

---

### Symptom: Nothing at All

**Example Output**:
```
(blank screen)
```

**Diagnosis**:
1. Wrong pin (not actually UART)
2. TX/RX swapped
3. Board not booting (power issue)
4. Board booting from different source (QSPI flash)

**Fix**:
1. Measure voltage on suspected TX pin - should toggle during boot
2. Swap TX/RX
3. Check power supply (12V, sufficient amperage)
4. Check for boot LEDs or other signs of life

---

### Symptom: Partial Text

**Example Output**:
```
U-Boot SPL 20��DRAM Init�����
```

**Diagnosis**: Intermittent connection or voltage droop.

**Fix**:
1. Re-seat connections
2. Check for cold solder joints
3. Ensure GND is solid connection
4. Use breadboard with jumpers for testing

---

## Recommended USB-TTL Adapters

### ✅ FTDI FT232RL (Best)

**Price**: $10-15
**Voltage**: 3.3V and 5V selectable (jumper)
**Driver**: Excellent Linux/Windows support
**ID**: `lsusb` shows "Future Technology Devices"

**Buy**: Ensure genuine FTDI chip (not clone).

---

### ✅ CP2102 (Good)

**Price**: $5-8
**Voltage**: 3.3V fixed
**Driver**: Silicon Labs driver required
**ID**: `lsusb` shows "Silicon Labs CP210x"

**Note**: Slightly less reliable than FTDI, but works well.

---

### ⚠️ CH340G (Budget)

**Price**: $2-5
**Voltage**: 3.3V or 5V (check datasheet)
**Driver**: Requires CH341SER driver on Windows
**ID**: `lsusb` shows "QinHeng Electronics"

**Warning**: Quality varies. Test with loopback first.

---

### ❌ PL2303 (Avoid)

**Price**: $2-3
**Issues**: Driver problems, fake chips common
**Advice**: Use FTDI or CP2102 instead.

---

## Summary

**Keys to Success**:

1. **Identify Pin 1**: Use square pad, triangle marker, or continuity to GND
2. **Cross TX/RX**: Adapter TX → Board RX, Adapter RX → Board TX
3. **Don't Connect VCC**: Use only GND, TX, RX
4. **Use 3.3V Adapter**: Never 5V on Stratix 10 pins
5. **Start at 115200 Baud**: Most common for ARM Linux systems
6. **Test Adapter First**: Loopback test (short TX to RX)

**Typical Success Timeline**:

- Pin identification: 10-30 minutes
- First connection: 5 minutes
- Troubleshooting (if needed): 10-30 minutes
- **Total**: 30-60 minutes to first boot message

---

**If you have photos of your specific board, I can provide targeted pinout identification!**
