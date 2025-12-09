#!/usr/bin/env python3
"""
PMIC I2C Flasher - Voltage/Frequency Tuning for Repurposed FPGAs
================================================================

This tool allows tuning of FPGA core voltage (VCCINT) and frequency for
AI workloads. Mining hardware is typically tuned for maximum hashrate;
AI workloads may benefit from different voltage/frequency curves.

Supported PMICs:
- Texas Instruments TPS53647 (common in Stratix 10 boards)
- Infineon IR3584 (common in Xilinx boards)
- Renesas ISL68137 (high-end mining rigs)

WARNING: Incorrect voltage settings can damage your FPGA!
- Stratix 10: VCCINT = 0.85V ±5% (0.80V to 0.89V safe range)
- Virtex UltraScale+: VCCINT = 0.85V ±5% (0.80V to 0.89V safe range)

Author: SNN Kernel Team
License: GPL-3.0
"""

import sys
import argparse
import subprocess
from typing import Optional, Dict

# ============================================================================
# I2C INTERFACE
# ============================================================================

class I2CDevice:
    """Linux I2C device interface using i2c-tools"""

    def __init__(self, bus: int, addr: int):
        self.bus = bus
        self.addr = addr

    def read_byte(self, register: int) -> Optional[int]:
        """Read single byte from I2C register"""
        try:
            result = subprocess.run(
                ["i2cget", "-y", str(self.bus), f"0x{self.addr:02x}", f"0x{register:02x}"],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip(), 16)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    def write_byte(self, register: int, value: int) -> bool:
        """Write single byte to I2C register"""
        try:
            subprocess.run(
                ["i2cset", "-y", str(self.bus), f"0x{self.addr:02x}", f"0x{register:02x}", f"0x{value:02x}"],
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def read_word(self, register: int) -> Optional[int]:
        """Read 16-bit word from I2C register (little-endian)"""
        try:
            result = subprocess.run(
                ["i2cget", "-y", str(self.bus), f"0x{self.addr:02x}", f"0x{register:02x}", "w"],
                capture_output=True,
                text=True,
                check=True
            )
            # i2cget returns 0xMMLL (little-endian), convert to integer
            return int(result.stdout.strip(), 16)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    def write_word(self, register: int, value: int) -> bool:
        """Write 16-bit word to I2C register"""
        try:
            subprocess.run(
                ["i2cset", "-y", str(self.bus), f"0x{self.addr:02x}", f"0x{register:02x}", f"0x{value:04x}", "w"],
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

# ============================================================================
# PMIC DRIVERS
# ============================================================================

class TPS53647:
    """Texas Instruments TPS53647 PMIC (common in Stratix 10 mining boards)"""

    # PMBus command codes
    CMD_VOUT_MODE = 0x20
    CMD_VOUT_COMMAND = 0x21
    CMD_VOUT_MAX = 0x24
    CMD_READ_VOUT = 0x8B
    CMD_READ_IOUT = 0x8C
    CMD_READ_TEMPERATURE_1 = 0x8D

    # VID table (linear voltage)
    VID_STEP = 0.005  # 5mV per step

    def __init__(self, i2c_bus: int, i2c_addr: int = 0x60):
        self.i2c = I2CDevice(i2c_bus, i2c_addr)
        self.i2c_addr = i2c_addr

    def read_voltage(self) -> Optional[float]:
        """Read current output voltage"""
        vout_raw = self.i2c.read_word(self.CMD_READ_VOUT)
        if vout_raw is not None:
            # Convert from linear11 format to voltage
            # Linear11: 5-bit exponent, 11-bit mantissa
            exponent = (vout_raw >> 11) & 0x1F
            mantissa = vout_raw & 0x7FF

            # Handle sign extension for exponent
            if exponent & 0x10:
                exponent |= 0xFFFFFFE0

            # Calculate voltage
            voltage = mantissa * (2 ** exponent)
            return voltage
        return None

    def set_voltage(self, target_voltage: float) -> bool:
        """Set output voltage (in volts)"""
        if not (0.5 <= target_voltage <= 1.0):
            print(f"ERROR: Voltage {target_voltage}V out of safe range (0.5V - 1.0V)")
            return False

        # Convert voltage to linear11 format
        # Use exponent = -9 (fixed for TPS53647)
        exponent = -9
        mantissa = int(target_voltage / (2 ** exponent))

        # Pack into linear11 format
        vout_command = ((exponent & 0x1F) << 11) | (mantissa & 0x7FF)

        print(f"  Setting VCCINT to {target_voltage:.3f}V (raw: 0x{vout_command:04x})")

        return self.i2c.write_word(self.CMD_VOUT_COMMAND, vout_command)

    def read_current(self) -> Optional[float]:
        """Read output current (in amperes)"""
        iout_raw = self.i2c.read_word(self.CMD_READ_IOUT)
        if iout_raw is not None:
            # Linear11 format (same as voltage)
            exponent = (iout_raw >> 11) & 0x1F
            mantissa = iout_raw & 0x7FF

            if exponent & 0x10:
                exponent |= 0xFFFFFFE0

            current = mantissa * (2 ** exponent)
            return current
        return None

    def read_temperature(self) -> Optional[float]:
        """Read PMIC temperature (in Celsius)"""
        temp_raw = self.i2c.read_word(self.CMD_READ_TEMPERATURE_1)
        if temp_raw is not None:
            # Linear11 format
            exponent = (temp_raw >> 11) & 0x1F
            mantissa = temp_raw & 0x7FF

            if exponent & 0x10:
                exponent |= 0xFFFFFFE0

            temperature = mantissa * (2 ** exponent)
            return temperature
        return None

class IR3584:
    """Infineon IR3584 PMIC (common in Xilinx UltraScale+ boards)"""

    # Similar to TPS53647 but with different VID table
    CMD_VOUT_COMMAND = 0x21
    CMD_READ_VOUT = 0x8B

    def __init__(self, i2c_bus: int, i2c_addr: int = 0x70):
        self.i2c = I2CDevice(i2c_bus, i2c_addr)

    def set_voltage(self, target_voltage: float) -> bool:
        """Set output voltage using VID code"""
        # IR3584 uses VID table (0.25V to 1.52V in 5mV steps)
        if not (0.5 <= target_voltage <= 1.0):
            print(f"ERROR: Voltage {target_voltage}V out of safe range")
            return False

        # VID code calculation for IR3584
        # VID = (Vout - 0.25V) / 0.005V
        vid_code = int((target_voltage - 0.25) / 0.005)

        print(f"  Setting VCCINT to {target_voltage:.3f}V (VID: {vid_code})")

        return self.i2c.write_byte(self.CMD_VOUT_COMMAND, vid_code)

# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_pmic(i2c_bus: int) -> Optional[object]:
    """Auto-detect PMIC type on I2C bus"""

    # Try common PMIC addresses
    pmic_configs = [
        (0x60, TPS53647),  # TI TPS53647
        (0x70, IR3584),    # Infineon IR3584
        (0x42, TPS53647),  # Alternate TI address
    ]

    print(f"\n[DETECT] Scanning I2C bus {i2c_bus} for PMICs...")

    for addr, pmic_class in pmic_configs:
        i2c_dev = I2CDevice(i2c_bus, addr)

        # Try to read a register (PMBus VOUT_MODE)
        result = i2c_dev.read_byte(0x20)

        if result is not None:
            print(f"  [FOUND] PMIC at address 0x{addr:02x} - {pmic_class.__name__}")
            return pmic_class(i2c_bus, addr)

    print(f"  [NOT FOUND] No PMICs detected on bus {i2c_bus}")
    return None

# ============================================================================
# VOLTAGE TUNING PRESETS
# ============================================================================

VOLTAGE_PRESETS = {
    "stock": {
        "vccint": 0.85,
        "description": "Factory default (mining optimized)"
    },
    "efficient": {
        "vccint": 0.80,
        "description": "Undervolt for lower power AI inference (may reduce stability)"
    },
    "performance": {
        "vccint": 0.88,
        "description": "Slight overvolt for AI training workloads (increased power)"
    },
    "safe": {
        "vccint": 0.85,
        "description": "Conservative setting for initial testing"
    }
}

# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PMIC I2C Flasher - Voltage tuning for repurposed FPGAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Auto-detect PMIC and read current voltage
  sudo ./pmic_flasher.py --bus 0 --read

  # Set voltage to 0.85V (stock)
  sudo ./pmic_flasher.py --bus 0 --voltage 0.85

  # Use preset (efficient AI inference)
  sudo ./pmic_flasher.py --bus 0 --preset efficient

Available Presets:
{chr(10).join([f'  {name}: {cfg["vccint"]}V - {cfg["description"]}' for name, cfg in VOLTAGE_PRESETS.items()])}

WARNING: Incorrect voltage settings can damage your FPGA!
         Only use this tool on hardware you own and understand.
         Start with conservative settings and monitor temperatures.
"""
    )

    parser.add_argument("--bus", type=int, required=True, help="I2C bus number (e.g., 0, 1, 2)")
    parser.add_argument("--addr", type=lambda x: int(x, 16), help="PMIC I2C address (hex, e.g., 0x60)")
    parser.add_argument("--voltage", type=float, help="Target VCCINT voltage (e.g., 0.85)")
    parser.add_argument("--preset", choices=VOLTAGE_PRESETS.keys(), help="Use voltage preset")
    parser.add_argument("--read", action="store_true", help="Read current voltage/current/temp")
    parser.add_argument("--type", choices=["tps53647", "ir3584"], help="Force PMIC type")

    args = parser.parse_args()

    # Detect or instantiate PMIC
    if args.type:
        pmic_map = {
            "tps53647": TPS53647,
            "ir3584": IR3584
        }
        addr = args.addr if args.addr else 0x60
        pmic = pmic_map[args.type](args.bus, addr)
        print(f"\n[INIT] Using {args.type} at 0x{addr:02x} on bus {args.bus}")
    else:
        pmic = detect_pmic(args.bus)

    if not pmic:
        print("\nERROR: No PMIC detected. Check:")
        print("  1. I2C bus number (use 'i2cdetect -l' to list buses)")
        print("  2. PMIC power (board must be powered on)")
        print("  3. I2C pullup resistors (hardware issue)")
        return 1

    # Read mode
    if args.read:
        print("\n[READ] Current PMIC status:")

        voltage = pmic.read_voltage()
        if voltage is not None:
            print(f"  VCCINT: {voltage:.3f}V")
        else:
            print("  VCCINT: [Read failed]")

        if hasattr(pmic, 'read_current'):
            current = pmic.read_current()
            if current is not None:
                print(f"  IOUT:   {current:.2f}A")
                if voltage:
                    print(f"  Power:  {voltage * current:.2f}W")

        if hasattr(pmic, 'read_temperature'):
            temp = pmic.read_temperature()
            if temp is not None:
                print(f"  PMIC Temp: {temp:.1f}°C")

        return 0

    # Set voltage mode
    target_voltage = None

    if args.preset:
        target_voltage = VOLTAGE_PRESETS[args.preset]["vccint"]
        print(f"\n[PRESET] Using '{args.preset}': {VOLTAGE_PRESETS[args.preset]['description']}")

    if args.voltage:
        target_voltage = args.voltage

    if not target_voltage:
        print("ERROR: Must specify --voltage or --preset or --read")
        return 1

    # Safety checks
    print(f"\n[WARNING] About to set VCCINT to {target_voltage:.3f}V")
    print(f"  Safe range: 0.80V - 0.89V (±5% from 0.85V nominal)")

    if target_voltage < 0.80 or target_voltage > 0.89:
        print(f"\n  ⚠️  DANGER: {target_voltage:.3f}V is outside safe range!")
        confirm = input("  Continue anyway? (type 'YES' in caps): ")
        if confirm != "YES":
            print("  Aborted.")
            return 1

    # Set voltage
    print(f"\n[SET] Programming PMIC...")
    success = pmic.set_voltage(target_voltage)

    if success:
        print(f"  [SUCCESS] Voltage set to {target_voltage:.3f}V")

        # Verify by reading back
        print(f"\n[VERIFY] Reading back voltage...")
        actual_voltage = pmic.read_voltage()

        if actual_voltage is not None:
            print(f"  Measured: {actual_voltage:.3f}V")
            error = abs(actual_voltage - target_voltage)

            if error < 0.01:  # Within 10mV
                print(f"  [OK] Voltage programming successful (error: {error*1000:.1f}mV)")
            else:
                print(f"  [WARNING] Voltage error: {error*1000:.1f}mV (may be acceptable)")

        return 0
    else:
        print(f"  [FAIL] Voltage programming failed")
        return 1

if __name__ == "__main__":
    import os
    if os.geteuid() != 0:
        print("ERROR: This tool requires root access for I2C operations")
        print("Try: sudo ./pmic_flasher.py --bus 0 --read")
        sys.exit(1)

    sys.exit(main())
