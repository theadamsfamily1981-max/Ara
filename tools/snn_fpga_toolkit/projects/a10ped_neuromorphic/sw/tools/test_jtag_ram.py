#!/usr/bin/env python3
"""
JTAG RAM Test Script

Tests the JTAG-to-Avalon bridge and on-chip RAM for Milestone 1.2.

This script uses Intel System Console Tcl commands via subprocess to:
  1. Connect to JTAG cable
  2. Open JTAG Master service
  3. Write test patterns to on-chip RAM
  4. Read back and verify
  5. Report success/failure

Prerequisites:
  - Quartus Prime Pro with System Console
  - FPGA programmed with bringup.sof
  - USB-Blaster II cable connected

Usage:
  python test_jtag_ram.py

Author: A10PED Neuromorphic Project
License: BSD-3-Clause
"""

import subprocess
import sys
import time
from pathlib import Path


class JTAGTester:
    """Test JTAG-to-Avalon bridge and on-chip RAM"""

    def __init__(self):
        """Initialize JTAG tester"""
        self.system_console = self._find_system_console()
        self.ram_base = 0x0000
        self.ram_size = 4096  # 4KB

    def _find_system_console(self) -> str:
        """Find system-console executable"""
        # Common Quartus installation paths
        paths = [
            "/opt/intelFPGA_pro/23.4/quartus/sopc_builder/bin/system-console",
            "/opt/intelFPGA/23.4/quartus/sopc_builder/bin/system-console",
            "C:/intelFPGA_pro/23.4/quartus/sopc_builder/bin/system-console.exe",
            "system-console"  # In PATH
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"✅ Found System Console: {path}")
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        print("ERROR: System Console not found!")
        print("Please install Intel Quartus Prime Pro and add system-console to PATH")
        sys.exit(1)

    def run_tcl_command(self, tcl_script: str) -> tuple[bool, str]:
        """
        Execute Tcl script via System Console

        Args:
            tcl_script: Tcl commands to execute

        Returns:
            (success, output) tuple
        """
        try:
            # Run system-console in CLI mode
            result = subprocess.run(
                [self.system_console, "--cli", "--script=-"],
                input=tcl_script.encode(),
                capture_output=True,
                timeout=30
            )

            output = result.stdout.decode() + result.stderr.decode()
            return (result.returncode == 0, output)

        except subprocess.TimeoutExpired:
            return (False, "Timeout waiting for System Console")
        except Exception as e:
            return (False, f"Error running Tcl command: {e}")

    def test_connection(self) -> bool:
        """Test JTAG cable connection and enumerate devices"""
        print("\n" + "="*60)
        print("TEST 1: JTAG Cable Connection")
        print("="*60)

        tcl_script = """
# Get available JTAG cables
set cables [get_service_paths master]
if {[llength $cables] == 0} {
    puts "ERROR: No JTAG cables found"
    exit 1
}

# Show available cables
puts "Available JTAG cables:"
foreach cable $cables {
    puts "  $cable"
}

# Open first cable
set cable [lindex $cables 0]
puts "\\nUsing cable: $cable"
open_service master $cable

# Check connection
set status [master_read_32 $cable 0x0 1]
puts "Initial read from 0x0000: $status"

close_service master $cable
puts "SUCCESS: JTAG connection verified"
exit 0
"""

        success, output = self.run_tcl_command(tcl_script)
        print(output)

        if success:
            print("✅ PASS: JTAG connection successful\n")
        else:
            print("❌ FAIL: Could not connect via JTAG\n")
            print("Troubleshooting:")
            print("  1. Check USB-Blaster II cable is connected")
            print("  2. Check FPGA is powered on")
            print("  3. Run: jtagconfig  (should show cable)")
            print("  4. Program FPGA: quartus_pgm -l (list cables)")

        return success

    def test_ram_readwrite(self) -> bool:
        """Test on-chip RAM read/write functionality"""
        print("\n" + "="*60)
        print("TEST 2: On-Chip RAM Read/Write")
        print("="*60)

        tcl_script = """
# Open JTAG master
set cables [get_service_paths master]
set cable [lindex $cables 0]
open_service master $cable

puts "Testing 4KB on-chip RAM at address 0x0000..."
puts ""

# Test pattern 1: Sequential values
puts "Pattern 1: Writing sequential values (0x00, 0x01, 0x02, ...)..."
for {set i 0} {$i < 256} {incr i} {
    master_write_32 $cable [expr $i * 4] $i
}

puts "Reading back and verifying..."
set errors 0
for {set i 0} {$i < 256} {incr i} {
    set addr [expr $i * 4]
    set readback [master_read_32 $cable $addr 1]
    if {$readback != $i} {
        puts "ERROR at 0x[format %04X $addr]: expected $i, got $readback"
        incr errors
    }
}
if {$errors == 0} {
    puts "✅ Pattern 1 PASS: All 256 words verified"
} else {
    puts "❌ Pattern 1 FAIL: $errors errors"
}
puts ""

# Test pattern 2: Alternating 0x55 / 0xAA
puts "Pattern 2: Writing alternating 0x55555555 / 0xAAAAAAAA..."
for {set i 0} {$i < 256} {incr i} {
    if {$i % 2 == 0} {
        master_write_32 $cable [expr $i * 4] 0x55555555
    } else {
        master_write_32 $cable [expr $i * 4] 0xAAAAAAAA
    }
}

puts "Reading back and verifying..."
set errors 0
for {set i 0} {$i < 256} {incr i} {
    set addr [expr $i * 4]
    set readback [master_read_32 $cable $addr 1]
    set expected [expr {$i % 2 == 0 ? 0x55555555 : 0xAAAAAAAA}]
    if {$readback != $expected} {
        puts "ERROR at 0x[format %04X $addr]: expected 0x[format %08X $expected], got 0x[format %08X $readback]"
        incr errors
    }
}
if {$errors == 0} {
    puts "✅ Pattern 2 PASS: All 256 words verified"
} else {
    puts "❌ Pattern 2 FAIL: $errors errors"
}
puts ""

# Test pattern 3: Walking 1s
puts "Pattern 3: Writing walking 1s (0x00000001, 0x00000002, 0x00000004, ...)..."
for {set i 0} {$i < 32} {incr i} {
    set pattern [expr 1 << $i]
    master_write_32 $cable [expr $i * 4] $pattern
}

puts "Reading back and verifying..."
set errors 0
for {set i 0} {$i < 32} {incr i} {
    set addr [expr $i * 4]
    set expected [expr 1 << $i]
    set readback [master_read_32 $cable $addr 1]
    if {$readback != $expected} {
        puts "ERROR at 0x[format %04X $addr]: expected 0x[format %08X $expected], got 0x[format %08X $readback]"
        incr errors
    }
}
if {$errors == 0} {
    puts "✅ Pattern 3 PASS: All 32 words verified"
} else {
    puts "❌ Pattern 3 FAIL: $errors errors"
}

close_service master $cable
exit 0
"""

        success, output = self.run_tcl_command(tcl_script)
        print(output)

        if success and "FAIL" not in output:
            print("✅ PASS: RAM read/write successful\n")
        else:
            print("❌ FAIL: RAM errors detected\n")

        return success and "FAIL" not in output

    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("\n")
        print("="*60)
        print(" A10PED Neuromorphic - JTAG RAM Test Suite")
        print(" Milestone 1.2: JTAG Bring-Up Validation")
        print("="*60)

        all_pass = True

        # Test 1: Connection
        if not self.test_connection():
            all_pass = False
            print("\n⚠️  Cannot proceed without JTAG connection")
            return False

        time.sleep(1)

        # Test 2: RAM read/write
        if not self.test_ram_readwrite():
            all_pass = False

        # Final summary
        print("\n" + "="*60)
        if all_pass:
            print(" ✅ ALL TESTS PASSED - MILESTONE 1.2 COMPLETE!")
        else:
            print(" ❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*60)
        print("")

        return all_pass


def main():
    """Main entry point"""
    tester = JTAGTester()

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
