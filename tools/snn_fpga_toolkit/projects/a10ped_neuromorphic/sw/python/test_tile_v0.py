#!/usr/bin/env python3
"""
A10PED AI Tile v0 - Validation Test Suite

Tests memcopy kernel and basic tile functionality.

Usage:
    python test_tile_v0.py [--tile-id TILE_ID]

Author: A10PED Neuromorphic Project
License: MIT
"""

import argparse
import sys
import time
import numpy as np
from typing import Tuple

from a10ped import AITile, AITileError


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg: str):
    """Print test section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_pass(msg: str):
    """Print test pass message"""
    print(f"{Colors.GREEN}✅ PASS{Colors.RESET}: {msg}")


def print_fail(msg: str):
    """Print test fail message"""
    print(f"{Colors.RED}❌ FAIL{Colors.RESET}: {msg}")


def print_info(msg: str):
    """Print informational message"""
    print(f"{Colors.YELLOW}ℹ️  INFO{Colors.RESET}: {msg}")


class TileV0Validator:
    """Validation test suite for AI Tile v0"""

    def __init__(self, tile_id: int = 0):
        """Initialize validator"""
        self.tile_id = tile_id
        self.tile: AITile = None
        self.tests_passed = 0
        self.tests_failed = 0

    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print_header("A10PED AI Tile v0 - Validation Test Suite")

        try:
            # Test 1: Connection and info
            if not self.test_connection():
                return False

            time.sleep(0.5)

            # Test 2: Status register access
            if not self.test_status_access():
                return False

            time.sleep(0.5)

            # Test 3: Memcopy basic functionality
            if not self.test_memcopy_basic():
                return False

            time.sleep(0.5)

            # Test 4: Memcopy alignment validation
            if not self.test_memcopy_alignment():
                return False

            time.sleep(0.5)

            # Test 5: Memcopy performance
            if not self.test_memcopy_performance():
                return False

        except Exception as e:
            print_fail(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._print_summary()

        return self.tests_failed == 0

    def test_connection(self) -> bool:
        """Test tile connection and read hardware info"""
        print_header(f"TEST 1: Connection and Hardware Info (Tile {self.tile_id})")

        try:
            self.tile = AITile(tile_id=self.tile_id)
            print_pass(f"Connected to {self.tile}")

            info = self.tile.get_info()
            print_info(f"Version: {info.version[0]}.{info.version[1]}.{info.version[2]}")
            print_info(f"Capabilities: 0x{info.capabilities:08X}")
            print_info(f"  - Memcopy:         {info.has_memcopy}")
            print_info(f"  - SNN:             {info.has_snn}")
            print_info(f"  - Topological:     {info.has_topological}")
            print_info(f"  - Interrupts:      {info.has_irq}")
            print_info(f"  - Multi-precision: {info.has_multi_precision}")

            if not info.has_memcopy:
                print_fail("Memcopy capability not available")
                self.tests_failed += 1
                return False

            self.tests_passed += 1
            return True

        except FileNotFoundError as e:
            print_fail(str(e))
            print_info("Make sure a10ped_driver.ko is loaded:")
            print_info("  cd ../driver && make && sudo insmod a10ped_driver.ko")
            self.tests_failed += 1
            return False

    def test_status_access(self) -> bool:
        """Test status register read access"""
        print_header("TEST 2: Status Register Access")

        try:
            status = self.tile.get_status()
            print_info(f"Raw status: 0x{status.raw:08X}")
            print_info(f"  - BUSY:            {status.busy}")
            print_info(f"  - DONE:            {status.done}")
            print_info(f"  - ERROR:           {status.error}")
            print_info(f"  - DDR_READY:       {status.ddr_ready}")
            print_info(f"  - THERMAL_WARNING: {status.thermal_warning}")

            if status.busy:
                print_fail("Tile is unexpectedly busy")
                self.tests_failed += 1
                return False

            if not status.ddr_ready:
                print_fail("DDR4 is not ready (EMIF calibration may have failed)")
                self.tests_failed += 1
                return False

            if status.error:
                error_code = self.tile.get_error_code()
                print_fail(f"Tile is in error state (error code: 0x{error_code:02X})")
                self.tests_failed += 1
                return False

            print_pass("Status register read successful")
            self.tests_passed += 1
            return True

        except Exception as e:
            print_fail(f"Status access failed: {e}")
            self.tests_failed += 1
            return False

    def test_memcopy_basic(self) -> bool:
        """Test basic memcopy functionality"""
        print_header("TEST 3: Basic Memcopy (4KB transfer)")

        try:
            # Simple 4KB memcopy test
            src_addr = 0x0000
            dst_addr = 0x100000  # 1MB offset
            length = 4096

            print_info(f"Memcopy: 0x{src_addr:X} → 0x{dst_addr:X} ({length} bytes)")

            start = time.time()
            self.tile.memcopy(src=src_addr, dst=dst_addr, length=length)
            elapsed = time.time() - start

            status = self.tile.get_status()
            if status.error:
                error_code = self.tile.get_error_code()
                print_fail(f"Memcopy failed (error code: 0x{error_code:02X})")
                self.tests_failed += 1
                return False

            perf_cycles = self.tile.get_perf_cycles()
            print_info(f"Completed in {elapsed*1000:.2f} ms ({perf_cycles} cycles)")

            print_pass("Basic memcopy successful")
            self.tests_passed += 1
            return True

        except AITileError as e:
            print_fail(f"Memcopy failed: {e}")
            self.tests_failed += 1
            return False

    def test_memcopy_alignment(self) -> bool:
        """Test memcopy alignment validation"""
        print_header("TEST 4: Memcopy Alignment Validation")

        # Test cases: (src, dst, length, should_fail)
        test_cases = [
            (0x0000, 0x1000, 4096, False, "Valid: all 64-byte aligned"),
            (0x0040, 0x1000, 4096, False, "Valid: src at 64-byte boundary"),
            (0x0001, 0x1000, 4096, True, "Invalid: src misaligned"),
            (0x0000, 0x1001, 4096, True, "Invalid: dst misaligned"),
            (0x0000, 0x1000, 4095, True, "Invalid: length misaligned"),
        ]

        passed = 0
        for src, dst, length, should_fail, desc in test_cases:
            try:
                self.tile.memcopy(src, dst, length)
                if should_fail:
                    print_fail(f"{desc} - Expected failure but succeeded")
                else:
                    print_pass(f"{desc}")
                    passed += 1
            except ValueError:
                if should_fail:
                    print_pass(f"{desc}")
                    passed += 1
                else:
                    print_fail(f"{desc} - Unexpected failure")

        if passed == len(test_cases):
            print_pass("All alignment validation tests passed")
            self.tests_passed += 1
            return True
        else:
            print_fail(f"Alignment validation: {passed}/{len(test_cases)} passed")
            self.tests_failed += 1
            return False

    def test_memcopy_performance(self) -> bool:
        """Test memcopy performance with various transfer sizes"""
        print_header("TEST 5: Memcopy Performance")

        transfer_sizes = [
            (4096, "4KB"),
            (65536, "64KB"),
            (1048576, "1MB"),
            (4194304, "4MB"),
        ]

        results = []

        for length, desc in transfer_sizes:
            try:
                src_addr = 0x0000
                dst_addr = 0x10000000  # 256MB offset

                # Warm-up
                self.tile.memcopy(src_addr, dst_addr, length)

                # Timed run
                start = time.time()
                self.tile.memcopy(src_addr, dst_addr, length)
                elapsed = time.time() - start

                perf_cycles = self.tile.get_perf_cycles()
                bandwidth_mbps = (length / (1024 * 1024)) / elapsed

                results.append((desc, length, elapsed, perf_cycles, bandwidth_mbps))
                print_info(f"{desc:>6}: {elapsed*1000:7.2f} ms  "
                          f"{perf_cycles:10} cycles  "
                          f"{bandwidth_mbps:7.1f} MB/s")

            except AITileError as e:
                print_fail(f"{desc} memcopy failed: {e}")
                self.tests_failed += 1
                return False

        # Check if performance is reasonable
        # For PCIe Gen3 x8, theoretical max is ~8GB/s
        # We expect at least 100 MB/s for small transfers
        min_bandwidth = min(r[4] for r in results)
        if min_bandwidth < 10.0:
            print_fail(f"Performance too low: {min_bandwidth:.1f} MB/s (expected >10 MB/s)")
            print_info("This may indicate PCIe or DDR4 configuration issues")
            self.tests_failed += 1
            return False

        print_pass(f"Performance acceptable (min: {min_bandwidth:.1f} MB/s)")
        self.tests_passed += 1
        return True

    def _print_summary(self):
        """Print test summary"""
        print_header("Test Summary")

        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"Total tests:  {total_tests}")
        print(f"Passed:       {Colors.GREEN}{self.tests_passed}{Colors.RESET}")
        print(f"Failed:       {Colors.RED}{self.tests_failed}{Colors.RESET}")
        print(f"Pass rate:    {pass_rate:.1f}%")
        print()

        if self.tests_failed == 0:
            print(f"{Colors.BOLD}{Colors.GREEN}✅ ALL TESTS PASSED{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.GREEN}AI Tile v0 is operational and ready for SNN workloads!{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}❌ SOME TESTS FAILED{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.RED}Review errors above and check hardware/driver setup{Colors.RESET}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="A10PED AI Tile v0 validation test suite"
    )
    parser.add_argument(
        '--tile-id',
        type=int,
        default=0,
        choices=[0, 1],
        help='Tile ID to test (0 or 1 for dual-tile boards)'
    )
    args = parser.parse_args()

    validator = TileV0Validator(tile_id=args.tile_id)
    success = validator.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
