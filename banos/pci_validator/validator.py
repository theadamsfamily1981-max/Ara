#!/usr/bin/env python3
"""
PCIe Link Validator

Validates PCIe link status, speed negotiation, and lane configuration.
Supports Gen1-Gen6, x1-x32 configurations.

Ara Validates -> Negotiates -> Autotunes -> Maximizes Bandwidth
"""

from __future__ import annotations
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Constants & Types
# =============================================================================

class PCIeGeneration(Enum):
    """PCIe generation with GT/s per lane."""
    GEN1 = (1, 2.5, 0.25)    # (gen, GT/s, GB/s per lane)
    GEN2 = (2, 5.0, 0.5)
    GEN3 = (3, 8.0, 0.985)   # 128b/130b encoding
    GEN4 = (4, 16.0, 1.969)
    GEN5 = (5, 32.0, 3.938)
    GEN6 = (6, 64.0, 7.563)  # PAM4

    @property
    def gen(self) -> int:
        return self.value[0]

    @property
    def gtps(self) -> float:
        return self.value[1]

    @property
    def gbps_per_lane(self) -> float:
        return self.value[2]

    @classmethod
    def from_speed(cls, gtps: float) -> "PCIeGeneration":
        """Get generation from GT/s value."""
        speed_map = {2.5: cls.GEN1, 5.0: cls.GEN2, 8.0: cls.GEN3,
                     16.0: cls.GEN4, 32.0: cls.GEN5, 64.0: cls.GEN6}
        return speed_map.get(gtps, cls.GEN1)


class AraTier(Enum):
    """Ara tiered compatibility based on bandwidth."""
    SOVEREIGN = ("sovereign", 32.0, "GPU Rescoring, HTC FPGA")
    HOMEOSTASIS = ("homeostasis", 8.0, "Reflex eBPF, Storage")
    REFLEX = ("reflex", 4.0, "Sensors, Serial")
    LEGACY = ("legacy", 0.5, "Fallback UIO")

    @property
    def name_str(self) -> str:
        return self.value[0]

    @property
    def min_gbps(self) -> float:
        return self.value[1]

    @property
    def workloads(self) -> str:
        return self.value[2]


@dataclass
class PCIeLinkStatus:
    """Current PCIe link status."""
    speed_gtps: float = 0.0
    width: int = 0
    generation: Optional[PCIeGeneration] = None
    bandwidth_gbps: float = 0.0

    # Capability (maximum)
    max_speed_gtps: float = 0.0
    max_width: int = 0
    max_generation: Optional[PCIeGeneration] = None
    max_bandwidth_gbps: float = 0.0

    # Status flags
    link_active: bool = False
    aspm_enabled: bool = False
    aer_errors: int = 0

    # Lane status (for detailed validation)
    lane_status: Dict[int, bool] = field(default_factory=dict)

    @property
    def negotiated_ratio(self) -> float:
        """Ratio of negotiated to maximum bandwidth."""
        if self.max_bandwidth_gbps > 0:
            return self.bandwidth_gbps / self.max_bandwidth_gbps
        return 0.0

    @property
    def is_downgraded(self) -> bool:
        """Check if link is running below max capability."""
        return (self.speed_gtps < self.max_speed_gtps or
                self.width < self.max_width)


@dataclass
class ValidationResult:
    """Result of PCIe link validation."""
    pci_address: str
    passed: bool
    tier: AraTier
    link_status: PCIeLinkStatus
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        ls = self.link_status
        return (
            f"PCIe {self.pci_address}: [{status}] "
            f"Gen{ls.generation.gen if ls.generation else '?'} x{ls.width} "
            f"({ls.bandwidth_gbps:.1f} GB/s) | "
            f"Tier: {self.tier.name_str.upper()}"
        )


# =============================================================================
# PCIe Link Validator
# =============================================================================

class PCIeLinkValidator:
    """
    Validates PCIe link status and configuration.

    Supports:
    - Reading link status from sysfs
    - Parsing lspci output
    - Tiered compatibility assessment
    - Recommendations for optimization
    """

    def __init__(self):
        self.sysfs_base = Path("/sys/bus/pci/devices")

    def validate(
        self,
        pci_address: str,
        target_gen: Optional[PCIeGeneration] = None,
        target_width: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate PCIe link for given device.

        Args:
            pci_address: PCI address (e.g., "0000:03:00.0")
            target_gen: Expected generation (optional)
            target_width: Expected lane width (optional)

        Returns:
            ValidationResult with status, tier, and recommendations
        """
        issues: List[str] = []
        recommendations: List[str] = []

        # Get link status
        link_status = self._read_link_status(pci_address)

        if not link_status.link_active:
            issues.append("Link not active")
            return ValidationResult(
                pci_address=pci_address,
                passed=False,
                tier=AraTier.LEGACY,
                link_status=link_status,
                issues=issues,
            )

        # Check if running at expected speed/width
        if target_gen and link_status.generation != target_gen:
            issues.append(
                f"Speed mismatch: expected Gen{target_gen.gen}, "
                f"got Gen{link_status.generation.gen if link_status.generation else '?'}"
            )

        if target_width and link_status.width != target_width:
            issues.append(
                f"Width mismatch: expected x{target_width}, got x{link_status.width}"
            )

        # Check for downgrade
        if link_status.is_downgraded:
            issues.append(
                f"Link downgraded: {link_status.max_speed_gtps} GT/s x{link_status.max_width} -> "
                f"{link_status.speed_gtps} GT/s x{link_status.width}"
            )
            recommendations.append("Check BIOS settings for PCIe speed")
            recommendations.append("Disable ASPM in BIOS")
            recommendations.append("Check physical seating of card")

        # Check ASPM
        if link_status.aspm_enabled:
            issues.append("ASPM enabled (may cause latency)")
            recommendations.append("Disable ASPM: setpci -s <addr> CAP_EXP+10.w=0")

        # Check AER errors
        if link_status.aer_errors > 0:
            issues.append(f"AER errors detected: {link_status.aer_errors}")
            recommendations.append("Check dmesg for AER details")

        # Determine tier
        tier = self._classify_tier(link_status.bandwidth_gbps)

        # Overall pass/fail
        passed = len(issues) == 0 or (
            link_status.link_active and
            link_status.negotiated_ratio >= 0.9 and
            link_status.aer_errors == 0
        )

        return ValidationResult(
            pci_address=pci_address,
            passed=passed,
            tier=tier,
            link_status=link_status,
            issues=issues,
            recommendations=recommendations,
        )

    def validate_all(self) -> List[ValidationResult]:
        """Validate all PCIe devices in system."""
        results = []
        for device_dir in self.sysfs_base.iterdir():
            if device_dir.is_dir():
                pci_addr = device_dir.name
                result = self.validate(pci_addr)
                results.append(result)
        return results

    def _read_link_status(self, pci_address: str) -> PCIeLinkStatus:
        """Read link status from sysfs and lspci."""
        status = PCIeLinkStatus()

        # Try sysfs first
        device_path = self.sysfs_base / pci_address
        if device_path.exists():
            status = self._read_sysfs_status(device_path, status)

        # Fall back to / augment with lspci
        status = self._read_lspci_status(pci_address, status)

        # Calculate bandwidth
        if status.generation:
            status.bandwidth_gbps = status.generation.gbps_per_lane * status.width

        if status.max_generation:
            status.max_bandwidth_gbps = status.max_generation.gbps_per_lane * status.max_width

        return status

    def _read_sysfs_status(self, device_path: Path, status: PCIeLinkStatus) -> PCIeLinkStatus:
        """Read link status from sysfs."""
        # Current speed
        speed_file = device_path / "current_link_speed"
        if speed_file.exists():
            speed_str = speed_file.read_text().strip()
            status.speed_gtps = self._parse_speed(speed_str)
            status.generation = PCIeGeneration.from_speed(status.speed_gtps)
            status.link_active = True

        # Current width
        width_file = device_path / "current_link_width"
        if width_file.exists():
            width_str = width_file.read_text().strip()
            status.width = int(width_str)

        # Max speed
        max_speed_file = device_path / "max_link_speed"
        if max_speed_file.exists():
            speed_str = max_speed_file.read_text().strip()
            status.max_speed_gtps = self._parse_speed(speed_str)
            status.max_generation = PCIeGeneration.from_speed(status.max_speed_gtps)

        # Max width
        max_width_file = device_path / "max_link_width"
        if max_width_file.exists():
            width_str = max_width_file.read_text().strip()
            status.max_width = int(width_str)

        return status

    def _read_lspci_status(self, pci_address: str, status: PCIeLinkStatus) -> PCIeLinkStatus:
        """Read link status from lspci."""
        try:
            result = subprocess.run(
                ["lspci", "-vvv", "-s", pci_address],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return status

            output = result.stdout

            # Parse LnkCap (capability)
            cap_match = re.search(
                r"LnkCap:.*?Speed\s+([\d.]+)GT/s.*?Width\s+x(\d+)",
                output, re.DOTALL
            )
            if cap_match:
                status.max_speed_gtps = float(cap_match.group(1))
                status.max_width = int(cap_match.group(2))
                status.max_generation = PCIeGeneration.from_speed(status.max_speed_gtps)

            # Parse LnkSta (status)
            sta_match = re.search(
                r"LnkSta:.*?Speed\s+([\d.]+)GT/s.*?Width\s+x(\d+)",
                output, re.DOTALL
            )
            if sta_match:
                status.speed_gtps = float(sta_match.group(1))
                status.width = int(sta_match.group(2))
                status.generation = PCIeGeneration.from_speed(status.speed_gtps)
                status.link_active = True

            # Check ASPM
            if "ASPM L0s" in output or "ASPM L1" in output:
                if "ASPM Enabled" in output or "L0s+ L1+" in output:
                    status.aspm_enabled = True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return status

    def _parse_speed(self, speed_str: str) -> float:
        """Parse speed string like '8 GT/s' or '8GT/s'."""
        match = re.search(r"([\d.]+)\s*GT", speed_str)
        if match:
            return float(match.group(1))
        return 0.0

    def _classify_tier(self, bandwidth_gbps: float) -> AraTier:
        """Classify device into Ara tier based on bandwidth."""
        if bandwidth_gbps >= AraTier.SOVEREIGN.min_gbps:
            return AraTier.SOVEREIGN
        elif bandwidth_gbps >= AraTier.HOMEOSTASIS.min_gbps:
            return AraTier.HOMEOSTASIS
        elif bandwidth_gbps >= AraTier.REFLEX.min_gbps:
            return AraTier.REFLEX
        else:
            return AraTier.LEGACY


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate PCIe link configuration"
    )
    parser.add_argument(
        "pci_address",
        nargs="?",
        help="PCI address to validate (e.g., 0000:03:00.0)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all PCIe devices"
    )
    parser.add_argument(
        "--gen", "-g",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Expected PCIe generation"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        help="Expected lane width"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    validator = PCIeLinkValidator()

    if args.all:
        results = validator.validate_all()
    elif args.pci_address:
        target_gen = PCIeGeneration(args.gen) if args.gen else None
        results = [validator.validate(
            args.pci_address,
            target_gen=target_gen,
            target_width=args.width
        )]
    else:
        parser.print_help()
        return 1

    if args.json:
        import json
        output = []
        for r in results:
            output.append({
                "pci_address": r.pci_address,
                "passed": r.passed,
                "tier": r.tier.name_str,
                "bandwidth_gbps": r.link_status.bandwidth_gbps,
                "generation": r.link_status.generation.gen if r.link_status.generation else None,
                "width": r.link_status.width,
                "issues": r.issues,
                "recommendations": r.recommendations,
            })
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print("Ara PCIe Link Validation")
        print("=" * 70)

        for result in results:
            print(f"\n{result.summary()}")
            if result.issues:
                print("  Issues:")
                for issue in result.issues:
                    print(f"    - {issue}")
            if result.recommendations:
                print("  Recommendations:")
                for rec in result.recommendations:
                    print(f"    - {rec}")

        # Summary
        passed = sum(1 for r in results if r.passed)
        print(f"\n{'=' * 70}")
        print(f"Summary: {passed}/{len(results)} devices passed validation")

        tier_counts = {}
        for r in results:
            tier_counts[r.tier.name_str] = tier_counts.get(r.tier.name_str, 0) + 1
        print(f"Tiers: {tier_counts}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    exit(main())
