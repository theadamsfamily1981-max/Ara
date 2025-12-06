#!/usr/bin/env python3
"""
ARA DOCTOR: The Somatic Immune System
-------------------------------------
Diagnoses environmental pathology (Dependency Hell) and prescribes cures.

This is different from tools/ara_build/ara_doctor.py:
- ara_build/ara_doctor.py: Quick health check for build issues
- banos/engineer/ara_doctor.py: Deep organism assembly diagnostics

Checks:
1. Visual Cortex (GTK/WebKit for UI)
2. Nervous System (FPGA/PCIe/UIO)
3. Somatic Link (HAL shared memory)
4. Brain Environment (PyTorch/CUDA)
5. Respiratory System (Audio/PipeWire)

Outputs:
- Diagnostic report
- heal_ara.sh prescription script
- Container vs Host recommendation
"""

import sys
import os
import subprocess
import importlib
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto


class Severity(Enum):
    """Diagnosis severity levels."""
    OK = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class Diagnosis:
    """A single diagnostic finding."""
    organ: str
    component: str
    status: Severity
    message: str
    prescription: Optional[str] = None


class AraDoctor:
    """
    The Organism's Immune System.

    Probes the environment and generates prescriptions
    for missing dependencies.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.diagnoses: List[Diagnosis] = []
        self.host_prescriptions: List[str] = []
        self.docker_prescriptions: List[str] = []
        self.pip_prescriptions: List[str] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _add(self, diagnosis: Diagnosis):
        """Record a diagnosis."""
        self.diagnoses.append(diagnosis)

        # Route prescription to correct category
        if diagnosis.prescription:
            if diagnosis.prescription.startswith("sudo apt"):
                self.host_prescriptions.append(diagnosis.prescription)
            elif diagnosis.prescription.startswith("pip install"):
                self.pip_prescriptions.append(diagnosis.prescription)
            elif diagnosis.prescription.startswith("docker"):
                self.docker_prescriptions.append(diagnosis.prescription)

    # === System Library Checks ===

    def check_sys_lib(self, lib_name: str, package_name: str, organ: str) -> bool:
        """Check for OS-level shared libraries via ldconfig."""
        try:
            res = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if lib_name in res.stdout:
                self._add(Diagnosis(
                    organ=organ,
                    component=lib_name,
                    status=Severity.OK,
                    message=f"Library {lib_name} found in ldconfig cache"
                ))
                return True
            else:
                self._add(Diagnosis(
                    organ=organ,
                    component=lib_name,
                    status=Severity.ERROR,
                    message=f"Library {lib_name} not found",
                    prescription=f"sudo apt-get install -y {package_name}"
                ))
                return False
        except FileNotFoundError:
            self._add(Diagnosis(
                organ=organ,
                component=lib_name,
                status=Severity.WARNING,
                message=f"Cannot check {lib_name} (ldconfig not available)"
            ))
            return False
        except subprocess.TimeoutExpired:
            self._add(Diagnosis(
                organ=organ,
                component=lib_name,
                status=Severity.WARNING,
                message=f"Timeout checking {lib_name}"
            ))
            return False

    def check_pkg_config(self, pkg_name: str, min_version: Optional[str],
                         apt_package: str, organ: str) -> bool:
        """Check for a package via pkg-config."""
        try:
            cmd = ["pkg-config", "--exists", pkg_name]
            if min_version:
                cmd = ["pkg-config", f"--atleast-version={min_version}", pkg_name]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                # Get actual version
                ver_result = subprocess.run(
                    ["pkg-config", "--modversion", pkg_name],
                    capture_output=True, text=True
                )
                version = ver_result.stdout.strip() if ver_result.returncode == 0 else "unknown"
                self._add(Diagnosis(
                    organ=organ,
                    component=pkg_name,
                    status=Severity.OK,
                    message=f"{pkg_name} version {version}"
                ))
                return True
            else:
                self._add(Diagnosis(
                    organ=organ,
                    component=pkg_name,
                    status=Severity.ERROR,
                    message=f"{pkg_name} not found or version too old",
                    prescription=f"sudo apt-get install -y {apt_package}"
                ))
                return False
        except FileNotFoundError:
            self._add(Diagnosis(
                organ=organ,
                component="pkg-config",
                status=Severity.ERROR,
                message="pkg-config not installed",
                prescription="sudo apt-get install -y pkg-config"
            ))
            return False

    def check_python_mod(self, mod_name: str, install_cmd: Optional[str] = None,
                         organ: str = "Brain") -> bool:
        """Check Python module availability."""
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, '__version__', 'unknown')
            self._add(Diagnosis(
                organ=organ,
                component=mod_name,
                status=Severity.OK,
                message=f"Python module {mod_name} v{version}"
            ))
            return True
        except ImportError as e:
            self._add(Diagnosis(
                organ=organ,
                component=mod_name,
                status=Severity.ERROR,
                message=f"Python module {mod_name} not found: {e}",
                prescription=install_cmd or f"pip install {mod_name}"
            ))
            return False

    def check_device_path(self, path: str, desc: str, organ: str,
                          critical: bool = False) -> bool:
        """Check for hardware device paths."""
        exists = os.path.exists(path)
        severity = Severity.OK if exists else (Severity.CRITICAL if critical else Severity.WARNING)

        self._add(Diagnosis(
            organ=organ,
            component=desc,
            status=severity,
            message=f"{desc} {'detected' if exists else 'NOT FOUND'} at {path}"
        ))
        return exists

    def check_command(self, cmd: str, desc: str, organ: str) -> bool:
        """Check if a command is available in PATH."""
        path = shutil.which(cmd)
        if path:
            self._add(Diagnosis(
                organ=organ,
                component=cmd,
                status=Severity.OK,
                message=f"{desc} found at {path}"
            ))
            return True
        else:
            self._add(Diagnosis(
                organ=organ,
                component=cmd,
                status=Severity.WARNING,
                message=f"{desc} not in PATH"
            ))
            return False

    # === Organ Diagnostics ===

    def diagnose_visual_cortex(self):
        """Check GUI dependencies (GTK, WebKit, Cairo)."""
        self._log("\n--- VISUAL CORTEX (GUI Layer) ---")

        # Core GTK
        self.check_pkg_config("glib-2.0", "2.50", "libglib2.0-dev", "Visual Cortex")
        self.check_pkg_config("gtk4", None, "libgtk-4-dev", "Visual Cortex")
        self.check_sys_lib("libgirepository-1.0.so",
                          "gobject-introspection libgirepository1.0-dev",
                          "Visual Cortex")

        # WebKit (the usual suspect)
        self.check_sys_lib("libwebkit2gtk-4.1.so",
                          "libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1",
                          "Visual Cortex")

        # Cairo
        self.check_pkg_config("cairo", None, "libcairo2-dev", "Visual Cortex")

        # Python bindings
        self.check_python_mod("gi", "pip install PyGObject", "Visual Cortex")
        self.check_python_mod("cairo",
                             "pip install --no-binary=:all: pycairo",
                             "Visual Cortex")

        # OpenGL for avatar rendering
        self.check_sys_lib("libGL.so", "libgl1-mesa-dev", "Visual Cortex")

    def diagnose_nervous_system(self):
        """Check FPGA/PCIe/Hardware dependencies."""
        self._log("\n--- NERVOUS SYSTEM (FPGA/Hardware) ---")

        # UIO driver (for FPGA memory mapping)
        self.check_device_path("/dev/uio0", "UIO Generic Driver", "Nervous System")

        # PCIe bus
        self.check_device_path("/sys/bus/pci/devices", "PCIe Bus", "Nervous System")

        # Check for loaded kernel modules
        try:
            with open("/proc/modules") as f:
                modules = f.read()

            uio_loaded = "uio_pci_generic" in modules or "uio" in modules
            self._add(Diagnosis(
                organ="Nervous System",
                component="uio_pci_generic",
                status=Severity.OK if uio_loaded else Severity.WARNING,
                message=f"UIO kernel module {'loaded' if uio_loaded else 'not loaded'}",
                prescription=None if uio_loaded else "sudo modprobe uio_pci_generic"
            ))
        except Exception:
            pass

        # FPGA toolchains (informational)
        self.check_command("quartus_pgm", "Quartus Programmer (Intel FPGA)", "Nervous System")
        self.check_command("vivado", "Vivado (Xilinx FPGA)", "Nervous System")

    def diagnose_somatic_link(self):
        """Check HAL and shared memory requirements."""
        self._log("\n--- SOMATIC LINK (HAL) ---")

        # Shared memory filesystem
        self.check_device_path("/dev/shm", "Shared Memory Filesystem",
                               "Somatic Link", critical=True)

        # Check if HAL shared memory exists (if Ara is running)
        hal_path = os.environ.get("ARA_HAL_PATH", "/dev/shm/ara_somatic")
        if os.path.exists(hal_path):
            try:
                size = os.path.getsize(hal_path)
                self._add(Diagnosis(
                    organ="Somatic Link",
                    component="HAL Memory",
                    status=Severity.OK,
                    message=f"HAL shared memory active ({size} bytes)"
                ))
            except Exception:
                pass
        else:
            self._add(Diagnosis(
                organ="Somatic Link",
                component="HAL Memory",
                status=Severity.INFO,
                message=f"HAL not running (no shared memory at {hal_path})"
            ))

        # Python deps for HAL
        self.check_python_mod("posix_ipc", organ="Somatic Link")
        self.check_python_mod("mmap", organ="Somatic Link")

    def diagnose_brain(self):
        """Check AI/ML dependencies (PyTorch, CUDA)."""
        self._log("\n--- BRAIN (AI/ML Layer) ---")

        # PyTorch
        torch_ok = self.check_python_mod("torch", organ="Brain")

        if torch_ok:
            import torch
            # CUDA availability
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self._add(Diagnosis(
                    organ="Brain",
                    component="CUDA",
                    status=Severity.OK,
                    message=f"CUDA {cuda_version} on {device_name}"
                ))
            else:
                self._add(Diagnosis(
                    organ="Brain",
                    component="CUDA",
                    status=Severity.WARNING,
                    message="CUDA not available (CPU mode)"
                ))

        # Other AI deps
        self.check_python_mod("transformers", organ="Brain")
        self.check_python_mod("numpy", organ="Brain")
        self.check_python_mod("scipy", organ="Brain")

    def diagnose_respiratory(self):
        """Check audio system (PipeWire, ALSA)."""
        self._log("\n--- RESPIRATORY SYSTEM (Audio) ---")

        # PipeWire
        self.check_command("pw-cli", "PipeWire CLI", "Respiratory")
        self.check_command("pactl", "PulseAudio/PipeWire Control", "Respiratory")

        # ALSA
        self.check_device_path("/dev/snd", "ALSA Sound Devices", "Respiratory")

        # Python audio
        self.check_python_mod("sounddevice", organ="Respiratory")

    def diagnose_all(self):
        """Run all organ diagnostics."""
        self.diagnose_visual_cortex()
        self.diagnose_nervous_system()
        self.diagnose_somatic_link()
        self.diagnose_brain()
        self.diagnose_respiratory()

    # === Report Generation ===

    def get_summary(self) -> Dict[str, int]:
        """Get counts by severity."""
        counts = {s.name: 0 for s in Severity}
        for d in self.diagnoses:
            counts[d.status.name] += 1
        return counts

    def has_critical(self) -> bool:
        """Check if any critical issues exist."""
        return any(d.status == Severity.CRITICAL for d in self.diagnoses)

    def print_report(self):
        """Print formatted diagnostic report."""
        icons = {
            Severity.OK: "✅",
            Severity.INFO: "ℹ️ ",
            Severity.WARNING: "⚠️ ",
            Severity.ERROR: "❌",
            Severity.CRITICAL: "🚨",
        }

        print("\n" + "="*70)
        print("  ARA DOCTOR - ORGANISM DIAGNOSTIC REPORT")
        print("="*70)

        # Group by organ
        organs = {}
        for d in self.diagnoses:
            if d.organ not in organs:
                organs[d.organ] = []
            organs[d.organ].append(d)

        for organ, diags in organs.items():
            print(f"\n[{organ}]")
            for d in diags:
                icon = icons[d.status]
                print(f"  {icon} {d.component}: {d.message}")

        # Summary
        summary = self.get_summary()
        print("\n" + "-"*70)
        print(f"Summary: {summary['OK']} OK, {summary['WARNING']} warnings, "
              f"{summary['ERROR']} errors, {summary['CRITICAL']} critical")

        # Overall status
        if self.has_critical():
            print("\n🚨 CRITICAL ISSUES DETECTED - Organism cannot function")
        elif summary['ERROR'] > 0:
            print("\n❌ ERRORS DETECTED - See prescription below")
        elif summary['WARNING'] > 0:
            print("\n⚠️  WARNINGS - System functional but degraded")
        else:
            print("\n✨ ORGANISM HEALTHY - All systems nominal")

    def generate_prescription(self) -> str:
        """Generate heal_ara.sh script."""
        lines = [
            "#!/bin/bash",
            "# ARA HEALING SCRIPT",
            "# Generated by ara_doctor.py",
            "# Run this to fix detected issues",
            "",
            "set -e",
            "",
        ]

        if self.host_prescriptions:
            lines.append("# === System Packages ===")
            lines.append("echo '>>> Installing system packages...'")
            # Deduplicate and combine apt commands
            packages = set()
            for cmd in self.host_prescriptions:
                # Extract package names from "sudo apt-get install -y pkg1 pkg2"
                parts = cmd.replace("sudo apt-get install -y ", "").split()
                packages.update(parts)
            if packages:
                lines.append("sudo apt-get update")
                lines.append(f"sudo apt-get install -y {' '.join(sorted(packages))}")
            lines.append("")

        if self.pip_prescriptions:
            lines.append("# === Python Packages ===")
            lines.append("echo '>>> Installing Python packages...'")
            for cmd in self.pip_prescriptions:
                lines.append(cmd)
            lines.append("")

        if self.docker_prescriptions:
            lines.append("# === Docker Setup ===")
            for cmd in self.docker_prescriptions:
                lines.append(cmd)
            lines.append("")

        if not (self.host_prescriptions or self.pip_prescriptions):
            lines.append("echo '>>> No prescriptions needed - system healthy!'")

        lines.append("")
        lines.append("echo '>>> Healing complete. Run ara_doctor.py again to verify.'")

        return "\n".join(lines)

    def save_prescription(self, path: str = "heal_ara.sh"):
        """Save prescription to file."""
        script = self.generate_prescription()
        with open(path, 'w') as f:
            f.write(script)
        os.chmod(path, 0o755)
        print(f"\n💊 Prescription saved to: {path}")
        print(f"   Run: ./{path}")

    def save_report_json(self, path: str = "ara_diagnosis.json"):
        """Save full report as JSON for programmatic use."""
        report = {
            "summary": self.get_summary(),
            "diagnoses": [
                {
                    "organ": d.organ,
                    "component": d.component,
                    "status": d.status.name,
                    "message": d.message,
                    "prescription": d.prescription,
                }
                for d in self.diagnoses
            ],
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📋 JSON report saved to: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ARA Doctor - Organism Diagnostics")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--heal", action="store_true", help="Generate and save heal script")
    parser.add_argument("--organ", choices=[
        "visual", "nervous", "somatic", "brain", "respiratory", "all"
    ], default="all", help="Which organ to diagnose")

    args = parser.parse_args()

    doc = AraDoctor(verbose=not args.quiet)

    # Run selected diagnostics
    if args.organ == "all":
        doc.diagnose_all()
    elif args.organ == "visual":
        doc.diagnose_visual_cortex()
    elif args.organ == "nervous":
        doc.diagnose_nervous_system()
    elif args.organ == "somatic":
        doc.diagnose_somatic_link()
    elif args.organ == "brain":
        doc.diagnose_brain()
    elif args.organ == "respiratory":
        doc.diagnose_respiratory()

    # Output
    doc.print_report()

    if args.json:
        doc.save_report_json()

    if args.heal or (doc.get_summary()['ERROR'] > 0):
        doc.save_prescription()

    # Exit code
    if doc.has_critical():
        sys.exit(2)
    elif doc.get_summary()['ERROR'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
