#!/usr/bin/env python3
"""
Ara Doctor - System Health Scanner.

Scans the system to check if all of Ara's organs are healthy:
- Python environment
- GPU / CUDA
- FPGA tools
- GTK / GUI dependencies
- Build tools
- HAL / shared memory

Usage:
    ara-doctor              Full scan
    ara-doctor --quick      Quick scan (skip slow checks)
    ara-doctor --json       Output as JSON
    ara-doctor --fix        Suggest fixes for each issue
"""

import sys
import os
import shutil
import subprocess
import json
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum


class HealthStatus(Enum):
    """Health check result."""
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    detail: str = ""
    fix_hint: str = ""
    fix_command: str = ""


@dataclass
class SystemHealth:
    """Overall system health report."""
    timestamp: str = ""
    platform: str = ""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == HealthStatus.OK)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == HealthStatus.WARN)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == HealthStatus.FAIL)

    @property
    def is_healthy(self) -> bool:
        return self.fail_count == 0


def check_command(name: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(name) is not None


def run_command(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a command and return (code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return 127, "", "command not found"
    except Exception as e:
        return 1, "", str(e)


def check_python_import(module: str, package: str = None) -> Tuple[bool, str]:
    """Try to import a Python module."""
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return True, str(version)
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# =============================================================================
# Individual Health Checks
# =============================================================================

def check_python() -> CheckResult:
    """Check Python version."""
    version = sys.version.split()[0]
    major, minor = sys.version_info[:2]

    if major < 3 or (major == 3 and minor < 9):
        return CheckResult(
            name="Python",
            status=HealthStatus.WARN,
            message=f"Python {version} (3.9+ recommended)",
            fix_hint="Upgrade Python for best compatibility"
        )

    return CheckResult(
        name="Python",
        status=HealthStatus.OK,
        message=f"Python {version}"
    )


def check_venv() -> CheckResult:
    """Check virtual environment."""
    venv = os.environ.get("VIRTUAL_ENV")
    conda = os.environ.get("CONDA_DEFAULT_ENV")

    if venv:
        return CheckResult(
            name="Virtual Env",
            status=HealthStatus.OK,
            message=f"venv: {Path(venv).name}"
        )
    elif conda:
        return CheckResult(
            name="Virtual Env",
            status=HealthStatus.OK,
            message=f"conda: {conda}"
        )
    else:
        return CheckResult(
            name="Virtual Env",
            status=HealthStatus.WARN,
            message="No virtual environment active",
            fix_hint="Using a venv prevents package conflicts",
            fix_command="python -m venv ~/.ara/venv && source ~/.ara/venv/bin/activate"
        )


def check_pytorch() -> CheckResult:
    """Check PyTorch installation."""
    ok, version = check_python_import("torch")

    if not ok:
        return CheckResult(
            name="PyTorch",
            status=HealthStatus.FAIL,
            message="Not installed",
            detail=version,
            fix_hint="Install PyTorch for neural network support",
            fix_command="pip install torch"
        )

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            return CheckResult(
                name="PyTorch",
                status=HealthStatus.OK,
                message=f"{version} (CUDA {cuda_version})",
                detail=f"GPU: {device_name}"
            )
        else:
            return CheckResult(
                name="PyTorch",
                status=HealthStatus.WARN,
                message=f"{version} (CPU only)",
                fix_hint="CUDA not available - GPU acceleration disabled"
            )
    except Exception as e:
        return CheckResult(
            name="PyTorch",
            status=HealthStatus.WARN,
            message=f"{version} (status check failed)",
            detail=str(e)
        )


def check_nvidia() -> CheckResult:
    """Check NVIDIA driver and GPU."""
    if not check_command("nvidia-smi"):
        return CheckResult(
            name="NVIDIA GPU",
            status=HealthStatus.WARN,
            message="nvidia-smi not found",
            fix_hint="Install NVIDIA drivers if you have an NVIDIA GPU"
        )

    code, out, err = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])

    if code != 0:
        return CheckResult(
            name="NVIDIA GPU",
            status=HealthStatus.FAIL,
            message="nvidia-smi failed",
            detail=err
        )

    # Parse output
    parts = out.strip().split(", ")
    if len(parts) >= 3:
        gpu_name, driver, memory = parts[0], parts[1], parts[2]
        return CheckResult(
            name="NVIDIA GPU",
            status=HealthStatus.OK,
            message=f"{gpu_name}",
            detail=f"Driver {driver}, {memory}"
        )

    return CheckResult(
        name="NVIDIA GPU",
        status=HealthStatus.OK,
        message=out.strip()[:50]
    )


def check_cuda_toolkit() -> CheckResult:
    """Check CUDA toolkit installation."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not check_command("nvcc"):
        if cuda_home:
            nvcc_path = Path(cuda_home) / "bin" / "nvcc"
            if nvcc_path.exists():
                return CheckResult(
                    name="CUDA Toolkit",
                    status=HealthStatus.WARN,
                    message=f"Found at {cuda_home} but nvcc not in PATH",
                    fix_command=f"export PATH={cuda_home}/bin:$PATH"
                )

        return CheckResult(
            name="CUDA Toolkit",
            status=HealthStatus.WARN,
            message="nvcc not found",
            fix_hint="CUDA toolkit not installed or not in PATH"
        )

    code, out, err = run_command(["nvcc", "--version"])
    if code == 0:
        # Parse version from output
        import re
        match = re.search(r"release (\d+\.\d+)", out)
        version = match.group(1) if match else "unknown"
        return CheckResult(
            name="CUDA Toolkit",
            status=HealthStatus.OK,
            message=f"CUDA {version}"
        )

    return CheckResult(
        name="CUDA Toolkit",
        status=HealthStatus.WARN,
        message="nvcc found but version check failed"
    )


def check_gtk() -> CheckResult:
    """Check GTK / PyGObject."""
    ok, msg = check_python_import("gi")

    if not ok:
        return CheckResult(
            name="PyGObject/GTK",
            status=HealthStatus.WARN,
            message="PyGObject not installed",
            fix_hint="Install GTK development packages and PyGObject",
            fix_command="sudo apt install libgirepository1.0-dev && pip install pygobject"
        )

    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk
        return CheckResult(
            name="PyGObject/GTK",
            status=HealthStatus.OK,
            message="GTK 4.0 available"
        )
    except ValueError:
        try:
            gi.require_version("Gtk", "3.0")
            from gi.repository import Gtk
            return CheckResult(
                name="PyGObject/GTK",
                status=HealthStatus.OK,
                message="GTK 3.0 available (4.0 not found)"
            )
        except Exception:
            pass
    except Exception as e:
        return CheckResult(
            name="PyGObject/GTK",
            status=HealthStatus.WARN,
            message="GTK import failed",
            detail=str(e)
        )

    return CheckResult(
        name="PyGObject/GTK",
        status=HealthStatus.WARN,
        message="GTK not available"
    )


def check_webkit() -> CheckResult:
    """Check WebKitGTK."""
    try:
        import gi
        gi.require_version("WebKit", "6.0")
        from gi.repository import WebKit  # noqa: F401
        return CheckResult(
            name="WebKitGTK",
            status=HealthStatus.OK,
            message="WebKit 6.0"
        )
    except ValueError:
        try:
            import gi
            gi.require_version("WebKit2", "4.1")
            from gi.repository import WebKit2  # noqa: F401
            return CheckResult(
                name="WebKitGTK",
                status=HealthStatus.OK,
                message="WebKit2 4.1"
            )
        except Exception:
            pass
    except ImportError:
        pass
    except Exception as e:
        return CheckResult(
            name="WebKitGTK",
            status=HealthStatus.FAIL,
            message="Import failed",
            detail=str(e)
        )

    return CheckResult(
        name="WebKitGTK",
        status=HealthStatus.WARN,
        message="Not available",
        fix_hint="WebKitGTK needed for vision cortex",
        fix_command="sudo apt install libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1"
    )


def check_hal() -> CheckResult:
    """Check Ara HAL shared memory."""
    hal_path = Path(os.environ.get("ARA_HAL_PATH", "/dev/shm/ara_somatic"))

    if hal_path.exists():
        size = hal_path.stat().st_size
        return CheckResult(
            name="Ara HAL",
            status=HealthStatus.OK,
            message=f"Somatic bus active ({size} bytes)"
        )
    else:
        return CheckResult(
            name="Ara HAL",
            status=HealthStatus.WARN,
            message="Somatic bus not found",
            fix_hint="Start ara-daemon to create HAL"
        )


def check_fpga() -> CheckResult:
    """Check FPGA availability."""
    # Check for Xilinx tools
    if check_command("vivado"):
        return CheckResult(
            name="FPGA Tools",
            status=HealthStatus.OK,
            message="Vivado found"
        )

    # Check for device files
    fpga_devices = list(Path("/dev").glob("xdma*")) + list(Path("/dev").glob("ara*"))
    if fpga_devices:
        return CheckResult(
            name="FPGA Device",
            status=HealthStatus.OK,
            message=f"Device: {fpga_devices[0].name}"
        )

    return CheckResult(
        name="FPGA",
        status=HealthStatus.SKIP,
        message="No FPGA tools/devices found"
    )


def check_build_tools() -> CheckResult:
    """Check common build tools."""
    tools = {
        "gcc": "C compiler",
        "g++": "C++ compiler",
        "make": "Make",
        "cmake": "CMake",
        "pkg-config": "pkg-config",
    }

    missing = []
    found = []

    for tool, desc in tools.items():
        if check_command(tool):
            found.append(tool)
        else:
            missing.append(tool)

    if not missing:
        return CheckResult(
            name="Build Tools",
            status=HealthStatus.OK,
            message=f"All found ({len(found)} tools)"
        )
    elif len(missing) <= 2:
        return CheckResult(
            name="Build Tools",
            status=HealthStatus.WARN,
            message=f"Missing: {', '.join(missing)}",
            fix_command=f"sudo apt install {' '.join(missing)}"
        )
    else:
        return CheckResult(
            name="Build Tools",
            status=HealthStatus.FAIL,
            message=f"Missing {len(missing)} tools",
            detail=f"Missing: {', '.join(missing)}",
            fix_command="sudo apt install build-essential cmake pkg-config"
        )


def check_rust() -> CheckResult:
    """Check Rust toolchain."""
    if not check_command("cargo"):
        return CheckResult(
            name="Rust",
            status=HealthStatus.SKIP,
            message="Not installed",
            fix_hint="Install Rust if needed for dependencies",
            fix_command="curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        )

    code, out, err = run_command(["rustc", "--version"])
    if code == 0:
        version = out.strip().split()[1] if out else "unknown"
        return CheckResult(
            name="Rust",
            status=HealthStatus.OK,
            message=f"rustc {version}"
        )

    return CheckResult(
        name="Rust",
        status=HealthStatus.WARN,
        message="Installed but version check failed"
    )


def check_disk_space() -> CheckResult:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)

        if free_gb < 5:
            return CheckResult(
                name="Disk Space",
                status=HealthStatus.FAIL,
                message=f"Only {free_gb:.1f} GB free",
                fix_hint="Critically low disk space",
                fix_command="sudo apt clean && pip cache purge"
            )
        elif free_gb < 20:
            return CheckResult(
                name="Disk Space",
                status=HealthStatus.WARN,
                message=f"{free_gb:.1f} GB free",
                fix_hint="Consider freeing up space"
            )
        else:
            return CheckResult(
                name="Disk Space",
                status=HealthStatus.OK,
                message=f"{free_gb:.1f} GB free"
            )
    except Exception as e:
        return CheckResult(
            name="Disk Space",
            status=HealthStatus.WARN,
            message="Check failed",
            detail=str(e)
        )


def check_memory() -> CheckResult:
    """Check system memory."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]  # kB
                    meminfo[key] = int(val)

        total_gb = meminfo.get("MemTotal", 0) / (1024 ** 2)
        avail_gb = meminfo.get("MemAvailable", 0) / (1024 ** 2)

        if avail_gb < 2:
            return CheckResult(
                name="Memory",
                status=HealthStatus.WARN,
                message=f"{avail_gb:.1f} GB available / {total_gb:.1f} GB total",
                fix_hint="Low memory may cause issues"
            )
        else:
            return CheckResult(
                name="Memory",
                status=HealthStatus.OK,
                message=f"{avail_gb:.1f} GB available / {total_gb:.1f} GB total"
            )
    except Exception as e:
        return CheckResult(
            name="Memory",
            status=HealthStatus.WARN,
            message="Check failed",
            detail=str(e)
        )


# =============================================================================
# Main Scanner
# =============================================================================

def run_full_scan(quick: bool = False) -> SystemHealth:
    """Run full system health scan."""
    import datetime

    health = SystemHealth(
        timestamp=datetime.datetime.now().isoformat(),
        platform=platform.platform()
    )

    # Core checks
    health.checks.append(check_python())
    health.checks.append(check_venv())
    health.checks.append(check_disk_space())
    health.checks.append(check_memory())

    # Build environment
    health.checks.append(check_build_tools())
    health.checks.append(check_rust())

    # GPU / CUDA
    health.checks.append(check_nvidia())
    health.checks.append(check_cuda_toolkit())

    # ML frameworks
    if not quick:
        health.checks.append(check_pytorch())

    # GUI
    health.checks.append(check_gtk())
    health.checks.append(check_webkit())

    # Ara-specific
    health.checks.append(check_hal())
    health.checks.append(check_fpga())

    return health


def print_health_report(health: SystemHealth, show_fixes: bool = False):
    """Print health report to terminal."""
    status_icons = {
        HealthStatus.OK: "✓",
        HealthStatus.WARN: "⚠",
        HealthStatus.FAIL: "✗",
        HealthStatus.SKIP: "○",
    }

    status_colors = {
        HealthStatus.OK: "\033[92m",     # Green
        HealthStatus.WARN: "\033[93m",   # Yellow
        HealthStatus.FAIL: "\033[91m",   # Red
        HealthStatus.SKIP: "\033[90m",   # Gray
    }
    reset = "\033[0m"

    print("\n" + "=" * 60)
    print("  [ARA DOCTOR] System Health Scan")
    print("=" * 60)
    print(f"Platform: {health.platform}")
    print(f"Scan time: {health.timestamp}")
    print("-" * 60)

    for check in health.checks:
        icon = status_icons[check.status]
        color = status_colors[check.status]
        print(f"{color}{icon}{reset} {check.name:20s} {check.message}")

        if check.detail:
            print(f"  └─ {check.detail}")

        if show_fixes and check.fix_hint:
            print(f"  └─ Hint: {check.fix_hint}")
            if check.fix_command:
                print(f"  └─ Fix:  {check.fix_command}")

    print("-" * 60)

    # Summary
    summary_color = "\033[92m" if health.is_healthy else "\033[91m"
    print(f"\nSummary: {summary_color}{health.ok_count} OK{reset}, "
          f"\033[93m{health.warn_count} warnings{reset}, "
          f"\033[91m{health.fail_count} failures{reset}")

    if health.is_healthy:
        print("\n🟢 Ara's organs are healthy!")
    else:
        print("\n🔴 Some organs need attention.")

    print("=" * 60 + "\n")


def print_json_report(health: SystemHealth):
    """Print health report as JSON."""
    data = {
        "timestamp": health.timestamp,
        "platform": health.platform,
        "summary": {
            "ok": health.ok_count,
            "warn": health.warn_count,
            "fail": health.fail_count,
            "healthy": health.is_healthy,
        },
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
                "detail": c.detail,
                "fix_hint": c.fix_hint,
                "fix_command": c.fix_command,
            }
            for c in health.checks
        ]
    }
    print(json.dumps(data, indent=2))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Doctor - System Health Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ara-doctor              Full health scan
    ara-doctor --quick      Skip slow checks
    ara-doctor --json       Output as JSON
    ara-doctor --fix        Show fix suggestions
        """
    )

    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick scan (skip slow checks)")
    parser.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--fix", "-f", action="store_true",
                       help="Show fix suggestions")

    args = parser.parse_args()

    health = run_full_scan(quick=args.quick)

    if args.json:
        print_json_report(health)
    else:
        print_health_report(health, show_fixes=args.fix)

    # Exit code based on health
    sys.exit(0 if health.is_healthy else 1)


if __name__ == "__main__":
    main()
