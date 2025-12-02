#!/usr/bin/env python3
"""
Hardware Bring-Up Readiness Validator

Run this script before Phase 2 hardware bring-up to ensure all
software components are validated and ready.

Usage:
    python scripts/validate_hardware_ready.py
    python scripts/validate_hardware_ready.py --export-hls build/hls
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, name: str, passed: bool, message: str, details: dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.message}"


def check_module_imports() -> ValidationResult:
    """Validate all required modules can be imported."""
    try:
        # Core modules
        from ara.cxl_control import (
            ControlPlane, FPGAEmulator, HLSExporter,
            ControlState, CXLMemoryManager,
            hls_lif_step, hls_l1_homeostat, hls_l3_metacontrol,
            export_hls_kernel
        )

        from ara.metacontrol import (
            L3MetacontrolService, PADState, ControlModulation,
            WorkspaceMode, get_metacontrol_service,
            update_antifragility_status
        )

        from tfan.system import (
            SemanticSystemOptimizer,
            CLVComputer, CognitiveLoadVector, RiskLevel,
            AtomicStructuralUpdater
        )

        return ValidationResult(
            "Module Imports",
            True,
            "All 3 core modules imported successfully",
            {"modules": ["ara.cxl_control", "ara.metacontrol", "tfan.system"]}
        )
    except ImportError as e:
        return ValidationResult(
            "Module Imports",
            False,
            f"Import failed: {e}"
        )


def check_hls_export() -> ValidationResult:
    """Validate HLS export generates all required files."""
    try:
        from ara.cxl_control import HLSExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = HLSExporter(target_device="xcu250-figd2104-2L-e")
            files = exporter.export_all(tmpdir)

            # Validate files
            required_files = [
                "pgu_cache_kernel.cpp",
                "pgu_cache_kernel.h",
                "pgu_cache_tb.cpp",
                "run_hls.tcl"
            ]

            missing = [f for f in required_files if f not in files]
            if missing:
                return ValidationResult(
                    "HLS Export",
                    False,
                    f"Missing files: {missing}"
                )

            # Validate content
            cpp_content = files["pgu_cache_kernel.cpp"]
            checks = {
                "HLS pragmas": "#pragma HLS" in cpp_content,
                "Kernel function": "pgu_cache_kernel" in cpp_content,
                "LIF step": "lif_step" in cpp_content,
                "L1 homeostat": "l1_homeostat" in cpp_content,
            }

            failed_checks = [k for k, v in checks.items() if not v]
            if failed_checks:
                return ValidationResult(
                    "HLS Export",
                    False,
                    f"Content validation failed: {failed_checks}"
                )

            file_sizes = {f: len(content) for f, content in files.items()}

            return ValidationResult(
                "HLS Export",
                True,
                f"Generated {len(files)} files with valid content",
                {"files": file_sizes}
            )

    except Exception as e:
        return ValidationResult(
            "HLS Export",
            False,
            f"Export failed: {e}"
        )


def check_control_plane() -> ValidationResult:
    """Validate control plane functionality."""
    try:
        from ara.cxl_control import ControlPlane

        plane = ControlPlane()
        result = plane.fast_control_step(
            valence=-0.3,
            arousal=0.8,
            dominance=0.5
        )

        # Validate outputs
        required_keys = ["temperature_mult", "memory_mult", "attention_gain", "latency_us"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            return ValidationResult(
                "Control Plane",
                False,
                f"Missing output keys: {missing}"
            )

        # Validate ranges
        if not (0.5 <= result["temperature_mult"] <= 1.5):
            return ValidationResult(
                "Control Plane",
                False,
                f"Temperature mult out of range: {result['temperature_mult']}"
            )

        return ValidationResult(
            "Control Plane",
            True,
            f"Latency: {result['latency_us']:.1f}μs",
            {"result": result}
        )

    except Exception as e:
        return ValidationResult(
            "Control Plane",
            False,
            f"Test failed: {e}"
        )


def check_hls_functions() -> ValidationResult:
    """Validate HLS-compatible functions."""
    try:
        from ara.cxl_control import hls_lif_step, hls_l1_homeostat, hls_l3_metacontrol

        # Test LIF step
        v, spike = hls_lif_step(0.5, 1.0, 10.0, 1.0)
        if not isinstance(v, float) or spike not in (0, 1):
            return ValidationResult("HLS Functions", False, "LIF step returned invalid types")

        # Test L1 homeostat
        val_c, aro_c, integral = hls_l1_homeostat(0.0, 0.5, 0.3, 0.6, 0.0)
        if not all(isinstance(x, float) for x in [val_c, aro_c, integral]):
            return ValidationResult("HLS Functions", False, "L1 homeostat returned invalid types")

        # Test L3 metacontrol
        temp, mem, attn = hls_l3_metacontrol(-0.3, 0.8, 0.5)
        if not all(isinstance(x, float) for x in [temp, mem, attn]):
            return ValidationResult("HLS Functions", False, "L3 metacontrol returned invalid types")

        return ValidationResult(
            "HLS Functions",
            True,
            "All 3 HLS functions validated",
            {"lif_spike": spike, "temp_mult": temp, "mem_mult": mem}
        )

    except Exception as e:
        return ValidationResult(
            "HLS Functions",
            False,
            f"Test failed: {e}"
        )


def check_metacontrol() -> ValidationResult:
    """Validate metacontrol service with antifragility metrics."""
    try:
        from ara.metacontrol import get_metacontrol_service

        service = get_metacontrol_service()

        # Update antifragility metrics
        service.update_antifragility_metrics(
            antifragility_score=2.21,
            delta_p99_ms=17.78,
            risk_level="LOW",
            clv_instability=0.12,
            clv_resource=0.08
        )

        # Get status
        status = service.get_status()

        # Validate antifragility fields
        required = ["antifragility_score", "last_delta_p99_ms", "risk_level",
                   "clv_instability", "clv_resource"]
        missing = [k for k in required if k not in status]
        if missing:
            return ValidationResult(
                "Metacontrol",
                False,
                f"Missing status fields: {missing}"
            )

        return ValidationResult(
            "Metacontrol",
            True,
            f"AF Score: {status['antifragility_score']}, Risk: {status['risk_level']}",
            {"status": status}
        )

    except Exception as e:
        return ValidationResult(
            "Metacontrol",
            False,
            f"Test failed: {e}"
        )


def check_required_files() -> ValidationResult:
    """Validate all required files exist."""
    required = [
        "ara/cxl_control/__init__.py",
        "ara/metacontrol/__init__.py",
        "tfan/system/__init__.py",
        "tfan/system/semantic_optimizer.py",
        "tfan/system/cognitive_load_vector.py",
        "tfan/system/atomic_updater.py",
        "scripts/certify_antifragility_delta.py",
        "scripts/demo_closed_loop_antifragility.py",
        ".github/workflows/a_cert.yml",
        "docs/PHASES.md",
        "docs/ANTIFRAGILITY_CERTIFICATION.md",
    ]

    missing = []
    for f in required:
        if not (ROOT / f).exists():
            missing.append(f)

    if missing:
        return ValidationResult(
            "Required Files",
            False,
            f"Missing {len(missing)} files",
            {"missing": missing}
        )

    return ValidationResult(
        "Required Files",
        True,
        f"All {len(required)} required files present",
        {"count": len(required)}
    )


def check_certification_results() -> ValidationResult:
    """Validate certification results exist."""
    results_path = ROOT / "results" / "phase1" / "certification.json"

    if not results_path.exists():
        return ValidationResult(
            "Certification Results",
            False,
            "Phase 1 results not found"
        )

    try:
        import json
        with open(results_path) as f:
            results = json.load(f)

        af_score = results.get("antifragility_score", 0)
        delta_p99 = results.get("delta_p99_burst_ms", 0)
        passed = results.get("certification_passed", False)

        if not passed:
            return ValidationResult(
                "Certification Results",
                False,
                f"Certification did not pass (AF: {af_score:.2f})"
            )

        return ValidationResult(
            "Certification Results",
            True,
            f"AF Score: {af_score:.2f}×, Δp99: +{delta_p99:.2f}ms",
            {"results": results}
        )

    except Exception as e:
        return ValidationResult(
            "Certification Results",
            False,
            f"Failed to read results: {e}"
        )


def run_all_validations() -> list:
    """Run all validation checks."""
    checks = [
        check_module_imports,
        check_hls_export,
        check_control_plane,
        check_hls_functions,
        check_metacontrol,
        check_required_files,
        check_certification_results,
    ]

    results = []
    for check in checks:
        result = check()
        results.append(result)

    return results


def export_hls_files(output_dir: str):
    """Export HLS files to specified directory."""
    from ara.cxl_control import export_hls_kernel

    print(f"\nExporting HLS files to {output_dir}...")
    files = export_hls_kernel(output_dir)

    print("Generated files:")
    for fname, content in files.items():
        fpath = os.path.join(output_dir, fname)
        print(f"  - {fname}: {os.path.getsize(fpath)} bytes")

    print(f"\n✓ HLS files exported to {output_dir}")
    print("\nNext steps:")
    print(f"  cd {output_dir}")
    print("  vitis_hls -f run_hls.tcl")


def main():
    parser = argparse.ArgumentParser(description="Validate hardware bring-up readiness")
    parser.add_argument("--export-hls", metavar="DIR", help="Export HLS files to directory")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  TF-A-N / Ara Hardware Bring-Up Readiness Validator")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    results = run_all_validations()

    # Print results
    passed = 0
    failed = 0

    for result in results:
        print(result)
        if result.passed:
            passed += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print(f"  Summary: {passed}/{len(results)} checks passed")
    print("=" * 60)

    if failed == 0:
        print()
        print("  ✅ SYSTEM READY FOR HARDWARE BRING-UP")
        print()
        print("  Next steps:")
        print("    1. Export HLS: python scripts/validate_hardware_ready.py --export-hls build/hls")
        print("    2. Run synthesis: cd build/hls && vitis_hls -f run_hls.tcl")
        print("    3. Follow docs/HARDWARE_BRINGUP_CHECKLIST.md")
        print()

        if args.export_hls:
            export_hls_files(args.export_hls)

        return 0
    else:
        print()
        print("  ❌ VALIDATION FAILED - Fix issues before hardware bring-up")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
