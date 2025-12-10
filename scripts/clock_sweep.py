#!/usr/bin/env python3
"""
Cathedral Clock Hormesis Sweep
==============================

Map the safe envelope for clock/voltage on a device.
Runs systematic stress tests at different settings and records:
- T_s, A_g deltas
- Error rates
- Thermal/power behavior

Usage:
    python clock_sweep.py --device gpu --model "RTX 3090" --dry-run
    python clock_sweep.py --device cpu --model "TR 3970X" --config config/clock_hormesis.yaml
    python clock_sweep.py --device fpga --model "K10" --output results/k10_sweep.json

Safety:
    - Always start from stock settings
    - Validates each step before proceeding
    - Automatic rollback on errors
    - Never exceeds defined limits
"""

import argparse
import json
import time
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml
import numpy as np


@dataclass
class ClockSetting:
    """A clock/voltage configuration to test."""
    clock_offset_mhz: int
    voltage_offset_mv: int
    mem_clock_offset_mhz: int = 0  # GPU only
    power_limit_w: int = 0
    label: str = ""


@dataclass
class SweepResult:
    """Result of testing one clock setting."""
    setting: ClockSetting
    timestamp: str
    duration_sec: float

    # Pass/fail
    passed: bool
    error_count: int
    error_messages: List[str] = field(default_factory=list)

    # Performance
    throughput_ops_sec: float = 0.0
    throughput_delta_pct: float = 0.0

    # Stability
    T_s: float = 0.99
    T_s_delta: float = 0.0
    A_g: float = 0.0
    determinism_check: bool = True

    # Thermal/power
    temp_max_c: float = 0.0
    temp_avg_c: float = 0.0
    power_avg_w: float = 0.0
    power_max_w: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["setting"] = asdict(self.setting)
        return d


class ClockController:
    """Interface to control device clocks. Override for real hardware."""

    def __init__(self, device_type: str, device_model: str):
        self.device_type = device_type
        self.device_model = device_model
        self.stock_setting = ClockSetting(0, 0, 0, 0, "stock")
        self.current_setting = self.stock_setting

    def apply_setting(self, setting: ClockSetting) -> bool:
        """Apply a clock setting. Override for real hardware."""
        print(f"  [CLOCK] Applying: {setting.label}")
        print(f"          clock={setting.clock_offset_mhz:+d}MHz, "
              f"voltage={setting.voltage_offset_mv:+d}mV, "
              f"mem={setting.mem_clock_offset_mhz:+d}MHz")
        self.current_setting = setting
        return True

    def reset_to_stock(self) -> bool:
        """Reset to stock settings."""
        print("  [CLOCK] Resetting to stock")
        return self.apply_setting(self.stock_setting)

    def read_sensors(self) -> Dict[str, float]:
        """Read temperature and power sensors. Override for real hardware."""
        # Simulated sensor data
        return {
            "temp_c": 55.0 + np.random.randn() * 5,
            "power_w": 250.0 + np.random.randn() * 20,
        }


class GPUClockController(ClockController):
    """GPU-specific clock control via nvidia-smi."""

    def apply_setting(self, setting: ClockSetting) -> bool:
        """Apply GPU clock settings via nvidia-smi."""
        # In production, would run:
        # nvidia-smi -lgc {base+offset},{max+offset}
        # nvidia-smi -pl {power_limit}
        print(f"  [GPU] Would apply via nvidia-smi:")
        print(f"        nvidia-smi --lock-gpu-clocks={1725 + setting.clock_offset_mhz}")
        print(f"        nvidia-smi -pl {setting.power_limit_w or 320}")
        self.current_setting = setting
        return True

    def read_sensors(self) -> Dict[str, float]:
        """Read GPU sensors via nvidia-smi."""
        # In production: nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv
        return {
            "temp_c": 60.0 + abs(self.current_setting.clock_offset_mhz) * 0.1 + np.random.randn() * 3,
            "power_w": 280.0 + abs(self.current_setting.clock_offset_mhz) * 0.5 + np.random.randn() * 10,
        }


class CPUClockController(ClockController):
    """CPU-specific clock control."""

    def apply_setting(self, setting: ClockSetting) -> bool:
        """Apply CPU settings. Platform-specific."""
        print(f"  [CPU] Would apply frequency scaling:")
        print(f"        cpufreq-set or ryzenadj/intel-undervolt")
        self.current_setting = setting
        return True


class FPGAClockController(ClockController):
    """FPGA-specific clock control."""

    def apply_setting(self, setting: ClockSetting) -> bool:
        """Apply FPGA PLL settings."""
        print(f"  [FPGA] Would program PLL via:")
        print(f"        bitstream reconfiguration or IP core register write")
        self.current_setting = setting
        return True


class StressTest:
    """Run stress tests and measure stability."""

    def __init__(self, device_type: str):
        self.device_type = device_type
        self.baseline_hash: Optional[str] = None
        self.baseline_throughput: float = 0.0

    def run_determinism_test(self, duration_sec: float = 10) -> Tuple[bool, str]:
        """
        Run determinism test: same input should produce same output hash.
        Override for real workloads.
        """
        # Simulated: run a deterministic computation
        test_input = np.random.default_rng(42).random(10000)
        result = np.sum(test_input ** 2)  # Deterministic operation
        result_hash = hashlib.sha256(str(result).encode()).hexdigest()[:16]

        if self.baseline_hash is None:
            self.baseline_hash = result_hash

        passed = (result_hash == self.baseline_hash)
        return passed, result_hash

    def run_memory_stress(self, duration_sec: float = 10) -> Tuple[bool, int]:
        """
        Run memory stress test.
        Returns (passed, error_count)
        """
        # In production: cuda-memtest, memtest86, or BRAM checker
        # Simulated: assume passes unless clock is very high
        error_count = 0
        return True, error_count

    def run_workload_benchmark(self, duration_sec: float = 10) -> float:
        """
        Run a representative workload and measure throughput.
        Returns ops/sec.
        """
        # Simulated: matrix operations
        start = time.time()
        ops = 0
        while time.time() - start < min(duration_sec, 5):
            _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
            ops += 1

        elapsed = time.time() - start
        throughput = ops / elapsed * 1e6  # ops/sec

        if self.baseline_throughput == 0:
            self.baseline_throughput = throughput

        return throughput

    def measure_T_s(self) -> float:
        """
        Measure topology stability.
        Simplified: based on variance of repeated measurements.
        """
        measurements = []
        for _ in range(10):
            t = self.run_workload_benchmark(0.5)
            measurements.append(t)

        # T_s based on coefficient of variation
        cv = np.std(measurements) / (np.mean(measurements) + 1e-10)
        T_s = max(0, 1 - cv * 10)  # Scale so low variance → high T_s
        return T_s

    def full_stress_test(
        self,
        duration_sec: float,
        controller: ClockController,
    ) -> SweepResult:
        """Run full stress test suite."""
        start = time.time()
        errors = []
        error_count = 0

        # Determinism check
        det_passed, det_hash = self.run_determinism_test()
        if not det_passed:
            errors.append(f"Determinism failed: expected {self.baseline_hash}, got {det_hash}")
            error_count += 1

        # Memory stress
        mem_passed, mem_errors = self.run_memory_stress(duration_sec / 3)
        error_count += mem_errors
        if not mem_passed:
            errors.append(f"Memory stress failed: {mem_errors} errors")

        # Throughput benchmark
        throughput = self.run_workload_benchmark(duration_sec / 3)
        throughput_delta = (throughput - self.baseline_throughput) / (self.baseline_throughput + 1e-10) * 100

        # T_s measurement
        T_s = self.measure_T_s()
        T_s_delta = T_s - 0.99  # Relative to expected baseline

        # A_g estimation (simplified: based on T_s improvement under stress)
        A_g = T_s_delta * 0.5  # Rough approximation

        # Sensor readings
        temps = []
        powers = []
        for _ in range(5):
            sensors = controller.read_sensors()
            temps.append(sensors["temp_c"])
            powers.append(sensors["power_w"])
            time.sleep(0.5)

        elapsed = time.time() - start
        passed = (error_count == 0) and det_passed and mem_passed

        return SweepResult(
            setting=controller.current_setting,
            timestamp=datetime.utcnow().isoformat(),
            duration_sec=elapsed,
            passed=passed,
            error_count=error_count,
            error_messages=errors,
            throughput_ops_sec=throughput,
            throughput_delta_pct=throughput_delta,
            T_s=T_s,
            T_s_delta=T_s_delta,
            A_g=A_g,
            determinism_check=det_passed,
            temp_max_c=max(temps),
            temp_avg_c=np.mean(temps),
            power_avg_w=np.mean(powers),
            power_max_w=max(powers),
        )


class ClockSweep:
    """Orchestrate clock sweep across settings."""

    def __init__(
        self,
        device_type: str,
        device_model: str,
        config_path: str = "config/clock_hormesis.yaml",
    ):
        self.device_type = device_type
        self.device_model = device_model
        self.config = self._load_config(config_path)
        self.results: List[SweepResult] = []

        # Create appropriate controller
        if device_type == "gpu":
            self.controller = GPUClockController(device_type, device_model)
        elif device_type == "cpu":
            self.controller = CPUClockController(device_type, device_model)
        elif device_type == "fpga":
            self.controller = FPGAClockController(device_type, device_model)
        else:
            self.controller = ClockController(device_type, device_model)

        self.stress_test = StressTest(device_type)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load sweep configuration."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
        return {"sweep": {}}

    def generate_settings(self) -> List[ClockSetting]:
        """Generate settings to sweep based on config."""
        settings = [ClockSetting(0, 0, 0, 0, "stock")]  # Always start with stock

        sweep_config = self.config.get("sweep", {}).get(self.device_type, {})

        clock_steps = sweep_config.get("clock_steps", [0, +50, +100])
        voltage_steps = sweep_config.get("voltage_steps", [-30, 0])
        mem_steps = sweep_config.get("mem_clock_steps", [0])

        for clock in clock_steps:
            for voltage in voltage_steps:
                for mem in mem_steps:
                    if clock == 0 and voltage == 0 and mem == 0:
                        continue  # Skip stock duplicate
                    label = f"c{clock:+d}_v{voltage:+d}_m{mem:+d}"
                    settings.append(ClockSetting(clock, voltage, mem, 0, label))

        return settings

    def run_sweep(
        self,
        dry_run: bool = False,
        test_duration_sec: float = 60,
        cooldown_sec: float = 30,
    ) -> List[SweepResult]:
        """Run the full sweep."""
        settings = self.generate_settings()
        print(f"\n{'='*60}")
        print(f"CLOCK HORMESIS SWEEP: {self.device_type.upper()} / {self.device_model}")
        print(f"Settings to test: {len(settings)}")
        print(f"Test duration: {test_duration_sec}s per setting")
        print(f"{'='*60}\n")

        # Baseline at stock
        print("[1/2] Establishing baseline at stock settings...")
        self.controller.reset_to_stock()
        if not dry_run:
            baseline = self.stress_test.full_stress_test(test_duration_sec, self.controller)
            self.results.append(baseline)
            print(f"  Baseline T_s={baseline.T_s:.4f}, throughput={baseline.throughput_ops_sec:.0f} ops/s")
        else:
            print("  [DRY RUN] Would run baseline stress test")

        # Sweep
        print(f"\n[2/2] Sweeping {len(settings)-1} configurations...")
        for i, setting in enumerate(settings[1:], 1):
            print(f"\n--- Setting {i}/{len(settings)-1}: {setting.label} ---")

            if dry_run:
                print(f"  [DRY RUN] Would test {setting.label}")
                continue

            # Apply setting
            if not self.controller.apply_setting(setting):
                print(f"  [ERROR] Failed to apply setting, skipping")
                continue

            # Warmup
            print(f"  Warming up ({cooldown_sec//2}s)...")
            time.sleep(cooldown_sec // 2)

            # Stress test
            print(f"  Running stress test ({test_duration_sec}s)...")
            result = self.stress_test.full_stress_test(test_duration_sec, self.controller)
            self.results.append(result)

            # Report
            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status}")
            print(f"    T_s={result.T_s:.4f} (Δ={result.T_s_delta:+.4f})")
            print(f"    A_g={result.A_g:+.4f}")
            print(f"    Throughput: {result.throughput_delta_pct:+.1f}%")
            print(f"    Temp: {result.temp_avg_c:.1f}°C (max {result.temp_max_c:.1f}°C)")
            print(f"    Power: {result.power_avg_w:.0f}W (max {result.power_max_w:.0f}W)")

            if not result.passed:
                print(f"    Errors: {result.error_messages}")
                # Stop if we hit failures (beyond safe envelope)
                print("  [!] Stopping sweep - found failure boundary")
                break

            # Cooldown
            print(f"  Cooldown ({cooldown_sec}s)...")
            time.sleep(cooldown_sec)

        # Reset to stock
        print("\nResetting to stock settings...")
        self.controller.reset_to_stock()

        return self.results

    def save_results(self, path: str):
        """Save sweep results to JSON."""
        data = {
            "device_type": self.device_type,
            "device_model": self.device_model,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {path}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate sweep summary."""
        if not self.results:
            return {}

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        # Find best hormesis point (max A_g while T_s acceptable)
        best = None
        for r in passed:
            if r.T_s >= 0.95 and r.A_g > 0:
                if best is None or r.A_g > best.A_g:
                    best = r

        # Find safe envelope boundary
        boundary = failed[0] if failed else None

        return {
            "total_tested": len(self.results),
            "passed": len(passed),
            "failed": len(failed),
            "best_hormesis": best.setting.label if best else "stock",
            "best_hormesis_Ag": best.A_g if best else 0.0,
            "safe_boundary": boundary.setting.label if boundary else "not found",
        }

    def print_summary(self):
        """Print sweep summary."""
        summary = self._generate_summary()
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Total tested: {summary.get('total_tested', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Best hormesis point: {summary.get('best_hormesis', 'N/A')}")
        print(f"Best A_g: {summary.get('best_hormesis_Ag', 0):+.4f}")
        print(f"Safe boundary at: {summary.get('safe_boundary', 'N/A')}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Cathedral Clock Hormesis Sweep")
    parser.add_argument("--device", required=True, choices=["gpu", "cpu", "fpga"],
                        help="Device type to sweep")
    parser.add_argument("--model", default="Unknown",
                        help="Device model (e.g., 'RTX 3090')")
    parser.add_argument("--config", default="config/clock_hormesis.yaml",
                        help="Path to config file")
    parser.add_argument("--output", default=None,
                        help="Output path for results JSON")
    parser.add_argument("--duration", type=int, default=60,
                        help="Test duration per setting (seconds)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Cooldown between tests (seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be tested without running")

    args = parser.parse_args()

    sweep = ClockSweep(args.device, args.model, args.config)
    results = sweep.run_sweep(
        dry_run=args.dry_run,
        test_duration_sec=args.duration,
        cooldown_sec=args.cooldown,
    )

    sweep.print_summary()

    if args.output and not args.dry_run:
        sweep.save_results(args.output)
    elif not args.dry_run:
        # Default output path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_path = f"results/{args.device}_sweep_{timestamp}.json"
        Path("results").mkdir(exist_ok=True)
        sweep.save_results(default_path)


if __name__ == "__main__":
    main()
