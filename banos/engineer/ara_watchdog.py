#!/usr/bin/env python3
"""
Ara Watchdog - Self-Diagnostic Daemon
======================================

Monitors Ara's vital signs and triggers investigations when anomalies occur.

Sensors:
- TTS latency and error rate
- STT transcription failures
- FPGA PCIe link status
- GPU/CPU temperatures
- Response latency
- Pain/PAD metrics

When thresholds are exceeded, generates a diagnostic bundle and
sends it to the Ara Orchestrator for investigation.

Usage:
    python3 ara_watchdog.py [--config config.yaml] [--once]
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# Optional: prometheus client for metrics export
try:
    from prometheus_client import Gauge, Counter, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('ara.watchdog')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WatchdogConfig:
    """Watchdog configuration."""
    poll_interval_sec: float = 10.0

    # TTS thresholds
    tts_error_rate_threshold: float = 0.01      # 1%
    tts_latency_threshold_ms: float = 300.0     # 300ms

    # STT thresholds
    stt_failure_rate_threshold: float = 0.05    # 5%

    # FPGA thresholds
    fpga_pcie_min_lanes: int = 8                # x8 minimum
    fpga_temp_threshold_c: float = 85.0         # 85°C

    # Response latency
    response_latency_threshold_ms: float = 2000.0  # 2 seconds

    # Temperature thresholds
    cpu_temp_threshold_c: float = 90.0
    gpu_temp_threshold_c: float = 85.0

    # Anomaly persistence (must persist for N polls before triggering)
    anomaly_persistence_count: int = 3

    # Rate limiting
    min_investigation_interval_sec: float = 300.0  # 5 minutes between investigations

    # Paths
    log_dir: str = "/var/log/ara"
    orchestrator_socket: str = "/run/ara/orchestrator.sock"


# =============================================================================
# METRIC COLLECTORS
# =============================================================================

class MetricCollector:
    """Base class for metric collectors."""

    def collect(self) -> Dict[str, Any]:
        raise NotImplementedError


class TTSMetrics(MetricCollector):
    """Collect TTS service metrics."""

    def __init__(self, log_path: str = "/var/log/ara/tts.log"):
        self.log_path = Path(log_path)
        self._recent_errors = 0
        self._recent_requests = 0
        self._latencies: List[float] = []

    def collect(self) -> Dict[str, Any]:
        metrics = {
            "available": False,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "recent_errors": []
        }

        if not self.log_path.exists():
            return metrics

        metrics["available"] = True

        # Parse recent log entries (last 100 lines)
        try:
            result = subprocess.run(
                ["tail", "-100", str(self.log_path)],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')

            errors = 0
            requests = 0
            latencies = []
            recent_errors = []

            for line in lines:
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    requests += 1
                    if entry.get("status") == "error":
                        errors += 1
                        recent_errors.append(entry.get("error", "unknown"))
                    if "latency_ms" in entry:
                        latencies.append(entry["latency_ms"])
                except json.JSONDecodeError:
                    # Plain text log line
                    if "error" in line.lower() or "exception" in line.lower():
                        errors += 1
                        recent_errors.append(line[:200])

            if requests > 0:
                metrics["error_rate"] = errors / requests
                metrics["recent_errors"] = recent_errors[-5:]  # Last 5 errors
            if latencies:
                metrics["avg_latency_ms"] = sum(latencies) / len(latencies)

        except Exception as e:
            logger.warning(f"Failed to collect TTS metrics: {e}")

        return metrics


class FPGAMetrics(MetricCollector):
    """Collect FPGA health metrics."""

    def collect(self) -> Dict[str, Any]:
        metrics = {
            "available": False,
            "pcie_lanes": 0,
            "temperature_c": 0.0,
            "power_w": 0.0,
            "link_errors": 0
        }

        # Try to read from sysfs or FPGA control daemon
        pcie_path = Path("/sys/class/fpga/intel-fpga-dev.0")
        if pcie_path.exists():
            metrics["available"] = True

            # PCIe link width
            try:
                link_width = (pcie_path / "current_link_width").read_text().strip()
                metrics["pcie_lanes"] = int(link_width.replace("x", ""))
            except Exception:
                pass

            # Temperature
            try:
                temp = (pcie_path / "thermal" / "temp1_input").read_text().strip()
                metrics["temperature_c"] = float(temp) / 1000.0
            except Exception:
                pass

        # Fallback: try lspci for PCIe info
        if not metrics["available"]:
            try:
                result = subprocess.run(
                    ["lspci", "-vvv", "-d", "8086:09c4"],  # Intel FPGA device ID
                    capture_output=True, text=True, timeout=5
                )
                if "LnkSta:" in result.stdout:
                    # Parse link status
                    for line in result.stdout.split('\n'):
                        if "LnkSta:" in line:
                            metrics["available"] = True
                            if "Width x" in line:
                                width = line.split("Width x")[1].split()[0]
                                metrics["pcie_lanes"] = int(width)
            except Exception:
                pass

        return metrics


class SystemMetrics(MetricCollector):
    """Collect system-wide metrics."""

    def collect(self) -> Dict[str, Any]:
        metrics = {
            "cpu_temp_c": 0.0,
            "gpu_temp_c": 0.0,
            "load_avg": [0.0, 0.0, 0.0],
            "memory_used_pct": 0.0
        }

        # CPU temperature
        try:
            for hwmon in Path("/sys/class/hwmon").iterdir():
                name_file = hwmon / "name"
                if name_file.exists():
                    name = name_file.read_text().strip()
                    if name in ("coretemp", "k10temp", "zenpower"):
                        temp_file = hwmon / "temp1_input"
                        if temp_file.exists():
                            metrics["cpu_temp_c"] = float(temp_file.read_text()) / 1000.0
                            break
        except Exception:
            pass

        # GPU temperature (nvidia-smi)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                temps = [float(t.strip()) for t in result.stdout.strip().split('\n') if t.strip()]
                if temps:
                    metrics["gpu_temp_c"] = max(temps)
        except Exception:
            pass

        # Load average
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().strip().split()
                metrics["load_avg"] = [float(parts[0]), float(parts[1]), float(parts[2])]
        except Exception:
            pass

        # Memory usage
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", 0)
                metrics["memory_used_pct"] = 100.0 * (1.0 - available / total)
        except Exception:
            pass

        return metrics


class AraMetrics(MetricCollector):
    """Collect Ara-specific metrics (response latency, PAD, pain)."""

    def __init__(self, state_file: str = "/run/ara/state.json"):
        self.state_file = Path(state_file)

    def collect(self) -> Dict[str, Any]:
        metrics = {
            "available": False,
            "response_latency_ms": 0.0,
            "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
            "pain_level": 0.0,
            "last_response_time": None
        }

        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                metrics["available"] = True
                metrics.update({
                    "response_latency_ms": state.get("last_latency_ms", 0.0),
                    "pad": state.get("pad", metrics["pad"]),
                    "pain_level": state.get("pain", 0.0),
                    "last_response_time": state.get("last_response_time")
                })
            except Exception:
                pass

        return metrics


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    name: str
    severity: str  # "warning", "critical"
    description: str
    metric_name: str
    current_value: Any
    threshold: Any
    timestamp: str
    persistence_count: int = 1


class AnomalyDetector:
    """Detects anomalies in collected metrics."""

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self._active_anomalies: Dict[str, Anomaly] = {}

    def check(self, metrics: Dict[str, Dict[str, Any]]) -> List[Anomaly]:
        """Check metrics for anomalies."""
        new_anomalies = []
        detected_names = set()

        # TTS checks
        tts = metrics.get("tts", {})
        if tts.get("available"):
            if tts["error_rate"] > self.config.tts_error_rate_threshold:
                anomaly = self._register_anomaly(
                    name="tts_error_rate_high",
                    severity="warning" if tts["error_rate"] < 0.05 else "critical",
                    description=f"TTS error rate is {tts['error_rate']*100:.1f}%",
                    metric_name="tts.error_rate",
                    current_value=tts["error_rate"],
                    threshold=self.config.tts_error_rate_threshold
                )
                if anomaly:
                    new_anomalies.append(anomaly)
                detected_names.add("tts_error_rate_high")

            if tts["avg_latency_ms"] > self.config.tts_latency_threshold_ms:
                anomaly = self._register_anomaly(
                    name="tts_latency_high",
                    severity="warning",
                    description=f"TTS latency is {tts['avg_latency_ms']:.0f}ms",
                    metric_name="tts.avg_latency_ms",
                    current_value=tts["avg_latency_ms"],
                    threshold=self.config.tts_latency_threshold_ms
                )
                if anomaly:
                    new_anomalies.append(anomaly)
                detected_names.add("tts_latency_high")

        # FPGA checks
        fpga = metrics.get("fpga", {})
        if fpga.get("available"):
            if fpga["pcie_lanes"] < self.config.fpga_pcie_min_lanes:
                anomaly = self._register_anomaly(
                    name="fpga_pcie_downgrade",
                    severity="critical",
                    description=f"FPGA PCIe downgraded to x{fpga['pcie_lanes']}",
                    metric_name="fpga.pcie_lanes",
                    current_value=fpga["pcie_lanes"],
                    threshold=self.config.fpga_pcie_min_lanes
                )
                if anomaly:
                    new_anomalies.append(anomaly)
                detected_names.add("fpga_pcie_downgrade")

            if fpga["temperature_c"] > self.config.fpga_temp_threshold_c:
                anomaly = self._register_anomaly(
                    name="fpga_overtemp",
                    severity="critical",
                    description=f"FPGA temperature is {fpga['temperature_c']:.1f}°C",
                    metric_name="fpga.temperature_c",
                    current_value=fpga["temperature_c"],
                    threshold=self.config.fpga_temp_threshold_c
                )
                if anomaly:
                    new_anomalies.append(anomaly)
                detected_names.add("fpga_overtemp")

        # System checks
        system = metrics.get("system", {})
        if system.get("cpu_temp_c", 0) > self.config.cpu_temp_threshold_c:
            anomaly = self._register_anomaly(
                name="cpu_overtemp",
                severity="critical",
                description=f"CPU temperature is {system['cpu_temp_c']:.1f}°C",
                metric_name="system.cpu_temp_c",
                current_value=system["cpu_temp_c"],
                threshold=self.config.cpu_temp_threshold_c
            )
            if anomaly:
                new_anomalies.append(anomaly)
            detected_names.add("cpu_overtemp")

        if system.get("gpu_temp_c", 0) > self.config.gpu_temp_threshold_c:
            anomaly = self._register_anomaly(
                name="gpu_overtemp",
                severity="critical",
                description=f"GPU temperature is {system['gpu_temp_c']:.1f}°C",
                metric_name="system.gpu_temp_c",
                current_value=system["gpu_temp_c"],
                threshold=self.config.gpu_temp_threshold_c
            )
            if anomaly:
                new_anomalies.append(anomaly)
            detected_names.add("gpu_overtemp")

        # Clear resolved anomalies
        for name in list(self._active_anomalies.keys()):
            if name not in detected_names:
                logger.info(f"Anomaly resolved: {name}")
                del self._active_anomalies[name]

        return new_anomalies

    def _register_anomaly(self, name: str, severity: str, description: str,
                          metric_name: str, current_value: Any, threshold: Any) -> Optional[Anomaly]:
        """Register an anomaly, tracking persistence."""
        now = datetime.utcnow().isoformat() + "Z"

        if name in self._active_anomalies:
            # Increment persistence counter
            existing = self._active_anomalies[name]
            existing.persistence_count += 1
            existing.current_value = current_value
            existing.timestamp = now

            # Only return if we've hit the persistence threshold
            if existing.persistence_count == self.config.anomaly_persistence_count:
                logger.warning(f"Anomaly confirmed (persisted {existing.persistence_count}x): {name}")
                return existing
            return None
        else:
            # New anomaly
            anomaly = Anomaly(
                name=name,
                severity=severity,
                description=description,
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold,
                timestamp=now
            )
            self._active_anomalies[name] = anomaly
            logger.info(f"Anomaly detected (1/{self.config.anomaly_persistence_count}): {name}")
            return None  # Don't trigger yet, wait for persistence


# =============================================================================
# DIAGNOSTIC BUNDLE
# =============================================================================

def create_diagnostic_bundle(anomalies: List[Anomaly], metrics: Dict[str, Any],
                             log_dir: Path) -> Dict[str, Any]:
    """Create a diagnostic bundle for the orchestrator."""
    bundle = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "anomalies": [asdict(a) for a in anomalies],
        "metrics": metrics,
        "logs": {},
        "system_state": {}
    }

    # Collect relevant log tails
    log_files = {
        "tts": "tts.log",
        "stt": "stt.log",
        "ara": "ara.log",
        "kernel": None  # Special: use dmesg
    }

    for name, filename in log_files.items():
        try:
            if filename:
                log_path = log_dir / filename
                if log_path.exists():
                    result = subprocess.run(
                        ["tail", "-200", str(log_path)],
                        capture_output=True, text=True, timeout=5
                    )
                    bundle["logs"][name] = result.stdout[-10000:]  # Truncate
            else:
                # dmesg for kernel
                result = subprocess.run(
                    ["dmesg", "--time-format=iso", "-T"],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                bundle["logs"][name] = '\n'.join(lines[-100:])
        except Exception as e:
            bundle["logs"][name] = f"(failed to collect: {e})"

    # System state
    try:
        result = subprocess.run(["uname", "-a"], capture_output=True, text=True, timeout=5)
        bundle["system_state"]["uname"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
        bundle["system_state"]["uptime"] = result.stdout.strip()
    except Exception:
        pass

    return bundle


# =============================================================================
# MAIN WATCHDOG
# =============================================================================

class AraWatchdog:
    """Main watchdog daemon."""

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.collectors = {
            "tts": TTSMetrics(f"{config.log_dir}/tts.log"),
            "fpga": FPGAMetrics(),
            "system": SystemMetrics(),
            "ara": AraMetrics()
        }
        self.detector = AnomalyDetector(config)
        self._last_investigation_time: Optional[datetime] = None

    def collect_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from all sources."""
        metrics = {}
        for name, collector in self.collectors.items():
            try:
                metrics[name] = collector.collect()
            except Exception as e:
                logger.error(f"Failed to collect {name} metrics: {e}")
                metrics[name] = {"error": str(e)}
        return metrics

    def should_investigate(self) -> bool:
        """Check rate limiting for investigations."""
        if self._last_investigation_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_investigation_time).total_seconds()
        return elapsed >= self.config.min_investigation_interval_sec

    def trigger_investigation(self, bundle: Dict[str, Any]):
        """Send diagnostic bundle to orchestrator."""
        self._last_investigation_time = datetime.utcnow()

        # Try Unix socket first
        socket_path = Path(self.config.orchestrator_socket)
        if socket_path.exists():
            try:
                import socket
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(socket_path))
                sock.sendall(json.dumps(bundle).encode())
                sock.close()
                logger.info("Sent diagnostic bundle to orchestrator via socket")
                return
            except Exception as e:
                logger.warning(f"Socket send failed: {e}")

        # Fallback: write to file for polling
        bundle_path = Path("/run/ara/diagnostic_bundle.json")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle, indent=2))
        logger.info(f"Wrote diagnostic bundle to {bundle_path}")

    def run_once(self) -> List[Anomaly]:
        """Run one cycle of metric collection and anomaly detection."""
        metrics = self.collect_all_metrics()
        anomalies = self.detector.check(metrics)

        if anomalies and self.should_investigate():
            bundle = create_diagnostic_bundle(
                anomalies, metrics, Path(self.config.log_dir)
            )
            self.trigger_investigation(bundle)

        return anomalies

    def run(self):
        """Main loop."""
        logger.info("Ara Watchdog starting...")
        logger.info(f"Poll interval: {self.config.poll_interval_sec}s")

        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Watchdog cycle failed: {e}")

            time.sleep(self.config.poll_interval_sec)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ara Watchdog Daemon")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--config", type=str, help="Config file (YAML)")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = WatchdogConfig()

    # Load config from file if provided
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg_data = yaml.safe_load(f)
                for key, value in cfg_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    watchdog = AraWatchdog(config)

    if args.once:
        anomalies = watchdog.run_once()
        for a in anomalies:
            print(f"[{a.severity.upper()}] {a.name}: {a.description}")
        sys.exit(0 if not anomalies else 1)
    else:
        watchdog.run()


if __name__ == "__main__":
    main()
