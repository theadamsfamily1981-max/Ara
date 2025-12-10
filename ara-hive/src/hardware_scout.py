#!/usr/bin/env python3
"""
Hardware Scout - Discovers and catalogs available hardware.

Scans local system for GPUs, FPGAs, and other compute devices,
then registers them in hardware_inventory table.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=127.0.0.1"
    python src/hardware_scout.py --once        # Single scan
    python src/hardware_scout.py --daemon      # Continuous monitoring
"""

import os
import time
import json
import socket
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: str          # 'gpu', 'fpga', 'tpu', 'npu', 'cpu'
    device_name: str          # 'RTX 4090', 'Alveo U250'
    vendor: str               # 'nvidia', 'amd', 'intel', 'xilinx'
    vram_gb: Optional[float] = None
    compute_units: Optional[int] = None
    clock_mhz: Optional[int] = None
    tdp_watts: Optional[int] = None
    driver_version: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)

    # Current state
    utilization_pct: float = 0.0
    temp_celsius: Optional[float] = None
    power_watts: Optional[float] = None


class HardwareScout:
    """
    Scans system for compute devices and registers them with the hive.
    """

    def __init__(self):
        self.hostname = socket.gethostname()
        self.node_id: Optional[int] = None
        self.devices: List[DeviceInfo] = []

    def _get_node_id(self) -> Optional[int]:
        """Get node_id from database."""
        if not HAS_PSYCOPG2:
            return None

        try:
            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = True
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT id FROM nodes WHERE hostname=%s", (self.hostname,))
            row = cur.fetchone()
            conn.close()
            return row["id"] if row else None
        except Exception as e:
            print(f"[scout] DB error: {e}")
            return None

    def scan_nvidia_gpus(self) -> List[DeviceInfo]:
        """Scan for NVIDIA GPUs using nvidia-smi."""
        devices = []

        try:
            # Query nvidia-smi for GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit,temperature.gpu,power.draw,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return devices

            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 8:
                    continue

                name, vram_mb, driver, clock, tdp, temp, power, util = parts

                try:
                    vram_gb = float(vram_mb) / 1024 if vram_mb != '[N/A]' else None
                    clock_mhz = int(clock) if clock != '[N/A]' else None
                    tdp_watts = int(float(tdp)) if tdp != '[N/A]' else None
                    temp_c = float(temp) if temp != '[N/A]' else None
                    power_w = float(power) if power != '[N/A]' else None
                    util_pct = float(util) if util != '[N/A]' else 0.0
                except (ValueError, TypeError):
                    vram_gb = clock_mhz = tdp_watts = temp_c = power_w = None
                    util_pct = 0.0

                # Estimate CUDA cores from name
                cuda_cores = self._estimate_cuda_cores(name)

                device = DeviceInfo(
                    device_type='gpu',
                    device_name=name,
                    vendor='nvidia',
                    vram_gb=vram_gb,
                    compute_units=cuda_cores,
                    clock_mhz=clock_mhz,
                    tdp_watts=tdp_watts,
                    driver_version=driver,
                    utilization_pct=util_pct,
                    temp_celsius=temp_c,
                    power_watts=power_w,
                    capabilities={
                        'cuda': True,
                        'tensor_cores': 'RTX' in name or 'A100' in name or 'H100' in name,
                    }
                )
                devices.append(device)

        except FileNotFoundError:
            # nvidia-smi not available
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"[scout] NVIDIA scan error: {e}")

        return devices

    def _estimate_cuda_cores(self, name: str) -> Optional[int]:
        """Estimate CUDA cores from GPU name."""
        # Common NVIDIA GPUs
        cuda_map = {
            'RTX 4090': 16384, 'RTX 4080': 9728, 'RTX 4070': 5888,
            'RTX 3090': 10496, 'RTX 3080': 8704, 'RTX 3070': 5888,
            'RTX 3060': 3584, 'RTX 2080': 2944, 'RTX 2070': 2304,
            'A100': 6912, 'A6000': 10752, 'A5000': 8192, 'A4000': 6144,
            'H100': 16896, 'L40': 18176,
            'GTX 1080': 2560, 'GTX 1070': 1920, 'GTX 1060': 1280,
        }

        for pattern, cores in cuda_map.items():
            if pattern in name:
                return cores
        return None

    def scan_amd_gpus(self) -> List[DeviceInfo]:
        """Scan for AMD GPUs using rocm-smi."""
        devices = []

        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse rocm-smi JSON output
                for gpu_id, gpu_info in data.items():
                    if not gpu_id.startswith('card'):
                        continue

                    name = gpu_info.get('Product Name', f'AMD GPU {gpu_id}')
                    vram_bytes = gpu_info.get('VRAM Total Memory (B)', 0)
                    vram_gb = vram_bytes / (1024**3) if vram_bytes else None

                    device = DeviceInfo(
                        device_type='gpu',
                        device_name=name,
                        vendor='amd',
                        vram_gb=vram_gb,
                        capabilities={'rocm': True}
                    )
                    devices.append(device)

        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except Exception as e:
            print(f"[scout] AMD scan error: {e}")

        return devices

    def scan_intel_gpus(self) -> List[DeviceInfo]:
        """Scan for Intel GPUs."""
        devices = []

        try:
            # Check for Intel discrete GPUs via sysfs
            import glob

            for card_dir in glob.glob('/sys/class/drm/card*/device'):
                vendor_path = os.path.join(card_dir, 'vendor')
                if os.path.exists(vendor_path):
                    with open(vendor_path) as f:
                        vendor_id = f.read().strip()

                    # Intel vendor ID: 0x8086
                    if vendor_id == '0x8086':
                        device_path = os.path.join(card_dir, 'device')
                        if os.path.exists(device_path):
                            with open(device_path) as f:
                                device_id = f.read().strip()

                            device = DeviceInfo(
                                device_type='gpu',
                                device_name=f'Intel GPU {device_id}',
                                vendor='intel',
                                capabilities={'oneapi': True}
                            )
                            devices.append(device)

        except Exception as e:
            print(f"[scout] Intel scan error: {e}")

        return devices

    def scan_fpgas(self) -> List[DeviceInfo]:
        """Scan for FPGAs (Xilinx, Intel)."""
        devices = []

        # Check for Xilinx FPGAs via xbutil
        try:
            result = subprocess.run(
                ["xbutil", "examine", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                for device_data in data.get('devices', []):
                    name = device_data.get('name', 'Xilinx FPGA')
                    device = DeviceInfo(
                        device_type='fpga',
                        device_name=name,
                        vendor='xilinx',
                        capabilities={'vitis': True}
                    )
                    devices.append(device)

        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return devices

    def scan_cpus(self) -> List[DeviceInfo]:
        """Scan CPU for compute capabilities."""
        devices = []

        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()

            # Extract model name
            model_name = 'Unknown CPU'
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    model_name = line.split(':')[1].strip()
                    break

            # Count cores
            cores = cpuinfo.count('processor')

            # Get max frequency
            freq_mhz = None
            try:
                with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq') as f:
                    freq_mhz = int(f.read().strip()) // 1000
            except:
                pass

            vendor = 'intel' if 'Intel' in model_name else 'amd' if 'AMD' in model_name else 'other'

            device = DeviceInfo(
                device_type='cpu',
                device_name=model_name,
                vendor=vendor,
                compute_units=cores,
                clock_mhz=freq_mhz,
                capabilities={
                    'avx2': 'avx2' in cpuinfo.lower(),
                    'avx512': 'avx512' in cpuinfo.lower(),
                }
            )
            devices.append(device)

        except Exception as e:
            print(f"[scout] CPU scan error: {e}")

        return devices

    def scan_all(self) -> List[DeviceInfo]:
        """Scan all device types."""
        self.devices = []

        print(f"[scout] Scanning hardware on {self.hostname}...")

        # GPUs
        nvidia_gpus = self.scan_nvidia_gpus()
        if nvidia_gpus:
            print(f"[scout] Found {len(nvidia_gpus)} NVIDIA GPU(s)")
            self.devices.extend(nvidia_gpus)

        amd_gpus = self.scan_amd_gpus()
        if amd_gpus:
            print(f"[scout] Found {len(amd_gpus)} AMD GPU(s)")
            self.devices.extend(amd_gpus)

        intel_gpus = self.scan_intel_gpus()
        if intel_gpus:
            print(f"[scout] Found {len(intel_gpus)} Intel GPU(s)")
            self.devices.extend(intel_gpus)

        # FPGAs
        fpgas = self.scan_fpgas()
        if fpgas:
            print(f"[scout] Found {len(fpgas)} FPGA(s)")
            self.devices.extend(fpgas)

        # CPUs (always include)
        cpus = self.scan_cpus()
        self.devices.extend(cpus)

        print(f"[scout] Total: {len(self.devices)} device(s)")
        return self.devices

    def register_devices(self):
        """Register scanned devices in database."""
        if not HAS_PSYCOPG2:
            print("[scout] psycopg2 not available, skipping DB registration")
            return

        self.node_id = self._get_node_id()
        if not self.node_id:
            print(f"[scout] Node {self.hostname} not registered, skipping")
            return

        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cur = conn.cursor()

        for device in self.devices:
            try:
                cur.execute("""
                    INSERT INTO hardware_inventory (
                        node_id, device_type, device_name, vendor,
                        vram_gb, compute_units, clock_mhz, tdp_watts,
                        driver_version, capabilities,
                        utilization_pct, temp_celsius, power_watts,
                        last_scout
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        now()
                    )
                    ON CONFLICT (node_id, device_type, device_name)
                    DO UPDATE SET
                        vram_gb = EXCLUDED.vram_gb,
                        compute_units = EXCLUDED.compute_units,
                        clock_mhz = EXCLUDED.clock_mhz,
                        tdp_watts = EXCLUDED.tdp_watts,
                        driver_version = EXCLUDED.driver_version,
                        capabilities = EXCLUDED.capabilities,
                        utilization_pct = EXCLUDED.utilization_pct,
                        temp_celsius = EXCLUDED.temp_celsius,
                        power_watts = EXCLUDED.power_watts,
                        last_scout = now()
                """, (
                    self.node_id,
                    device.device_type,
                    device.device_name,
                    device.vendor,
                    device.vram_gb,
                    device.compute_units,
                    device.clock_mhz,
                    device.tdp_watts,
                    device.driver_version,
                    json.dumps(device.capabilities),
                    device.utilization_pct,
                    device.temp_celsius,
                    device.power_watts,
                ))
                print(f"[scout] Registered: {device.device_type} {device.device_name}")

            except Exception as e:
                print(f"[scout] Failed to register {device.device_name}: {e}")

        conn.close()

    def run_daemon(self, interval_s: int = 60):
        """Run continuous monitoring."""
        print(f"[scout] Starting daemon (interval={interval_s}s)")
        while True:
            self.scan_all()
            self.register_devices()
            time.sleep(interval_s)

    def print_summary(self):
        """Print summary of discovered hardware."""
        print("\n" + "=" * 60)
        print("Hardware Scout Summary")
        print("=" * 60)

        for device in self.devices:
            print(f"\n{device.device_type.upper()}: {device.device_name}")
            print(f"  Vendor: {device.vendor}")
            if device.vram_gb:
                print(f"  VRAM: {device.vram_gb:.1f} GB")
            if device.compute_units:
                print(f"  Compute Units: {device.compute_units}")
            if device.clock_mhz:
                print(f"  Clock: {device.clock_mhz} MHz")
            if device.tdp_watts:
                print(f"  TDP: {device.tdp_watts} W")
            if device.utilization_pct:
                print(f"  Utilization: {device.utilization_pct:.1f}%")
            if device.temp_celsius:
                print(f"  Temperature: {device.temp_celsius:.0f}°C")
            if device.power_watts:
                print(f"  Power: {device.power_watts:.1f} W")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hardware Scout")
    parser.add_argument("--once", action="store_true", help="Single scan")
    parser.add_argument("--daemon", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval (seconds)")
    parser.add_argument("--no-register", action="store_true", help="Don't register to DB")
    args = parser.parse_args()

    scout = HardwareScout()

    if args.daemon:
        scout.run_daemon(args.interval)
    else:
        scout.scan_all()
        scout.print_summary()
        if not args.no_register:
            scout.register_devices()


if __name__ == "__main__":
    main()
