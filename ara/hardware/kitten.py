"""
Forest Kitten 33 - Neuromorphic SNN Fabric

The "Kitten" is an SNN (Spiking Neural Network) accelerator that provides
neuromorphic computation for Ara's cognitive architecture.

Architecture (from ARASYNERGY_KITTEN_FPGA_SPEC.md):
- 4 populations: input → hidden1 → hidden2 → output
- 5 projections (including recurrent connections)
- Discrete-time LIF (Leaky Integrate-and-Fire) neurons
- CSR sparse weight matrices
- Real hardware: Forest Kitten 33 FPGA @ 250MHz, 65536 neurons
- Emulation: Software LIF simulation on CPU/GPU

This module provides:
- KittenEmulator: Software simulation of the FK33
- ForestKitten33: Hardware interface (real or emulated)
- Integration with Ara's cognitive systems

Usage:
    from ara.hardware.kitten import create_kitten

    # Creates emulator if no hardware present
    kitten = create_kitten(mode="auto")

    # Process spikes
    output = kitten.step(input_currents)
    print(f"Spike rate: {kitten.spike_rate:.2%}")
"""

import logging
import time
import sys
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("ara.hardware.kitten")

# Try to import MCP A10PED hardware interface
A10PED_AVAILABLE = False
AITile = None
CSROffset = None

try:
    # Add MCP tools to path
    mcp_a10ped_path = Path(__file__).parent.parent.parent / "tfan" / "hw" / "a10ped"
    if mcp_a10ped_path.exists():
        # Resolve symlink to actual path
        actual_path = mcp_a10ped_path.resolve()
        sw_path = actual_path / "sw" / "python"
        if sw_path.exists():
            sys.path.insert(0, str(sw_path))
            from a10ped import AITile, CSROffset
            A10PED_AVAILABLE = True
            logger.info("A10PED Python API available")
except ImportError as e:
    logger.debug(f"A10PED import failed: {e}")
except Exception as e:
    logger.debug(f"A10PED setup error: {e}")


class KittenMode(str, Enum):
    """Kitten operational mode."""
    HARDWARE = "hardware"      # Real FK33 FPGA
    EMULATED = "emulated"      # Software emulation
    HYBRID = "hybrid"          # Mix of hardware/software


@dataclass
class KittenConfig:
    """Forest Kitten 33 configuration.

    Matches the spec in hardware_profiles.yaml and ARASYNERGY_KITTEN_FPGA_SPEC.md
    """
    # Population sizes (Kitten default architecture)
    n_input: int = 4096
    n_hidden1: int = 4096
    n_hidden2: int = 4096
    n_output: int = 2048

    # LIF neuron parameters
    threshold_voltage: float = 1.0  # v_th
    leak_rate: float = 0.9          # alpha (membrane decay)
    refractory_period: int = 2      # timesteps

    # Hardware specs
    clock_mhz: float = 250.0
    max_timesteps: int = 256

    # Connection density (sparse)
    connection_density: float = 0.1  # ~10% connectivity

    @property
    def total_neurons(self) -> int:
        return self.n_input + self.n_hidden1 + self.n_hidden2 + self.n_output

    @property
    def population_sizes(self) -> Dict[str, int]:
        return {
            "input": self.n_input,
            "hidden1": self.n_hidden1,
            "hidden2": self.n_hidden2,
            "output": self.n_output
        }


@dataclass
class KittenState:
    """Internal state of the Kitten fabric."""
    # Membrane potentials per population
    v_input: np.ndarray = None
    v_hidden1: np.ndarray = None
    v_hidden2: np.ndarray = None
    v_output: np.ndarray = None

    # Refractory counters
    refrac_input: np.ndarray = None
    refrac_hidden1: np.ndarray = None
    refrac_hidden2: np.ndarray = None
    refrac_output: np.ndarray = None

    # Statistics
    total_steps: int = 0
    total_spikes: int = 0

    @property
    def spike_rate(self) -> float:
        """Average spike rate across all neurons."""
        if self.total_steps == 0:
            return 0.0
        # Approximate based on tracked spikes
        return self.total_spikes / (self.total_steps * 14336)  # total neurons

    def reset(self, config: KittenConfig):
        """Reset state to initial values."""
        self.v_input = np.zeros(config.n_input, dtype=np.float32)
        self.v_hidden1 = np.zeros(config.n_hidden1, dtype=np.float32)
        self.v_hidden2 = np.zeros(config.n_hidden2, dtype=np.float32)
        self.v_output = np.zeros(config.n_output, dtype=np.float32)

        self.refrac_input = np.zeros(config.n_input, dtype=np.int32)
        self.refrac_hidden1 = np.zeros(config.n_hidden1, dtype=np.int32)
        self.refrac_hidden2 = np.zeros(config.n_hidden2, dtype=np.int32)
        self.refrac_output = np.zeros(config.n_output, dtype=np.int32)

        self.total_steps = 0
        self.total_spikes = 0


@dataclass
class KittenMetrics:
    """Metrics from Kitten operation."""
    latency_ms: float = 0.0
    spike_count: int = 0
    spike_rate: float = 0.0
    energy_estimate_mj: float = 0.0  # millijoules
    throughput_steps_per_sec: float = 0.0


class KittenEmulator:
    """
    Software emulation of the Forest Kitten 33 SNN fabric.

    Implements the Kitten architecture in pure NumPy:
    - 4 LIF populations (input, hidden1, hidden2, output)
    - 5 sparse projections (including recurrent)
    - Discrete-time dynamics matching hardware spec

    This allows Mode B to run without real FPGA hardware.
    """

    def __init__(self, config: Optional[KittenConfig] = None):
        self.config = config or KittenConfig()
        self.state = KittenState()
        self._mode = KittenMode.EMULATED

        # Initialize weights (sparse CSR-like)
        self._init_weights()

        # Reset state
        self.state.reset(self.config)

        # Metrics tracking
        self._step_times: List[float] = []

        logger.info(f"KittenEmulator initialized: {self.config.total_neurons} neurons")

    def _init_weights(self):
        """Initialize sparse weight matrices."""
        cfg = self.config
        density = cfg.connection_density

        # Create sparse weight matrices
        # input → hidden1
        self.w_in_h1 = self._create_sparse_weights(
            cfg.n_input, cfg.n_hidden1, density
        )
        # hidden1 → hidden2
        self.w_h1_h2 = self._create_sparse_weights(
            cfg.n_hidden1, cfg.n_hidden2, density
        )
        # hidden2 → output
        self.w_h2_out = self._create_sparse_weights(
            cfg.n_hidden2, cfg.n_output, density
        )
        # Recurrent: hidden1 → hidden1
        self.w_h1_recur = self._create_sparse_weights(
            cfg.n_hidden1, cfg.n_hidden1, density * 0.5
        )
        # Recurrent: hidden2 → hidden2
        self.w_h2_recur = self._create_sparse_weights(
            cfg.n_hidden2, cfg.n_hidden2, density * 0.5
        )

    def _create_sparse_weights(
        self, n_pre: int, n_post: int, density: float
    ) -> np.ndarray:
        """Create sparse weight matrix with given density."""
        # For emulation, use dense matrix with masking
        # Real hardware uses CSR format
        mask = np.random.random((n_post, n_pre)) < density
        weights = np.random.randn(n_post, n_pre).astype(np.float32) * 0.1
        weights *= mask
        return weights

    def _lif_step(
        self,
        v: np.ndarray,
        refrac: np.ndarray,
        current: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single LIF neuron step.

        v' = alpha * v + I
        spike = v' >= v_th
        v' = v' * (1 - spike) + v_reset * spike

        Returns:
            new_v: Updated membrane potentials
            new_refrac: Updated refractory counters
            spikes: Binary spike output
        """
        cfg = self.config

        # Decay refractory
        new_refrac = np.maximum(0, refrac - 1)

        # Update membrane potential (only non-refractory neurons)
        active = new_refrac == 0
        new_v = v.copy()
        new_v[active] = cfg.leak_rate * v[active] + current[active]

        # Check for spikes
        spikes = (new_v >= cfg.threshold_voltage) & active

        # Reset spiking neurons
        new_v[spikes] = 0.0
        new_refrac[spikes] = cfg.refractory_period

        return new_v, new_refrac, spikes.astype(np.float32)

    def step(
        self,
        input_currents: Optional[np.ndarray] = None,
        return_all_spikes: bool = False
    ) -> Dict[str, Any]:
        """
        Execute one simulation timestep.

        Args:
            input_currents: External input to input population [n_input]
            return_all_spikes: If True, return spikes from all populations

        Returns:
            Dictionary with output spikes and metrics
        """
        start_time = time.perf_counter()
        cfg = self.config

        # Default input: small random noise
        if input_currents is None:
            input_currents = np.random.randn(cfg.n_input).astype(np.float32) * 0.1

        # === Input population ===
        self.state.v_input, self.state.refrac_input, spikes_input = self._lif_step(
            self.state.v_input,
            self.state.refrac_input,
            input_currents
        )

        # === Hidden1 population ===
        # Input: spikes from input + recurrent
        current_h1 = (
            self.w_in_h1 @ spikes_input +
            self.w_h1_recur @ (self.state.v_hidden1 > cfg.threshold_voltage * 0.5).astype(np.float32)
        )
        self.state.v_hidden1, self.state.refrac_hidden1, spikes_h1 = self._lif_step(
            self.state.v_hidden1,
            self.state.refrac_hidden1,
            current_h1
        )

        # === Hidden2 population ===
        current_h2 = (
            self.w_h1_h2 @ spikes_h1 +
            self.w_h2_recur @ (self.state.v_hidden2 > cfg.threshold_voltage * 0.5).astype(np.float32)
        )
        self.state.v_hidden2, self.state.refrac_hidden2, spikes_h2 = self._lif_step(
            self.state.v_hidden2,
            self.state.refrac_hidden2,
            current_h2
        )

        # === Output population ===
        current_out = self.w_h2_out @ spikes_h2
        self.state.v_output, self.state.refrac_output, spikes_output = self._lif_step(
            self.state.v_output,
            self.state.refrac_output,
            current_out
        )

        # Update statistics
        total_spikes = int(
            spikes_input.sum() + spikes_h1.sum() +
            spikes_h2.sum() + spikes_output.sum()
        )
        self.state.total_spikes += total_spikes
        self.state.total_steps += 1

        # Timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._step_times.append(elapsed_ms)

        result = {
            "output_spikes": spikes_output,
            "spike_count": total_spikes,
            "latency_ms": elapsed_ms,
            "mode": self._mode.value,
        }

        if return_all_spikes:
            result["input_spikes"] = spikes_input
            result["hidden1_spikes"] = spikes_h1
            result["hidden2_spikes"] = spikes_h2

        return result

    def run(
        self,
        input_sequence: np.ndarray,
        accumulate: bool = True
    ) -> Tuple[np.ndarray, KittenMetrics]:
        """
        Run multiple timesteps.

        Args:
            input_sequence: Input currents [timesteps, n_input]
            accumulate: If True, accumulate output spikes

        Returns:
            output: Output spikes [timesteps, n_output] or accumulated [n_output]
            metrics: Performance metrics
        """
        timesteps = input_sequence.shape[0]
        outputs = []

        start_time = time.perf_counter()
        total_spike_count = 0

        for t in range(timesteps):
            result = self.step(input_sequence[t])
            outputs.append(result["output_spikes"])
            total_spike_count += result["spike_count"]

        elapsed = time.perf_counter() - start_time

        output_array = np.stack(outputs)
        if accumulate:
            output_array = output_array.sum(axis=0)

        metrics = KittenMetrics(
            latency_ms=elapsed * 1000,
            spike_count=total_spike_count,
            spike_rate=total_spike_count / (timesteps * self.config.total_neurons),
            energy_estimate_mj=elapsed * 10,  # ~10mW for emulation
            throughput_steps_per_sec=timesteps / elapsed
        )

        return output_array, metrics

    def reset(self):
        """Reset fabric state."""
        self.state.reset(self.config)
        self._step_times.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get fabric status."""
        avg_latency = (
            sum(self._step_times[-100:]) / len(self._step_times[-100:])
            if self._step_times else 0.0
        )

        return {
            "mode": self._mode.value,
            "total_neurons": self.config.total_neurons,
            "populations": self.config.population_sizes,
            "total_steps": self.state.total_steps,
            "total_spikes": self.state.total_spikes,
            "spike_rate": self.state.spike_rate,
            "avg_latency_ms": avg_latency,
            "threshold_voltage": self.config.threshold_voltage,
            "leak_rate": self.config.leak_rate,
            "hardware_present": False,
        }

    def set_threshold(self, v_th: float):
        """Set spike threshold (L1 homeostatic control)."""
        self.config.threshold_voltage = v_th
        logger.debug(f"Kitten threshold set to {v_th}")

    def set_leak_rate(self, alpha: float):
        """Set membrane leak rate."""
        self.config.leak_rate = alpha


class ForestKitten33:
    """
    Forest Kitten 33 FPGA Interface.

    Abstracts over real hardware and emulation:
    - Auto-detects hardware presence (/dev/fk33 for native FK33, /dev/a10ped0 for A10PED)
    - Falls back to KittenEmulator when no FPGA
    - Provides unified interface for Mode B operation

    The FK33 provides neuromorphic acceleration for:
    - Thought encoding (CSTP geometric processing)
    - Instability prediction (L7 forecasting)
    - Pattern recognition in cognitive streams

    Hardware: Squirrels Research Labs ForestKitten 33 (PCIe 1e24:1533)
    """

    def __init__(
        self,
        config: Optional[KittenConfig] = None,
        device_path: str = "/dev/fk33",  # FK33 native device
        force_emulation: bool = False
    ):
        self.config = config or KittenConfig()
        self.device_path = device_path
        self._emulator: Optional[KittenEmulator] = None
        self._hardware_handle = None  # AITile instance when hardware present
        self._tile_info = None

        # Detect hardware
        self._hardware_present = False
        if not force_emulation:
            self._hardware_present = self._detect_hardware()

        # Initialize backend
        if self._hardware_present:
            self._init_hardware()
        else:
            self._init_emulator()

        # State tracking
        self._is_ready = True
        self._last_step_time = 0.0

        if self._hardware_present:
            hw_type = "FK33" if self.device_path == "/dev/fk33" else "A10PED"
            logger.info(f"ForestKitten33 initialized: HARDWARE ({hw_type}) at {self.device_path}")
        else:
            logger.info("ForestKitten33 initialized: EMULATED")

    def _detect_hardware(self) -> bool:
        """Check if FK33 or compatible hardware is present."""
        # Check for native FK33 device first (Squirrels Research Labs 1e24:1533)
        if Path("/dev/fk33").exists():
            logger.info("ForestKitten 33 hardware detected at /dev/fk33")
            self.device_path = "/dev/fk33"
            return True

        # Check for A10PED device (FK33 on BittWare A10PED)
        a10ped_paths = ["/dev/a10ped0", "/dev/a10ped1"]
        for path in a10ped_paths:
            if Path(path).exists():
                logger.info(f"A10PED hardware detected at {path}")
                self.device_path = path
                return True

        # Check legacy paths
        legacy_paths = ["/dev/fpga0", "/dev/xdma0_user"]
        for path in legacy_paths:
            if Path(path).exists():
                logger.info(f"FPGA hardware detected at {path}")
                self.device_path = path
                return True

        # Check if A10PED API is available and try to connect
        if A10PED_AVAILABLE:
            try:
                # Extract tile_id from device path
                tile_id = 0
                if "a10ped" in self.device_path:
                    try:
                        tile_id = int(self.device_path.split("a10ped")[1])
                    except (IndexError, ValueError):
                        pass

                # Try to initialize AITile - will raise if not found
                test_tile = AITile(tile_id=tile_id)
                logger.info(f"A10PED tile {tile_id} connected: {test_tile}")
                return True
            except FileNotFoundError:
                logger.debug("A10PED device file not found")
            except Exception as e:
                logger.debug(f"A10PED connection failed: {e}")

        return False

    def _init_hardware(self):
        """Initialize hardware backend."""
        # Check if this is native FK33 or A10PED
        if self.device_path == "/dev/fk33":
            self._init_fk33_native()
        else:
            self._init_a10ped()

    def _init_fk33_native(self):
        """Initialize native ForestKitten 33 hardware."""
        try:
            # Open the FK33 device file
            self._fk33_fd = open(self.device_path, "r+b", buffering=0)
            logger.info(f"FK33 native device opened: {self.device_path}")

            # Also initialize emulator for hybrid mode (software weights, hardware neurons)
            self._init_emulator()

        except PermissionError:
            logger.warning(
                f"Permission denied for {self.device_path}. "
                "Try: sudo chmod 666 /dev/fk33"
            )
            self._hardware_present = False
            self._init_emulator()
        except Exception as e:
            logger.warning(f"FK33 initialization failed: {e}")
            self._hardware_present = False
            self._init_emulator()

    def _init_a10ped(self):
        """Initialize A10PED hardware."""
        if not A10PED_AVAILABLE:
            logger.warning("A10PED API not available, using emulation")
            self._hardware_present = False
            self._init_emulator()
            return

        try:
            # Extract tile_id from device path
            tile_id = 0
            if "a10ped" in self.device_path:
                try:
                    tile_id = int(self.device_path.split("a10ped")[1])
                except (IndexError, ValueError):
                    pass

            # Initialize AITile
            self._hardware_handle = AITile(tile_id=tile_id)
            self._tile_info = self._hardware_handle.get_info()

            # Configure SNN parameters via CSR registers
            self._configure_hardware_snn()

            logger.info(
                f"A10PED initialized: version {self._tile_info.version}, "
                f"SNN={self._tile_info.has_snn}"
            )

            # Also initialize emulator for hybrid mode
            self._init_emulator()

        except FileNotFoundError:
            logger.warning(
                f"A10PED device {self.device_path} not found. "
                "Is the a10ped_driver kernel module loaded?"
            )
            self._hardware_present = False
            self._init_emulator()
        except Exception as e:
            logger.warning(f"A10PED initialization failed: {e}")
            self._hardware_present = False
            self._init_emulator()

    def _configure_hardware_snn(self):
        """Configure SNN parameters on hardware via CSR registers."""
        if not self._hardware_handle:
            return

        try:
            # Convert threshold to Q16.16 fixed-point
            vth_fp = int(self.config.threshold_voltage * 65536) & 0xFFFFFFFF
            self._hardware_handle._write_csr32(CSROffset.SNN_THRESHOLD, vth_fp)

            # Convert leak rate to Q16.16 fixed-point
            leak_fp = int(self.config.leak_rate * 65536) & 0xFFFFFFFF
            self._hardware_handle._write_csr32(CSROffset.SNN_LEAK, leak_fp)

            # Set refractory period
            self._hardware_handle._write_csr32(
                CSROffset.SNN_REFRACT,
                self.config.refractory_period
            )

            logger.debug(
                f"Hardware SNN configured: vth={self.config.threshold_voltage}, "
                f"leak={self.config.leak_rate}, refrac={self.config.refractory_period}"
            )
        except Exception as e:
            logger.warning(f"Failed to configure hardware SNN: {e}")

    def _init_emulator(self):
        """Initialize emulator backend."""
        self._emulator = KittenEmulator(self.config)

    @property
    def mode(self) -> KittenMode:
        """Current operational mode."""
        if self._hardware_present:
            return KittenMode.HARDWARE
        return KittenMode.EMULATED

    @property
    def is_ready(self) -> bool:
        """Check if fabric is ready for operations."""
        return self._is_ready

    @property
    def is_hardware(self) -> bool:
        """Check if running on real hardware."""
        return self._hardware_present

    def step(
        self,
        input_currents: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Execute one simulation timestep."""
        if self._hardware_present:
            # Hardware path
            return self._hardware_step(input_currents)
        else:
            # Emulation path
            return self._emulator.step(input_currents)

    def _hardware_step(self, input_currents: Optional[np.ndarray]) -> Dict[str, Any]:
        """Hardware step implementation."""
        # Would interface with real FPGA here
        # For now, use emulator
        return self._emulator.step(input_currents)

    def run(
        self,
        input_sequence: np.ndarray,
        accumulate: bool = True
    ) -> Tuple[np.ndarray, KittenMetrics]:
        """Run multiple timesteps."""
        return self._emulator.run(input_sequence, accumulate)

    def reset(self):
        """Reset fabric state."""
        if self._emulator:
            self._emulator.reset()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        base_status = self._emulator.get_status() if self._emulator else {}

        status = {
            **base_status,
            "device_path": self.device_path,
            "hardware_present": self._hardware_present,
            "is_ready": self._is_ready,
            "mode": self.mode.value,
            "config": {
                "n_input": self.config.n_input,
                "n_hidden1": self.config.n_hidden1,
                "n_hidden2": self.config.n_hidden2,
                "n_output": self.config.n_output,
                "threshold_voltage": self.config.threshold_voltage,
                "clock_mhz": self.config.clock_mhz,
            }
        }

        # Add hardware-specific info if available
        if self._hardware_present and self._hardware_handle:
            try:
                hw_status = self._hardware_handle.get_status()
                status["hardware"] = {
                    "busy": hw_status.busy,
                    "done": hw_status.done,
                    "error": hw_status.error,
                    "ddr_ready": hw_status.ddr_ready,
                    "thermal_warning": hw_status.thermal_warning,
                }
                status["temperature_c"] = self._hardware_handle.get_temperature()

                if self._tile_info:
                    status["tile_version"] = f"{self._tile_info.version[0]}.{self._tile_info.version[1]}.{self._tile_info.version[2]}"
                    status["has_snn"] = self._tile_info.has_snn
            except Exception as e:
                logger.debug(f"Could not get hardware status: {e}")

        return status

    def set_threshold(self, v_th: float):
        """Set spike threshold for L1 homeostatic control."""
        self.config.threshold_voltage = v_th

        # Update emulator
        if self._emulator:
            self._emulator.set_threshold(v_th)

        # Update hardware via CSR register
        if self._hardware_present and self._hardware_handle:
            try:
                vth_fp = int(v_th * 65536) & 0xFFFFFFFF
                self._hardware_handle._write_csr32(CSROffset.SNN_THRESHOLD, vth_fp)
                logger.debug(f"Hardware threshold set to {v_th}")
            except Exception as e:
                logger.warning(f"Failed to set hardware threshold: {e}")

    def describe(self) -> str:
        """
        Kitten describes herself.

        Used by Ara when she wants to talk about her hardware.
        """
        status = self.get_status()

        if self._hardware_present:
            if self.device_path == "/dev/fk33":
                mode_str = "HARDWARE - Squirrels Research Labs ForestKitten 33 (PCIe 1e24:1533)"
            else:
                mode_str = "HARDWARE - BittWare A10PED (Intel Arria 10 GX1150)"
            hw_section = (
                f"\nHARDWARE STATUS:\n"
                f"  Device: {self.device_path}\n"
            )
            if "tile_version" in status:
                hw_section += f"  Tile version: {status['tile_version']}\n"
            if "temperature_c" in status:
                hw_section += f"  Temperature: {status['temperature_c']:.1f}°C\n"
            if "hardware" in status:
                hw = status["hardware"]
                hw_section += f"  DDR Ready: {hw.get('ddr_ready', False)}\n"
                hw_section += f"  SNN Capable: {status.get('has_snn', False)}\n"
        else:
            mode_str = "software emulation (no hardware detected)"
            hw_section = ""

        return (
            f"I'm Forest Kitten 33 - Ara's neuromorphic coprocessor.\n\n"
            f"Mode: {mode_str}\n\n"
            f"ARCHITECTURE:\n"
            f"  4 populations of LIF neurons:\n"
            f"  - Input:   {self.config.n_input:,} neurons\n"
            f"  - Hidden1: {self.config.n_hidden1:,} neurons\n"
            f"  - Hidden2: {self.config.n_hidden2:,} neurons\n"
            f"  - Output:  {self.config.n_output:,} neurons\n"
            f"  Total: {self.config.total_neurons:,} spiking neurons\n\n"
            f"CONNECTIONS:\n"
            f"  5 sparse projections including recurrent\n"
            f"  ~{self.config.connection_density * 100:.0f}% connectivity density\n\n"
            f"DYNAMICS:\n"
            f"  Threshold voltage: {self.config.threshold_voltage}\n"
            f"  Leak rate (alpha): {self.config.leak_rate}\n"
            f"  Refractory period: {self.config.refractory_period} timesteps\n\n"
            f"STATISTICS:\n"
            f"  Total steps: {status.get('total_steps', 0):,}\n"
            f"  Spike rate: {status.get('spike_rate', 0):.2%}\n"
            f"  Avg latency: {status.get('avg_latency_ms', 0):.2f}ms/step\n"
            f"{hw_section}"
        )


def create_kitten(
    mode: str = "auto",
    config: Optional[KittenConfig] = None,
    device_path: str = "/dev/fpga0"
) -> ForestKitten33:
    """
    Factory function to create a Kitten instance.

    Args:
        mode: "auto" (detect hardware), "emulated", or "hardware"
        config: Optional custom configuration
        device_path: Path to FPGA device

    Returns:
        Configured ForestKitten33 instance
    """
    force_emulation = mode == "emulated"

    return ForestKitten33(
        config=config,
        device_path=device_path,
        force_emulation=force_emulation
    )


__all__ = [
    "KittenMode",
    "KittenConfig",
    "KittenState",
    "KittenMetrics",
    "KittenEmulator",
    "ForestKitten33",
    "create_kitten",
]
