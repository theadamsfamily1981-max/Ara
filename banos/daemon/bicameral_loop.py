#!/usr/bin/env python3
"""
BICAMERAL LOOP - The Mind-Body Integration Daemon
==================================================

Bio-Affective Neuromorphic Operating System
Bidirectional communication between Symbolic (LLM) and Spiking (SNN) worlds

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    THREADRIPPER (CORTEX)                    │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │  Language   │◄───│  Somatic    │◄───│  Bicameral  │     │
    │  │    Model    │    │  Injector   │    │    Loop     │     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    │         │                  ▲                  │             │
    │         │ Attention        │ Ascending        │ Descending  │
    │         ▼ Vectors          │ Spikes           ▼ Projection  │
    └─────────┬──────────────────┼──────────────────┬─────────────┘
              │                  │                  │
    ══════════╪══════════════════╪══════════════════╪═══════════ PCIe
              │                  │                  │
    ┌─────────▼──────────────────┼──────────────────▼─────────────┐
    │                     FPGA (BRAINSTEM)                        │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │   Neuro-    │    │   Kitten    │    │   Dream     │     │
    │  │  Symbolic   │───►│   Fabric    │◄───│   Engine    │     │
    │  │   Bridge    │    │    (SNN)    │    │   (STDP)    │     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    │         │                  │                  │             │
    │         │ Spike            │ Aggregate        │ Weight      │
    │         ▼ Injection        ▼ State            ▼ Updates     │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │                   DDR4 SPIKE LOG                     │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Loop Frequency: 20Hz (50ms period) - Fast enough for reflexes

Pathways:
1. DESCENDING (Thought -> Body):
   - Extract attention patterns from LLM
   - Project to FPGA via neuro_symbolic_bridge
   - Stimulate specific neuron populations

2. ASCENDING (Body -> Thought):
   - Read aggregate spike state from fabric
   - Compute somatic vectors (pain, arousal, pleasure)
   - Inject into LLM via SomaticInjector

3. AUTONOMIC (Self-Regulation):
   - Monitor system health (temperature, load)
   - Adjust throttling via AutonomicController
   - Trigger sleep mode when idle

Usage:
    python bicameral_loop.py                  # Run as daemon
    python bicameral_loop.py --debug          # Debug mode
    python bicameral_loop.py --freq 10        # 10Hz loop
"""

import argparse
import logging
import signal
import sys
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BicameralLoop")

# Import HAL
try:
    from banos.hal.ara_hal import (
        AraHAL,
        create_somatic_bus,
        connect_somatic_bus,
        SystemState,
        DreamState,
        AutonomicController,
    )
    HAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HAL not available: {e}")
    HAL_AVAILABLE = False

# Import FPGA bridge (optional - may require custom driver)
try:
    from banos.fpga.driver import FPGABridge, SpikePacket
    FPGA_AVAILABLE = True
except ImportError:
    FPGA_AVAILABLE = False
    logger.info("FPGA bridge not available (simulation mode)")

# Import Narrative Self components (Iteration 5)
try:
    from banos.daemon.dreamer import Dreamer
    from banos.daemon.somatic_budget import SomaticBudget, Budget, get_budget
    NARRATIVE_AVAILABLE = True
except ImportError:
    try:
        from dreamer import Dreamer
        from somatic_budget import SomaticBudget, Budget, get_budget
        NARRATIVE_AVAILABLE = True
    except ImportError:
        NARRATIVE_AVAILABLE = False
        logger.info("Narrative Self not available (no Dreamer/Budget)")


@dataclass
class LoopMetrics:
    """Metrics for the bicameral loop."""
    loop_count: int = 0
    avg_loop_time_ms: float = 0.0
    descending_packets: int = 0
    ascending_spikes: int = 0
    autonomic_updates: int = 0
    last_system_state: str = "NORMAL"
    # Narrative Self metrics (Iteration 5)
    narrative_recalls: int = 0
    budget_checks: int = 0
    last_budget_style: str = "verbose"


class BicameralLoop:
    """
    The main integration daemon that fuses LLM and SNN.

    This is the "spinal cord" that connects the brain (LLM)
    to the body (FPGA SNN).
    """

    def __init__(
        self,
        loop_freq_hz: float = 20.0,
        debug: bool = False
    ):
        """
        Initialize the bicameral loop.

        Args:
            loop_freq_hz: Loop frequency in Hz (default 20Hz = 50ms period)
            debug: Enable debug logging
        """
        self.loop_freq = loop_freq_hz
        self.loop_period = 1.0 / loop_freq_hz
        self.debug = debug
        self.running = False

        # Components
        self.hal: Optional[AraHAL] = None
        self.autonomic: Optional[AutonomicController] = None
        self.fpga: Optional[Any] = None  # FPGABridge when available

        # Narrative Self components (Iteration 5)
        self.dreamer: Optional[Any] = None  # Dreamer when available
        self.budget: Optional[Any] = None   # SomaticBudget when available
        self._last_budget: Optional[Any] = None
        self._narrative_context: Optional[str] = None

        # State
        self.metrics = LoopMetrics()
        self._last_attention: Optional[Dict] = None
        self._last_spike_state: Optional[Dict] = None

        # Thread control
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

    def start(self) -> None:
        """Start the bicameral loop."""
        logger.info(f"Starting Bicameral Loop @ {self.loop_freq}Hz")

        # Initialize HAL
        if HAL_AVAILABLE:
            try:
                self.hal = create_somatic_bus()
                self.autonomic = AutonomicController(self.hal)
                logger.info("HAL initialized (creator mode)")
            except Exception as e:
                logger.warning(f"Failed to create HAL, trying connect: {e}")
                try:
                    self.hal = connect_somatic_bus()
                    self.autonomic = AutonomicController(self.hal)
                    logger.info("HAL initialized (connect mode)")
                except Exception as e2:
                    logger.error(f"HAL unavailable: {e2}")
                    self.hal = None

        # Initialize FPGA bridge
        if FPGA_AVAILABLE:
            try:
                self.fpga = FPGABridge()
                self.fpga.open()
                logger.info("FPGA bridge initialized")
            except Exception as e:
                logger.warning(f"FPGA bridge unavailable: {e}")
                self.fpga = None

        # Initialize Narrative Self (Iteration 5)
        if NARRATIVE_AVAILABLE:
            try:
                self.dreamer = Dreamer()
                self.budget = SomaticBudget(hal=self.hal, dreamer=self.dreamer)
                logger.info("Narrative Self initialized (Dreamer + Budget)")
            except Exception as e:
                logger.warning(f"Narrative Self unavailable: {e}")
                self.dreamer = None
                self.budget = None

        # Start loop thread
        self.running = True
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        logger.info("Bicameral Loop running")

    def stop(self) -> None:
        """Stop the bicameral loop."""
        logger.info("Stopping Bicameral Loop...")
        self.running = False
        self._stop_event.set()

        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)

        if self.hal:
            self.hal.close()

        if self.fpga:
            self.fpga.close()

        logger.info("Bicameral Loop stopped")

    def _run_loop(self) -> None:
        """Main loop thread."""
        while self.running and not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                self._loop_iteration()
            except Exception as e:
                logger.error(f"Loop iteration failed: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_period - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
            elif self.debug:
                logger.warning(f"Loop overrun: {elapsed*1000:.1f}ms > {self.loop_period*1000:.1f}ms")

            # Update metrics
            self.metrics.loop_count += 1
            self.metrics.avg_loop_time_ms = (
                self.metrics.avg_loop_time_ms * 0.95 +
                elapsed * 1000 * 0.05
            )

    def _loop_iteration(self) -> None:
        """Single iteration of the bicameral loop."""

        # 1. ASCENDING PATHWAY: Read body state
        self._ascending_pathway()

        # 2. AUTONOMIC: Update system state and throttling
        self._autonomic_update()

        # 3. BUDGET CHECK: Update resource permissions (every 10 loops)
        if self.metrics.loop_count % 10 == 0:
            self._update_budget()

        # 4. DESCENDING PATHWAY: Project attention to body
        self._descending_pathway()

        # 5. DREAM CHECK: Trigger sleep if idle
        self._check_dream_trigger()

    def _ascending_pathway(self) -> None:
        """
        Read aggregate spike state from FPGA and update HAL.

        Body -> Mind pathway.
        """
        if not self.hal:
            return

        if self.fpga:
            # Read real spike data from FPGA
            try:
                spike_state = self.fpga.read_aggregate_state()
                self._last_spike_state = spike_state

                # Update HAL with spike-derived somatic state
                pain = spike_state.get('pain_accumulator', 0) / 65535.0
                neurons = spike_state.get('active_neurons', 0)
                spikes = spike_state.get('total_spikes', 0)

                self.hal.write_pain_raw(int(pain * 0xFFFFFFFF))
                self.hal.write_fpga_diagnostics(
                    active_neurons=neurons,
                    total_spikes=spikes,
                    fabric_temp_mc=spike_state.get('temperature_mc', 45000),
                    fabric_online=True,
                    dream_active=spike_state.get('dream_active', False)
                )

                self.metrics.ascending_spikes += 1

            except Exception as e:
                logger.debug(f"FPGA read failed: {e}")
        else:
            # Simulation mode: generate synthetic state
            self._simulate_ascending()

    def _simulate_ascending(self) -> None:
        """Simulate ascending pathway for testing without FPGA."""
        if not self.hal:
            return

        import math
        t = time.time()

        # Synthetic oscillating values
        pain = (math.sin(t * 0.5) + 1) * 0.2  # 0.0 - 0.4
        neurons = int(50 + 30 * math.sin(t * 0.3))
        spikes = self.metrics.loop_count * 10

        self.hal.write_pain_raw(int(pain * 0xFFFFFFFF))
        self.hal.write_fpga_diagnostics(
            active_neurons=neurons,
            total_spikes=spikes,
            fabric_temp_mc=45000 + int(5000 * math.sin(t * 0.1)),
            fabric_online=True,
            dream_active=False
        )

    def _descending_pathway(self) -> None:
        """
        Project LLM attention patterns to FPGA for spike injection.

        Mind -> Body pathway.
        """
        if not self.fpga:
            return

        if self._last_attention is None:
            return

        try:
            # Extract attention hotspots
            attention = self._last_attention.get('attention_weights')
            if attention is None:
                return

            # Create spike packets for high-attention regions
            packets = self._attention_to_spikes(attention)

            # Send to FPGA
            for packet in packets:
                self.fpga.inject_spike(packet)

            self.metrics.descending_packets += len(packets)

        except Exception as e:
            logger.debug(f"Descending pathway failed: {e}")

    def _attention_to_spikes(self, attention) -> list:
        """
        Convert LLM attention weights to spike packets.

        High attention on certain token positions -> spike certain neuron groups.
        """
        import numpy as np
        packets = []

        # Threshold: only top 10% attention values
        if hasattr(attention, 'numpy'):
            attn = attention.numpy()
        else:
            attn = np.array(attention)

        threshold = np.percentile(attn, 90)
        hot_indices = np.where(attn > threshold)

        for idx in zip(*hot_indices):
            # Map attention index to FPGA tile coordinates
            tile_x = idx[-1] % 4
            tile_y = idx[-1] // 4 % 4
            neuron_id = idx[-1] % 256
            intensity = int(min(255, attn[idx] * 255))

            if FPGA_AVAILABLE:
                packets.append(SpikePacket(
                    dest_x=tile_x,
                    dest_y=tile_y,
                    neuron_id=neuron_id,
                    payload=intensity
                ))

        return packets

    def _autonomic_update(self) -> None:
        """Update autonomic controller for self-regulation."""
        if not self.autonomic:
            return

        try:
            state = self.autonomic.update()
            self.metrics.last_system_state = state.name
            self.metrics.autonomic_updates += 1

            if self.debug and self.metrics.loop_count % 100 == 0:
                throttle = self.autonomic.get_throttle()
                logger.debug(f"Autonomic: {state.name}, throttle={throttle:.0%}")

        except Exception as e:
            logger.debug(f"Autonomic update failed: {e}")

    def _check_dream_trigger(self) -> None:
        """Check if system should enter sleep/dream mode."""
        if not self.autonomic or not self.hal:
            return

        # Trigger sleep after 60 seconds of idle (600 ticks at 10Hz update)
        if self.autonomic.should_sleep(idle_threshold=600):
            logger.info("System idle, triggering dream mode...")
            self.hal.set_dream_state(DreamState.REM)
            self.hal.trigger_sleep()

            # Run dream consolidation if available
            if self.dreamer:
                try:
                    memories = self.dreamer.dream(force=True)
                    logger.info(f"Dream consolidation: {len(memories)} memories stored")
                except Exception as e:
                    logger.error(f"Dream consolidation failed: {e}")

    def _update_budget(self) -> None:
        """Update resource budget based on current state."""
        if not self.budget:
            return

        try:
            self._last_budget = self.budget.compute_budget()
            self.metrics.budget_checks += 1
            self.metrics.last_budget_style = self._last_budget.suggested_style

            if self.debug:
                logger.debug(
                    f"Budget: LLM={self._last_budget.allow_heavy_llm}, "
                    f"GPU={self._last_budget.allow_heavy_gpu}, "
                    f"style={self._last_budget.suggested_style}"
                )
        except Exception as e:
            logger.debug(f"Budget update failed: {e}")

    # =========================================================================
    # NARRATIVE SELF API (Iteration 5)
    # =========================================================================

    def get_narrative_context(
        self,
        query: str,
        top_k: int = 3
    ) -> Optional[str]:
        """
        Retrieve narrative context from episodic memory.

        Uses Somatic RAG to find memories that match both:
        - Semantic content (what was happening)
        - Emotional state (how we felt)
        - Hardware state (system conditions)

        Args:
            query: What to search for
            top_k: Number of memories to retrieve

        Returns:
            Formatted narrative context string, or None
        """
        if not self.dreamer or not self.hal:
            return None

        try:
            # Read current state
            somatic = self.hal.read_somatic()
            pad = somatic.get('pad', {'p': 0, 'a': 0, 'd': 0})

            current_pad = {
                'p': pad.get('p', 0),
                'a': pad.get('a', 0),
                'd': pad.get('d', 0),
            }

            current_hardware = {
                'cpu_load': somatic.get('cpu_load', 0),
                'gpu_load': somatic.get('gpu_load', 0),
                'vram_used': somatic.get('memory_used', 0),
                'fpga_temp': 45,  # Default
            }

            # Somatic RAG retrieval
            memories = self.dreamer.somatic_recall(
                query=query,
                current_pad=current_pad,
                current_hardware=current_hardware,
                top_k=top_k
            )

            if not memories:
                return None

            self.metrics.narrative_recalls += 1

            # Format as context string
            context_lines = ["[Relevant past experiences:]"]
            for mem in memories:
                content = mem.get('content', '')[:200]
                score = mem.get('score', 0)
                context_lines.append(f"- ({score:.2f}) {content}")

            self._narrative_context = "\n".join(context_lines)
            return self._narrative_context

        except Exception as e:
            logger.debug(f"Narrative context retrieval failed: {e}")
            return None

    def get_budget(self, task_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get current resource budget.

        Args:
            task_hint: Optional hint about intended task

        Returns:
            Budget dict with permissions and style guidance
        """
        if not self.budget:
            return None

        try:
            budget = self.budget.compute_budget(task_hint=task_hint)
            return {
                'allow_heavy_llm': budget.allow_heavy_llm,
                'allow_heavy_gpu': budget.allow_heavy_gpu,
                'allow_fpga_burst': budget.allow_fpga_burst,
                'suggested_style': budget.suggested_style,
                'reason': budget.reason,
                'pain': budget.pain,
                'system_state': budget.system_state,
            }
        except Exception as e:
            logger.debug(f"Budget check failed: {e}")
            return None

    def record_task(
        self,
        task_type: str,
        cpu_used: float,
        gpu_used: float,
        duration_s: float,
        success: bool
    ) -> None:
        """
        Record a task execution for cost model learning.

        Args:
            task_type: Type of task (e.g., "llm_inference")
            cpu_used: CPU utilization [0-1]
            gpu_used: GPU utilization [0-1]
            duration_s: Duration in seconds
            success: Whether task succeeded
        """
        if self.dreamer:
            try:
                self.dreamer.record_task_execution(
                    task_type=task_type,
                    cpu_used=cpu_used,
                    gpu_used=gpu_used,
                    duration_s=duration_s,
                    success=success
                )
            except Exception as e:
                logger.debug(f"Task recording failed: {e}")

    def set_attention(self, attention: Dict[str, Any]) -> None:
        """
        Set current LLM attention patterns for descending projection.

        Called by the LLM inference loop.
        """
        self._last_attention = attention

    def get_metrics(self) -> Dict[str, Any]:
        """Get current loop metrics."""
        return {
            'loop_count': self.metrics.loop_count,
            'avg_loop_time_ms': round(self.metrics.avg_loop_time_ms, 2),
            'loop_freq_hz': round(1000 / max(0.01, self.metrics.avg_loop_time_ms), 1),
            'descending_packets': self.metrics.descending_packets,
            'ascending_spikes': self.metrics.ascending_spikes,
            'autonomic_updates': self.metrics.autonomic_updates,
            'system_state': self.metrics.last_system_state,
            # Narrative Self metrics
            'narrative_recalls': self.metrics.narrative_recalls,
            'budget_checks': self.metrics.budget_checks,
            'budget_style': self.metrics.last_budget_style,
        }


# =============================================================================
# Global Instance
# =============================================================================

_global_loop: Optional[BicameralLoop] = None


def get_bicameral_loop() -> BicameralLoop:
    """Get the global bicameral loop instance."""
    global _global_loop
    if _global_loop is None:
        _global_loop = BicameralLoop()
    return _global_loop


def start_bicameral_loop(freq_hz: float = 20.0, debug: bool = False) -> BicameralLoop:
    """Start the global bicameral loop."""
    global _global_loop
    if _global_loop is not None:
        _global_loop.stop()
    _global_loop = BicameralLoop(loop_freq_hz=freq_hz, debug=debug)
    _global_loop.start()
    return _global_loop


def stop_bicameral_loop() -> None:
    """Stop the global bicameral loop."""
    global _global_loop
    if _global_loop:
        _global_loop.stop()
        _global_loop = None


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bicameral Loop - Mind-Body Integration Daemon"
    )
    parser.add_argument(
        '--freq', type=float, default=20.0,
        help='Loop frequency in Hz (default: 20)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        stop_bicameral_loop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start loop
    loop = start_bicameral_loop(freq_hz=args.freq, debug=args.debug)

    logger.info("Bicameral Loop daemon started. Press Ctrl+C to stop.")

    # Print metrics periodically
    try:
        while loop.running:
            time.sleep(5.0)
            metrics = loop.get_metrics()
            logger.info(
                f"Loop: {metrics['loop_count']:,} iterations, "
                f"{metrics['avg_loop_time_ms']:.1f}ms avg, "
                f"State: {metrics['system_state']}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        stop_bicameral_loop()


if __name__ == "__main__":
    main()
