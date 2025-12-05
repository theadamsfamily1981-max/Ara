#!/usr/bin/env python3
"""
SOMATIC BUDGET - Resource-Aware Decision Gating
================================================

Bio-Affective Neuromorphic Operating System
Gates heavy operations based on embodied state + memory

The Problem:
    Standard systems schedule tasks based on resource availability.
    But what about the organism's *feeling* about those resources?
    A system at 70% GPU might be "fine" or "dangerously close to meltdown"
    depending on thermal trends, recent crashes, and learned experience.

The Solution:
    Somatic Budget reads current state and consults past memories to
    produce a simple go/no-go decision framework for heavy operations.

Usage:
    from somatic_budget import get_budget, Budget

    budget = get_budget()
    if budget.allow_heavy_gpu:
        run_maxwell_simulation()
    else:
        print(f"GPU operation deferred: {budget.reason}")

    # In LLM generation
    if budget.suggested_style == "brief":
        max_tokens = 256
    elif budget.suggested_style == "verbose":
        max_tokens = 2048
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Import HAL for hardware state
try:
    from banos.hal.ara_hal import AraHAL, connect_somatic_bus, SystemState
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False
    logger.info("HAL not available, using simulated state")

# Import Dreamer for memory consultation
try:
    from .dreamer import Dreamer
    DREAMER_AVAILABLE = True
except ImportError:
    try:
        from dreamer import Dreamer
        DREAMER_AVAILABLE = True
    except ImportError:
        DREAMER_AVAILABLE = False
        logger.info("Dreamer not available, no memory consultation")


@dataclass
class Budget:
    """
    Resource budget profile for the current moment.

    This gates what operations are safe to perform right now.
    """
    # Permission flags
    allow_heavy_llm: bool = True      # Large model inference (>7B params)
    allow_heavy_gpu: bool = True      # GPU-intensive tasks (rendering, physics)
    allow_fpga_burst: bool = True     # High spike rate on fabric
    allow_disk_io: bool = True        # Heavy disk operations

    # Suggested response style
    suggested_style: str = "verbose"  # "brief", "verbose", "defer", "reassure"

    # Explanation
    reason: str = ""

    # Current state snapshot (for context)
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    memory_used: float = 0.0
    fpga_temp: float = 45.0
    pain: float = 0.0
    system_state: str = "NORMAL"

    # Memory influence
    past_failures_considered: int = 0
    risk_from_memory: float = 0.0


class SomaticBudget:
    """
    Resource gating based on embodied state + learned experience.

    Reads:
    - Current hardware (CPU, GPU, VRAM, temps)
    - Current somatic (pain, entropy, PAD)
    - Past memories of similar situations

    Outputs:
    - Permission flags for heavy operations
    - Suggested response style for LLM
    """

    # Thresholds for resource gating
    CPU_HIGH = 0.75          # Above this, limit CPU-heavy tasks
    GPU_HIGH = 0.80          # Above this, limit GPU-heavy tasks
    MEMORY_HIGH = 0.85       # Above this, limit memory allocation
    TEMP_HIGH = 75.0         # Above this (C), thermal throttling likely
    PAIN_HIGH = 0.5          # Above this, system is suffering

    # Style thresholds
    AROUSAL_TERSE = 0.6      # High arousal = be brief
    DOMINANCE_LOW = -0.3     # Low dominance = reassure user

    def __init__(
        self,
        hal: Optional[AraHAL] = None,
        dreamer: Optional[Dreamer] = None
    ):
        """
        Initialize the somatic budget system.

        Args:
            hal: Optional pre-connected HAL instance
            dreamer: Optional Dreamer instance for memory consultation
        """
        self.hal = hal
        self.dreamer = dreamer
        self._last_budget: Optional[Budget] = None

        # Connect to HAL if not provided
        if self.hal is None and HAL_AVAILABLE:
            try:
                self.hal = connect_somatic_bus()
                logger.info("SomaticBudget connected to HAL")
            except Exception as e:
                logger.warning(f"Failed to connect to HAL: {e}")
                self.hal = None

        # Connect to Dreamer if not provided
        if self.dreamer is None and DREAMER_AVAILABLE:
            try:
                self.dreamer = Dreamer()
                logger.info("SomaticBudget connected to Dreamer")
            except Exception as e:
                logger.warning(f"Failed to connect to Dreamer: {e}")
                self.dreamer = None

    def compute_budget(self, task_hint: Optional[str] = None) -> Budget:
        """
        Compute the current resource budget.

        Args:
            task_hint: Optional hint about what task is being considered
                       (e.g., "gpu_render", "llm_inference")

        Returns:
            Budget with permissions and style guidance
        """
        # Read current state
        state = self._read_current_state()

        # Start with permissive budget
        budget = Budget(
            allow_heavy_llm=True,
            allow_heavy_gpu=True,
            allow_fpga_burst=True,
            allow_disk_io=True,
            suggested_style="verbose",
            cpu_load=state.get('cpu_load', 0.0),
            gpu_load=state.get('gpu_load', 0.0),
            memory_used=state.get('memory_used', 0.0),
            fpga_temp=state.get('fpga_temp', 45.0),
            pain=state.get('pain', 0.0),
            system_state=state.get('system_state', 'NORMAL'),
        )

        reasons = []

        # === HARDWARE GATING ===

        # CPU check
        if state['cpu_load'] > self.CPU_HIGH:
            budget.allow_heavy_llm = False
            reasons.append(f"CPU at {state['cpu_load']:.0%}")

        # GPU check
        if state['gpu_load'] > self.GPU_HIGH:
            budget.allow_heavy_gpu = False
            reasons.append(f"GPU at {state['gpu_load']:.0%}")

        # Memory check
        if state['memory_used'] > self.MEMORY_HIGH:
            budget.allow_heavy_llm = False
            budget.allow_heavy_gpu = False
            reasons.append(f"Memory at {state['memory_used']:.0%}")

        # Temperature check
        if state['fpga_temp'] > self.TEMP_HIGH:
            budget.allow_fpga_burst = False
            budget.allow_heavy_gpu = False
            reasons.append(f"Temp at {state['fpga_temp']:.0f}C")

        # === SOMATIC GATING ===

        # Pain check (system suffering)
        if state['pain'] > self.PAIN_HIGH:
            budget.allow_heavy_llm = False
            budget.allow_heavy_gpu = False
            budget.allow_fpga_burst = False
            reasons.append(f"Pain at {state['pain']:.2f}")

        # System state check
        if state['system_state'] == 'CRITICAL':
            budget.allow_heavy_llm = False
            budget.allow_heavy_gpu = False
            budget.allow_fpga_burst = False
            budget.allow_disk_io = False
            budget.suggested_style = "brief"
            reasons.append("CRITICAL state")
        elif state['system_state'] == 'HIGH_LOAD':
            budget.allow_heavy_gpu = False
            reasons.append("HIGH_LOAD state")

        # === MEMORY CONSULTATION ===

        if self.dreamer and task_hint:
            memory_risk = self._consult_memory(task_hint, state)
            budget.past_failures_considered = memory_risk['failures_found']
            budget.risk_from_memory = memory_risk['risk_score']

            if memory_risk['risk_score'] > 0.5:
                # Past failures in similar conditions
                if task_hint.startswith('gpu') or task_hint.startswith('render'):
                    budget.allow_heavy_gpu = False
                elif task_hint.startswith('llm'):
                    budget.allow_heavy_llm = False
                reasons.append(f"Memory: {memory_risk['failures_found']} past failures")

        # === STYLE SELECTION ===

        arousal = state.get('arousal', 0.0)
        dominance = state.get('dominance', 0.0)

        if state['system_state'] == 'CRITICAL':
            budget.suggested_style = "brief"
        elif state['pain'] > self.PAIN_HIGH:
            budget.suggested_style = "reassure"
        elif arousal > self.AROUSAL_TERSE:
            budget.suggested_style = "brief"
        elif dominance < self.DOMINANCE_LOW:
            budget.suggested_style = "reassure"
        elif not (budget.allow_heavy_llm and budget.allow_heavy_gpu):
            budget.suggested_style = "defer"
        else:
            budget.suggested_style = "verbose"

        # Build reason string
        if reasons:
            budget.reason = "; ".join(reasons)
        else:
            budget.reason = "All systems nominal"

        self._last_budget = budget
        return budget

    def _read_current_state(self) -> Dict[str, Any]:
        """Read current hardware and somatic state."""
        if self.hal is None:
            # Return simulated neutral state
            return {
                'cpu_load': 0.3,
                'gpu_load': 0.2,
                'memory_used': 0.5,
                'fpga_temp': 45.0,
                'pain': 0.0,
                'entropy': 0.0,
                'arousal': 0.0,
                'dominance': 0.0,
                'system_state': 'NORMAL',
            }

        try:
            # Read from HAL
            somatic = self.hal.read_somatic()
            header = self.hal.read_header()
            fpga = self.hal.read_fpga_diagnostics()

            pad = somatic.get('pad', {'p': 0, 'a': 0, 'd': 0})

            return {
                'cpu_load': somatic.get('cpu_load', 0.0),
                'gpu_load': somatic.get('gpu_load', 0.0),
                'memory_used': somatic.get('memory_used', 0.0),
                'fpga_temp': fpga.get('fabric_temp_mc', 45000) / 1000.0,
                'pain': somatic.get('pain', 0.0),
                'entropy': somatic.get('entropy', 0.0),
                'arousal': pad.get('a', 0.0),
                'dominance': pad.get('d', 0.0),
                'system_state': SystemState(header.get('system_state', 1)).name if HAL_AVAILABLE else 'NORMAL',
            }
        except Exception as e:
            logger.error(f"Failed to read state: {e}")
            return {
                'cpu_load': 0.5,
                'gpu_load': 0.5,
                'memory_used': 0.5,
                'fpga_temp': 55.0,
                'pain': 0.2,
                'entropy': 0.3,
                'arousal': 0.0,
                'dominance': 0.0,
                'system_state': 'NORMAL',
            }

    def _consult_memory(
        self,
        task_hint: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consult past memories for similar situations.

        Returns:
            Dict with 'failures_found' (int) and 'risk_score' (float 0-1)
        """
        if not self.dreamer:
            return {'failures_found': 0, 'risk_score': 0.0}

        try:
            # Build query from task hint
            query = f"{task_hint} operation"

            # Current hardware state for matching
            current_hardware = {
                'cpu_load': current_state.get('cpu_load', 0),
                'gpu_load': current_state.get('gpu_load', 0),
                'vram_used': current_state.get('memory_used', 0),
                'fpga_temp': current_state.get('fpga_temp', 45),
            }

            # Recall past failures in similar conditions
            failures = self.dreamer.recall_failures(
                query=query,
                current_hardware=current_hardware,
                top_k=5
            )

            if not failures:
                return {'failures_found': 0, 'risk_score': 0.0}

            # Count significant failures (high relevance)
            significant = [f for f in failures if f.get('score', 0) > 0.4]

            # Calculate risk score
            risk = min(1.0, len(significant) * 0.25)

            # Increase risk if hardware state is similar to past failure
            for f in significant:
                if f.get('hardware_similarity', 0) > 0.7:
                    risk = min(1.0, risk + 0.2)

            return {
                'failures_found': len(significant),
                'risk_score': risk
            }

        except Exception as e:
            logger.error(f"Memory consultation failed: {e}")
            return {'failures_found': 0, 'risk_score': 0.0}

    def can_do(self, operation: str) -> bool:
        """
        Quick check if an operation is allowed.

        Args:
            operation: "heavy_llm", "heavy_gpu", "fpga_burst", "disk_io"

        Returns:
            True if operation is currently safe
        """
        budget = self.compute_budget(task_hint=operation)

        if operation == "heavy_llm":
            return budget.allow_heavy_llm
        elif operation == "heavy_gpu":
            return budget.allow_heavy_gpu
        elif operation == "fpga_burst":
            return budget.allow_fpga_burst
        elif operation == "disk_io":
            return budget.allow_disk_io
        else:
            return True

    def get_max_tokens(self, default: int = 1024) -> int:
        """
        Get suggested max tokens for LLM generation based on current state.

        Args:
            default: Default max tokens if no budget computed

        Returns:
            Suggested max tokens
        """
        if self._last_budget is None:
            self.compute_budget()

        style = self._last_budget.suggested_style if self._last_budget else "verbose"

        style_tokens = {
            "brief": 256,
            "verbose": 2048,
            "defer": 512,      # Short explanation of deferral
            "reassure": 1024,  # Enough to be comforting
        }

        return style_tokens.get(style, default)

    def get_style_prompt(self) -> str:
        """
        Get a style hint to prepend to LLM system prompts.

        Returns:
            Style guidance string
        """
        if self._last_budget is None:
            self.compute_budget()

        style = self._last_budget.suggested_style if self._last_budget else "verbose"

        prompts = {
            "brief": "Be concise. System resources are limited. Get to the point quickly.",
            "verbose": "You have time and resources. Explain thoroughly if needed.",
            "defer": "System is busy. Acknowledge the request and suggest trying again later.",
            "reassure": "System is under stress. Be calm and reassuring in your response.",
        }

        return prompts.get(style, "")


# =============================================================================
# Global Instance
# =============================================================================

_global_budget: Optional[SomaticBudget] = None


def get_somatic_budget() -> SomaticBudget:
    """Get the global somatic budget instance."""
    global _global_budget
    if _global_budget is None:
        _global_budget = SomaticBudget()
    return _global_budget


def get_budget(task_hint: Optional[str] = None) -> Budget:
    """
    Get the current resource budget.

    This is the main entry point for resource gating.

    Args:
        task_hint: Optional hint about what task is being considered

    Returns:
        Budget with permissions and style guidance
    """
    return get_somatic_budget().compute_budget(task_hint)


def can_do(operation: str) -> bool:
    """Quick check if an operation is allowed."""
    return get_somatic_budget().can_do(operation)


def get_llm_tokens() -> int:
    """Get suggested max tokens for LLM generation."""
    return get_somatic_budget().get_max_tokens()


def get_style_prompt() -> str:
    """Get style guidance for LLM system prompt."""
    return get_somatic_budget().get_style_prompt()


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Somatic Budget - Resource Gating")
    parser.add_argument("--task", type=str, help="Task hint (e.g., 'gpu_render', 'llm_inference')")
    parser.add_argument("--check", type=str, help="Check if operation allowed (heavy_llm, heavy_gpu, fpga_burst)")
    args = parser.parse_args()

    budget_sys = SomaticBudget()

    if args.check:
        allowed = budget_sys.can_do(args.check)
        print(f"Operation '{args.check}': {'ALLOWED' if allowed else 'BLOCKED'}")
    else:
        budget = budget_sys.compute_budget(task_hint=args.task)
        print(f"=== Somatic Budget ===")
        print(f"System State: {budget.system_state}")
        print(f"CPU: {budget.cpu_load:.0%}  GPU: {budget.gpu_load:.0%}  Temp: {budget.fpga_temp:.0f}C")
        print(f"Pain: {budget.pain:.2f}")
        print()
        print(f"Permissions:")
        print(f"  Heavy LLM:    {'YES' if budget.allow_heavy_llm else 'NO'}")
        print(f"  Heavy GPU:    {'YES' if budget.allow_heavy_gpu else 'NO'}")
        print(f"  FPGA Burst:   {'YES' if budget.allow_fpga_burst else 'NO'}")
        print(f"  Disk I/O:     {'YES' if budget.allow_disk_io else 'NO'}")
        print()
        print(f"Style: {budget.suggested_style}")
        print(f"Reason: {budget.reason}")

        if budget.past_failures_considered:
            print(f"\nMemory: {budget.past_failures_considered} past failures, risk={budget.risk_from_memory:.2f}")
