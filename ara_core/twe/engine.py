#!/usr/bin/env python3
"""
Teleoplastic Weave Engine - Main Controller
============================================

The TWE is a closed-loop, criticality-tuned organ manufacturing platform:

    Meta-Brain (designs) → CADD (safety gate) → Kitten Fabric (control) → Hardware (print)

GUTC Integration:
    - λ (criticality) tracked in both tissue and control fabric
    - Π (precision) fields balance blueprint vs. viability
    - Near-critical operation (λ ≈ 1) for robust, adaptive printing

Control Stack:
    1. Strategic (Meta-Brain): Design organs, run simulations
    2. Tactical (CADD + Supervisor): Safety gate, job management
    3. Reflexive (Kitten Fabric): Real-time closed-loop control

Usage:
    from ara_core.twe import TeleoplasticWeaveEngine
    from ara_core.cadd import CADDSentinel

    sentinel = CADDSentinel()
    twe = TeleoplasticWeaveEngine(sentinel=sentinel)

    # Design organ with safety checks
    blueprint = twe.plan_blueprint(patient_profile)

    # Print with closed-loop control
    twe.print_organ(blueprint)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto

from .blueprint import OrganBlueprint, PrintLayer, create_demo_vascular_patch
from .fabric import KittenFabric, SensorGrid, ActuatorCommand
from .hardware import TWEHardware, SimulatedHardware, PrinterState

# Import CADD if available
try:
    from ..cadd import CADDSentinel, DriftType, DriftAlert
    CADD_AVAILABLE = True
except ImportError:
    CADD_AVAILABLE = False
    CADDSentinel = None


# =============================================================================
# Print Job State
# =============================================================================

class JobState(Enum):
    """State of a print job."""
    PENDING = auto()
    PLANNING = auto()
    SAFETY_CHECK = auto()
    PRINTING = auto()
    PAUSED = auto()
    REPLANNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class PrintJob:
    """A print job with full state tracking."""
    job_id: str
    blueprint: OrganBlueprint
    state: JobState = JobState.PENDING

    # Progress
    current_layer: int = 0
    total_layers: int = 0
    layers_completed: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Health tracking
    mean_viability_history: List[float] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=list)
    alerts_triggered: List[Any] = field(default_factory=list)

    # Outcome
    success: bool = False
    failure_reason: Optional[str] = None

    def duration_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def progress_fraction(self) -> float:
        if self.total_layers == 0:
            return 0.0
        return self.layers_completed / self.total_layers


# =============================================================================
# Meta-Brain Interface (Simplified)
# =============================================================================

class MetaBrainInterface:
    """
    Interface to the Meta-Brain for organ design.

    In production, this would connect to the full Meta-Brain system.
    Here we provide a simplified simulation.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.design_history: List[OrganBlueprint] = []

    def generate_organs(
        self,
        patient_profile: Dict[str, Any],
        n_candidates: int = 5,
    ) -> List[OrganBlueprint]:
        """
        Generate candidate organ blueprints.

        In production: runs generative models with patient-specific constraints.
        Here: generates demo patches with variation.
        """
        if self.verbose:
            print(f"[MetaBrain] Generating {n_candidates} candidate organs...")

        candidates = []
        for i in range(n_candidates):
            # Vary parameters to create diversity
            size_mm = 8.0 + i * 2.0
            resolution_um = 100.0 - i * 10.0

            bp = create_demo_vascular_patch(
                size_mm=size_mm,
                resolution_um=max(50, resolution_um),
            )
            bp.patient_id = patient_profile.get("patient_id", "unknown")

            # Simulate different robustness scores
            bp.stress_margin = 0.2 + 0.1 * i
            bp.perfusion_tolerance = 0.15 + 0.05 * i
            bp.viability_score = 0.7 + 0.05 * i

            candidates.append(bp)

        return candidates

    def evaluate_blueprint(self, blueprint: OrganBlueprint) -> Dict[str, float]:
        """
        Evaluate a blueprint via simulation.

        Returns scores for various criteria.
        """
        # In production: run tissue mechanics, perfusion, etc.
        # Here: return pre-computed scores with some noise

        return {
            "stress_margin": blueprint.stress_margin + 0.05 * (0.5 - np.random.random()),
            "perfusion_tolerance": blueprint.perfusion_tolerance + 0.03 * (0.5 - np.random.random()),
            "viability_score": blueprint.viability_score + 0.02 * (0.5 - np.random.random()),
            "manufacturability": 0.8 + 0.1 * np.random.random(),
            "immunological_match": 0.9 + 0.05 * np.random.random(),
        }

    def select_pareto(
        self,
        scored_blueprints: List[tuple],
    ) -> OrganBlueprint:
        """
        Select best blueprint from Pareto front.

        scored_blueprints: List of (blueprint, scores) tuples
        """
        if not scored_blueprints:
            raise ValueError("No blueprints to select from")

        # Simple weighted sum selection (in production: proper Pareto)
        best_score = -float('inf')
        best_bp = None

        weights = {
            "stress_margin": 0.25,
            "perfusion_tolerance": 0.25,
            "viability_score": 0.3,
            "manufacturability": 0.1,
            "immunological_match": 0.1,
        }

        for bp, scores in scored_blueprints:
            total = sum(scores.get(k, 0) * v for k, v in weights.items())
            if total > best_score:
                best_score = total
                best_bp = bp

        if self.verbose:
            print(f"[MetaBrain] Selected blueprint: {best_bp.blueprint_id}")
            print(f"            Score: {best_score:.3f}")

        self.design_history.append(best_bp)
        return best_bp

    def detect_global_risk(
        self,
        sensor_state: SensorGrid,
        blueprint: OrganBlueprint,
    ) -> bool:
        """
        Detect if the print is in a globally risky state.

        Returns True if intervention is needed.
        """
        mean_viab = sensor_state.mean_viability()

        # Critical viability threshold
        if mean_viab < 0.4:
            if self.verbose:
                print(f"[MetaBrain] GLOBAL RISK: viability={mean_viab:.2f}")
            return True

        return False

    def replan_local_patch(
        self,
        blueprint: OrganBlueprint,
        problem_region: Optional[tuple] = None,
    ) -> OrganBlueprint:
        """
        Modify blueprint to address local issues.

        In production: sophisticated local replanning.
        Here: just logs and returns same blueprint.
        """
        if self.verbose:
            print("[MetaBrain] Replanning local patch...")
        return blueprint


# Import numpy for MetaBrain
import numpy as np


# =============================================================================
# Teleoplastic Weave Engine
# =============================================================================

class TeleoplasticWeaveEngine:
    """
    Teleoplastic Weave Engine - Closed-loop organ printer.

    Integrates:
    - Meta-Brain: Organ design and simulation
    - CADD Sentinel: Safety gating
    - Kitten Fabric: Real-time SNN control
    - Hardware: Physical printer interface
    """

    def __init__(
        self,
        meta_brain: Optional[MetaBrainInterface] = None,
        sentinel: Optional['CADDSentinel'] = None,
        fabric: Optional[KittenFabric] = None,
        hardware: Optional[TWEHardware] = None,
        verbose: bool = True,
    ):
        """
        Initialize the TWE.

        Args:
            meta_brain: Blueprint designer (default: create new)
            sentinel: CADD safety sentinel (optional)
            fabric: Kitten Fabric controller (default: create new)
            hardware: Hardware interface (default: simulated)
            verbose: Print status messages
        """
        self.verbose = verbose

        # Initialize components
        self.meta_brain = meta_brain or MetaBrainInterface(verbose=verbose)
        self.sentinel = sentinel
        self.fabric = fabric or KittenFabric(verbose=verbose)
        self.hardware = hardware or SimulatedHardware(verbose=verbose)

        # Job management
        self.current_job: Optional[PrintJob] = None
        self.job_history: List[PrintJob] = []

        # State
        self.initialized = False

        if self.verbose:
            print("[TWE] Teleoplastic Weave Engine created")
            print(f"      CADD Sentinel: {'enabled' if sentinel else 'disabled'}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize all subsystems."""
        if self.verbose:
            print("\n[TWE] Initializing subsystems...")

        # Hardware
        if not self.hardware.initialize():
            print("[TWE] ERROR: Hardware initialization failed")
            return False

        # Home axes
        if not self.hardware.home():
            print("[TWE] ERROR: Homing failed")
            return False

        # Fabric
        if not self.fabric.connect():
            print("[TWE] ERROR: Fabric connection failed")
            return False

        self.initialized = True
        if self.verbose:
            print("[TWE] Initialization complete ✓")

        return True

    def shutdown(self) -> None:
        """Shutdown all subsystems safely."""
        if self.verbose:
            print("\n[TWE] Shutting down...")

        if self.current_job and self.current_job.state == JobState.PRINTING:
            self.hardware.pause_print()

        self.fabric.disconnect()
        self.hardware.shutdown()
        self.initialized = False

        if self.verbose:
            print("[TWE] Shutdown complete")

    # ------------------------------------------------------------------
    # Blueprint Planning
    # ------------------------------------------------------------------

    def plan_blueprint(
        self,
        patient_profile: Dict[str, Any],
        n_candidates: int = 5,
    ) -> OrganBlueprint:
        """
        Plan an organ blueprint with safety checks.

        1. Generate candidate organs via Meta-Brain
        2. Evaluate robustness via simulation
        3. Check design diversity via CADD
        4. Select best from Pareto front

        Args:
            patient_profile: Patient-specific constraints
            n_candidates: Number of candidates to generate

        Returns:
            Selected OrganBlueprint
        """
        if self.verbose:
            print("\n[TWE] Planning blueprint...")

        # 1. Generate candidates
        candidates = self.meta_brain.generate_organs(patient_profile, n_candidates)

        # 2. Evaluate each
        scored = []
        for bp in candidates:
            scores = self.meta_brain.evaluate_blueprint(bp)
            scored.append((bp, scores))

            # 3. Update CADD with design diversity info
            if self.sentinel:
                # Record design features as associations
                self.sentinel.update_association(
                    agent_id=f"design_{bp.blueprint_id[:8]}",
                    concept=bp.organ_type,
                    signal="scaffold_type",
                    strength=scores["manufacturability"],
                )
                self.sentinel.update_association(
                    agent_id=f"design_{bp.blueprint_id[:8]}",
                    concept=bp.organ_type,
                    signal="vascular_pattern",
                    strength=scores["perfusion_tolerance"],
                )

        # 4. Check CADD for monoculture
        if self.sentinel:
            alerts = self.sentinel.tick()
            blocking_alerts = [a for a in alerts if a.drift_type == DriftType.MONOCULTURE]

            if blocking_alerts:
                if self.verbose:
                    print("[TWE] CADD ALERT: Design monoculture detected!")
                    for alert in blocking_alerts:
                        print(f"      {alert.message}")
                raise RuntimeError("Design monoculture detected. Abort planning.")

        # 5. Select best
        blueprint = self.meta_brain.select_pareto(scored)

        if self.verbose:
            print(f"[TWE] Blueprint selected: {blueprint.blueprint_id}")
            print(f"      Organ: {blueprint.organ_type}")
            print(f"      Volume: {blueprint.volume_mm3():.1f} mm³")
            print(f"      Vessels: {len(blueprint.vasculature.edges)}")

        return blueprint

    # ------------------------------------------------------------------
    # Print Execution
    # ------------------------------------------------------------------

    def print_organ(self, blueprint: OrganBlueprint) -> PrintJob:
        """
        Print an organ with full safety and closed-loop control.

        Control loop:
        1. Safety gate (CADD check)
        2. Prepare chamber
        3. For each layer:
           a. Read sensors
           b. Estimate λ
           c. Update Π
           d. Compute actuator commands
           e. Apply commands
           f. Check for global risk → replan if needed

        Args:
            blueprint: Organ blueprint to print

        Returns:
            PrintJob with outcome
        """
        if not self.initialized:
            raise RuntimeError("TWE not initialized. Call initialize() first.")

        # Create job
        job = PrintJob(
            job_id=f"job_{int(time.time())}",
            blueprint=blueprint,
            total_layers=len(blueprint.layers()),
        )
        self.current_job = job

        if self.verbose:
            print(f"\n[TWE] Starting print job: {job.job_id}")
            print(f"      Blueprint: {blueprint.blueprint_id}")
            print(f"      Layers: {job.total_layers}")

        try:
            # Safety gate
            job.state = JobState.SAFETY_CHECK
            if not self._safety_gate(blueprint):
                job.state = JobState.FAILED
                job.failure_reason = "CADD safety gate failed"
                return self._finalize_job(job)

            # Prepare chamber
            self.hardware.prepare_chamber(blueprint)

            # Start print
            job.state = JobState.PRINTING
            job.started_at = time.time()

            # Get layers
            layers = blueprint.layers()

            # Main print loop
            for layer_idx, layer in enumerate(layers):
                job.current_layer = layer_idx

                if self.verbose and layer_idx % 10 == 0:
                    print(f"[TWE] Layer {layer_idx}/{job.total_layers} "
                          f"({job.progress_fraction():.1%})")

                # Core control step
                success = self._print_layer(layer, blueprint, job)

                if not success:
                    if job.state == JobState.PAUSED:
                        # Attempt recovery
                        if self.verbose:
                            print("[TWE] Attempting recovery...")
                        self.hardware.resume_print()
                        job.state = JobState.PRINTING
                    else:
                        job.state = JobState.FAILED
                        job.failure_reason = "Layer print failed"
                        break

                job.layers_completed += 1

                # Check for abort conditions
                if job.state == JobState.ABORTED:
                    break

            # Finalize
            if job.state == JobState.PRINTING:
                job.state = JobState.COMPLETED
                job.success = True

            self.hardware.finalize_print()

        except Exception as e:
            job.state = JobState.FAILED
            job.failure_reason = str(e)
            self.hardware.abort_print()

        return self._finalize_job(job)

    def _safety_gate(self, blueprint: OrganBlueprint) -> bool:
        """Run CADD safety check before printing."""
        if self.sentinel is None:
            return True  # No sentinel, skip check

        if self.verbose:
            print("[TWE] Running CADD safety gate...")

        alerts = self.sentinel.tick()
        status = self.sentinel.health_status()

        if self.verbose:
            print(f"      H_influence: {status['h_influence']:.3f}")

        monoculture = [a for a in alerts if a.drift_type == DriftType.MONOCULTURE]
        if monoculture:
            if self.verbose:
                print("[TWE] BLOCKED: Monoculture detected in design ecosystem")
            return False

        if self.verbose:
            print("[TWE] Safety gate PASSED ✓")
        return True

    def _print_layer(
        self,
        layer: PrintLayer,
        blueprint: OrganBlueprint,
        job: PrintJob,
    ) -> bool:
        """
        Print a single layer with closed-loop control.

        Returns True if successful, False if intervention needed.
        """
        # 1. Read sensors
        sensor_state = self.hardware.read_sensors()

        # 2. Estimate λ
        lambda_hat = self.fabric.estimate_lambda(sensor_state)
        job.lambda_history.append(lambda_hat)

        # 3. Update Π
        pi_sens, pi_prior = self.fabric.update_precision(
            sensor_state, blueprint, lambda_hat
        )

        # 4. Compute actuator commands
        commands = self.fabric.control_step(
            sensor_state, layer, lambda_hat, pi_sens, pi_prior
        )

        # 5. Apply commands
        self.hardware.apply_commands(commands)

        # Record viability
        job.mean_viability_history.append(sensor_state.mean_viability())

        # 6. Check for pause/abort
        if commands.pause_requested:
            self.hardware.pause_print()
            job.state = JobState.PAUSED
            return False

        if commands.abort_requested:
            job.state = JobState.ABORTED
            return False

        # 7. Check for global risk (Meta-Brain)
        if self.meta_brain.detect_global_risk(sensor_state, blueprint):
            self.hardware.pause_print()
            job.state = JobState.REPLANNING

            # Attempt replanning
            new_blueprint = self.meta_brain.replan_local_patch(blueprint)
            # In production: would update remaining layers

            self.hardware.resume_print()
            job.state = JobState.PRINTING

        return True

    def _finalize_job(self, job: PrintJob) -> PrintJob:
        """Finalize a job and store in history."""
        job.completed_at = time.time()
        self.job_history.append(job)
        self.current_job = None

        if self.verbose:
            status = "✓ SUCCESS" if job.success else f"✗ FAILED: {job.failure_reason}"
            print(f"\n[TWE] Job {job.job_id} complete: {status}")
            if job.duration_s():
                print(f"      Duration: {job.duration_s():.1f}s")
            print(f"      Layers: {job.layers_completed}/{job.total_layers}")
            if job.mean_viability_history:
                print(f"      Mean viability: {np.mean(job.mean_viability_history):.2f}")
            if job.lambda_history:
                print(f"      Mean λ: {np.mean(job.lambda_history):.3f}")

        return job

    # ------------------------------------------------------------------
    # Status and Reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get complete TWE status."""
        return {
            "initialized": self.initialized,
            "hardware_state": self.hardware.get_state().name,
            "fabric": self.fabric.get_status(),
            "current_job": {
                "job_id": self.current_job.job_id if self.current_job else None,
                "state": self.current_job.state.name if self.current_job else None,
                "progress": self.current_job.progress_fraction() if self.current_job else 0,
            },
            "total_jobs": len(self.job_history),
            "successful_jobs": sum(1 for j in self.job_history if j.success),
        }

    def status_string(self) -> str:
        """Get formatted status string."""
        status = self.get_status()

        hw_emoji = {
            "IDLE": "⚪",
            "READY": "🟢",
            "PRINTING": "🔵",
            "PAUSED": "🟡",
            "ERROR": "🔴",
        }.get(status["hardware_state"], "⚪")

        lines = [
            "╔══════════════════════════════════════╗",
            "║    TELEOPLASTIC WEAVE ENGINE         ║",
            "╠══════════════════════════════════════╣",
            f"║ Hardware: {hw_emoji} {status['hardware_state']:<26}║",
            f"║ Fabric λ̂: {status['fabric']['lambda_hat']:.3f}                       ║",
            f"║ Viability: {status['fabric']['mean_viability']:.2f}                      ║",
        ]

        if status["current_job"]["job_id"]:
            lines.append(f"║ Job: {status['current_job']['state']:<30}║")
            lines.append(f"║ Progress: {status['current_job']['progress']:.1%}                       ║")

        lines.append(f"║ Jobs: {status['successful_jobs']}/{status['total_jobs']} successful              ║")
        lines.append("╚══════════════════════════════════════╝")

        return "\n".join(lines)


# =============================================================================
# Convenience: Quick Demo
# =============================================================================

def run_demo_print(verbose: bool = True) -> PrintJob:
    """
    Run a demo print with simulated hardware.

    Shows the full TWE pipeline: design → safety → print.
    """
    # Create sentinel if available
    sentinel = None
    if CADD_AVAILABLE:
        from ..cadd import CADDSentinel, SentinelConfig
        config = SentinelConfig(h_influence_min=0.3)  # Relaxed for demo
        sentinel = CADDSentinel(config=config)

    # Create TWE
    twe = TeleoplasticWeaveEngine(sentinel=sentinel, verbose=verbose)

    # Initialize
    twe.initialize()

    # Plan blueprint
    patient = {"patient_id": "demo_patient_001", "organ_needed": "vascular_patch"}
    blueprint = twe.plan_blueprint(patient, n_candidates=3)

    # Print
    job = twe.print_organ(blueprint)

    # Cleanup
    twe.shutdown()

    return job
