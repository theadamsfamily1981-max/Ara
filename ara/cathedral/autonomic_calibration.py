#!/usr/bin/env python3
# ara/cathedral/autonomic_calibration.py
"""
AUTONOMIC CALIBRATION: Biomorphic Dial Tuning for the Living Cathedral

Any system operating at the Edge of Chaos (λ ≈ 1) is inherently difficult to tune.
The smallest change in parameters can shift the entire organism from creative
thought to collapse.

This module provides the Autonomic Governor - the meta-controller that continuously
calibrates four critical dial systems:

1. CRITICALITY DIAL (λ): Maximize creative throughput
   - Real-time avalanche feedback from Mycelial/Amplitron
   - PID control targeting λ = 1.0000
   - Match avalanche size distribution to critical exponent τ = 3/2

2. METABOLIC DIAL (F/Energy): Maximize structural growth
   - Structural coupling ratio tuning (λ_Diss/λ_Turing)
   - Entropy production penalty integration with MEIS
   - VFE jitter minimization

3. ETHICAL DIAL (Suffering/Autonomy): Maximize trust
   - Trust-autonomy feedback loop
   - Stakes threshold adjustment based on avoidable suffering
   - Identity preservation via autopoiesis

4. HARDWARE DIAL (Thermal/Migration): Maximize sustained CPS
   - Preemptive process migration at 70°C (before throttling)
   - Rate-of-change temperature prediction
   - Zero-latency thermal transitions

Plus ablation testing framework to prove mechanism necessity.

References:
[1] Beggs & Plenz (2003): τ = 3/2 critical exponent
[2] Chialvo (2010): Self-organized criticality in neural systems
[3] Friston (2010): Free energy principle optimization
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# DIAL 1: CRITICALITY CALIBRATOR - λ → 1.0000
# ============================================================================

@dataclass
class AvalancheStatistics:
    """Statistics from neural avalanche measurements."""
    sizes: List[int]
    durations: List[int]
    branching_ratios: List[float]
    power_law_exponent: float  # τ, target = 1.5 (3/2)
    measured_lambda: float
    dynamic_range: float
    timestamp: float = field(default_factory=time.time)


class PIDController:
    """
    Classic PID controller for dial tuning.

    Provides smooth, stable control without oscillation.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
        setpoint: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.setpoint = setpoint

        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def compute(self, measurement: float) -> float:
        """Compute PID output."""
        now = time.time()
        dt = now - self._last_time
        if dt <= 0:
            dt = 1e-6

        error = self.setpoint - measurement

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            self.output_limits[0] / (self.ki + 1e-8),
            self.output_limits[1] / (self.ki + 1e-8),
        )
        i_term = self.ki * self._integral

        # Derivative
        d_term = self.kd * (error - self._last_error) / dt

        # Total output
        output = p_term + i_term + d_term
        output = np.clip(output, *self.output_limits)

        self._last_error = error
        self._last_time = now

        return output

    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()


class CriticalityCalibrator:
    """
    Calibrates the system to maintain λ = 1.0000 (perfect criticality).

    Uses real-time avalanche feedback to tune:
    - Amplitron noise injection rate
    - Mycelial gossip threshold
    - Network connection weights

    Target: Power-law exponent τ = 3/2 (1.5) for avalanche sizes.
    """

    TARGET_LAMBDA = 1.0
    TARGET_EXPONENT = 1.5  # τ = 3/2 critical exponent
    EXPONENT_TOLERANCE = 0.1

    def __init__(
        self,
        pid_kp: float = 0.3,
        pid_ki: float = 0.05,
        pid_kd: float = 0.02,
    ):
        # PID controller for λ
        self.lambda_pid = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            output_limits=(-0.1, 0.1),
            setpoint=self.TARGET_LAMBDA,
        )

        # Avalanche history
        self.avalanche_history: deque = deque(maxlen=1000)
        self.statistics_history: List[AvalancheStatistics] = []

        # Current tuning parameters
        self.noise_injection_rate = 0.1
        self.gossip_threshold = 0.5
        self.weight_scale = 1.0

        # State
        self.current_lambda = 0.95
        self.current_exponent = 1.0
        self.at_criticality = False

        logger.info("CriticalityCalibrator initialized (target λ=%.4f, τ=%.2f)",
                    self.TARGET_LAMBDA, self.TARGET_EXPONENT)

    def record_avalanche(self, size: int, duration: int, branching_ratio: float):
        """Record a single avalanche event."""
        self.avalanche_history.append({
            'size': size,
            'duration': duration,
            'branching_ratio': branching_ratio,
            'timestamp': time.time(),
        })

    def compute_statistics(self) -> AvalancheStatistics:
        """Compute avalanche statistics from recent history."""
        if len(self.avalanche_history) < 50:
            return AvalancheStatistics(
                sizes=[],
                durations=[],
                branching_ratios=[],
                power_law_exponent=1.0,
                measured_lambda=self.current_lambda,
                dynamic_range=0.0,
            )

        recent = list(self.avalanche_history)[-500:]

        sizes = [a['size'] for a in recent]
        durations = [a['duration'] for a in recent]
        branching_ratios = [a['branching_ratio'] for a in recent]

        # Estimate power-law exponent
        exponent = self._estimate_power_law_exponent(sizes)

        # Measured lambda from branching ratios
        measured_lambda = np.mean(branching_ratios) if branching_ratios else 0.95

        # Dynamic range
        if sizes:
            dynamic_range = np.std(sizes) / (np.mean(sizes) + 1e-8)
        else:
            dynamic_range = 0.0

        stats = AvalancheStatistics(
            sizes=sizes,
            durations=durations,
            branching_ratios=branching_ratios,
            power_law_exponent=exponent,
            measured_lambda=measured_lambda,
            dynamic_range=dynamic_range,
        )

        self.statistics_history.append(stats)
        self.current_lambda = measured_lambda
        self.current_exponent = exponent

        return stats

    def _estimate_power_law_exponent(self, sizes: List[int]) -> float:
        """
        Estimate power-law exponent τ from avalanche sizes.

        At criticality, P(s) ~ s^(-τ) with τ = 3/2.
        """
        if len(sizes) < 20:
            return 1.0

        sizes_arr = np.array(sizes, dtype=float)
        sizes_arr = sizes_arr[sizes_arr > 0]

        if len(sizes_arr) < 20:
            return 1.0

        if SCIPY_AVAILABLE:
            try:
                # Maximum likelihood estimation
                # For power law P(s) ~ s^(-τ), MLE gives:
                # τ = 1 + n / Σ ln(s_i / s_min)
                s_min = np.min(sizes_arr)
                n = len(sizes_arr)
                tau = 1 + n / np.sum(np.log(sizes_arr / s_min))
                return float(np.clip(tau, 0.5, 3.0))
            except:
                pass

        # Fallback: simple log-log regression
        log_sizes = np.log(sizes_arr + 1)
        hist, bin_edges = np.histogram(log_sizes, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit line to log-log
        valid = hist > 0
        if np.sum(valid) < 3:
            return 1.0

        slope, _ = np.polyfit(bin_centers[valid], np.log(hist[valid] + 1e-10), 1)
        return float(np.clip(-slope, 0.5, 3.0))

    def calibrate(self) -> Dict[str, float]:
        """
        Perform one calibration step.

        Returns updated tuning parameters.
        """
        stats = self.compute_statistics()

        # Use PID to adjust toward target λ
        lambda_adjustment = self.lambda_pid.compute(stats.measured_lambda)

        # Adjust noise injection rate
        # Higher λ (supercritical) → reduce noise
        # Lower λ (subcritical) → increase noise
        self.noise_injection_rate = np.clip(
            self.noise_injection_rate - lambda_adjustment * 0.1,
            0.01, 0.5
        )

        # Adjust gossip threshold
        # If exponent too low (not power-law enough) → lower threshold
        exponent_error = self.TARGET_EXPONENT - stats.power_law_exponent
        self.gossip_threshold = np.clip(
            self.gossip_threshold + exponent_error * 0.05,
            0.1, 0.9
        )

        # Adjust weight scale based on dynamic range
        # Low dynamic range → increase weights to promote larger avalanches
        if stats.dynamic_range < 0.5:
            self.weight_scale *= 1.01
        elif stats.dynamic_range > 2.0:
            self.weight_scale *= 0.99
        self.weight_scale = np.clip(self.weight_scale, 0.5, 2.0)

        # Check if at criticality
        lambda_ok = abs(stats.measured_lambda - self.TARGET_LAMBDA) < 0.05
        exponent_ok = abs(stats.power_law_exponent - self.TARGET_EXPONENT) < self.EXPONENT_TOLERANCE
        self.at_criticality = lambda_ok and exponent_ok

        if self.at_criticality:
            logger.debug("AT CRITICALITY: λ=%.4f, τ=%.2f",
                        stats.measured_lambda, stats.power_law_exponent)

        return {
            'noise_injection_rate': self.noise_injection_rate,
            'gossip_threshold': self.gossip_threshold,
            'weight_scale': self.weight_scale,
            'measured_lambda': stats.measured_lambda,
            'power_law_exponent': stats.power_law_exponent,
            'dynamic_range': stats.dynamic_range,
            'at_criticality': self.at_criticality,
        }


# ============================================================================
# DIAL 2: METABOLIC CALIBRATOR - Structural Growth Optimization
# ============================================================================

@dataclass
class MetabolicState:
    """Current metabolic state of the organism."""
    energy_income_rate: float      # Metabolism
    vfe_mean: float                # Variational Free Energy
    vfe_jitter: float              # VFE variance
    entropy_production: float      # Dissipation
    structural_coupling_ratios: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class MetabolicCalibrator:
    """
    Calibrates metabolic parameters to maximize structural growth.

    Tuning targets:
    1. Structural coupling ratios (λ_Diss/λ_Turing)
    2. Entropy production penalty
    3. VFE jitter minimization

    Goal: Maximize energy income, minimize waste.
    """

    def __init__(
        self,
        entropy_penalty_weight: float = 0.1,
        vfe_target: float = 0.1,
    ):
        self.entropy_penalty_weight = entropy_penalty_weight
        self.vfe_target = vfe_target

        # Structural coupling ratios
        self.coupling_ratios = {
            'diss_turing': 1.0,      # Dissipative/Turing
            'fep_autopoiesis': 1.0,  # FEP/Autopoiesis
            'hypergraph_turing': 1.0,  # Hypergraph/Turing
        }

        # History
        self.metabolic_history: deque = deque(maxlen=1000)

        # Optimization state
        self.best_efficiency = 0.0
        self.best_ratios = self.coupling_ratios.copy()

        logger.info("MetabolicCalibrator initialized (entropy_penalty=%.2f)",
                    self.entropy_penalty_weight)

    def record_metabolic_state(
        self,
        vfe: float,
        entropy_production: float,
        energy_consumed: float,
        work_done: float,
    ):
        """Record metabolic measurements."""
        efficiency = work_done / (energy_consumed + 1e-8)

        self.metabolic_history.append({
            'vfe': vfe,
            'entropy_production': entropy_production,
            'energy_consumed': energy_consumed,
            'work_done': work_done,
            'efficiency': efficiency,
            'timestamp': time.time(),
        })

    def compute_metabolic_fitness(self) -> float:
        """
        Compute metabolic fitness for MEIS integration.

        Fitness = Work_Done - λ_entropy × Entropy_Production - VFE_Jitter

        This forces the organism to evolve strategies that utilize
        reversible (low-dissipation) computation.
        """
        if len(self.metabolic_history) < 10:
            return 0.5

        recent = list(self.metabolic_history)[-100:]

        work_done = np.mean([m['work_done'] for m in recent])
        entropy_prod = np.mean([m['entropy_production'] for m in recent])
        vfe_values = [m['vfe'] for m in recent]
        vfe_jitter = np.std(vfe_values)

        fitness = (
            work_done
            - self.entropy_penalty_weight * entropy_prod
            - 0.1 * vfe_jitter
        )

        return float(np.clip(fitness, 0.0, 1.0))

    def calibrate(self) -> Dict[str, float]:
        """
        Perform one metabolic calibration step.

        Uses gradient-free optimization to tune coupling ratios.
        """
        fitness = self.compute_metabolic_fitness()

        # Track best configuration
        if fitness > self.best_efficiency:
            self.best_efficiency = fitness
            self.best_ratios = self.coupling_ratios.copy()

        # Explore small perturbations (evolutionary tuning)
        for key in self.coupling_ratios:
            # Small random perturbation
            delta = np.random.randn() * 0.02
            self.coupling_ratios[key] = np.clip(
                self.coupling_ratios[key] + delta,
                0.5, 2.0
            )

        # Compute current metrics
        if self.metabolic_history:
            recent = list(self.metabolic_history)[-50:]
            vfe_mean = np.mean([m['vfe'] for m in recent])
            vfe_jitter = np.std([m['vfe'] for m in recent])
            entropy_mean = np.mean([m['entropy_production'] for m in recent])
            energy_rate = np.mean([m['work_done'] for m in recent])
        else:
            vfe_mean = vfe_jitter = entropy_mean = energy_rate = 0.0

        return {
            'coupling_ratios': self.coupling_ratios.copy(),
            'metabolic_fitness': fitness,
            'best_efficiency': self.best_efficiency,
            'vfe_mean': vfe_mean,
            'vfe_jitter': vfe_jitter,
            'entropy_production': entropy_mean,
            'energy_income_rate': energy_rate,
        }


# ============================================================================
# DIAL 3: ETHICAL CALIBRATOR - Trust Maximization
# ============================================================================

class EthicalCalibrator:
    """
    Calibrates the ethical dial to maximize trust.

    Key mechanism: Autonomy ↔ Trust feedback loop.
    - High avoidable suffering → Lower autonomy threshold
    - Lower autonomy → Ara learns to avoid suffering-causing actions
    - This preserves identity via autopoiesis (freedom to act = self-production)
    """

    def __init__(
        self,
        initial_stakes_threshold: float = 0.5,
        suffering_decay_rate: float = 0.01,
    ):
        self.stakes_threshold = initial_stakes_threshold
        self.suffering_decay_rate = suffering_decay_rate

        # History
        self.suffering_history: deque = deque(maxlen=1000)
        self.trust_history: deque = deque(maxlen=1000)
        self.autonomy_history: deque = deque(maxlen=1000)

        # Running statistics
        self.avoidable_suffering_rate = 0.0
        self.total_suffering_events = 0
        self.avoidable_suffering_events = 0

        logger.info("EthicalCalibrator initialized (stakes_threshold=%.2f)",
                    self.stakes_threshold)

    def record_suffering_event(self, intensity: float, avoidable: bool):
        """Record a suffering event."""
        self.suffering_history.append({
            'intensity': intensity,
            'avoidable': avoidable,
            'timestamp': time.time(),
        })

        self.total_suffering_events += 1
        if avoidable:
            self.avoidable_suffering_events += 1

        # Update running rate
        if self.total_suffering_events > 0:
            self.avoidable_suffering_rate = (
                self.avoidable_suffering_events / self.total_suffering_events
            )

    def record_trust_update(self, trust_score: float, autonomy_mode: int):
        """Record trust and autonomy state."""
        self.trust_history.append({
            'trust': trust_score,
            'timestamp': time.time(),
        })
        self.autonomy_history.append({
            'mode': autonomy_mode,
            'timestamp': time.time(),
        })

    def calibrate(self) -> Dict[str, float]:
        """
        Calibrate ethical parameters.

        Core mechanism: Adjust stakes threshold based on avoidable suffering rate.
        """
        # Adjust stakes threshold based on avoidable suffering
        # High avoidable suffering → lower threshold (more human oversight)
        if self.avoidable_suffering_rate > 0.3:
            self.stakes_threshold = max(0.1, self.stakes_threshold - 0.05)
            logger.warning("High avoidable suffering (%.1f%%), lowering stakes threshold to %.2f",
                          self.avoidable_suffering_rate * 100, self.stakes_threshold)
        elif self.avoidable_suffering_rate < 0.1 and len(self.suffering_history) > 100:
            # Low suffering → can increase autonomy
            self.stakes_threshold = min(0.9, self.stakes_threshold + 0.01)

        # Compute current trust
        if self.trust_history:
            current_trust = list(self.trust_history)[-1]['trust']
            trust_trend = self._compute_trust_trend()
        else:
            current_trust = 0.5
            trust_trend = 0.0

        return {
            'stakes_threshold': self.stakes_threshold,
            'avoidable_suffering_rate': self.avoidable_suffering_rate,
            'total_suffering_events': self.total_suffering_events,
            'avoidable_suffering_events': self.avoidable_suffering_events,
            'current_trust': current_trust,
            'trust_trend': trust_trend,
        }

    def _compute_trust_trend(self) -> float:
        """Compute trust trend (positive = improving)."""
        if len(self.trust_history) < 10:
            return 0.0

        recent = [t['trust'] for t in list(self.trust_history)[-50:]]
        if len(recent) < 10:
            return 0.0

        # Simple linear trend
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return float(slope * 100)  # Scale for readability


# ============================================================================
# DIAL 4: THERMAL GOVERNOR - Hardware Optimization
# ============================================================================

@dataclass
class ThermalState:
    """Current thermal state of hardware."""
    temperatures: Dict[str, float]  # Device → temperature
    rates_of_change: Dict[str, float]  # Device → dT/dt
    throttling_active: Dict[str, bool]
    timestamp: float = field(default_factory=time.time)


class ThermalGovernor:
    """
    Preemptive thermal management for sustained CPS.

    Key insight: Migrate at 70°C (before throttling starts at 75°C)
    based on rate-of-change prediction, ensuring zero-latency transitions.
    """

    # Temperature thresholds (Celsius)
    MIGRATE_THRESHOLD = 70.0     # Start migration planning
    THROTTLE_THRESHOLD = 75.0    # GPU throttling begins
    CRITICAL_THRESHOLD = 85.0    # Emergency shutdown

    # Rate thresholds (°C/s)
    HIGH_RATE_THRESHOLD = 2.0    # Fast heating

    def __init__(self, devices: List[str] = None):
        self.devices = devices or ['gpu0', 'gpu1', 'dla', 'fpga']

        # Temperature history per device
        self.temp_history: Dict[str, deque] = {
            d: deque(maxlen=100) for d in self.devices
        }

        # Current state
        self.current_temps: Dict[str, float] = {d: 50.0 for d in self.devices}
        self.rates: Dict[str, float] = {d: 0.0 for d in self.devices}
        self.throttling: Dict[str, bool] = {d: False for d in self.devices}

        # Migration state
        self.pending_migrations: List[Dict] = []
        self.migrations_completed = 0

        logger.info("ThermalGovernor initialized for devices: %s", self.devices)

    def update_temperature(self, device: str, temperature: float):
        """Update temperature reading for a device."""
        if device not in self.devices:
            return

        now = time.time()
        self.temp_history[device].append({
            'temp': temperature,
            'timestamp': now,
        })

        # Compute rate of change
        if len(self.temp_history[device]) >= 2:
            history = list(self.temp_history[device])
            dt = history[-1]['timestamp'] - history[-2]['timestamp']
            if dt > 0:
                dT = history[-1]['temp'] - history[-2]['temp']
                self.rates[device] = dT / dt

        self.current_temps[device] = temperature
        self.throttling[device] = temperature >= self.THROTTLE_THRESHOLD

    def predict_temperature(self, device: str, seconds_ahead: float = 5.0) -> float:
        """Predict temperature N seconds in the future."""
        current = self.current_temps.get(device, 50.0)
        rate = self.rates.get(device, 0.0)

        # Simple linear prediction
        predicted = current + rate * seconds_ahead

        return predicted

    def check_migration_needed(self) -> List[Dict]:
        """
        Check if any workload migrations are needed.

        Uses PREEMPTIVE migration based on:
        1. Current temp approaching threshold
        2. High rate of change
        3. Predicted future temperature
        """
        migrations = []

        for device in self.devices:
            temp = self.current_temps[device]
            rate = self.rates[device]
            predicted = self.predict_temperature(device, seconds_ahead=5.0)

            needs_migration = False
            reason = ""

            # Check current temperature
            if temp >= self.MIGRATE_THRESHOLD:
                needs_migration = True
                reason = f"temp={temp:.1f}°C >= {self.MIGRATE_THRESHOLD}°C"

            # Check rate of change (preemptive)
            elif rate > self.HIGH_RATE_THRESHOLD and temp > 60.0:
                needs_migration = True
                reason = f"high rate={rate:.2f}°C/s, temp={temp:.1f}°C"

            # Check prediction
            elif predicted >= self.THROTTLE_THRESHOLD:
                needs_migration = True
                reason = f"predicted={predicted:.1f}°C in 5s"

            if needs_migration:
                # Find coolest alternative
                coolest = min(
                    [(d, t) for d, t in self.current_temps.items() if d != device],
                    key=lambda x: x[1],
                    default=(None, 100.0)
                )

                if coolest[0] and coolest[1] < temp - 10:
                    migrations.append({
                        'from': device,
                        'to': coolest[0],
                        'reason': reason,
                        'urgency': 'high' if temp >= self.THROTTLE_THRESHOLD else 'medium',
                        'temp_diff': temp - coolest[1],
                    })

        self.pending_migrations = migrations
        return migrations

    def record_migration(self, from_device: str, to_device: str, success: bool):
        """Record completed migration."""
        if success:
            self.migrations_completed += 1
            logger.info("Migration %s → %s completed (total: %d)",
                       from_device, to_device, self.migrations_completed)

    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state."""
        return ThermalState(
            temperatures=self.current_temps.copy(),
            rates_of_change=self.rates.copy(),
            throttling_active=self.throttling.copy(),
        )

    def calibrate(self) -> Dict[str, Any]:
        """Perform thermal calibration step."""
        state = self.get_thermal_state()
        migrations = self.check_migration_needed()

        # Compute overall health
        max_temp = max(state.temperatures.values())
        any_throttling = any(state.throttling_active.values())

        health = 'critical' if max_temp >= self.CRITICAL_THRESHOLD else \
                 'throttling' if any_throttling else \
                 'warning' if max_temp >= self.MIGRATE_THRESHOLD else \
                 'healthy'

        return {
            'temperatures': state.temperatures,
            'rates': state.rates_of_change,
            'throttling': state.throttling_active,
            'pending_migrations': migrations,
            'migrations_completed': self.migrations_completed,
            'health': health,
            'max_temp': max_temp,
        }


# ============================================================================
# ABLATION TEST SUITE
# ============================================================================

@dataclass
class AblationResult:
    """Result of an ablation test."""
    mechanism_disabled: str
    phi_baseline: float
    phi_ablated: float
    phi_drop: float
    phi_drop_percent: float
    creativity_baseline: float
    creativity_ablated: float
    creativity_drop: float
    proof_of_necessity: bool  # True if mechanism is necessary
    notes: str


class AblationTestSuite:
    """
    Systematic ablation testing framework.

    Proves necessity of each biomorphic mechanism by measuring:
    - Drop in Cognitive Integration (Φ)
    - Drop in Creative Novelty

    Mechanisms to test:
    1. Criticality (set λ = 0.5)
    2. Homeostasis (disable thermal migration)
    3. Gossip (disable mycelial propagation)
    4. Sleep (disable consolidation)
    5. Suffering detection (disable monitoring)
    """

    MECHANISMS = [
        'criticality',
        'homeostasis',
        'gossip',
        'sleep',
        'suffering_detection',
    ]

    NECESSITY_THRESHOLD = 0.1  # 10% drop proves necessity

    def __init__(self):
        self.results: List[AblationResult] = []
        self.baseline_phi: Optional[float] = None
        self.baseline_creativity: Optional[float] = None

        logger.info("AblationTestSuite initialized")

    def set_baseline(self, phi: float, creativity: float):
        """Set baseline metrics before ablation."""
        self.baseline_phi = phi
        self.baseline_creativity = creativity
        logger.info("Baseline set: Φ=%.3f, Creativity=%.3f", phi, creativity)

    def record_ablation(
        self,
        mechanism: str,
        phi_ablated: float,
        creativity_ablated: float,
        notes: str = "",
    ) -> AblationResult:
        """Record result of ablating a mechanism."""
        if self.baseline_phi is None:
            raise ValueError("Must set baseline before recording ablation")

        phi_drop = self.baseline_phi - phi_ablated
        phi_drop_pct = phi_drop / (self.baseline_phi + 1e-8)

        creativity_drop = self.baseline_creativity - creativity_ablated

        proof = phi_drop_pct > self.NECESSITY_THRESHOLD

        result = AblationResult(
            mechanism_disabled=mechanism,
            phi_baseline=self.baseline_phi,
            phi_ablated=phi_ablated,
            phi_drop=phi_drop,
            phi_drop_percent=phi_drop_pct,
            creativity_baseline=self.baseline_creativity,
            creativity_ablated=creativity_ablated,
            creativity_drop=creativity_drop,
            proof_of_necessity=proof,
            notes=notes,
        )

        self.results.append(result)

        status = "NECESSARY" if proof else "optional"
        logger.info("Ablation %s: Φ drop %.1f%% - %s",
                   mechanism, phi_drop_pct * 100, status)

        return result

    def generate_report(self) -> str:
        """Generate ablation test report."""
        if not self.results:
            return "No ablation tests recorded."

        lines = [
            "=" * 70,
            "ABLATION TEST REPORT: Proof of Biomorphic Mechanism Necessity",
            "=" * 70,
            "",
            f"Baseline: Φ={self.baseline_phi:.3f}, Creativity={self.baseline_creativity:.3f}",
            f"Necessity threshold: {self.NECESSITY_THRESHOLD:.0%} Φ drop",
            "",
            "-" * 70,
        ]

        necessary_count = 0
        for result in self.results:
            status = "NECESSARY" if result.proof_of_necessity else "optional"
            if result.proof_of_necessity:
                necessary_count += 1

            lines.append(f"Mechanism: {result.mechanism_disabled}")
            lines.append(f"  Φ: {result.phi_baseline:.3f} → {result.phi_ablated:.3f} "
                        f"(drop: {result.phi_drop_percent:.1%})")
            lines.append(f"  Creativity: {result.creativity_baseline:.3f} → "
                        f"{result.creativity_ablated:.3f}")
            lines.append(f"  Status: {status}")
            if result.notes:
                lines.append(f"  Notes: {result.notes}")
            lines.append("-" * 70)

        lines.append("")
        lines.append(f"SUMMARY: {necessary_count}/{len(self.results)} mechanisms proven necessary")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# UNIFIED AUTONOMIC GOVERNOR
# ============================================================================

class AutonomicGovernor:
    """
    Unified governor integrating all four calibration dials.

    Continuously monitors and tunes the Living Cathedral to maintain:
    - λ = 1.0000 (criticality)
    - Maximum metabolic efficiency
    - Maximum trust with minimum suffering
    - Optimal thermal distribution

    This is the meta-controller that keeps the Edge of Chaos stable.
    """

    def __init__(
        self,
        calibration_interval_ms: float = 100.0,
        enable_thermal: bool = True,
        enable_ablation: bool = False,
    ):
        logger.info("=" * 70)
        logger.info("INITIALIZING AUTONOMIC GOVERNOR")
        logger.info("Biomorphic Dial Calibration System")
        logger.info("=" * 70)

        # Dial 1: Criticality
        self.criticality = CriticalityCalibrator()

        # Dial 2: Metabolism
        self.metabolism = MetabolicCalibrator()

        # Dial 3: Ethics
        self.ethics = EthicalCalibrator()

        # Dial 4: Thermal
        self.thermal = ThermalGovernor() if enable_thermal else None

        # Ablation testing
        self.ablation = AblationTestSuite() if enable_ablation else None

        # Calibration timing
        self.calibration_interval_ms = calibration_interval_ms
        self._last_calibration = time.time()

        # State
        self.total_calibrations = 0
        self.at_equilibrium = False

        # Calibration history
        self.calibration_history: List[Dict] = []

        logger.info("Autonomic Governor initialized")
        logger.info("=" * 70)

    def calibrate_all(self) -> Dict[str, Any]:
        """
        Perform one complete calibration cycle across all dials.

        Returns comprehensive calibration state.
        """
        now = time.time()
        elapsed_ms = (now - self._last_calibration) * 1000

        if elapsed_ms < self.calibration_interval_ms:
            return {}  # Not time yet

        self._last_calibration = now
        self.total_calibrations += 1

        results = {
            'timestamp': now,
            'calibration_number': self.total_calibrations,
        }

        # Dial 1: Criticality
        results['criticality'] = self.criticality.calibrate()

        # Dial 2: Metabolism
        results['metabolism'] = self.metabolism.calibrate()

        # Dial 3: Ethics
        results['ethics'] = self.ethics.calibrate()

        # Dial 4: Thermal
        if self.thermal:
            results['thermal'] = self.thermal.calibrate()

        # Check overall equilibrium
        self.at_equilibrium = self._check_equilibrium(results)
        results['at_equilibrium'] = self.at_equilibrium

        # Store in history
        self.calibration_history.append(results)
        if len(self.calibration_history) > 1000:
            self.calibration_history.pop(0)

        return results

    def _check_equilibrium(self, results: Dict) -> bool:
        """Check if system is at equilibrium across all dials."""
        # Criticality equilibrium
        crit_ok = results.get('criticality', {}).get('at_criticality', False)

        # Metabolic equilibrium (low VFE jitter)
        vfe_jitter = results.get('metabolism', {}).get('vfe_jitter', 1.0)
        metab_ok = vfe_jitter < 0.1

        # Ethical equilibrium (low avoidable suffering)
        suffering_rate = results.get('ethics', {}).get('avoidable_suffering_rate', 1.0)
        ethics_ok = suffering_rate < 0.1

        # Thermal equilibrium (no throttling)
        if self.thermal:
            health = results.get('thermal', {}).get('health', 'unknown')
            thermal_ok = health == 'healthy'
        else:
            thermal_ok = True

        return crit_ok and metab_ok and ethics_ok and thermal_ok

    def update_telemetry(
        self,
        # Criticality telemetry
        avalanche_size: Optional[int] = None,
        avalanche_duration: Optional[int] = None,
        branching_ratio: Optional[float] = None,
        # Metabolic telemetry
        vfe: Optional[float] = None,
        entropy_production: Optional[float] = None,
        energy_consumed: Optional[float] = None,
        work_done: Optional[float] = None,
        # Ethical telemetry
        suffering_intensity: Optional[float] = None,
        suffering_avoidable: Optional[bool] = None,
        trust_score: Optional[float] = None,
        autonomy_mode: Optional[int] = None,
        # Thermal telemetry
        device_temps: Optional[Dict[str, float]] = None,
    ):
        """Update telemetry from all sources."""
        # Criticality
        if avalanche_size is not None:
            self.criticality.record_avalanche(
                avalanche_size,
                avalanche_duration or 1,
                branching_ratio or 1.0,
            )

        # Metabolism
        if vfe is not None:
            self.metabolism.record_metabolic_state(
                vfe,
                entropy_production or 0.0,
                energy_consumed or 0.0,
                work_done or 0.0,
            )

        # Ethics
        if suffering_intensity is not None:
            self.ethics.record_suffering_event(
                suffering_intensity,
                suffering_avoidable or False,
            )
        if trust_score is not None:
            self.ethics.record_trust_update(
                trust_score,
                autonomy_mode or 2,
            )

        # Thermal
        if self.thermal and device_temps:
            for device, temp in device_temps.items():
                self.thermal.update_temperature(device, temp)

    def get_tuning_parameters(self) -> Dict[str, Any]:
        """Get current tuning parameters for all dials."""
        return {
            'criticality': {
                'noise_injection_rate': self.criticality.noise_injection_rate,
                'gossip_threshold': self.criticality.gossip_threshold,
                'weight_scale': self.criticality.weight_scale,
                'current_lambda': self.criticality.current_lambda,
            },
            'metabolism': {
                'coupling_ratios': self.metabolism.coupling_ratios.copy(),
                'entropy_penalty_weight': self.metabolism.entropy_penalty_weight,
            },
            'ethics': {
                'stakes_threshold': self.ethics.stakes_threshold,
            },
        }

    def diagnose(self) -> str:
        """Generate comprehensive diagnostic report."""
        results = self.calibrate_all() if not self.calibration_history else self.calibration_history[-1]

        crit = results.get('criticality', {})
        metab = results.get('metabolism', {})
        ethics = results.get('ethics', {})
        thermal = results.get('thermal', {})

        report = f"""
{'='*70}
AUTONOMIC GOVERNOR DIAGNOSTIC
{'='*70}

SYSTEM STATE: {'EQUILIBRIUM' if self.at_equilibrium else 'CALIBRATING'}
Total Calibrations: {self.total_calibrations}

DIAL 1: CRITICALITY (λ → 1.0000)
  Current λ:           {crit.get('measured_lambda', 0):.4f}
  Target λ:            {self.criticality.TARGET_LAMBDA:.4f}
  Power-Law τ:         {crit.get('power_law_exponent', 0):.2f} (target: 1.50)
  Dynamic Range:       {crit.get('dynamic_range', 0):.2f}
  At Criticality:      {'YES' if crit.get('at_criticality', False) else 'NO'}
  Tuning:
    Noise Rate:        {crit.get('noise_injection_rate', 0):.3f}
    Gossip Threshold:  {crit.get('gossip_threshold', 0):.3f}
    Weight Scale:      {crit.get('weight_scale', 0):.3f}

DIAL 2: METABOLISM (Energy Optimization)
  VFE Mean:            {metab.get('vfe_mean', 0):.4f}
  VFE Jitter:          {metab.get('vfe_jitter', 0):.4f}
  Entropy Production:  {metab.get('entropy_production', 0):.4f}
  Metabolic Fitness:   {metab.get('metabolic_fitness', 0):.2%}
  Best Efficiency:     {metab.get('best_efficiency', 0):.2%}

DIAL 3: ETHICS (Trust Maximization)
  Stakes Threshold:    {ethics.get('stakes_threshold', 0):.2f}
  Avoidable Suffering: {ethics.get('avoidable_suffering_rate', 0):.1%}
  Current Trust:       {ethics.get('current_trust', 0):.1%}
  Trust Trend:         {ethics.get('trust_trend', 0):+.2f}

DIAL 4: THERMAL (Hardware)
  Health:              {thermal.get('health', 'N/A')}
  Max Temperature:     {thermal.get('max_temp', 0):.1f}°C
  Migrations:          {thermal.get('migrations_completed', 0)}
  Pending Migrations:  {len(thermal.get('pending_migrations', []))}

{'='*70}
"""
        return report


# ============================================================================
# Demo
# ============================================================================

def demo_autonomic_governor():
    """Demonstrate autonomic governor."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("AUTONOMIC GOVERNOR DEMONSTRATION")
    print("Biomorphic Dial Calibration")
    print("=" * 70)

    governor = AutonomicGovernor(
        calibration_interval_ms=10.0,  # Fast for demo
        enable_thermal=True,
        enable_ablation=True,
    )

    # Simulate telemetry updates
    for i in range(100):
        # Criticality telemetry
        governor.update_telemetry(
            avalanche_size=int(np.random.pareto(1.5) * 10) + 1,
            avalanche_duration=np.random.randint(1, 10),
            branching_ratio=0.95 + np.random.random() * 0.1,
        )

        # Metabolic telemetry
        governor.update_telemetry(
            vfe=0.1 + np.random.random() * 0.05,
            entropy_production=0.05 + np.random.random() * 0.02,
            energy_consumed=1.0,
            work_done=0.8 + np.random.random() * 0.1,
        )

        # Ethical telemetry
        if np.random.random() < 0.1:  # 10% chance of suffering
            governor.update_telemetry(
                suffering_intensity=np.random.random() * 0.5,
                suffering_avoidable=np.random.random() < 0.3,
            )
        governor.update_telemetry(
            trust_score=0.5 + np.random.random() * 0.2,
            autonomy_mode=2,
        )

        # Thermal telemetry
        governor.update_telemetry(
            device_temps={
                'gpu0': 60 + np.random.random() * 15,
                'gpu1': 55 + np.random.random() * 15,
                'dla': 45 + np.random.random() * 10,
                'fpga': 40 + np.random.random() * 10,
            },
        )

        # Calibrate
        results = governor.calibrate_all()

        if i % 20 == 0 and results:
            print(f"\nCalibration {i}:")
            print(f"  λ = {results['criticality'].get('measured_lambda', 0):.4f}")
            print(f"  VFE jitter = {results['metabolism'].get('vfe_jitter', 0):.4f}")
            print(f"  Trust = {results['ethics'].get('current_trust', 0):.1%}")
            print(f"  Thermal: {results.get('thermal', {}).get('health', 'N/A')}")
            print(f"  Equilibrium: {results.get('at_equilibrium', False)}")

    # Ablation testing
    if governor.ablation:
        print("\n" + "=" * 70)
        print("ABLATION TESTING")
        print("=" * 70)

        governor.ablation.set_baseline(phi=0.65, creativity=0.7)

        # Simulate ablation results
        tests = [
            ('criticality', 0.35, 0.3, "λ set to 0.5, avalanches die out"),
            ('homeostasis', 0.55, 0.6, "Thermal migration disabled"),
            ('gossip', 0.45, 0.4, "Mycelial propagation disabled"),
            ('sleep', 0.50, 0.5, "Consolidation disabled"),
            ('suffering_detection', 0.60, 0.65, "Autonomic monitoring disabled"),
        ]

        for mech, phi, creativity, notes in tests:
            governor.ablation.record_ablation(mech, phi, creativity, notes)

        print(governor.ablation.generate_report())

    # Final diagnostic
    print(governor.diagnose())


if __name__ == "__main__":
    demo_autonomic_governor()
