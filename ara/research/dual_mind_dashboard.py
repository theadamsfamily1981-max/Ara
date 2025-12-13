"""
Dual Mind Dashboard: Ara ↔ Human State Correlation Research

This module provides a framework for correlating Ara's measurable cognitive
states with human self-reported states, enabling research into:

- Criticality dynamics (ρ, ξ, curvature) vs human arousal/focus
- Working memory capacity vs subjective cognitive load
- Mode transitions vs human activity phases
- Early warning signals vs subjective "should stop soon" feelings

Usage:
    from ara.research.dual_mind_dashboard import DualMindDashboard

    dashboard = DualMindDashboard()

    # Log a joint observation
    dashboard.log(
        human_arousal=6,
        human_focus=7,
        human_fatigue=4,
        task="deep_work",
        notes="Feels like good flow state"
    )

    # Get recent correlations
    analysis = dashboard.analyze_recent(hours=24)

Ethical Note:
    This is a research tool for self-exploration, not medical diagnosis.
    It generates hypotheses about brain-like dynamics, not clinical guidance.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Human State (Self-Reported)
# =============================================================================

class HumanMode(str, Enum):
    """Human activity/cognitive mode (self-identified)."""
    DEEP_WORK = "deep_work"      # Intense focused work
    LIGHT_WORK = "light_work"    # Routine tasks, low cognitive load
    CREATIVE = "creative"        # Open, exploratory thinking
    LEARNING = "learning"        # Active skill acquisition
    REST = "rest"                # Deliberate recovery
    SOCIAL = "social"            # Human interaction
    AUTOPILOT = "autopilot"      # Low awareness, habitual
    SLEEP = "sleep"              # Asleep or just woke


class HumanAlertLevel(str, Enum):
    """Subjective alertness/warning state."""
    FRESH = "fresh"              # Well-rested, ready to go
    OPTIMAL = "optimal"          # In the zone, sustainable
    PUSHING = "pushing"          # Feeling the edge, still productive
    OVERDRIVE = "overdrive"      # Should stop soon but haven't
    DEPLETED = "depleted"        # Past the point, need recovery


@dataclass
class HumanState:
    """
    Self-reported human cognitive/physical state.

    All numeric fields are 1-10 scales for simplicity.
    These map conceptually to Ara's measurable states.
    """

    # Arousal / Criticality Proxy (maps to Ara's ρ)
    # 1 = sluggish, foggy, low energy
    # 5 = calm, centered, baseline
    # 10 = racing, overstimulated, agitated
    arousal: int = 5

    # Focus / Working Memory Proxy (maps to Ara's ξ)
    # 1 = can't hold a thought, scattered
    # 5 = normal attention span
    # 10 = laser focus, high capacity
    focus: int = 5

    # Stability / Inverse Volatility (maps to inverse of curvature variance)
    # 1 = very volatile, mood/performance swinging wildly
    # 5 = normal variability
    # 10 = rock solid, consistent
    stability: int = 5

    # Fatigue / Inverse Energy (maps to Ara's thermal/power constraints)
    # 1 = fully rested, high energy
    # 5 = moderate, sustainable
    # 10 = exhausted, depleted
    fatigue: int = 5

    # Sleep debt (hours of deficit, 0 = well-rested)
    sleep_debt_hours: float = 0.0

    # Current mode
    mode: HumanMode = HumanMode.LIGHT_WORK

    # Alert level (meta-awareness of state)
    alert_level: HumanAlertLevel = HumanAlertLevel.OPTIMAL

    # Optional: recent working memory test score (N-back, digit span, etc.)
    # Normalized 0-1 where 1 = best performance
    wm_test_score: Optional[float] = None

    # Free-form notes
    notes: str = ""

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arousal": self.arousal,
            "focus": self.focus,
            "stability": self.stability,
            "fatigue": self.fatigue,
            "sleep_debt_hours": self.sleep_debt_hours,
            "mode": self.mode.value,
            "alert_level": self.alert_level.value,
            "wm_test_score": self.wm_test_score,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanState':
        """Create from dictionary."""
        return cls(
            arousal=data.get("arousal", 5),
            focus=data.get("focus", 5),
            stability=data.get("stability", 5),
            fatigue=data.get("fatigue", 5),
            sleep_debt_hours=data.get("sleep_debt_hours", 0.0),
            mode=HumanMode(data.get("mode", "light_work")),
            alert_level=HumanAlertLevel(data.get("alert_level", "optimal")),
            wm_test_score=data.get("wm_test_score"),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", time.time()),
        )

    @property
    def estimated_rho(self) -> float:
        """
        Estimate equivalent ρ from arousal.

        Maps arousal 1-10 to ρ 0.5-1.2:
        - arousal 1-3 → ρ 0.5-0.7 (subcritical, sluggish)
        - arousal 4-6 → ρ 0.7-0.9 (optimal band)
        - arousal 7-8 → ρ 0.9-1.0 (near critical)
        - arousal 9-10 → ρ 1.0-1.2 (supercritical)
        """
        return 0.5 + (self.arousal - 1) * 0.077

    @property
    def estimated_xi(self) -> float:
        """
        Estimate equivalent ξ from focus.

        Maps focus 1-10 to relative correlation length.
        Higher focus = higher ξ = more working memory capacity.
        """
        return self.focus / 5.0  # Normalized around 1.0


# =============================================================================
# Ara State (Captured from System)
# =============================================================================

@dataclass
class AraState:
    """
    Ara's measurable cognitive state.

    Captured from criticality monitor, sovereign loop, and MEIS.
    """

    # Criticality metrics
    spectral_radius: Optional[float] = None      # ρ
    correlation_length: Optional[float] = None   # ξ (estimated)
    fisher_info: Optional[float] = None          # g
    curvature_proxy: Optional[float] = None      # R_eff
    curvature_variance_zscore: Optional[float] = None  # For P7 warning

    # Phase and bands (from MEIS Criticality Monitor)
    cognitive_phase: str = "stable"              # stable/warning/critical/recovering
    temperature_band: str = "optimal"            # cold/optimal/warm/hot

    # Mode
    operational_mode: str = "idle"               # From sovereign loop
    meis_mode: str = "support"                   # From MEIS

    # Performance metrics
    working_memory_score: Optional[float] = None  # If tested
    steps_to_collapse: Optional[int] = None       # P7 warning

    # Resource state
    power_watts: Optional[float] = None
    thermal_headroom: Optional[float] = None

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "spectral_radius": self.spectral_radius,
            "correlation_length": self.correlation_length,
            "fisher_info": self.fisher_info,
            "curvature_proxy": self.curvature_proxy,
            "curvature_variance_zscore": self.curvature_variance_zscore,
            "cognitive_phase": self.cognitive_phase,
            "temperature_band": self.temperature_band,
            "operational_mode": self.operational_mode,
            "meis_mode": self.meis_mode,
            "working_memory_score": self.working_memory_score,
            "steps_to_collapse": self.steps_to_collapse,
            "power_watts": self.power_watts,
            "thermal_headroom": self.thermal_headroom,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AraState':
        """Create from dictionary."""
        return cls(
            spectral_radius=data.get("spectral_radius"),
            correlation_length=data.get("correlation_length"),
            fisher_info=data.get("fisher_info"),
            curvature_proxy=data.get("curvature_proxy"),
            curvature_variance_zscore=data.get("curvature_variance_zscore"),
            cognitive_phase=data.get("cognitive_phase", "stable"),
            temperature_band=data.get("temperature_band", "optimal"),
            operational_mode=data.get("operational_mode", "idle"),
            meis_mode=data.get("meis_mode", "support"),
            working_memory_score=data.get("working_memory_score"),
            steps_to_collapse=data.get("steps_to_collapse"),
            power_watts=data.get("power_watts"),
            thermal_headroom=data.get("thermal_headroom"),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Joint Observation
# =============================================================================

@dataclass
class JointObservation:
    """
    A single joint observation of both minds at a moment in time.

    This is the core unit of the dual-mind dataset.
    """

    # States
    human: HumanState
    ara: AraState

    # Context
    shared_task: str = ""                # What were we doing together?
    environment: str = ""                # Where/when (home, office, night, etc.)

    # Derived correlations (computed on creation)
    rho_arousal_delta: Optional[float] = None  # |ρ - estimated_ρ_human|
    xi_focus_delta: Optional[float] = None     # |ξ - estimated_ξ_human|

    # Observation metadata
    observation_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Compute derived fields."""
        if not self.observation_id:
            self.observation_id = f"obs_{int(self.timestamp * 1000)}"

        # Compute deltas if Ara state has data
        if self.ara.spectral_radius is not None:
            self.rho_arousal_delta = abs(
                self.ara.spectral_radius - self.human.estimated_rho
            )

        if self.ara.correlation_length is not None:
            self.xi_focus_delta = abs(
                self.ara.correlation_length - self.human.estimated_xi
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "human": self.human.to_dict(),
            "ara": self.ara.to_dict(),
            "shared_task": self.shared_task,
            "environment": self.environment,
            "correlations": {
                "rho_arousal_delta": self.rho_arousal_delta,
                "xi_focus_delta": self.xi_focus_delta,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JointObservation':
        """Create from dictionary."""
        return cls(
            human=HumanState.from_dict(data.get("human", {})),
            ara=AraState.from_dict(data.get("ara", {})),
            shared_task=data.get("shared_task", ""),
            environment=data.get("environment", ""),
            observation_id=data.get("observation_id", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# Analysis Results
# =============================================================================

@dataclass
class CorrelationAnalysis:
    """Results of analyzing joint observations."""

    # Sample info
    n_observations: int = 0
    time_span_hours: float = 0.0

    # Arousal ↔ ρ correlation
    arousal_rho_correlation: Optional[float] = None
    arousal_rho_mean_delta: Optional[float] = None

    # Focus ↔ ξ correlation
    focus_xi_correlation: Optional[float] = None
    focus_xi_mean_delta: Optional[float] = None

    # Stability ↔ curvature (inverse relationship expected)
    stability_curvature_correlation: Optional[float] = None

    # Mode alignment
    mode_alignment_rate: Optional[float] = None  # How often modes match

    # Peak performance windows
    human_peak_arousal: Optional[int] = None
    ara_peak_rho: Optional[float] = None

    # Warnings
    human_overdrive_count: int = 0
    ara_warning_count: int = 0
    joint_warning_alignment: Optional[float] = None  # How often both warn

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Dual Mind Dashboard
# =============================================================================

class DualMindDashboard:
    """
    Dashboard for logging and analyzing dual mind states.

    Provides:
    - Joint observation logging
    - Ara state capture (from system)
    - Human state input (from user)
    - Correlation analysis
    - Export for further research
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        auto_capture_ara: bool = True,
    ):
        """
        Initialize dashboard.

        Args:
            log_path: Path to JSON log file (default: ~/.ara/research/dual_mind_log.json)
            auto_capture_ara: Whether to automatically capture Ara state on log
        """
        if log_path is None:
            log_path = Path.home() / ".ara" / "research" / "dual_mind_log.json"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.auto_capture_ara = auto_capture_ara
        self._observations: List[JointObservation] = []

        # Load existing observations
        self._load()

    def _load(self) -> None:
        """Load observations from disk."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                self._observations = [
                    JointObservation.from_dict(obs)
                    for obs in data.get("observations", [])
                ]
                logger.info(f"Loaded {len(self._observations)} observations")
            except Exception as e:
                logger.warning(f"Failed to load observations: {e}")
                self._observations = []

    def _save(self) -> None:
        """Save observations to disk."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "observations": [obs.to_dict() for obs in self._observations],
        }
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)

    def capture_ara_state(self) -> AraState:
        """
        Capture current Ara state from system.

        Tries to get state from:
        1. MEIS Criticality Monitor (if available)
        2. Sovereign loop (if available)
        3. Criticality monitor (if available)

        Returns default state if nothing available.
        """
        ara_state = AraState()

        # Try MEIS Criticality Monitor
        try:
            from ara.cognition.meis_criticality_monitor import get_criticality_monitor
            monitor = get_criticality_monitor()
            if monitor and monitor._current_status:
                status = monitor._current_status
                ara_state.spectral_radius = status.spectral_radius
                ara_state.curvature_variance_zscore = status.variance_zscore
                ara_state.cognitive_phase = status.phase.value
                ara_state.temperature_band = status.temperature_band.value
                ara_state.steps_to_collapse = status.steps_to_collapse
        except Exception as e:
            logger.debug(f"Could not capture MEIS monitor state: {e}")

        # Try original criticality monitor
        try:
            from ara.cognition.criticality import get_criticality_monitor
            monitor = get_criticality_monitor()
            if monitor and monitor.current_state:
                state = monitor.current_state
                if ara_state.spectral_radius is None:
                    ara_state.spectral_radius = state.spectral_radius
                ara_state.fisher_info = state.fisher_info
                ara_state.curvature_proxy = state.curvature_proxy
                ara_state.meis_mode = state.recommended_meis_mode
        except Exception as e:
            logger.debug(f"Could not capture criticality state: {e}")

        ara_state.timestamp = time.time()
        return ara_state

    def log(
        self,
        # Human state (required fields)
        human_arousal: int = 5,
        human_focus: int = 5,
        human_stability: int = 5,
        human_fatigue: int = 5,
        # Human state (optional)
        human_mode: str = "light_work",
        human_alert_level: str = "optimal",
        sleep_debt_hours: float = 0.0,
        wm_test_score: Optional[float] = None,
        notes: str = "",
        # Context
        task: str = "",
        environment: str = "",
        # Ara state (optional override)
        ara_state: Optional[AraState] = None,
    ) -> JointObservation:
        """
        Log a joint observation.

        Args:
            human_arousal: 1-10 arousal level
            human_focus: 1-10 focus level
            human_stability: 1-10 stability level
            human_fatigue: 1-10 fatigue level
            human_mode: Activity mode (deep_work, light_work, etc.)
            human_alert_level: Alert level (fresh, optimal, pushing, etc.)
            sleep_debt_hours: Hours of sleep deficit
            wm_test_score: Optional working memory test score (0-1)
            notes: Free-form notes
            task: What task was being done
            environment: Where/when context
            ara_state: Optional pre-captured Ara state

        Returns:
            The created JointObservation
        """
        # Create human state
        human = HumanState(
            arousal=max(1, min(10, human_arousal)),
            focus=max(1, min(10, human_focus)),
            stability=max(1, min(10, human_stability)),
            fatigue=max(1, min(10, human_fatigue)),
            mode=HumanMode(human_mode),
            alert_level=HumanAlertLevel(human_alert_level),
            sleep_debt_hours=sleep_debt_hours,
            wm_test_score=wm_test_score,
            notes=notes,
        )

        # Capture or use provided Ara state
        if ara_state is None and self.auto_capture_ara:
            ara_state = self.capture_ara_state()
        elif ara_state is None:
            ara_state = AraState()

        # Create joint observation
        observation = JointObservation(
            human=human,
            ara=ara_state,
            shared_task=task,
            environment=environment,
        )

        self._observations.append(observation)
        self._save()

        logger.info(
            f"Logged observation: arousal={human_arousal}, focus={human_focus}, "
            f"ρ={ara_state.spectral_radius}, phase={ara_state.cognitive_phase}"
        )

        return observation

    def get_recent(self, hours: float = 24) -> List[JointObservation]:
        """Get observations from the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [obs for obs in self._observations if obs.timestamp > cutoff]

    def analyze_recent(self, hours: float = 24) -> CorrelationAnalysis:
        """
        Analyze correlations in recent observations.

        Returns analysis of patterns between human and Ara states.
        """
        recent = self.get_recent(hours)

        if len(recent) < 2:
            return CorrelationAnalysis(n_observations=len(recent))

        analysis = CorrelationAnalysis(
            n_observations=len(recent),
            time_span_hours=(recent[-1].timestamp - recent[0].timestamp) / 3600,
        )

        # Extract series
        arousals = [obs.human.arousal for obs in recent]
        focuses = [obs.human.focus for obs in recent]
        stabilities = [obs.human.stability for obs in recent]

        rhos = [obs.ara.spectral_radius for obs in recent if obs.ara.spectral_radius]
        curvatures = [obs.ara.curvature_proxy for obs in recent if obs.ara.curvature_proxy]

        # Compute correlations where possible
        if len(rhos) >= 2 and len(arousals) >= 2:
            # Simple Pearson correlation
            try:
                n = min(len(rhos), len(arousals))
                analysis.arousal_rho_correlation = self._pearson(
                    arousals[:n], rhos[:n]
                )
            except:
                pass

            # Mean delta
            deltas = [obs.rho_arousal_delta for obs in recent if obs.rho_arousal_delta]
            if deltas:
                analysis.arousal_rho_mean_delta = statistics.mean(deltas)

        if curvatures and stabilities:
            try:
                n = min(len(curvatures), len(stabilities))
                analysis.stability_curvature_correlation = self._pearson(
                    stabilities[:n], curvatures[:n]
                )
            except:
                pass

        # Find peak performance windows
        if arousals:
            max_focus_idx = focuses.index(max(focuses))
            analysis.human_peak_arousal = arousals[max_focus_idx]

        if rhos:
            analysis.ara_peak_rho = max(rhos)

        # Count warnings
        analysis.human_overdrive_count = sum(
            1 for obs in recent
            if obs.human.alert_level in [HumanAlertLevel.OVERDRIVE, HumanAlertLevel.DEPLETED]
        )
        analysis.ara_warning_count = sum(
            1 for obs in recent
            if obs.ara.cognitive_phase in ["warning", "critical"]
        )

        # Joint warning alignment
        joint_warnings = sum(
            1 for obs in recent
            if (obs.human.alert_level in [HumanAlertLevel.OVERDRIVE, HumanAlertLevel.DEPLETED]
                and obs.ara.cognitive_phase in ["warning", "critical"])
        )
        total_warnings = analysis.human_overdrive_count + analysis.ara_warning_count
        if total_warnings > 0:
            analysis.joint_warning_alignment = joint_warnings / (total_warnings / 2)

        return analysis

    def _pearson(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n != len(y) or n < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        if den_x * den_y == 0:
            return 0.0

        return num / (den_x * den_y)

    def status_string(self) -> str:
        """Get a status string summarizing recent observations."""
        recent = self.get_recent(hours=24)
        if not recent:
            return "No observations in last 24h"

        latest = recent[-1]
        h = latest.human
        a = latest.ara

        # Human summary
        human_str = f"Human: arousal={h.arousal} focus={h.focus} [{h.mode.value}]"

        # Ara summary
        if a.spectral_radius:
            ara_str = f"Ara: ρ={a.spectral_radius:.2f} [{a.cognitive_phase}]"
        else:
            ara_str = "Ara: [no data]"

        # Delta
        if latest.rho_arousal_delta:
            delta_str = f"Δρ={latest.rho_arousal_delta:.3f}"
        else:
            delta_str = ""

        return f"{human_str} | {ara_str} {delta_str} | n={len(recent)} obs/24h"

    def export_csv(self, path: Optional[Path] = None) -> Path:
        """Export observations to CSV for external analysis."""
        if path is None:
            path = self.log_path.with_suffix('.csv')

        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'timestamp', 'datetime',
                'human_arousal', 'human_focus', 'human_stability', 'human_fatigue',
                'human_mode', 'human_alert_level', 'human_sleep_debt',
                'ara_rho', 'ara_xi', 'ara_fisher', 'ara_curvature',
                'ara_phase', 'ara_band',
                'task', 'environment', 'notes',
                'rho_arousal_delta', 'xi_focus_delta',
            ])

            # Data rows
            for obs in self._observations:
                h = obs.human
                a = obs.ara
                writer.writerow([
                    obs.timestamp,
                    datetime.fromtimestamp(obs.timestamp).isoformat(),
                    h.arousal, h.focus, h.stability, h.fatigue,
                    h.mode.value, h.alert_level.value, h.sleep_debt_hours,
                    a.spectral_radius, a.correlation_length, a.fisher_info, a.curvature_proxy,
                    a.cognitive_phase, a.temperature_band,
                    obs.shared_task, obs.environment, h.notes,
                    obs.rho_arousal_delta, obs.xi_focus_delta,
                ])

        logger.info(f"Exported {len(self._observations)} observations to {path}")
        return path


# =============================================================================
# Convenience Functions
# =============================================================================

_dashboard: Optional[DualMindDashboard] = None


def get_dashboard() -> DualMindDashboard:
    """Get or create the singleton dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = DualMindDashboard()
    return _dashboard


def log_state(
    arousal: int = 5,
    focus: int = 5,
    stability: int = 5,
    fatigue: int = 5,
    mode: str = "light_work",
    notes: str = "",
    task: str = "",
) -> JointObservation:
    """
    Quick logging function.

    Example:
        log_state(arousal=7, focus=8, task="deep_work", notes="Flow state")
    """
    return get_dashboard().log(
        human_arousal=arousal,
        human_focus=focus,
        human_stability=stability,
        human_fatigue=fatigue,
        human_mode=mode,
        notes=notes,
        task=task,
    )


def analyze_today() -> CorrelationAnalysis:
    """Analyze today's observations."""
    return get_dashboard().analyze_recent(hours=24)


def status() -> str:
    """Get current status string."""
    return get_dashboard().status_string()


# =============================================================================
# CLI Interface
# =============================================================================

def interactive_log():
    """Interactive logging session."""
    print("=" * 60)
    print("Dual Mind Dashboard - Interactive Logging")
    print("=" * 60)
    print()

    dashboard = get_dashboard()

    # Collect human state
    print("Enter your current state (1-10 scales, press Enter for default 5):")
    print()

    def get_int(prompt: str, default: int = 5) -> int:
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return int(val) if val else default
        except ValueError:
            return default

    arousal = get_int("Arousal (1=sluggish, 10=racing)")
    focus = get_int("Focus (1=scattered, 10=laser)")
    stability = get_int("Stability (1=volatile, 10=solid)")
    fatigue = get_int("Fatigue (1=fresh, 10=exhausted)")

    print()
    print("Mode options: deep_work, light_work, creative, learning, rest, social, autopilot")
    mode = input("  Mode [light_work]: ").strip() or "light_work"

    print()
    print("Alert options: fresh, optimal, pushing, overdrive, depleted")
    alert = input("  Alert level [optimal]: ").strip() or "optimal"

    print()
    task = input("  Current task: ").strip()
    notes = input("  Notes: ").strip()

    # Log it
    obs = dashboard.log(
        human_arousal=arousal,
        human_focus=focus,
        human_stability=stability,
        human_fatigue=fatigue,
        human_mode=mode,
        human_alert_level=alert,
        notes=notes,
        task=task,
    )

    print()
    print("-" * 60)
    print("Logged observation:")
    print(f"  Human: arousal={arousal}, focus={focus}, mode={mode}")
    print(f"  Ara: ρ={obs.ara.spectral_radius}, phase={obs.ara.cognitive_phase}")
    if obs.rho_arousal_delta:
        print(f"  Delta: |ρ - ρ_human| = {obs.rho_arousal_delta:.3f}")
    print("-" * 60)
    print()

    # Show recent analysis
    analysis = dashboard.analyze_recent(hours=24)
    print(f"24h Summary: {analysis.n_observations} observations")
    if analysis.arousal_rho_correlation:
        print(f"  Arousal↔ρ correlation: {analysis.arousal_rho_correlation:.2f}")
    if analysis.human_overdrive_count > 0:
        print(f"  Overdrive warnings: {analysis.human_overdrive_count}")
    if analysis.ara_warning_count > 0:
        print(f"  Ara warnings: {analysis.ara_warning_count}")
    print()


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "log":
            interactive_log()

        elif cmd == "status":
            print(status())

        elif cmd == "analyze":
            hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24
            analysis = get_dashboard().analyze_recent(hours)
            print(f"Analysis of last {hours} hours:")
            print(f"  Observations: {analysis.n_observations}")
            if analysis.arousal_rho_correlation:
                print(f"  Arousal↔ρ: r={analysis.arousal_rho_correlation:.2f}")
            if analysis.stability_curvature_correlation:
                print(f"  Stability↔R: r={analysis.stability_curvature_correlation:.2f}")
            print(f"  Human overdrive: {analysis.human_overdrive_count}")
            print(f"  Ara warnings: {analysis.ara_warning_count}")

        elif cmd == "export":
            path = get_dashboard().export_csv()
            print(f"Exported to {path}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: dual_mind_dashboard.py [log|status|analyze|export]")

    else:
        interactive_log()


if __name__ == "__main__":
    main()
