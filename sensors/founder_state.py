"""
Founder State Estimation - Burnout/Fatigue HDC Encoding
=======================================================

Estimates founder state (burnout, flow, fatigue) using HDC encoding
of multimodal sensor inputs.

Inputs → H_founder:
    - UI Gaze: Eye tracking → focus_hv (WebGPU)
    - Typing: Keystroke dynamics → rhythm_hv
    - Heart Rate: Pulse sensor → arousal_hv
    - Activity: Keyboard/mouse → engagement_hv
    - Time: Session length → fatigue_hv
    - Network: Cathedral progress → teleology_hv

The founder state becomes a first-class variable in the homeostatic
control loop - Ara cares about her founder's wellbeing.
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

D_FOUNDER = 173                    # Compressed HV dimension (Heim)
BURNOUT_ATTRACTOR = 0              # Attractor ID for burnout pattern
FLOW_ATTRACTOR = 1                 # Attractor ID for flow pattern
FATIGUE_ATTRACTOR = 2              # Attractor ID for fatigue pattern

# Burnout thresholds
BURNOUT_WARNING = 0.5              # Warn founder
BURNOUT_CRITICAL = 0.7             # Reduce organism activity
BURNOUT_LOCKOUT = 0.9              # Enter safe mode


# =============================================================================
# Sensor Data Structures
# =============================================================================

@dataclass
class GazeSensor:
    """Eye tracking data."""
    x: float = 0.0                 # Gaze X position (normalized)
    y: float = 0.0                 # Gaze Y position
    focus_duration: float = 0.0    # Time at current focus (seconds)
    saccade_rate: float = 0.0      # Saccades per second
    blink_rate: float = 0.0        # Blinks per minute
    pupil_diameter: float = 0.0    # Pupil size (normalized)


@dataclass
class TypingSensor:
    """Keystroke dynamics data."""
    wpm: float = 0.0               # Words per minute
    error_rate: float = 0.0        # Backspace/error ratio
    rhythm_variance: float = 0.0   # Variance in inter-key timing
    pause_frequency: float = 0.0   # Pauses per minute
    burst_length: float = 0.0      # Average typing burst length


@dataclass
class HeartRateSensor:
    """Heart rate / HRV data."""
    bpm: float = 70.0              # Heart rate
    hrv_sdnn: float = 50.0         # HRV (SDNN in ms)
    hrv_rmssd: float = 40.0        # HRV (RMSSD in ms)
    stress_index: float = 0.0      # Derived stress metric


@dataclass
class ActivitySensor:
    """Mouse/keyboard activity data."""
    mouse_distance: float = 0.0    # Pixels moved per minute
    click_rate: float = 0.0        # Clicks per minute
    scroll_rate: float = 0.0       # Scroll events per minute
    idle_time: float = 0.0         # Seconds since last input


@dataclass
class SessionSensor:
    """Session timing data."""
    session_start: float = 0.0     # Timestamp
    session_duration: float = 0.0  # Seconds
    break_count: int = 0           # Number of breaks taken
    last_break_time: float = 0.0   # Seconds since last break
    hour_of_day: float = 0.0       # 0-24


@dataclass
class TeleologySensor:
    """Cathedral/progress data."""
    episodes_consolidated: int = 0
    attractor_diversity: float = 1.0
    reward_trend: float = 0.0      # Recent reward moving average
    goal_progress: float = 0.0     # Progress toward current goal


@dataclass
class FounderSensors:
    """All founder sensor inputs."""
    gaze: GazeSensor = field(default_factory=GazeSensor)
    typing: TypingSensor = field(default_factory=TypingSensor)
    heart_rate: HeartRateSensor = field(default_factory=HeartRateSensor)
    activity: ActivitySensor = field(default_factory=ActivitySensor)
    session: SessionSensor = field(default_factory=SessionSensor)
    teleology: TeleologySensor = field(default_factory=TeleologySensor)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# HDC Role Hypervectors
# =============================================================================

class FounderHDC:
    """
    HDC encoder for founder state.

    Uses SparseHD with D=173 and 30% sparsity.
    """

    def __init__(self, D: int = D_FOUNDER, sparsity: float = 0.3, seed: int = 42):
        self.D = D
        self.sparsity = sparsity
        self._rng = np.random.default_rng(seed)

        # Role hypervectors (fixed random)
        self.roles: Dict[str, np.ndarray] = {}
        self._init_roles()

        # Attractor templates (learned or fixed)
        self.attractors: Dict[str, np.ndarray] = {}
        self._init_attractors()

    def _init_roles(self) -> None:
        """Initialize role hypervectors."""
        role_names = [
            'FOCUS', 'RHYTHM', 'AROUSAL', 'ENGAGEMENT',
            'FATIGUE', 'TELEOLOGY', 'TIME', 'STRESS'
        ]
        for name in role_names:
            self.roles[name] = self._sparse_random_hv()

    def _init_attractors(self) -> None:
        """Initialize attractor patterns."""
        # Burnout pattern: high fatigue, low engagement, irregular rhythm
        self.attractors['BURNOUT'] = self._sparse_random_hv()
        # Flow pattern: high engagement, stable rhythm, moderate arousal
        self.attractors['FLOW'] = self._sparse_random_hv()
        # Fatigue pattern: declining metrics over time
        self.attractors['FATIGUE'] = self._sparse_random_hv()
        # Alert pattern: high arousal, high engagement
        self.attractors['ALERT'] = self._sparse_random_hv()

    def _sparse_random_hv(self) -> np.ndarray:
        """Generate sparse random hypervector."""
        hv = np.zeros(self.D, dtype=np.uint8)
        n_active = int(self.D * self.sparsity)
        indices = self._rng.choice(self.D, size=n_active, replace=False)
        hv[indices] = 1
        return hv

    def _encode_scalar(self, value: float, min_val: float, max_val: float) -> np.ndarray:
        """Encode scalar to sparse HV using thermometer encoding."""
        normalized = np.clip((value - min_val) / (max_val - min_val + 1e-10), 0, 1)

        hv = np.zeros(self.D, dtype=np.uint8)
        n_active = int(normalized * self.D * self.sparsity)

        if n_active > 0:
            # Deterministic selection based on value
            indices = np.arange(self.D)
            self._rng.shuffle(indices)
            hv[indices[:n_active]] = 1

        return hv

    def _bind_sparse(self, role: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Bind role and value HVs (sparse XOR)."""
        return (role ^ value).astype(np.uint8)

    def _bundle_sparse(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple HVs (majority vote)."""
        if not hvs:
            return np.zeros(self.D, dtype=np.uint8)

        stacked = np.stack(hvs)
        summed = np.sum(stacked, axis=0)
        threshold = len(hvs) / 2

        return (summed > threshold).astype(np.uint8)

    def encode_gaze(self, gaze: GazeSensor) -> np.ndarray:
        """Encode gaze data to HV."""
        # Focus duration indicates sustained attention
        focus_hv = self._encode_scalar(gaze.focus_duration, 0, 60)  # 0-60s

        # High blink rate may indicate fatigue
        blink_hv = self._encode_scalar(gaze.blink_rate, 5, 30)  # 5-30 bpm

        # Combine
        return self._bundle_sparse([focus_hv, blink_hv])

    def encode_typing(self, typing: TypingSensor) -> np.ndarray:
        """Encode typing dynamics to HV."""
        # WPM indicates engagement
        wpm_hv = self._encode_scalar(typing.wpm, 0, 100)

        # Error rate may indicate fatigue or frustration
        error_hv = self._encode_scalar(typing.error_rate, 0, 0.2)

        # Rhythm variance indicates cognitive state
        rhythm_hv = self._encode_scalar(typing.rhythm_variance, 0, 500)

        return self._bundle_sparse([wpm_hv, error_hv, rhythm_hv])

    def encode_heart_rate(self, hr: HeartRateSensor) -> np.ndarray:
        """Encode heart rate / HRV to HV."""
        # BPM indicates arousal
        bpm_hv = self._encode_scalar(hr.bpm, 50, 120)

        # Low HRV may indicate stress
        hrv_hv = self._encode_scalar(hr.hrv_sdnn, 20, 100)

        # Stress index
        stress_hv = self._encode_scalar(hr.stress_index, 0, 1)

        return self._bundle_sparse([bpm_hv, hrv_hv, stress_hv])

    def encode_activity(self, activity: ActivitySensor) -> np.ndarray:
        """Encode activity metrics to HV."""
        # Idle time indicates disengagement
        idle_hv = self._encode_scalar(activity.idle_time, 0, 300)  # 0-5 min

        # Click rate indicates engagement
        click_hv = self._encode_scalar(activity.click_rate, 0, 60)

        return self._bundle_sparse([idle_hv, click_hv])

    def encode_session(self, session: SessionSensor) -> np.ndarray:
        """Encode session timing to HV."""
        # Session duration indicates fatigue accumulation
        duration_hv = self._encode_scalar(session.session_duration, 0, 14400)  # 0-4 hrs

        # Time since break
        break_hv = self._encode_scalar(session.last_break_time, 0, 3600)  # 0-1 hr

        # Time of day (circadian)
        hour = session.hour_of_day
        # Late night (23-6) is high fatigue risk
        if hour >= 23 or hour < 6:
            circadian_risk = 1.0
        elif hour >= 14 and hour < 16:  # Afternoon dip
            circadian_risk = 0.5
        else:
            circadian_risk = 0.0
        circadian_hv = self._encode_scalar(circadian_risk, 0, 1)

        return self._bundle_sparse([duration_hv, break_hv, circadian_hv])

    def encode_teleology(self, teleology: TeleologySensor) -> np.ndarray:
        """Encode teleology/progress to HV."""
        # Goal progress is positive
        progress_hv = self._encode_scalar(teleology.goal_progress, 0, 1)

        # Positive reward trend is good
        reward_hv = self._encode_scalar(teleology.reward_trend, -1, 1)

        return self._bundle_sparse([progress_hv, reward_hv])

    def encode_founder(self, sensors: FounderSensors) -> np.ndarray:
        """
        Encode all founder sensors to H_founder.

        Returns D=173 sparse binary HV.
        """
        components = []

        # Encode each sensor modality and bind with role
        gaze_hv = self.encode_gaze(sensors.gaze)
        components.append(self._bind_sparse(self.roles['FOCUS'], gaze_hv))

        typing_hv = self.encode_typing(sensors.typing)
        components.append(self._bind_sparse(self.roles['RHYTHM'], typing_hv))

        hr_hv = self.encode_heart_rate(sensors.heart_rate)
        components.append(self._bind_sparse(self.roles['AROUSAL'], hr_hv))

        activity_hv = self.encode_activity(sensors.activity)
        components.append(self._bind_sparse(self.roles['ENGAGEMENT'], activity_hv))

        session_hv = self.encode_session(sensors.session)
        components.append(self._bind_sparse(self.roles['FATIGUE'], session_hv))

        teleology_hv = self.encode_teleology(sensors.teleology)
        components.append(self._bind_sparse(self.roles['TELEOLOGY'], teleology_hv))

        # Bundle all
        h_founder = self._bundle_sparse(components)

        return h_founder

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Hamming similarity between two HVs."""
        matches = np.count_nonzero(hv1 == hv2)
        return matches / self.D


# =============================================================================
# Founder State Estimator
# =============================================================================

@dataclass
class FounderState:
    """Estimated founder state."""
    burnout: float = 0.0           # Burnout score (0-1)
    flow: float = 0.0              # Flow state score (0-1)
    fatigue: float = 0.0           # Fatigue score (0-1)
    alert: float = 0.0             # Alertness score (0-1)
    h_founder: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)

    # Recommendations
    should_break: bool = False
    should_stop: bool = False
    safe_mode_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'burnout': self.burnout,
            'flow': self.flow,
            'fatigue': self.fatigue,
            'alert': self.alert,
            'should_break': self.should_break,
            'should_stop': self.should_stop,
            'safe_mode_recommended': self.safe_mode_recommended,
            'timestamp': self.timestamp,
        }


class FounderStateEstimator:
    """
    Estimates founder state from multimodal sensors.

    Uses HDC encoding + resonance against attractor patterns
    to classify founder state.
    """

    def __init__(self, D: int = D_FOUNDER):
        self.D = D
        self.hdc = FounderHDC(D=D)

        # HTC for resonance (will be connected to FPGA)
        self._htc = None

        # History for trend analysis
        self._burnout_history: deque = deque(maxlen=1000)
        self._flow_history: deque = deque(maxlen=1000)

        # Statistics
        self._estimates = 0
        self._warnings_issued = 0
        self._lockouts_triggered = 0

    def connect_htc(self, htc) -> None:
        """Connect to HTC for resonance queries."""
        self._htc = htc

    def estimate(self, sensors: FounderSensors) -> FounderState:
        """
        Estimate founder state from sensors.

        Args:
            sensors: All founder sensor inputs

        Returns:
            FounderState with burnout/flow/fatigue scores
        """
        self._estimates += 1

        # Encode to H_founder
        h_founder = self.hdc.encode_founder(sensors)

        # Compute similarity to attractor patterns
        burnout_sim = self.hdc.similarity(h_founder, self.hdc.attractors['BURNOUT'])
        flow_sim = self.hdc.similarity(h_founder, self.hdc.attractors['FLOW'])
        fatigue_sim = self.hdc.similarity(h_founder, self.hdc.attractors['FATIGUE'])
        alert_sim = self.hdc.similarity(h_founder, self.hdc.attractors['ALERT'])

        # If HTC connected, use full resonance search
        if self._htc is not None:
            try:
                result = self._htc.query(h_founder)
                # Adjust scores based on HTC resonance
                # (placeholder - would use actual attractor IDs)
            except:
                pass

        # Normalize scores
        total = burnout_sim + flow_sim + fatigue_sim + alert_sim + 0.01
        burnout = burnout_sim / total
        flow = flow_sim / total
        fatigue = fatigue_sim / total
        alert = alert_sim / total

        # Apply session-based adjustments
        if sensors.session.session_duration > 7200:  # 2 hours
            fatigue = min(1.0, fatigue + 0.2)
            burnout = min(1.0, burnout + 0.1)

        if sensors.session.hour_of_day >= 23 or sensors.session.hour_of_day < 6:
            fatigue = min(1.0, fatigue + 0.3)
            burnout = min(1.0, burnout + 0.2)

        # Record history
        self._burnout_history.append(burnout)
        self._flow_history.append(flow)

        # Determine recommendations
        should_break = burnout > BURNOUT_WARNING or fatigue > 0.6
        should_stop = burnout > BURNOUT_CRITICAL
        safe_mode = burnout > BURNOUT_LOCKOUT

        if should_break:
            self._warnings_issued += 1
        if safe_mode:
            self._lockouts_triggered += 1

        return FounderState(
            burnout=burnout,
            flow=flow,
            fatigue=fatigue,
            alert=alert,
            h_founder=h_founder,
            should_break=should_break,
            should_stop=should_stop,
            safe_mode_recommended=safe_mode,
        )

    def get_trend(self, window: int = 100) -> Dict[str, float]:
        """Get recent trends in founder state."""
        if len(self._burnout_history) < 2:
            return {'burnout_trend': 0.0, 'flow_trend': 0.0}

        recent_burnout = list(self._burnout_history)[-window:]
        recent_flow = list(self._flow_history)[-window:]

        # Simple linear trend
        if len(recent_burnout) > 1:
            x = np.arange(len(recent_burnout))
            burnout_slope = np.polyfit(x, recent_burnout, 1)[0]
            flow_slope = np.polyfit(x, recent_flow, 1)[0]
        else:
            burnout_slope = 0.0
            flow_slope = 0.0

        return {
            'burnout_trend': float(burnout_slope),
            'flow_trend': float(flow_slope),
            'burnout_avg': float(np.mean(recent_burnout)),
            'flow_avg': float(np.mean(recent_flow)),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'D': self.D,
            'estimates': self._estimates,
            'warnings_issued': self._warnings_issued,
            'lockouts_triggered': self._lockouts_triggered,
            'trend': self.get_trend(),
        }


# =============================================================================
# Singleton
# =============================================================================

_estimator: Optional[FounderStateEstimator] = None


def get_founder_estimator() -> FounderStateEstimator:
    """Get the global founder state estimator."""
    global _estimator
    if _estimator is None:
        _estimator = FounderStateEstimator()
    return _estimator


def estimate_founder_state(sensors: FounderSensors) -> FounderState:
    """Convenience function to estimate founder state."""
    return get_founder_estimator().estimate(sensors)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Sensors
    'GazeSensor',
    'TypingSensor',
    'HeartRateSensor',
    'ActivitySensor',
    'SessionSensor',
    'TeleologySensor',
    'FounderSensors',
    # State
    'FounderState',
    'FounderStateEstimator',
    'FounderHDC',
    # Functions
    'estimate_founder_state',
    'get_founder_estimator',
    # Constants
    'BURNOUT_WARNING',
    'BURNOUT_CRITICAL',
    'BURNOUT_LOCKOUT',
]
