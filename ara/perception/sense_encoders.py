"""
Ara Sense Encoders - Per-Sense HV Encoding
==========================================

Converts raw sensor readings into role-bound hypervectors.

Each encoder:
1. Bins numeric values into discrete levels
2. Binds features with roles: H_attr = bind(H_ROLE, bind(H_FEATURE, H_VALUE))
3. Bundles all attributes into a single sense HV

Architecture follows the HD/VSA role-filler binding pattern:
- Each sense has a base role HV (VISION, HEARING, etc.)
- Features are bound to value bins (e.g., BRIGHTNESS + HIGH)
- Semantic tags add qualitative info (DANGER, SAFE, etc.)

Usage:
    from ara.hd import get_vocab
    from ara.perception.sense_encoders import encode_vision

    vocab = get_vocab()
    vision_data = {"brightness": 0.8, "motion_level": "HIGH", "face_present": True}
    h_vision = encode_vision(vocab, vision_data)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional

from ara.hd.vocab import HDVocab
from ara.hd.ops import bind, bundle, DIM


# =============================================================================
# Binning Functions
# =============================================================================

def temp_bin(t: float) -> str:
    """Bin temperature (Celsius) into levels."""
    if t < 40:
        return "LOW"
    if t < 55:
        return "MED"
    if t < 70:
        return "HIGH"
    if t < 85:
        return "CRITICAL"
    return "EXTREME"


def voltage_bin(v: float) -> str:
    """Bin voltage deviation from nominal."""
    if v < 0.85:
        return "CRITICAL"  # Undervoltage
    if v < 0.95:
        return "LOW"
    if v < 1.05:
        return "MED"  # Nominal
    if v < 1.15:
        return "HIGH"
    return "CRITICAL"  # Overvoltage


def normalized_bin(x: float) -> str:
    """Bin a [0, 1] normalized value."""
    if x < 0.2:
        return "MINIMAL"
    if x < 0.4:
        return "LOW"
    if x < 0.6:
        return "MED"
    if x < 0.8:
        return "HIGH"
    return "CRITICAL"


def fatigue_bin(x: float) -> str:
    """Bin fatigue level [0, 1]."""
    if x < 0.25:
        return "LOW"
    if x < 0.5:
        return "MED"
    if x < 0.75:
        return "HIGH"
    return "CRITICAL"


def angle_bin(a: float) -> str:
    """Bin angle deviation (degrees)."""
    a = abs(a)
    if a < 2:
        return "ZERO"
    if a < 5:
        return "LOW"
    if a < 15:
        return "MED"
    if a < 30:
        return "HIGH"
    return "CRITICAL"


def volume_bin(v: float) -> str:
    """Bin audio volume/RMS level [0, 1]."""
    if v < 0.1:
        return "MINIMAL"
    if v < 0.3:
        return "LOW"
    if v < 0.6:
        return "MED"
    if v < 0.85:
        return "HIGH"
    return "CRITICAL"


def utilization_bin(u: float) -> str:
    """Bin CPU/GPU/memory utilization [0, 1]."""
    if u < 0.2:
        return "LOW"
    if u < 0.5:
        return "MED"
    if u < 0.8:
        return "HIGH"
    if u < 0.95:
        return "CRITICAL"
    return "EXTREME"


# =============================================================================
# Per-Sense Encoders
# =============================================================================

def encode_vision(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode vision sense into HV.

    Expected snap fields:
    - brightness: float [0, 1]
    - motion_level: str "LOW"/"MED"/"HIGH" or float
    - face_present: bool
    - screen_busy: bool
    - color_temp: float (Kelvin, optional)
    """
    hvs = []

    # Brightness
    b = snap.get("brightness", 0.5)
    b_bin = normalized_bin(b) if isinstance(b, (int, float)) else b
    hvs.append(bind(vocab.feature("BRIGHTNESS"), vocab.bin(b_bin)))

    # Motion
    m = snap.get("motion_level", "LOW")
    if isinstance(m, (int, float)):
        m = normalized_bin(m)
    hvs.append(bind(vocab.feature("MOTION"), vocab.bin(m)))

    # Face present (founder detection)
    if snap.get("face_present", False):
        hvs.append(bind(vocab.feature("FACE"), vocab.tag("SAFE")))

    # Screen busy (high activity)
    if snap.get("screen_busy", False):
        hvs.append(bind(vocab.feature("SCREEN_BUSY"), vocab.tag("WARNING")))

    # Color temperature (circadian)
    if "color_temp" in snap:
        ct = snap["color_temp"]
        ct_bin = "LOW" if ct < 4000 else "MED" if ct < 5500 else "HIGH"
        hvs.append(bind(vocab.feature("COLOR_TEMP"), vocab.bin(ct_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("VISION"), bundle(hvs))


def encode_hearing(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode hearing sense into HV.

    Expected snap fields:
    - rms_volume: float [0, 1]
    - noise_level: str or float
    - voice_detected: bool
    - voice_strain: bool (stressed voice)
    - coil_whine: bool
    - fan_rpm: float (optional)
    """
    hvs = []

    # Volume
    vol = snap.get("rms_volume", 0.0)
    vol_bin = volume_bin(vol) if isinstance(vol, (int, float)) else vol
    hvs.append(bind(vocab.feature("VOLUME"), vocab.bin(vol_bin)))

    # Noise level
    n = snap.get("noise_level", "LOW")
    if isinstance(n, (int, float)):
        n = volume_bin(n)
    hvs.append(bind(vocab.feature("NOISE"), vocab.bin(n)))

    # Voice detection
    if snap.get("voice_detected", False):
        tag = "DANGER" if snap.get("voice_strain", False) else "SAFE"
        hvs.append(bind(vocab.feature("VOICE_PRESENT"), vocab.tag(tag)))

    # Coil whine (hardware stress indicator)
    if snap.get("coil_whine", False):
        hvs.append(bind(vocab.feature("COIL_WHINE"), vocab.tag("WARNING")))

    # Fan RPM
    if "fan_rpm" in snap:
        rpm = snap["fan_rpm"]
        rpm_bin = "LOW" if rpm < 1000 else "MED" if rpm < 2500 else "HIGH" if rpm < 4000 else "CRITICAL"
        hvs.append(bind(vocab.feature("FAN_RPM"), vocab.bin(rpm_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("HEARING"), bundle(hvs))


def encode_touch(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode touch sense (thermal) into HV.

    Expected snap fields:
    - board_temp_c: float
    - cpu_temp_c: float
    - gpu_temp_c: float (optional)
    - ambient_temp_c: float (optional)
    - hotspot_detected: bool
    """
    hvs = []

    # Board temperature
    bt = snap.get("board_temp_c", 40.0)
    bt_bin = temp_bin(bt)
    hvs.append(bind(vocab.feature("BOARD_TEMP"), vocab.bin(bt_bin)))

    # CPU temperature
    ct = snap.get("cpu_temp_c", 40.0)
    ct_bin = temp_bin(ct)
    hvs.append(bind(vocab.feature("CPU_TEMP"), vocab.bin(ct_bin)))

    # GPU temperature
    if "gpu_temp_c" in snap:
        gt = snap["gpu_temp_c"]
        gt_bin = temp_bin(gt)
        hvs.append(bind(vocab.feature("GPU_TEMP"), vocab.bin(gt_bin)))

    # Ambient temperature
    if "ambient_temp_c" in snap:
        at = snap["ambient_temp_c"]
        at_bin = "LOW" if at < 20 else "MED" if at < 25 else "HIGH" if at < 30 else "CRITICAL"
        hvs.append(bind(vocab.feature("AMBIENT_TEMP"), vocab.bin(at_bin)))

    # Hotspot detection
    if snap.get("hotspot_detected", False):
        hvs.append(bind(vocab.feature("HOTSPOT"), vocab.tag("DANGER")))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("TOUCH"), bundle(hvs))


def encode_smell(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode smell sense (air quality) into HV.

    Expected snap fields:
    - ozone_level: float [0, 1]
    - smell_anomaly: bool
    - air_quality_index: float (optional)
    """
    hvs = []

    # Ozone level
    oz = snap.get("ozone_level", 0.0)
    o_bin = "LOW" if oz < 0.1 else "MED" if oz < 0.3 else "HIGH" if oz < 0.6 else "CRITICAL"
    hvs.append(bind(vocab.feature("OZONE"), vocab.bin(o_bin)))

    # Smell anomaly (burning, etc.)
    if snap.get("smell_anomaly", False):
        hvs.append(bind(vocab.feature("SMELL_ANOMALY"), vocab.tag("DANGER")))

    # Air quality index
    if "air_quality_index" in snap:
        aqi = snap["air_quality_index"]
        aqi_bin = "LOW" if aqi < 50 else "MED" if aqi < 100 else "HIGH" if aqi < 150 else "CRITICAL"
        hvs.append(bind(vocab.feature("AIR_QUALITY"), vocab.bin(aqi_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("SMELL"), bundle(hvs))


def encode_taste(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode taste sense (power quality) into HV.

    Expected snap fields:
    - voltage_v: float (normalized to 1.0 nominal)
    - current_a: float (optional)
    - ripple_mv: float (optional)
    - power_factor: float (optional)
    - danger_flag: bool
    """
    hvs = []

    # Voltage
    v = snap.get("voltage_v", 1.0)
    v_bin = voltage_bin(v)
    hvs.append(bind(vocab.feature("VOLTAGE"), vocab.bin(v_bin)))

    # Danger flag
    if snap.get("danger_flag", False):
        hvs.append(bind(vocab.feature("VOLTAGE"), vocab.tag("DANGER")))

    # Ripple
    if "ripple_mv" in snap:
        r = snap["ripple_mv"]
        r_bin = "LOW" if r < 20 else "MED" if r < 50 else "HIGH" if r < 100 else "CRITICAL"
        hvs.append(bind(vocab.feature("RIPPLE"), vocab.bin(r_bin)))

    # Power factor
    if "power_factor" in snap:
        pf = snap["power_factor"]
        pf_bin = "CRITICAL" if pf < 0.7 else "LOW" if pf < 0.85 else "MED" if pf < 0.95 else "HIGH"
        hvs.append(bind(vocab.feature("POWER_FACTOR"), vocab.bin(pf_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("TASTE"), bundle(hvs))


def encode_vestibular(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode vestibular sense (orientation/motion) into HV.

    Expected snap fields:
    - pitch_deg: float
    - roll_deg: float
    - yaw_deg: float (optional)
    - motion_state: str "STILL"/"WALKING"/"PACING"/"MOVING"
    - vibration_level: float (optional)
    """
    hvs = []

    # Pitch
    pb = angle_bin(snap.get("pitch_deg", 0.0))
    hvs.append(bind(vocab.feature("PITCH"), vocab.bin(pb)))

    # Roll
    rb = angle_bin(snap.get("roll_deg", 0.0))
    hvs.append(bind(vocab.feature("ROLL"), vocab.bin(rb)))

    # Yaw (optional)
    if "yaw_deg" in snap:
        yb = angle_bin(snap["yaw_deg"])
        hvs.append(bind(vocab.feature("YAW"), vocab.bin(yb)))

    # Motion state
    m = snap.get("motion_state", "STILL")
    hvs.append(bind(vocab.feature("MOTION_STATE"), vocab.tag(m.upper())))

    # Vibration
    if "vibration_level" in snap:
        vib = snap["vibration_level"]
        vib_bin = normalized_bin(vib)
        hvs.append(bind(vocab.feature("VIBRATION"), vocab.bin(vib_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("VESTIBULAR"), bundle(hvs))


def encode_proprioception(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode proprioception sense (self-monitoring) into HV.

    Expected snap fields:
    - cpu_load: float [0, 1]
    - memory_used: float [0, 1]
    - disk_io: float [0, 1] (optional)
    - network_io: float [0, 1] (optional)
    - gpu_util: float [0, 1] (optional)
    """
    hvs = []

    # CPU load
    cpu = snap.get("cpu_load", 0.0)
    cpu_bin = utilization_bin(cpu)
    hvs.append(bind(vocab.feature("CPU_LOAD"), vocab.bin(cpu_bin)))

    # Memory
    mem = snap.get("memory_used", 0.0)
    mem_bin = utilization_bin(mem)
    hvs.append(bind(vocab.feature("MEMORY_USED"), vocab.bin(mem_bin)))

    # Disk I/O
    if "disk_io" in snap:
        dio = snap["disk_io"]
        dio_bin = utilization_bin(dio)
        hvs.append(bind(vocab.feature("DISK_IO"), vocab.bin(dio_bin)))

    # Network I/O
    if "network_io" in snap:
        nio = snap["network_io"]
        nio_bin = utilization_bin(nio)
        hvs.append(bind(vocab.feature("NETWORK_IO"), vocab.bin(nio_bin)))

    # GPU utilization
    if "gpu_util" in snap:
        gpu = snap["gpu_util"]
        gpu_bin = utilization_bin(gpu)
        hvs.append(bind(vocab.feature("GPU_UTIL"), vocab.bin(gpu_bin)))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("PROPRIOCEPTION"), bundle(hvs))


def encode_interoception(vocab: HDVocab, snap: Dict[str, Any]) -> np.ndarray:
    """
    Encode interoception sense (founder state) into HV.

    Expected snap fields:
    - fatigue: float [0, 1]
    - attention: float [0, 1]
    - stress: float [0, 1] (optional)
    - flow_state: float [0, 1] (optional)
    - burnout_risk: bool
    """
    hvs = []

    # Fatigue
    fb = fatigue_bin(snap.get("fatigue", 0.0))
    hvs.append(bind(vocab.feature("FATIGUE"), vocab.bin(fb)))

    # Attention drift (inverse of attention)
    att = snap.get("attention", 1.0)
    ab = fatigue_bin(1.0 - att)
    hvs.append(bind(vocab.feature("ATTENTION_DRIFT"), vocab.bin(ab)))

    # Stress
    if "stress" in snap:
        stress = snap["stress"]
        stress_bin = fatigue_bin(stress)
        hvs.append(bind(vocab.feature("STRESS"), vocab.bin(stress_bin)))

    # Flow state
    if "flow_state" in snap:
        flow = snap["flow_state"]
        flow_bin = "HIGH" if flow > 0.7 else "MED" if flow > 0.4 else "LOW"
        hvs.append(bind(vocab.feature("FLOW_STATE"), vocab.bin(flow_bin)))

    # Burnout risk
    if snap.get("burnout_risk", False):
        hvs.append(bind(vocab.feature("BURNOUT"), vocab.tag("DANGER")))

    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    return bind(vocab.role("INTEROCEPTION"), bundle(hvs))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Binning functions
    'temp_bin',
    'voltage_bin',
    'normalized_bin',
    'fatigue_bin',
    'angle_bin',
    'volume_bin',
    'utilization_bin',
    # Sense encoders
    'encode_vision',
    'encode_hearing',
    'encode_touch',
    'encode_smell',
    'encode_taste',
    'encode_vestibular',
    'encode_proprioception',
    'encode_interoception',
]
