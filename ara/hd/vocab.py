"""
Ara HD Vocabulary - Symbol to Hypervector Mapping
=================================================

Manages the global HD vocabulary for Ara's sensorium, time, and tasks.

All symbols are lazily generated on first access and cached for consistency.
Seeds are deterministic from symbol names, ensuring stability across runs.

Vocabulary categories:
- Roles: VISION, HEARING, TOUCH, SMELL, TASTE, VESTIBULAR, INTEROCEPTION, TIME, TASK
- Features: BRIGHTNESS, MOTION, VOLUME, TEMP, VOLTAGE, FATIGUE, etc.
- Bins: LOW, MED, HIGH, CRITICAL
- Tags: SAFE, WARNING, DANGER, PROTECT
- Time slots: MORNING, AFTERNOON, EVENING, NIGHT, LATE_NIGHT
- Tasks: THERMAL_GUARDIAN, FPGA_BUILD, FOUNDER_FOCUS, etc.

Usage:
    from ara.hd import get_vocab

    vocab = get_vocab()
    h_vision = vocab.role("VISION")
    h_brightness = vocab.feature("BRIGHTNESS")
    h_high = vocab.bin("HIGH")

    # All HVs are deterministic and cached
    assert np.array_equal(vocab.role("VISION"), vocab.role("VISION"))
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .ops import DIM, random_hv_from_string


class HDVocab:
    """
    Global HD vocabulary manager for Ara's sensorium.

    All HVs are 16,384-bit binary {0,1} arrays.
    Generated deterministically from symbol names for reproducibility.
    """

    # Canonical roles (senses + context)
    CANONICAL_ROLES = [
        "VISION",
        "HEARING",
        "TOUCH",
        "SMELL",
        "TASTE",
        "VESTIBULAR",
        "PROPRIOCEPTION",
        "INTEROCEPTION",
        "TIME",
        "TASK",
        "TELEOLOGY",
    ]

    # Canonical features per sense
    CANONICAL_FEATURES = [
        # Vision
        "BRIGHTNESS", "MOTION", "FACE", "SCREEN_BUSY", "COLOR_TEMP",
        # Hearing
        "VOLUME", "NOISE", "VOICE_PRESENT", "COIL_WHINE", "FAN_RPM",
        # Touch
        "BOARD_TEMP", "CPU_TEMP", "GPU_TEMP", "HOTSPOT", "AMBIENT_TEMP",
        # Smell
        "OZONE", "SMELL_ANOMALY", "AIR_QUALITY",
        # Taste
        "VOLTAGE", "CURRENT", "RIPPLE", "POWER_FACTOR",
        # Vestibular
        "PITCH", "ROLL", "YAW", "MOTION_STATE", "VIBRATION",
        # Proprioception
        "CPU_LOAD", "MEMORY_USED", "DISK_IO", "NETWORK_IO", "GPU_UTIL",
        # Interoception
        "FATIGUE", "ATTENTION_DRIFT", "BURNOUT", "STRESS", "FLOW_STATE",
    ]

    # Canonical bins (quantization levels)
    CANONICAL_BINS = [
        "ZERO", "MINIMAL", "LOW", "MED", "HIGH", "CRITICAL", "EXTREME",
    ]

    # Canonical tags (semantic qualifiers)
    CANONICAL_TAGS = [
        "SAFE", "OK", "WARNING", "DANGER", "CRITICAL",
        "PROTECT", "ALERT", "NOMINAL", "OPTIMAL", "DEGRADED",
        "STILL", "WALKING", "PACING", "MOVING",
    ]

    # Canonical time slots
    CANONICAL_TIME_SLOTS = [
        "EARLY_MORNING",  # 5-8
        "MORNING",        # 8-12
        "AFTERNOON",      # 12-17
        "EVENING",        # 17-21
        "NIGHT",          # 21-24
        "LATE_NIGHT",     # 0-5
    ]

    # Canonical tasks
    CANONICAL_TASKS = [
        "UNKNOWN_TASK",
        "THERMAL_GUARDIAN",
        "FPGA_BUILD",
        "FPGA_SYNTHESIS",
        "FOUNDER_FOCUS",
        "FOUNDER_REST",
        "MAINTENANCE",
        "CATHEDRAL_WORK",
        "RESEARCH",
        "RECOVERY",
        "EMERGENCY",
        "IDLE",
    ]

    def __init__(self, dim: int = DIM, seed_prefix: str = "ara_vocab_v1"):
        """
        Initialize the vocabulary.

        Args:
            dim: HV dimension (default: 16,384)
            seed_prefix: Prefix for deterministic seeding
        """
        self.dim = dim
        self.seed_prefix = seed_prefix

        # Cached HV tables
        self._roles: Dict[str, np.ndarray] = {}
        self._features: Dict[str, np.ndarray] = {}
        self._bins: Dict[str, np.ndarray] = {}
        self._tags: Dict[str, np.ndarray] = {}
        self._time_slots: Dict[str, np.ndarray] = {}
        self._tasks: Dict[str, np.ndarray] = {}

        # Generic cache for custom symbols
        self._custom: Dict[str, np.ndarray] = {}

    def _make_seed(self, category: str, name: str) -> str:
        """Create deterministic seed string."""
        return f"{self.seed_prefix}:{category}:{name}"

    def _get_or_create(
        self,
        table: Dict[str, np.ndarray],
        category: str,
        name: str,
    ) -> np.ndarray:
        """Get from cache or create new HV."""
        if name not in table:
            seed_str = self._make_seed(category, name)
            table[name] = random_hv_from_string(seed_str, self.dim)
        return table[name]

    # =========================================================================
    # Public Accessors
    # =========================================================================

    def role(self, name: str) -> np.ndarray:
        """
        Get role HV (sense modality or context type).

        Standard roles: VISION, HEARING, TOUCH, SMELL, TASTE,
                        VESTIBULAR, PROPRIOCEPTION, INTEROCEPTION, TIME, TASK
        """
        return self._get_or_create(self._roles, "role", name.upper())

    def feature(self, name: str) -> np.ndarray:
        """
        Get feature HV (attribute within a sense).

        Examples: BRIGHTNESS, VOLUME, BOARD_TEMP, FATIGUE
        """
        return self._get_or_create(self._features, "feature", name.upper())

    def bin(self, name: str) -> np.ndarray:
        """
        Get bin HV (quantization level).

        Standard bins: ZERO, MINIMAL, LOW, MED, HIGH, CRITICAL, EXTREME
        """
        return self._get_or_create(self._bins, "bin", name.upper())

    def tag(self, name: str) -> np.ndarray:
        """
        Get tag HV (semantic qualifier).

        Standard tags: SAFE, WARNING, DANGER, PROTECT, NOMINAL, etc.
        """
        return self._get_or_create(self._tags, "tag", name.upper())

    def time_slot(self, name: str) -> np.ndarray:
        """
        Get time slot HV.

        Standard slots: EARLY_MORNING, MORNING, AFTERNOON, EVENING, NIGHT, LATE_NIGHT
        """
        return self._get_or_create(self._time_slots, "time", name.upper())

    def task(self, name: str) -> np.ndarray:
        """
        Get task HV.

        Standard tasks: THERMAL_GUARDIAN, FPGA_BUILD, FOUNDER_FOCUS, etc.
        """
        return self._get_or_create(self._tasks, "task", name.upper())

    def custom(self, category: str, name: str) -> np.ndarray:
        """
        Get custom HV for any category/name pair.

        Useful for extending vocabulary without modifying this class.
        """
        key = f"{category}:{name}"
        if key not in self._custom:
            seed_str = self._make_seed(category, name)
            self._custom[key] = random_hv_from_string(seed_str, self.dim)
        return self._custom[key]

    # =========================================================================
    # Preloading (Optional)
    # =========================================================================

    def preload_canonical(self) -> None:
        """
        Preload all canonical symbols.

        Call this at startup to ensure all standard HVs are cached
        before any time-sensitive operations.
        """
        for name in self.CANONICAL_ROLES:
            self.role(name)
        for name in self.CANONICAL_FEATURES:
            self.feature(name)
        for name in self.CANONICAL_BINS:
            self.bin(name)
        for name in self.CANONICAL_TAGS:
            self.tag(name)
        for name in self.CANONICAL_TIME_SLOTS:
            self.time_slot(name)
        for name in self.CANONICAL_TASKS:
            self.task(name)

    # =========================================================================
    # Inspection
    # =========================================================================

    def stats(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        return {
            "roles": len(self._roles),
            "features": len(self._features),
            "bins": len(self._bins),
            "tags": len(self._tags),
            "time_slots": len(self._time_slots),
            "tasks": len(self._tasks),
            "custom": len(self._custom),
            "total": (
                len(self._roles) + len(self._features) + len(self._bins) +
                len(self._tags) + len(self._time_slots) + len(self._tasks) +
                len(self._custom)
            ),
        }

    def list_symbols(self, category: str) -> list:
        """List all symbols in a category."""
        tables = {
            "role": self._roles,
            "feature": self._features,
            "bin": self._bins,
            "tag": self._tags,
            "time": self._time_slots,
            "task": self._tasks,
        }
        table = tables.get(category.lower(), {})
        return list(table.keys())


# =============================================================================
# Singleton
# =============================================================================

_vocab: Optional[HDVocab] = None


def get_vocab(dim: int = DIM) -> HDVocab:
    """
    Get the global HD vocabulary instance.

    Creates and preloads canonical symbols on first access.
    """
    global _vocab
    if _vocab is None or _vocab.dim != dim:
        _vocab = HDVocab(dim=dim)
        _vocab.preload_canonical()
    return _vocab


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HDVocab',
    'get_vocab',
]
