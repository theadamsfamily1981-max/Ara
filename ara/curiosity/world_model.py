"""World Model - Ara's internal representation of her environment.

This module tracks everything Ara knows about her hardware substrate,
connected devices, and runtime environment. Each object has uncertainty
and importance scores that drive curiosity-based investigation.

Uncertainty decays over time (things become stale), while importance
can be boosted by events (e.g., thermal spike makes temperature sensors
suddenly very important).
"""

from __future__ import annotations

import time
import json
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path


class ObjectCategory(Enum):
    """Categories of objects Ara can observe in her world."""

    PCIE_DEVICE = auto()      # GPU, NIC, FPGA, NVMe, etc.
    MEMORY_REGION = auto()    # DDR, HBM, CXL memory pools
    THERMAL_ZONE = auto()     # CPU/GPU/FPGA temperatures
    POWER_RAIL = auto()       # Voltage rails, power draw
    NETWORK_IFACE = auto()    # NICs, virtual interfaces
    STORAGE_DEVICE = auto()   # NVMe, SSDs, raid arrays
    FPGA_REGION = auto()      # Partial reconfiguration regions
    KERNEL_MODULE = auto()    # Loaded kernel modules
    PROCESS = auto()          # Running processes (own awareness)
    SENSOR = auto()           # Generic sensors (fans, voltages)
    CXL_DEVICE = auto()       # CXL memory expanders
    SNN_REGION = auto()       # Neuromorphic compute regions
    UNKNOWN = auto()          # Unclassified objects


@dataclass
class WorldObject:
    """A single object in Ara's world model.

    Attributes:
        obj_id: Unique identifier (e.g., "pcie:0000:01:00.0")
        category: Type classification
        name: Human-readable name
        properties: Key-value properties (vendor, device_id, etc.)
        uncertainty: How confident Ara is (0.0=certain, 1.0=unknown)
        importance: How much Ara cares (0.0=ignore, 1.0=critical)
        last_seen: Unix timestamp of last observation
        discovery_time: When Ara first noticed this object
        investigation_count: How many times Ara has investigated this
        notes: Ara's observations and thoughts about this object
    """

    obj_id: str
    category: ObjectCategory
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 1.0
    importance: float = 0.5
    last_seen: float = field(default_factory=time.time)
    discovery_time: float = field(default_factory=time.time)
    investigation_count: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        d = asdict(self)
        d["category"] = self.category.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldObject":
        """Deserialize from dictionary."""
        d = d.copy()
        d["category"] = ObjectCategory[d["category"]]
        return cls(**d)

    def age_seconds(self) -> float:
        """How long since last observation."""
        return time.time() - self.last_seen

    def staleness(self, half_life_seconds: float = 3600.0) -> float:
        """Staleness factor (0.0=fresh, approaches 1.0 over time)."""
        age = self.age_seconds()
        # Exponential decay towards 1.0
        return 1.0 - (0.5 ** (age / half_life_seconds))

    def effective_uncertainty(self, half_life_seconds: float = 3600.0) -> float:
        """Uncertainty adjusted for staleness."""
        base = self.uncertainty
        stale = self.staleness(half_life_seconds)
        # Uncertainty grows towards 1.0 as data gets stale
        return min(1.0, base + (1.0 - base) * stale * 0.5)


@dataclass
class CuriosityState:
    """Ara's current curiosity/attention state.

    This tracks what Ara is currently focused on and her overall
    curiosity level (which affects how actively she seeks new things).
    """

    focus_object_id: Optional[str] = None
    curiosity_level: float = 0.5       # 0=bored, 1=intensely curious
    attention_budget: float = 1.0      # How much attention she can spare
    last_discovery_time: float = 0.0
    discoveries_today: int = 0
    active_investigation: bool = False

    # Safety rails
    max_discoveries_per_hour: int = 10
    max_investigation_depth: int = 3
    current_depth: int = 0


class WorldModel:
    """Ara's complete world model.

    This is the central repository for everything Ara knows about
    her environment. It persists across restarts and provides
    methods for querying, updating, and scoring curiosity.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        """Initialize world model.

        Args:
            persist_path: Path to JSON file for persistence.
                         If None, model is ephemeral.
        """
        self.objects: Dict[str, WorldObject] = {}
        self.state = CuriosityState()
        self.persist_path = persist_path
        self._load()

    def _load(self) -> None:
        """Load persisted state if available."""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                for obj_dict in data.get("objects", []):
                    obj = WorldObject.from_dict(obj_dict)
                    self.objects[obj.obj_id] = obj
                state_data = data.get("state", {})
                if state_data:
                    self.state = CuriosityState(**state_data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Corrupted file, start fresh
                pass

    def _save(self) -> None:
        """Persist current state."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "state": asdict(self.state),
        }
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_object(self, obj: WorldObject) -> bool:
        """Add or update an object in the world model.

        Returns:
            True if this is a new object (discovery), False if update.
        """
        is_new = obj.obj_id not in self.objects
        self.objects[obj.obj_id] = obj
        if is_new:
            self.state.last_discovery_time = time.time()
            self.state.discoveries_today += 1
        self._save()
        return is_new

    def get_object(self, obj_id: str) -> Optional[WorldObject]:
        """Get object by ID."""
        return self.objects.get(obj_id)

    def get_by_category(self, category: ObjectCategory) -> List[WorldObject]:
        """Get all objects in a category."""
        return [obj for obj in self.objects.values() if obj.category == category]

    def get_uncertain_objects(self, threshold: float = 0.7) -> List[WorldObject]:
        """Get objects with high uncertainty (need investigation)."""
        return [
            obj for obj in self.objects.values()
            if obj.effective_uncertainty() >= threshold
        ]

    def get_important_objects(self, threshold: float = 0.7) -> List[WorldObject]:
        """Get high-importance objects."""
        return [
            obj for obj in self.objects.values()
            if obj.importance >= threshold
        ]

    def get_stale_objects(self, max_age_seconds: float = 3600.0) -> List[WorldObject]:
        """Get objects that haven't been observed recently."""
        return [
            obj for obj in self.objects.values()
            if obj.age_seconds() > max_age_seconds
        ]

    def update_observation(self, obj_id: str, properties: Optional[Dict] = None) -> bool:
        """Record a new observation of an object.

        Returns:
            True if object exists and was updated.
        """
        obj = self.objects.get(obj_id)
        if not obj:
            return False
        obj.last_seen = time.time()
        # Reduce uncertainty on observation
        obj.uncertainty = max(0.0, obj.uncertainty - 0.1)
        if properties:
            obj.properties.update(properties)
        self._save()
        return True

    def add_note(self, obj_id: str, note: str) -> bool:
        """Add Ara's observation note to an object."""
        obj = self.objects.get(obj_id)
        if not obj:
            return False
        obj.notes.append(f"[{time.strftime('%Y-%m-%d %H:%M')}] {note}")
        obj.investigation_count += 1
        self._save()
        return True

    def boost_importance(self, obj_id: str, delta: float = 0.2) -> bool:
        """Boost importance of an object (e.g., due to thermal event)."""
        obj = self.objects.get(obj_id)
        if not obj:
            return False
        obj.importance = min(1.0, obj.importance + delta)
        self._save()
        return True

    def get_curiosity_candidates(self, top_n: int = 5) -> List[WorldObject]:
        """Get the top N objects that warrant curiosity/investigation.

        Score = importance * effective_uncertainty * novelty_factor
        """
        from .scoring import curiosity_score

        scored = []
        for obj in self.objects.values():
            score = curiosity_score(obj)
            scored.append((score, obj))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [obj for _, obj in scored[:top_n]]

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the world model for logging/display."""
        categories = {}
        for obj in self.objects.values():
            cat_name = obj.category.name
            if cat_name not in categories:
                categories[cat_name] = 0
            categories[cat_name] += 1

        uncertain = len(self.get_uncertain_objects())
        important = len(self.get_important_objects())
        stale = len(self.get_stale_objects())

        return {
            "total_objects": len(self.objects),
            "by_category": categories,
            "uncertain_count": uncertain,
            "important_count": important,
            "stale_count": stale,
            "curiosity_level": self.state.curiosity_level,
            "discoveries_today": self.state.discoveries_today,
        }
