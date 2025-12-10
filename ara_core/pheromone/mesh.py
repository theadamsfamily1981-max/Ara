#!/usr/bin/env python3
"""
Pheromone Mesh - Digital Chemical Gradients
============================================

10KB pheromone files control 1000+ agents via chemical gradients,
mapping to T-FAN field evolution (Theorem T1: d_B < ε).

Pheromone Types:
    - TASK: Work allocation signals
    - PRIORITY: Urgency/stress indicators (σ* = 0.10)
    - REWARD: NIB D_value gradients for learning
    - ALARM: Emergency coordination
    - TERRITORY: Resource boundaries

Golden Controller Parameters:
    - τ_decay = w = 10 (window size)
    - α_diffuse = 0.12 (correction strength)
    - Evaporation rate = 1/τ_decay per tick
"""

import time
import json
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import hashlib


class PheromoneType(str, Enum):
    """Types of pheromones in the mesh."""
    TASK = "task"           # Work allocation
    PRIORITY = "priority"   # Urgency signals (maps to σ*)
    REWARD = "reward"       # NIB D_value gradients
    ALARM = "alarm"         # Emergency coordination
    TERRITORY = "territory" # Resource boundaries
    TRAIL = "trail"         # Path markers for routing


@dataclass
class Pheromone:
    """A single pheromone deposit."""
    ptype: PheromoneType
    location: str
    intensity: float          # 0.0 to 1.0
    deposited_at: float       # timestamp
    deposited_by: str         # agent ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, tau: float = 10.0, elapsed: float = 1.0) -> float:
        """Apply exponential decay. Returns new intensity."""
        decay_factor = np.exp(-elapsed / tau)
        self.intensity *= decay_factor
        return self.intensity

    def is_expired(self, threshold: float = 0.01) -> bool:
        """Check if pheromone has decayed below threshold."""
        return self.intensity < threshold


@dataclass
class MeshConfig:
    """Configuration for pheromone mesh."""
    # Golden controller parameters
    tau_decay: float = 10.0       # w = 10 (decay window)
    alpha_diffuse: float = 0.12  # α = 0.12 (diffusion strength)

    # Mesh parameters
    max_pheromones_per_location: int = 100
    evaporation_threshold: float = 0.01
    diffusion_radius: int = 2     # Hop distance for diffusion

    # Stress dosing (σ* = 0.10)
    sigma_star: float = 0.10
    priority_boost: float = 1.5   # Boost for priority pheromones

    # Capacity limits (10KB target)
    max_locations: int = 1000
    max_total_pheromones: int = 10000


class PheromoneRing:
    """
    Ring buffer for pheromone deposits at a location.

    Implements T_s = 0.94 cluster routing via gradient aggregation.
    """

    def __init__(self, location: str, max_size: int = 100):
        self.location = location
        self.max_size = max_size
        self.pheromones: List[Pheromone] = []
        self.aggregate: Dict[PheromoneType, float] = defaultdict(float)
        self.last_update: float = time.time()

    def deposit(self, pheromone: Pheromone):
        """Add a pheromone to the ring."""
        self.pheromones.append(pheromone)
        self.aggregate[pheromone.ptype] += pheromone.intensity

        # Enforce max size (FIFO eviction)
        while len(self.pheromones) > self.max_size:
            evicted = self.pheromones.pop(0)
            self.aggregate[evicted.ptype] -= evicted.intensity
            self.aggregate[evicted.ptype] = max(0, self.aggregate[evicted.ptype])

        self.last_update = time.time()

    def decay_all(self, tau: float, elapsed: float) -> int:
        """Decay all pheromones. Returns count of expired ones."""
        expired_count = 0
        remaining = []

        self.aggregate = defaultdict(float)

        for p in self.pheromones:
            p.decay(tau, elapsed)
            if p.is_expired():
                expired_count += 1
            else:
                remaining.append(p)
                self.aggregate[p.ptype] += p.intensity

        self.pheromones = remaining
        self.last_update = time.time()
        return expired_count

    def read_gradient(self) -> Dict[PheromoneType, float]:
        """Read aggregate pheromone gradient."""
        return dict(self.aggregate)

    def total_intensity(self) -> float:
        """Total intensity across all types."""
        return sum(self.aggregate.values())

    def dominant_type(self) -> Optional[PheromoneType]:
        """Get the dominant pheromone type."""
        if not self.aggregate:
            return None
        return max(self.aggregate.keys(), key=lambda k: self.aggregate[k])

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        # Format: location_hash(8) + n_types(1) + [type(1) + intensity(4)]*n
        loc_hash = hashlib.sha256(self.location.encode()).digest()[:8]
        data = bytearray(loc_hash)

        active = [(t, i) for t, i in self.aggregate.items() if i > 0.01]
        data.append(len(active))

        for ptype, intensity in active:
            # Map type to byte
            type_map = {t: i for i, t in enumerate(PheromoneType)}
            data.append(type_map.get(ptype, 0))
            data.extend(struct.pack('f', intensity))

        return bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes, location: str) -> 'PheromoneRing':
        """Deserialize from binary format."""
        ring = cls(location)
        if len(data) < 9:
            return ring

        n_types = data[8]
        offset = 9
        type_list = list(PheromoneType)

        for _ in range(n_types):
            if offset + 5 > len(data):
                break
            type_idx = data[offset]
            intensity = struct.unpack('f', data[offset+1:offset+5])[0]

            if type_idx < len(type_list):
                ring.aggregate[type_list[type_idx]] = intensity

            offset += 5

        return ring


class PriorityTrail:
    """
    Priority trail for stress signaling.

    Maps to σ* = 0.10 optimal stress level.
    High-priority trails trigger antifragile response.
    """

    def __init__(self, sigma_star: float = 0.10):
        self.sigma_star = sigma_star
        self.trails: Dict[str, float] = {}  # path -> priority
        self.stress_level: float = 0.0
        self.last_update: float = time.time()

    def add_trail(self, path: str, priority: float):
        """Add or update a priority trail."""
        self.trails[path] = max(self.trails.get(path, 0), priority)
        self._update_stress()

    def _update_stress(self):
        """Calculate aggregate stress level."""
        if not self.trails:
            self.stress_level = 0.0
        else:
            # Stress = mean priority, clamped to [0, 1]
            self.stress_level = min(1.0, np.mean(list(self.trails.values())))

    def is_optimal_stress(self) -> bool:
        """Check if stress is near σ* = 0.10 (antifragile zone)."""
        return abs(self.stress_level - self.sigma_star) < 0.05

    def stress_adjustment(self) -> float:
        """
        Return adjustment needed to reach σ*.

        Positive = need more stress (increase priority)
        Negative = need less stress (dampen)
        """
        return self.sigma_star - self.stress_level

    def decay(self, tau: float = 10.0, elapsed: float = 1.0):
        """Decay all trail priorities."""
        decay_factor = np.exp(-elapsed / tau)
        self.trails = {
            path: priority * decay_factor
            for path, priority in self.trails.items()
            if priority * decay_factor > 0.01
        }
        self._update_stress()

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "trails": self.trails,
            "stress_level": self.stress_level,
            "sigma_star": self.sigma_star,
            "is_optimal": self.is_optimal_stress(),
        }


class RewardPheromones:
    """
    Reward pheromones for NIB D_value gradients.

    Implements distributed value learning via chemical signals.
    """

    def __init__(self):
        self.values: Dict[str, float] = {}  # location -> accumulated reward
        self.visit_counts: Dict[str, int] = defaultdict(int)
        self.last_rewards: List[Tuple[str, float, float]] = []  # (loc, reward, time)

    def deposit_reward(self, location: str, reward: float):
        """Deposit a reward signal at location."""
        # Exponential moving average
        alpha = 0.1
        current = self.values.get(location, 0.0)
        self.values[location] = current * (1 - alpha) + reward * alpha
        self.visit_counts[location] += 1
        self.last_rewards.append((location, reward, time.time()))

        # Keep last 1000 rewards
        if len(self.last_rewards) > 1000:
            self.last_rewards = self.last_rewards[-1000:]

    def get_value(self, location: str) -> float:
        """Get accumulated value at location."""
        return self.values.get(location, 0.0)

    def get_gradient(self, from_loc: str, neighbors: List[str]) -> Dict[str, float]:
        """Get value gradient from location to neighbors."""
        base = self.get_value(from_loc)
        return {
            n: self.get_value(n) - base
            for n in neighbors
        }

    def best_neighbor(self, from_loc: str, neighbors: List[str]) -> Optional[str]:
        """Find neighbor with highest value gradient."""
        gradient = self.get_gradient(from_loc, neighbors)
        if not gradient:
            return None
        return max(gradient.keys(), key=lambda k: gradient[k])

    def decay(self, tau: float = 10.0, elapsed: float = 1.0):
        """Decay all values."""
        decay_factor = np.exp(-elapsed / tau)
        self.values = {
            loc: val * decay_factor
            for loc, val in self.values.items()
            if val * decay_factor > 0.01
        }

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        # Format: n_locs(2) + [loc_hash(8) + value(4)]*n
        data = bytearray()
        data.extend(struct.pack('H', len(self.values)))

        for loc, val in self.values.items():
            loc_hash = hashlib.sha256(loc.encode()).digest()[:8]
            data.extend(loc_hash)
            data.extend(struct.pack('f', val))

        return bytes(data)


class PheromoneMesh:
    """
    Complete pheromone mesh for Cathedral coordination.

    Coordinates 1000+ agents via 10KB digital chemistry.
    """

    def __init__(self, config: MeshConfig = None):
        self.config = config or MeshConfig()

        # Location rings
        self.rings: Dict[str, PheromoneRing] = {}

        # Specialized subsystems
        self.priority_trail = PriorityTrail(self.config.sigma_star)
        self.reward_pheromones = RewardPheromones()

        # Topology (for diffusion)
        self.neighbors: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self.total_deposits: int = 0
        self.total_evaporated: int = 0
        self.last_tick: float = time.time()

    def add_location(self, location: str, neighbors: List[str] = None):
        """Register a location in the mesh."""
        if location not in self.rings:
            self.rings[location] = PheromoneRing(
                location, self.config.max_pheromones_per_location
            )
        if neighbors:
            self.neighbors[location].update(neighbors)
            # Bidirectional
            for n in neighbors:
                self.neighbors[n].add(location)

    def deposit(self, ptype: PheromoneType, location: str,
                intensity: float, agent_id: str = "system",
                metadata: Dict = None):
        """Deposit a pheromone at a location."""
        # Auto-create location
        if location not in self.rings:
            self.add_location(location)

        pheromone = Pheromone(
            ptype=ptype,
            location=location,
            intensity=intensity,
            deposited_at=time.time(),
            deposited_by=agent_id,
            metadata=metadata or {},
        )

        # Priority boost for stress signals
        if ptype == PheromoneType.PRIORITY:
            pheromone.intensity *= self.config.priority_boost
            self.priority_trail.add_trail(location, intensity)

        # Reward tracking
        if ptype == PheromoneType.REWARD:
            self.reward_pheromones.deposit_reward(location, intensity)

        self.rings[location].deposit(pheromone)
        self.total_deposits += 1

    def read_gradient(self, location: str) -> Dict[PheromoneType, float]:
        """Read pheromone gradient at a location."""
        if location not in self.rings:
            return {}
        return self.rings[location].read_gradient()

    def get_neighbor_gradients(self, location: str) -> Dict[str, Dict[PheromoneType, float]]:
        """Get gradients from all neighbors."""
        return {
            n: self.read_gradient(n)
            for n in self.neighbors.get(location, set())
        }

    def follow_gradient(self, location: str, ptype: PheromoneType) -> Optional[str]:
        """Find neighbor with strongest gradient of given type."""
        current = self.read_gradient(location).get(ptype, 0)
        best_loc = None
        best_gradient = 0

        for neighbor in self.neighbors.get(location, set()):
            neighbor_val = self.read_gradient(neighbor).get(ptype, 0)
            gradient = neighbor_val - current
            if gradient > best_gradient:
                best_gradient = gradient
                best_loc = neighbor

        return best_loc

    def tick(self, elapsed: float = None) -> Dict[str, Any]:
        """
        Run one mesh tick: decay + diffuse.

        Returns statistics about the tick.
        """
        now = time.time()
        if elapsed is None:
            elapsed = now - self.last_tick
        self.last_tick = now

        evaporated = 0
        diffused = 0

        # Decay all rings
        for ring in self.rings.values():
            evaporated += ring.decay_all(self.config.tau_decay, elapsed)

        # Decay subsystems
        self.priority_trail.decay(self.config.tau_decay, elapsed)
        self.reward_pheromones.decay(self.config.tau_decay, elapsed)

        # Diffusion (spread to neighbors)
        diffusion_deposits = []
        for location, ring in self.rings.items():
            neighbors = self.neighbors.get(location, set())
            if not neighbors:
                continue

            for ptype, intensity in ring.read_gradient().items():
                if intensity < 0.1:  # Only diffuse significant amounts
                    continue

                diffuse_amount = intensity * self.config.alpha_diffuse / len(neighbors)
                for neighbor in neighbors:
                    diffusion_deposits.append((ptype, neighbor, diffuse_amount))

        # Apply diffusion (separate pass to avoid iteration issues)
        for ptype, loc, amount in diffusion_deposits:
            if loc in self.rings:
                p = Pheromone(
                    ptype=ptype, location=loc, intensity=amount,
                    deposited_at=now, deposited_by="diffusion"
                )
                self.rings[loc].deposit(p)
                diffused += 1

        self.total_evaporated += evaporated

        return {
            "elapsed": elapsed,
            "evaporated": evaporated,
            "diffused": diffused,
            "active_locations": len([r for r in self.rings.values() if r.total_intensity() > 0.01]),
            "total_intensity": sum(r.total_intensity() for r in self.rings.values()),
            "stress_level": self.priority_trail.stress_level,
            "optimal_stress": self.priority_trail.is_optimal_stress(),
        }

    def get_hotspots(self, ptype: PheromoneType = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get locations with highest pheromone intensity."""
        if ptype:
            scored = [
                (loc, ring.aggregate.get(ptype, 0))
                for loc, ring in self.rings.items()
            ]
        else:
            scored = [
                (loc, ring.total_intensity())
                for loc, ring in self.rings.items()
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def influence_entropy(self) -> float:
        """
        Calculate H_influence across locations.

        Target: > 1.8 bits for healthy diversity.
        """
        intensities = [r.total_intensity() for r in self.rings.values()]
        total = sum(intensities)

        if total < 0.01:
            return 0.0

        probs = [i / total for i in intensities if i > 0]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def health_status(self) -> Dict[str, Any]:
        """Get mesh health status for Cathedral monitoring."""
        h_influence = self.influence_entropy()
        stress = self.priority_trail.stress_level

        return {
            "h_influence": h_influence,
            "h_influence_ok": h_influence > 1.8,
            "stress_level": stress,
            "optimal_stress": self.priority_trail.is_optimal_stress(),
            "active_locations": len([r for r in self.rings.values() if r.total_intensity() > 0.01]),
            "total_locations": len(self.rings),
            "total_deposits": self.total_deposits,
            "total_evaporated": self.total_evaporated,
            "size_bytes": self.size_bytes(),
        }

    def size_bytes(self) -> int:
        """Estimate serialized size in bytes."""
        # Rough estimate: 8 bytes per location hash + 5 bytes per active pheromone type
        size = 0
        for ring in self.rings.values():
            size += 8  # location hash
            size += 1  # n_types
            size += len([i for i in ring.aggregate.values() if i > 0.01]) * 5
        return size

    def to_json(self) -> str:
        """Export mesh state as JSON."""
        return json.dumps({
            "locations": {
                loc: {
                    "gradient": {t.value: i for t, i in ring.read_gradient().items()},
                    "total": ring.total_intensity(),
                }
                for loc, ring in self.rings.items()
                if ring.total_intensity() > 0.01
            },
            "priority_trail": self.priority_trail.to_dict(),
            "health": self.health_status(),
        }, indent=2)

    def save(self, path: str):
        """Save mesh to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'PheromoneMesh':
        """Load mesh from file."""
        mesh = cls()
        with open(path, 'r') as f:
            data = json.load(f)

        for loc, info in data.get("locations", {}).items():
            mesh.add_location(loc)
            for ptype_str, intensity in info.get("gradient", {}).items():
                ptype = PheromoneType(ptype_str)
                mesh.deposit(ptype, loc, intensity, "loaded")

        return mesh


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_mesh: Optional[PheromoneMesh] = None


def get_mesh(hive_size: int = 1000) -> PheromoneMesh:
    """Get the global pheromone mesh instance."""
    global _mesh
    if _mesh is None:
        config = MeshConfig(max_locations=hive_size)
        _mesh = PheromoneMesh(config)
    return _mesh


def mesh_tick() -> Dict[str, Any]:
    """Run a pheromone mesh tick."""
    return get_mesh().tick()


def mesh_status() -> str:
    """Get mesh health status string."""
    health = get_mesh().health_status()
    if health["h_influence_ok"] and health["optimal_stress"]:
        return "🟢 PHEROMONE MESH: HEALTHY"
    elif health["h_influence_ok"] or health["optimal_stress"]:
        return "🟡 PHEROMONE MESH: PARTIAL"
    else:
        return "🔴 PHEROMONE MESH: DEGRADED"


def mesh_gradient(location: str) -> Dict[PheromoneType, float]:
    """Read gradient at a location."""
    return get_mesh().read_gradient(location)
