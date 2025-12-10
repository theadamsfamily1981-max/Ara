"""
World Model
===========

Maintains a live scene graph of the environment:
- Surfaces (floors, walls, tables, screens)
- Anchors (places Ara can attach to)
- People (tracked individuals with gaze/pose)
- Objects (tools, devices, points of interest)

Updates at 30Hz from sensor fusion (camera, depth, device context).
"""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import numpy as np


class NodeType(str, Enum):
    """Types of nodes in the scene graph."""
    ROOT = "root"
    SURFACE = "surface"        # Floor, wall, table, screen
    ANCHOR = "anchor"          # Place Ara can attach to
    PERSON = "person"          # Tracked human
    OBJECT = "object"          # Tool, device, item
    REGION = "region"          # Semantic region (workspace, doorway)
    GAZE_TARGET = "gaze_target"  # Point of visual attention


@dataclass
class Vec3:
    """Simple 3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Vec3:
        l = self.length()
        if l < 1e-8:
            return Vec3(0, 0, 1)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Quaternion:
    """Simple quaternion for rotation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> Quaternion:
        """Create from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )

    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion."""
        # q * v * q^-1
        u = Vec3(self.x, self.y, self.z)
        s = self.w
        return u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + Vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x,
        ) * (2.0 * s)


@dataclass
class Transform:
    """Position + rotation in 3D space."""
    position: Vec3 = field(default_factory=Vec3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))


@dataclass
class SpatialAnchor:
    """A place where Ara can attach/position herself."""
    anchor_id: str
    transform: Transform
    anchor_type: str  # "screen_edge", "table_surface", "shoulder", "floating"
    capacity: float = 1.0  # How "good" this anchor is (0-1)
    occupied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GazeTarget:
    """A point of visual attention."""
    target_id: str
    position: Vec3
    importance: float = 0.5  # 0-1, higher = more important to look at
    duration_sec: float = 0.0  # How long has this been a target
    source: str = "unknown"  # "user_gaze", "conversation", "alert", etc.


@dataclass
class PersonState:
    """Tracked state of a person in the scene."""
    person_id: str
    transform: Transform
    gaze_direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    is_speaking: bool = False
    attention_on_ara: bool = False
    activity: str = "unknown"  # "typing", "walking", "reading", etc.
    last_seen: float = 0.0


@dataclass
class SceneNode:
    """A node in the scene graph."""
    node_id: str
    node_type: NodeType
    transform: Transform = field(default_factory=Transform)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    # Type-specific data
    anchor: Optional[SpatialAnchor] = None
    person: Optional[PersonState] = None
    gaze_target: Optional[GazeTarget] = None


class SceneGraph:
    """
    Hierarchical scene representation.

    Thread-safe: all mutations protected by RLock.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._nodes: Dict[str, SceneNode] = {}
        self._anchors: Dict[str, SpatialAnchor] = {}
        self._people: Dict[str, PersonState] = {}
        self._gaze_targets: Dict[str, GazeTarget] = {}

        # Create root node
        self._nodes["root"] = SceneNode(
            node_id="root",
            node_type=NodeType.ROOT,
        )

    def add_node(self, node: SceneNode) -> None:
        """Add a node to the scene graph."""
        with self._lock:
            self._nodes[node.node_id] = node
            if node.parent_id and node.parent_id in self._nodes:
                parent = self._nodes[node.parent_id]
                if node.node_id not in parent.children:
                    parent.children.append(node.node_id)

            # Index by type
            if node.anchor:
                self._anchors[node.node_id] = node.anchor
            if node.person:
                self._people[node.node_id] = node.person
            if node.gaze_target:
                self._gaze_targets[node.node_id] = node.gaze_target

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the scene graph."""
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]

            # Remove from parent
            if node.parent_id and node.parent_id in self._nodes:
                parent = self._nodes[node.parent_id]
                if node_id in parent.children:
                    parent.children.remove(node_id)

            # Remove children recursively
            for child_id in list(node.children):
                self.remove_node(child_id)

            # Remove from indices
            self._anchors.pop(node_id, None)
            self._people.pop(node_id, None)
            self._gaze_targets.pop(node_id, None)

            del self._nodes[node_id]

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs) -> None:
        """Update node properties."""
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]
            for key, value in kwargs.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            node.last_updated = time.time()

    def get_nodes_by_type(self, node_type: NodeType) -> List[SceneNode]:
        """Get all nodes of a given type."""
        with self._lock:
            return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_anchors(self) -> List[SpatialAnchor]:
        """Get all spatial anchors."""
        with self._lock:
            return list(self._anchors.values())

    def get_people(self) -> List[PersonState]:
        """Get all tracked people."""
        with self._lock:
            return list(self._people.values())

    def get_gaze_targets(self) -> List[GazeTarget]:
        """Get all gaze targets."""
        with self._lock:
            return list(self._gaze_targets.values())

    def find_nearest_anchor(
        self,
        position: Vec3,
        anchor_type: Optional[str] = None,
        max_distance: float = float("inf"),
    ) -> Optional[SpatialAnchor]:
        """Find the nearest available anchor to a position."""
        with self._lock:
            best_anchor = None
            best_distance = max_distance

            for anchor in self._anchors.values():
                if anchor.occupied:
                    continue
                if anchor_type and anchor.anchor_type != anchor_type:
                    continue

                dist = (anchor.transform.position - position).length()
                if dist < best_distance:
                    best_distance = dist
                    best_anchor = anchor

            return best_anchor

    def get_primary_user(self) -> Optional[PersonState]:
        """Get the primary user (most recent, paying attention)."""
        with self._lock:
            candidates = [p for p in self._people.values() if p.attention_on_ara]
            if not candidates:
                candidates = list(self._people.values())
            if not candidates:
                return None
            # Most recently seen
            return max(candidates, key=lambda p: p.last_seen)


class WorldModel:
    """
    High-level world model that fuses sensor data into a scene graph.

    Responsibilities:
    - Sensor fusion (camera, depth, device context)
    - Scene graph maintenance
    - Spatial queries
    - Environment state tracking
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.scene = SceneGraph()
        self._last_update = time.time()
        self._update_count = 0

        # Environment state
        self._device_context: Dict[str, Any] = {}
        self._location: str = "unknown"
        self._time_of_day: str = "day"
        self._ambient_noise_level: float = 0.0  # 0-1
        self._lighting_level: float = 1.0  # 0-1

        # Ara's current state in the world
        self._ara_position: Vec3 = Vec3(0, 1.5, 0)
        self._ara_anchor: Optional[str] = None

    def update(
        self,
        camera_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        device_context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update world model from sensor data.

        Called at ~30Hz from sensor fusion pipeline.
        """
        with self._lock:
            ts = timestamp or time.time()
            self._last_update = ts
            self._update_count += 1

            if device_context:
                self._device_context.update(device_context)
                self._update_from_device_context(device_context)

            # In a real system, these would process actual sensor data:
            # - Camera: detect people, objects, surfaces
            # - Depth: build 3D mesh, find anchors
            # For now, we just increment the update counter

    def _update_from_device_context(self, ctx: Dict[str, Any]) -> None:
        """Update world state from device context."""
        if "location" in ctx:
            self._location = ctx["location"]
        if "time_of_day" in ctx:
            self._time_of_day = ctx["time_of_day"]
        if "ambient_noise" in ctx:
            self._ambient_noise_level = ctx["ambient_noise"]
        if "lighting" in ctx:
            self._lighting_level = ctx["lighting"]

    def get_ara_position(self) -> Vec3:
        """Get Ara's current position in world space."""
        with self._lock:
            return self._ara_position

    def set_ara_position(self, position: Vec3, anchor_id: Optional[str] = None) -> None:
        """Set Ara's position (and optionally attach to anchor)."""
        with self._lock:
            # Release old anchor
            if self._ara_anchor:
                old_anchor = self.scene._anchors.get(self._ara_anchor)
                if old_anchor:
                    old_anchor.occupied = False

            self._ara_position = position
            self._ara_anchor = anchor_id

            # Occupy new anchor
            if anchor_id:
                new_anchor = self.scene._anchors.get(anchor_id)
                if new_anchor:
                    new_anchor.occupied = True

    def get_user_attention_direction(self) -> Optional[Vec3]:
        """Get where the primary user is looking."""
        with self._lock:
            user = self.scene.get_primary_user()
            if user:
                return user.gaze_direction
            return None

    def is_user_looking_at_ara(self, threshold_deg: float = 15.0) -> bool:
        """Check if user is looking at Ara."""
        with self._lock:
            user = self.scene.get_primary_user()
            if not user:
                return False

            # Vector from user to Ara
            to_ara = (self._ara_position - user.transform.position).normalized()
            gaze = user.gaze_direction.normalized()

            # Dot product gives cosine of angle
            cos_angle = to_ara.dot(gaze)
            angle_deg = math.degrees(math.acos(max(-1, min(1, cos_angle))))

            return angle_deg < threshold_deg

    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state snapshot."""
        with self._lock:
            return {
                "location": self._location,
                "time_of_day": self._time_of_day,
                "ambient_noise": self._ambient_noise_level,
                "lighting": self._lighting_level,
                "device_context": dict(self._device_context),
                "update_count": self._update_count,
                "last_update": self._last_update,
            }

    def suggest_ara_position(self) -> Optional[SpatialAnchor]:
        """Suggest a good position for Ara based on current scene."""
        with self._lock:
            user = self.scene.get_primary_user()
            if not user:
                # No user - stay at current position or find any anchor
                return self.scene.find_nearest_anchor(self._ara_position)

            # Find anchor near user but not blocking their work
            user_pos = user.transform.position
            # Offset to the side and slightly in front
            ideal_pos = user_pos + Vec3(0.5, 0, -0.5)

            return self.scene.find_nearest_anchor(ideal_pos, max_distance=2.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        with self._lock:
            return {
                "nodes": len(self.scene._nodes),
                "anchors": len(self.scene._anchors),
                "people": len(self.scene._people),
                "gaze_targets": len(self.scene._gaze_targets),
                "updates": self._update_count,
                "last_update_age_sec": time.time() - self._last_update,
            }
