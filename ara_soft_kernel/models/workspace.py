"""
Workspace Models
================

Workspace and surface specifications for visual environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class WorkspaceType(str, Enum):
    """Types of workspaces."""
    DESKTOP = "desktop"
    AR = "ar"
    VR = "vr"
    MIXED_REALITY = "mixed_reality"


class SurfaceKind(str, Enum):
    """Types of surfaces in a workspace."""
    PANEL = "panel"        # 2D panel/window
    HUD = "hud"            # Head-up display element
    OVERLAY = "overlay"    # Semi-transparent overlay
    WINDOW = "window"      # Traditional window
    VOLUME = "volume"      # 3D volumetric display
    AVATAR = "avatar"      # Ara's avatar


class AnchorType(str, Enum):
    """Types of spatial anchors."""
    DESK = "desk"
    HAND = "hand"
    HEAD = "head"
    WORLD = "world"
    SCREEN = "screen"
    WALL = "wall"
    FLOOR = "floor"


@dataclass
class Pose:
    """Spatial pose for a surface."""
    anchor: AnchorType = AnchorType.SCREEN
    offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position: Optional[str] = None  # For HUD: "top-right", "bottom-left", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor.value,
            "offset": self.offset,
            "rotation": self.rotation,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pose:
        anchor = data.get("anchor", "screen")
        if isinstance(anchor, str):
            anchor = AnchorType(anchor)
        return cls(
            anchor=anchor,
            offset=data.get("offset", [0.0, 0.0, 0.0]),
            rotation=data.get("rotation", [0.0, 0.0, 0.0]),
            position=data.get("position"),
        )


@dataclass
class SurfaceSize:
    """Size of a surface."""
    width_deg: Optional[float] = None   # For AR/VR (angular size)
    height_deg: Optional[float] = None
    width_px: Optional[int] = None      # For desktop
    height_px: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.width_deg is not None:
            result["width_deg"] = self.width_deg
        if self.height_deg is not None:
            result["height_deg"] = self.height_deg
        if self.width_px is not None:
            result["width_px"] = self.width_px
        if self.height_px is not None:
            result["height_px"] = self.height_px
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SurfaceSize:
        return cls(
            width_deg=data.get("width_deg"),
            height_deg=data.get("height_deg"),
            width_px=data.get("width_px"),
            height_px=data.get("height_px"),
        )


@dataclass
class Interaction:
    """Interaction properties of a surface."""
    grabbable: bool = False
    resizable: bool = False
    dismissable: bool = True
    clickable: bool = True
    scrollable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grabbable": self.grabbable,
            "resizable": self.resizable,
            "dismissable": self.dismissable,
            "clickable": self.clickable,
            "scrollable": self.scrollable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Interaction:
        return cls(
            grabbable=data.get("grabbable", False),
            resizable=data.get("resizable", False),
            dismissable=data.get("dismissable", True),
            clickable=data.get("clickable", True),
            scrollable=data.get("scrollable", False),
        )


@dataclass
class Visibility:
    """Visibility properties of a surface."""
    min_attention: float = 0.0       # Minimum user attention to show
    fade_when_occluded: bool = True
    auto_hide_after_s: Optional[float] = None
    scale_with_distance: bool = False
    always_face_user: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_attention": self.min_attention,
            "fade_when_occluded": self.fade_when_occluded,
            "auto_hide_after_s": self.auto_hide_after_s,
            "scale_with_distance": self.scale_with_distance,
            "always_face_user": self.always_face_user,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Visibility:
        return cls(
            min_attention=data.get("min_attention", 0.0),
            fade_when_occluded=data.get("fade_when_occluded", True),
            auto_hide_after_s=data.get("auto_hide_after_s"),
            scale_with_distance=data.get("scale_with_distance", False),
            always_face_user=data.get("always_face_user", True),
        )


@dataclass
class Surface:
    """A visual surface in a workspace."""
    id: str
    kind: SurfaceKind
    source: str  # ara://streams/..., agent://..., or URL

    pose: Pose = field(default_factory=Pose)
    size: SurfaceSize = field(default_factory=SurfaceSize)
    interaction: Interaction = field(default_factory=Interaction)
    visibility: Visibility = field(default_factory=Visibility)

    # For avatars
    avatar_id: Optional[str] = None

    # Metadata
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "source": self.source,
            "pose": self.pose.to_dict(),
            "size": self.size.to_dict(),
            "interaction": self.interaction.to_dict(),
            "visibility": self.visibility.to_dict(),
            "avatar_id": self.avatar_id,
            "title": self.title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Surface:
        kind = data.get("kind", "panel")
        if isinstance(kind, str):
            kind = SurfaceKind(kind)
        return cls(
            id=data["id"],
            kind=kind,
            source=data["source"],
            pose=Pose.from_dict(data.get("pose", {})),
            size=SurfaceSize.from_dict(data.get("size", {})),
            interaction=Interaction.from_dict(data.get("interaction", {})),
            visibility=Visibility.from_dict(data.get("visibility", {})),
            avatar_id=data.get("avatar_id"),
            title=data.get("title"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Transitions:
    """Transition animations for workspace."""
    enter: str = "fade_in"
    exit: str = "fade_out"
    duration_ms: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enter": self.enter,
            "exit": self.exit,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transitions:
        return cls(
            enter=data.get("enter", "fade_in"),
            exit=data.get("exit", "fade_out"),
            duration_ms=data.get("duration_ms", 300),
        )


@dataclass
class WorkspaceSpec:
    """Complete specification of a workspace."""
    id: str
    name: str
    type: WorkspaceType

    surfaces: List[Surface] = field(default_factory=list)
    transitions: Transitions = field(default_factory=Transitions)

    # Context
    active_goal: Optional[str] = None
    related_agents: List[str] = field(default_factory=list)

    # State
    active: bool = True
    created_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "surfaces": [s.to_dict() for s in self.surfaces],
            "transitions": self.transitions.to_dict(),
            "context": {
                "active_goal": self.active_goal,
                "related_agents": self.related_agents,
            },
            "active": self.active,
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkspaceSpec:
        ws_type = data.get("type", "desktop")
        if isinstance(ws_type, str):
            ws_type = WorkspaceType(ws_type)
        context = data.get("context", {})
        return cls(
            id=data["id"],
            name=data["name"],
            type=ws_type,
            surfaces=[Surface.from_dict(s) for s in data.get("surfaces", [])],
            transitions=Transitions.from_dict(data.get("transitions", {})),
            active_goal=context.get("active_goal"),
            related_agents=context.get("related_agents", []),
            active=data.get("active", True),
            created_at=data.get("created_at"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> WorkspaceSpec:
        return cls.from_dict(json.loads(json_str))

    def get_surface(self, surface_id: str) -> Optional[Surface]:
        """Get a surface by ID."""
        for surface in self.surfaces:
            if surface.id == surface_id:
                return surface
        return None

    def add_surface(self, surface: Surface) -> None:
        """Add a surface to the workspace."""
        self.surfaces.append(surface)

    def remove_surface(self, surface_id: str) -> bool:
        """Remove a surface by ID."""
        for i, surface in enumerate(self.surfaces):
            if surface.id == surface_id:
                self.surfaces.pop(i)
                return True
        return False
