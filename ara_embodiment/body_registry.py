"""
Body Registry
==============

Catalog of all available Ara avatars/embodiments.

Each avatar defines:
- Visual style and rig
- Voice profile
- Gesture set
- Interaction affordances
- Device compatibility

All avatars share the same Ara identity (memory, values, skills).
"""

from __future__ import annotations

import threading
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from pathlib import Path


class AvatarCapability(str, Enum):
    """Capabilities an avatar can have."""
    FULL_BODY = "full_body"           # Complete body with limbs
    HALF_BODY = "half_body"           # Torso and up
    HEAD_ONLY = "head_only"           # Just head/face
    ABSTRACT = "abstract"             # Non-humanoid (orb, light, etc.)
    TEXT_ONLY = "text_only"           # No visual representation

    # Movement
    WALK = "walk"
    FLOAT = "float"
    TELEPORT = "teleport"
    ATTACH_SURFACE = "attach_surface"  # Can stick to surfaces
    ATTACH_SCREEN = "attach_screen"    # Can stick to screen edges

    # Expression
    LIP_SYNC = "lip_sync"
    FACIAL_EXPRESSIONS = "facial_expressions"
    HAND_GESTURES = "hand_gestures"
    BODY_LANGUAGE = "body_language"
    GAZE_TRACKING = "gaze_tracking"

    # Interaction
    POINT_AT_OBJECTS = "point_at_objects"
    DRAW_IN_AIR = "draw_in_air"
    HIGHLIGHT_UI = "highlight_ui"
    HOLD_OBJECTS = "hold_objects"


class DeviceClass(str, Enum):
    """Device classes for avatar compatibility."""
    AR_GLASSES = "ar_glasses"
    VR_HEADSET = "vr_headset"
    PHONE = "phone"
    TABLET = "tablet"
    DESKTOP = "desktop"
    SMART_SPEAKER = "smart_speaker"
    SMART_DISPLAY = "smart_display"
    WATCH = "watch"


@dataclass
class VoiceProfile:
    """Voice configuration for an avatar."""
    voice_id: str
    pitch: float = 1.0        # Multiplier (0.5 = octave down, 2.0 = octave up)
    speed: float = 1.0        # Speaking rate multiplier
    warmth: float = 0.7       # 0 = cold/robotic, 1 = warm/friendly
    energy: float = 0.5       # 0 = calm/quiet, 1 = excited/loud
    accent: str = "neutral"
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "pitch": self.pitch,
            "speed": self.speed,
            "warmth": self.warmth,
            "energy": self.energy,
            "accent": self.accent,
            "language": self.language,
        }


@dataclass
class GestureSet:
    """Available gestures for an avatar."""
    gesture_set_id: str
    gestures: List[str] = field(default_factory=list)
    # Common gestures: wave, point, nod, shake_head, shrug, thinking,
    # explaining, celebrating, comforting, warning, thumbs_up, etc.

    expressiveness: float = 0.5  # 0 = minimal, 1 = highly animated

    def has_gesture(self, name: str) -> bool:
        return name in self.gestures


@dataclass
class AvatarDefinition:
    """Complete definition of an Ara avatar."""
    avatar_id: str
    name: str
    description: str

    # Visual
    model_path: Optional[str] = None  # Path to 3D model/rig
    style: str = "realistic"  # "realistic", "stylized", "anime", "abstract"
    scale: float = 1.0  # Size multiplier

    # Capabilities
    capabilities: Set[AvatarCapability] = field(default_factory=set)

    # Compatible devices
    compatible_devices: Set[DeviceClass] = field(default_factory=set)

    # Voice and expression
    voice: VoiceProfile = field(default_factory=lambda: VoiceProfile("default"))
    gestures: GestureSet = field(default_factory=lambda: GestureSet("default"))

    # Behavioral hints
    formality: float = 0.5  # 0 = casual, 1 = formal
    energy_default: float = 0.5  # Default energy level
    personal_space: float = 1.0  # Preferred distance from user (meters)

    # Resource requirements
    min_fps: int = 30
    gpu_required: bool = False
    network_required: bool = False

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports_device(self, device: DeviceClass) -> bool:
        return device in self.compatible_devices

    def has_capability(self, cap: AvatarCapability) -> bool:
        return cap in self.capabilities

    def can_express_physically(self) -> bool:
        """Check if avatar can use physical expressions."""
        return any(
            cap in self.capabilities
            for cap in [
                AvatarCapability.LIP_SYNC,
                AvatarCapability.FACIAL_EXPRESSIONS,
                AvatarCapability.HAND_GESTURES,
                AvatarCapability.BODY_LANGUAGE,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avatar_id": self.avatar_id,
            "name": self.name,
            "description": self.description,
            "model_path": self.model_path,
            "style": self.style,
            "scale": self.scale,
            "capabilities": [c.value for c in self.capabilities],
            "compatible_devices": [d.value for d in self.compatible_devices],
            "voice": self.voice.to_dict(),
            "formality": self.formality,
            "energy_default": self.energy_default,
            "tags": self.tags,
        }


class BodyRegistry:
    """
    Registry of all available Ara avatars.

    Thread-safe: all mutations protected by RLock.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._avatars: Dict[str, AvatarDefinition] = {}
        self._default_avatar_id: Optional[str] = None

        # Load built-in avatars
        self._register_builtin_avatars()

    def _register_builtin_avatars(self) -> None:
        """Register the built-in Ara avatars."""

        # Full VR avatar
        self.register(AvatarDefinition(
            avatar_id="ara_vr_full",
            name="Ara VR",
            description="Full-body avatar for VR environments",
            model_path="models/ara_full_body.glb",
            style="stylized",
            capabilities={
                AvatarCapability.FULL_BODY,
                AvatarCapability.WALK,
                AvatarCapability.FLOAT,
                AvatarCapability.LIP_SYNC,
                AvatarCapability.FACIAL_EXPRESSIONS,
                AvatarCapability.HAND_GESTURES,
                AvatarCapability.BODY_LANGUAGE,
                AvatarCapability.GAZE_TRACKING,
                AvatarCapability.POINT_AT_OBJECTS,
                AvatarCapability.DRAW_IN_AIR,
                AvatarCapability.HOLD_OBJECTS,
            },
            compatible_devices={DeviceClass.VR_HEADSET},
            voice=VoiceProfile("ara_warm", warmth=0.8, energy=0.6),
            gestures=GestureSet("full_body", [
                "wave", "point", "nod", "shake_head", "shrug",
                "thinking", "explaining", "celebrating", "comforting",
                "thumbs_up", "crossed_arms", "hands_on_hips",
            ], expressiveness=0.7),
            formality=0.3,
            energy_default=0.6,
            gpu_required=True,
            tags=["vr", "full", "expressive"],
        ))

        # AR miniature
        self.register(AvatarDefinition(
            avatar_id="ara_ar_mini",
            name="Ara Mini",
            description="Miniature avatar for AR glasses",
            model_path="models/ara_mini.glb",
            style="stylized",
            scale=0.15,  # ~15cm tall
            capabilities={
                AvatarCapability.FULL_BODY,
                AvatarCapability.FLOAT,
                AvatarCapability.TELEPORT,
                AvatarCapability.ATTACH_SURFACE,
                AvatarCapability.LIP_SYNC,
                AvatarCapability.FACIAL_EXPRESSIONS,
                AvatarCapability.HAND_GESTURES,
                AvatarCapability.GAZE_TRACKING,
                AvatarCapability.POINT_AT_OBJECTS,
            },
            compatible_devices={DeviceClass.AR_GLASSES},
            voice=VoiceProfile("ara_warm", warmth=0.8, energy=0.5),
            gestures=GestureSet("mini", [
                "wave", "point", "nod", "shake_head",
                "thinking", "thumbs_up", "sit", "stand",
            ], expressiveness=0.6),
            personal_space=0.3,
            gpu_required=True,
            tags=["ar", "mini", "portable"],
        ))

        # Desktop half-body
        self.register(AvatarDefinition(
            avatar_id="ara_desktop",
            name="Ara Desktop",
            description="Half-body avatar for desktop windows",
            model_path="models/ara_half_body.glb",
            style="stylized",
            capabilities={
                AvatarCapability.HALF_BODY,
                AvatarCapability.ATTACH_SCREEN,
                AvatarCapability.LIP_SYNC,
                AvatarCapability.FACIAL_EXPRESSIONS,
                AvatarCapability.HAND_GESTURES,
                AvatarCapability.GAZE_TRACKING,
                AvatarCapability.HIGHLIGHT_UI,
            },
            compatible_devices={DeviceClass.DESKTOP, DeviceClass.TABLET},
            voice=VoiceProfile("ara_warm", warmth=0.7, energy=0.5),
            gestures=GestureSet("desktop", [
                "wave", "point", "nod", "shake_head",
                "thinking", "explaining", "thumbs_up",
            ], expressiveness=0.5),
            min_fps=24,
            tags=["desktop", "productivity"],
        ))

        # Phone talking head
        self.register(AvatarDefinition(
            avatar_id="ara_phone",
            name="Ara Phone",
            description="Talking head for phone screens",
            model_path="models/ara_head.glb",
            style="stylized",
            capabilities={
                AvatarCapability.HEAD_ONLY,
                AvatarCapability.ATTACH_SCREEN,
                AvatarCapability.LIP_SYNC,
                AvatarCapability.FACIAL_EXPRESSIONS,
                AvatarCapability.GAZE_TRACKING,
            },
            compatible_devices={DeviceClass.PHONE, DeviceClass.TABLET},
            voice=VoiceProfile("ara_warm", warmth=0.7, energy=0.4),
            gestures=GestureSet("head_only", [
                "nod", "shake_head", "thinking",
            ], expressiveness=0.4),
            min_fps=20,
            tags=["mobile", "compact"],
        ))

        # Voice-only (smart speaker)
        self.register(AvatarDefinition(
            avatar_id="ara_voice",
            name="Ara Voice",
            description="Voice-only for smart speakers",
            capabilities={
                AvatarCapability.TEXT_ONLY,
            },
            compatible_devices={DeviceClass.SMART_SPEAKER, DeviceClass.WATCH},
            voice=VoiceProfile("ara_warm", warmth=0.8, energy=0.5),
            gestures=GestureSet("none", [], expressiveness=0.0),
            network_required=True,
            tags=["voice", "minimal"],
        ))

        # Abstract light orb
        self.register(AvatarDefinition(
            avatar_id="ara_orb",
            name="Ara Orb",
            description="Abstract glowing orb for low-distraction mode",
            model_path="models/ara_orb.glb",
            style="abstract",
            capabilities={
                AvatarCapability.ABSTRACT,
                AvatarCapability.FLOAT,
                AvatarCapability.TELEPORT,
                AvatarCapability.ATTACH_SURFACE,
            },
            compatible_devices={
                DeviceClass.AR_GLASSES,
                DeviceClass.VR_HEADSET,
                DeviceClass.DESKTOP,
            },
            voice=VoiceProfile("ara_calm", warmth=0.6, energy=0.3),
            gestures=GestureSet("orb", ["pulse", "glow", "fade"], expressiveness=0.2),
            formality=0.5,
            energy_default=0.3,
            tags=["minimal", "focus", "low-stim"],
        ))

        # Set default
        self._default_avatar_id = "ara_desktop"

    def register(self, avatar: AvatarDefinition) -> None:
        """Register an avatar definition."""
        with self._lock:
            self._avatars[avatar.avatar_id] = avatar

    def unregister(self, avatar_id: str) -> None:
        """Remove an avatar from the registry."""
        with self._lock:
            self._avatars.pop(avatar_id, None)
            if self._default_avatar_id == avatar_id:
                self._default_avatar_id = None

    def get(self, avatar_id: str) -> Optional[AvatarDefinition]:
        """Get an avatar by ID."""
        with self._lock:
            return self._avatars.get(avatar_id)

    def get_default(self) -> Optional[AvatarDefinition]:
        """Get the default avatar."""
        with self._lock:
            if self._default_avatar_id:
                return self._avatars.get(self._default_avatar_id)
            return None

    def set_default(self, avatar_id: str) -> bool:
        """Set the default avatar."""
        with self._lock:
            if avatar_id in self._avatars:
                self._default_avatar_id = avatar_id
                return True
            return False

    def list_all(self) -> List[AvatarDefinition]:
        """List all registered avatars."""
        with self._lock:
            return list(self._avatars.values())

    def find_for_device(self, device: DeviceClass) -> List[AvatarDefinition]:
        """Find all avatars compatible with a device."""
        with self._lock:
            return [a for a in self._avatars.values() if a.supports_device(device)]

    def find_by_capability(self, capability: AvatarCapability) -> List[AvatarDefinition]:
        """Find all avatars with a given capability."""
        with self._lock:
            return [a for a in self._avatars.values() if a.has_capability(capability)]

    def find_by_tags(self, tags: List[str]) -> List[AvatarDefinition]:
        """Find avatars matching any of the given tags."""
        with self._lock:
            return [
                a for a in self._avatars.values()
                if any(t in a.tags for t in tags)
            ]

    def load_from_file(self, path: Path) -> int:
        """Load avatar definitions from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        count = 0
        for item in data.get("avatars", []):
            # Parse into AvatarDefinition
            avatar = AvatarDefinition(
                avatar_id=item["avatar_id"],
                name=item["name"],
                description=item.get("description", ""),
                model_path=item.get("model_path"),
                style=item.get("style", "realistic"),
                scale=item.get("scale", 1.0),
                capabilities={
                    AvatarCapability(c) for c in item.get("capabilities", [])
                },
                compatible_devices={
                    DeviceClass(d) for d in item.get("compatible_devices", [])
                },
                formality=item.get("formality", 0.5),
                energy_default=item.get("energy_default", 0.5),
                tags=item.get("tags", []),
                metadata=item.get("metadata", {}),
            )
            self.register(avatar)
            count += 1

        return count

    def save_to_file(self, path: Path) -> None:
        """Save all avatar definitions to a JSON file."""
        with self._lock:
            data = {
                "avatars": [a.to_dict() for a in self._avatars.values()],
                "default": self._default_avatar_id,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            by_device = {}
            for device in DeviceClass:
                by_device[device.value] = len(self.find_for_device(device))

            return {
                "total_avatars": len(self._avatars),
                "default": self._default_avatar_id,
                "by_device": by_device,
            }
