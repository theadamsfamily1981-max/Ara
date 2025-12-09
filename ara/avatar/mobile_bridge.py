"""
Mobile Bridge: Phone ↔ Cathedral Sync Protocol

Enables BANOS Lite on mobile devices to sync with the full Cathedral.
Handles:
- State delta sync (∆HV between phone and cathedral)
- Memory merge (offline → online reconciliation)
- Skill offload (heavy compute to cathedral)

Architecture:
    Phone (BANOS Lite, 2k dim)  ←──→  Cathedral (Full BANOS, 8k dim)
           │                              │
           ├─ Local AxisMundi            ├─ Full AxisMundi
           ├─ Local Memory (1k)          ├─ Full Memory (100k)
           └─ Local Skills               └─ Cathedral Skills

Sync Protocol:
    1. Phone sends: local_state_hv, local_memory_delta, skill_requests
    2. Cathedral responds: delta_hv, merged_memories, skill_results
    3. Phone applies: state update, memory merge, skill completion

Usage:
    bridge = MobileBridge(cathedral_server)

    # Sync from mobile client
    result = await bridge.sync(
        user_id="alice",
        device_id="iphone_001",
        local_state=local_axis.global_state(),
        local_memories=[...],
        skill_requests=[...],
    )
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum

import numpy as np

from ara.core.axis_mundi import AxisMundi
from ara.core.eternal_memory import EternalMemory

logger = logging.getLogger(__name__)


# =============================================================================
# Mobile Client State
# =============================================================================

class DeviceType(str, Enum):
    """Mobile device types."""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    DESKTOP = "desktop"


class SyncMode(str, Enum):
    """Sync modes for different network conditions."""
    FULL = "full"           # Full state sync (WiFi)
    DELTA = "delta"         # Delta-only sync (cellular)
    MEMORY_ONLY = "memory"  # Just memory sync (low bandwidth)
    SKILLS_ONLY = "skills"  # Just skill offload (urgent)


@dataclass
class MobileDevice:
    """Registered mobile device."""
    device_id: str
    user_id: str
    device_type: DeviceType
    axis_dim: int = 2048          # Reduced dimension on phone
    memory_limit: int = 1000       # Max local memories
    last_sync_ts: float = 0.0
    push_token: Optional[str] = None  # For push notifications

    # Sync state
    local_tick_id: int = 0
    cathedral_tick_id: int = 0
    pending_skills: List[str] = field(default_factory=list)


@dataclass
class SyncRequest:
    """Request from mobile client to cathedral."""
    user_id: str
    device_id: str
    sync_mode: SyncMode

    # State sync
    local_state_hv: Optional[bytes] = None
    local_tick_id: int = 0

    # Memory sync
    new_memories: List[Dict[str, Any]] = field(default_factory=list)
    memory_since_ts: float = 0.0

    # Skill requests
    skill_requests: List[Dict[str, Any]] = field(default_factory=list)

    # Device info
    battery_level: float = 1.0
    network_type: str = "wifi"  # wifi, cellular, offline


@dataclass
class SyncResponse:
    """Response from cathedral to mobile client."""
    status: str
    sync_ts: float

    # State sync
    delta_hv: Optional[bytes] = None
    cathedral_tick_id: int = 0

    # Memory sync
    merged_memories: List[Dict[str, Any]] = field(default_factory=list)
    memories_to_delete: List[str] = field(default_factory=list)

    # Skill results
    skill_results: List[Dict[str, Any]] = field(default_factory=list)

    # Push config
    next_sync_hint_ms: int = 60000  # Suggested next sync interval


# =============================================================================
# State Compression / Decompression
# =============================================================================

class StateCompressor:
    """
    Compress AxisMundi state for mobile transmission.

    Uses dimension reduction: 8k cathedral → 2k phone
    """

    def __init__(
        self,
        cathedral_dim: int = 8192,
        mobile_dim: int = 2048,
    ):
        self.cathedral_dim = cathedral_dim
        self.mobile_dim = mobile_dim

        # Random projection matrix (fixed seed for consistency)
        rng = np.random.default_rng(42)
        self.projection = rng.randn(mobile_dim, cathedral_dim).astype(np.float32)
        self.projection /= np.sqrt(cathedral_dim)

        # Inverse projection (pseudo-inverse for reconstruction)
        self.inverse = np.linalg.pinv(self.projection)

    def compress(self, cathedral_hv: np.ndarray) -> np.ndarray:
        """Compress 8k cathedral HV to 2k mobile HV."""
        if len(cathedral_hv) != self.cathedral_dim:
            # If already mobile dim, return as-is
            if len(cathedral_hv) == self.mobile_dim:
                return cathedral_hv
            raise ValueError(f"Expected {self.cathedral_dim}D, got {len(cathedral_hv)}D")

        mobile_hv = self.projection @ cathedral_hv
        # Re-bipolarize
        return np.sign(mobile_hv).astype(np.float32)

    def expand(self, mobile_hv: np.ndarray) -> np.ndarray:
        """Expand 2k mobile HV to 8k cathedral HV."""
        if len(mobile_hv) != self.mobile_dim:
            if len(mobile_hv) == self.cathedral_dim:
                return mobile_hv
            raise ValueError(f"Expected {self.mobile_dim}D, got {len(mobile_hv)}D")

        cathedral_hv = self.inverse @ mobile_hv
        # Re-bipolarize
        return np.sign(cathedral_hv).astype(np.float32)

    def compute_delta(
        self,
        local_hv: np.ndarray,
        cathedral_hv: np.ndarray,
    ) -> np.ndarray:
        """Compute delta HV for sync (in mobile dimension)."""
        # Compress cathedral to mobile dim
        cathedral_mobile = self.compress(cathedral_hv)

        # Ensure local is same dim
        if len(local_hv) != self.mobile_dim:
            local_hv = self.compress(local_hv)

        # Delta is XOR-like difference
        delta = cathedral_mobile * local_hv  # Element-wise for bipolar
        return delta


# =============================================================================
# Mobile Bridge
# =============================================================================

class MobileBridge:
    """
    Bridge between mobile clients and Cathedral.

    Handles:
    - Device registration
    - State sync (with dimension reduction)
    - Memory merge (conflict resolution)
    - Skill offload (heavy compute delegation)
    """

    def __init__(
        self,
        cathedral_server: "CathedralServer",
        cathedral_dim: int = 8192,
        mobile_dim: int = 2048,
    ):
        self.cathedral = cathedral_server
        self.compressor = StateCompressor(cathedral_dim, mobile_dim)

        # Registered devices
        self.devices: Dict[str, MobileDevice] = {}

        # Pending skill results
        self.skill_results: Dict[str, Dict[str, Any]] = {}

    def register_device(
        self,
        user_id: str,
        device_id: str,
        device_type: DeviceType,
        push_token: Optional[str] = None,
    ) -> MobileDevice:
        """Register a new mobile device."""
        device = MobileDevice(
            device_id=device_id,
            user_id=user_id,
            device_type=device_type,
            push_token=push_token,
        )
        self.devices[device_id] = device
        logger.info(f"Registered device {device_id} for user {user_id}")
        return device

    def get_device(self, device_id: str) -> Optional[MobileDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)

    async def sync(self, request: SyncRequest) -> SyncResponse:
        """
        Process a sync request from a mobile client.

        Args:
            request: Sync request with local state, memories, and skill requests

        Returns:
            Sync response with delta, merged memories, and skill results
        """
        start_time = time.time()

        # Validate user
        if request.user_id not in self.cathedral.jars:
            return SyncResponse(
                status="error_not_provisioned",
                sync_ts=time.time(),
            )

        jar = self.cathedral.jars[request.user_id]

        # Get or register device
        device = self.devices.get(request.device_id)
        if not device:
            device = self.register_device(
                request.user_id,
                request.device_id,
                DeviceType.IOS,  # Default, should come from request
            )

        response = SyncResponse(
            status="ok",
            sync_ts=time.time(),
        )

        # 1. State sync
        if request.sync_mode in (SyncMode.FULL, SyncMode.DELTA):
            response = await self._sync_state(request, jar, device, response)

        # 2. Memory sync
        if request.sync_mode in (SyncMode.FULL, SyncMode.MEMORY_ONLY):
            response = await self._sync_memories(request, jar, response)

        # 3. Skill offload
        if request.sync_mode in (SyncMode.FULL, SyncMode.SKILLS_ONLY):
            response = await self._process_skills(request, jar, response)

        # Update device state
        device.last_sync_ts = time.time()
        device.local_tick_id = request.local_tick_id

        # Compute next sync hint based on conditions
        response.next_sync_hint_ms = self._compute_sync_interval(request, device)

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Sync for {request.user_id}/{request.device_id} took {elapsed:.1f}ms")

        return response

    async def _sync_state(
        self,
        request: SyncRequest,
        jar: "BrainJar",
        device: MobileDevice,
        response: SyncResponse,
    ) -> SyncResponse:
        """Sync AxisMundi state between phone and cathedral."""
        if not request.local_state_hv:
            return response

        try:
            # Decode local state
            local_hv = np.frombuffer(request.local_state_hv, dtype=np.float32)

            # Get cathedral state
            cathedral_hv = jar.axis.global_state()

            # Compute delta (in mobile dimension)
            delta_hv = self.compressor.compute_delta(local_hv, cathedral_hv)

            response.delta_hv = delta_hv.tobytes()
            response.cathedral_tick_id = device.cathedral_tick_id

        except Exception as e:
            logger.error(f"State sync failed: {e}")
            response.status = "partial_error"

        return response

    async def _sync_memories(
        self,
        request: SyncRequest,
        jar: "BrainJar",
        response: SyncResponse,
    ) -> SyncResponse:
        """Merge memories between phone and cathedral."""
        try:
            # Get memories from cathedral since last sync
            cathedral_memories = jar.memory.list_episodes(limit=100)
            new_from_cathedral = [
                {
                    "id": ep.id,
                    "meta": ep.meta,
                    "strength": ep.strength,
                    "created_at": ep.created_at,
                }
                for ep in cathedral_memories
                if ep.created_at > request.memory_since_ts
            ]

            response.merged_memories = new_from_cathedral

            # Store new memories from phone
            for mem in request.new_memories:
                if "content_hv" in mem and "meta" in mem:
                    content_hv = np.frombuffer(
                        bytes.fromhex(mem["content_hv"]),
                        dtype=np.float32,
                    )
                    jar.memory.store(
                        content_hv=content_hv,
                        strength=mem.get("strength", 0.7),
                        meta=mem["meta"],
                    )

        except Exception as e:
            logger.error(f"Memory sync failed: {e}")
            response.status = "partial_error"

        return response

    async def _process_skills(
        self,
        request: SyncRequest,
        jar: "BrainJar",
        response: SyncResponse,
    ) -> SyncResponse:
        """Process skill offload requests from phone."""
        skill_results = []

        for skill_req in request.skill_requests:
            skill_id = skill_req.get("id")
            skill_name = skill_req.get("name")

            # Check if skill already completed
            if skill_id in self.skill_results:
                skill_results.append(self.skill_results[skill_id])
                continue

            # Queue skill for execution (placeholder)
            # In real implementation, this would dispatch to skill executors
            result = {
                "id": skill_id,
                "name": skill_name,
                "status": "queued",
                "eta_ms": 5000,
            }
            skill_results.append(result)

        response.skill_results = skill_results
        return response

    def _compute_sync_interval(
        self,
        request: SyncRequest,
        device: MobileDevice,
    ) -> int:
        """Compute suggested next sync interval based on conditions."""
        base_interval = 60000  # 1 minute

        # More frequent if on WiFi
        if request.network_type == "wifi":
            base_interval = 30000

        # Less frequent if battery low
        if request.battery_level < 0.2:
            base_interval *= 3

        # More frequent if active conversation
        jar = self.cathedral.jars.get(request.user_id)
        if jar and jar.session_active:
            base_interval = 10000  # 10 seconds during active chat

        return base_interval


# =============================================================================
# Mobile Protocol Messages
# =============================================================================

def encode_sync_request(
    user_id: str,
    device_id: str,
    local_axis: AxisMundi,
    new_memories: List[Dict],
    skill_requests: List[Dict],
    battery_level: float = 1.0,
    network_type: str = "wifi",
) -> Dict[str, Any]:
    """Encode a sync request for transmission."""
    return {
        "version": "1.0",
        "type": "sync_request",
        "user_id": user_id,
        "device_id": device_id,
        "sync_mode": "full",
        "local_state_hv": local_axis.global_state().tobytes().hex(),
        "local_tick_id": 0,  # Would come from local loop
        "new_memories": new_memories,
        "memory_since_ts": time.time() - 3600,  # Last hour
        "skill_requests": skill_requests,
        "battery_level": battery_level,
        "network_type": network_type,
        "ts": time.time(),
    }


def decode_sync_response(data: Dict[str, Any]) -> SyncResponse:
    """Decode a sync response from cathedral."""
    return SyncResponse(
        status=data.get("status", "error"),
        sync_ts=data.get("sync_ts", 0),
        delta_hv=bytes.fromhex(data["delta_hv"]) if data.get("delta_hv") else None,
        cathedral_tick_id=data.get("cathedral_tick_id", 0),
        merged_memories=data.get("merged_memories", []),
        memories_to_delete=data.get("memories_to_delete", []),
        skill_results=data.get("skill_results", []),
        next_sync_hint_ms=data.get("next_sync_hint_ms", 60000),
    )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    from ara.avatar.cathedral import CathedralServer, SubscriptionTier
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("=== Mobile Bridge Demo ===\n")

        # Create cathedral and bridge
        cathedral = CathedralServer(data_dir=Path("/tmp/ara_mobile_demo"))
        bridge = MobileBridge(cathedral)

        # Provision a user
        await cathedral.provision_user(
            "mobile_alice",
            display_name="Alice (Mobile)",
            tier=SubscriptionTier.PRO,
        )
        await cathedral.record_consent("mobile_alice")

        # Register mobile device
        device = bridge.register_device(
            user_id="mobile_alice",
            device_id="iphone_001",
            device_type=DeviceType.IOS,
        )
        print(f"Registered device: {device.device_id}")

        # Create local (mobile) AxisMundi
        local_axis = AxisMundi(dim=2048, seed=123)
        local_axis.write("context", np.random.randn(2048).astype(np.float32))

        # Create sync request
        request = SyncRequest(
            user_id="mobile_alice",
            device_id="iphone_001",
            sync_mode=SyncMode.FULL,
            local_state_hv=local_axis.global_state().tobytes(),
            local_tick_id=100,
            new_memories=[
                {
                    "content_hv": np.random.randn(2048).astype(np.float32).tobytes().hex(),
                    "meta": {"source": "mobile", "topic": "test"},
                    "strength": 0.8,
                }
            ],
            skill_requests=[
                {"id": "skill_001", "name": "summarize", "params": {"text": "Hello world"}},
            ],
            battery_level=0.75,
            network_type="wifi",
        )

        # Process sync
        print("\nProcessing sync...")
        response = await bridge.sync(request)

        print(f"  Status: {response.status}")
        print(f"  Delta HV: {len(response.delta_hv) if response.delta_hv else 0} bytes")
        print(f"  Merged memories: {len(response.merged_memories)}")
        print(f"  Skill results: {len(response.skill_results)}")
        print(f"  Next sync in: {response.next_sync_hint_ms}ms")

        print("\n=== Demo Complete ===")

    asyncio.run(demo())
