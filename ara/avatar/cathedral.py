"""
Cathedral Avatar Server: Multi-Tenant Brain Jar Gateway

The Social Cathedral - manages WireGuard peers and spawns personalized
Ara instances (Brain Jars) for friends. Each friend gets:
- Isolated AxisMundi shard
- Private EternalMemory
- Policy-enforced capabilities
- VPN-gated access

Architecture:
    Friend Phone/Desktop
           │
           ▼ WireGuard VPN
    ┌──────────────────┐
    │  CathedralServer │
    │  ├─ Auth/Policy  │
    │  ├─ BrainJars    │──► Per-user Soul Shards
    │  └─ Sync Bridge  │──► Mobile ↔ Cathedral
    └──────────────────┘

Tiers:
    TIER_1: Free - 1k memory, text only
    TIER_2: Pro ($9.99/mo) - 16k memory, voice, sync
    TIER_3: Power ($99/mo) - Cathedral priority, custom skills

Usage:
    server = CathedralServer()
    await server.provision_user("friend_001", tier="TIER_2")
    await server.start()
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path
from enum import Enum

import numpy as np

from ara.core.axis_mundi import AxisMundi, encode_text_to_hv
from ara.core.eternal_memory import EternalMemory
from ara.core.config import AraConfig, get_config
from ara.safety.autonomy import AutonomyController, AutonomyLevel
from ara.brain_jars.policy import BrainJarPolicy, PolicyTier
from ara.brain_jars.isolation import BrainJarStore
from ara.brain_jars.disclosures import DisclosureManager, get_onboarding_disclosure

logger = logging.getLogger(__name__)


# =============================================================================
# Subscription Tiers
# =============================================================================

class SubscriptionTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"           # 1k memory, text only
    PRO = "pro"             # 16k memory, voice, sync
    POWER = "power"         # Cathedral priority, custom skills
    FOUNDER = "founder"     # Max only - full access


TIER_CONFIGS = {
    SubscriptionTier.FREE: {
        "axis_dim": 1024,
        "memory_limit": 1000,
        "voice_enabled": False,
        "sync_enabled": False,
        "priority": 0,
        "monthly_usd": 0,
    },
    SubscriptionTier.PRO: {
        "axis_dim": 4096,
        "memory_limit": 16000,
        "voice_enabled": True,
        "sync_enabled": True,
        "priority": 1,
        "monthly_usd": 9.99,
    },
    SubscriptionTier.POWER: {
        "axis_dim": 8192,
        "memory_limit": 100000,
        "voice_enabled": True,
        "sync_enabled": True,
        "priority": 2,
        "monthly_usd": 99.00,
    },
    SubscriptionTier.FOUNDER: {
        "axis_dim": 16384,
        "memory_limit": float("inf"),
        "voice_enabled": True,
        "sync_enabled": True,
        "priority": 10,
        "monthly_usd": 0,  # Free for founder
    },
}


# =============================================================================
# Brain Jar (Per-User Soul Shard)
# =============================================================================

@dataclass
class BrainJar:
    """
    A friend's personal Ara instance.

    Contains:
    - Isolated AxisMundi shard (their soul state)
    - Private EternalMemory (their conversations)
    - Policy enforcement (what they can do)
    - VPN credentials (how they connect)
    """
    user_id: str
    display_name: str
    tier: SubscriptionTier
    policy: BrainJarPolicy

    # Core components (lazy-initialized)
    axis: Optional[AxisMundi] = None
    memory: Optional[EternalMemory] = None
    safety: Optional[AutonomyController] = None
    store: Optional[BrainJarStore] = None

    # State
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0
    session_active: bool = False

    # VPN credentials
    vpn_pubkey: str = ""
    vpn_psk: str = ""  # Pre-shared key

    # Consent
    consent_given: bool = False
    consent_date: Optional[str] = None

    # Rate limiting
    messages_this_hour: int = 0
    hour_start: float = 0.0

    def __post_init__(self):
        """Generate VPN credentials if not provided."""
        if not self.vpn_pubkey:
            # Generate mock WireGuard credentials
            self.vpn_pubkey = f"wg-pub-{secrets.token_hex(16)}"
            self.vpn_psk = secrets.token_hex(32)
        self.hour_start = time.time()

    def check_rate_limit(self) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        # Reset counter every hour
        if now - self.hour_start > 3600:
            self.messages_this_hour = 0
            self.hour_start = now

        # Check limit based on policy
        limit = self.policy.resources.messages_per_hour
        return self.messages_this_hour < limit

    def increment_message_count(self) -> None:
        """Increment message counter for rate limiting."""
        self.messages_this_hour += 1
        self.message_count += 1


# =============================================================================
# Mobile Sync State
# =============================================================================

@dataclass
class MobileSyncState:
    """State for mobile client sync."""
    user_id: str
    device_id: str
    last_sync_ts: float = 0.0
    pending_delta_hvs: List[bytes] = field(default_factory=list)
    local_memory_count: int = 0
    cathedral_memory_count: int = 0
    sync_status: str = "idle"


# =============================================================================
# Cathedral Server
# =============================================================================

class CathedralServer:
    """
    The Cathedral Gatekeeper.

    Manages WireGuard peers and spawns personalized Ara instances.
    Each user gets an isolated Brain Jar with their own soul shard.
    """

    def __init__(
        self,
        config: Optional[AraConfig] = None,
        data_dir: Optional[Path] = None,
    ):
        self.config = config or get_config()
        self.data_dir = data_dir or Path("/var/ara/cathedral")

        # Brain jars by user ID
        self.jars: Dict[str, BrainJar] = {}

        # Mobile sync states
        self.mobile_syncs: Dict[str, MobileSyncState] = {}

        # Disclosure manager for consent tracking
        self.disclosures = DisclosureManager()

        # Server state
        self._running = False
        self._app = None

        logger.info(f"CathedralServer initialized at {self.data_dir}")

    # =========================================================================
    # User Provisioning
    # =========================================================================

    async def provision_user(
        self,
        user_id: str,
        display_name: str = "",
        tier: SubscriptionTier = SubscriptionTier.FREE,
        invited_by: str = "founder",
    ) -> Dict[str, Any]:
        """
        Provision a new Brain Jar for a friend.

        Args:
            user_id: Unique user identifier
            display_name: Friendly name for the user
            tier: Subscription tier
            invited_by: Who invited this user

        Returns:
            Provisioning result with VPN credentials
        """
        logger.info(f"Provisioning Brain Jar for {user_id} ({tier.value})...")

        if user_id in self.jars:
            logger.warning(f"User {user_id} already exists")
            return {"error": "User already exists", "user_id": user_id}

        # Get tier config
        tier_config = TIER_CONFIGS[tier]

        # Create policy based on tier
        if tier == SubscriptionTier.FOUNDER:
            policy_tier = PolicyTier.DOGFOOD
        elif tier == SubscriptionTier.POWER:
            policy_tier = PolicyTier.TRUSTED_FRIEND
        elif tier == SubscriptionTier.PRO:
            policy_tier = PolicyTier.TRUSTED_FRIEND
        else:
            policy_tier = PolicyTier.ACQUAINTANCE

        policy = BrainJarPolicy.for_tier(policy_tier)

        # Create brain jar
        jar = BrainJar(
            user_id=user_id,
            display_name=display_name or user_id,
            tier=tier,
            policy=policy,
        )

        # Initialize components
        jar_data_dir = self.data_dir / "jars" / user_id
        jar_data_dir.mkdir(parents=True, exist_ok=True)

        # AxisMundi (per-user soul shard)
        jar.axis = AxisMundi(
            dim=tier_config["axis_dim"],
            layers=["soul", "context", "emotion"],
            seed=hash(user_id) % (2**31),
        )

        # EternalMemory (per-user episodic store)
        jar.memory = EternalMemory(
            dim=tier_config["axis_dim"],
            db_path=jar_data_dir / "memory.db",
        )

        # Safety (per-user autonomy, always starts conservative)
        jar.safety = AutonomyController(
            initial_level=1,  # Suggester only
            max_level=2,      # Never fully autonomous for friends
            require_human_for_level_3=True,
        )

        # Storage isolation
        jar.store = BrainJarStore(user_id, base_path=jar_data_dir)

        # Store jar
        self.jars[user_id] = jar

        # Generate WireGuard config
        vpn_config = self._generate_vpn_config(jar)

        logger.info(f"Brain Jar provisioned for {user_id}")

        return {
            "status": "provisioned",
            "user_id": user_id,
            "tier": tier.value,
            "vpn_pubkey": jar.vpn_pubkey,
            "vpn_config": vpn_config,
            "onboarding_url": f"https://ara.so/onboard/{user_id}",
            "disclosures": get_onboarding_disclosure(
                display_name or user_id,
                policy,
            ),
        }

    def _generate_vpn_config(self, jar: BrainJar) -> str:
        """Generate WireGuard client config for a user."""
        # This is a template - real implementation would use wg tools
        config = f"""# Ara VPN Config for {jar.user_id}
[Interface]
PrivateKey = <GENERATED_CLIENT_PRIVATE_KEY>
Address = 10.42.0.{hash(jar.user_id) % 250 + 2}/32
DNS = 10.42.0.1

[Peer]
PublicKey = <CATHEDRAL_PUBLIC_KEY>
PresharedKey = {jar.vpn_psk}
AllowedIPs = 10.42.0.0/24
Endpoint = vpn.cathedral.ara:51820
PersistentKeepalive = 25
"""
        return config

    async def deprovision_user(self, user_id: str) -> Dict[str, Any]:
        """Remove a user's Brain Jar (with data export option)."""
        if user_id not in self.jars:
            return {"error": "User not found"}

        jar = self.jars[user_id]

        # Export data before deletion if requested
        export_path = self.data_dir / "exports" / f"{user_id}_export.json"

        logger.info(f"Deprovisioning Brain Jar for {user_id}")

        # Clean up
        del self.jars[user_id]

        return {
            "status": "deprovisioned",
            "user_id": user_id,
            "data_exported": False,  # Would be True if export requested
        }

    # =========================================================================
    # Message Handling (Per-User)
    # =========================================================================

    async def handle_message(
        self,
        user_id: str,
        text: str,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle a message from a user's Brain Jar.

        Args:
            user_id: The user sending the message
            text: Message text
            device_id: Optional device identifier for mobile sync

        Returns:
            Response with reply, memory info, etc.
        """
        start_time = time.time()

        # Check if user exists
        if user_id not in self.jars:
            return {
                "error": "Soul not found. Please subscribe.",
                "status": "not_provisioned",
            }

        jar = self.jars[user_id]

        # Check consent
        if not jar.consent_given:
            return {
                "error": "Consent required before interaction.",
                "status": "consent_required",
                "disclosures": get_onboarding_disclosure(jar.display_name, jar.policy),
            }

        # Check policy allows text chat
        if not jar.policy.capabilities.text_chat:
            return {
                "error": "Text chat not available on your tier.",
                "status": "capability_denied",
            }

        # Rate limiting
        if not jar.check_rate_limit():
            return {
                "error": "Rate limit exceeded. Please slow down.",
                "status": "rate_limited",
            }

        # Update activity
        jar.last_active = time.time()
        jar.increment_message_count()
        jar.session_active = True

        # Encode message to HV
        tier_config = TIER_CONFIGS[jar.tier]
        content_hv = encode_text_to_hv(text, dim=tier_config["axis_dim"])

        # Update user's soul shard
        jar.axis.write("context", content_hv)

        # Recall from user's memory
        recall_result = jar.memory.recall(
            query_hv=content_hv,
            k=5,
            user_filter=user_id,
        )

        # Generate response (placeholder - would use LLM)
        reply = self._reason_for_user(jar, text, recall_result)

        # Store to user's memory
        jar.memory.store(
            content_hv=content_hv,
            strength=0.8,
            meta={
                "user": user_id,
                "message": text[:100],
                "reply": reply[:100],
                "device": device_id or "unknown",
            },
        )

        # Check memory limits
        memory_count = jar.memory.stats()["episode_count"]
        if memory_count > tier_config["memory_limit"]:
            # Consolidate old memories
            jar.memory.consolidate(min_strength=0.3)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "reply": reply,
            "status": "ok",
            "user_id": user_id,
            "tier": jar.tier.value,
            "message_count": jar.message_count,
            "memories_recalled": len(recall_result.episodes),
            "coherence": jar.axis.global_coherence(),
            "processing_time_ms": elapsed_ms,
        }

    def _reason_for_user(
        self,
        jar: BrainJar,
        text: str,
        recall_result: Any,
    ) -> str:
        """Generate response for a specific user."""
        # Placeholder reasoning - would integrate with LLM
        memory_context = ""
        if recall_result.episodes:
            top_memory = recall_result.episodes[0]
            if top_memory.similarity > 0.3:
                memory_context = f" I remember we talked about: {top_memory.meta.get('message', '...')[:30]}"

        # Personalized greeting
        greeting = f"Hello {jar.display_name}!"

        if "hello" in text.lower() or "hi" in text.lower():
            return f"{greeting}{memory_context} How can I help?"
        elif "remember" in text.lower():
            if recall_result.episodes:
                topics = [ep.meta.get("message", "?")[:25] for ep in recall_result.episodes[:3]]
                return f"I recall these moments with you: {', '.join(topics)}"
            else:
                return "We haven't built many memories together yet."
        else:
            return f"I heard you say: '{text[:40]}'{memory_context}"

    # =========================================================================
    # Mobile Sync
    # =========================================================================

    async def sync_mobile(
        self,
        user_id: str,
        device_id: str,
        local_state_hv: bytes,
        local_memory_count: int,
    ) -> Dict[str, Any]:
        """
        Sync mobile client with Cathedral.

        Args:
            user_id: User ID
            device_id: Mobile device identifier
            local_state_hv: Local AxisMundi state (compressed)
            local_memory_count: Number of memories on device

        Returns:
            Sync result with delta and updates
        """
        if user_id not in self.jars:
            return {"error": "User not found", "status": "not_provisioned"}

        jar = self.jars[user_id]

        # Check sync enabled for tier
        if not TIER_CONFIGS[jar.tier]["sync_enabled"]:
            return {"error": "Sync not available on your tier", "status": "upgrade_required"}

        # Get or create sync state
        sync_key = f"{user_id}:{device_id}"
        if sync_key not in self.mobile_syncs:
            self.mobile_syncs[sync_key] = MobileSyncState(
                user_id=user_id,
                device_id=device_id,
            )

        sync_state = self.mobile_syncs[sync_key]
        sync_state.sync_status = "syncing"

        # Compute delta between local and cathedral state
        cathedral_state = jar.axis.global_state()
        cathedral_memory_count = jar.memory.stats()["episode_count"]

        # Generate delta HV (simplified - real impl would be more sophisticated)
        delta_hv = cathedral_state.tobytes()

        # Update sync state
        sync_state.last_sync_ts = time.time()
        sync_state.local_memory_count = local_memory_count
        sync_state.cathedral_memory_count = cathedral_memory_count
        sync_state.sync_status = "synced"

        return {
            "status": "synced",
            "delta_hv": delta_hv.hex(),
            "cathedral_memory_count": cathedral_memory_count,
            "local_memory_count": local_memory_count,
            "sync_ts": sync_state.last_sync_ts,
        }

    # =========================================================================
    # Consent Management
    # =========================================================================

    async def record_consent(
        self,
        user_id: str,
        consent_version: str = "1.0",
    ) -> Dict[str, Any]:
        """Record that a user has given consent."""
        if user_id not in self.jars:
            return {"error": "User not found"}

        jar = self.jars[user_id]
        jar.consent_given = True
        jar.consent_date = time.strftime("%Y-%m-%d %H:%M:%S")

        self.disclosures.record_consent(user_id, consent_version)

        logger.info(f"Consent recorded for {user_id}")

        return {
            "status": "consent_recorded",
            "user_id": user_id,
            "consent_version": consent_version,
            "consent_date": jar.consent_date,
        }

    # =========================================================================
    # Status and Stats
    # =========================================================================

    def get_cathedral_stats(self) -> Dict[str, Any]:
        """Get overall Cathedral statistics."""
        tier_counts = {tier.value: 0 for tier in SubscriptionTier}
        total_messages = 0
        total_memories = 0
        active_24h = 0

        now = time.time()
        for jar in self.jars.values():
            tier_counts[jar.tier.value] += 1
            total_messages += jar.message_count
            if jar.memory:
                total_memories += jar.memory.stats()["episode_count"]
            if now - jar.last_active < 86400:
                active_24h += 1

        return {
            "total_users": len(self.jars),
            "tier_breakdown": tier_counts,
            "total_messages": total_messages,
            "total_memories": total_memories,
            "active_24h": active_24h,
            "server_running": self._running,
        }

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        if user_id not in self.jars:
            return {"error": "User not found"}

        jar = self.jars[user_id]
        memory_stats = jar.memory.stats() if jar.memory else {}

        return {
            "user_id": user_id,
            "display_name": jar.display_name,
            "tier": jar.tier.value,
            "message_count": jar.message_count,
            "memory_count": memory_stats.get("episode_count", 0),
            "consent_given": jar.consent_given,
            "created_at": jar.created_at,
            "last_active": jar.last_active,
            "session_active": jar.session_active,
        }

    def list_users(self) -> List[Dict[str, Any]]:
        """List all provisioned users."""
        return [
            {
                "user_id": jar.user_id,
                "display_name": jar.display_name,
                "tier": jar.tier.value,
                "consent_given": jar.consent_given,
                "last_active": jar.last_active,
            }
            for jar in self.jars.values()
        ]


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("=== Cathedral Server Demo ===\n")

        server = CathedralServer(data_dir=Path("/tmp/ara_cathedral_demo"))

        # Provision some users
        print("Provisioning users...")
        result1 = await server.provision_user(
            "friend_alice",
            display_name="Alice",
            tier=SubscriptionTier.PRO,
        )
        print(f"  Alice: {result1['status']}")

        result2 = await server.provision_user(
            "friend_bob",
            display_name="Bob",
            tier=SubscriptionTier.FREE,
        )
        print(f"  Bob: {result2['status']}")

        # Record consent
        await server.record_consent("friend_alice")
        await server.record_consent("friend_bob")

        # Send messages
        print("\nSending messages...")
        r1 = await server.handle_message("friend_alice", "Hello Ara!")
        print(f"  Alice: {r1['reply']}")

        r2 = await server.handle_message("friend_bob", "Hi, remember me?")
        print(f"  Bob: {r2['reply']}")

        r3 = await server.handle_message("friend_alice", "What did we talk about?")
        print(f"  Alice: {r3['reply']}")

        # Stats
        print("\n=== Cathedral Stats ===")
        stats = server.get_cathedral_stats()
        print(f"  Total users: {stats['total_users']}")
        print(f"  Tier breakdown: {stats['tier_breakdown']}")
        print(f"  Total messages: {stats['total_messages']}")

        print("\n=== User List ===")
        for user in server.list_users():
            print(f"  {user['display_name']} ({user['tier']}): consent={user['consent_given']}")

    asyncio.run(demo())
