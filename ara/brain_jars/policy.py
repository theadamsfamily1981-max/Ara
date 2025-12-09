"""
Brain Jar Policy Schema

Defines what each friend tenant can and cannot do.
Policies are stored per-user and enforced at the API boundary.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class PolicyTier(str, Enum):
    """
    Pre-defined policy tiers for quick setup.

    DOGFOOD: Internal testing (you as friend_000)
    TRUSTED_FRIEND: Close friends, more latitude
    ACQUAINTANCE: More restricted, shorter retention
    PUBLIC_BETA: Maximum restrictions, explicit consent
    """
    DOGFOOD = "dogfood"
    TRUSTED_FRIEND = "trusted_friend"
    ACQUAINTANCE = "acquaintance"
    PUBLIC_BETA = "public_beta"


@dataclass
class ResourceLimits:
    """Resource envelope for this tenant."""
    memory_mb: int = 256          # EternalMemory shard size
    session_minutes: int = 60     # Max session duration
    daily_minutes: int = 180      # Daily usage cap
    messages_per_hour: int = 100  # Rate limit
    max_context_tokens: int = 8192  # LLM context window


@dataclass
class RetentionPolicy:
    """Data retention rules."""
    conversation_days: int = 30   # How long to keep chat logs
    memory_days: int = 90         # How long to keep episodic memory
    voice_retention: bool = False # Store voice recordings?
    allow_export: bool = True     # Can user download their data?
    allow_delete: bool = True     # Can user nuke their jar?


@dataclass
class CapabilityFlags:
    """What this jar is allowed to do."""
    # Conversation
    text_chat: bool = True
    voice_chat: bool = True
    avatar_video: bool = True

    # Memory
    episodic_memory: bool = True
    emotional_tracking: bool = True
    preference_learning: bool = True

    # Tools (all default OFF for safety)
    file_upload: bool = False
    code_execution: bool = False
    web_search: bool = False
    calendar_access: bool = False

    # NEVER for friend jars
    hardware_control: bool = False
    bios_access: bool = False
    fpga_control: bool = False
    network_config: bool = False
    wallet_access: bool = False
    founder_data_access: bool = False


@dataclass
class SafetyConfig:
    """Safety guardrails for this jar."""
    show_ai_disclaimer: bool = True
    show_crisis_resources: bool = True
    crisis_hotline: str = "988 (US) or your local emergency services"
    no_therapy_claims: bool = True
    hallucination_warning: bool = True
    max_emotional_intensity: float = 0.7  # Cap arousal/valence extremes


@dataclass
class BrainJarPolicy:
    """
    Complete policy for a brain jar tenant.

    Example usage:
        policy = BrainJarPolicy.for_tier(PolicyTier.TRUSTED_FRIEND)
        policy.user_id = "max_001"
        policy.display_name = "Max"
        save_policy(policy, "/var/ara/brain_jars/max_001/policy.yaml")
    """
    # Identity
    user_id: str = ""
    display_name: str = ""
    tier: PolicyTier = PolicyTier.ACQUAINTANCE

    # Consent
    consent_given: bool = False
    consent_timestamp: Optional[str] = None
    consent_version: str = "1.0"

    # Sub-policies
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    capabilities: CapabilityFlags = field(default_factory=CapabilityFlags)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""

    @classmethod
    def for_tier(cls, tier: PolicyTier) -> "BrainJarPolicy":
        """Create a policy with tier-appropriate defaults."""
        policy = cls(tier=tier)

        if tier == PolicyTier.DOGFOOD:
            # Internal testing - maximum access
            policy.resources = ResourceLimits(
                memory_mb=1024,
                session_minutes=480,
                daily_minutes=720,
                messages_per_hour=500,
                max_context_tokens=32768,
            )
            policy.retention = RetentionPolicy(
                conversation_days=365,
                memory_days=365,
                voice_retention=True,
                allow_export=True,
                allow_delete=True,
            )
            policy.capabilities = CapabilityFlags(
                text_chat=True,
                voice_chat=True,
                avatar_video=True,
                episodic_memory=True,
                emotional_tracking=True,
                preference_learning=True,
                file_upload=True,
                code_execution=False,  # Still no code exec
                web_search=True,
                calendar_access=True,
                # STILL NEVER for any jar
                hardware_control=False,
                bios_access=False,
                fpga_control=False,
                network_config=False,
                wallet_access=False,
                founder_data_access=False,
            )

        elif tier == PolicyTier.TRUSTED_FRIEND:
            # Close friends - generous but bounded
            policy.resources = ResourceLimits(
                memory_mb=512,
                session_minutes=120,
                daily_minutes=360,
                messages_per_hour=200,
                max_context_tokens=16384,
            )
            policy.retention = RetentionPolicy(
                conversation_days=90,
                memory_days=180,
                voice_retention=False,
                allow_export=True,
                allow_delete=True,
            )
            policy.capabilities = CapabilityFlags(
                text_chat=True,
                voice_chat=True,
                avatar_video=True,
                episodic_memory=True,
                emotional_tracking=True,
                preference_learning=True,
                file_upload=False,
                code_execution=False,
                web_search=True,
                calendar_access=False,
            )

        elif tier == PolicyTier.ACQUAINTANCE:
            # Default - restricted
            policy.resources = ResourceLimits(
                memory_mb=256,
                session_minutes=60,
                daily_minutes=120,
                messages_per_hour=100,
                max_context_tokens=8192,
            )
            policy.retention = RetentionPolicy(
                conversation_days=30,
                memory_days=60,
                voice_retention=False,
                allow_export=True,
                allow_delete=True,
            )
            policy.capabilities = CapabilityFlags(
                text_chat=True,
                voice_chat=True,
                avatar_video=False,
                episodic_memory=True,
                emotional_tracking=False,
                preference_learning=True,
            )

        elif tier == PolicyTier.PUBLIC_BETA:
            # Maximum restrictions
            policy.resources = ResourceLimits(
                memory_mb=128,
                session_minutes=30,
                daily_minutes=60,
                messages_per_hour=50,
                max_context_tokens=4096,
            )
            policy.retention = RetentionPolicy(
                conversation_days=7,
                memory_days=14,
                voice_retention=False,
                allow_export=True,
                allow_delete=True,
            )
            policy.capabilities = CapabilityFlags(
                text_chat=True,
                voice_chat=False,
                avatar_video=False,
                episodic_memory=False,
                emotional_tracking=False,
                preference_learning=False,
            )

        return policy

    def record_consent(self) -> None:
        """Mark that user has given informed consent."""
        self.consent_given = True
        self.consent_timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "tier": self.tier.value,
            "consent_given": self.consent_given,
            "consent_timestamp": self.consent_timestamp,
            "consent_version": self.consent_version,
            "resources": asdict(self.resources),
            "retention": asdict(self.retention),
            "capabilities": asdict(self.capabilities),
            "safety": asdict(self.safety),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainJarPolicy":
        """Create from dictionary."""
        policy = cls()
        policy.user_id = data.get("user_id", "")
        policy.display_name = data.get("display_name", "")
        policy.tier = PolicyTier(data.get("tier", "acquaintance"))
        policy.consent_given = data.get("consent_given", False)
        policy.consent_timestamp = data.get("consent_timestamp")
        policy.consent_version = data.get("consent_version", "1.0")
        policy.created_at = data.get("created_at", "")
        policy.updated_at = data.get("updated_at", "")
        policy.notes = data.get("notes", "")

        if "resources" in data:
            policy.resources = ResourceLimits(**data["resources"])
        if "retention" in data:
            policy.retention = RetentionPolicy(**data["retention"])
        if "capabilities" in data:
            policy.capabilities = CapabilityFlags(**data["capabilities"])
        if "safety" in data:
            policy.safety = SafetyConfig(**data["safety"])

        return policy


def save_policy(policy: BrainJarPolicy, path: Path) -> None:
    """Save policy to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    policy.updated_at = datetime.utcnow().isoformat()

    with open(path, "w") as f:
        yaml.dump(policy.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_policy(path: Path) -> BrainJarPolicy:
    """Load policy from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return BrainJarPolicy.from_dict(data)


# =============================================================================
# Example Policy Template
# =============================================================================

EXAMPLE_POLICY_YAML = """
# Brain Jar Policy for: {user_id}
# Tier: {tier}
# Created: {created_at}

user_id: "{user_id}"
display_name: "{display_name}"
tier: "{tier}"

# Consent tracking
consent_given: false
consent_timestamp: null
consent_version: "1.0"

# Resource limits
resources:
  memory_mb: 256
  session_minutes: 60
  daily_minutes: 180
  messages_per_hour: 100
  max_context_tokens: 8192

# Data retention
retention:
  conversation_days: 30
  memory_days: 90
  voice_retention: false
  allow_export: true
  allow_delete: true

# Capabilities (what this jar can do)
capabilities:
  # Conversation
  text_chat: true
  voice_chat: true
  avatar_video: true

  # Memory
  episodic_memory: true
  emotional_tracking: true
  preference_learning: true

  # Tools (conservative defaults)
  file_upload: false
  code_execution: false
  web_search: false
  calendar_access: false

  # NEVER for friend jars (hardcoded false)
  hardware_control: false
  bios_access: false
  fpga_control: false
  network_config: false
  wallet_access: false
  founder_data_access: false

# Safety configuration
safety:
  show_ai_disclaimer: true
  show_crisis_resources: true
  crisis_hotline: "988 (US) or your local emergency services"
  no_therapy_claims: true
  hallucination_warning: true
  max_emotional_intensity: 0.7

notes: ""
"""
