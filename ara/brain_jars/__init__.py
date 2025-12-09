"""
Ara Brain Jars: Multi-Tenant Friend Access System

Each "brain jar" is an isolated tenant instance containing:
- Per-friend EternalMemory shard
- Per-friend AxisMundi instance
- Per-friend conversation history
- Per-friend preferences and state

Key principles:
1. ISOLATION: Friends never see each other's data or Founder's private state
2. CONSENT: Explicit disclosure of what's stored and why
3. SAFETY: Clear disclaimers, crisis resources, no therapy promises
4. BOUNDARIES: Friend jars have NO authority over hardware/infrastructure

Usage:
    from ara.brain_jars import BrainJarManager, BrainJarPolicy

    manager = BrainJarManager()
    jar = manager.create_jar("friend_001", policy=BrainJarPolicy.TRUSTED_FRIEND)
"""

from .policy import BrainJarPolicy, PolicyTier, load_policy, save_policy
from .isolation import BrainJarStore, get_jar_store
from .api_boundary import FriendJarAPI, JarCapability
from .disclosures import (
    DisclosureManager,
    get_onboarding_disclosure,
    get_session_disclaimer,
    get_consent_form,
    get_data_transparency_notice,
    get_hallucination_warning,
)

__all__ = [
    # Policy
    "BrainJarPolicy",
    "PolicyTier",
    "load_policy",
    "save_policy",
    # Isolation
    "BrainJarStore",
    "get_jar_store",
    # API
    "FriendJarAPI",
    "JarCapability",
    # Disclosures
    "DisclosureManager",
    "get_onboarding_disclosure",
    "get_session_disclaimer",
    "get_consent_form",
    "get_data_transparency_notice",
    "get_hallucination_warning",
]
