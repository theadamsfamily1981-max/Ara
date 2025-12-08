"""
Ara Reflex API - High-Level Control of Network Reflexes
=======================================================

High-level API for controlling eBPF/XDP reflex programs.

The reflex layer operates at the NIC level (microseconds) while
this API provides the Python-side interface for:
- Updating reflex policies
- Encoding reflex events as HVs for learning
- Computing teleological priorities

The actual eBPF programs are in ebpf/reflex_xdp.c
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np

from ara.hd.ops import bind, bundle, random_hv_from_string
from ara.hd.vocab import get_vocab
from ara.io.types import HDInputEvent, IOChannel


# =============================================================================
# Reflex Policy
# =============================================================================

class ReflexAction(str, Enum):
    """Actions the reflex layer can take."""
    PASS = "pass"          # Allow packet through
    DROP = "drop"          # Drop packet
    MARK = "mark"          # Mark for monitoring
    RATE_LIMIT = "rate_limit"  # Apply rate limiting
    REDIRECT = "redirect"  # Redirect to different queue


@dataclass
class ReflexRule:
    """A single reflex rule."""
    rule_id: str
    priority: int                  # Higher = checked first
    match: Dict[str, Any]          # Match criteria
    action: ReflexAction
    rate_limit_bps: Optional[int] = None  # For rate_limit action
    enabled: bool = True


@dataclass
class ReflexPolicy:
    """
    A complete reflex policy for the eBPF layer.

    Policies are pushed down to eBPF maps for fast matching.
    """
    policy_id: str
    rules: List[ReflexRule] = field(default_factory=list)
    default_action: ReflexAction = ReflexAction.PASS
    teleology_threshold: int = 200  # Priority threshold for "pain packets"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_rule(self, rule: ReflexRule) -> None:
        """Add a rule to the policy."""
        self.rules.append(rule)
        # Keep sorted by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "priority": r.priority,
                    "match": r.match,
                    "action": r.action.value,
                    "rate_limit_bps": r.rate_limit_bps,
                    "enabled": r.enabled,
                }
                for r in self.rules
            ],
            "default_action": self.default_action.value,
            "teleology_threshold": self.teleology_threshold,
        }


# =============================================================================
# Priority Computation
# =============================================================================

def compute_reflex_priority(
    flow_hv: np.ndarray,
    h_good_template: Optional[np.ndarray] = None,
    h_bad_template: Optional[np.ndarray] = None,
    teleology_bias: float = 0.0,
) -> int:
    """
    Compute teleological priority for a flow.

    Priority is 0-255:
    - 0-50: Background traffic
    - 50-150: Normal traffic
    - 150-200: Important traffic
    - 200-255: Critical/"pain" traffic

    Args:
        flow_hv: Hypervector encoding the flow
        h_good_template: Template for "good" flows
        h_bad_template: Template for "bad" flows
        teleology_bias: Additional bias from teleology state

    Returns:
        Priority value 0-255
    """
    from ara.hd.ops import cosine

    base_priority = 128  # Default middle priority

    # Adjust based on similarity to templates
    if h_good_template is not None:
        sim_good = cosine(flow_hv, h_good_template)
        if sim_good > 0.3:
            base_priority -= int(sim_good * 50)  # Good = lower priority (less urgent)

    if h_bad_template is not None:
        sim_bad = cosine(flow_hv, h_bad_template)
        if sim_bad > 0.3:
            base_priority += int(sim_bad * 100)  # Bad = higher priority (more urgent)

    # Apply teleology bias
    base_priority += int(teleology_bias * 50)

    return max(0, min(255, base_priority))


# =============================================================================
# Reflex Event Encoding
# =============================================================================

def encode_reflex_event(event) -> HDInputEvent:
    """
    Encode a reflex event into an HDInputEvent for HTC learning.

    This allows the soul to learn from network reflexes.

    Args:
        event: ReflexEvent from lan_reflex_bridge

    Returns:
        HDInputEvent for HTC
    """
    vocab = get_vocab()

    # Bin severity
    severity = event.severity
    if severity < 0.3:
        severity_bin = "LOW"
    elif severity < 0.6:
        severity_bin = "MED"
    elif severity < 0.8:
        severity_bin = "HIGH"
    else:
        severity_bin = "CRITICAL"

    # Create event type HV
    h_type = random_hv_from_string(f"REFLEX:{event.event_type.value}")
    h_severity = vocab.bin(severity_bin)

    # Add source node if available
    components = [
        bind(vocab.feature("REFLEX_TYPE"), h_type),
        bind(vocab.feature("SEVERITY"), h_severity),
    ]

    if event.source_node:
        h_node = random_hv_from_string(f"NODE:{event.source_node}")
        components.append(bind(vocab.feature("SOURCE_NODE"), h_node))

    # Bundle and bind with network role
    h_event = bundle(components)
    h_bound = bind(vocab.role("NETWORK"), h_event)

    # Higher severity = higher priority for learning
    priority = 0.5 + event.severity * 0.5

    return HDInputEvent(
        channel=IOChannel.NETWORK,
        role="ROLE_NET_REFLEX",
        meta={
            "event_type": event.event_type.value,
            "severity": event.severity,
            "source_node": event.source_node,
            "priority": event.priority,
        },
        hv=h_bound,
        priority=min(1.0, priority),
    )


# =============================================================================
# Essential Services
# =============================================================================

ESSENTIAL_SERVICES = [
    "dns",
    "dhcp",
    "ntp",
    "ssh",
    "kubernetes",
    "prometheus",
    "logging",
    "backup",
]


def is_essential_service(service_name: str) -> bool:
    """Check if a service is in the essential services whitelist."""
    return service_name.lower() in ESSENTIAL_SERVICES


def create_essential_service_rules() -> List[ReflexRule]:
    """Create reflex rules to protect essential services."""
    rules = []
    for i, service in enumerate(ESSENTIAL_SERVICES):
        rules.append(ReflexRule(
            rule_id=f"essential_{service}",
            priority=1000 - i,  # High priority
            match={"service": service},
            action=ReflexAction.PASS,
            enabled=True,
        ))
    return rules


# =============================================================================
# Reflex Controller
# =============================================================================

class ReflexController:
    """
    High-level controller for the reflex layer.

    Manages policies and provides interface for Sovereign to update reflexes.
    """

    def __init__(self):
        self._current_policy: Optional[ReflexPolicy] = None
        self._h_good: Optional[np.ndarray] = None
        self._h_bad: Optional[np.ndarray] = None

    def set_policy(self, policy: ReflexPolicy) -> None:
        """Set the current reflex policy."""
        self._current_policy = policy
        # In production: push to eBPF maps
        print(f"[Reflex] Policy updated: {policy.policy_id} with {len(policy.rules)} rules")

    def get_policy(self) -> Optional[ReflexPolicy]:
        """Get the current reflex policy."""
        return self._current_policy

    def update_templates(
        self,
        h_good: Optional[np.ndarray] = None,
        h_bad: Optional[np.ndarray] = None,
    ) -> None:
        """Update good/bad flow templates."""
        if h_good is not None:
            self._h_good = h_good
        if h_bad is not None:
            self._h_bad = h_bad

    def compute_priority(self, flow_hv: np.ndarray) -> int:
        """Compute priority for a flow using current templates."""
        return compute_reflex_priority(
            flow_hv,
            self._h_good,
            self._h_bad,
        )

    def create_default_policy(self) -> ReflexPolicy:
        """Create a sensible default policy."""
        policy = ReflexPolicy(policy_id="default")

        # Add essential service rules
        for rule in create_essential_service_rules():
            policy.add_rule(rule)

        return policy


# =============================================================================
# Factory
# =============================================================================

_reflex_controller: Optional[ReflexController] = None


def get_reflex_controller() -> ReflexController:
    """Get the global reflex controller."""
    global _reflex_controller
    if _reflex_controller is None:
        _reflex_controller = ReflexController()
    return _reflex_controller


__all__ = [
    'ReflexAction',
    'ReflexRule',
    'ReflexPolicy',
    'compute_reflex_priority',
    'encode_reflex_event',
    'is_essential_service',
    'ESSENTIAL_SERVICES',
    'create_essential_service_rules',
    'ReflexController',
    'get_reflex_controller',
]
