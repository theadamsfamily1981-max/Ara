"""
Ara LAN Reflex Search - Flow Lookup and Rule Matching
======================================================

Python interface for the LAN nervous system search components:
- Hash CAM for flow signature lookup
- LUT-TCAM for reflex rule matching

This is the "have I seen this before?" fast path for network events.

Architecture:
    Packet → Header Extract → Flow Hash → Hash CAM → Candidate IDs
                                       ↓
                           LUT-TCAM → Action (drop/throttle/pass)
                                       ↓
                            Optional: XNOR-CAM for HDC similarity

Mythic Spec:
    The spinal cord reflex - before conscious thought,
    the body reacts. Pain packets trigger automatic responses;
    familiar flows are recognized instantly.

Physical Spec:
    - Hash CAM: ~ns latency, 4 tables × 4k entries
    - LUT-TCAM: single-cycle, 256 rules
    - Reflex action latency: < 10 ns typical

Usage:
    from ara.core.lan.reflex_search import ReflexSearch, FlowSignature

    reflex = ReflexSearch()

    # Add a reflex rule
    reflex.add_rule(
        key=FlowSignature(dst_port=22),
        mask=FlowSignature(dst_port=0xFFFF),
        action=ReflexAction.PRIORITY_BOOST
    )

    # Lookup a flow
    result = reflex.lookup_flow(sig)
    if result.matched:
        print(f"Action: {result.action}")

Software Mode:
    Falls back to dict-based lookup when no FPGA.
"""

from __future__ import annotations

import numpy as np
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, Dict, List, Tuple, Any
import time


# =============================================================================
# Constants
# =============================================================================

# Hash CAM parameters
HASH_TABLES = 4
HASH_TABLE_SIZE = 4096
HASH_ID_BITS = 16
SIG_BITS = 64

# TCAM parameters
TCAM_ENTRIES = 256
TCAM_KEY_WIDTH = 56
TCAM_ACTION_BITS = 8

# Hash seeds (must match RTL)
HASH_SEEDS = [
    0x9E3779B97F4A7C15,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA6B27D4A7F3,
]


# =============================================================================
# Reflex Actions
# =============================================================================

class ReflexAction(IntEnum):
    """Actions for reflex rules."""
    PASS = 0x00           # Default: allow through
    DROP = 0x01           # Drop packet
    THROTTLE = 0x02       # Rate limit
    PRIORITY_BOOST = 0x03 # Increase priority
    GLITCH_TRIGGER = 0x04 # Trigger visual glitch
    LOG_ONLY = 0x05       # Log but don't act
    REDIRECT = 0x06       # Redirect to different queue
    MARK_SUSPICIOUS = 0x07 # Flag for deeper analysis


# =============================================================================
# Flow Signature
# =============================================================================

@dataclass
class FlowSignature:
    """
    Compact flow signature for hash CAM lookup.

    Can be created from raw fields or computed from full 5-tuple.
    """
    src_ip: int = 0
    dst_ip: int = 0
    src_port: int = 0
    dst_port: int = 0
    proto: int = 0

    def to_hash(self) -> int:
        """Compute 64-bit signature hash."""
        # Match RTL hash computation
        mixed = 0
        mixed ^= (self.src_ip << 32) | self.dst_ip
        mixed ^= (
            ((self.dst_ip & 0xFFFF) << 48) |
            (self.src_port << 32) |
            (self.dst_port << 16) |
            (self.proto << 8)
        )
        mixed ^= (self.src_port << 32) | self.dst_port

        # Additional mixing
        mixed ^= (mixed >> 17)
        mixed ^= ((mixed << 31) & ((1 << 64) - 1))

        return mixed & ((1 << 64) - 1)

    def to_tcam_key(self) -> int:
        """Compute 56-bit TCAM key (hash + proto + port)."""
        flow_hash = self.to_hash() & 0xFFFFFFFF  # 32-bit hash
        return (flow_hash << 24) | (self.proto << 16) | self.dst_port

    @classmethod
    def from_packet(cls, packet: bytes) -> 'FlowSignature':
        """Extract signature from raw packet (simplified)."""
        # This is a placeholder - real implementation would parse headers
        if len(packet) < 20:
            return cls()

        # Assume IPv4 + TCP/UDP
        # IP header starts at byte 14 (after Ethernet)
        # This is a simplified extraction
        return cls()


# =============================================================================
# Reflex Rule
# =============================================================================

@dataclass
class ReflexRule:
    """A TCAM rule for reflex matching."""
    key: FlowSignature
    mask: FlowSignature  # 1 = care, 0 = don't care
    action: ReflexAction
    priority: int = 0
    rule_id: int = 0
    hit_count: int = 0
    created: datetime = field(default_factory=datetime.utcnow)

    def matches(self, sig: FlowSignature) -> bool:
        """Check if signature matches this rule (software path)."""
        key_val = self.key.to_tcam_key()
        mask_val = self.mask.to_tcam_key()
        sig_val = sig.to_tcam_key()

        return (sig_val & mask_val) == (key_val & mask_val)


# =============================================================================
# Lookup Results
# =============================================================================

@dataclass
class FlowLookupResult:
    """Result of hash CAM flow lookup."""
    found: bool
    candidate_ids: List[int]
    latency_ns: float
    tables_hit: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "candidate_ids": self.candidate_ids,
            "latency_ns": self.latency_ns,
            "tables_hit": self.tables_hit,
        }


@dataclass
class ReflexLookupResult:
    """Result of TCAM reflex lookup."""
    matched: bool
    action: ReflexAction
    rule_id: int
    latency_ns: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "action": self.action.name if self.matched else None,
            "rule_id": self.rule_id,
            "latency_ns": self.latency_ns,
        }


# =============================================================================
# Software Hash Table (for simulation)
# =============================================================================

class SoftwareHashTable:
    """Software simulation of hash CAM."""

    def __init__(self, table_size: int = HASH_TABLE_SIZE):
        self.table_size = table_size
        self.tables: List[Dict[int, int]] = [{} for _ in range(HASH_TABLES)]
        self.entries = 0
        self.collisions = 0

    def _compute_hash(self, sig: int, seed_idx: int) -> int:
        """Compute hash address."""
        seed = HASH_SEEDS[seed_idx]
        mixed = sig ^ seed

        # Fold to table size
        addr = 0
        for i in range(0, 64, 12):  # log2(4096) = 12
            addr ^= (mixed >> i) & (self.table_size - 1)

        return addr

    def insert(self, sig: int, entry_id: int) -> None:
        """Insert entry into all tables."""
        self.entries += 1

        for t in range(HASH_TABLES):
            addr = self._compute_hash(sig, t)
            if addr in self.tables[t]:
                self.collisions += 1
            self.tables[t][addr] = entry_id

    def lookup(self, sig: int) -> List[Tuple[int, bool]]:
        """Lookup signature, return (id, found) for each table."""
        results = []
        for t in range(HASH_TABLES):
            addr = self._compute_hash(sig, t)
            if addr in self.tables[t]:
                results.append((self.tables[t][addr], True))
            else:
                results.append((0, False))
        return results


# =============================================================================
# Software TCAM (for simulation)
# =============================================================================

class SoftwareTCAM:
    """Software simulation of LUT-TCAM."""

    def __init__(self, max_entries: int = TCAM_ENTRIES):
        self.max_entries = max_entries
        self.rules: List[Optional[ReflexRule]] = [None] * max_entries
        self.entry_count = 0

    def program_rule(self, index: int, rule: ReflexRule) -> bool:
        """Program a rule at index."""
        if index >= self.max_entries:
            return False

        if self.rules[index] is None:
            self.entry_count += 1

        rule.rule_id = index
        self.rules[index] = rule
        return True

    def lookup(self, sig: FlowSignature) -> Tuple[bool, Optional[ReflexRule]]:
        """Find first matching rule."""
        # Priority-based: check higher priority first
        sorted_rules = [
            (i, r) for i, r in enumerate(self.rules)
            if r is not None
        ]
        sorted_rules.sort(key=lambda x: x[1].priority, reverse=True)

        for idx, rule in sorted_rules:
            if rule.matches(sig):
                rule.hit_count += 1
                return True, rule

        return False, None


# =============================================================================
# Reflex Search Interface
# =============================================================================

class ReflexSearch:
    """
    LAN reflex search interface.

    Combines hash CAM for flow lookup and LUT-TCAM for rule matching.
    Falls back to software when no FPGA available.
    """

    def __init__(self, fpga_device=None):
        """
        Initialize reflex search.

        Args:
            fpga_device: FPGA handle (None = software mode)
        """
        self.is_hardware = fpga_device is not None
        self.fpga = fpga_device

        # Software fallbacks
        self.hash_table = SoftwareHashTable()
        self.tcam = SoftwareTCAM()

        # Flow database (for software mode)
        self._flows: Dict[int, Any] = {}
        self._next_flow_id = 1

        # Statistics
        self._lookups = 0
        self._rule_matches = 0
        self._flow_hits = 0

    # =========================================================================
    # Rule Management
    # =========================================================================

    def add_rule(
        self,
        key: FlowSignature,
        mask: FlowSignature,
        action: ReflexAction,
        priority: int = 0,
        index: Optional[int] = None,
    ) -> int:
        """
        Add a reflex rule to the TCAM.

        Args:
            key: Match key
            mask: Care mask (1 = care, 0 = wildcard)
            action: Action to take on match
            priority: Rule priority (higher = checked first)
            index: Optional fixed index (auto-assigned if None)

        Returns:
            Rule index
        """
        # Find free index if not specified
        if index is None:
            for i in range(self.tcam.max_entries):
                if self.tcam.rules[i] is None:
                    index = i
                    break
            if index is None:
                raise ValueError("TCAM full")

        rule = ReflexRule(
            key=key,
            mask=mask,
            action=action,
            priority=priority,
        )

        if self.is_hardware:
            # Program FPGA TCAM
            # self.fpga.program_tcam_rule(index, key, mask, action)
            pass

        self.tcam.program_rule(index, rule)
        return index

    def remove_rule(self, index: int) -> bool:
        """Remove a rule by index."""
        if index >= self.tcam.max_entries:
            return False

        self.tcam.rules[index] = None
        return True

    # =========================================================================
    # Flow Management
    # =========================================================================

    def register_flow(
        self,
        sig: FlowSignature,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Register a flow in the hash CAM.

        Args:
            sig: Flow signature
            metadata: Optional metadata to associate

        Returns:
            Flow ID
        """
        flow_id = self._next_flow_id
        self._next_flow_id += 1

        sig_hash = sig.to_hash()

        if self.is_hardware:
            # Program FPGA hash CAM
            # self.fpga.program_hash_entry(sig_hash, flow_id)
            pass

        self.hash_table.insert(sig_hash, flow_id)
        self._flows[flow_id] = {
            "sig": sig,
            "metadata": metadata or {},
            "created": datetime.utcnow(),
        }

        return flow_id

    # =========================================================================
    # Lookup
    # =========================================================================

    def lookup_flow(self, sig: FlowSignature) -> FlowLookupResult:
        """
        Lookup flow in hash CAM.

        Args:
            sig: Flow signature to lookup

        Returns:
            FlowLookupResult with candidate IDs
        """
        start = time.perf_counter()
        self._lookups += 1

        sig_hash = sig.to_hash()

        if self.is_hardware:
            # Query FPGA
            # candidates = self.fpga.query_hash_cam(sig_hash)
            candidates = []
        else:
            results = self.hash_table.lookup(sig_hash)
            candidates = [r[0] for r in results if r[1]]

        end = time.perf_counter()
        latency_ns = (end - start) * 1e9

        if candidates:
            self._flow_hits += 1

        return FlowLookupResult(
            found=len(candidates) > 0,
            candidate_ids=candidates,
            latency_ns=latency_ns,
            tables_hit=len(candidates),
        )

    def lookup_reflex(self, sig: FlowSignature) -> ReflexLookupResult:
        """
        Lookup reflex rule in TCAM.

        Args:
            sig: Flow signature to match

        Returns:
            ReflexLookupResult with action
        """
        start = time.perf_counter()
        self._lookups += 1

        if self.is_hardware:
            # Query FPGA TCAM
            # matched, action, rule_id = self.fpga.query_tcam(sig)
            matched, action, rule_id = False, ReflexAction.PASS, 0
        else:
            matched, rule = self.tcam.lookup(sig)
            if matched and rule:
                action = rule.action
                rule_id = rule.rule_id
                self._rule_matches += 1
            else:
                action = ReflexAction.PASS
                rule_id = 0

        end = time.perf_counter()
        latency_ns = (end - start) * 1e9

        return ReflexLookupResult(
            matched=matched,
            action=action,
            rule_id=rule_id,
            latency_ns=latency_ns,
        )

    def lookup_combined(
        self,
        sig: FlowSignature,
    ) -> Tuple[FlowLookupResult, ReflexLookupResult]:
        """
        Combined flow + reflex lookup.

        This is the typical path: check both hash CAM and TCAM.
        """
        flow_result = self.lookup_flow(sig)
        reflex_result = self.lookup_reflex(sig)
        return flow_result, reflex_result

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "lookups": self._lookups,
            "flow_hits": self._flow_hits,
            "rule_matches": self._rule_matches,
            "flows_registered": len(self._flows),
            "rules_active": self.tcam.entry_count,
            "hash_collisions": self.hash_table.collisions,
            "is_hardware": self.is_hardware,
        }


# =============================================================================
# Singleton
# =============================================================================

_reflex_search: Optional[ReflexSearch] = None


def get_reflex_search() -> ReflexSearch:
    """Get the global reflex search instance."""
    global _reflex_search
    if _reflex_search is None:
        _reflex_search = ReflexSearch()
    return _reflex_search


# =============================================================================
# Convenience Functions
# =============================================================================

def add_pain_rule(
    pattern: str,
    action: ReflexAction = ReflexAction.MARK_SUSPICIOUS,
) -> int:
    """
    Add a "pain" detection rule.

    Args:
        pattern: Pattern description (parsed to signature)
        action: Action on match

    Returns:
        Rule index
    """
    reflex = get_reflex_search()

    # Parse pattern (simplified)
    # Format: "port:22" or "proto:6" etc.
    sig = FlowSignature()
    mask = FlowSignature()

    if pattern.startswith("port:"):
        port = int(pattern.split(":")[1])
        sig.dst_port = port
        mask.dst_port = 0xFFFF

    elif pattern.startswith("proto:"):
        proto = int(pattern.split(":")[1])
        sig.proto = proto
        mask.proto = 0xFF

    return reflex.add_rule(sig, mask, action)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ReflexAction',
    'FlowSignature',
    'ReflexRule',
    'FlowLookupResult',
    'ReflexLookupResult',
    'ReflexSearch',
    'get_reflex_search',
    'add_pain_rule',
]
