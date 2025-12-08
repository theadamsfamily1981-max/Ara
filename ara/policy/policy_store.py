"""
Policy Store - HPV Policy Management
=====================================

Persistent storage and management of learned policies.

Each policy contains:
- A hypervector for pattern matching
- An action to take when matched
- Metadata (source, timestamp, usage stats)

Policies can be:
- Pre-defined (system defaults)
- Learned from LLM (via NEW_POLICY_HDC)
- User-defined
"""

from __future__ import annotations
import json
import time
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class Policy:
    """A single policy with HPV and metadata."""
    name: str                           # Unique identifier
    hpv: np.ndarray                     # Hypervector for matching
    action: str                         # What to do when matched
    threshold: float = 0.5              # Min similarity to trigger

    # Metadata
    description: str = ""
    source: str = "unknown"             # "predefined", "llm", "user"
    category: str = "general"
    created_at: float = field(default_factory=time.time)

    # Usage statistics
    match_count: int = 0
    last_matched: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "hpv_b64": base64.b64encode(self.hpv.tobytes()).decode(),
            "hpv_dim": len(self.hpv),
            "action": self.action,
            "threshold": self.threshold,
            "description": self.description,
            "source": self.source,
            "category": self.category,
            "created_at": self.created_at,
            "match_count": self.match_count,
            "last_matched": self.last_matched,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        """Reconstruct from dictionary."""
        hpv_bytes = base64.b64decode(data["hpv_b64"])
        hpv = np.frombuffer(hpv_bytes, dtype=np.int8).copy()
        return cls(
            name=data["name"],
            hpv=hpv,
            action=data["action"],
            threshold=data.get("threshold", 0.5),
            description=data.get("description", ""),
            source=data.get("source", "unknown"),
            category=data.get("category", "general"),
            created_at=data.get("created_at", time.time()),
            match_count=data.get("match_count", 0),
            last_matched=data.get("last_matched", 0.0),
        )


class PolicyStore:
    """
    Persistent storage for policies.

    Supports:
    - CRUD operations on policies
    - Matching state against all policies
    - Persistence to JSON file
    - Statistics tracking
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path

        # In-memory policy store
        self._policies: Dict[str, Policy] = {}

        # Index by category for faster lookup
        self._by_category: Dict[str, List[str]] = {}

        # Statistics
        self._stats = {
            "total_matches": 0,
            "policies_created": 0,
            "policies_removed": 0,
        }

        # Load from disk if path provided
        if storage_path and storage_path.exists():
            self.load()

    @property
    def count(self) -> int:
        """Number of policies in store."""
        return len(self._policies)

    def add(self, policy: Policy) -> bool:
        """
        Add a policy to the store.

        Returns True if added, False if name already exists.
        """
        if policy.name in self._policies:
            return False

        self._policies[policy.name] = policy
        self._stats["policies_created"] += 1

        # Update category index
        if policy.category not in self._by_category:
            self._by_category[policy.category] = []
        self._by_category[policy.category].append(policy.name)

        return True

    def get(self, name: str) -> Optional[Policy]:
        """Get a policy by name."""
        return self._policies.get(name)

    def remove(self, name: str) -> bool:
        """Remove a policy by name."""
        if name not in self._policies:
            return False

        policy = self._policies.pop(name)
        self._stats["policies_removed"] += 1

        # Update category index
        if policy.category in self._by_category:
            self._by_category[policy.category].remove(name)

        return True

    def update(self, policy: Policy) -> bool:
        """Update an existing policy."""
        if policy.name not in self._policies:
            return False
        self._policies[policy.name] = policy
        return True

    def list_all(self) -> List[str]:
        """List all policy names."""
        return list(self._policies.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List policies in a category."""
        return self._by_category.get(category, [])

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._by_category.keys())

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / len(a)

    def match(self, state_hpv: np.ndarray) -> Tuple[Optional[Policy], float]:
        """
        Find best matching policy for a state.

        Returns (policy, similarity) or (None, 0.0) if no match.
        """
        best_policy = None
        best_sim = 0.0

        for policy in self._policies.values():
            sim = self._similarity(state_hpv, policy.hpv)
            if sim >= policy.threshold and sim > best_sim:
                best_policy = policy
                best_sim = sim

        if best_policy:
            best_policy.match_count += 1
            best_policy.last_matched = time.time()
            self._stats["total_matches"] += 1

        return best_policy, best_sim

    def find_similar(self, state_hpv: np.ndarray,
                     threshold: float = 0.3) -> List[Tuple[Policy, float]]:
        """Find all policies above similarity threshold."""
        results = []
        for policy in self._policies.values():
            sim = self._similarity(state_hpv, policy.hpv)
            if sim >= threshold:
                results.append((policy, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        usage_stats = {}
        for policy in self._policies.values():
            usage_stats[policy.name] = {
                "match_count": policy.match_count,
                "last_matched": policy.last_matched,
            }

        return {
            **self._stats,
            "policy_count": self.count,
            "categories": list(self._by_category.keys()),
            "usage": usage_stats,
        }

    def save(self):
        """Save policies to disk."""
        if not self.storage_path:
            return

        data = {
            "policies": {
                name: policy.to_dict()
                for name, policy in self._policies.items()
            },
            "stats": self._stats,
            "saved_at": time.time(),
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load policies from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        for name, policy_data in data.get("policies", {}).items():
            policy = Policy.from_dict(policy_data)
            self.add(policy)


# ============================================================================
# Pre-defined Policies
# ============================================================================

def create_default_policies(encoder) -> List[Policy]:
    """
    Create default system policies.

    These handle common situations without needing LLM.
    """
    policies = []

    # CPU spike → suggest process investigation
    policies.append(Policy(
        name="handle_cpu_spike",
        hpv=encoder.encode_metrics({"cpu": 0.95, "memory": 0.3}),
        action="log_and_alert",
        threshold=0.6,
        description="High CPU usage detected",
        source="predefined",
        category="system",
    ))

    # Memory pressure → suggest cleanup
    policies.append(Policy(
        name="handle_memory_pressure",
        hpv=encoder.encode_metrics({"memory": 0.95, "swap": 0.5}),
        action="suggest_cleanup",
        threshold=0.6,
        description="Memory pressure detected",
        source="predefined",
        category="system",
    ))

    # User frustration → offer help
    policies.append(Policy(
        name="handle_user_frustration",
        hpv=encoder.encode_text("rapid clicking error retry backspace"),
        action="offer_assistance",
        threshold=0.5,
        description="User appears frustrated",
        source="predefined",
        category="user",
    ))

    # Idle period → reduce monitoring
    policies.append(Policy(
        name="handle_idle",
        hpv=encoder.encode_text("idle inactive no input"),
        action="reduce_polling",
        threshold=0.6,
        description="System is idle",
        source="predefined",
        category="power",
    ))

    return policies


def create_policy_store_with_defaults(encoder,
                                      storage_path: Optional[Path] = None) -> PolicyStore:
    """Create a policy store with default policies loaded."""
    store = PolicyStore(storage_path)

    for policy in create_default_policies(encoder):
        store.add(policy)

    return store
