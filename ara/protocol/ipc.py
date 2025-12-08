"""
NeuroSymbiosis IPC Protocol
============================

Communication protocol between neuromorphic card (subcortex)
and GPU LLM (cortex).

Messages:
---------

STATE_HPV_QUERY (Card → LLM):
    - state_hpv: Binary hypervector (compressed state)
    - summary: Text summary from HDP probe
    - context: Recent event window
    - urgency: 0-1 score

NEW_POLICY_HDC (LLM → Card):
    - policy_name: Identifier for the new policy
    - policy_hpv: Hypervector for pattern matching
    - action: What to do when matched
    - confidence_threshold: Min similarity to trigger
    - weight_deltas: Optional SNN weight updates

The protocol is designed to minimize bandwidth:
- HPVs are binary, can be compressed
- Most communication is text summaries
- Policies are learned, reducing future calls
"""

from __future__ import annotations
import json
import base64
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class StateHPVQuery:
    """
    STATE_HPV_QUERY: Request from Card to LLM.

    Sent when the subcortex detects an anomaly it can't handle locally.
    """
    # Core data
    state_hpv: np.ndarray              # The current state hypervector
    summary: str                        # Text summary from HDP probe
    urgency: float                      # 0-1, how urgent

    # Context
    top_concepts: List[tuple]           # [(concept, similarity), ...]
    recent_events: List[str]            # Recent event descriptions
    decision_reason: str                # Why we're escalating

    # Metadata
    timestamp: float = field(default_factory=time.time)
    query_id: str = ""                  # Unique query identifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "STATE_HPV_QUERY",
            "state_hpv_b64": base64.b64encode(self.state_hpv.tobytes()).decode(),
            "state_hpv_dim": len(self.state_hpv),
            "summary": self.summary,
            "urgency": self.urgency,
            "top_concepts": self.top_concepts,
            "recent_events": self.recent_events,
            "decision_reason": self.decision_reason,
            "timestamp": self.timestamp,
            "query_id": self.query_id,
        }

    def to_llm_prompt(self) -> str:
        """
        Convert to a prompt for the LLM.

        This is what the LLM actually sees.
        """
        lines = [
            "# Subcortex Escalation",
            "",
            f"**Urgency**: {self.urgency:.2f}",
            f"**Reason**: {self.decision_reason}",
            "",
            "## State Summary",
            self.summary,
            "",
        ]

        if self.top_concepts:
            lines.append("## Matched Patterns")
            for concept, sim in self.top_concepts[:5]:
                lines.append(f"- {concept}: {sim:.2f}")
            lines.append("")

        if self.recent_events:
            lines.append("## Recent Events")
            for event in self.recent_events[-10:]:
                lines.append(f"- {event}")
            lines.append("")

        lines.extend([
            "## Requested Response",
            "Please analyze this state and provide:",
            "1. Assessment of the situation",
            "2. Recommended action",
            "3. If appropriate, a new policy to handle similar situations",
            "",
            "Format policy as:",
            "```json",
            '{"policy_name": "...", "pattern_description": "...", "action": "...", "threshold": 0.X}',
            "```",
        ])

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateHPVQuery":
        """Reconstruct from dictionary."""
        hpv_bytes = base64.b64decode(data["state_hpv_b64"])
        state_hpv = np.frombuffer(hpv_bytes, dtype=np.int8)
        return cls(
            state_hpv=state_hpv,
            summary=data["summary"],
            urgency=data["urgency"],
            top_concepts=data.get("top_concepts", []),
            recent_events=data.get("recent_events", []),
            decision_reason=data.get("decision_reason", ""),
            timestamp=data.get("timestamp", time.time()),
            query_id=data.get("query_id", ""),
        )


@dataclass
class NewPolicyHDC:
    """
    NEW_POLICY_HDC: Response from LLM to Card.

    Contains a new policy to be learned by the subcortex.
    """
    # Policy definition
    policy_name: str                    # Unique identifier
    policy_hpv: np.ndarray              # Hypervector for matching
    action: str                         # What to do when matched
    confidence_threshold: float         # Min similarity to trigger

    # Metadata
    description: str = ""               # Human-readable description
    source_query_id: str = ""           # Query that triggered this
    timestamp: float = field(default_factory=time.time)

    # Optional: SNN weight deltas
    weight_deltas: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": "NEW_POLICY_HDC",
            "policy_name": self.policy_name,
            "policy_hpv_b64": base64.b64encode(self.policy_hpv.tobytes()).decode(),
            "policy_hpv_dim": len(self.policy_hpv),
            "action": self.action,
            "confidence_threshold": self.confidence_threshold,
            "description": self.description,
            "source_query_id": self.source_query_id,
            "timestamp": self.timestamp,
        }
        if self.weight_deltas is not None:
            result["weight_deltas_b64"] = base64.b64encode(
                self.weight_deltas.tobytes()
            ).decode()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NewPolicyHDC":
        """Reconstruct from dictionary."""
        hpv_bytes = base64.b64decode(data["policy_hpv_b64"])
        policy_hpv = np.frombuffer(hpv_bytes, dtype=np.int8)

        weight_deltas = None
        if "weight_deltas_b64" in data:
            weight_bytes = base64.b64decode(data["weight_deltas_b64"])
            weight_deltas = np.frombuffer(weight_bytes, dtype=np.float32)

        return cls(
            policy_name=data["policy_name"],
            policy_hpv=policy_hpv,
            action=data["action"],
            confidence_threshold=data["confidence_threshold"],
            description=data.get("description", ""),
            source_query_id=data.get("source_query_id", ""),
            timestamp=data.get("timestamp", time.time()),
            weight_deltas=weight_deltas,
        )

    @classmethod
    def from_llm_response(cls, response_text: str, encoder,
                          state_hpv: np.ndarray) -> Optional["NewPolicyHDC"]:
        """
        Parse LLM response to extract policy.

        The LLM should include a JSON block with policy definition.
        """
        import re

        # Find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if not json_match:
            return None

        try:
            policy_data = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None

        # Extract fields
        policy_name = policy_data.get("policy_name", f"policy_{int(time.time())}")
        action = policy_data.get("action", "log")
        threshold = float(policy_data.get("threshold", 0.5))
        description = policy_data.get("pattern_description", "")

        # Create policy HPV by binding state with action encoding
        action_hv = encoder.encode_text(action)
        policy_hpv = encoder.bind(state_hpv, action_hv)

        return cls(
            policy_name=policy_name,
            policy_hpv=policy_hpv,
            action=action,
            confidence_threshold=threshold,
            description=description,
        )


class NeuroSymbiosisProtocol:
    """
    Protocol handler for Card ↔ LLM communication.

    Manages the query/response flow and policy learning.
    """

    def __init__(self, encoder, subcortex):
        self.encoder = encoder
        self.subcortex = subcortex

        # Pending queries
        self._pending: Dict[str, StateHPVQuery] = {}

        # Learned policies
        self._policies: List[NewPolicyHDC] = []

        # Statistics
        self._stats = {
            "queries_sent": 0,
            "policies_received": 0,
            "policies_applied": 0,
        }

    def create_query(self, decision, state_hpv: np.ndarray,
                     recent_events: Optional[List[str]] = None) -> StateHPVQuery:
        """
        Create a STATE_HPV_QUERY from a decision.

        Called when subcortex decides to escalate.
        """
        query_id = f"q_{int(time.time() * 1000)}"

        # Get probe result for top concepts
        probe_result = self.subcortex.probe.probe(state_hpv)

        query = StateHPVQuery(
            state_hpv=state_hpv,
            summary=self.subcortex.get_state_summary(),
            urgency=decision.severity,
            top_concepts=probe_result.top_concepts,
            recent_events=recent_events or [],
            decision_reason=decision.reason,
            query_id=query_id,
        )

        self._pending[query_id] = query
        self._stats["queries_sent"] += 1

        return query

    def process_response(self, response_text: str, query_id: str) -> Optional[NewPolicyHDC]:
        """
        Process LLM response and extract policy.

        Applies the policy to the subcortex if valid.
        """
        if query_id not in self._pending:
            return None

        query = self._pending.pop(query_id)

        # Try to parse policy from response
        policy = NewPolicyHDC.from_llm_response(
            response_text,
            self.encoder,
            query.state_hpv
        )

        if policy:
            policy.source_query_id = query_id
            self._policies.append(policy)
            self._stats["policies_received"] += 1

            # Apply to subcortex
            self.subcortex.learn_policy(
                policy.policy_name,
                policy.policy_hpv,
                {
                    "action": policy.action,
                    "threshold": policy.confidence_threshold,
                    "description": policy.description,
                    "source": "llm",
                }
            )
            self._stats["policies_applied"] += 1

        return policy

    def get_pending_queries(self) -> List[StateHPVQuery]:
        """Get list of pending queries."""
        return list(self._pending.values())

    def get_learned_policies(self) -> List[NewPolicyHDC]:
        """Get list of learned policies."""
        return self._policies.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            **self._stats,
            "pending_queries": len(self._pending),
        }
