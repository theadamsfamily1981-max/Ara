"""
Card Runtime - Neuromorphic Subcortex
=====================================

The card-side runtime that:
1. Ingests telemetry from various sources
2. Encodes into rolling state hypervector
3. Runs CorrSpike-HDC for local decisions
4. Applies policies or escalates to GPU cortex

This is the "always-on" nervous system that filters 80-90% of events
locally, only escalating truly novel or high-risk situations.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple
from collections import deque
from enum import Enum
import numpy as np

from ara.hdc.encoder import HDEncoder, HDEncoderConfig
from ara.hdc.state_stream import StateStream, StateStreamConfig
from ara.bridge.messages import StateHPVQuery, NewPolicy

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CardConfig:
    """Configuration for the card runtime."""
    hv_dim: int = 8192
    decay: float = 0.95
    anomaly_threshold: float = 0.6
    escalation_threshold: float = 0.7
    max_events_per_sec: int = 1000
    policy_cache_size: int = 64
    telemetry_buffer_size: int = 1000


class Decision(Enum):
    """Decision made by the card."""
    IGNORE = "ignore"
    LOG = "log"
    LOCAL_POLICY = "local_policy"
    ESCALATE = "escalate"
    EMERGENCY = "emergency"


@dataclass
class CardEvent:
    """An event processed by the card."""
    timestamp: float
    source: str
    event_type: str
    data: Dict[str, Any]
    hv: Optional[np.ndarray] = None


@dataclass
class CardStats:
    """Statistics for the card runtime."""
    events_processed: int = 0
    events_ignored: int = 0
    events_logged: int = 0
    local_policies_applied: int = 0
    escalations: int = 0
    emergencies: int = 0
    total_latency_us: float = 0.0

    @property
    def filter_rate(self) -> float:
        """Fraction of events handled locally (not escalated)."""
        if self.events_processed == 0:
            return 1.0
        escalated = self.escalations + self.emergencies
        return 1.0 - (escalated / self.events_processed)

    @property
    def avg_latency_us(self) -> float:
        """Average processing latency in microseconds."""
        if self.events_processed == 0:
            return 0.0
        return self.total_latency_us / self.events_processed


# ============================================================================
# Policy Store
# ============================================================================

@dataclass
class InstalledPolicy:
    """A policy installed on the card."""
    policy_id: str
    policy_hv: np.ndarray
    action_type: str
    conditions: Dict[str, Any]
    expires_at: float
    snn_deltas: Dict[str, float]
    applied_count: int = 0


class PolicyStore:
    """Manages installed policies on the card."""

    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._policies: Dict[str, InstalledPolicy] = {}
        self._by_expiry: List[Tuple[float, str]] = []  # (expires_at, policy_id)

    def install(self, policy: NewPolicy) -> None:
        """Install a new policy."""
        # Remove oldest if at capacity
        if len(self._policies) >= self.max_size:
            self._evict_oldest()

        installed = InstalledPolicy(
            policy_id=policy.policy_id,
            policy_hv=policy.get_policy_hv(),
            action_type=policy.action_type,
            conditions=policy.conditions,
            expires_at=policy.expires_at_ms / 1000.0 if policy.expires_at_ms > 0 else float('inf'),
            snn_deltas=policy.get_snn_deltas()
        )

        self._policies[policy.policy_id] = installed
        self._by_expiry.append((installed.expires_at, policy.policy_id))
        self._by_expiry.sort()

        logger.info(f"Installed policy: {policy.policy_id}")

    def find_matching(self, state_hv: np.ndarray, threshold: float = 0.6) -> Optional[InstalledPolicy]:
        """Find a policy that matches the current state."""
        self._expire_old()

        best_match = None
        best_sim = threshold

        for policy in self._policies.values():
            # Compute cosine similarity
            sim = np.dot(state_hv, policy.policy_hv) / len(state_hv)

            if sim > best_sim:
                best_sim = sim
                best_match = policy

        if best_match:
            best_match.applied_count += 1

        return best_match

    def _expire_old(self) -> None:
        """Remove expired policies."""
        now = time.time()
        expired = [pid for pid, p in self._policies.items() if p.expires_at < now]
        for pid in expired:
            del self._policies[pid]
            logger.debug(f"Expired policy: {pid}")

    def _evict_oldest(self) -> None:
        """Evict the oldest policy."""
        if self._by_expiry:
            _, pid = self._by_expiry.pop(0)
            if pid in self._policies:
                del self._policies[pid]

    def get_recent_ids(self, n: int = 5) -> List[str]:
        """Get IDs of recently applied policies."""
        sorted_policies = sorted(
            self._policies.values(),
            key=lambda p: p.applied_count,
            reverse=True
        )
        return [p.policy_id for p in sorted_policies[:n]]


# ============================================================================
# Card Runtime
# ============================================================================

class CardRuntime:
    """
    Main runtime for the neuromorphic card.

    Processes telemetry, maintains state HPV, makes local decisions,
    and escalates to cortex when needed.
    """

    def __init__(self, config: Optional[CardConfig] = None):
        self.config = config or CardConfig()

        # HDC components
        self.encoder = HDEncoder(HDEncoderConfig(dim=self.config.hv_dim))
        self.state_stream = StateStream(StateStreamConfig(dim=self.config.hv_dim, decay_rate=self.config.decay))

        # Policy store
        self.policy_store = PolicyStore(max_size=self.config.policy_cache_size)

        # Statistics
        self.stats = CardStats()

        # Event buffer
        self._event_buffer: deque = deque(maxlen=self.config.telemetry_buffer_size)

        # Callbacks
        self._escalation_callback: Optional[Callable[[StateHPVQuery], None]] = None
        self._decision_callback: Optional[Callable[[CardEvent, Decision], None]] = None

        # Running state
        self._running = False
        self._lock = threading.Lock()

    def set_escalation_callback(self, callback: Callable[[StateHPVQuery], None]) -> None:
        """Set callback for when escalation is needed."""
        self._escalation_callback = callback

    def set_decision_callback(self, callback: Callable[[CardEvent, Decision], None]) -> None:
        """Set callback for all decisions."""
        self._decision_callback = callback

    def process_event(self, event: CardEvent) -> Decision:
        """
        Process a single event through the card pipeline.

        Returns the decision made (ignore, local_policy, escalate, etc.)
        """
        start_time = time.time()

        with self._lock:
            self.stats.events_processed += 1

            # 1. Encode event to HPV
            event.hv = self.encoder.encode_event(
                event.event_type,
                event.data,
                time.localtime().tm_hour
            )

            # 2. Update rolling state
            self.state_stream.add_event(event.hv)
            state_hv = self.state_stream.get_state()

            # 3. Check for matching policy
            matching_policy = self.policy_store.find_matching(
                state_hv,
                threshold=self.config.anomaly_threshold
            )

            # 4. Compute anomaly score (how novel is this state?)
            anomaly_score = self._compute_anomaly_score(state_hv)

            # 5. Make decision
            decision = self._decide(event, state_hv, anomaly_score, matching_policy)

            # 6. Execute decision
            self._execute_decision(event, decision, state_hv, anomaly_score, matching_policy)

            # Update stats
            latency_us = (time.time() - start_time) * 1e6
            self.stats.total_latency_us += latency_us

            # Callback
            if self._decision_callback:
                self._decision_callback(event, decision)

            return decision

    def _compute_anomaly_score(self, state_hv: np.ndarray) -> float:
        """
        Compute how anomalous the current state is.

        Uses the policy store as a reference - if no policies match well,
        the state is considered more anomalous.
        """
        # Check similarity against all installed policies
        max_sim = 0.0
        for policy in self.policy_store._policies.values():
            sim = np.dot(state_hv, policy.policy_hv) / len(state_hv)
            max_sim = max(max_sim, sim)

        # Higher anomaly score = less similar to known policies
        return 1.0 - max_sim

    def _decide(
        self,
        event: CardEvent,
        state_hv: np.ndarray,
        anomaly_score: float,
        matching_policy: Optional[InstalledPolicy]
    ) -> Decision:
        """Make a decision about how to handle the event."""

        # Emergency: very high anomaly + specific event types
        if anomaly_score > 0.9 and event.event_type in ('error', 'crash', 'security'):
            return Decision.EMERGENCY

        # Escalate: high anomaly, no matching policy
        if anomaly_score > self.config.escalation_threshold and matching_policy is None:
            return Decision.ESCALATE

        # Local policy: we have a matching policy
        if matching_policy is not None:
            return Decision.LOCAL_POLICY

        # Log: moderate anomaly
        if anomaly_score > self.config.anomaly_threshold:
            return Decision.LOG

        # Ignore: normal operation
        return Decision.IGNORE

    def _execute_decision(
        self,
        event: CardEvent,
        decision: Decision,
        state_hv: np.ndarray,
        anomaly_score: float,
        matching_policy: Optional[InstalledPolicy]
    ) -> None:
        """Execute the decision."""

        if decision == Decision.IGNORE:
            self.stats.events_ignored += 1

        elif decision == Decision.LOG:
            self.stats.events_logged += 1
            self._event_buffer.append(event)

        elif decision == Decision.LOCAL_POLICY:
            self.stats.local_policies_applied += 1
            logger.debug(f"Applied local policy: {matching_policy.policy_id}")

        elif decision == Decision.ESCALATE:
            self.stats.escalations += 1
            self._escalate(event, state_hv, anomaly_score)

        elif decision == Decision.EMERGENCY:
            self.stats.emergencies += 1
            self._escalate(event, state_hv, anomaly_score, urgent=True)

    def _escalate(
        self,
        event: CardEvent,
        state_hv: np.ndarray,
        anomaly_score: float,
        urgent: bool = False
    ) -> None:
        """Escalate to the cortex (GPU LLM)."""
        if self._escalation_callback is None:
            logger.warning("No escalation callback set, dropping escalation")
            return

        # Build query
        query = StateHPVQuery.create(
            hv=state_hv,
            features={
                'event_type': event.event_type,
                'source': event.source,
                **event.data
            },
            anomaly_score=anomaly_score,
            urgency=1.0 if urgent else 0.5 + anomaly_score * 0.5,
            recent_policies=self.policy_store.get_recent_ids(),
            context=f"Event: {event.event_type} from {event.source}"
        )

        logger.info(f"Escalating to cortex: {query.trace_id} (anomaly={anomaly_score:.2f})")
        self._escalation_callback(query)

    def install_policy(self, policy: NewPolicy) -> None:
        """Install a policy received from the cortex."""
        with self._lock:
            self.policy_store.install(policy)

    def ingest_telemetry(
        self,
        source: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> Decision:
        """Convenience method to ingest telemetry."""
        event = CardEvent(
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            data=data
        )
        return self.process_event(event)

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            'events_processed': self.stats.events_processed,
            'events_ignored': self.stats.events_ignored,
            'local_policies_applied': self.stats.local_policies_applied,
            'escalations': self.stats.escalations,
            'emergencies': self.stats.emergencies,
            'filter_rate': self.stats.filter_rate,
            'avg_latency_us': self.stats.avg_latency_us,
            'policies_installed': len(self.policy_store._policies),
            'state_magnitude': self.state_stream.get_state_magnitude()
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the card runtime."""
    print("=" * 60)
    print("Card Runtime Demo")
    print("=" * 60)

    # Create runtime
    config = CardConfig(hv_dim=1024)
    runtime = CardRuntime(config)

    # Track escalations
    escalations = []
    def on_escalate(query: StateHPVQuery):
        escalations.append(query)
        print(f"  ESCALATED: {query.trace_id[:8]}... anomaly={query.anomaly_score:.2f}")

    runtime.set_escalation_callback(on_escalate)

    # Simulate telemetry stream
    print("\n--- Simulating 100 events ---")

    import random
    random.seed(42)

    for i in range(100):
        # Generate event
        if i % 20 == 0:
            # Anomalous event
            event_type = random.choice(['error', 'crash', 'security'])
            data = {'severity': random.uniform(0.7, 1.0), 'count': random.randint(5, 20)}
        else:
            # Normal event
            event_type = random.choice(['metric', 'log', 'heartbeat'])
            data = {'value': random.uniform(0.1, 0.5), 'count': random.randint(1, 5)}

        decision = runtime.ingest_telemetry(
            source=f"source_{i % 5}",
            event_type=event_type,
            data=data
        )

        if i < 5 or decision in (Decision.ESCALATE, Decision.EMERGENCY):
            print(f"  Event {i}: {event_type} -> {decision.value}")

    # Show stats
    print(f"\n--- Card Stats ---")
    stats = runtime.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  Total escalations: {len(escalations)}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
