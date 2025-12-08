"""
Bridge Messages - Wire Protocol for Card <-> Cortex
===================================================

Defines the message types exchanged between the neuromorphic card
(subcortex) and the GPU-based LLM (cortex).

Message Types:
    0x01 STATE_HPV_QUERY - Card asks cortex for guidance
    0x02 NEW_POLICY      - Cortex sends updated policy to card
    0x03 HEARTBEAT       - Keep-alive between card and cortex
    0x04 POLICY_ACK      - Card confirms policy installation

Wire Format:
    [msg_type: u8][version: u8][payload_len: u32][payload: bytes][crc32: u32]
"""

import struct
import json
import zlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# ============================================================================
# Constants
# ============================================================================

MSG_STATE_HPV_QUERY = 0x01
MSG_NEW_POLICY = 0x02
MSG_HEARTBEAT = 0x03
MSG_POLICY_ACK = 0x04

PROTOCOL_VERSION = 1
DEFAULT_HV_DIM = 8192  # Default hypervector dimension


# ============================================================================
# Encoding Utilities
# ============================================================================

def pack_bipolar_to_bytes(hv: np.ndarray) -> bytes:
    """
    Pack a bipolar {-1, +1} hypervector into bytes.

    Each element becomes one bit: +1 -> 1, -1 -> 0
    """
    # Convert to {0, 1}
    bits = ((hv + 1) // 2).astype(np.uint8)

    # Pack into bytes (8 bits per byte)
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= (1 << (i % 8))

    return packed.tobytes()


def unpack_bytes_to_bipolar(data: bytes, dim: int) -> np.ndarray:
    """
    Unpack bytes back to a bipolar {-1, +1} hypervector.
    """
    packed = np.frombuffer(data, dtype=np.uint8)
    hv = np.zeros(dim, dtype=np.int8)

    for i in range(dim):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(packed):
            bit = (packed[byte_idx] >> bit_idx) & 1
            hv[i] = 1 if bit else -1
        else:
            hv[i] = -1  # Default to -1 for missing bits

    return hv


def serialize_snn_deltas(deltas: Dict[str, float]) -> bytes:
    """
    Serialize SNN weight deltas as JSON + compression.

    Format: [n_entries: u16][entries: (name_len, name, delta_fp16)...]

    For card consumption, this would be further translated to
    neuron indices based on the card's codebook.
    """
    # Simple JSON for now; can optimize to binary later
    return zlib.compress(json.dumps(deltas).encode('utf-8'))


def deserialize_snn_deltas(data: bytes) -> Dict[str, float]:
    """Deserialize SNN weight deltas."""
    return json.loads(zlib.decompress(data).decode('utf-8'))


# ============================================================================
# STATE_HPV_QUERY: Card -> Cortex
# ============================================================================

@dataclass
class StateHPVQuery:
    """
    Message from card to cortex requesting guidance.

    Sent when the card's subcortex decides to escalate an event
    to the GPU-based LLM for higher-level reasoning.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds
        trace_id: UUID for request/response correlation
        hv_bytes: Packed hypervector representing current state
        hv_dim: Dimension of the hypervector
        features: Telemetry features extracted by the card
        anomaly_score: How anomalous the current state is (0-1)
        urgency: How urgent the query is (0-1)
        recent_policies: List of recently applied policy IDs
        context: Optional additional context string
    """
    timestamp_ms: int
    trace_id: str
    hv_bytes: bytes
    hv_dim: int
    features: Dict[str, Any]
    anomaly_score: float = 0.5
    urgency: float = 0.5
    recent_policies: List[str] = field(default_factory=list)
    context: Optional[str] = None

    @classmethod
    def create(
        cls,
        hv: np.ndarray,
        features: Dict[str, Any],
        anomaly_score: float = 0.5,
        urgency: float = 0.5,
        recent_policies: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> 'StateHPVQuery':
        """Create a new query from a hypervector and features."""
        return cls(
            timestamp_ms=int(time.time() * 1000),
            trace_id=str(uuid.uuid4()),
            hv_bytes=pack_bipolar_to_bytes(hv),
            hv_dim=len(hv),
            features=features,
            anomaly_score=anomaly_score,
            urgency=urgency,
            recent_policies=recent_policies or [],
            context=context
        )

    def get_hv(self) -> np.ndarray:
        """Unpack the hypervector."""
        return unpack_bytes_to_bipolar(self.hv_bytes, self.hv_dim)

    def to_bytes(self) -> bytes:
        """Serialize to wire format."""
        # Build payload
        features_json = json.dumps(self.features).encode('utf-8')
        policies_json = json.dumps(self.recent_policies).encode('utf-8')
        context_bytes = (self.context or '').encode('utf-8')

        payload = struct.pack(
            '<QI',  # timestamp_ms, hv_dim
            self.timestamp_ms,
            self.hv_dim
        )
        payload += struct.pack('<H', len(self.trace_id)) + self.trace_id.encode('utf-8')
        payload += struct.pack('<I', len(self.hv_bytes)) + self.hv_bytes
        payload += struct.pack('<I', len(features_json)) + features_json
        payload += struct.pack('<ff', self.anomaly_score, self.urgency)
        payload += struct.pack('<I', len(policies_json)) + policies_json
        payload += struct.pack('<I', len(context_bytes)) + context_bytes

        # Add header and CRC
        header = struct.pack('<BBI', MSG_STATE_HPV_QUERY, PROTOCOL_VERSION, len(payload))
        crc = zlib.crc32(payload)

        return header + payload + struct.pack('<I', crc)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'StateHPVQuery':
        """Deserialize from wire format."""
        # Parse header
        msg_type, version, payload_len = struct.unpack('<BBI', data[:6])
        assert msg_type == MSG_STATE_HPV_QUERY
        assert version == PROTOCOL_VERSION

        payload = data[6:6+payload_len]
        crc_expected = struct.unpack('<I', data[6+payload_len:6+payload_len+4])[0]
        crc_actual = zlib.crc32(payload)
        assert crc_actual == crc_expected, "CRC mismatch"

        # Parse payload
        offset = 0
        timestamp_ms, hv_dim = struct.unpack_from('<QI', payload, offset)
        offset += 12

        trace_id_len = struct.unpack_from('<H', payload, offset)[0]
        offset += 2
        trace_id = payload[offset:offset+trace_id_len].decode('utf-8')
        offset += trace_id_len

        hv_len = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        hv_bytes = payload[offset:offset+hv_len]
        offset += hv_len

        features_len = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        features = json.loads(payload[offset:offset+features_len].decode('utf-8'))
        offset += features_len

        anomaly_score, urgency = struct.unpack_from('<ff', payload, offset)
        offset += 8

        policies_len = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        recent_policies = json.loads(payload[offset:offset+policies_len].decode('utf-8'))
        offset += policies_len

        context_len = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        context = payload[offset:offset+context_len].decode('utf-8') if context_len > 0 else None

        return cls(
            timestamp_ms=timestamp_ms,
            trace_id=trace_id,
            hv_bytes=hv_bytes,
            hv_dim=hv_dim,
            features=features,
            anomaly_score=anomaly_score,
            urgency=urgency,
            recent_policies=recent_policies,
            context=context
        )

    def to_llm_prompt(self, probe_summary: str) -> str:
        """
        Generate an LLM prompt from this query.

        Args:
            probe_summary: Human-readable summary from HyperdimensionalProbe
        """
        features_str = '\n'.join(f"  - {k}: {v}" for k, v in self.features.items())
        policies_str = ', '.join(self.recent_policies) if self.recent_policies else 'none'

        prompt = f"""You are Ara's Cortex - the reasoning layer of a neuromorphic AI system.

The subcortex (card-side SNN+HDC) has escalated an event for your analysis.

## Compressed State (Hyperdimensional Probe):
{probe_summary}

## Telemetry Features:
{features_str}

## Metrics:
  - Anomaly Score: {self.anomaly_score:.2f}
  - Urgency: {self.urgency:.2f}

## Recent Policies Applied:
  {policies_str}

## Task:
1. Analyze what is happening based on the compressed state and features.
2. Decide if the current policies are sufficient or need adjustment.
3. If needed, propose a new or updated policy.

## Response Format (JSON):
```json
{{
  "analysis": "Brief explanation of the situation",
  "policy_needed": true/false,
  "policy": {{
    "name": "policy_name_v1",
    "conditions": {{"metric_name": {{"op": ">", "value": 0.7}}}},
    "action": {{"type": "throttle|alert|block|allow", "params": {{}}}},
    "risk_level": "low|medium|high",
    "expires_in_s": 3600
  }}
}}
```
"""
        if self.context:
            prompt += f"\n## Additional Context:\n{self.context}\n"

        return prompt


# ============================================================================
# NEW_POLICY: Cortex -> Card
# ============================================================================

@dataclass
class NewPolicy:
    """
    Message from cortex to card with a new/updated policy.

    Contains both the HDC-encoded policy (for correlation) and
    SNN weight deltas (for Hebbian updates on the card).

    Attributes:
        trace_id: UUID matching the original query
        policy_id: Unique identifier for this policy
        policy_hv_bytes: Packed hypervector encoding the policy
        hv_dim: Dimension of the policy hypervector
        snn_deltas_bytes: Serialized SNN weight adjustments
        action_type: High-level action type (throttle, alert, etc.)
        conditions: Structured conditions that trigger the policy
        risk_level: Risk assessment (low, medium, high)
        expires_at_ms: When the policy expires (0 = never)
        human_description: Human-readable description
    """
    trace_id: str
    policy_id: str
    policy_hv_bytes: bytes
    hv_dim: int
    snn_deltas_bytes: bytes
    action_type: str
    conditions: Dict[str, Any]
    risk_level: str = "medium"
    expires_at_ms: int = 0
    human_description: str = ""

    @classmethod
    def create(
        cls,
        trace_id: str,
        policy_id: str,
        policy_hv: np.ndarray,
        snn_deltas: Dict[str, float],
        action_type: str,
        conditions: Dict[str, Any],
        risk_level: str = "medium",
        expires_in_s: int = 0,
        human_description: str = ""
    ) -> 'NewPolicy':
        """Create a new policy message."""
        expires_at_ms = int((time.time() + expires_in_s) * 1000) if expires_in_s > 0 else 0

        return cls(
            trace_id=trace_id,
            policy_id=policy_id,
            policy_hv_bytes=pack_bipolar_to_bytes(policy_hv),
            hv_dim=len(policy_hv),
            snn_deltas_bytes=serialize_snn_deltas(snn_deltas),
            action_type=action_type,
            conditions=conditions,
            risk_level=risk_level,
            expires_at_ms=expires_at_ms,
            human_description=human_description
        )

    def get_policy_hv(self) -> np.ndarray:
        """Unpack the policy hypervector."""
        return unpack_bytes_to_bipolar(self.policy_hv_bytes, self.hv_dim)

    def get_snn_deltas(self) -> Dict[str, float]:
        """Deserialize SNN weight deltas."""
        return deserialize_snn_deltas(self.snn_deltas_bytes)

    def is_expired(self) -> bool:
        """Check if the policy has expired."""
        if self.expires_at_ms == 0:
            return False
        return time.time() * 1000 > self.expires_at_ms

    def to_bytes(self) -> bytes:
        """Serialize to wire format."""
        # Build payload
        conditions_json = json.dumps(self.conditions).encode('utf-8')

        payload = struct.pack('<H', len(self.trace_id)) + self.trace_id.encode('utf-8')
        payload += struct.pack('<H', len(self.policy_id)) + self.policy_id.encode('utf-8')
        payload += struct.pack('<I', self.hv_dim)
        payload += struct.pack('<I', len(self.policy_hv_bytes)) + self.policy_hv_bytes
        payload += struct.pack('<I', len(self.snn_deltas_bytes)) + self.snn_deltas_bytes
        payload += struct.pack('<H', len(self.action_type)) + self.action_type.encode('utf-8')
        payload += struct.pack('<I', len(conditions_json)) + conditions_json
        payload += struct.pack('<H', len(self.risk_level)) + self.risk_level.encode('utf-8')
        payload += struct.pack('<Q', self.expires_at_ms)
        payload += struct.pack('<I', len(self.human_description)) + self.human_description.encode('utf-8')

        # Add header and CRC
        header = struct.pack('<BBI', MSG_NEW_POLICY, PROTOCOL_VERSION, len(payload))
        crc = zlib.crc32(payload)

        return header + payload + struct.pack('<I', crc)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'NewPolicy':
        """Deserialize from wire format."""
        msg_type, version, payload_len = struct.unpack('<BBI', data[:6])
        assert msg_type == MSG_NEW_POLICY
        assert version == PROTOCOL_VERSION

        payload = data[6:6+payload_len]
        crc_expected = struct.unpack('<I', data[6+payload_len:6+payload_len+4])[0]
        assert zlib.crc32(payload) == crc_expected, "CRC mismatch"

        offset = 0

        def read_string(off):
            slen = struct.unpack_from('<H', payload, off)[0]
            return payload[off+2:off+2+slen].decode('utf-8'), off+2+slen

        def read_bytes(off):
            blen = struct.unpack_from('<I', payload, off)[0]
            return payload[off+4:off+4+blen], off+4+blen

        trace_id, offset = read_string(offset)
        policy_id, offset = read_string(offset)

        hv_dim = struct.unpack_from('<I', payload, offset)[0]
        offset += 4

        policy_hv_bytes, offset = read_bytes(offset)
        snn_deltas_bytes, offset = read_bytes(offset)

        action_type, offset = read_string(offset)

        conditions_bytes, offset = read_bytes(offset)
        conditions = json.loads(conditions_bytes.decode('utf-8'))

        risk_level, offset = read_string(offset)

        expires_at_ms = struct.unpack_from('<Q', payload, offset)[0]
        offset += 8

        desc_len = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        human_description = payload[offset:offset+desc_len].decode('utf-8')

        return cls(
            trace_id=trace_id,
            policy_id=policy_id,
            policy_hv_bytes=policy_hv_bytes,
            hv_dim=hv_dim,
            snn_deltas_bytes=snn_deltas_bytes,
            action_type=action_type,
            conditions=conditions,
            risk_level=risk_level,
            expires_at_ms=expires_at_ms,
            human_description=human_description
        )


# ============================================================================
# Test
# ============================================================================

def test_messages():
    """Test message serialization/deserialization."""
    print("Testing StateHPVQuery...")

    # Create a test query
    hv = np.random.choice([-1, 1], size=1024).astype(np.int8)
    query = StateHPVQuery.create(
        hv=hv,
        features={'cpu_usage': 0.75, 'friction': 0.6, 'error_count': 42},
        anomaly_score=0.8,
        urgency=0.7,
        recent_policies=['throttle_v1', 'alert_v2'],
        context="Router flapping detected"
    )

    # Round-trip
    wire = query.to_bytes()
    print(f"  Wire size: {len(wire)} bytes")

    query2 = StateHPVQuery.from_bytes(wire)
    assert query2.trace_id == query.trace_id
    assert abs(query2.anomaly_score - query.anomaly_score) < 1e-5, f"anomaly: {query2.anomaly_score} vs {query.anomaly_score}"
    assert np.array_equal(query2.get_hv(), hv)
    print("  PASS")

    print("\nTesting NewPolicy...")

    policy_hv = np.random.choice([-1, 1], size=1024).astype(np.int8)
    policy = NewPolicy.create(
        trace_id=query.trace_id,
        policy_id="throttle_flap_v3",
        policy_hv=policy_hv,
        snn_deltas={'route_flap': 0.5, 'friction': 0.3},
        action_type="throttle",
        conditions={'route_flap_score': {'op': '>', 'value': 0.7}},
        risk_level="medium",
        expires_in_s=3600,
        human_description="Throttle on route flap"
    )

    wire = policy.to_bytes()
    print(f"  Wire size: {len(wire)} bytes")

    policy2 = NewPolicy.from_bytes(wire)
    assert policy2.policy_id == policy.policy_id
    assert policy2.action_type == "throttle"
    assert np.array_equal(policy2.get_policy_hv(), policy_hv)
    print("  PASS")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_messages()
