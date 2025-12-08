"""
Policy Compiler - LLM Output to HDC+SNN
=======================================

Full pipeline for compiling LLM-generated policies into:
1. HDC policy vectors (for correlation-based matching on card)
2. SNN weight deltas (for Hebbian updates on card)
3. Structured rule representation (for human inspection)

The compiler handles:
- Parsing LLM JSON output with validation
- Encoding policies as hypervectors using binding/bundling
- Generating sparse SNN weight updates
- Version management for policy evolution
"""

import json
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PolicyCompilerConfig:
    """Configuration for the policy compiler."""
    hv_dim: int = 8192
    snn_delta_scale: float = 1.0
    max_conditions: int = 10
    max_action_params: int = 10
    default_expires_s: int = 3600
    validate_json: bool = True


# ============================================================================
# Structured Policy
# ============================================================================

@dataclass
class StructuredPolicy:
    """
    Structured representation of a policy.

    Extracted from LLM output and used to generate HDC vectors and SNN deltas.
    """
    name: str
    conditions: Dict[str, Any]
    action: Dict[str, Any]
    risk_level: str
    expires_in_s: int = 0
    version: int = 1
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique ID for this policy."""
        content = f"{self.name}:{self.version}:{json.dumps(self.conditions, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'conditions': self.conditions,
            'action': self.action,
            'risk_level': self.risk_level,
            'expires_in_s': self.expires_in_s,
            'version': self.version,
            'parent_id': self.parent_id,
            'tags': self.tags,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredPolicy':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            conditions=data['conditions'],
            action=data['action'],
            risk_level=data.get('risk_level', 'medium'),
            expires_in_s=data.get('expires_in_s', 0),
            version=data.get('version', 1),
            parent_id=data.get('parent_id'),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )


# ============================================================================
# Condition Operators
# ============================================================================

@dataclass
class Condition:
    """A single condition in a policy."""
    metric: str
    operator: str  # '>', '<', '>=', '<=', '==', '!=', 'in', 'not_in'
    value: Any
    weight: float = 1.0

    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate this condition against metrics."""
        if self.metric not in metrics:
            return False

        actual = metrics[self.metric]

        if self.operator == '>':
            return actual > self.value
        elif self.operator == '<':
            return actual < self.value
        elif self.operator == '>=':
            return actual >= self.value
        elif self.operator == '<=':
            return actual <= self.value
        elif self.operator == '==':
            return actual == self.value
        elif self.operator == '!=':
            return actual != self.value
        elif self.operator == 'in':
            return actual in self.value
        elif self.operator == 'not_in':
            return actual not in self.value
        else:
            return False


def parse_conditions(conditions_dict: Dict[str, Any]) -> List[Condition]:
    """Parse conditions dictionary into Condition objects."""
    conditions = []

    for metric, spec in conditions_dict.items():
        if isinstance(spec, dict):
            # Format: {"op": ">", "value": 0.7}
            op = spec.get('op', '>')
            value = spec.get('value', 0)
            weight = spec.get('weight', 1.0)
        else:
            # Simple format: just a threshold value
            op = '>'
            value = spec
            weight = 1.0

        conditions.append(Condition(metric, op, value, weight))

    return conditions


# ============================================================================
# Policy Compiler
# ============================================================================

class PolicyCompiler:
    """
    Compiles LLM output into HDC policy vectors and SNN weight deltas.

    The compilation process:
    1. Parse LLM JSON output
    2. Validate and normalize policy structure
    3. Generate HDC vector using binding/bundling
    4. Generate SNN weight deltas for card
    """

    def __init__(
        self,
        codebook: Dict[str, np.ndarray],
        config: Optional[PolicyCompilerConfig] = None
    ):
        self.codebook = codebook
        self.config = config or PolicyCompilerConfig()
        self.dim = config.hv_dim if config else 8192

        # Ensure codebook has required base vectors
        self._ensure_base_vectors()

        # Track compiled policies
        self._compiled_count = 0

    def _ensure_base_vectors(self) -> None:
        """Ensure codebook has required base vectors."""
        required = ['POLICY', 'RISK_LOW', 'RISK_MEDIUM', 'RISK_HIGH',
                   'THROTTLE', 'ALERT', 'BLOCK', 'ALLOW']

        rng = np.random.default_rng(42)
        for name in required:
            if name not in self.codebook:
                self.codebook[name] = rng.choice([-1, 1], size=self.dim).astype(np.int8)

    def parse_llm_output(self, text: str) -> Optional[StructuredPolicy]:
        """
        Parse LLM output text into a StructuredPolicy.

        Handles various JSON formats and extracts policy information.
        """
        # Try to find JSON in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())

            # Check if policy is needed
            if not data.get('policy_needed', True):
                return None

            # Extract policy object (might be nested)
            policy_data = data.get('policy', data)

            # Validate required fields
            if 'name' not in policy_data and 'policy_name' not in policy_data:
                return None

            name = policy_data.get('name', policy_data.get('policy_name', 'unnamed'))
            conditions = policy_data.get('conditions', {})
            action = policy_data.get('action', {'type': 'alert'})
            risk_level = policy_data.get('risk_level', 'medium')
            expires_in_s = policy_data.get('expires_in_s', self.config.default_expires_s)

            # Validate
            if self.config.validate_json:
                if len(conditions) > self.config.max_conditions:
                    conditions = dict(list(conditions.items())[:self.config.max_conditions])

                if risk_level not in ('low', 'medium', 'high'):
                    risk_level = 'medium'

            return StructuredPolicy(
                name=name,
                conditions=conditions,
                action=action,
                risk_level=risk_level,
                expires_in_s=expires_in_s,
                metadata={'raw_response': text[:500]}  # Store truncated original
            )

        except json.JSONDecodeError as e:
            return None

    def encode_policy_hv(self, policy: StructuredPolicy) -> np.ndarray:
        """
        Encode a policy as a hypervector.

        Uses HDC operations:
        - Binding (XOR for binary, multiply for bipolar): associates concepts
        - Bundling (majority): combines multiple concepts

        Result is a bipolar {-1, +1} vector.
        """
        hv = np.zeros(self.dim, dtype=np.float32)

        # Start with POLICY base vector
        hv += self.codebook['POLICY'].astype(np.float32)

        # Bind risk level
        risk_key = f"RISK_{policy.risk_level.upper()}"
        if risk_key in self.codebook:
            hv = self._bind(hv, self.codebook[risk_key].astype(np.float32))

        # Bind each condition
        for cond_name, cond_spec in policy.conditions.items():
            key = cond_name.upper().replace(' ', '_')

            if key in self.codebook:
                hv = self._bind(hv, self.codebook[key].astype(np.float32))
            else:
                # Generate deterministic vector for unknown concepts
                hv = self._bind(hv, self._generate_concept_hv(key))

        # Bind action type
        action_type = policy.action.get('type', 'alert').upper()
        if action_type in self.codebook:
            hv = self._bind(hv, self.codebook[action_type].astype(np.float32))

        # Normalize to bipolar
        self._compiled_count += 1
        return np.sign(hv + 1e-6).astype(np.int8)

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bipolar binding (element-wise multiply)."""
        return a * b

    def _bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple vectors (element-wise sum then sign)."""
        result = np.sum(vectors, axis=0)
        return np.sign(result + 1e-6)

    def _generate_concept_hv(self, name: str) -> np.ndarray:
        """Generate a deterministic HV for an unknown concept."""
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=self.dim).astype(np.float32)

    def policy_to_snn_deltas(self, policy: StructuredPolicy) -> Dict[str, float]:
        """
        Generate SNN weight deltas from a policy.

        Maps policy conditions and actions to weight adjustments
        that can be applied via Hebbian learning on the card.
        """
        deltas = {}
        scale = self.config.snn_delta_scale

        # Each condition increases sensitivity to that metric
        for cond_name, cond_spec in policy.conditions.items():
            weight = 1.0
            if isinstance(cond_spec, dict):
                weight = cond_spec.get('weight', 1.0)
            deltas[cond_name] = weight * scale

        # Risk level affects overall response strength
        risk_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }.get(policy.risk_level, 1.0)
        deltas['_risk_multiplier'] = risk_multiplier

        # Action type determines response direction
        action_type = policy.action.get('type', 'alert')
        action_strength = {
            'throttle': 0.5,
            'block': 1.0,
            'alert': 0.3,
            'allow': -0.5,
            'ignore': -1.0
        }.get(action_type, 0.3)
        deltas['_action_strength'] = action_strength * scale

        return deltas

    def compile(self, llm_output: str) -> Optional[Tuple[StructuredPolicy, np.ndarray, Dict[str, float]]]:
        """
        Full compilation pipeline.

        Returns:
            Tuple of (policy, policy_hv, snn_deltas) or None if no policy needed
        """
        policy = self.parse_llm_output(llm_output)
        if policy is None:
            return None

        policy_hv = self.encode_policy_hv(policy)
        snn_deltas = self.policy_to_snn_deltas(policy)

        return policy, policy_hv, snn_deltas

    def get_stats(self) -> Dict[str, Any]:
        """Get compiler statistics."""
        return {
            'compiled_count': self._compiled_count,
            'codebook_size': len(self.codebook),
            'hv_dim': self.dim
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the policy compiler."""
    print("=" * 60)
    print("Policy Compiler Demo")
    print("=" * 60)

    # Create compiler with test codebook
    dim = 1024
    rng = np.random.default_rng(42)

    codebook = {
        'POLICY': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_LOW': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_MEDIUM': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_HIGH': rng.choice([-1, 1], size=dim).astype(np.int8),
        'THROTTLE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'ALERT': rng.choice([-1, 1], size=dim).astype(np.int8),
        'CPU_USAGE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'MEMORY_PRESSURE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'ROUTE_FLAP': rng.choice([-1, 1], size=dim).astype(np.int8),
    }

    config = PolicyCompilerConfig(hv_dim=dim)
    compiler = PolicyCompiler(codebook, config)

    # Test LLM output
    llm_output = '''
    Based on the telemetry, I recommend the following policy:

    ```json
    {
        "analysis": "High CPU usage combined with route flapping indicates network stress",
        "policy_needed": true,
        "policy": {
            "name": "network_stress_throttle_v1",
            "conditions": {
                "cpu_usage": {"op": ">", "value": 0.8},
                "route_flap": {"op": ">", "value": 5}
            },
            "action": {
                "type": "throttle",
                "params": {"amount": 0.3, "duration_s": 300}
            },
            "risk_level": "medium",
            "expires_in_s": 3600
        }
    }
    ```
    '''

    print("\n--- Input LLM Output ---")
    print(llm_output[:200] + "...")

    print("\n--- Compiling Policy ---")
    result = compiler.compile(llm_output)

    if result:
        policy, policy_hv, snn_deltas = result

        print(f"\nPolicy Name: {policy.name}")
        print(f"Policy ID: {policy.id}")
        print(f"Risk Level: {policy.risk_level}")
        print(f"Conditions: {policy.conditions}")
        print(f"Action: {policy.action}")

        print(f"\nPolicy HV shape: {policy_hv.shape}")
        print(f"Policy HV sample: {policy_hv[:10]}")

        print(f"\nSNN Deltas:")
        for k, v in snn_deltas.items():
            print(f"  {k}: {v:.2f}")
    else:
        print("No policy generated")

    print(f"\n--- Compiler Stats ---")
    stats = compiler.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
