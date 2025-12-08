"""
Cortex Bridge - Card <-> GPU LLM Communication
==============================================

High-level bridge that:
1. Receives STATE_HPV_QUERY from the card
2. Uses HyperdimensionalProbe to decode the HPV
3. Builds LLM prompt and gets response
4. Uses PolicyCompiler to generate NEW_POLICY
5. Sends policy back to card

This is the "synapse" between the subcortex (card) and cortex (GPU).
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from abc import ABC, abstractmethod
import numpy as np

from ara.bridge.messages import StateHPVQuery, NewPolicy
from ara.hdc.probe import HDProbe

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Backend Interface
# ============================================================================

class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a response from the LLM."""
        pass


class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Return a mock policy response."""
        return json.dumps({
            "analysis": "Mock analysis: detected anomalous pattern requiring throttling",
            "policy_needed": True,
            "policy": {
                "name": "mock_throttle_v1",
                "conditions": {"anomaly_score": {"op": ">", "value": 0.7}},
                "action": {"type": "throttle", "params": {"amount": 0.3}},
                "risk_level": "medium",
                "expires_in_s": 3600
            }
        })


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("OpenAI not installed, using mock response")
            return MockLLMBackend().generate(prompt)


# ============================================================================
# Policy Compiler (simplified - full version in policy/compiler.py)
# ============================================================================

@dataclass
class StructuredPolicy:
    """Structured policy extracted from LLM response."""
    name: str
    conditions: Dict[str, Any]
    action: Dict[str, Any]
    risk_level: str
    expires_in_s: int = 0


class PolicyCompiler:
    """
    Compiles LLM output into HDC policy vectors and SNN deltas.

    This is a simplified version - the full implementation is in
    ara/policy/compiler.py
    """

    def __init__(self, codebook: Dict[str, np.ndarray]):
        self.codebook = codebook
        self.dim = next(iter(codebook.values())).shape[0] if codebook else 1024

    def parse_llm_output(self, text: str) -> Optional[StructuredPolicy]:
        """Parse LLM JSON output into StructuredPolicy."""
        try:
            # Try to extract JSON from the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])

                if not data.get('policy_needed', False):
                    return None

                policy = data.get('policy', {})
                return StructuredPolicy(
                    name=policy.get('name', 'unnamed_policy'),
                    conditions=policy.get('conditions', {}),
                    action=policy.get('action', {}),
                    risk_level=policy.get('risk_level', 'medium'),
                    expires_in_s=policy.get('expires_in_s', 0)
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM output: {e}")
            return None

        return None

    def encode_policy_hv(self, policy: StructuredPolicy) -> np.ndarray:
        """Encode a policy as a hypervector."""
        hv = np.zeros(self.dim, dtype=np.float32)

        def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Bipolar binding (element-wise multiply)."""
            return a * b

        # Start with POLICY base vector
        if 'POLICY' in self.codebook:
            hv = self.codebook['POLICY'].astype(np.float32)

        # Bind risk level
        risk_key = f"RISK_{policy.risk_level.upper()}"
        if risk_key in self.codebook:
            hv = bind(hv, self.codebook[risk_key].astype(np.float32))

        # Bind conditions
        for cond_name in policy.conditions.keys():
            key = cond_name.upper()
            if key in self.codebook:
                hv = bind(hv, self.codebook[key].astype(np.float32))

        # Bind action type
        action_key = policy.action.get('type', 'unknown').upper()
        if action_key in self.codebook:
            hv = bind(hv, self.codebook[action_key].astype(np.float32))

        # Normalize to bipolar
        return np.sign(hv + 1e-6).astype(np.int8)

    def policy_to_snn_deltas(self, policy: StructuredPolicy) -> Dict[str, float]:
        """
        Generate SNN weight deltas from a policy.

        This encodes "increase sensitivity" for each condition.
        """
        deltas = {}
        for name in policy.conditions.keys():
            # Positive delta = increase sensitivity to this feature
            deltas[name] = 1.0

        # Action type affects how strongly we respond
        action_type = policy.action.get('type', 'alert')
        if action_type == 'throttle':
            deltas['_action_strength'] = 0.5
        elif action_type == 'block':
            deltas['_action_strength'] = 1.0
        elif action_type == 'allow':
            deltas['_action_strength'] = -0.5

        return deltas


# ============================================================================
# Cortex Bridge
# ============================================================================

@dataclass
class BridgeConfig:
    """Configuration for the cortex bridge."""
    hv_dim: int = 8192
    max_response_tokens: int = 1024
    timeout_s: float = 30.0
    log_queries: bool = True
    log_policies: bool = True


@dataclass
class BridgeStats:
    """Statistics for the bridge."""
    queries_received: int = 0
    policies_sent: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0


class CortexBridge:
    """
    High-level bridge between card (subcortex) and GPU (cortex).

    Handles the full query -> probe -> LLM -> compile -> response flow.
    """

    def __init__(
        self,
        llm: LLMBackend,
        probe: HDProbe,
        compiler: PolicyCompiler,
        config: Optional[BridgeConfig] = None
    ):
        self.llm = llm
        self.probe = probe
        self.compiler = compiler
        self.config = config or BridgeConfig()
        self.stats = BridgeStats()

        # Policy cache (avoid re-generating same policies)
        self._policy_cache: Dict[str, NewPolicy] = {}

    def handle_query(self, query: StateHPVQuery) -> Optional[NewPolicy]:
        """
        Handle a STATE_HPV_QUERY and return a NEW_POLICY if needed.

        Returns None if no policy update is required.
        """
        start_time = time.time()
        self.stats.queries_received += 1

        try:
            # 1. Decode HPV using probe
            hv = query.get_hv()
            probe_result = self.probe.probe(hv)
            probe_summary = self._format_probe_summary(probe_result)

            if self.config.log_queries:
                logger.info(f"Query {query.trace_id}: anomaly={query.anomaly_score:.2f}, "
                           f"top_concept={probe_result.top_concepts[0] if probe_result.top_concepts else 'none'}")

            # 2. Build LLM prompt
            prompt = query.to_llm_prompt(probe_summary)

            # 3. Get LLM response
            llm_response = self.llm.generate(prompt, self.config.max_response_tokens)

            # 4. Parse into structured policy
            structured = self.compiler.parse_llm_output(llm_response)

            if structured is None:
                # No policy needed
                latency = (time.time() - start_time) * 1000
                self._update_stats(latency)
                return None

            # 5. Compile to HPV + SNN deltas
            policy_hv = self.compiler.encode_policy_hv(structured)
            snn_deltas = self.compiler.policy_to_snn_deltas(structured)

            # 6. Create NEW_POLICY message
            policy = NewPolicy.create(
                trace_id=query.trace_id,
                policy_id=structured.name,
                policy_hv=policy_hv,
                snn_deltas=snn_deltas,
                action_type=structured.action.get('type', 'alert'),
                conditions=structured.conditions,
                risk_level=structured.risk_level,
                expires_in_s=structured.expires_in_s,
                human_description=f"Auto-generated policy: {structured.name}"
            )

            self.stats.policies_sent += 1

            if self.config.log_policies:
                logger.info(f"Policy {policy.policy_id} generated for {query.trace_id}")

            latency = (time.time() - start_time) * 1000
            self._update_stats(latency)

            return policy

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Error handling query {query.trace_id}: {e}")
            return None

    def _format_probe_summary(self, probe_result) -> str:
        """Format probe result as text for LLM."""
        lines = ["Hyperdimensional State Probe:"]

        for concept, sim in probe_result.top_concepts:
            lines.append(f"  - {concept}: {sim:.2f}")

        if probe_result.is_anomaly:
            lines.append(f"  [ANOMALY: max_sim={probe_result.max_similarity:.2f} below threshold]")

        if probe_result.is_novel:
            lines.append("  [NOVEL: no good match in codebook]")

        return '\n'.join(lines)

    def _update_stats(self, latency_ms: float) -> None:
        """Update statistics."""
        self.stats.total_latency_ms += latency_ms
        total = self.stats.queries_received
        self.stats.avg_latency_ms = self.stats.total_latency_ms / max(1, total)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'queries_received': self.stats.queries_received,
            'policies_sent': self.stats.policies_sent,
            'errors': self.stats.errors,
            'avg_latency_ms': self.stats.avg_latency_ms,
            'policy_rate': self.stats.policies_sent / max(1, self.stats.queries_received)
        }


# ============================================================================
# Async Bridge (for production use)
# ============================================================================

class AsyncCortexBridge:
    """
    Async version of CortexBridge for high-throughput scenarios.

    Uses a queue to batch queries and process them efficiently.
    """

    def __init__(self, sync_bridge: CortexBridge):
        self.bridge = sync_bridge
        self._pending: List[StateHPVQuery] = []

    async def handle_query(self, query: StateHPVQuery) -> Optional[NewPolicy]:
        """Async query handling (delegates to sync for now)."""
        # In production, this would use asyncio properly
        import asyncio
        return await asyncio.to_thread(self.bridge.handle_query, query)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the cortex bridge."""
    print("=" * 60)
    print("Cortex Bridge Demo")
    print("=" * 60)

    # Create components
    dim = 1024

    # Build a simple codebook
    rng = np.random.default_rng(42)
    codebook = {
        'POLICY': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_LOW': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_MEDIUM': rng.choice([-1, 1], size=dim).astype(np.int8),
        'RISK_HIGH': rng.choice([-1, 1], size=dim).astype(np.int8),
        'THROTTLE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'ALERT': rng.choice([-1, 1], size=dim).astype(np.int8),
        'BLOCK': rng.choice([-1, 1], size=dim).astype(np.int8),
        'ROUTE_FLAP': rng.choice([-1, 1], size=dim).astype(np.int8),
        'CPU_SPIKE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'MEMORY_PRESSURE': rng.choice([-1, 1], size=dim).astype(np.int8),
        'NETWORK_ERROR': rng.choice([-1, 1], size=dim).astype(np.int8),
    }

    # Create probe
    probe = HDProbe(dim=dim)
    for name, vec in codebook.items():
        if not name.startswith('RISK_') and name != 'POLICY':
            probe.add_concept(name, vec)

    # Create compiler
    compiler = PolicyCompiler(codebook)

    # Create mock LLM
    llm = MockLLMBackend()

    # Create bridge
    bridge = CortexBridge(llm, probe, compiler)

    # Simulate a query from the card
    print("\n--- Simulating Card Query ---")

    state_hv = rng.choice([-1, 1], size=dim).astype(np.int8)
    # Make it somewhat similar to ROUTE_FLAP
    state_hv[:dim//2] = codebook['ROUTE_FLAP'][:dim//2]

    query = StateHPVQuery.create(
        hv=state_hv,
        features={
            'cpu_usage': 0.45,
            'memory_mb': 2048,
            'route_flap_count': 15,
            'friction': 0.72
        },
        anomaly_score=0.81,
        urgency=0.65,
        recent_policies=['baseline_v1'],
        context="Multiple BGP route changes in last 5 minutes"
    )

    print(f"Query ID: {query.trace_id}")
    print(f"Anomaly Score: {query.anomaly_score}")
    print(f"Features: {query.features}")

    # Process through bridge
    print("\n--- Processing Query ---")
    policy = bridge.handle_query(query)

    if policy:
        print(f"\n--- Policy Generated ---")
        print(f"Policy ID: {policy.policy_id}")
        print(f"Action: {policy.action_type}")
        print(f"Risk Level: {policy.risk_level}")
        print(f"Conditions: {policy.conditions}")
        print(f"SNN Deltas: {policy.get_snn_deltas()}")
    else:
        print("\nNo policy update needed")

    # Show stats
    print(f"\n--- Bridge Stats ---")
    stats = bridge.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
