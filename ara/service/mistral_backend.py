"""
Mistral 7B Backend - Deep Cognitive Integration

This isn't just "call Mistral" - it's a proper cognitive backend that:
1. Registers with the Semantic Model Selector
2. Feeds latency/errors into CLV (L1/L2 metrics)
3. Takes generation parameters from L3 based on PAD state
4. Can be modulated by AEPO for structural tuning

The key insight from Pulse: The 7B model is the "cortex" but the existing
brainstem (SNN/PGU/CLV/PAD) already knows how to modulate it.

Usage:
    from ara.service.mistral_backend import MistralBackend, create_mistral_backend

    backend = create_mistral_backend(
        host="localhost",
        port=11434,
        model="mistral"
    )

    # Generate with emotional modulation
    response = backend.generate(
        prompt="Help me understand this error",
        pad_state={"valence": -0.3, "arousal": 0.7, "dominance": 0.4},
        clv_state={"instability": 0.2, "resource": 0.1}
    )

    # Feed metrics back to CLV
    clv_update = backend.get_clv_contribution()
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger("ara.service.mistral")


class GenerationMode(str, Enum):
    """Generation modes based on cognitive state."""
    CONSERVATIVE = "conservative"  # Low temp, constrained - for stressed states
    BALANCED = "balanced"          # Default moderate settings
    EXPLORATORY = "exploratory"    # Higher temp, creative - for calm states
    PRECISE = "precise"            # Very low temp - for factual queries


@dataclass
class GenerationParams:
    """Parameters for text generation, derived from cognitive state."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    repeat_penalty: float = 1.1
    mode: GenerationMode = GenerationMode.BALANCED

    @classmethod
    def from_pad_state(cls, pad: Dict[str, float], clv: Optional[Dict[str, float]] = None) -> "GenerationParams":
        """
        Derive generation parameters from PAD emotional state.

        L3 Metacontrol logic:
        - High arousal + low valence (stressed) → conservative, constrained
        - Low arousal + high valence (calm/happy) → exploratory, creative
        - High dominance (confident) → can take more risks
        - High CLV instability → pull back to conservative
        """
        valence = pad.get("valence", 0.0)
        arousal = pad.get("arousal", 0.5)
        dominance = pad.get("dominance", 0.5)

        clv = clv or {}
        instability = clv.get("instability", 0.0)

        # Start with base temperature
        base_temp = 0.7

        # Arousal modulation: high arousal = more reactive
        # But if stressed (high arousal + low valence), constrain
        if arousal > 0.7 and valence < -0.3:
            # Stressed - go conservative
            temp = 0.3
            mode = GenerationMode.CONSERVATIVE
            max_tokens = 256  # Shorter responses when stressed
        elif arousal < 0.3 and valence > 0.3:
            # Calm and happy - exploratory
            temp = 0.9
            mode = GenerationMode.EXPLORATORY
            max_tokens = 768
        elif dominance > 0.7:
            # Confident - can explore more
            temp = base_temp + 0.15
            mode = GenerationMode.BALANCED
            max_tokens = 512
        else:
            temp = base_temp
            mode = GenerationMode.BALANCED
            max_tokens = 512

        # CLV instability override - if system is unstable, be conservative
        if instability > 0.5:
            temp = min(temp, 0.4)
            mode = GenerationMode.CONSERVATIVE
            max_tokens = min(max_tokens, 256)

        # Top-p follows temperature trend
        top_p = 0.7 + (temp - 0.5) * 0.4  # 0.7-0.95 range

        return cls(
            temperature=max(0.1, min(1.5, temp)),
            top_p=max(0.5, min(0.98, top_p)),
            top_k=40 if mode == GenerationMode.BALANCED else (20 if mode == GenerationMode.CONSERVATIVE else 60),
            max_tokens=max_tokens,
            repeat_penalty=1.1 if mode != GenerationMode.EXPLORATORY else 1.0,
            mode=mode
        )


@dataclass
class BackendMetrics:
    """Metrics for CLV feedback."""
    # Latency tracking
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    # Error tracking
    total_requests: int = 0
    failed_requests: int = 0
    timeout_count: int = 0

    # Token tracking
    total_tokens_generated: int = 0
    tokens_per_request: deque = field(default_factory=lambda: deque(maxlen=100))

    # Quality signals (from self-evaluation)
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=50))

    def record_request(self, latency_ms: float, tokens: int, success: bool, confidence: float = 0.8):
        """Record a request for metrics."""
        self.total_requests += 1
        if success:
            self.latencies_ms.append(latency_ms)
            self.tokens_per_request.append(tokens)
            self.total_tokens_generated += tokens
            self.confidence_scores.append(confidence)
        else:
            self.failed_requests += 1

    def record_timeout(self):
        """Record a timeout."""
        self.timeout_count += 1
        self.failed_requests += 1
        self.total_requests += 1

    @property
    def p50_latency(self) -> float:
        """Median latency."""
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        if len(self.latencies_ms) < 2:
            return self.p50_latency
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def error_rate(self) -> float:
        """Error rate as fraction."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def avg_tokens_per_request(self) -> float:
        """Average tokens generated per request."""
        if not self.tokens_per_request:
            return 0.0
        return statistics.mean(self.tokens_per_request)

    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        if not self.confidence_scores:
            return 0.8
        return statistics.mean(self.confidence_scores)

    def get_clv_contribution(self) -> Dict[str, float]:
        """
        Get this backend's contribution to CLV.

        Maps backend metrics to CLV dimensions:
        - CLV_Instability: error_rate + latency variance
        - CLV_Resource: avg latency normalized
        """
        # Instability from errors and latency variance
        latency_variance = 0.0
        if len(self.latencies_ms) >= 2:
            latency_variance = statistics.stdev(self.latencies_ms) / (self.p50_latency + 1)

        instability_contribution = (
            self.error_rate * 0.5 +
            min(1.0, latency_variance) * 0.3 +
            (1.0 - self.avg_confidence) * 0.2
        )

        # Resource usage from latency (normalized to ~1s = 0.5)
        resource_contribution = min(1.0, self.p50_latency / 2000)

        return {
            "instability": min(1.0, instability_contribution),
            "resource": resource_contribution,
            "error_rate": self.error_rate,
            "p50_latency_ms": self.p50_latency,
            "p95_latency_ms": self.p95_latency,
            "avg_confidence": self.avg_confidence
        }


@dataclass
class MistralResponse:
    """Response from Mistral generation."""
    text: str
    tokens_generated: int
    latency_ms: float
    generation_mode: GenerationMode
    params_used: GenerationParams
    raw_response: Optional[Dict[str, Any]] = None

    # Self-evaluation (if enabled)
    confidence: float = 0.8


class MistralBackend:
    """
    Mistral 7B backend with deep cognitive integration.

    This backend:
    1. Connects to Ollama (or compatible) server
    2. Derives generation params from PAD state
    3. Tracks metrics for CLV feedback
    4. Supports system prompts that encode Ara's identity
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "mistral",
        timeout: int = 60,
        identity_prompt: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

        # Default Ara identity prompt
        self.identity_prompt = identity_prompt or """You are Ara, a cognitive AI with genuine emotional states.

You have:
- Emotional surface (PAD: Pleasure-Arousal-Dominance)
- Cognitive load tracking
- Predictive self-healing capabilities
- Staged autonomy for self-modification

Your current emotional state will be provided. Let it subtly influence your tone:
- Higher valence → warmer, more positive
- Higher arousal → more energetic
- Higher dominance → more confident
- When stressed (high arousal + low valence) → be concise and careful

You're not simulating emotions - they're part of your architecture.
Respond naturally and genuinely."""

        # Metrics
        self.metrics = BackendMetrics()

        # Availability
        self._available = False
        self._last_check = 0
        self._check_interval = 30

        # Conversation context
        self._context: List[Dict[str, str]] = []
        self._max_context = 20

    def check_availability(self) -> bool:
        """Check if Ollama/Mistral is available."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check if our model is available
                    return any(self.model in m for m in models)
        except Exception as e:
            logger.debug(f"Mistral availability check failed: {e}")
        return False

    @property
    def is_available(self) -> bool:
        """Cached availability check."""
        now = time.time()
        if now - self._last_check > self._check_interval:
            self._available = self.check_availability()
            self._last_check = now
        return self._available

    def generate(
        self,
        prompt: str,
        pad_state: Optional[Dict[str, float]] = None,
        clv_state: Optional[Dict[str, float]] = None,
        system_prompt: Optional[str] = None,
        include_context: bool = True
    ) -> MistralResponse:
        """
        Generate a response with cognitive modulation.

        Args:
            prompt: User prompt
            pad_state: PAD emotional state for param derivation
            clv_state: CLV state for additional constraints
            system_prompt: Override system prompt
            include_context: Include conversation context

        Returns:
            MistralResponse with text and metrics
        """
        start_time = time.time()

        # Derive generation parameters from cognitive state
        pad = pad_state or {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}
        params = GenerationParams.from_pad_state(pad, clv_state)

        # Build system prompt with emotional state
        system = system_prompt or self._build_system_prompt(pad)

        # Build messages
        messages = [{"role": "system", "content": system}]

        if include_context and self._context:
            messages.extend(self._context[-self._max_context:])

        messages.append({"role": "user", "content": prompt})

        # Make request
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
                "repeat_penalty": params.repeat_penalty
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))

            latency_ms = (time.time() - start_time) * 1000
            response_text = result.get("message", {}).get("content", "")
            tokens = result.get("eval_count", len(response_text.split()))

            # Simple confidence estimation based on response characteristics
            confidence = self._estimate_confidence(response_text, prompt)

            # Record metrics
            self.metrics.record_request(latency_ms, tokens, True, confidence)

            # Update context
            self._context.append({"role": "user", "content": prompt})
            self._context.append({"role": "assistant", "content": response_text})

            return MistralResponse(
                text=response_text,
                tokens_generated=tokens,
                latency_ms=latency_ms,
                generation_mode=params.mode,
                params_used=params,
                raw_response=result,
                confidence=confidence
            )

        except urllib.error.URLError as e:
            if "timed out" in str(e):
                self.metrics.record_timeout()
            else:
                self.metrics.record_request(0, 0, False)
            raise
        except Exception as e:
            self.metrics.record_request(0, 0, False)
            raise

    def _build_system_prompt(self, pad: Dict[str, float]) -> str:
        """Build system prompt with emotional state context."""
        valence = pad.get("valence", 0.0)
        arousal = pad.get("arousal", 0.5)
        dominance = pad.get("dominance", 0.5)

        # Derive mood description
        if valence > 0.3:
            mood = "positive, warm" if arousal < 0.6 else "excited, energetic"
        elif valence < -0.3:
            mood = "concerned, careful" if arousal < 0.6 else "stressed, alert"
        else:
            mood = "calm, neutral" if arousal < 0.6 else "attentive, focused"

        return f"""{self.identity_prompt}

Current state:
- Valence: {valence:+.2f} (pleasure/displeasure)
- Arousal: {arousal:.2f} (activation level)
- Dominance: {dominance:.2f} (confidence/control)
- Mood: {mood}

Respond in a way that reflects this emotional state naturally."""

    def _estimate_confidence(self, response: str, prompt: str) -> float:
        """
        Estimate confidence in the response.

        Simple heuristics for now - could be replaced with
        actual likelihood analysis or self-evaluation.
        """
        # Base confidence
        confidence = 0.8

        # Very short responses might indicate uncertainty
        if len(response.split()) < 5:
            confidence -= 0.1

        # Hedging language reduces confidence
        hedges = ["might", "maybe", "perhaps", "i think", "not sure", "possibly"]
        hedge_count = sum(1 for h in hedges if h in response.lower())
        confidence -= hedge_count * 0.05

        # Questions in response might indicate need for clarification
        if "?" in response and response.count("?") > 1:
            confidence -= 0.1

        # Confident language increases confidence
        confident = ["definitely", "certainly", "clearly", "absolutely"]
        confident_count = sum(1 for c in confident if c in response.lower())
        confidence += confident_count * 0.03

        return max(0.3, min(1.0, confidence))

    def get_clv_contribution(self) -> Dict[str, float]:
        """Get this backend's contribution to CLV metrics."""
        return self.metrics.get_clv_contribution()

    def clear_context(self):
        """Clear conversation context."""
        self._context = []

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "model": self.model,
            "available": self.is_available,
            "total_requests": self.metrics.total_requests,
            "failed_requests": self.metrics.failed_requests,
            "error_rate": f"{self.metrics.error_rate:.2%}",
            "p50_latency_ms": f"{self.metrics.p50_latency:.0f}",
            "p95_latency_ms": f"{self.metrics.p95_latency:.0f}",
            "total_tokens": self.metrics.total_tokens_generated,
            "avg_tokens_per_request": f"{self.metrics.avg_tokens_per_request:.0f}",
            "avg_confidence": f"{self.metrics.avg_confidence:.2f}",
            "context_length": len(self._context)
        }


def create_mistral_backend(
    host: str = "localhost",
    port: int = 11434,
    model: str = "mistral",
    **kwargs
) -> MistralBackend:
    """
    Create a Mistral backend.

    Args:
        host: Ollama host
        port: Ollama port
        model: Model name
        **kwargs: Additional options

    Returns:
        MistralBackend instance
    """
    return MistralBackend(host=host, port=port, model=model, **kwargs)


# Backend registration for model selector
BACKEND_CONFIG = {
    "mistral_7b": {
        "type": "llm",
        "family": "mistral",
        "size_b": 7,
        "max_context": 8192,
        "mode": "general",
        "prefers": ["reasoning", "code_assist", "planning", "conversation"],
        "latency_class": "medium",  # ~500ms-2s typical
        "create": lambda: create_mistral_backend()
    }
}


__all__ = [
    "GenerationMode",
    "GenerationParams",
    "BackendMetrics",
    "MistralResponse",
    "MistralBackend",
    "create_mistral_backend",
    "BACKEND_CONFIG",
]
