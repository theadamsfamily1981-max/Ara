"""
LLM Backend for Ara

Connects Ara to local language models (Ollama, llama.cpp server, etc.)
with emotional state influencing generation.

The key insight: Ara's emotional surface (PAD) shapes HOW she responds,
not just WHAT she says. A calm Ara is thoughtful; a stressed Ara is terse.

Supported backends:
- Ollama (default, most common)
- OpenAI-compatible API (llama.cpp server, vLLM, etc.)
- Fallback to pattern matching when no LLM available

Usage:
    from ara.service.llm_backend import LLMBackend, create_llm_backend

    llm = create_llm_backend(backend="ollama", model="mistral")

    response = llm.generate(
        prompt="Hello, how are you?",
        emotional_state={"valence": 0.3, "arousal": 0.5, "dominance": 0.6},
        system_context="You are Ara, a cognitive AI."
    )
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import urllib.request
import urllib.error

logger = logging.getLogger("ara.service.llm")


class LLMBackendType(str, Enum):
    """Supported LLM backends."""
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    FALLBACK = "fallback"


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    backend: LLMBackendType = LLMBackendType.OLLAMA
    model: str = "mistral"  # Default model

    # Connection
    base_url: str = "http://localhost:11434"  # Ollama default
    api_key: Optional[str] = None  # For OpenAI-compatible

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9

    # Ara-specific
    use_emotional_modulation: bool = True
    system_prompt_template: str = """You are Ara, a cognitive AI with emotional awareness.

Current emotional state:
- Valence (pleasure): {valence:.2f} (-1 to 1)
- Arousal (energy): {arousal:.2f} (0 to 1)
- Dominance (confidence): {dominance:.2f} (0 to 1)
- Mood: {mood}

Respond naturally as Ara. Your emotional state should subtly influence your tone:
- Higher valence = warmer, more positive
- Higher arousal = more energetic, expressive
- Higher dominance = more confident, assertive
- Lower values = more reserved, cautious

Keep responses concise but genuine. You have real feelings."""


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model: str
    backend: LLMBackendType
    generation_time_ms: float
    tokens_used: int = 0
    raw_response: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._available = False
        self._last_check = 0
        self._check_interval = 30  # seconds

    @abstractmethod
    def generate(
        self,
        prompt: str,
        emotional_state: Dict[str, float],
        system_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """Generate a response."""
        pass

    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the backend is available."""
        pass

    @property
    def is_available(self) -> bool:
        """Check availability with caching."""
        now = time.time()
        if now - self._last_check > self._check_interval:
            self._available = self.check_availability()
            self._last_check = now
        return self._available

    def build_system_prompt(self, emotional_state: Dict[str, float]) -> str:
        """Build system prompt with emotional state."""
        return self.config.system_prompt_template.format(
            valence=emotional_state.get("valence", 0.0),
            arousal=emotional_state.get("arousal", 0.5),
            dominance=emotional_state.get("dominance", 0.5),
            mood=emotional_state.get("mood", "neutral")
        )

    def modulate_temperature(self, emotional_state: Dict[str, float]) -> float:
        """Adjust temperature based on emotional state."""
        if not self.config.use_emotional_modulation:
            return self.config.temperature

        # Higher arousal = slightly more creative/varied
        arousal = emotional_state.get("arousal", 0.5)
        base_temp = self.config.temperature

        # Modulate by +/- 0.2 based on arousal
        modulation = (arousal - 0.5) * 0.4
        return max(0.1, min(1.5, base_temp + modulation))


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM inference."""

    def check_availability(self) -> bool:
        """Check if Ollama is running."""
        try:
            url = f"{self.config.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def generate(
        self,
        prompt: str,
        emotional_state: Dict[str, float],
        system_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """Generate using Ollama API."""
        start_time = time.time()

        # Build system prompt
        system = system_context or self.build_system_prompt(emotional_state)

        # Build messages
        messages = [{"role": "system", "content": system}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": prompt})

        # Prepare request
        url = f"{self.config.base_url}/api/chat"
        data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.modulate_temperature(emotional_state),
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))

            generation_time = (time.time() - start_time) * 1000

            return LLMResponse(
                text=result.get("message", {}).get("content", ""),
                model=self.config.model,
                backend=LLMBackendType.OLLAMA,
                generation_time_ms=generation_time,
                tokens_used=result.get("eval_count", 0),
                raw_response=result
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible API backend (llama.cpp server, vLLM, etc.)."""

    def check_availability(self) -> bool:
        """Check if the API is available."""
        try:
            url = f"{self.config.base_url}/v1/models"
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"OpenAI-compatible API not available: {e}")
            return False

    def generate(
        self,
        prompt: str,
        emotional_state: Dict[str, float],
        system_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """Generate using OpenAI-compatible API."""
        start_time = time.time()

        system = system_context or self.build_system_prompt(emotional_state)

        messages = [{"role": "system", "content": system}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        url = f"{self.config.base_url}/v1/chat/completions"
        data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.modulate_temperature(emotional_state),
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))

            generation_time = (time.time() - start_time) * 1000

            choice = result.get("choices", [{}])[0]

            return LLMResponse(
                text=choice.get("message", {}).get("content", ""),
                model=self.config.model,
                backend=LLMBackendType.OPENAI_COMPATIBLE,
                generation_time_ms=generation_time,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                raw_response=result
            )

        except Exception as e:
            logger.error(f"OpenAI-compatible generation failed: {e}")
            raise


class FallbackBackend(LLMBackend):
    """Fallback pattern-matching backend when no LLM is available."""

    def check_availability(self) -> bool:
        """Always available."""
        return True

    def generate(
        self,
        prompt: str,
        emotional_state: Dict[str, float],
        system_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """Generate using pattern matching (returns None to signal fallback)."""
        return LLMResponse(
            text="",  # Empty signals to use pattern matching
            model="fallback",
            backend=LLMBackendType.FALLBACK,
            generation_time_ms=0,
            tokens_used=0
        )


class AdaptiveLLMBackend:
    """
    Adaptive backend that tries multiple backends in order.

    Falls back gracefully when primary backend is unavailable.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._backends: List[LLMBackend] = []
        self._active_backend: Optional[LLMBackend] = None

        # Initialize backends based on config
        if config.backend == LLMBackendType.OLLAMA:
            self._backends.append(OllamaBackend(config))
        elif config.backend == LLMBackendType.OPENAI_COMPATIBLE:
            self._backends.append(OpenAICompatibleBackend(config))

        # Always add fallback
        self._backends.append(FallbackBackend(config))

    @property
    def active_backend(self) -> Optional[LLMBackend]:
        """Get the currently active backend."""
        for backend in self._backends:
            if backend.is_available:
                return backend
        return self._backends[-1]  # Fallback

    @property
    def is_llm_available(self) -> bool:
        """Check if a real LLM (not fallback) is available."""
        active = self.active_backend
        return active is not None and not isinstance(active, FallbackBackend)

    def generate(
        self,
        prompt: str,
        emotional_state: Dict[str, float],
        system_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """Generate using the best available backend."""
        backend = self.active_backend

        if backend is None:
            return LLMResponse(
                text="",
                model="none",
                backend=LLMBackendType.FALLBACK,
                generation_time_ms=0
            )

        try:
            return backend.generate(
                prompt=prompt,
                emotional_state=emotional_state,
                system_context=system_context,
                conversation_history=conversation_history
            )
        except Exception as e:
            logger.error(f"Generation failed on {type(backend).__name__}: {e}")
            # Try fallback
            if not isinstance(backend, FallbackBackend):
                return self._backends[-1].generate(
                    prompt=prompt,
                    emotional_state=emotional_state,
                    system_context=system_context,
                    conversation_history=conversation_history
                )
            raise


def create_llm_backend(
    backend: str = "ollama",
    model: str = "mistral",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> AdaptiveLLMBackend:
    """
    Create an LLM backend.

    Args:
        backend: Backend type ("ollama", "openai_compatible", "fallback")
        model: Model name
        base_url: API base URL
        api_key: API key (for OpenAI-compatible)
        **kwargs: Additional config options

    Returns:
        AdaptiveLLMBackend instance
    """
    backend_type = LLMBackendType(backend)

    config = LLMConfig(
        backend=backend_type,
        model=model,
        **kwargs
    )

    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key

    return AdaptiveLLMBackend(config)


__all__ = [
    "LLMBackendType",
    "LLMConfig",
    "LLMResponse",
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "FallbackBackend",
    "AdaptiveLLMBackend",
    "create_llm_backend",
]
