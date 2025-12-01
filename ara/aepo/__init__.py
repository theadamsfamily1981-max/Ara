"""
ARA AEPO Module (Adaptive Entropy Policy Optimizer)

Multi-agent routing and tool-use policy.
Decides when/how to use Claude, ChatGPT, Gemini, local models,
or specialized tools based on request characteristics.

API Contract:
    POST /aepo/route
    - Input: text, context, session_id, available_backends
    - Output: selected_backend, prompt_template, confidence

AEPO controls:
    - Which LLM backend to use (Claude, ChatGPT, Gemini, Ollama)
    - Whether to use tools (code execution, search, etc.)
    - How to structure the prompt for the selected backend
    - Fallback strategies if primary choice fails
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum
import json

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class Backend(str, Enum):
    """Available LLM backends."""
    CLAUDE = "claude"           # Anthropic Claude (heavy R&D, refactoring)
    CHATGPT = "chatgpt"         # OpenAI GPT-4 (design, reasoning)
    GEMINI = "gemini"           # Google Gemini (multimodal, search)
    OLLAMA = "ollama"           # Local models (privacy, offline)
    TFAN = "tfan"               # Local TF-A-N model (specialized)
    TOOL = "tool"               # Direct tool use (no LLM)


class TaskType(str, Enum):
    """Task type classification."""
    CODE_WRITE = "code_write"           # Write new code
    CODE_REFACTOR = "code_refactor"     # Refactor existing code
    CODE_DEBUG = "code_debug"           # Debug/fix issues
    CODE_REVIEW = "code_review"         # Review code
    RESEARCH = "research"               # Research/analysis
    CREATIVE = "creative"               # Creative writing
    CONVERSATION = "conversation"       # General chat
    SEARCH = "search"                   # Information lookup
    CALCULATION = "calculation"         # Math/computation
    SUMMARIZE = "summarize"             # Summarization
    TRANSLATE = "translate"             # Translation
    MULTIMODAL = "multimodal"           # Image/audio/video


@dataclass
class BackendCapabilities:
    """Capabilities and characteristics of a backend."""
    name: Backend
    available: bool = True
    supports_code: bool = True
    supports_multimodal: bool = False
    supports_tools: bool = False
    max_context: int = 100000
    latency_ms: float = 1000.0
    cost_per_1k: float = 0.01
    strengths: List[TaskType] = field(default_factory=list)
    api_key_required: bool = True


# Default backend capabilities
DEFAULT_CAPABILITIES = {
    Backend.CLAUDE: BackendCapabilities(
        name=Backend.CLAUDE,
        supports_code=True,
        supports_multimodal=True,
        supports_tools=True,
        max_context=200000,
        latency_ms=1500,
        cost_per_1k=0.015,
        strengths=[TaskType.CODE_WRITE, TaskType.CODE_REFACTOR, TaskType.CODE_REVIEW, TaskType.RESEARCH],
    ),
    Backend.CHATGPT: BackendCapabilities(
        name=Backend.CHATGPT,
        supports_code=True,
        supports_multimodal=True,
        supports_tools=True,
        max_context=128000,
        latency_ms=1200,
        cost_per_1k=0.01,
        strengths=[TaskType.CREATIVE, TaskType.CONVERSATION, TaskType.RESEARCH],
    ),
    Backend.GEMINI: BackendCapabilities(
        name=Backend.GEMINI,
        supports_code=True,
        supports_multimodal=True,
        supports_tools=True,
        max_context=1000000,
        latency_ms=1000,
        cost_per_1k=0.005,
        strengths=[TaskType.MULTIMODAL, TaskType.SEARCH, TaskType.SUMMARIZE],
    ),
    Backend.OLLAMA: BackendCapabilities(
        name=Backend.OLLAMA,
        supports_code=True,
        supports_multimodal=False,
        supports_tools=False,
        max_context=32000,
        latency_ms=500,
        cost_per_1k=0.0,  # Free (local)
        strengths=[TaskType.CONVERSATION],
        api_key_required=False,
    ),
    Backend.TFAN: BackendCapabilities(
        name=Backend.TFAN,
        supports_code=False,
        supports_multimodal=True,
        supports_tools=False,
        max_context=32768,
        latency_ms=200,
        cost_per_1k=0.0,
        strengths=[],
        api_key_required=False,
    ),
}


@dataclass
class RoutingDecision:
    """
    AEPO routing decision.

    Contains the selected backend, confidence, and
    prompt modifications for that backend.
    """
    # Primary selection
    selected_backend: Backend
    confidence: float = 0.5

    # Task analysis
    task_type: TaskType = TaskType.CONVERSATION
    complexity: float = 0.5  # [0, 1]

    # Prompt modifications
    system_prompt_additions: List[str] = field(default_factory=list)
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None

    # Fallback
    fallback_backend: Optional[Backend] = None
    fallback_reason: Optional[str] = None

    # Tools
    tools_to_enable: List[str] = field(default_factory=list)

    # Metadata
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_backend": self.selected_backend.value,
            "confidence": self.confidence,
            "task_type": self.task_type.value,
            "complexity": self.complexity,
            "system_prompt_additions": self.system_prompt_additions,
            "temperature_override": self.temperature_override,
            "max_tokens_override": self.max_tokens_override,
            "fallback_backend": self.fallback_backend.value if self.fallback_backend else None,
            "fallback_reason": self.fallback_reason,
            "tools_to_enable": self.tools_to_enable,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AEPORouter:
    """
    AEPO (Adaptive Entropy Policy Optimizer) Router.

    Decides which backend to use for a given request based on:
    - Task type and complexity
    - Backend capabilities and availability
    - Cost/latency constraints
    - User preferences

    In production, this wraps TF-A-N's AEPO policy network.
    For now, uses heuristic routing.
    """

    def __init__(
        self,
        available_backends: Optional[List[Backend]] = None,
        default_backend: Backend = Backend.CLAUDE,
        prefer_local: bool = False,
        cost_sensitive: bool = False,
        use_tfan_policy: bool = False,
    ):
        self.default_backend = default_backend
        self.prefer_local = prefer_local
        self.cost_sensitive = cost_sensitive
        self.use_tfan_policy = use_tfan_policy

        # Set available backends
        if available_backends:
            self.available_backends = set(available_backends)
        else:
            self.available_backends = {Backend.CLAUDE, Backend.CHATGPT, Backend.OLLAMA}

        self.capabilities = DEFAULT_CAPABILITIES.copy()

        # Try to load TFAN AEPO policy
        self._tfan_available = False
        if use_tfan_policy:
            self._init_tfan_policy()

    def _init_tfan_policy(self):
        """Initialize TF-A-N AEPO policy network."""
        try:
            from tfan.agent.aepo import AEPOPolicy
            self._tfan_available = True
        except ImportError:
            self._tfan_available = False

    def route(
        self,
        text: str,
        context: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a request to the appropriate backend.

        Args:
            text: User input text
            context: Conversation history
            session_id: Session identifier
            constraints: Optional constraints (max_cost, max_latency, etc.)

        Returns:
            RoutingDecision with selected backend and configuration
        """
        # Classify task
        task_type = self._classify_task(text)
        complexity = self._estimate_complexity(text, context)

        # Get available backends that support this task
        candidates = self._filter_candidates(task_type, constraints)

        if not candidates:
            # Fallback to default
            return RoutingDecision(
                selected_backend=self.default_backend,
                confidence=0.3,
                task_type=task_type,
                complexity=complexity,
                reasoning="No suitable backend found, using default",
            )

        # Score candidates
        scored = self._score_candidates(candidates, task_type, complexity, constraints)

        # Select best
        best_backend, best_score = max(scored.items(), key=lambda x: x[1])

        # Determine fallback
        fallback = None
        if len(scored) > 1:
            remaining = {k: v for k, v in scored.items() if k != best_backend}
            fallback, _ = max(remaining.items(), key=lambda x: x[1])

        # Build decision
        decision = RoutingDecision(
            selected_backend=best_backend,
            confidence=best_score,
            task_type=task_type,
            complexity=complexity,
            fallback_backend=fallback,
            reasoning=self._explain_choice(best_backend, task_type, complexity),
        )

        # Add task-specific modifications
        self._add_task_modifications(decision, task_type, complexity)

        return decision

    def _classify_task(self, text: str) -> TaskType:
        """Classify the task type from input text."""
        text_lower = text.lower()

        # Code indicators
        if any(m in text_lower for m in ["write code", "implement", "create a function", "build"]):
            return TaskType.CODE_WRITE
        if any(m in text_lower for m in ["refactor", "improve", "clean up", "optimize"]):
            return TaskType.CODE_REFACTOR
        if any(m in text_lower for m in ["debug", "fix", "error", "bug", "not working"]):
            return TaskType.CODE_DEBUG
        if any(m in text_lower for m in ["review", "check", "look at this code"]):
            return TaskType.CODE_REVIEW

        # Research
        if any(m in text_lower for m in ["research", "find out", "what is", "explain", "how does"]):
            return TaskType.RESEARCH

        # Creative
        if any(m in text_lower for m in ["write a story", "create", "imagine", "brainstorm"]):
            return TaskType.CREATIVE

        # Search
        if any(m in text_lower for m in ["search", "find", "look up", "latest"]):
            return TaskType.SEARCH

        # Summarize
        if any(m in text_lower for m in ["summarize", "tldr", "brief", "key points"]):
            return TaskType.SUMMARIZE

        # Multimodal
        if any(m in text_lower for m in ["image", "picture", "photo", "video", "audio"]):
            return TaskType.MULTIMODAL

        # Default to conversation
        return TaskType.CONVERSATION

    def _estimate_complexity(
        self,
        text: str,
        context: Optional[List[str]],
    ) -> float:
        """Estimate task complexity (0-1)."""
        complexity = 0.3  # Base

        # Length adds complexity
        if len(text) > 500:
            complexity += 0.2
        if len(text) > 1000:
            complexity += 0.1

        # Context adds complexity
        if context and len(context) > 5:
            complexity += 0.1

        # Code blocks add complexity
        if "```" in text:
            complexity += 0.2

        # Questions add complexity
        if text.count("?") > 2:
            complexity += 0.1

        return min(1.0, complexity)

    def _filter_candidates(
        self,
        task_type: TaskType,
        constraints: Optional[Dict[str, Any]],
    ) -> List[Backend]:
        """Filter backends that can handle this task."""
        candidates = []

        for backend in self.available_backends:
            caps = self.capabilities.get(backend)
            if not caps or not caps.available:
                continue

            # Check task-specific requirements
            if task_type in [TaskType.CODE_WRITE, TaskType.CODE_REFACTOR, TaskType.CODE_DEBUG]:
                if not caps.supports_code:
                    continue

            if task_type == TaskType.MULTIMODAL:
                if not caps.supports_multimodal:
                    continue

            # Check constraints
            if constraints:
                if "max_latency_ms" in constraints:
                    if caps.latency_ms > constraints["max_latency_ms"]:
                        continue
                if "max_cost_per_1k" in constraints:
                    if caps.cost_per_1k > constraints["max_cost_per_1k"]:
                        continue

            candidates.append(backend)

        return candidates

    def _score_candidates(
        self,
        candidates: List[Backend],
        task_type: TaskType,
        complexity: float,
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[Backend, float]:
        """Score each candidate backend."""
        scores = {}

        for backend in candidates:
            caps = self.capabilities.get(backend)
            score = 0.5  # Base score

            # Bonus for task strength
            if task_type in caps.strengths:
                score += 0.3

            # Complexity bonus for powerful backends
            if complexity > 0.6 and backend in [Backend.CLAUDE, Backend.CHATGPT]:
                score += 0.2

            # Cost preference
            if self.cost_sensitive:
                score -= caps.cost_per_1k * 10  # Penalize cost

            # Local preference
            if self.prefer_local and backend == Backend.OLLAMA:
                score += 0.2

            # Default backend bonus
            if backend == self.default_backend:
                score += 0.1

            scores[backend] = min(1.0, max(0.0, score))

        return scores

    def _explain_choice(
        self,
        backend: Backend,
        task_type: TaskType,
        complexity: float,
    ) -> str:
        """Generate explanation for routing choice."""
        caps = self.capabilities.get(backend)

        reasons = []
        if task_type in caps.strengths:
            reasons.append(f"strong at {task_type.value}")
        if complexity > 0.6:
            reasons.append(f"handles complex tasks well")
        if self.prefer_local and backend == Backend.OLLAMA:
            reasons.append("local execution preferred")
        if self.cost_sensitive and caps.cost_per_1k == 0:
            reasons.append("no API cost")

        if not reasons:
            reasons.append("default selection")

        return f"Selected {backend.value}: {', '.join(reasons)}"

    def _add_task_modifications(
        self,
        decision: RoutingDecision,
        task_type: TaskType,
        complexity: float,
    ):
        """Add task-specific prompt/parameter modifications."""
        # Code tasks: lower temperature, enable tools
        if task_type in [TaskType.CODE_WRITE, TaskType.CODE_REFACTOR, TaskType.CODE_DEBUG]:
            decision.temperature_override = 0.3
            decision.tools_to_enable.append("code_execution")
            decision.system_prompt_additions.append(
                "Focus on clean, well-documented code. Include error handling."
            )

        # Creative tasks: higher temperature
        if task_type == TaskType.CREATIVE:
            decision.temperature_override = 0.9
            decision.system_prompt_additions.append(
                "Be creative and exploratory. Offer multiple perspectives."
            )

        # Research: enable search
        if task_type in [TaskType.RESEARCH, TaskType.SEARCH]:
            decision.tools_to_enable.append("web_search")
            decision.system_prompt_additions.append(
                "Cite sources when possible. Be thorough but concise."
            )

        # Complex tasks: more tokens
        if complexity > 0.7:
            decision.max_tokens_override = 4000


# Convenience function
def route_request(
    text: str,
    context: Optional[List[str]] = None,
    available_backends: Optional[List[str]] = None,
    prefer_local: bool = False,
) -> Dict[str, Any]:
    """
    Route a request to the appropriate backend.

    Args:
        text: User input
        context: Conversation history
        available_backends: List of available backend names
        prefer_local: Prefer local execution

    Returns:
        Routing decision dict
    """
    backends = None
    if available_backends:
        backends = [Backend(b) for b in available_backends if b in Backend._value2member_map_]

    router = AEPORouter(
        available_backends=backends,
        prefer_local=prefer_local,
    )

    decision = router.route(text, context)
    return decision.to_dict()


__all__ = [
    "Backend",
    "TaskType",
    "BackendCapabilities",
    "RoutingDecision",
    "AEPORouter",
    "route_request",
]
