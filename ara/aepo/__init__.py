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
from typing import Optional, Dict, Any, List, Callable, Tuple
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


# =============================================================================
# PHASE 5.1: STRUCTURAL LEARNING ENVIRONMENT
# =============================================================================
# AEPOEnv exposes SNN parameters (r, k, v_th) as action space
# Uses Pareto metrics (HV, IGD) and EPR-CV as state/reward signals

import time
import logging
import math
import random

logger = logging.getLogger("ara.aepo")

# Try numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class ParetoMetrics:
    """
    Pareto multi-objective metrics for structural learning.

    HV (Hypervolume): Volume dominated by Pareto front - higher is better
    IGD (Inverted Generational Distance): Distance to true Pareto front - lower is better
    EPR_CV (Edge-of-chaos Criticality Variance): Stability metric
    """
    hypervolume: float = 0.0           # HV indicator
    igd: float = float('inf')          # IGD indicator
    epr_cv: float = 0.0                # Criticality variance
    accuracy: float = 0.0              # Task accuracy
    energy: float = 0.0                # Energy consumption (spike rate)
    latency_us: float = 0.0            # Control loop latency
    stability: float = 0.0             # Topology stability
    density: float = 0.0               # Network density

    def weighted_reward(
        self,
        hv_weight: float = 1.0,
        igd_weight: float = 0.5,
        epr_weight: float = 0.3,
    ) -> float:
        """Compute weighted reward signal."""
        # HV is to be maximized, IGD and EPR_CV to be minimized
        reward = hv_weight * self.hypervolume
        reward -= igd_weight * min(1.0, self.igd)  # Clamp IGD contribution
        reward -= epr_weight * self.epr_cv
        return reward

    def to_vector(self) -> List[float]:
        """Convert to feature vector for neural network input."""
        return [
            self.hypervolume,
            self.igd,
            self.epr_cv,
            self.accuracy,
            self.energy,
            self.latency_us / 1000.0,  # Normalize to ms
            self.stability,
            self.density,
        ]


@dataclass
class SNNParameters:
    """
    SNN parameters exposed as action space.

    r: Refractory period vector (per neuron)
    k: Coupling strength matrix
    v_th: Threshold voltage vector
    tau: Membrane time constant
    """
    num_neurons: int = 128
    r: List[float] = field(default_factory=list)      # Refractory periods [0.001, 0.1]
    k: List[List[float]] = field(default_factory=list)  # Coupling matrix [-1, 1]
    v_th: List[float] = field(default_factory=list)   # Thresholds [0.5, 2.0]
    tau: float = 0.02                                  # Membrane time constant

    def __post_init__(self):
        """Initialize with defaults if empty."""
        if not self.r:
            self.r = [0.02] * self.num_neurons
        if not self.k:
            # Sparse random initialization
            self.k = [[0.0] * self.num_neurons for _ in range(self.num_neurons)]
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if random.random() < 0.1:  # 10% connectivity
                        self.k[i][j] = random.gauss(0, 0.3)
        if not self.v_th:
            self.v_th = [1.0] * self.num_neurons

    def density(self) -> float:
        """Compute network connection density."""
        total = self.num_neurons * self.num_neurons
        connections = sum(1 for row in self.k for v in row if abs(v) > 0.01)
        return connections / total

    def apply_mask(self, mask: List[List[float]]):
        """Apply topology mask from TP-RL."""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                self.k[i][j] *= mask[i][j]


@dataclass
class AEPOState:
    """
    State observation for AEPO environment.

    Combines network metrics with Pareto indicators.
    """
    pareto_metrics: ParetoMetrics
    snn_params: SNNParameters
    episode: int = 0
    step: int = 0

    def to_vector(self) -> List[float]:
        """Flatten state to feature vector."""
        features = self.pareto_metrics.to_vector()
        features.extend([
            self.snn_params.tau,
            self.snn_params.density(),
            float(self.episode) / 1000.0,  # Normalized episode
            float(self.step) / 500.0,       # Normalized step
        ])
        return features


@dataclass
class AEPOAction:
    """
    Action space for structural learning.

    Actions modify SNN parameters and topology.
    """
    # Parameter deltas
    delta_tau: float = 0.0           # Change to membrane time constant
    delta_v_th: List[float] = field(default_factory=list)  # Threshold changes

    # Topology actions
    add_connection: Optional[tuple] = None   # (i, j, weight) to add
    remove_connection: Optional[tuple] = None  # (i, j) to remove

    # Mask modulation (from TP-RL)
    mask_update: Optional[List[List[float]]] = None

    @classmethod
    def from_vector(cls, vec: List[float], num_neurons: int = 128) -> "AEPOAction":
        """Parse action from neural network output."""
        action = cls()

        if len(vec) >= 1:
            action.delta_tau = vec[0] * 0.01  # Scale to reasonable range

        if len(vec) >= num_neurons + 1:
            action.delta_v_th = [v * 0.1 for v in vec[1:num_neurons+1]]

        return action


class AEPOEnv:
    """
    AEPO Environment for Autonomous Structural Learning.

    Exposes SNN parameters as action space and reports
    Pareto metrics as state/reward signals.

    This environment implements the structural learning loop:
    1. Agent observes current SNN state + Pareto metrics
    2. Agent outputs parameter/topology modifications
    3. Environment applies modifications and simulates SNN
    4. Environment computes new Pareto metrics as reward

    Compatible with standard RL APIs (Gym-like interface).
    """

    def __init__(
        self,
        num_neurons: int = 128,
        max_steps: int = 500,
        target_accuracy: float = 0.95,
        target_energy: float = 0.1,
        target_latency_us: float = 200.0,
        use_tprl: bool = True,
    ):
        self.num_neurons = num_neurons
        self.max_steps = max_steps
        self.target_accuracy = target_accuracy
        self.target_energy = target_energy
        self.target_latency_us = target_latency_us
        self.use_tprl = use_tprl

        # State
        self.params: Optional[SNNParameters] = None
        self.metrics: Optional[ParetoMetrics] = None
        self.episode = 0
        self.step = 0
        self.done = False

        # History for Pareto front computation
        self.pareto_front: List[ParetoMetrics] = []
        self.episode_history: List[Dict[str, float]] = []

        # TP-RL integration
        self._tprl_trainer = None
        if use_tprl:
            self._init_tprl()

        logger.info(f"AEPOEnv initialized: neurons={num_neurons}, max_steps={max_steps}")

    def _init_tprl(self):
        """Initialize TP-RL trainer for mask optimization."""
        try:
            from ara.tprl import TPRLTrainer, TrainingConfig
            config = TrainingConfig(
                num_episodes=1,  # Single episode per step
                max_steps_per_episode=1,
            )
            self._tprl_trainer = TPRLTrainer(config)
            logger.info("TP-RL trainer integrated with AEPOEnv")
        except ImportError:
            logger.warning("TP-RL not available for AEPOEnv")
            self._tprl_trainer = None

    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return 12  # ParetoMetrics(8) + tau + density + episode + step

    @property
    def action_dim(self) -> int:
        """Dimension of action vector."""
        return 1 + self.num_neurons  # delta_tau + delta_v_th

    def reset(self) -> AEPOState:
        """Reset environment for new episode."""
        self.episode += 1
        self.step = 0
        self.done = False

        # Initialize SNN parameters
        self.params = SNNParameters(num_neurons=self.num_neurons)

        # Initial metrics (simulated)
        self.metrics = self._simulate_snn()

        self.episode_history = []

        return AEPOState(
            pareto_metrics=self.metrics,
            snn_params=self.params,
            episode=self.episode,
            step=self.step,
        )

    def step_env(self, action: AEPOAction) -> tuple:
        """
        Execute action and return (state, reward, done, info).

        Args:
            action: AEPOAction with parameter modifications

        Returns:
            (state, reward, done, info) tuple
        """
        self.step += 1

        # Apply action to SNN parameters
        self._apply_action(action)

        # Run TP-RL step if available
        if self._tprl_trainer and action.mask_update:
            self.params.apply_mask(action.mask_update)

        # Simulate SNN with new parameters
        self.metrics = self._simulate_snn()

        # Compute reward
        reward = self._compute_reward()

        # Update Pareto front
        self._update_pareto_front()

        # Check termination
        self.done = self.step >= self.max_steps
        if self.metrics.accuracy >= self.target_accuracy:
            if self.metrics.energy <= self.target_energy:
                self.done = True  # Target achieved

        # Build state
        state = AEPOState(
            pareto_metrics=self.metrics,
            snn_params=self.params,
            episode=self.episode,
            step=self.step,
        )

        info = {
            "pareto_front_size": len(self.pareto_front),
            "hypervolume": self._compute_hypervolume(),
            "target_achieved": self.metrics.accuracy >= self.target_accuracy,
        }

        return state, reward, self.done, info

    def _apply_action(self, action: AEPOAction):
        """Apply action to SNN parameters."""
        # Update tau
        self.params.tau = max(0.001, min(0.1, self.params.tau + action.delta_tau))

        # Update thresholds
        if action.delta_v_th:
            for i, delta in enumerate(action.delta_v_th[:self.num_neurons]):
                self.params.v_th[i] = max(0.5, min(2.0, self.params.v_th[i] + delta))

        # Add connection
        if action.add_connection:
            i, j, w = action.add_connection
            if 0 <= i < self.num_neurons and 0 <= j < self.num_neurons:
                self.params.k[i][j] = max(-1.0, min(1.0, w))

        # Remove connection
        if action.remove_connection:
            i, j = action.remove_connection
            if 0 <= i < self.num_neurons and 0 <= j < self.num_neurons:
                self.params.k[i][j] = 0.0

    def _simulate_snn(self) -> ParetoMetrics:
        """
        Simulate SNN and compute metrics.

        In production, this calls the actual SNN simulator.
        For now, uses analytical approximations.
        """
        # Compute network properties
        density = self.params.density()
        mean_coupling = sum(
            abs(self.params.k[i][j])
            for i in range(self.num_neurons)
            for j in range(self.num_neurons)
        ) / (self.num_neurons ** 2)

        mean_threshold = sum(self.params.v_th) / self.num_neurons

        # Approximate metrics based on parameters
        # Higher coupling + lower threshold = higher activity = more energy
        base_accuracy = 0.5 + 0.3 * math.tanh(mean_coupling * 3)
        base_energy = 0.05 + 0.15 * density + 0.1 * (1.5 - mean_threshold)

        # Add some stochasticity
        accuracy = base_accuracy + random.gauss(0, 0.02)
        energy = base_energy + random.gauss(0, 0.01)

        # Stability inversely related to coupling variance
        coupling_var = sum(
            (self.params.k[i][j] - mean_coupling) ** 2
            for i in range(self.num_neurons)
            for j in range(self.num_neurons)
        ) / (self.num_neurons ** 2)
        stability = 1.0 - min(1.0, coupling_var * 5)

        # EPR-CV: criticality variance (want it low for edge-of-chaos)
        epr_cv = abs(density - 0.1) + abs(mean_coupling - 0.3)

        # Latency (simulated)
        latency_us = 10.0 + self.num_neurons * 0.1 + random.gauss(0, 2)

        return ParetoMetrics(
            hypervolume=0.0,  # Computed later
            igd=0.0,
            epr_cv=epr_cv,
            accuracy=max(0.0, min(1.0, accuracy)),
            energy=max(0.0, min(1.0, energy)),
            latency_us=max(1.0, latency_us),
            stability=max(0.0, min(1.0, stability)),
            density=density,
        )

    def _compute_reward(self) -> float:
        """Compute reward signal from current metrics."""
        # Multi-objective reward
        accuracy_reward = self.metrics.accuracy
        energy_penalty = self.metrics.energy
        latency_penalty = min(1.0, self.metrics.latency_us / self.target_latency_us)
        stability_bonus = self.metrics.stability

        # Criticality bonus (want EPR-CV near 0)
        criticality_bonus = max(0, 1.0 - self.metrics.epr_cv)

        reward = (
            1.0 * accuracy_reward
            - 0.3 * energy_penalty
            - 0.2 * latency_penalty
            + 0.2 * stability_bonus
            + 0.3 * criticality_bonus
        )

        # Bonus for achieving targets
        if self.metrics.accuracy >= self.target_accuracy:
            reward += 0.5
        if self.metrics.energy <= self.target_energy:
            reward += 0.3

        return reward

    def _update_pareto_front(self):
        """Update Pareto front with current metrics."""
        # Check if current solution is non-dominated
        dominated = False
        for existing in self.pareto_front:
            # Check if existing dominates current
            if (existing.accuracy >= self.metrics.accuracy and
                existing.energy <= self.metrics.energy and
                existing.latency_us <= self.metrics.latency_us and
                (existing.accuracy > self.metrics.accuracy or
                 existing.energy < self.metrics.energy or
                 existing.latency_us < self.metrics.latency_us)):
                dominated = True
                break

        if not dominated:
            # Remove solutions dominated by current
            self.pareto_front = [
                p for p in self.pareto_front
                if not (self.metrics.accuracy >= p.accuracy and
                       self.metrics.energy <= p.energy and
                       self.metrics.latency_us <= p.latency_us and
                       (self.metrics.accuracy > p.accuracy or
                        self.metrics.energy < p.energy or
                        self.metrics.latency_us < p.latency_us))
            ]
            self.pareto_front.append(self.metrics)

    def _compute_hypervolume(self) -> float:
        """Compute hypervolume indicator of Pareto front."""
        if not self.pareto_front:
            return 0.0

        # Reference point (worst case)
        ref_accuracy = 0.0
        ref_energy = 1.0
        ref_latency = 1000.0

        # Simple 3D hypervolume approximation
        hv = 0.0
        for p in self.pareto_front:
            # Volume contribution
            vol = (
                (p.accuracy - ref_accuracy) *
                (ref_energy - p.energy) *
                (ref_latency - p.latency_us) / 1000.0
            )
            hv += max(0, vol)

        # Update metrics
        self.metrics.hypervolume = hv
        return hv


class AEPOAgent:
    """
    Policy gradient agent for AEPO structural learning.

    Uses a simple MLP policy to map states to actions.
    Trained with REINFORCE + baseline.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Simple policy network (weights)
        # In production, use PyTorch/JAX
        self.w1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(state_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[random.gauss(0, 0.1) for _ in range(action_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * action_dim

        # Training history
        self.episode_rewards: List[float] = []
        self.baseline = 0.0

        logger.info(f"AEPOAgent initialized: state_dim={state_dim}, action_dim={action_dim}")

    def _forward(self, state: List[float]) -> List[float]:
        """Forward pass through policy network."""
        # Hidden layer
        hidden = []
        for j in range(self.hidden_dim):
            h = self.b1[j]
            for i in range(min(len(state), self.state_dim)):
                h += state[i] * self.w1[i][j]
            hidden.append(math.tanh(h))

        # Output layer
        output = []
        for j in range(self.action_dim):
            o = self.b2[j]
            for i in range(self.hidden_dim):
                o += hidden[i] * self.w2[i][j]
            output.append(math.tanh(o))  # Bounded output

        return output

    def select_action(self, state: AEPOState, explore: bool = True) -> AEPOAction:
        """Select action given state."""
        state_vec = state.to_vector()
        action_vec = self._forward(state_vec)

        # Add exploration noise
        if explore:
            action_vec = [a + random.gauss(0, 0.1) for a in action_vec]

        return AEPOAction.from_vector(action_vec, state.snn_params.num_neurons)

    def update(self, rewards: List[float], states: List[AEPOState], actions: List[AEPOAction]):
        """Update policy using REINFORCE."""
        if not rewards:
            return

        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)

        # Update baseline (moving average)
        self.baseline = 0.9 * self.baseline + 0.1 * total_reward

        # Compute advantage
        advantage = total_reward - self.baseline

        # Simple gradient update (approximate)
        # In production, use autograd
        for t, (state, action) in enumerate(zip(states, actions)):
            state_vec = state.to_vector()

            # Compute return-to-go
            G = sum(rewards[t:])

            # Update weights with gradient (simplified)
            scale = self.learning_rate * (G - self.baseline) * 0.01

            for i in range(min(len(state_vec), self.state_dim)):
                for j in range(self.hidden_dim):
                    self.w1[i][j] += scale * state_vec[i]

        logger.debug(f"Policy updated: total_reward={total_reward:.3f}, advantage={advantage:.3f}")


def train_aepo(
    num_episodes: int = 500,
    max_steps: int = 500,
    num_neurons: int = 128,
    checkpoint_dir: str = "./checkpoints/aepo",
) -> Dict[str, Any]:
    """
    Train AEPO agent for structural learning.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_neurons: Number of SNN neurons
        checkpoint_dir: Directory for checkpoints

    Returns:
        Training results dict
    """
    # Create environment and agent
    env = AEPOEnv(num_neurons=num_neurons, max_steps=max_steps)
    agent = AEPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    results = {
        "episodes": [],
        "best_accuracy": 0.0,
        "best_hypervolume": 0.0,
        "pareto_front_size": 0,
    }

    logger.info(f"Starting AEPO training: {num_episodes} episodes")
    start_time = time.time()

    for ep in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        episode_states = []
        episode_actions = []

        while not env.done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step_env(action)

            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action)

            state = next_state

        # Update policy
        agent.update(episode_rewards, episode_states, episode_actions)

        # Track best results
        if env.metrics.accuracy > results["best_accuracy"]:
            results["best_accuracy"] = env.metrics.accuracy
        if env.metrics.hypervolume > results["best_hypervolume"]:
            results["best_hypervolume"] = env.metrics.hypervolume

        results["pareto_front_size"] = len(env.pareto_front)

        # Logging
        if (ep + 1) % 50 == 0:
            logger.info(
                f"Episode {ep + 1}/{num_episodes}: "
                f"acc={env.metrics.accuracy:.3f}, "
                f"energy={env.metrics.energy:.3f}, "
                f"HV={env.metrics.hypervolume:.4f}, "
                f"front={len(env.pareto_front)}"
            )

        results["episodes"].append({
            "episode": ep + 1,
            "accuracy": env.metrics.accuracy,
            "energy": env.metrics.energy,
            "hypervolume": env.metrics.hypervolume,
            "reward": sum(episode_rewards),
        })

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["episodes_per_second"] = num_episodes / elapsed

    logger.info(
        f"AEPO training complete: {num_episodes} episodes in {elapsed:.1f}s, "
        f"best_acc={results['best_accuracy']:.3f}, best_HV={results['best_hypervolume']:.4f}"
    )

    return results


# =============================================================================
# PHASE 5.4: AUTONOMOUS MODEL SELECTOR
# =============================================================================
# Auto-selects between Claude, local TF-A-N, and other backends based on:
# - Task complexity and type
# - Latency requirements
# - Cost constraints
# - Current PAD state (emotional context)
# - System load and availability


@dataclass
class ModelSelectionCriteria:
    """Criteria for autonomous model selection."""
    # Task characteristics
    task_type: TaskType = TaskType.CONVERSATION
    complexity_score: float = 0.5  # [0, 1]
    context_length: int = 0        # Token count

    # Performance requirements
    max_latency_ms: float = 5000.0
    min_accuracy: float = 0.8
    max_cost_per_1k: float = 0.02

    # Emotional context (from PAD)
    valence: float = 0.0   # Affects risk tolerance
    arousal: float = 0.5   # Affects speed preference

    # System state
    prefer_local: bool = False
    require_tools: bool = False
    require_multimodal: bool = False


@dataclass
class ModelSelectionResult:
    """Result of autonomous model selection."""
    selected_model: Backend
    confidence: float
    reasoning: str
    alternatives: List[Tuple[Backend, float]]  # (model, score)
    estimated_latency_ms: float
    estimated_cost: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [(m.value, s) for m, s in self.alternatives],
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost": self.estimated_cost,
            "timestamp": self.timestamp,
        }


class AutonomousModelSelector:
    """
    Autonomous model selector using multi-criteria decision making.

    Combines:
    - Task-based scoring (complexity, type, requirements)
    - Performance-based scoring (latency, accuracy, cost)
    - Emotional context (PAD state affects risk tolerance)
    - Learning from feedback (success/failure history)

    This is the "brain" that decides when to use:
    - Claude: Complex reasoning, code generation, research
    - ChatGPT: Creative tasks, general conversation
    - Gemini: Multimodal, search-integrated tasks
    - Local (Ollama/TF-A-N): Privacy-sensitive, low-latency
    """

    def __init__(
        self,
        available_backends: Optional[List[Backend]] = None,
        learning_rate: float = 0.1,
    ):
        """
        Initialize autonomous model selector.

        Args:
            available_backends: List of available backend models
            learning_rate: Rate for updating preferences from feedback
        """
        if available_backends:
            self.available_backends = set(available_backends)
        else:
            self.available_backends = {Backend.CLAUDE, Backend.CHATGPT, Backend.OLLAMA, Backend.TFAN}

        self.learning_rate = learning_rate

        # Backend capabilities
        self.capabilities = DEFAULT_CAPABILITIES.copy()

        # Learned preferences (updated from feedback)
        self.backend_scores: Dict[Backend, float] = {
            Backend.CLAUDE: 0.9,
            Backend.CHATGPT: 0.85,
            Backend.GEMINI: 0.8,
            Backend.OLLAMA: 0.6,
            Backend.TFAN: 0.5,
            Backend.TOOL: 0.7,
        }

        # Task-specific preferences
        self.task_preferences: Dict[TaskType, Dict[Backend, float]] = {
            TaskType.CODE_WRITE: {Backend.CLAUDE: 0.95, Backend.CHATGPT: 0.8},
            TaskType.CODE_REFACTOR: {Backend.CLAUDE: 0.95, Backend.CHATGPT: 0.75},
            TaskType.CODE_DEBUG: {Backend.CLAUDE: 0.9, Backend.CHATGPT: 0.8},
            TaskType.RESEARCH: {Backend.CLAUDE: 0.9, Backend.GEMINI: 0.85},
            TaskType.CREATIVE: {Backend.CHATGPT: 0.9, Backend.CLAUDE: 0.8},
            TaskType.CONVERSATION: {Backend.OLLAMA: 0.8, Backend.CHATGPT: 0.85},
            TaskType.MULTIMODAL: {Backend.GEMINI: 0.95, Backend.CLAUDE: 0.8},
            TaskType.SEARCH: {Backend.GEMINI: 0.9},
        }

        # Selection history for learning
        self.selection_history: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []

        logger.info(f"AutonomousModelSelector initialized with {len(self.available_backends)} backends")

    def select(self, criteria: ModelSelectionCriteria) -> ModelSelectionResult:
        """
        Select the best model for given criteria.

        Args:
            criteria: Selection criteria

        Returns:
            ModelSelectionResult with selected model and reasoning
        """
        scores: Dict[Backend, float] = {}
        reasons: Dict[Backend, List[str]] = {}

        for backend in self.available_backends:
            caps = self.capabilities.get(backend)
            if not caps or not caps.available:
                continue

            score = 0.0
            reason_parts = []

            # 1. Base score from learned preferences
            base_score = self.backend_scores.get(backend, 0.5)
            score += base_score * 0.3
            reason_parts.append(f"base={base_score:.2f}")

            # 2. Task-specific preference
            task_prefs = self.task_preferences.get(criteria.task_type, {})
            task_score = task_prefs.get(backend, 0.5)
            score += task_score * 0.25
            if task_score > 0.7:
                reason_parts.append(f"task_fit={task_score:.2f}")

            # 3. Capability requirements check
            if criteria.require_tools and not caps.supports_tools:
                score *= 0.5
                reason_parts.append("no_tools")
            if criteria.require_multimodal and not caps.supports_multimodal:
                score *= 0.3
                reason_parts.append("no_multimodal")

            # 4. Latency requirement
            if caps.latency_ms <= criteria.max_latency_ms:
                latency_score = 1.0 - (caps.latency_ms / criteria.max_latency_ms) * 0.5
                score += latency_score * 0.15
            else:
                score *= 0.5
                reason_parts.append("too_slow")

            # 5. Cost constraint
            if caps.cost_per_1k <= criteria.max_cost_per_1k:
                cost_score = 1.0 - (caps.cost_per_1k / criteria.max_cost_per_1k) * 0.5
                score += cost_score * 0.1
            else:
                score *= 0.7
                reason_parts.append("too_expensive")

            # 6. Complexity alignment
            # Complex tasks → powerful models; simple tasks → fast/cheap models
            if criteria.complexity_score > 0.7:
                if backend in [Backend.CLAUDE, Backend.CHATGPT]:
                    score += 0.1
                    reason_parts.append("complex_fit")
            elif criteria.complexity_score < 0.3:
                if backend in [Backend.OLLAMA, Backend.TFAN]:
                    score += 0.1
                    reason_parts.append("simple_fit")

            # 7. Local preference
            if criteria.prefer_local and backend == Backend.OLLAMA:
                score += 0.15
                reason_parts.append("local_preferred")

            # 8. Emotional context modulation
            # Low valence → more conservative (proven models)
            if criteria.valence < -0.3:
                if backend == Backend.CLAUDE:
                    score += 0.05
                    reason_parts.append("risk_averse")
            # High arousal → faster models
            if criteria.arousal > 0.7:
                if caps.latency_ms < 1000:
                    score += 0.05
                    reason_parts.append("speed_preferred")

            scores[backend] = min(1.0, max(0.0, score))
            reasons[backend] = reason_parts

        # Select best
        if not scores:
            # Fallback
            return ModelSelectionResult(
                selected_model=Backend.CLAUDE,
                confidence=0.3,
                reasoning="No suitable backend found, using default",
                alternatives=[],
                estimated_latency_ms=1500,
                estimated_cost=0.015,
            )

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_backend, best_score = sorted_scores[0]
        best_caps = self.capabilities.get(best_backend, DEFAULT_CAPABILITIES[Backend.CLAUDE])

        # Build reasoning
        reasoning = f"Selected {best_backend.value}: " + ", ".join(reasons.get(best_backend, []))

        result = ModelSelectionResult(
            selected_model=best_backend,
            confidence=best_score,
            reasoning=reasoning,
            alternatives=sorted_scores[1:4],  # Top 3 alternatives
            estimated_latency_ms=best_caps.latency_ms,
            estimated_cost=best_caps.cost_per_1k,
        )

        # Record selection
        self._record_selection(criteria, result)

        return result

    def provide_feedback(
        self,
        selection_timestamp: str,
        success: bool,
        actual_latency_ms: float,
        quality_score: float = 0.5,
    ):
        """
        Provide feedback on a selection to improve future decisions.

        Args:
            selection_timestamp: Timestamp of the selection
            success: Whether the task was completed successfully
            actual_latency_ms: Actual observed latency
            quality_score: Quality rating [0, 1]
        """
        # Find corresponding selection
        selection = None
        for s in reversed(self.selection_history):
            if s["timestamp"] == selection_timestamp:
                selection = s
                break

        if not selection:
            logger.warning(f"Selection not found for feedback: {selection_timestamp}")
            return

        backend = Backend(selection["selected_model"])

        # Update backend score based on feedback
        current_score = self.backend_scores.get(backend, 0.5)

        # Compute reward
        reward = 0.0
        if success:
            reward += 0.3
        if quality_score > 0.7:
            reward += 0.2
        if actual_latency_ms < selection["estimated_latency_ms"] * 1.2:
            reward += 0.1

        # Penalize failures
        if not success:
            reward -= 0.3
        if quality_score < 0.3:
            reward -= 0.2

        # Update score with learning rate
        new_score = current_score + self.learning_rate * reward
        self.backend_scores[backend] = max(0.1, min(1.0, new_score))

        # Record feedback
        self.feedback_history.append({
            "selection_timestamp": selection_timestamp,
            "backend": backend.value,
            "success": success,
            "quality_score": quality_score,
            "actual_latency_ms": actual_latency_ms,
            "reward": reward,
            "new_score": self.backend_scores[backend],
        })

        logger.info(f"Feedback processed: {backend.value} score {current_score:.3f} → {self.backend_scores[backend]:.3f}")

    def _record_selection(self, criteria: ModelSelectionCriteria, result: ModelSelectionResult):
        """Record selection for future learning."""
        record = {
            "timestamp": result.timestamp,
            "task_type": criteria.task_type.value,
            "complexity": criteria.complexity_score,
            "selected_model": result.selected_model.value,
            "confidence": result.confidence,
            "estimated_latency_ms": result.estimated_latency_ms,
        }
        self.selection_history.append(record)

        # Keep last 1000 selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

    def get_status(self) -> Dict[str, Any]:
        """Get selector status."""
        return {
            "available_backends": [b.value for b in self.available_backends],
            "backend_scores": {b.value: s for b, s in self.backend_scores.items()},
            "selection_count": len(self.selection_history),
            "feedback_count": len(self.feedback_history),
        }


# Global selector
_model_selector: Optional[AutonomousModelSelector] = None


def get_model_selector() -> AutonomousModelSelector:
    """Get or create the global model selector."""
    global _model_selector
    if _model_selector is None:
        _model_selector = AutonomousModelSelector()
    return _model_selector


def select_model(
    task_type: str = "conversation",
    complexity: float = 0.5,
    max_latency_ms: float = 5000.0,
    prefer_local: bool = False,
    valence: float = 0.0,
    arousal: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function for model selection.

    Args:
        task_type: Type of task
        complexity: Complexity score [0, 1]
        max_latency_ms: Maximum acceptable latency
        prefer_local: Prefer local models
        valence: PAD valence for emotional context
        arousal: PAD arousal for emotional context

    Returns:
        Selection result dict

    Example:
        result = select_model(
            task_type="code_write",
            complexity=0.8,
            max_latency_ms=10000,
        )
        print(f"Use: {result['selected_model']}")
    """
    selector = get_model_selector()

    try:
        tt = TaskType(task_type.lower())
    except ValueError:
        tt = TaskType.CONVERSATION

    criteria = ModelSelectionCriteria(
        task_type=tt,
        complexity_score=complexity,
        max_latency_ms=max_latency_ms,
        prefer_local=prefer_local,
        valence=valence,
        arousal=arousal,
    )

    result = selector.select(criteria)
    return result.to_dict()


__all__ = [
    # Routing exports (original)
    "Backend",
    "TaskType",
    "BackendCapabilities",
    "RoutingDecision",
    "AEPORouter",
    "route_request",
    # Structural learning exports (Phase 5.1)
    "ParetoMetrics",
    "SNNParameters",
    "AEPOState",
    "AEPOAction",
    "AEPOEnv",
    "AEPOAgent",
    "train_aepo",
    # Autonomous model selection (Phase 5.4)
    "ModelSelectionCriteria",
    "ModelSelectionResult",
    "AutonomousModelSelector",
    "get_model_selector",
    "select_model",
]
