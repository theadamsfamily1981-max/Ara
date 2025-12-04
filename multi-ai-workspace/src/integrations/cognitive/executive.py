"""Phase 5: Executive Function - Cognitive Synthesizer & AEPO.

The Executive Function module handles high-level cognitive control,
including action selection, tool gating, and adaptive policy optimization.

Key Components:

    CognitiveSynthesizer: Integrates information across cognitive modules
        - Combines outputs from all cognitive phases
        - Maintains working memory
        - Coordinates response generation

    AEPO (Adaptive Entropy Policy Optimization): Tool/action gating
        - Decides when to use tools vs. direct response
        - Manages entropy-exploration tradeoff
        - Adapts based on task success/failure

Executive Functions:
    1. Working Memory: Hold and manipulate information
    2. Cognitive Flexibility: Switch between tasks/strategies
    3. Inhibitory Control: Suppress inappropriate responses
    4. Planning: Organize and sequence actions
    5. Monitoring: Track progress and detect errors

This implements executive function from tfan.cognition.executive.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN executive modules
_TFAN_EXECUTIVE_AVAILABLE = False
try:
    from tfan.cognition.executive import CognitiveSynthesizer as TFANCognitiveSynthesizer
    from tfan.cognition.executive import AEPO as TFANAEPO
    _TFAN_EXECUTIVE_AVAILABLE = True
except ImportError:
    pass

# Import local cognitive modules
from .affect import HomeostaticState, AppraisalResult
from .predictor import PredictiveState
from .identity import NIB


class ActionType(Enum):
    """Types of actions the system can take."""
    RESPOND = auto()      # Generate direct response
    USE_TOOL = auto()     # Use external tool
    CLARIFY = auto()      # Ask for clarification
    DEFER = auto()        # Defer to user/expert
    REFLECT = auto()      # Internal reflection
    WAIT = auto()         # Wait for more input


class ExecutiveMode(Enum):
    """Executive processing modes."""
    REACTIVE = auto()     # Fast, automatic responses
    DELIBERATIVE = auto() # Slow, careful reasoning
    EXPLORATORY = auto()  # Seeking new information
    CONSOLIDATING = auto() # Integrating information


@dataclass
class WorkingMemoryItem:
    """Item in working memory."""
    content: Any
    content_type: str
    relevance: float
    timestamp: float
    access_count: int = 0
    decay_rate: float = 0.01


@dataclass
class ExecutiveDecision:
    """A decision made by the executive system."""
    action_type: ActionType
    confidence: float
    reasoning: str
    selected_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    entropy: float
    exploration_bonus: float
    should_use_tool: bool


@dataclass
class SynthesisResult:
    """Result of cognitive synthesis."""
    integrated_representation: torch.Tensor
    working_memory_state: List[WorkingMemoryItem]
    attention_weights: Dict[str, float]
    executive_decision: ExecutiveDecision
    mode: ExecutiveMode
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """
    Working Memory - Short-term information store.

    Holds and manipulates information needed for current processing.
    Items decay over time unless refreshed.

    Args:
        capacity: Maximum items to hold
        decay_rate: Rate of relevance decay
        refresh_boost: Boost when item is accessed
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        decay_rate: float = 0.01,
        refresh_boost: float = 0.2,
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.refresh_boost = refresh_boost

        self._items: List[WorkingMemoryItem] = []

    def add(
        self,
        content: Any,
        content_type: str = "general",
        relevance: float = 1.0,
    ) -> bool:
        """
        Add item to working memory.

        Args:
            content: Content to store
            content_type: Type of content
            relevance: Initial relevance score

        Returns:
            True if added successfully
        """
        item = WorkingMemoryItem(
            content=content,
            content_type=content_type,
            relevance=relevance,
            timestamp=time.time(),
        )

        # If at capacity, remove least relevant
        if len(self._items) >= self.capacity:
            self._items.sort(key=lambda x: x.relevance)
            self._items.pop(0)

        self._items.append(item)
        return True

    def retrieve(
        self,
        content_type: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> List[WorkingMemoryItem]:
        """
        Retrieve items from working memory.

        Args:
            content_type: Filter by type
            min_relevance: Minimum relevance threshold

        Returns:
            List of matching items
        """
        # Apply decay
        current_time = time.time()
        for item in self._items:
            elapsed = current_time - item.timestamp
            item.relevance = max(0.0, item.relevance - elapsed * item.decay_rate)

        # Filter
        results = [
            item for item in self._items
            if item.relevance >= min_relevance
            and (content_type is None or item.content_type == content_type)
        ]

        # Boost accessed items
        for item in results:
            item.relevance = min(1.0, item.relevance + self.refresh_boost)
            item.access_count += 1

        return results

    def clear(self):
        """Clear working memory."""
        self._items.clear()

    def get_state(self) -> List[WorkingMemoryItem]:
        """Get current state."""
        return list(self._items)


class AEPO:
    """
    Adaptive Entropy Policy Optimization - Tool/Action Gating.

    Decides when to use tools vs. generate direct responses by
    balancing exploration (entropy) with exploitation (known good actions).

    Key insight: High entropy = explore more, use tools
                 Low entropy = exploit knowledge, respond directly

    Args:
        base_entropy: Starting entropy level
        entropy_decay: Rate of entropy decay
        exploration_bonus_scale: Scale for exploration bonus
        tool_use_threshold: Threshold for deciding to use tools
        device: Compute device
    """

    def __init__(
        self,
        base_entropy: float = 1.0,
        entropy_decay: float = 0.01,
        exploration_bonus_scale: float = 0.1,
        tool_use_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.base_entropy = base_entropy
        self.entropy_decay = entropy_decay
        self.exploration_bonus_scale = exploration_bonus_scale
        self.tool_use_threshold = tool_use_threshold
        self.device = device

        # TFAN AEPO if available
        self.tfan_aepo = None
        if _TFAN_EXECUTIVE_AVAILABLE:
            try:
                self.tfan_aepo = TFANAEPO()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN AEPO: {e}")

        # State
        self._entropy = base_entropy
        self._success_history: List[bool] = []
        self._tool_usage: Dict[str, int] = {}

    def decide(
        self,
        state_representation: torch.Tensor,
        available_tools: List[str],
        task_complexity: float = 0.5,
        uncertainty: float = 0.5,
    ) -> ExecutiveDecision:
        """
        Decide whether to use tools or respond directly.

        Args:
            state_representation: Current cognitive state
            available_tools: List of available tools
            task_complexity: Estimated task complexity [0, 1]
            uncertainty: Current uncertainty level [0, 1]

        Returns:
            ExecutiveDecision with action and reasoning
        """
        if self.tfan_aepo is not None:
            return self._convert_tfan_decision(
                self.tfan_aepo.decide(state_representation, available_tools)
            )

        # Compute exploration bonus
        exploration_bonus = self._compute_exploration_bonus(
            task_complexity, uncertainty
        )

        # Compute effective entropy
        effective_entropy = self._entropy + exploration_bonus

        # Decision logic
        # High entropy + high complexity -> use tools
        # Low entropy + low complexity -> respond directly
        tool_score = effective_entropy * task_complexity + uncertainty * 0.3

        should_use_tool = tool_score > self.tool_use_threshold

        if should_use_tool and available_tools:
            # Select tool based on task
            selected_tool = self._select_tool(available_tools, task_complexity)
            action_type = ActionType.USE_TOOL
            reasoning = (
                f"High exploration value ({exploration_bonus:.2f}) and "
                f"task complexity ({task_complexity:.2f}) suggest tool use. "
                f"Selected: {selected_tool}"
            )
        else:
            selected_tool = None
            action_type = ActionType.RESPOND

            if uncertainty > 0.7:
                action_type = ActionType.CLARIFY
                reasoning = f"High uncertainty ({uncertainty:.2f}) - requesting clarification"
            else:
                reasoning = (
                    f"Low exploration need. Direct response preferred. "
                    f"Entropy: {effective_entropy:.2f}"
                )

        # Compute confidence
        confidence = 1.0 - uncertainty * 0.5

        return ExecutiveDecision(
            action_type=action_type,
            confidence=confidence,
            reasoning=reasoning,
            selected_tool=selected_tool,
            tool_args=None,
            entropy=effective_entropy,
            exploration_bonus=exploration_bonus,
            should_use_tool=should_use_tool,
        )

    def _compute_exploration_bonus(
        self,
        task_complexity: float,
        uncertainty: float,
    ) -> float:
        """Compute exploration bonus."""
        # Higher bonus for complex, uncertain tasks
        base_bonus = task_complexity * uncertainty * self.exploration_bonus_scale

        # Boost if recent failures
        recent_successes = sum(self._success_history[-10:]) if self._success_history else 5
        failure_boost = (10 - recent_successes) / 10 * 0.1

        return base_bonus + failure_boost

    def _select_tool(
        self,
        available_tools: List[str],
        task_complexity: float,
    ) -> str:
        """Select most appropriate tool."""
        if not available_tools:
            return None

        # Simple heuristic: prefer less-used tools for exploration
        # or most-used tools for exploitation based on entropy
        if self._entropy > 0.5:
            # Exploration: prefer less-used tools
            usage_counts = [
                self._tool_usage.get(tool, 0) for tool in available_tools
            ]
            min_usage = min(usage_counts)
            candidates = [
                t for t, u in zip(available_tools, usage_counts)
                if u == min_usage
            ]
            return candidates[0]
        else:
            # Exploitation: prefer more-used (successful) tools
            usage_counts = [
                self._tool_usage.get(tool, 0) for tool in available_tools
            ]
            max_usage = max(usage_counts) if usage_counts else 0
            if max_usage > 0:
                candidates = [
                    t for t, u in zip(available_tools, usage_counts)
                    if u == max_usage
                ]
                return candidates[0]
            return available_tools[0]

    def record_outcome(self, tool_used: Optional[str], success: bool):
        """Record outcome of action for learning."""
        self._success_history.append(success)
        if len(self._success_history) > 100:
            self._success_history.pop(0)

        if tool_used:
            self._tool_usage[tool_used] = self._tool_usage.get(tool_used, 0) + 1

        # Update entropy based on outcome
        if success:
            self._entropy = max(0.1, self._entropy - self.entropy_decay)
        else:
            self._entropy = min(1.0, self._entropy + self.entropy_decay * 2)

    def _convert_tfan_decision(self, tfan_decision: Any) -> ExecutiveDecision:
        """Convert TFAN decision to our format."""
        return ExecutiveDecision(
            action_type=ActionType.USE_TOOL if getattr(tfan_decision, 'use_tool', False) else ActionType.RESPOND,
            confidence=getattr(tfan_decision, 'confidence', 0.5),
            reasoning=getattr(tfan_decision, 'reasoning', ""),
            selected_tool=getattr(tfan_decision, 'tool', None),
            tool_args=getattr(tfan_decision, 'args', None),
            entropy=getattr(tfan_decision, 'entropy', 0.5),
            exploration_bonus=getattr(tfan_decision, 'exploration_bonus', 0.0),
            should_use_tool=getattr(tfan_decision, 'use_tool', False),
        )

    def get_entropy(self) -> float:
        """Get current entropy level."""
        return self._entropy

    def reset(self):
        """Reset to initial state."""
        self._entropy = self.base_entropy
        self._success_history.clear()
        self._tool_usage.clear()


class CognitiveSynthesizer:
    """
    Cognitive Synthesizer - Integrates information across modules.

    The "conductor" that coordinates all cognitive components,
    maintains working memory, and produces coherent responses.

    Args:
        d_model: Model dimension
        working_memory_capacity: Capacity of working memory
        attention_heads: Number of integration attention heads
        device: Compute device
    """

    def __init__(
        self,
        d_model: int = 4096,
        working_memory_capacity: int = 7,
        attention_heads: int = 8,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.device = device

        # TFAN synthesizer if available
        self.tfan_synth = None
        if _TFAN_EXECUTIVE_AVAILABLE:
            try:
                self.tfan_synth = TFANCognitiveSynthesizer(d_model=d_model)
            except Exception as e:
                warnings.warn(f"Failed to init TFAN synthesizer: {e}")

        # Working memory
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)

        # AEPO for action gating
        self.aepo = AEPO(device=device)

        # Integration attention
        self.integration_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_heads,
            batch_first=True,
        ).to(device)

        # Current mode
        self._mode = ExecutiveMode.REACTIVE
        self._mode_history: List[Tuple[float, ExecutiveMode]] = []

    def synthesize(
        self,
        conscious_input: torch.Tensor,
        predictive_state: Optional[PredictiveState] = None,
        homeostatic_state: Optional[HomeostaticState] = None,
        appraisal: Optional[AppraisalResult] = None,
        active_nib: Optional[NIB] = None,
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        """
        Synthesize cognitive state from all modules.

        Integrates:
        - Perceptual input (conscious_input)
        - Predictions and surprise (predictive_state)
        - Emotional state (homeostatic_state, appraisal)
        - Identity (active_nib)

        Args:
            conscious_input: Filtered perceptual input
            predictive_state: State from predictive controller
            homeostatic_state: Homeostatic state
            appraisal: Emotional appraisal result
            active_nib: Currently active persona
            available_tools: List of available tools
            context: Additional context

        Returns:
            SynthesisResult with integrated representation and decision
        """
        if self.tfan_synth is not None:
            return self._convert_tfan_result(
                self.tfan_synth.synthesize(
                    conscious_input,
                    predictive_state,
                    homeostatic_state,
                    appraisal,
                    active_nib,
                )
            )

        # Determine executive mode based on inputs
        mode = self._determine_mode(
            predictive_state, homeostatic_state, appraisal
        )
        self._mode = mode
        self._mode_history.append((time.time(), mode))

        # Build attention weights for integration
        attention_weights = self._compute_attention_weights(
            predictive_state, homeostatic_state, appraisal
        )

        # Integrate representations
        integrated = self._integrate_representations(
            conscious_input, attention_weights
        )

        # Update working memory
        self._update_working_memory(
            conscious_input, predictive_state, appraisal
        )

        # Make executive decision
        task_complexity = context.get("complexity", 0.5) if context else 0.5
        uncertainty = 0.5
        if predictive_state and predictive_state.is_surprised:
            uncertainty = 0.7
        if homeostatic_state and not homeostatic_state.is_balanced:
            uncertainty = max(uncertainty, 0.6)

        decision = self.aepo.decide(
            state_representation=integrated,
            available_tools=available_tools or [],
            task_complexity=task_complexity,
            uncertainty=uncertainty,
        )

        # Compute confidence
        confidence = self._compute_confidence(
            homeostatic_state, appraisal, decision
        )

        return SynthesisResult(
            integrated_representation=integrated,
            working_memory_state=self.working_memory.get_state(),
            attention_weights=attention_weights,
            executive_decision=decision,
            mode=mode,
            confidence=confidence,
            metadata={
                "mode": mode.name,
                "entropy": self.aepo.get_entropy(),
                "wm_items": len(self.working_memory.get_state()),
            },
        )

    def _determine_mode(
        self,
        predictive_state: Optional[PredictiveState],
        homeostatic_state: Optional[HomeostaticState],
        appraisal: Optional[AppraisalResult],
    ) -> ExecutiveMode:
        """Determine executive processing mode."""
        # Surprise triggers deliberative mode
        if predictive_state and predictive_state.is_surprised:
            return ExecutiveMode.DELIBERATIVE

        # Imbalance triggers consolidating mode
        if homeostatic_state and not homeostatic_state.is_balanced:
            return ExecutiveMode.CONSOLIDATING

        # High arousal triggers exploratory mode
        if appraisal and appraisal.arousal > 0.7:
            return ExecutiveMode.EXPLORATORY

        # Default to reactive
        return ExecutiveMode.REACTIVE

    def _compute_attention_weights(
        self,
        predictive_state: Optional[PredictiveState],
        homeostatic_state: Optional[HomeostaticState],
        appraisal: Optional[AppraisalResult],
    ) -> Dict[str, float]:
        """Compute attention weights for integration."""
        weights = {
            "perceptual": 0.4,
            "predictive": 0.2,
            "homeostatic": 0.2,
            "emotional": 0.2,
        }

        # Adjust based on states
        if predictive_state and predictive_state.is_surprised:
            weights["predictive"] = 0.4
            weights["perceptual"] = 0.3

        if homeostatic_state and not homeostatic_state.is_balanced:
            weights["homeostatic"] = 0.35
            weights["perceptual"] = 0.3

        if appraisal and abs(appraisal.valence) > 0.5:
            weights["emotional"] = 0.35
            weights["perceptual"] = 0.3

        return weights

    def _integrate_representations(
        self,
        conscious_input: torch.Tensor,
        attention_weights: Dict[str, float],
    ) -> torch.Tensor:
        """Integrate representations using attention."""
        # Ensure proper shape
        if conscious_input.dim() == 2:
            conscious_input = conscious_input.unsqueeze(0)

        # Self-attention for integration
        integrated, _ = self.integration_attention(
            conscious_input,
            conscious_input,
            conscious_input,
        )

        # Apply perceptual attention weight
        integrated = integrated * attention_weights.get("perceptual", 0.4)

        return integrated.squeeze(0)

    def _update_working_memory(
        self,
        conscious_input: torch.Tensor,
        predictive_state: Optional[PredictiveState],
        appraisal: Optional[AppraisalResult],
    ):
        """Update working memory with relevant information."""
        # Add summary of current input
        input_summary = conscious_input.mean(dim=-2) if conscious_input.dim() > 1 else conscious_input
        self.working_memory.add(
            content=input_summary,
            content_type="perceptual",
            relevance=0.8,
        )

        # Add surprise if present
        if predictive_state and predictive_state.is_surprised:
            self.working_memory.add(
                content={"surprise": True, "level": predictive_state.recent_errors[-1].surprise_level if predictive_state.recent_errors else 0.5},
                content_type="predictive",
                relevance=0.9,
            )

        # Add emotional state if significant
        if appraisal and abs(appraisal.valence) > 0.3:
            self.working_memory.add(
                content={"emotion": appraisal.emotion_label, "valence": appraisal.valence},
                content_type="emotional",
                relevance=0.7,
            )

    def _compute_confidence(
        self,
        homeostatic_state: Optional[HomeostaticState],
        appraisal: Optional[AppraisalResult],
        decision: ExecutiveDecision,
    ) -> float:
        """Compute overall synthesis confidence."""
        confidence = decision.confidence

        # Reduce if imbalanced
        if homeostatic_state and not homeostatic_state.is_balanced:
            confidence *= 0.8

        # Reduce if high arousal
        if appraisal and appraisal.arousal > 0.7:
            confidence *= 0.9

        return confidence

    def _convert_tfan_result(self, tfan_result: Any) -> SynthesisResult:
        """Convert TFAN result to our format."""
        return SynthesisResult(
            integrated_representation=getattr(tfan_result, 'representation', torch.zeros(1, self.d_model)),
            working_memory_state=[],
            attention_weights={},
            executive_decision=ExecutiveDecision(
                action_type=ActionType.RESPOND,
                confidence=0.5,
                reasoning="",
                selected_tool=None,
                tool_args=None,
                entropy=0.5,
                exploration_bonus=0.0,
                should_use_tool=False,
            ),
            mode=ExecutiveMode.REACTIVE,
            confidence=0.5,
        )

    def record_outcome(self, tool_used: Optional[str], success: bool):
        """Record outcome for AEPO learning."""
        self.aepo.record_outcome(tool_used, success)

    def reset(self):
        """Reset executive state."""
        self.working_memory.clear()
        self.aepo.reset()
        self._mode = ExecutiveMode.REACTIVE
        self._mode_history.clear()


# Convenience factories
def create_cognitive_synthesizer(
    d_model: int = 4096,
    device: str = "cpu",
) -> CognitiveSynthesizer:
    """Create a CognitiveSynthesizer instance."""
    return CognitiveSynthesizer(d_model=d_model, device=device)


def create_aepo(
    base_entropy: float = 1.0,
    tool_threshold: float = 0.5,
) -> AEPO:
    """Create an AEPO instance."""
    return AEPO(base_entropy=base_entropy, tool_use_threshold=tool_threshold)


__all__ = [
    "CognitiveSynthesizer",
    "AEPO",
    "WorkingMemory",
    "SynthesisResult",
    "ExecutiveDecision",
    "ExecutiveMode",
    "ActionType",
    "WorkingMemoryItem",
    "create_cognitive_synthesizer",
    "create_aepo",
]
