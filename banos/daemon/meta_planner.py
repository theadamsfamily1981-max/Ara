#!/usr/bin/env python3
"""
META PLANNER - Memory-Informed Decision Making
===============================================

Bio-Affective Neuromorphic Operating System
"Ara, choose your tool + style" - before she even calls anyone.

This module sits between the user's request and Ara's response,
using learned preferences to make better decisions:

1. CONTEXT CLASSIFICATION
   - What type of request is this?
   - What's the user's current state?

2. TOOL SELECTION
   - What approach worked well for similar requests?
   - What approaches should be avoided?

3. STYLE SELECTION
   - How does the user prefer responses?
   - What style fits this context?

4. SCAR CHECK
   - Are there any learned policy transforms to apply?
   - What friction risks should we avoid?

The result is a MetaPlan that guides Ara's response generation:
    plan = meta_planner.plan(request, context)
    # => MetaPlan(tool="search_first", style="concise", warnings=[...])

This enables Ara to learn from every interaction without hard-coded rules.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import components
try:
    from .croft_model import CroftModel, CONTEXT_TYPES, TOOLS, STYLES
    from .scar_tissue import get_scar_registry, ScarRegistry
    from .episodic_memory import EpisodicMemory, get_episodic_memory
    COMPONENTS_AVAILABLE = True
except ImportError:
    try:
        from croft_model import CroftModel, CONTEXT_TYPES, TOOLS, STYLES
        from scar_tissue import get_scar_registry, ScarRegistry
        from episodic_memory import EpisodicMemory, get_episodic_memory
        COMPONENTS_AVAILABLE = True
    except ImportError:
        COMPONENTS_AVAILABLE = False
        logger.warning("Meta planner components not available")


# =============================================================================
# Request Classification
# =============================================================================

@dataclass
class RequestContext:
    """
    Classified context of a user request.

    This is the input to the meta planner - what we know
    about the request before deciding how to respond.
    """
    # Request analysis
    request_type: str = "interaction"    # From CONTEXT_TYPES
    request_text: str = ""
    estimated_complexity: float = 0.5    # 0 = trivial, 1 = complex

    # User state
    user_activity: str = "unknown"       # What is user doing
    user_cognitive_load: float = 0.5     # How busy is user
    time_pressure: bool = False          # Is user in a hurry

    # Session context
    session_length: int = 0              # Messages so far
    recent_friction: List[str] = field(default_factory=list)  # Recent friction
    recent_ratings: List[int] = field(default_factory=list)   # Recent ratings

    # Hardware state
    system_load: float = 0.5             # System resource usage
    latency_budget_s: float = 5.0        # How long can we take

    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_type': self.request_type,
            'estimated_complexity': self.estimated_complexity,
            'user_activity': self.user_activity,
            'user_cognitive_load': self.user_cognitive_load,
            'time_pressure': self.time_pressure,
            'session_length': self.session_length,
            'recent_friction': self.recent_friction,
            'system_load': self.system_load,
            'latency_budget_s': self.latency_budget_s,
        }


@dataclass
class MetaPlan:
    """
    The output of meta planning - how Ara should respond.

    This guides response generation before it even starts.
    """
    # Selected approach
    tool: str = "direct_answer"          # Which tool/approach to use
    style: str = "concise"               # What style to use

    # Predictions
    expected_rating: float = 0.0         # What rating do we expect
    expected_latency_s: float = 2.0      # How long should this take
    confidence: float = 0.5              # How confident is this plan

    # Warnings
    friction_risks: Dict[str, float] = field(default_factory=dict)
    active_scars: List[str] = field(default_factory=list)

    # Constraints from scars
    max_intrusiveness: float = 1.0       # Cap on intrusiveness
    avoid_tools: List[str] = field(default_factory=list)
    avoid_styles: List[str] = field(default_factory=list)

    # Reasoning
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool': self.tool,
            'style': self.style,
            'expected_rating': self.expected_rating,
            'expected_latency_s': self.expected_latency_s,
            'confidence': self.confidence,
            'friction_risks': self.friction_risks,
            'active_scars': self.active_scars,
            'max_intrusiveness': self.max_intrusiveness,
            'avoid_tools': self.avoid_tools,
            'avoid_styles': self.avoid_styles,
            'rationale': self.rationale,
        }


# =============================================================================
# Request Classifier
# =============================================================================

class RequestClassifier:
    """
    Classify user requests into context types.

    Uses simple heuristics for now - could be upgraded to ML later.
    """

    # Keywords for classification
    CLASSIFICATION_HINTS = {
        'code_review': ['review', 'check', 'look at', 'is this ok', 'feedback'],
        'debugging': ['error', 'bug', 'broken', 'not working', 'fix', 'crash', 'exception'],
        'explanation': ['explain', 'what is', 'how does', 'why', 'understand'],
        'generation': ['create', 'write', 'generate', 'implement', 'add', 'new'],
        'refactoring': ['refactor', 'improve', 'clean up', 'better', 'optimize'],
        'testing': ['test', 'unittest', 'pytest', 'spec', 'coverage'],
        'documentation': ['document', 'readme', 'docstring', 'comment'],
        'exploration': ['find', 'search', 'where', 'which files', 'grep'],
        'planning': ['plan', 'how should', 'approach', 'design', 'architecture'],
    }

    def classify(self, request_text: str) -> Tuple[str, float]:
        """
        Classify a request into a context type.

        Returns:
            (context_type, confidence)
        """
        text_lower = request_text.lower()

        # Score each context type
        scores = {}
        for ctx_type, hints in self.CLASSIFICATION_HINTS.items():
            score = sum(1 for hint in hints if hint in text_lower)
            if score > 0:
                scores[ctx_type] = score

        if not scores:
            return 'interaction', 0.3

        # Return highest scoring type
        best_type = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_type] / 3.0)

        return best_type, confidence

    def estimate_complexity(self, request_text: str) -> float:
        """
        Estimate request complexity (0-1).

        Higher complexity = more work needed.
        """
        # Simple heuristics
        complexity = 0.3  # Base

        # Length factor
        word_count = len(request_text.split())
        if word_count > 50:
            complexity += 0.2
        if word_count > 100:
            complexity += 0.2

        # Complexity keywords
        complex_hints = [
            'all', 'every', 'entire', 'whole', 'complete',
            'refactor', 'redesign', 'architecture',
            'multiple', 'several', 'various',
        ]
        for hint in complex_hints:
            if hint in request_text.lower():
                complexity += 0.1

        return min(1.0, complexity)


# =============================================================================
# Meta Planner
# =============================================================================

class MetaPlanner:
    """
    Memory-informed planning for response generation.

    Integrates:
    - Croft Model (preference prediction)
    - Scar Tissue (policy transforms)
    - EpisodicMemory (historical context)
    """

    def __init__(
        self,
        croft_model: Optional[CroftModel] = None,
        scar_registry: Optional[ScarRegistry] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        self.croft = croft_model
        self.scars = scar_registry
        self.memory = episodic_memory
        self.classifier = RequestClassifier()

        # Initialize components lazily if not provided
        if COMPONENTS_AVAILABLE:
            if self.croft is None:
                self.croft = CroftModel()
            if self.scars is None:
                self.scars = get_scar_registry()
            if self.memory is None:
                try:
                    self.memory = get_episodic_memory()
                except:
                    pass

        # Session tracking
        self._session_friction: List[str] = []
        self._session_ratings: List[int] = []
        self._session_start = time.time()

    def classify_request(self, request_text: str) -> RequestContext:
        """
        Create a RequestContext from the request text.

        Args:
            request_text: The user's request

        Returns:
            Classified RequestContext
        """
        # Classify request type
        req_type, type_confidence = self.classifier.classify(request_text)

        # Estimate complexity
        complexity = self.classifier.estimate_complexity(request_text)

        # Check for time pressure signals
        time_pressure = any(
            hint in request_text.lower()
            for hint in ['quick', 'fast', 'asap', 'urgent', 'hurry']
        )

        return RequestContext(
            request_type=req_type,
            request_text=request_text,
            estimated_complexity=complexity,
            time_pressure=time_pressure,
            session_length=len(self._session_ratings),
            recent_friction=self._session_friction[-5:],
            recent_ratings=self._session_ratings[-5:],
        )

    def plan(
        self,
        request_text: str,
        context: Optional[RequestContext] = None,
    ) -> MetaPlan:
        """
        Create a meta plan for responding to a request.

        Args:
            request_text: The user's request
            context: Optional pre-classified context

        Returns:
            MetaPlan with tool, style, and warnings
        """
        # Classify request if not provided
        if context is None:
            context = self.classify_request(request_text)

        plan = MetaPlan()
        reasons = []

        # Step 1: Check scars for policy transforms
        if self.scars:
            context_dict = context.to_dict()
            warnings = self.scars.get_warnings(context_dict, {})

            for warning in warnings:
                plan.active_scars.append(warning['scar_id'])

                # Extract constraints from scar transforms
                for transform in warning.get('transforms', []):
                    if transform['target'] == 'tool.preference' and 'avoid:' in str(transform.get('value', '')):
                        avoid_tool = transform['value'].replace('avoid:', '')
                        plan.avoid_tools.append(avoid_tool)
                    if transform['target'] == 'mode.intrusiveness':
                        if transform['operation'] == 'multiply':
                            plan.max_intrusiveness *= transform['value']

            if plan.active_scars:
                reasons.append(f"Constrained by {len(plan.active_scars)} scars")

        # Step 2: Get recommendation from Croft model
        if self.croft:
            rec = self.croft.recommend(
                context_type=context.request_type,
                exclude_tools=plan.avoid_tools,
                exclude_styles=plan.avoid_styles,
            )
            plan.tool = rec['tool']
            plan.style = rec['style']
            plan.expected_rating = rec['expected_rating']
            reasons.append(f"Croft recommends {rec['tool']}/{rec['style']}")

            # Get friction predictions
            pred = self.croft.predict(
                context_type=context.request_type,
                tool=plan.tool,
                style=plan.style,
            )
            plan.friction_risks = pred['friction_risk']
            plan.expected_latency_s = pred['latency_target']
            plan.confidence = pred['confidence']

        else:
            # Fallback: simple heuristics
            plan.tool, plan.style = self._fallback_selection(context)
            plan.confidence = 0.3
            reasons.append("Using fallback heuristics")

        # Step 3: Adjust for context factors
        if context.time_pressure:
            plan.style = 'concise'
            plan.expected_latency_s *= 0.5
            reasons.append("Time pressure: forcing concise style")

        if context.estimated_complexity > 0.7:
            if plan.tool == 'direct_answer':
                plan.tool = 'multi_step'
                reasons.append("Complex request: using multi-step")

        # Step 4: Check recent session friction
        if self._session_friction:
            recent_friction = set(self._session_friction[-3:])
            if 'too_verbose' in recent_friction:
                plan.style = 'concise'
                reasons.append("Recent verbose complaints: forcing concise")
            if 'too_slow' in recent_friction:
                plan.expected_latency_s *= 0.7
                reasons.append("Recent slow complaints: reducing latency budget")

        # Step 5: Check recent ratings
        if self._session_ratings and len(self._session_ratings) >= 3:
            avg_recent = sum(self._session_ratings[-3:]) / 3
            if avg_recent < 0:
                # Recent poor performance - be more conservative
                plan.confidence *= 0.7
                reasons.append("Recent poor ratings: reducing confidence")

        plan.rationale = "; ".join(reasons)
        return plan

    def _fallback_selection(
        self,
        context: RequestContext,
    ) -> Tuple[str, str]:
        """Fallback tool/style selection when Croft unavailable."""
        # Default mappings
        type_to_tool = {
            'code_review': 'read_file',
            'debugging': 'search_first',
            'explanation': 'direct_answer',
            'generation': 'direct_answer',
            'refactoring': 'read_file',
            'testing': 'read_file',
            'documentation': 'direct_answer',
            'exploration': 'grep',
            'planning': 'explain_steps',
        }

        type_to_style = {
            'code_review': 'technical',
            'debugging': 'step_by_step',
            'explanation': 'verbose',
            'generation': 'code_heavy',
            'refactoring': 'technical',
            'testing': 'code_heavy',
            'documentation': 'narrative',
            'exploration': 'concise',
            'planning': 'step_by_step',
        }

        tool = type_to_tool.get(context.request_type, 'direct_answer')
        style = type_to_style.get(context.request_type, 'concise')

        return tool, style

    def record_outcome(
        self,
        rating: int,
        friction_flags: Optional[List[str]] = None,
    ) -> None:
        """
        Record the outcome of a response for session learning.

        Called after each response to update session state.
        """
        self._session_ratings.append(rating)
        if friction_flags:
            self._session_friction.extend(friction_flags)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session performance."""
        if not self._session_ratings:
            return {
                'duration_s': time.time() - self._session_start,
                'interactions': 0,
                'avg_rating': 0.0,
                'friction_count': 0,
            }

        return {
            'duration_s': time.time() - self._session_start,
            'interactions': len(self._session_ratings),
            'avg_rating': sum(self._session_ratings) / len(self._session_ratings),
            'friction_count': len(self._session_friction),
            'common_friction': list(set(self._session_friction)),
        }

    def reset_session(self) -> None:
        """Reset session state."""
        self._session_friction = []
        self._session_ratings = []
        self._session_start = time.time()


# =============================================================================
# Convenience Functions
# =============================================================================

_planner: Optional[MetaPlanner] = None


def get_meta_planner() -> MetaPlanner:
    """Get the global meta planner instance."""
    global _planner
    if _planner is None:
        _planner = MetaPlanner()
    return _planner


def plan_response(request_text: str) -> MetaPlan:
    """
    Quick function to get a plan for a request.

    Usage:
        plan = plan_response("Help me debug this error")
        # => MetaPlan(tool='search_first', style='step_by_step', ...)
    """
    return get_meta_planner().plan(request_text)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta Planner - Memory-Informed Decisions")
    parser.add_argument("request", type=str, nargs="?", help="Request to plan for")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    planner = MetaPlanner()

    if args.interactive:
        print("Meta Planner - Interactive Mode")
        print("Enter requests to see planning (Ctrl+C to exit)\n")

        while True:
            try:
                request = input("Request: ").strip()
                if not request:
                    continue

                context = planner.classify_request(request)
                print(f"  Context: {context.request_type} (complexity={context.estimated_complexity:.2f})")

                plan = planner.plan(request, context)
                print(f"  Plan: {plan.tool} / {plan.style}")
                print(f"  Expected rating: {plan.expected_rating:.2f}")
                print(f"  Confidence: {plan.confidence:.2f}")
                if plan.friction_risks:
                    print(f"  Friction risks: {plan.friction_risks}")
                if plan.active_scars:
                    print(f"  Active scars: {plan.active_scars}")
                print(f"  Rationale: {plan.rationale}")
                print()

            except KeyboardInterrupt:
                print("\nBye!")
                break

    elif args.request:
        plan = planner.plan(args.request)
        print(json.dumps(plan.to_dict(), indent=2))

    else:
        parser.print_help()
