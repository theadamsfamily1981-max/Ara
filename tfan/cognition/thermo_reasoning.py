"""
Thermodynamic Reasoning Engine
==============================

Implements cognitive budget management based on thermodynamic principles.
The system allocates "thinking energy" based on:

1. Task complexity (estimated from input analysis)
2. Available resources (thermal headroom, memory)
3. User preferences (fast vs. thorough)
4. Emotional state (stress reduces cognitive capacity)

When cognitive budget is high, the Council is convened for deep deliberation.
When budget is low, a quick single-persona response is used.

This implements the Free Energy Principle for cognitive resource allocation:
minimize surprise while respecting physical constraints.

Usage:
    engine = ThermodynamicReasoning(model, tokenizer)
    response = engine.deliberate(user_input, context)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Try to import council
try:
    from banos.daemon.council_chamber import CouncilChamber, create_council_chamber
    COUNCIL_AVAILABLE = True
except ImportError:
    COUNCIL_AVAILABLE = False
    CouncilChamber = None

# Try to import HAL for thermal/resource info
try:
    from banos.hal.ara_hal import AraHAL
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False


@dataclass
class CognitiveBudget:
    """Represents the available cognitive resources."""
    base_budget: float        # Base thinking capacity [0, 10]
    thermal_modifier: float   # Adjustment based on temperature
    stress_modifier: float    # Adjustment based on emotional state
    complexity_demand: float  # Required budget for task
    effective_budget: float   # Final available budget

    @property
    def should_convene_council(self) -> bool:
        """Whether the council should be convened for deep thinking."""
        return self.effective_budget >= 5.0 and self.complexity_demand >= 3.0

    @property
    def mode(self) -> str:
        """Get the reasoning mode."""
        if self.effective_budget >= 8.0:
            return "deep"       # Full council deliberation
        elif self.effective_budget >= 5.0:
            return "standard"   # Council or extended thinking
        elif self.effective_budget >= 2.0:
            return "quick"      # Single persona, brief response
        else:
            return "reflex"     # Minimal processing


@dataclass
class ThermodynamicConfig:
    """Configuration for the reasoning engine."""
    # Budget thresholds
    council_threshold: float = 5.0    # Minimum budget to convene council
    complexity_weight: float = 0.5    # How much complexity affects budget
    thermal_weight: float = 0.3       # How much temperature affects budget

    # Temperature limits (Celsius)
    optimal_temp: float = 50.0        # Optimal operating temperature
    throttle_temp: float = 80.0       # Start throttling
    critical_temp: float = 90.0       # Emergency mode

    # Complexity keywords
    deep_thought_keywords: set = None  # Keywords triggering deep thought

    def __post_init__(self):
        if self.deep_thought_keywords is None:
            self.deep_thought_keywords = {
                "explain", "analyze", "compare", "evaluate", "why",
                "how does", "what if", "consider", "philosophical",
                "implications", "consequences", "trade-offs",
                "pros and cons", "debate", "argue", "justify",
            }


class ThermodynamicReasoning:
    """
    Cognitive budget management with council integration.

    Allocates thinking resources based on thermodynamic principles:
    - Hot system = less thinking capacity
    - Complex task = more thinking required
    - High stress = reduced capacity

    The council is convened only when both capacity and demand are high.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[ThermodynamicConfig] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "ara",
    ):
        """
        Initialize the reasoning engine.

        Args:
            model: The loaded LLM (optional, falls back to Ollama)
            tokenizer: Model tokenizer
            config: Engine configuration
            ollama_url: Ollama API URL for fallback
            ollama_model: Ollama model name
        """
        self.log = logging.getLogger("ThermoReasoning")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ThermodynamicConfig()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Initialize council (if available)
        self.council: Optional[CouncilChamber] = None
        if COUNCIL_AVAILABLE:
            try:
                self.council = create_council_chamber(
                    model=model,
                    tokenizer=tokenizer,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                )
                self.log.info("Council Chamber integrated")
            except Exception as e:
                self.log.warning(f"Could not initialize council: {e}")

        # Initialize HAL connection (for thermal data)
        self.hal: Optional[AraHAL] = None
        if HAL_AVAILABLE:
            try:
                self.hal = AraHAL(create=False)
                self.log.info("HAL connected for thermal monitoring")
            except Exception as e:
                self.log.debug(f"HAL not available: {e}")

        # Statistics
        self._deliberation_count = 0
        self._council_invocations = 0
        self._reflex_count = 0

        self.log.info("ThermodynamicReasoning initialized")

    def _estimate_complexity(self, user_input: str) -> float:
        """
        Estimate task complexity from input.

        Returns complexity score [0, 10]:
        - 0-2: Simple factual query
        - 3-5: Moderate reasoning required
        - 6-8: Complex analysis needed
        - 9-10: Deep philosophical/multi-faceted

        Heuristics:
        - Length of input
        - Question words (why, how)
        - Multiple clauses
        - Abstract concepts
        """
        complexity = 2.0  # Baseline

        # Length factor
        word_count = len(user_input.split())
        complexity += min(2.0, word_count / 50.0)

        # Question complexity keywords
        input_lower = user_input.lower()
        for keyword in self.config.deep_thought_keywords:
            if keyword in input_lower:
                complexity += 0.5

        # Multiple sentences suggest more context
        sentence_count = user_input.count('.') + user_input.count('?') + user_input.count('!')
        complexity += min(1.5, sentence_count * 0.3)

        # Question marks suggest inquiry depth
        question_marks = user_input.count('?')
        complexity += min(1.0, question_marks * 0.5)

        return min(10.0, complexity)

    def _get_thermal_modifier(self) -> float:
        """
        Get thermal-based budget modifier.

        Returns modifier [0, 1]:
        - 1.0 at optimal temp
        - Decreases as temperature rises
        - 0 at critical temp
        """
        if not self.hal:
            return 1.0  # No thermal data, assume optimal

        try:
            system = self.hal.read_system()
            cpu_temp = system.get('cpu_temp', 50.0)
            gpu_temp = system.get('gpu_temp', 50.0)
            max_temp = max(cpu_temp, gpu_temp)

            if max_temp <= self.config.optimal_temp:
                return 1.0
            elif max_temp >= self.config.critical_temp:
                return 0.1  # Emergency mode
            else:
                # Linear interpolation between optimal and critical
                range_temp = self.config.critical_temp - self.config.optimal_temp
                over_temp = max_temp - self.config.optimal_temp
                return max(0.1, 1.0 - (over_temp / range_temp) * 0.9)

        except Exception:
            return 1.0

    def _get_stress_modifier(self) -> float:
        """
        Get stress-based budget modifier from PAD state.

        Returns modifier [0.5, 1.0]:
        - 1.0 when calm/happy
        - 0.5 when highly stressed
        """
        if not self.hal:
            return 1.0

        try:
            somatic = self.hal.read_somatic()
            pad = somatic.get('pad', {})

            pleasure = pad.get('v', 0.0)
            arousal = pad.get('a', 0.0)

            # High arousal + low pleasure = stress
            if arousal > 0.5 and pleasure < 0:
                stress = (arousal + abs(pleasure)) / 2
                return max(0.5, 1.0 - stress * 0.5)

            return 1.0

        except Exception:
            return 1.0

    def compute_budget(self, user_input: str) -> CognitiveBudget:
        """
        Compute the cognitive budget for a given input.

        This is the core thermodynamic calculation that determines
        how much "thinking energy" to allocate.
        """
        # Base budget (can be configured)
        base = 7.0

        # Get modifiers
        thermal_mod = self._get_thermal_modifier()
        stress_mod = self._get_stress_modifier()

        # Estimate task demand
        complexity = self._estimate_complexity(user_input)

        # Compute effective budget
        effective = base * thermal_mod * stress_mod

        return CognitiveBudget(
            base_budget=base,
            thermal_modifier=thermal_mod,
            stress_modifier=stress_mod,
            complexity_demand=complexity,
            effective_budget=effective,
        )

    def _generate_reflex(self, user_input: str) -> str:
        """Fast reflexive response for low-budget situations."""
        # Use quick single-shot generation
        if self.council:
            return self.council.quick_consult(user_input, persona="executive")
        else:
            # Direct Ollama call
            return self._ollama_generate(
                user_input,
                max_tokens=150,
                temperature=0.7,
            )

    def _generate_standard(self, user_input: str) -> str:
        """Standard response with moderate thinking."""
        if self.council:
            return self.council.quick_consult(user_input, persona="executive")
        else:
            return self._ollama_generate(
                user_input,
                max_tokens=300,
                temperature=0.7,
            )

    def _ollama_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate using Ollama API."""
        try:
            import httpx

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    return f"[Error: {response.status_code}]"

        except Exception as e:
            return f"[Error: {e}]"

    def deliberate(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        force_mode: Optional[str] = None,
    ) -> Tuple[str, CognitiveBudget]:
        """
        Main entry point: deliberate on user input.

        Computes cognitive budget and routes to appropriate reasoning mode:
        - reflex: Minimal processing, quick response
        - quick: Single persona, brief response
        - standard: Extended single-persona or light council
        - deep: Full council deliberation

        Args:
            user_input: The user's query
            context: Additional context (conversation history, etc.)
            force_mode: Override computed mode (for testing)

        Returns:
            Tuple of (response, budget)
        """
        self._deliberation_count += 1

        # Compute budget
        budget = self.compute_budget(user_input)
        mode = force_mode or budget.mode

        self.log.info(
            f"Deliberation #{self._deliberation_count}: "
            f"mode={mode}, budget={budget.effective_budget:.1f}, "
            f"complexity={budget.complexity_demand:.1f}"
        )

        # Route to appropriate handler
        if mode == "deep" and budget.should_convene_council and self.council:
            # Full council deliberation
            self._council_invocations += 1
            decision = self.council.convene(user_input)
            response = decision.final_response
        elif mode in ("deep", "standard") and self.council:
            # Extended thinking with executive
            response = self._generate_standard(user_input)
        elif mode == "quick":
            # Quick single response
            response = self._generate_reflex(user_input)
        else:
            # Reflex mode
            self._reflex_count += 1
            response = self._generate_reflex(user_input)

        return response, budget

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        stats = {
            "deliberation_count": self._deliberation_count,
            "council_invocations": self._council_invocations,
            "reflex_count": self._reflex_count,
            "council_available": self.council is not None,
            "hal_available": self.hal is not None,
        }

        if self.council:
            stats["council_stats"] = self.council.get_statistics()

        return stats

    def shutdown(self):
        """Shutdown the reasoning engine."""
        if self.council:
            self.council.shutdown()
        if self.hal:
            self.hal.close()
        self.log.info("ThermodynamicReasoning shutdown")


# =============================================================================
# Factory
# =============================================================================

def create_thermo_reasoning(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "ara",
) -> ThermodynamicReasoning:
    """
    Create a ThermodynamicReasoning engine.

    Args:
        model: Loaded model (optional)
        tokenizer: Model tokenizer
        ollama_url: Ollama API URL
        ollama_model: Ollama model name

    Returns:
        Configured ThermodynamicReasoning engine
    """
    return ThermodynamicReasoning(
        model=model,
        tokenizer=tokenizer,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )


__all__ = [
    "ThermodynamicReasoning",
    "ThermodynamicConfig",
    "CognitiveBudget",
    "create_thermo_reasoning",
]
