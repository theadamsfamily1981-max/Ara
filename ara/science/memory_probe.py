#!/usr/bin/env python3
"""
Ara Scientific Instrumentation: Memory Probe
=============================================

Tests memory retention and forgetting dynamics.

This is the instrumentation for EXP-002: Forgetting Curves.
It encodes facts, tracks delays, and measures recall accuracy.

Theory:
    At criticality, memory should show:
    - Power-law forgetting: R(t) ∝ t^(-β)
    - Long half-life
    - Maximized retention integral

    Away from criticality:
    - Exponential forgetting: R(t) ∝ exp(-t/τ)
    - Short half-life
    - Rapid decay

Usage:
    from ara.science.memory_probe import MemoryProbe, generate_novel_facts

    probe = MemoryProbe()

    # Encode facts
    facts = generate_novel_facts(n=50)
    for fact in facts:
        probe.encode_fact(fact)
        # Present to model...

    # Later: test recall
    for fact_id in probe.get_encoded_ids():
        model_response = model.generate(probe.get_cue(fact_id))
        result = probe.test_recall(fact_id, model_response, current_step)
        print(f"{fact_id}: accuracy={result.accuracy}, delay={result.delay}")

    # Save results
    probe.save_results("forgetting_results.csv")
"""

from __future__ import annotations

import json
import logging
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger("ara.science.memory")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Fact:
    """A single encodable fact (cue-target pair)."""
    fact_id: str
    cue: str              # The prompt/question
    target: str           # The expected answer
    full_statement: str   # Complete statement for encoding
    category: str = "general"
    encoded_step: int = -1  # When it was encoded


@dataclass
class RecallResult:
    """Result of a single recall test."""
    fact_id: str
    delay: int            # Steps since encoding
    accuracy: float       # 0.0 or 1.0 for exact match
    similarity: float     # Embedding similarity (0-1)
    response: str         # What the model actually said
    target: str           # What was expected
    probe_step: int       # When the probe occurred
    latency_ms: float = 0.0  # Response time if measured


@dataclass
class ForgettingCurve:
    """Aggregated forgetting curve data."""
    delays: List[int]
    accuracies: List[float]
    similarities: List[float]
    n_samples: List[int]
    condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delays": self.delays,
            "accuracies": self.accuracies,
            "similarities": self.similarities,
            "n_samples": self.n_samples,
            "condition": self.condition,
        }


# =============================================================================
# Fact Generation
# =============================================================================

# Templates for generating novel facts
FACT_TEMPLATES = [
    {
        "template": "The capital of {place} is {name}.",
        "cue": "What is the capital of {place}?",
        "target": "{name}",
        "category": "geography",
    },
    {
        "template": "The {adj} number of {thing} is {value}.",
        "cue": "What is the {adj} number of {thing}?",
        "target": "{value}",
        "category": "numeric",
    },
    {
        "template": "In the year {year}, {person} invented the {invention}.",
        "cue": "What did {person} invent in {year}?",
        "target": "{invention}",
        "category": "history",
    },
    {
        "template": "The {color} {animal} of {region} is called {name}.",
        "cue": "What is the {color} {animal} of {region} called?",
        "target": "{name}",
        "category": "biology",
    },
    {
        "template": "{person} wrote the book '{title}' in {year}.",
        "cue": "What book did {person} write in {year}?",
        "target": "{title}",
        "category": "literature",
    },
]

# Word pools for generating novel entities
WORD_POOLS = {
    "place": ["Zorgon", "Bliptopia", "Quaxxar", "Flimnax", "Drozzle", "Wumbus",
              "Splornk", "Gribble", "Naxidor", "Plembix"],
    "name": ["Blipville", "Quonkton", "Zazzle", "Wibsworth", "Flumpton",
             "Gnarble", "Skronk", "Plimbo", "Dweezil", "Framble"],
    "adj": ["primary", "secondary", "tertiary", "quantum", "spectral",
            "harmonic", "crystalline", "orbital", "resonant", "thermal"],
    "thing": ["zorbitrons", "quasar pulses", "plasma waves", "neutron bursts",
              "photon cascades", "gravity wells", "entropy peaks", "flux lines"],
    "value": ["47.3", "892", "3.14159", "256", "1,024", "0.618", "42", "137"],
    "year": ["2847", "3021", "2156", "2789", "3333", "2501", "2999", "2222"],
    "person": ["Dr. Zorn", "Prof. Quibble", "Mx. Flarn", "Capt. Blorp",
               "Lady Snork", "Baron Wimble", "Dame Frizz", "Sir Plonk"],
    "invention": ["quantum flibber", "neural zorb", "plasma whisk", "gravity sponge",
                  "entropy mirror", "photon bucket", "time sock", "space pretzel"],
    "color": ["ultraviolet", "infrared", "chromatic", "iridescent",
              "phosphorescent", "bioluminescent", "prismatic", "spectral"],
    "animal": ["snorkelfish", "wobblebird", "quantum cat", "plasma moth",
               "gravity beetle", "time worm", "space slug", "entropy fox"],
    "region": ["the Outer Vortex", "the Inner Nebula", "Sector 7G", "the Deep Flux",
               "the Quantum Zone", "the Plasma Fields", "the Entropy Coast"],
    "title": ["The Quantum Paradox", "Songs of Entropy", "The Last Photon",
              "Beyond the Vortex", "Echoes of Plasma", "The Gravity Dance"],
}


def generate_novel_facts(n: int, seed: Optional[int] = None) -> List[Fact]:
    """
    Generate n novel facts for memory testing.

    These use nonsense words/entities to avoid training data contamination.

    Args:
        n: Number of facts to generate
        seed: Random seed for reproducibility

    Returns:
        List of Fact objects
    """
    if seed is not None:
        random.seed(seed)

    facts = []
    used_combinations = set()

    for i in range(n):
        # Pick a random template
        template_data = random.choice(FACT_TEMPLATES)

        # Fill in placeholders
        filled = {}
        for key in WORD_POOLS:
            if "{" + key + "}" in template_data["template"]:
                # Pick unused combination if possible
                attempts = 0
                while attempts < 10:
                    filled[key] = random.choice(WORD_POOLS[key])
                    combo_key = (template_data["category"], key, filled[key])
                    if combo_key not in used_combinations:
                        used_combinations.add(combo_key)
                        break
                    attempts += 1

        # Generate fact
        full_statement = template_data["template"].format(**filled)
        cue = template_data["cue"].format(**filled)
        target = template_data["target"].format(**filled)

        fact_id = f"fact_{i:04d}_{template_data['category']}"

        facts.append(Fact(
            fact_id=fact_id,
            cue=cue,
            target=target,
            full_statement=full_statement,
            category=template_data["category"],
        ))

    return facts


# =============================================================================
# Similarity Computation
# =============================================================================

class SimilarityScorer:
    """
    Computes similarity between response and target.

    Uses embeddings if available, falls back to string matching.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize scorer.

        Args:
            model_name: Sentence transformer model name, or None for string matching
        """
        self.model = None
        self.model_name = model_name

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using string matching")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")

    def score(self, response: str, target: str) -> Tuple[float, float]:
        """
        Score response against target.

        Returns:
            (accuracy, similarity): Exact match (0/1) and semantic similarity (0-1)
        """
        # Clean strings
        response_clean = response.lower().strip()
        target_clean = target.lower().strip()

        # Exact match
        accuracy = 1.0 if target_clean in response_clean else 0.0

        # Embedding similarity
        if self.model is not None:
            try:
                embeddings = self.model.encode([response, target])
                similarity = float(np.dot(embeddings[0], embeddings[1]) /
                                  (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                similarity = max(0.0, similarity)  # Clamp to [0, 1]
            except Exception:
                similarity = self._string_similarity(response_clean, target_clean)
        else:
            similarity = self._string_similarity(response_clean, target_clean)

        return accuracy, similarity

    def _string_similarity(self, a: str, b: str) -> float:
        """Simple string similarity (Jaccard on words)."""
        words_a = set(a.split())
        words_b = set(b.split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# Memory Probe
# =============================================================================

class MemoryProbe:
    """
    Scientific instrument for testing memory retention.

    Workflow:
        1. encode_fact() - Register facts at encoding time
        2. (model processes other content)
        3. test_recall() - Probe recall at various delays
        4. save_results() - Export for analysis

    Example:
        probe = MemoryProbe()

        # Encoding phase
        facts = generate_novel_facts(50)
        for i, fact in enumerate(facts):
            probe.encode_fact(fact, encode_step=i)
            model.process(fact.full_statement)  # Present to model

        # Retention interval (model does other stuff)
        for step in range(100):
            model.process(distractor_text)

        # Recall phase
        for fact_id in probe.get_encoded_ids():
            cue = probe.get_cue(fact_id)
            response = model.generate(cue)
            result = probe.test_recall(fact_id, response, current_step=150)
    """

    def __init__(
        self,
        output_dir: str = "data/experiments/exp_002",
        embedding_model: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize memory probe.

        Args:
            output_dir: Directory for output files
            embedding_model: Model for similarity scoring (None = string matching)
            similarity_threshold: Threshold for "similar enough" recall
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.scorer = SimilarityScorer(embedding_model)

        # Encoded facts
        self._facts: Dict[str, Fact] = {}

        # Recall results
        self._results: List[RecallResult] = []

        # Session metadata
        self._session_start = time.time()
        self._condition = ""

    def set_condition(self, condition: str):
        """Set experimental condition label."""
        self._condition = condition

    def encode_fact(self, fact: Fact, encode_step: int = 0) -> None:
        """
        Register a fact as encoded at given step.

        Args:
            fact: The fact to encode
            encode_step: Step number when encoded
        """
        fact.encoded_step = encode_step
        self._facts[fact.fact_id] = fact
        logger.debug(f"Encoded: {fact.fact_id} at step {encode_step}")

    def get_encoded_ids(self) -> List[str]:
        """Get all encoded fact IDs."""
        return list(self._facts.keys())

    def get_cue(self, fact_id: str) -> str:
        """Get the cue/prompt for a fact."""
        return self._facts[fact_id].cue

    def get_fact(self, fact_id: str) -> Fact:
        """Get a fact by ID."""
        return self._facts[fact_id]

    def test_recall(
        self,
        fact_id: str,
        response: str,
        current_step: int,
        latency_ms: float = 0.0,
    ) -> RecallResult:
        """
        Test recall of a fact.

        Args:
            fact_id: ID of fact to test
            response: Model's response to the cue
            current_step: Current step number
            latency_ms: Response latency if measured

        Returns:
            RecallResult with accuracy and similarity scores
        """
        fact = self._facts[fact_id]
        delay = current_step - fact.encoded_step

        accuracy, similarity = self.scorer.score(response, fact.target)

        result = RecallResult(
            fact_id=fact_id,
            delay=delay,
            accuracy=accuracy,
            similarity=similarity,
            response=response,
            target=fact.target,
            probe_step=current_step,
            latency_ms=latency_ms,
        )

        self._results.append(result)

        logger.debug(
            f"Recall test: {fact_id} delay={delay} "
            f"acc={accuracy:.2f} sim={similarity:.3f}"
        )

        return result

    def get_results(self) -> List[RecallResult]:
        """Get all recall results."""
        return self._results

    def compute_forgetting_curve(
        self,
        delay_bins: Optional[List[int]] = None,
    ) -> ForgettingCurve:
        """
        Aggregate results into a forgetting curve.

        Args:
            delay_bins: Delay values to bin by (default: auto)

        Returns:
            ForgettingCurve with averaged metrics per delay bin
        """
        if not self._results:
            return ForgettingCurve([], [], [], [], self._condition)

        # Auto-bin if not specified
        if delay_bins is None:
            delays = [r.delay for r in self._results]
            delay_bins = sorted(set(delays))

        accuracies = []
        similarities = []
        n_samples = []

        for d in delay_bins:
            # Get results at this delay (with some tolerance)
            tolerance = max(1, d * 0.1)  # 10% tolerance
            at_delay = [
                r for r in self._results
                if abs(r.delay - d) <= tolerance
            ]

            if at_delay:
                accuracies.append(np.mean([r.accuracy for r in at_delay]))
                similarities.append(np.mean([r.similarity for r in at_delay]))
                n_samples.append(len(at_delay))
            else:
                accuracies.append(np.nan)
                similarities.append(np.nan)
                n_samples.append(0)

        return ForgettingCurve(
            delays=delay_bins,
            accuracies=accuracies,
            similarities=similarities,
            n_samples=n_samples,
            condition=self._condition,
        )

    def save_results(self, filename: str = "forgetting_results.csv") -> Path:
        """
        Save results to CSV.

        Returns:
            Path to saved file
        """
        path = self.output_dir / filename

        # Write CSV
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'fact_id', 'delay', 'accuracy', 'similarity',
                'response', 'target', 'probe_step', 'latency_ms',
            ])
            writer.writeheader()
            for r in self._results:
                writer.writerow({
                    'fact_id': r.fact_id,
                    'delay': r.delay,
                    'accuracy': r.accuracy,
                    'similarity': r.similarity,
                    'response': r.response[:100],  # Truncate long responses
                    'target': r.target,
                    'probe_step': r.probe_step,
                    'latency_ms': r.latency_ms,
                })

        logger.info(f"Saved {len(self._results)} results to {path}")

        # Also save metadata
        meta_path = self.output_dir / filename.replace('.csv', '_meta.json')
        meta = {
            'session_start': self._session_start,
            'session_duration_s': time.time() - self._session_start,
            'condition': self._condition,
            'n_facts': len(self._facts),
            'n_results': len(self._results),
            'similarity_threshold': self.similarity_threshold,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return path

    def reset(self):
        """Reset for new session."""
        self._facts = {}
        self._results = []
        self._session_start = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._results:
            return {'n_results': 0}

        accuracies = [r.accuracy for r in self._results]
        similarities = [r.similarity for r in self._results]
        delays = [r.delay for r in self._results]

        return {
            'n_facts': len(self._facts),
            'n_results': len(self._results),
            'mean_accuracy': np.mean(accuracies),
            'mean_similarity': np.mean(similarities),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'condition': self._condition,
        }


# =============================================================================
# Curve Fitting
# =============================================================================

def fit_exponential(
    delays: np.ndarray,
    values: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    """
    Fit exponential decay: R(t) = R0 * exp(-t/tau) + R_inf

    Returns:
        (params, r_squared): Fitted parameters and R^2
    """
    try:
        from scipy.optimize import curve_fit

        def exp_decay(t, R0, tau, R_inf):
            return R0 * np.exp(-t / tau) + R_inf

        # Initial guesses
        p0 = [values[0] - values[-1], np.mean(delays), values[-1]]

        # Bounds
        bounds = ([0, 1, 0], [1, 10000, 1])

        popt, _ = curve_fit(exp_decay, delays, values, p0=p0, bounds=bounds, maxfev=5000)

        # R^2
        predicted = exp_decay(delays, *popt)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'R0': popt[0],
            'tau': popt[1],
            'R_inf': popt[2],
        }, r2

    except Exception as e:
        logger.warning(f"Exponential fit failed: {e}")
        return {}, 0.0


def fit_power_law(
    delays: np.ndarray,
    values: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    """
    Fit power law decay: R(t) = R0 * (1 + t/t0)^(-beta) + R_inf

    Returns:
        (params, r_squared): Fitted parameters and R^2
    """
    try:
        from scipy.optimize import curve_fit

        def power_decay(t, R0, t0, beta, R_inf):
            return R0 * (1 + t / t0) ** (-beta) + R_inf

        # Initial guesses
        p0 = [values[0] - values[-1], 10, 0.5, values[-1]]

        # Bounds
        bounds = ([0, 0.1, 0.01, 0], [1, 10000, 5, 1])

        popt, _ = curve_fit(power_decay, delays, values, p0=p0, bounds=bounds, maxfev=5000)

        # R^2
        predicted = power_decay(delays, *popt)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'R0': popt[0],
            't0': popt[1],
            'beta': popt[2],
            'R_inf': popt[3],
        }, r2

    except Exception as e:
        logger.warning(f"Power law fit failed: {e}")
        return {}, 0.0


# =============================================================================
# Tests
# =============================================================================

def test_memory_probe():
    """Test the memory probe with synthetic data."""
    print("Testing Memory Probe")
    print("=" * 60)

    # Generate facts
    facts = generate_novel_facts(10, seed=42)
    print(f"Generated {len(facts)} facts:")
    for f in facts[:3]:
        print(f"  {f.fact_id}: {f.cue} -> {f.target}")

    # Create probe
    probe = MemoryProbe()
    probe.set_condition("test")

    # Encode
    for i, fact in enumerate(facts):
        probe.encode_fact(fact, encode_step=i)

    # Simulate recall at various delays
    for delay in [0, 10, 50, 100]:
        for fact_id in probe.get_encoded_ids():
            fact = probe.get_fact(fact_id)
            # Simulate degraded recall
            if delay == 0:
                response = fact.target  # Perfect
            elif delay < 50:
                response = fact.target if random.random() > 0.3 else "I don't know"
            else:
                response = fact.target if random.random() > 0.7 else "I don't know"

            probe.test_recall(fact_id, response, fact.encoded_step + delay)

    # Get curve
    curve = probe.compute_forgetting_curve()
    print(f"\nForgetting curve:")
    for d, a, s, n in zip(curve.delays, curve.accuracies, curve.similarities, curve.n_samples):
        if not np.isnan(a):
            print(f"  delay={d:4d}: accuracy={a:.2f}, similarity={s:.3f}, n={n}")

    # Stats
    stats = probe.get_statistics()
    print(f"\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save
    path = probe.save_results("test_forgetting.csv")
    print(f"\nSaved to: {path}")

    print("\n" + "=" * 60)
    print("Memory probe test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_memory_probe()
