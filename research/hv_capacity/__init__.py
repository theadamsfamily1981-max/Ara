"""
Hypervector Capacity Analysis
=============================

Tools for analyzing HV system capacity and limits.

Key Questions:
1. How many modalities can we bind before interference kills us?
2. What's the optimal dimensionality for our use case?
3. When does sparse bundling beat dense bundling?

Theoretical Background:
- HV capacity scales as O(D / log(D)) for random vectors
- Binding (circular convolution) is invertible but spreads energy
- Bundling (addition) creates interference that grows with item count

This module provides empirical analysis tools to validate
the choices made in ara/nervous/axis_mundi.py

NOT RUNTIME CODE - Analysis/validation only.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CapacityResult:
    """Result of a capacity analysis experiment."""
    dimension: int
    num_items: int
    operation: str  # 'bundle', 'bind', 'mixed'
    retrieval_accuracy: float
    similarity_mean: float
    similarity_std: float
    interference_level: float


def generate_random_hv(dim: int, bipolar: bool = True) -> np.ndarray:
    """Generate a random hypervector."""
    if bipolar:
        return np.random.choice([-1, 1], size=dim).astype(np.float32)
    else:
        return np.random.randn(dim).astype(np.float32)


def circular_bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """Circular convolution binding."""
    return np.real(np.fft.ifft(np.fft.fft(hv1) * np.fft.fft(hv2)))


def circular_unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Circular correlation unbinding."""
    return np.real(np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(key))))


def bundle(hvs: List[np.ndarray], normalize: bool = True) -> np.ndarray:
    """Bundle multiple HVs via addition."""
    result = np.sum(hvs, axis=0)
    if normalize:
        result = result / np.linalg.norm(result)
    return result


def sparse_topk(hv: np.ndarray, k: int) -> np.ndarray:
    """Keep only top-k magnitude components."""
    indices = np.argsort(np.abs(hv))[-k:]
    sparse = np.zeros_like(hv)
    sparse[indices] = hv[indices]
    return sparse


def cosine_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2) + 1e-8))


class CapacityAnalyzer:
    """
    Analyze HV capacity for Ara's use case.

    Experiments:
    1. Bundle capacity: How many items can we bundle before retrieval fails?
    2. Bind capacity: How deep can we nest bindings?
    3. Mixed capacity: Ara's actual usage pattern (bind modalities, bundle time)
    """

    def __init__(self, dim: int = 8192):
        self.dim = dim
        self.results: List[CapacityResult] = []

    def analyze_bundle_capacity(self,
                                max_items: int = 100,
                                trials: int = 10) -> List[CapacityResult]:
        """
        Test how many items can be bundled before retrieval accuracy drops.

        For each item count n:
        1. Generate n random HVs and one query
        2. Bundle them all
        3. Try to retrieve query via similarity
        4. Record accuracy across trials
        """
        results = []

        for n in range(2, max_items + 1, 5):
            accuracies = []
            similarities = []

            for _ in range(trials):
                # Generate items including one query
                items = [generate_random_hv(self.dim) for _ in range(n)]
                query = items[0]

                # Bundle all
                bundled = bundle(items)

                # Check if query is most similar to bundled
                sim_query = cosine_similarity(bundled, query)

                # Compare to random vector similarity
                random_sim = cosine_similarity(bundled, generate_random_hv(self.dim))

                accuracies.append(1.0 if sim_query > random_sim else 0.0)
                similarities.append(sim_query)

            result = CapacityResult(
                dimension=self.dim,
                num_items=n,
                operation='bundle',
                retrieval_accuracy=np.mean(accuracies),
                similarity_mean=np.mean(similarities),
                similarity_std=np.std(similarities),
                interference_level=1.0 - np.mean(similarities)
            )
            results.append(result)
            self.results.append(result)

        return results

    def analyze_bind_depth(self,
                          max_depth: int = 20,
                          trials: int = 10) -> List[CapacityResult]:
        """
        Test how deep we can nest bindings before unbinding fails.

        For each depth d:
        1. Generate d keys and one value
        2. Bind: bound = ((value ⊗ key_1) ⊗ key_2) ⊗ ... ⊗ key_d
        3. Unbind: recovered = bound ⊘ key_d ⊘ ... ⊘ key_1
        4. Check similarity to original value
        """
        results = []

        for depth in range(1, max_depth + 1):
            similarities = []

            for _ in range(trials):
                # Generate keys and value
                keys = [generate_random_hv(self.dim) for _ in range(depth)]
                value = generate_random_hv(self.dim)

                # Bind progressively
                bound = value.copy()
                for key in keys:
                    bound = circular_bind(bound, key)

                # Unbind in reverse order
                recovered = bound.copy()
                for key in reversed(keys):
                    recovered = circular_unbind(recovered, key)

                sim = cosine_similarity(recovered, value)
                similarities.append(sim)

            result = CapacityResult(
                dimension=self.dim,
                num_items=depth,
                operation='bind',
                retrieval_accuracy=np.mean([1.0 if s > 0.5 else 0.0 for s in similarities]),
                similarity_mean=np.mean(similarities),
                similarity_std=np.std(similarities),
                interference_level=1.0 - np.mean(similarities)
            )
            results.append(result)
            self.results.append(result)

        return results

    def analyze_ara_pattern(self,
                           num_modalities: int = 8,
                           temporal_bundle_size: int = 10,
                           trials: int = 10) -> CapacityResult:
        """
        Test Ara's actual usage: bind modalities, bundle over time.

        Pattern:
        1. Generate num_modalities modality HVs
        2. Bind each modality with its phase code
        3. Bundle all bound modalities into world_hv
        4. Do this for temporal_bundle_size timesteps
        5. Bundle timesteps (with temporal decay)
        6. Try to recover a specific modality at a specific time
        """
        accuracies = []

        for _ in range(trials):
            # Generate phase codes (one per modality)
            phase_codes = [generate_random_hv(self.dim) for _ in range(num_modalities)]

            # Generate temporal sequence
            world_hvs = []
            all_modality_hvs = []

            for t in range(temporal_bundle_size):
                # Generate modality HVs for this timestep
                modality_hvs = [generate_random_hv(self.dim) for _ in range(num_modalities)]
                all_modality_hvs.append(modality_hvs)

                # Bind each modality with its phase code
                bound_hvs = [circular_bind(m, p) for m, p in zip(modality_hvs, phase_codes)]

                # Bundle into world_hv for this timestep
                world_hv = bundle(bound_hvs)
                world_hvs.append(world_hv)

            # Bundle timesteps (simple average for now)
            temporal_bundle = bundle(world_hvs)

            # Try to recover modality 0 at time 0
            target_modality = all_modality_hvs[0][0]

            # Unbind from temporal bundle using phase code
            recovered = circular_unbind(temporal_bundle, phase_codes[0])

            sim = cosine_similarity(recovered, target_modality)
            accuracies.append(1.0 if sim > 0.3 else 0.0)  # Lower threshold due to temporal mixing

        result = CapacityResult(
            dimension=self.dim,
            num_items=num_modalities * temporal_bundle_size,
            operation='mixed',
            retrieval_accuracy=np.mean(accuracies),
            similarity_mean=np.mean(accuracies),  # Simplified
            similarity_std=np.std(accuracies),
            interference_level=1.0 - np.mean(accuracies)
        )
        self.results.append(result)
        return result

    def recommend_parameters(self) -> dict:
        """
        Based on analysis, recommend parameters for Ara.

        Returns recommended dim, max modalities, temporal window, etc.
        """
        # Find capacity limits from results
        bundle_limit = 16  # Conservative default
        bind_limit = 10

        for r in self.results:
            if r.operation == 'bundle' and r.retrieval_accuracy < 0.9:
                bundle_limit = min(bundle_limit, r.num_items - 5)
                break

        for r in self.results:
            if r.operation == 'bind' and r.retrieval_accuracy < 0.9:
                bind_limit = min(bind_limit, r.num_items - 2)
                break

        return {
            'recommended_dim': self.dim,
            'max_modalities': min(bundle_limit, 16),  # Ara uses 16 max
            'max_bind_depth': bind_limit,
            'temporal_window_safe': 10,
            'sparsity_k_suggested': self.dim // 8,
            'notes': [
                f'Analyzed at D={self.dim}',
                f'Bundle capacity ~{bundle_limit} items at 90% accuracy',
                f'Bind depth ~{bind_limit} at 90% recovery',
                'Use sparse_topk for long-term storage'
            ]
        }


__all__ = [
    'CapacityResult',
    'generate_random_hv',
    'circular_bind',
    'circular_unbind',
    'bundle',
    'sparse_topk',
    'cosine_similarity',
    'CapacityAnalyzer',
]
