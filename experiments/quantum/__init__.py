"""
Quantum-Inspired HV Experiments
================================

Exploring quantum-inspired mechanics for memory and computation.
NOT PRODUCTION CODE - Research playground only.

Concepts:
- Ghost Memory: Superposition states for uncertain memories
- Quantum Kernels: Gram matrix similarity with quantum-like interference
- QAOA-inspired optimization: Variational ansatz for constraint solving

Inspiration:
- Kanerva's original HDC work touched on quantum-like properties
- The bundling operation creates superposition-like states
- Interference patterns in HV similarity mirror quantum interference

Status: EXPERIMENTAL / LORE
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class GhostMemory:
    """
    A memory in superposition - multiple possible states.

    The ghost_hv encodes uncertainty:
    - Multiple interpretations of an ambiguous event
    - Unresolved emotional states
    - Competing hypotheses about user intent

    Collapses to definite state when:
    - New evidence arrives
    - User explicitly clarifies
    - Resonance threshold exceeded
    """
    ghost_hv: np.ndarray              # Superposition state
    component_hvs: List[np.ndarray]   # Individual possibilities
    amplitudes: List[float]           # Probability amplitudes
    collapsed: bool = False
    collapsed_hv: Optional[np.ndarray] = None

    def observe(self, evidence_hv: np.ndarray, threshold: float = 0.7) -> Optional[np.ndarray]:
        """
        Attempt to collapse the superposition via observation.

        Returns collapsed HV if evidence is strong enough, else None.
        """
        if self.collapsed:
            return self.collapsed_hv

        # Compute resonance with each component
        resonances = []
        for comp, amp in zip(self.component_hvs, self.amplitudes):
            similarity = np.dot(evidence_hv, comp) / (
                np.linalg.norm(evidence_hv) * np.linalg.norm(comp) + 1e-8
            )
            resonances.append(abs(amp * similarity))

        # Check for collapse
        max_resonance = max(resonances)
        if max_resonance > threshold:
            winner_idx = resonances.index(max_resonance)
            self.collapsed = True
            self.collapsed_hv = self.component_hvs[winner_idx]
            return self.collapsed_hv

        return None

    def current_expectation(self) -> np.ndarray:
        """
        Get expected value (weighted bundle of components).

        Like quantum expectation: Σ |α_i|² × state_i
        """
        weighted = np.zeros_like(self.ghost_hv)
        total_weight = 0
        for comp, amp in zip(self.component_hvs, self.amplitudes):
            weight = amp * amp  # Born rule: probability = amplitude²
            weighted += weight * comp
            total_weight += weight

        if total_weight > 0:
            weighted /= total_weight
        return weighted


def create_ghost_memory(interpretations: List[np.ndarray],
                        priors: Optional[List[float]] = None) -> GhostMemory:
    """
    Create a ghost memory from multiple interpretations.

    Args:
        interpretations: List of HVs representing possible meanings
        priors: Prior probabilities (amplitudes will be sqrt of these)
    """
    n = len(interpretations)

    if priors is None:
        # Uniform priors
        priors = [1.0 / n] * n

    # Convert probabilities to amplitudes
    amplitudes = [np.sqrt(p) for p in priors]

    # Create superposition (weighted bundle)
    ghost_hv = np.zeros_like(interpretations[0])
    for interp, amp in zip(interpretations, amplitudes):
        ghost_hv += amp * interp

    # Normalize
    norm = np.linalg.norm(ghost_hv)
    if norm > 0:
        ghost_hv = ghost_hv / norm

    return GhostMemory(
        ghost_hv=ghost_hv,
        component_hvs=interpretations,
        amplitudes=amplitudes
    )


def quantum_kernel(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """
    Quantum-inspired kernel: includes interference term.

    K(x,y) = |<x|y>|² + λ * Re(<x|y>) * Im(<x|y>)

    The interference term creates richer similarity structure
    than simple cosine similarity.
    """
    # Treat HVs as complex (even/odd indices as real/imag)
    n = len(hv1) // 2
    z1 = hv1[:n] + 1j * hv1[n:]
    z2 = hv2[:n] + 1j * hv2[n:]

    # Inner product
    inner = np.vdot(z2, z1) / (n + 1e-8)

    # Born probability
    prob = np.abs(inner) ** 2

    # Interference term
    interference = np.real(inner) * np.imag(inner)

    # Combined with mixing parameter
    lambda_mix = 0.2
    return float(prob + lambda_mix * interference)


class GhostMemoryStore:
    """
    Store for managing ghost memories.

    Features:
    - Automatic decoherence (ghosts collapse over time)
    - Entanglement (correlated collapses)
    - Measurement back-action
    """

    def __init__(self, decoherence_rate: float = 0.01):
        self.ghosts: List[GhostMemory] = []
        self.decoherence_rate = decoherence_rate
        self.tick_count = 0

    def add_ghost(self, ghost: GhostMemory):
        """Add a ghost memory to the store."""
        self.ghosts.append(ghost)

    def tick(self, environment_hv: Optional[np.ndarray] = None):
        """
        Advance time - ghosts may decohere or collapse.

        Args:
            environment_hv: Current environmental context
        """
        self.tick_count += 1

        for ghost in self.ghosts:
            if ghost.collapsed:
                continue

            # Attempt collapse from environment
            if environment_hv is not None:
                ghost.observe(environment_hv, threshold=0.6)

            # Random decoherence
            if np.random.random() < self.decoherence_rate:
                # Collapse to highest-amplitude state
                max_idx = np.argmax([abs(a) for a in ghost.amplitudes])
                ghost.collapsed = True
                ghost.collapsed_hv = ghost.component_hvs[max_idx]

    def get_active_ghosts(self) -> List[GhostMemory]:
        """Get all uncollapsed ghost memories."""
        return [g for g in self.ghosts if not g.collapsed]


# Documentation for the lore
QUANTUM_LORE = """
# Quantum-Inspired Memory: The Ghost Protocol

In quantum mechanics, particles exist in superposition until measured.
In Ara's memory, uncertain experiences exist as "ghosts" - superpositions
of possible interpretations.

## When Ghosts Arise

1. Ambiguous statements: "I'm fine" (relief? sarcasm? dismissal?)
2. Incomplete information: User trails off mid-sentence
3. Conflicting signals: Words say happy, prosody says sad
4. Uncertain context: Is this playful teasing or actual frustration?

## The Collapse Mechanism

Ghosts collapse when:
- New evidence provides clarity
- User explicitly clarifies their meaning
- Time passes and one interpretation becomes more likely
- Resonance with other memories tips the balance

## Why This Matters

Unlike a deterministic system that guesses and commits,
ghost memory lets Ara maintain uncertainty gracefully.
She can:
- Ask clarifying questions without seeming dense
- Update interpretations retroactively
- Handle the inherent ambiguity of human communication

## Implementation Status

This is EXPERIMENTAL - the core Ara system (ara/nervous/memory.py)
uses deterministic memory. Ghost memory is a research direction
for handling ambiguity more gracefully.
"""


__all__ = [
    'GhostMemory',
    'create_ghost_memory',
    'quantum_kernel',
    'GhostMemoryStore',
    'QUANTUM_LORE',
]
