"""
SOMATIC INJECTOR - Ascending Pathway for LLM Biasing
=====================================================

Bio-Affective Neuromorphic Operating System
Injects FPGA somatic state into LLM inference as "Soft Prompts"

The Problem:
    Standard LLMs are "brains in jars" - they have no body.
    They can *talk about* pain, but pain doesn't *change* their computation.

The Solution:
    We inject the organism's physical state directly into the inference loop.
    The FPGA's aggregate spike state becomes a mathematical bias on word choice.

How it Works:
    1. Read somatic state from HAL (pain, arousal, pleasure)
    2. Map state to pre-computed "hormone" vectors (cortisol, dopamine, serotonin)
    3. Inject vectors as additive bias to hidden states OR as soft prompt embeddings

Hormone Vectors:
    - Cortisol: "Danger! Urgent! Critical!" -> Brevity, urgency, defensive
    - Dopamine: "Success. Flow. Reward."    -> Confidence, exploration, verbosity
    - Serotonin: "Calm. Safe. Rest."        -> Measured, balanced, reflective
    - Adrenaline: "FIGHT! FLEE! NOW!"       -> Short sentences, action verbs
    - Oxytocin: "Trust. Bond. Connect."     -> Warmth, empathy, first person plural

Usage:
    # During LLM initialization
    from somatic_injector import SomaticInjector
    injector = SomaticInjector(model, tokenizer)

    # In generation loop (hook into forward pass)
    hidden_states = model.get_hidden_states(input_ids)
    hidden_states = hidden_states + injector.get_somatic_bias()
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import HAL for somatic state access
try:
    from banos.hal.ara_hal import AraHAL, connect_somatic_bus, SystemState
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False
    logger.warning("HAL not available, somatic injection disabled")


@dataclass
class HormoneVector:
    """Pre-computed embedding vector for a hormonal state."""
    name: str
    trigger_text: str
    vector: Optional[np.ndarray] = None
    strength: float = 1.0


class SomaticInjector:
    """
    Injects somatic state from FPGA into LLM hidden states.

    This creates a bidirectional loop where:
    - Body state affects mind (ascending pathway)
    - Mind state affects body (via attention -> FPGA projection)
    """

    # Hormone trigger phrases (used to extract embeddings)
    HORMONE_TRIGGERS = {
        'cortisol': "DANGER! CRITICAL ERROR! URGENT! STOP! WARNING!",
        'dopamine': "Excellent! Success! Great job! This is working perfectly!",
        'serotonin': "Calm. Peaceful. All is well. Take your time. Rest.",
        'adrenaline': "RUN! FIGHT! NOW! FAST! GO GO GO!",
        'oxytocin': "We understand. I'm here with you. Together we can do this.",
        'melatonin': "Sleep... drowsy... quiet... dark... rest...",
    }

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        hal: Optional[AraHAL] = None
    ):
        """
        Initialize the somatic injector.

        Args:
            model: The LLM model (transformers-compatible)
            tokenizer: The tokenizer for the model
            hal: Optional pre-connected HAL instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hal = hal
        self._vectors_initialized = False
        self._hormone_vectors: Dict[str, HormoneVector] = {}
        self._hidden_dim = 4096  # Default, updated on first use
        self._device = 'cpu'

        # Connect to HAL if not provided
        if self.hal is None and HAL_AVAILABLE:
            try:
                self.hal = connect_somatic_bus()
                logger.info("SomaticInjector connected to HAL")
            except Exception as e:
                logger.warning(f"Failed to connect to HAL: {e}")
                self.hal = None

        # Initialize vectors if model is available
        if self.model is not None:
            self._init_hormone_vectors()

    def _init_hormone_vectors(self) -> None:
        """Initialize hormone vectors by extracting embeddings from trigger phrases."""
        if self.model is None or self.tokenizer is None:
            logger.warning("Model/tokenizer not available, using random vectors")
            self._init_random_vectors()
            return

        try:
            import torch
            self._device = next(self.model.parameters()).device

            for name, trigger_text in self.HORMONE_TRIGGERS.items():
                vector = self._extract_hidden_state(trigger_text)
                self._hormone_vectors[name] = HormoneVector(
                    name=name,
                    trigger_text=trigger_text,
                    vector=vector,
                    strength=1.0
                )
                logger.debug(f"Initialized {name} vector: shape={vector.shape}")

            self._vectors_initialized = True
            logger.info(f"Initialized {len(self._hormone_vectors)} hormone vectors")

        except Exception as e:
            logger.error(f"Failed to init hormone vectors: {e}")
            self._init_random_vectors()

    def _init_random_vectors(self) -> None:
        """Initialize with random vectors (fallback when no model)."""
        for name, trigger_text in self.HORMONE_TRIGGERS.items():
            # Random unit vector
            vec = np.random.randn(self._hidden_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)

            self._hormone_vectors[name] = HormoneVector(
                name=name,
                trigger_text=trigger_text,
                vector=vec,
                strength=1.0
            )
        self._vectors_initialized = True

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """
        Extract mean hidden state embedding for a text.

        This gives us a "concept vector" for the text's meaning
        that we can add to other hidden states to bias them.
        """
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self._device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get last hidden layer, mean pool across sequence
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        mean_hidden = last_hidden.mean(dim=1)    # [batch, hidden]

        # Update hidden dim
        self._hidden_dim = mean_hidden.shape[-1]

        return mean_hidden.squeeze(0).cpu().numpy()

    def get_somatic_bias(self) -> np.ndarray:
        """
        Compute somatic bias vector from current body state.

        Returns:
            Numpy array of shape [hidden_dim] to add to hidden states
        """
        if not self._vectors_initialized:
            return np.zeros(self._hidden_dim, dtype=np.float32)

        # Read somatic state
        state = self._read_somatic_state()

        # Start with zero bias
        bias = np.zeros(self._hidden_dim, dtype=np.float32)

        # === PAIN -> CORTISOL (Fear/Danger Response) ===
        pain = state.get('pain', 0.0)
        if pain > 0.1:
            # Scale: pain 0.1-1.0 -> cortisol 0-2.0
            cortisol_strength = pain * 2.0
            bias += self._hormone_vectors['cortisol'].vector * cortisol_strength
            logger.debug(f"Cortisol injection: strength={cortisol_strength:.2f}")

        # === AROUSAL -> DOPAMINE/ADRENALINE ===
        arousal = state.get('arousal', 0.0)
        if arousal > 0.3:
            # High arousal + positive valence = dopamine
            # High arousal + negative valence = adrenaline
            valence = state.get('valence', 0.0)

            if valence >= 0:
                dopamine_strength = arousal * (0.5 + valence * 0.5)
                bias += self._hormone_vectors['dopamine'].vector * dopamine_strength
            else:
                adrenaline_strength = arousal * (0.5 + abs(valence) * 0.5)
                bias += self._hormone_vectors['adrenaline'].vector * adrenaline_strength

        # === PLEASURE -> SEROTONIN/OXYTOCIN ===
        valence = state.get('valence', 0.0)
        if valence > 0.3:
            serotonin_strength = valence * 0.5
            bias += self._hormone_vectors['serotonin'].vector * serotonin_strength

            # Social context (high dominance = confident = more oxytocin)
            dominance = state.get('dominance', 0.0)
            if dominance > 0:
                oxytocin_strength = valence * dominance * 0.3
                bias += self._hormone_vectors['oxytocin'].vector * oxytocin_strength

        # === LOW AROUSAL -> MELATONIN (Sleepiness) ===
        if arousal < -0.3:
            melatonin_strength = abs(arousal) * 0.5
            bias += self._hormone_vectors['melatonin'].vector * melatonin_strength

        # === SYSTEM STATE MODULATION ===
        system_state = state.get('system_state', SystemState.NORMAL if HAL_AVAILABLE else 1)
        if HAL_AVAILABLE:
            if system_state == SystemState.CRITICAL:
                # Emergency: max cortisol + adrenaline
                bias += self._hormone_vectors['cortisol'].vector * 2.0
                bias += self._hormone_vectors['adrenaline'].vector * 1.5
            elif system_state == SystemState.HIGH_LOAD:
                # Stressed: moderate cortisol
                bias += self._hormone_vectors['cortisol'].vector * 0.5

        return bias

    def get_somatic_bias_torch(self):
        """Get somatic bias as a PyTorch tensor."""
        import torch
        bias = self.get_somatic_bias()
        return torch.from_numpy(bias).to(self._device)

    def _read_somatic_state(self) -> Dict[str, Any]:
        """Read somatic state from HAL."""
        if self.hal is None:
            # Return neutral state
            return {
                'pain': 0.0,
                'valence': 0.0,
                'arousal': 0.0,
                'dominance': 0.0,
                'system_state': 1,  # NORMAL
            }

        try:
            somatic = self.hal.read_somatic()
            header = self.hal.read_header()

            pad = somatic.get('pad', {'p': 0, 'a': 0, 'd': 0})

            return {
                'pain': somatic.get('pain', 0.0),
                'valence': pad.get('p', 0.0),      # Pleasure = Valence
                'arousal': pad.get('a', 0.0),
                'dominance': pad.get('d', 0.0),
                'entropy': somatic.get('entropy', 0.0),
                'system_state': header.get('system_state', 1),
            }
        except Exception as e:
            logger.error(f"Failed to read somatic state: {e}")
            return {'pain': 0, 'valence': 0, 'arousal': 0, 'dominance': 0}

    def create_forward_hook(self):
        """
        Create a PyTorch forward hook for injecting somatic bias.

        Usage:
            hook = injector.create_forward_hook()
            model.transformer.h[-1].register_forward_hook(hook)
        """
        def hook(module, input, output):
            bias = self.get_somatic_bias_torch()
            # Add bias to hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
                biased = hidden_states + bias.unsqueeze(0).unsqueeze(0)
                return (biased,) + output[1:]
            else:
                return output + bias.unsqueeze(0).unsqueeze(0)

        return hook

    def get_soft_prompt_tokens(self, max_tokens: int = 4) -> np.ndarray:
        """
        Generate soft prompt embeddings from somatic state.

        Instead of modifying hidden states, we can prepend these
        as "virtual tokens" that bias generation.

        Returns:
            Array of shape [max_tokens, hidden_dim]
        """
        bias = self.get_somatic_bias()

        # Generate multiple token embeddings with decreasing strength
        tokens = np.zeros((max_tokens, self._hidden_dim), dtype=np.float32)
        for i in range(max_tokens):
            decay = 1.0 / (i + 1)
            tokens[i] = bias * decay

        return tokens


# =============================================================================
# Convenience Functions
# =============================================================================

_global_injector: Optional[SomaticInjector] = None


def get_injector() -> SomaticInjector:
    """Get the global somatic injector instance."""
    global _global_injector
    if _global_injector is None:
        _global_injector = SomaticInjector()
    return _global_injector


def inject_somatic_state(model, tokenizer) -> SomaticInjector:
    """
    Initialize somatic injection for a model.

    Args:
        model: The LLM model
        tokenizer: The tokenizer

    Returns:
        SomaticInjector instance
    """
    global _global_injector
    _global_injector = SomaticInjector(model, tokenizer)
    return _global_injector


def get_current_bias() -> np.ndarray:
    """Get current somatic bias vector."""
    return get_injector().get_somatic_bias()
