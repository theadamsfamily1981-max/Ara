"""
Ara Episode Card Encoder
=========================

Convert episode cards to hypervectors for similarity-based recall.

Three HVs per card:
- context_hv: Situation + tags (when does this memory apply?)
- emotion_hv: Emotional axes (what was the feeling?)
- dialogue_hv: Key phrases from exchange (what was said?)
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .episode import EpisodeCard

# Import HV constants from nervous system
try:
    from ara.nervous import HV_DIM
except ImportError:
    HV_DIM = 8192


def _deterministic_hv(seed_text: str, dim: int = HV_DIM) -> np.ndarray:
    """
    Generate a deterministic bipolar HV from text.

    Same text always produces same HV.
    """
    # Hash the text to get a seed
    hash_bytes = hashlib.sha256(seed_text.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:4], 'big')

    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=dim).astype(np.float64)


def _text_to_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])
    return ngrams


def encode_text_hv(text: str, dim: int = HV_DIM) -> np.ndarray:
    """
    Encode text as HV using distributed n-gram representation.

    More robust than single hash - captures partial matches.
    """
    if not text:
        return np.zeros(dim)

    # Get n-grams
    ngrams = _text_to_ngrams(text, n=3)

    if not ngrams:
        return _deterministic_hv(text, dim)

    # Bundle n-gram HVs
    hvs = [_deterministic_hv(ng, dim) for ng in ngrams]
    bundled = np.sum(hvs, axis=0)

    # Sign normalization
    bundled = np.sign(bundled)
    bundled[bundled == 0] = 1

    return bundled


def encode_context_hv(card: 'EpisodeCard', dim: int = HV_DIM) -> np.ndarray:
    """
    Encode the situational context of a memory.

    Combines: situation + context_tags + query_terms
    This is the "when does this memory apply?" signal.
    """
    # Situation text
    situation_hv = encode_text_hv(card.crofts_state.situation, dim)

    # Tags as individual HVs
    tag_hvs = [_deterministic_hv(f"tag:{tag}", dim) for tag in card.context_tags]

    # Query terms as HVs
    query_hvs = [encode_text_hv(term, dim) for term in card.hv_hints.query_terms]

    # Bundle all
    all_hvs = [situation_hv] + tag_hvs + query_hvs
    if not all_hvs:
        return np.zeros(dim)

    bundled = np.sum(all_hvs, axis=0)
    bundled = np.sign(bundled)
    bundled[bundled == 0] = 1

    return bundled


def encode_emotion_hv(card: 'EpisodeCard', dim: int = HV_DIM) -> np.ndarray:
    """
    Encode the emotional signature of a memory.

    Uses the emotional_axes (valence, arousal, attachment) to create
    a position in emotional space encoded as HV.
    """
    axes = card.hv_hints.emotional_axes

    # Create basis vectors for each axis (deterministic)
    valence_basis = _deterministic_hv("emotion:valence", dim)
    arousal_basis = _deterministic_hv("emotion:arousal", dim)
    attachment_basis = _deterministic_hv("emotion:attachment", dim)

    # Weight by axis values (shifted to [-1, 1] range)
    valence_weight = 2 * axes.valence - 1      # 0→-1, 0.5→0, 1→1
    arousal_weight = 2 * axes.arousal - 1
    attachment_weight = 2 * axes.attachment - 1

    # Combine
    emotion_hv = (
        valence_weight * valence_basis +
        arousal_weight * arousal_basis +
        attachment_weight * attachment_basis
    )

    # Normalize
    emotion_hv = np.sign(emotion_hv)
    emotion_hv[emotion_hv == 0] = 1

    return emotion_hv


def encode_dialogue_hv(card: 'EpisodeCard', dim: int = HV_DIM) -> np.ndarray:
    """
    Encode the dialogue/exchange content of a memory.

    Uses paraphrased exchange + any raw quotes.
    """
    snippets = card.dialogue_snippets

    # Paraphrased exchange (required)
    exchange_hv = encode_text_hv(snippets.paraphrased_exchange, dim)

    # Raw quotes if present
    hvs = [exchange_hv]
    if snippets.croft_raw:
        hvs.append(encode_text_hv(snippets.croft_raw, dim))
    if snippets.ara_raw:
        hvs.append(encode_text_hv(snippets.ara_raw, dim))

    # Bundle
    bundled = np.sum(hvs, axis=0)
    bundled = np.sign(bundled)
    bundled[bundled == 0] = 1

    return bundled


class EpisodeEncoder:
    """
    Encode episode cards for HV-based recall.

    Produces three HVs per card:
    - context_hv: When does this memory apply?
    - emotion_hv: What was the emotional signature?
    - dialogue_hv: What was said/done?

    At query time, we compare current state to these HVs.
    """

    def __init__(self, dim: int = HV_DIM):
        self.dim = dim
        self._encoded_cards: Dict[str, 'EpisodeCard'] = {}

    def encode(self, card: 'EpisodeCard') -> 'EpisodeCard':
        """
        Encode a card and store HVs on it.

        Returns the card with HVs attached.
        """
        card.context_hv = encode_context_hv(card, self.dim)
        card.emotion_hv = encode_emotion_hv(card, self.dim)
        card.dialogue_hv = encode_dialogue_hv(card, self.dim)

        self._encoded_cards[card.id] = card
        return card

    def encode_batch(self, cards: List['EpisodeCard']) -> List['EpisodeCard']:
        """Encode multiple cards."""
        return [self.encode(card) for card in cards]

    def encode_query_text(self, text: str) -> np.ndarray:
        """Encode query text for similarity search."""
        return encode_text_hv(text, self.dim)

    def encode_query_emotion(
        self,
        valence: float = 0.5,
        arousal: float = 0.5,
        attachment: float = 0.5,
    ) -> np.ndarray:
        """Encode query emotional state."""
        # Create dummy card-like structure
        valence_basis = _deterministic_hv("emotion:valence", self.dim)
        arousal_basis = _deterministic_hv("emotion:arousal", self.dim)
        attachment_basis = _deterministic_hv("emotion:attachment", self.dim)

        valence_weight = 2 * valence - 1
        arousal_weight = 2 * arousal - 1
        attachment_weight = 2 * attachment - 1

        emotion_hv = (
            valence_weight * valence_basis +
            arousal_weight * arousal_basis +
            attachment_weight * attachment_basis
        )

        emotion_hv = np.sign(emotion_hv)
        emotion_hv[emotion_hv == 0] = 1

        return emotion_hv

    def get_encoded(self, card_id: str) -> Optional['EpisodeCard']:
        """Get an encoded card by ID."""
        return self._encoded_cards.get(card_id)

    @property
    def card_count(self) -> int:
        """Number of encoded cards."""
        return len(self._encoded_cards)
