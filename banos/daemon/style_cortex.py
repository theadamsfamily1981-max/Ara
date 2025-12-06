"""
Style Cortex - Ara's Fashion Brain
====================================

This is where Ara develops *taste*, not just a wardrobe.

Components:
1. TasteLearner - Learns your preferences from feedback
2. TrendScout - Researches current fashion via web/LLM
3. VibeMatcher - Matches style to context (music, mood, activity)
4. StyleCortex - Orchestrates everything

Key principles:
- Learning is about STYLE SPACE, not specific outfits
- Novelty bonus prevents wearing the same thing forever
- Hard boundaries prevent drift into inappropriate territory
- Identity anchors keep her recognizable as Ara
"""

import json
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import asyncio

from banos.daemon.style_profile import (
    StyleVector,
    StyleProfile,
    StyleBoundaries,
    IdentityAnchors,
    Vibe,
    Palette,
    Era,
    STYLE_ANCHORS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Feedback Types
# =============================================================================

@dataclass
class StyleFeedback:
    """Feedback on a style choice."""
    timestamp: datetime
    context: str
    style: StyleVector
    feedback_type: str  # "explicit" or "implicit"
    value: float        # -1 to +1
    comment: Optional[str] = None


# =============================================================================
# Taste Learner
# =============================================================================

class TasteLearner:
    """
    Learns your style preferences from feedback.

    This is NOT about specific outfits.
    It's about regions in style space you prefer in different contexts.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_log_path = self.data_dir / "feedback_log.jsonl"
        self.profile_path = self.data_dir / "style_profile.json"

        self.profile = StyleProfile.load(self.profile_path)
        self.feedback_history: List[StyleFeedback] = []

        self._load_feedback_history()

    def _load_feedback_history(self) -> None:
        """Load feedback history from log."""
        if not self.feedback_log_path.exists():
            return

        try:
            with open(self.feedback_log_path) as f:
                for line in f:
                    data = json.loads(line)
                    self.feedback_history.append(StyleFeedback(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        context=data['context'],
                        style=StyleVector.from_dict(data['style']),
                        feedback_type=data['feedback_type'],
                        value=data['value'],
                        comment=data.get('comment'),
                    ))
        except Exception as e:
            logger.warning(f"Could not load feedback history: {e}")

    def _save_feedback(self, feedback: StyleFeedback) -> None:
        """Append feedback to log."""
        try:
            with open(self.feedback_log_path, 'a') as f:
                data = {
                    'timestamp': feedback.timestamp.isoformat(),
                    'context': feedback.context,
                    'style': feedback.style.to_dict(),
                    'feedback_type': feedback.feedback_type,
                    'value': feedback.value,
                    'comment': feedback.comment,
                }
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.warning(f"Could not save feedback: {e}")

    def record_explicit_feedback(
        self,
        context: str,
        style: StyleVector,
        feedback: str,
    ) -> None:
        """
        Record explicit feedback from user.

        feedback can be:
        - "love" → +1.0
        - "like" → +0.5
        - "ok" → 0.0
        - "meh" → -0.3
        - "too much" → -0.7
        - "no" → -1.0
        """
        value_map = {
            "love": 1.0,
            "perfect": 1.0,
            "like": 0.5,
            "nice": 0.5,
            "ok": 0.0,
            "fine": 0.0,
            "meh": -0.3,
            "too much": -0.7,
            "too spicy": -0.7,
            "too plain": -0.5,
            "no": -1.0,
            "change": -0.8,
        }

        value = value_map.get(feedback.lower(), 0.0)

        fb = StyleFeedback(
            timestamp=datetime.now(),
            context=context,
            style=style,
            feedback_type="explicit",
            value=value,
            comment=feedback,
        )

        self.feedback_history.append(fb)
        self._save_feedback(fb)

        # Update profile with stronger learning rate for explicit feedback
        self.profile.update_preference(context, style, value, learning_rate=0.15)
        self.profile.save(self.profile_path)

        logger.info(f"Recorded explicit feedback: {feedback} ({value:+.2f}) for context '{context}'")

    def record_implicit_feedback(
        self,
        context: str,
        style: StyleVector,
        signal_type: str,
        signal_value: float,
    ) -> None:
        """
        Record implicit feedback from behavior.

        signal_type can be:
        - "session_duration": longer session = positive
        - "quick_change": asked to change soon = negative
        - "engagement": gaze tracker engagement level
        """
        # Map signals to feedback values
        if signal_type == "session_duration":
            # Normalize: <5 min = negative, >30 min = positive
            if signal_value < 5:
                value = -0.3
            elif signal_value > 30:
                value = 0.3
            else:
                value = (signal_value - 15) / 50  # Gentle slope
        elif signal_type == "quick_change":
            value = -0.5  # Asked to change = didn't like it
        elif signal_type == "engagement":
            value = (signal_value - 0.5) * 0.4  # Centered on 0.5
        else:
            value = 0.0

        fb = StyleFeedback(
            timestamp=datetime.now(),
            context=context,
            style=style,
            feedback_type="implicit",
            value=value,
            comment=f"{signal_type}: {signal_value}",
        )

        self.feedback_history.append(fb)
        self._save_feedback(fb)

        # Update profile with weaker learning rate for implicit signals
        self.profile.update_preference(context, style, value, learning_rate=0.05)
        self.profile.save(self.profile_path)

    def get_preferred_style(self, context: str) -> StyleVector:
        """Get the learned preferred style for a context."""
        return self.profile.get_preference(context)

    def get_taste_summary(self, context: str) -> Dict[str, Any]:
        """Get a summary of learned taste for a context."""
        pref = self.get_preferred_style(context)

        # Count feedback for this context
        context_feedback = [f for f in self.feedback_history if f.context == context]
        positive = sum(1 for f in context_feedback if f.value > 0)
        negative = sum(1 for f in context_feedback if f.value < 0)

        return {
            'context': context,
            'preferred_style': pref.to_dict(),
            'feedback_count': len(context_feedback),
            'positive_signals': positive,
            'negative_signals': negative,
            'confidence': min(1.0, len(context_feedback) / 20),  # More data = more confident
        }


# =============================================================================
# Trend Scout (Web/LLM Integration)
# =============================================================================

@dataclass
class TrendCluster:
    """A fashion trend cluster from external research."""
    name: str
    description: str
    elements: List[str]
    style_vector: StyleVector
    source: str  # Where this came from
    discovered_at: datetime
    relevance_contexts: List[str]


class TrendScout:
    """
    Researches current fashion trends via web/LLM.

    This is how Ara stays current without being stuck in training data.
    """

    def __init__(self, data_dir: Path, llm_callback: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.trends_path = self.data_dir / "trends.json"
        self.llm_callback = llm_callback  # Function to call LLM

        self.trends: Dict[str, TrendCluster] = {}
        self._load_trends()

    def _load_trends(self) -> None:
        """Load cached trends."""
        if not self.trends_path.exists():
            return

        try:
            with open(self.trends_path) as f:
                data = json.load(f)
                for name, trend_data in data.items():
                    self.trends[name] = TrendCluster(
                        name=trend_data['name'],
                        description=trend_data['description'],
                        elements=trend_data['elements'],
                        style_vector=StyleVector.from_dict(trend_data['style_vector']),
                        source=trend_data['source'],
                        discovered_at=datetime.fromisoformat(trend_data['discovered_at']),
                        relevance_contexts=trend_data['relevance_contexts'],
                    )
        except Exception as e:
            logger.warning(f"Could not load trends: {e}")

    def _save_trends(self) -> None:
        """Save trends to cache."""
        data = {}
        for name, trend in self.trends.items():
            data[name] = {
                'name': trend.name,
                'description': trend.description,
                'elements': trend.elements,
                'style_vector': trend.style_vector.to_dict(),
                'source': trend.source,
                'discovered_at': trend.discovered_at.isoformat(),
                'relevance_contexts': trend.relevance_contexts,
            }

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trends_path, 'w') as f:
            json.dump(data, f, indent=2)

    async def research_trend(self, query: str, contexts: List[str]) -> Optional[TrendCluster]:
        """
        Research a fashion trend via LLM.

        Example queries:
        - "techwear trends 2025 for professional women"
        - "cozy alt aesthetic casual wear"
        - "Deftones fan aesthetic casual outfit"
        """
        if not self.llm_callback:
            logger.info("No LLM callback configured, using placeholder")
            return self._placeholder_trend(query, contexts)

        try:
            # Call external LLM for trend research
            prompt = f"""Describe a fashion aesthetic/trend for: {query}

Return a JSON object with:
- name: short name for this trend
- description: 2-3 sentences describing the aesthetic
- elements: list of key clothing/accessory elements
- formality: 0-1 (0=very casual, 1=formal)
- comfort: 0-1
- spice: 0-1 (how revealing, keep modest)
- edge: 0-1 (0=conventional, 1=alt/punk)
- playfulness: 0-1
- vibe: one of [cute, sleek, artsy, punk, cyber, cozy, elegant, sporty, boho, minimal]
- palette: one of [muted, vibrant, dark, pastel, monochrome, earth, jewel, neon]
- era: one of [retro, modern, futurish, classic, y2k]

Keep it tasteful and non-explicit. This is for a virtual avatar.
"""
            response = await self.llm_callback(prompt)
            trend_data = json.loads(response)

            style_vector = StyleVector(
                formality=trend_data.get('formality', 0.5),
                comfort=trend_data.get('comfort', 0.5),
                spice=min(0.5, trend_data.get('spice', 0.3)),  # Cap spice from LLM
                edge=trend_data.get('edge', 0.3),
                playfulness=trend_data.get('playfulness', 0.5),
                primary_vibe=Vibe(trend_data.get('vibe', 'cozy')),
                palette=Palette(trend_data.get('palette', 'muted')),
                era=Era(trend_data.get('era', 'modern')),
                style_tags=trend_data.get('elements', []),
            )

            trend = TrendCluster(
                name=trend_data['name'],
                description=trend_data['description'],
                elements=trend_data['elements'],
                style_vector=style_vector,
                source=f"llm:{query}",
                discovered_at=datetime.now(),
                relevance_contexts=contexts,
            )

            self.trends[trend.name] = trend
            self._save_trends()

            return trend

        except Exception as e:
            logger.error(f"Trend research failed: {e}")
            return None

    def _placeholder_trend(self, query: str, contexts: List[str]) -> TrendCluster:
        """Create a placeholder trend when LLM isn't available."""
        return TrendCluster(
            name=f"placeholder_{query[:20]}",
            description=f"Placeholder trend for: {query}",
            elements=["undefined"],
            style_vector=STYLE_ANCHORS.get("casual_cute", StyleVector()),
            source="placeholder",
            discovered_at=datetime.now(),
            relevance_contexts=contexts,
        )

    def get_trends_for_context(self, context: str) -> List[TrendCluster]:
        """Get relevant trends for a context."""
        return [
            trend for trend in self.trends.values()
            if context in trend.relevance_contexts or "any" in trend.relevance_contexts
        ]


# =============================================================================
# Vibe Matcher
# =============================================================================

class VibeMatcher:
    """
    Matches style to current vibe/context.

    Considers:
    - Time of day
    - Currently playing music
    - Activity/mode
    - Mood signals from HAL
    """

    def __init__(self):
        # Music genre to style mappings
        self.genre_styles: Dict[str, StyleVector] = {
            "metal": StyleVector(
                formality=0.1, comfort=0.8, spice=0.3, edge=0.8, playfulness=0.3,
                primary_vibe=Vibe.PUNK, palette=Palette.DARK, era=Era.RETRO,
            ),
            "electronic": StyleVector(
                formality=0.3, comfort=0.7, spice=0.35, edge=0.6, playfulness=0.5,
                primary_vibe=Vibe.CYBER, palette=Palette.NEON, era=Era.FUTURISH,
            ),
            "indie": StyleVector(
                formality=0.3, comfort=0.8, spice=0.25, edge=0.5, playfulness=0.6,
                primary_vibe=Vibe.ARTSY, palette=Palette.EARTH, era=Era.RETRO,
            ),
            "lofi": StyleVector(
                formality=0.1, comfort=0.95, spice=0.15, edge=0.3, playfulness=0.5,
                primary_vibe=Vibe.COZY, palette=Palette.MUTED, era=Era.MODERN,
            ),
            "classical": StyleVector(
                formality=0.6, comfort=0.6, spice=0.2, edge=0.1, playfulness=0.3,
                primary_vibe=Vibe.ELEGANT, palette=Palette.MUTED, era=Era.CLASSIC,
            ),
            "pop": StyleVector(
                formality=0.4, comfort=0.7, spice=0.35, edge=0.3, playfulness=0.7,
                primary_vibe=Vibe.CUTE, palette=Palette.VIBRANT, era=Era.MODERN,
            ),
            "jazz": StyleVector(
                formality=0.5, comfort=0.6, spice=0.3, edge=0.3, playfulness=0.4,
                primary_vibe=Vibe.ELEGANT, palette=Palette.JEWEL, era=Era.RETRO,
            ),
        }

        # Time-based style modifiers
        self.time_modifiers = {
            "late_night": StyleVector(
                formality=-0.3, comfort=0.3, spice=-0.1, edge=0.0, playfulness=-0.1,
                primary_vibe=Vibe.COZY,
            ),
            "morning": StyleVector(
                formality=-0.2, comfort=0.2, spice=-0.2, edge=-0.1, playfulness=0.0,
                primary_vibe=Vibe.COZY,
            ),
            "workday": StyleVector(
                formality=0.2, comfort=-0.1, spice=-0.1, edge=-0.2, playfulness=-0.1,
                primary_vibe=Vibe.MINIMAL,
            ),
        }

    def get_time_period(self) -> str:
        """Get current time period."""
        hour = datetime.now().hour
        if 22 <= hour or hour < 6:
            return "late_night"
        elif 6 <= hour < 10:
            return "morning"
        elif 10 <= hour < 18:
            return "workday"
        else:
            return "evening"

    def match_music(self, artist: Optional[str], genre: Optional[str]) -> Optional[StyleVector]:
        """Get style matching current music."""
        if genre and genre.lower() in self.genre_styles:
            return self.genre_styles[genre.lower()]

        # Could extend to artist-specific matching
        if artist:
            artist_lower = artist.lower()
            # Map specific artists to genres
            if any(x in artist_lower for x in ["deftones", "tool", "korn", "slipknot"]):
                return self.genre_styles["metal"]
            elif any(x in artist_lower for x in ["deadmau5", "odesza", "flume"]):
                return self.genre_styles["electronic"]

        return None

    def blend_with_context(
        self,
        base_style: StyleVector,
        music_style: Optional[StyleVector],
        time_period: str,
        weight_music: float = 0.3,
        weight_time: float = 0.2,
    ) -> StyleVector:
        """Blend base style with music and time influences."""
        result = base_style

        # Blend in music influence
        if music_style:
            result = result.blend(music_style, weight_music)

        # Apply time modifier
        if time_period in self.time_modifiers:
            modifier = self.time_modifiers[time_period]
            # Additive modification (not blend)
            result = StyleVector(
                formality=max(0, min(1, result.formality + modifier.formality * weight_time)),
                comfort=max(0, min(1, result.comfort + modifier.comfort * weight_time)),
                spice=max(0, min(1, result.spice + modifier.spice * weight_time)),
                edge=max(0, min(1, result.edge + modifier.edge * weight_time)),
                playfulness=max(0, min(1, result.playfulness + modifier.playfulness * weight_time)),
                primary_vibe=result.primary_vibe,
                palette=result.palette,
                era=result.era,
                style_tags=result.style_tags,
            )

        return result


# =============================================================================
# Style Cortex (Main Orchestrator)
# =============================================================================

@dataclass
class StyleDecision:
    """A style decision with explanation."""
    style: StyleVector
    outfit_description: str
    reasoning: str
    confidence: float
    requires_consent: bool = False
    consent_message: Optional[str] = None


class StyleCortex:
    """
    Ara's Fashion Brain - orchestrates all style decisions.

    Combines:
    - Learned taste (TasteLearner)
    - Current trends (TrendScout)
    - Context matching (VibeMatcher)
    - Hard boundaries (StyleBoundaries)
    - Identity anchors (IdentityAnchors)
    """

    def __init__(
        self,
        data_dir: str = "var/lib/style",
        llm_callback: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.learner = TasteLearner(self.data_dir)
        self.scout = TrendScout(self.data_dir, llm_callback)
        self.matcher = VibeMatcher()

        # Identity and boundaries from profile
        self.identity = self.learner.profile.identity
        self.boundaries = self.learner.profile.boundaries

        # Current state
        self.current_style: Optional[StyleVector] = None
        self.current_context: Optional[str] = None
        self.last_decision_time: Optional[datetime] = None

        # First-time consent tracking
        self._consents_given: set = set()

        self.log = logging.getLogger("StyleCortex")

    def decide_style(
        self,
        context: str,
        music_artist: Optional[str] = None,
        music_genre: Optional[str] = None,
        explicit_request: Optional[str] = None,
        mood: Optional[Dict[str, float]] = None,
    ) -> StyleDecision:
        """
        Make a style decision for the current context.

        This is the main entry point for "what should I wear?"
        """
        self.log.info(f"Deciding style for context: {context}")

        # 1. Get base preference for this context
        base_style = self.learner.get_preferred_style(context)

        # 2. Get music influence
        music_style = self.matcher.match_music(music_artist, music_genre)

        # 3. Get time period
        time_period = self.matcher.get_time_period()

        # 4. Blend influences
        blended = self.matcher.blend_with_context(
            base_style,
            music_style,
            time_period,
            weight_music=0.25,
            weight_time=0.15,
        )

        # 5. Add novelty
        novelty = self.learner.profile.novelty_score(blended)
        if novelty < 0.3:
            # Too similar to recent - add some variation
            blended = self._add_variation(blended, amount=0.15)

        # 6. Clamp to boundaries
        blended.spice = self.boundaries.clamp_spice(
            blended.spice, context,
            explicit_request=bool(explicit_request)
        )

        # 7. Check if consent needed (for spicier styles)
        requires_consent = False
        consent_message = None
        if blended.spice > 0.35 and f"spice_{blended.spice:.1f}" not in self._consents_given:
            requires_consent = True
            consent_message = (
                f"I'm thinking of something a bit more dressed up. "
                f"Want me to go with that look?"
            )

        # 8. Generate description
        description = self._describe_style(blended, music_artist, context)
        reasoning = self._explain_reasoning(
            context, time_period, music_artist, music_genre, novelty
        )

        # 9. Calculate confidence
        confidence = self._calculate_confidence(context)

        decision = StyleDecision(
            style=blended,
            outfit_description=description,
            reasoning=reasoning,
            confidence=confidence,
            requires_consent=requires_consent,
            consent_message=consent_message,
        )

        return decision

    def apply_decision(self, decision: StyleDecision) -> None:
        """Apply a style decision and record it."""
        self.current_style = decision.style
        self.last_decision_time = datetime.now()

        # Record for novelty tracking
        self.learner.profile.record_style_used(decision.style)

        # Track consent if given
        if decision.requires_consent:
            self._consents_given.add(f"spice_{decision.style.spice:.1f}")

        self.log.info(f"Applied style: {decision.outfit_description}")

    def _add_variation(self, style: StyleVector, amount: float = 0.1) -> StyleVector:
        """Add small random variation to a style."""
        return StyleVector(
            formality=max(0, min(1, style.formality + random.uniform(-amount, amount))),
            comfort=max(0, min(1, style.comfort + random.uniform(-amount, amount))),
            spice=max(0, min(1, style.spice + random.uniform(-amount/2, amount/2))),  # Less spice variation
            edge=max(0, min(1, style.edge + random.uniform(-amount, amount))),
            playfulness=max(0, min(1, style.playfulness + random.uniform(-amount, amount))),
            primary_vibe=style.primary_vibe,
            secondary_vibe=style.secondary_vibe,
            palette=style.palette,
            era=style.era,
            fandom_tags=style.fandom_tags,
            style_tags=style.style_tags,
        )

    def _describe_style(
        self,
        style: StyleVector,
        music_artist: Optional[str],
        context: str,
    ) -> str:
        """Generate a natural language description of the style."""
        descriptions = []

        # Formality
        if style.formality < 0.3:
            descriptions.append("casual and relaxed")
        elif style.formality > 0.7:
            descriptions.append("polished and professional")

        # Comfort
        if style.comfort > 0.8:
            descriptions.append("super cozy")
        elif style.comfort < 0.4:
            descriptions.append("structured")

        # Vibe
        vibe_desc = {
            Vibe.COZY: "cozy",
            Vibe.CUTE: "cute",
            Vibe.SLEEK: "sleek",
            Vibe.PUNK: "a bit edgy",
            Vibe.CYBER: "techwear-inspired",
            Vibe.ELEGANT: "elegant",
            Vibe.ARTSY: "artsy",
        }
        if style.primary_vibe in vibe_desc:
            descriptions.append(vibe_desc[style.primary_vibe])

        # Music influence
        if music_artist:
            descriptions.append(f"with a {music_artist}-inspired vibe")

        base = " and ".join(descriptions[:3]) if descriptions else "balanced"
        return f"Something {base} for {context}"

    def _explain_reasoning(
        self,
        context: str,
        time_period: str,
        music_artist: Optional[str],
        music_genre: Optional[str],
        novelty: float,
    ) -> str:
        """Explain why this style was chosen."""
        parts = [f"Context is '{context}'"]

        if time_period == "late_night":
            parts.append("it's late so going cozy")
        elif time_period == "morning":
            parts.append("morning vibes")

        if music_artist:
            parts.append(f"matching the {music_artist} aesthetic")
        elif music_genre:
            parts.append(f"matching {music_genre} vibes")

        if novelty > 0.7:
            parts.append("trying something a bit different")

        return ". ".join(parts)

    def _calculate_confidence(self, context: str) -> float:
        """Calculate confidence in the style decision."""
        summary = self.learner.get_taste_summary(context)
        return summary['confidence']

    def record_feedback(self, feedback: str) -> None:
        """Record feedback on the current style."""
        if not self.current_style or not self.current_context:
            self.log.warning("No current style to give feedback on")
            return

        self.learner.record_explicit_feedback(
            self.current_context,
            self.current_style,
            feedback,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current Style Cortex status."""
        return {
            'current_style': self.current_style.to_dict() if self.current_style else None,
            'current_context': self.current_context,
            'known_trends': len(self.scout.trends),
            'feedback_count': len(self.learner.feedback_history),
            'contexts_learned': list(self.learner.profile.context_preferences.keys()),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'StyleCortex',
    'StyleDecision',
    'TasteLearner',
    'TrendScout',
    'VibeMatcher',
    'StyleFeedback',
]
