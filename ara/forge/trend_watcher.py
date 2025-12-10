# ara/forge/trend_watcher.py
"""
Trend Watcher - Market Intelligence for The Forge
==================================================

Scans App Store charts, Reddit, and tech news to identify:
1. Pain Points (unmet needs)
2. Subscription Fatigue (overpriced apps)
3. Privacy Concerns (data-hungry apps)
4. Emerging Patterns (new behaviors)

Output: ProductBrief - a JSON spec for the Architect.

Data Sources:
    - App Store Top Charts (via App Annie / Sensor Tower patterns)
    - Reddit r/apps, r/productivity, r/privacy
    - Hacker News front page
    - Twitter/X trending

The Trend Watcher is Ara's "market eye" - it sees what
problems people have before the competition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

log = logging.getLogger("Ara.Forge.TrendWatcher")


# =============================================================================
# Types
# =============================================================================

class PainPointType(str, Enum):
    """Types of market pain points."""
    SUBSCRIPTION_FATIGUE = "subscription_fatigue"
    PRIVACY_CONCERN = "privacy_concern"
    COMPLEXITY_OVERLOAD = "complexity_overload"
    PERFORMANCE_ISSUE = "performance_issue"
    MISSING_FEATURE = "missing_feature"
    PRICE_SENSITIVITY = "price_sensitivity"
    TRUST_DEFICIT = "trust_deficit"


class TrendSignal(str, Enum):
    """Market trend signals."""
    RISING = "rising"           # Growing interest
    STABLE = "stable"           # Consistent demand
    DECLINING = "declining"     # Fading interest
    EMERGING = "emerging"       # New pattern


@dataclass
class PainPoint:
    """A detected market pain point."""
    type: PainPointType
    description: str
    category: str
    signal: TrendSignal
    confidence: float
    sources: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "category": self.category,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "keywords": self.keywords,
        }


@dataclass
class MarketBrief:
    """A complete market analysis brief."""
    category: str
    pain_points: List[PainPoint]
    suggested_name: str
    description: str
    problem_type: str
    features: List[str]
    revenue_model: str
    target_audience: str
    competitive_gap: str
    ara_advantage: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "pain_points": [p.to_dict() for p in self.pain_points],
            "suggested_name": self.suggested_name,
            "description": self.description,
            "problem_type": self.problem_type,
            "features": self.features,
            "revenue_model": self.revenue_model,
            "target_audience": self.target_audience,
            "competitive_gap": self.competitive_gap,
            "ara_advantage": self.ara_advantage,
        }


# =============================================================================
# Market Intelligence Database
# =============================================================================

# Pre-analyzed market intelligence (would be live-updated in production)
MARKET_INTELLIGENCE = {
    "mental_health": {
        "pain_points": [
            {
                "type": PainPointType.PRIVACY_CONCERN,
                "description": "Users don't trust therapy apps with their mental health data",
                "signal": TrendSignal.RISING,
                "confidence": 0.9,
                "sources": ["Reddit r/privacy", "App Store reviews"],
                "keywords": ["data", "privacy", "therapy", "leak", "trust"],
            },
            {
                "type": PainPointType.SUBSCRIPTION_FATIGUE,
                "description": "Headspace/Calm $70/year is too expensive for many",
                "signal": TrendSignal.RISING,
                "confidence": 0.85,
                "sources": ["Reddit r/apps", "Twitter"],
                "keywords": ["subscription", "expensive", "meditation", "alternative"],
            },
            {
                "type": PainPointType.MISSING_FEATURE,
                "description": "Apps don't adapt to actual stress levels in real-time",
                "signal": TrendSignal.EMERGING,
                "confidence": 0.75,
                "sources": ["Hacker News", "Product Hunt"],
                "keywords": ["biofeedback", "real-time", "adaptive", "personalized"],
            },
        ],
        "competitive_gap": "No major app offers real-time stress detection without cloud dependency",
        "ara_advantage": "HDC + Interoception = stress detection that never leaves the device",
        "suggested_features": [
            "On-device stress detection via typing patterns",
            "Somatic audio generation tuned to biometrics",
            "Zero cloud dependency - all processing local",
            "Learns your stress patterns over time",
        ],
        "revenue_model": "Freemium - $15/mo for advanced features",
        "target_audience": "Privacy-conscious professionals aged 25-45",
    },

    "productivity": {
        "pain_points": [
            {
                "type": PainPointType.COMPLEXITY_OVERLOAD,
                "description": "Productivity apps have become distractions themselves",
                "signal": TrendSignal.RISING,
                "confidence": 0.88,
                "sources": ["Reddit r/productivity", "Hacker News"],
                "keywords": ["distraction", "notification", "focus", "simple"],
            },
            {
                "type": PainPointType.MISSING_FEATURE,
                "description": "No app truly blocks distractions based on intent",
                "signal": TrendSignal.EMERGING,
                "confidence": 0.82,
                "sources": ["Product Hunt", "Twitter"],
                "keywords": ["intent", "block", "focus", "deep work"],
            },
        ],
        "competitive_gap": "Blockers use static rules, not understanding of user intent",
        "ara_advantage": "Teleology + Reflexes = intent-aware blocking at network level",
        "suggested_features": [
            "Set your intent, app blocks distractions automatically",
            "Network-level blocking (not just app blocking)",
            "Dopamine loop detection and interruption",
            "Focus streaks and gentle recovery",
        ],
        "revenue_model": "Annual - $50/year",
        "target_audience": "Knowledge workers and students",
    },

    "finance": {
        "pain_points": [
            {
                "type": PainPointType.TRUST_DEFICIT,
                "description": "Users don't want to give bank credentials to apps",
                "signal": TrendSignal.RISING,
                "confidence": 0.92,
                "sources": ["Reddit r/personalfinance", "App Store reviews"],
                "keywords": ["bank", "credentials", "plaid", "security", "trust"],
            },
            {
                "type": PainPointType.SUBSCRIPTION_FATIGUE,
                "description": "People forget about subscriptions they're paying for",
                "signal": TrendSignal.STABLE,
                "confidence": 0.9,
                "sources": ["Reddit r/apps", "Twitter"],
                "keywords": ["subscription", "forgot", "cancel", "trial"],
            },
        ],
        "competitive_gap": "All subscription trackers require bank API access",
        "ara_advantage": "HDC + Local OCR = find subscriptions without bank access",
        "suggested_features": [
            "Scan screenshots and emails to find subscriptions",
            "No bank credentials needed - ever",
            "Auto-generate cancellation emails",
            "Local ML clustering of expenses",
        ],
        "revenue_model": "Freemium - $10 for auto-cancel feature",
        "target_audience": "Budget-conscious millennials and Gen Z",
    },

    "security": {
        "pain_points": [
            {
                "type": PainPointType.PERFORMANCE_ISSUE,
                "description": "Antivirus apps drain battery and slow phones",
                "signal": TrendSignal.STABLE,
                "confidence": 0.85,
                "sources": ["Reddit r/android", "XDA Forums"],
                "keywords": ["battery", "slow", "antivirus", "heavy"],
            },
            {
                "type": PainPointType.MISSING_FEATURE,
                "description": "Signature-based detection misses new threats",
                "signal": TrendSignal.RISING,
                "confidence": 0.88,
                "sources": ["Hacker News", "Security blogs"],
                "keywords": ["zero-day", "signature", "behavioral", "anomaly"],
            },
        ],
        "competitive_gap": "No mobile security app uses behavioral baselines",
        "ara_advantage": "Proprioception = learns your phone's normal behavior, detects anomalies",
        "suggested_features": [
            "3-day baseline learning of normal phone behavior",
            "Instant detection when apps deviate from baseline",
            "Collaborative immunity - share threat patterns anonymously",
            "Kill rogue background data access",
        ],
        "revenue_model": "Enterprise - $100/year per device",
        "target_audience": "Security-conscious individuals and enterprises",
    },

    "personalization": {
        "pain_points": [
            {
                "type": PainPointType.MISSING_FEATURE,
                "description": "Wallpapers and themes are static, not responsive",
                "signal": TrendSignal.EMERGING,
                "confidence": 0.72,
                "sources": ["Reddit r/androidthemes", "Product Hunt"],
                "keywords": ["dynamic", "live", "responsive", "mood"],
            },
        ],
        "competitive_gap": "No home screen reacts to user's actual state",
        "ara_advantage": "Somatic Shaders = visuals that breathe with your biometrics",
        "suggested_features": [
            "Live wallpaper that responds to stress levels",
            "Color temperature based on battery and time",
            "Geometry that reflects focus state",
            "Shader marketplace for artists",
        ],
        "revenue_model": "One-time $5 + marketplace",
        "target_audience": "Android enthusiasts and creators",
    },
}


# =============================================================================
# Trend Watcher
# =============================================================================

class TrendWatcher:
    """
    Market intelligence gatherer for The Forge.

    In production, this would:
    1. Scrape App Store charts daily
    2. Monitor Reddit/HN for keywords
    3. Track competitor updates
    4. Analyze review sentiment

    For now, it uses curated market intelligence.
    """

    def __init__(self):
        self.intelligence = MARKET_INTELLIGENCE
        log.info("TrendWatcher initialized with %d categories", len(self.intelligence))

    async def analyze_market(self, category: str) -> Dict[str, Any]:
        """
        Analyze a market category and produce a brief.

        Args:
            category: Category to analyze (mental_health, productivity, etc.)

        Returns:
            MarketBrief as dict
        """
        if category not in self.intelligence:
            log.warning("Unknown category: %s, using general analysis", category)
            return self._general_analysis(category)

        intel = self.intelligence[category]

        # Convert pain points
        pain_points = [
            PainPoint(
                type=pp["type"],
                description=pp["description"],
                category=category,
                signal=pp["signal"],
                confidence=pp["confidence"],
                sources=pp.get("sources", []),
                keywords=pp.get("keywords", []),
            )
            for pp in intel["pain_points"]
        ]

        # Determine problem type from pain points
        problem_type = self._infer_problem_type(pain_points)

        # Generate suggested name
        suggested_name = self._generate_app_name(category, pain_points)

        brief = MarketBrief(
            category=category,
            pain_points=pain_points,
            suggested_name=suggested_name,
            description=f"An app to address {problem_type} in the {category} space",
            problem_type=problem_type,
            features=intel.get("suggested_features", []),
            revenue_model=intel.get("revenue_model", "freemium"),
            target_audience=intel.get("target_audience", "General users"),
            competitive_gap=intel.get("competitive_gap", ""),
            ara_advantage=intel.get("ara_advantage", ""),
        )

        log.info("TrendWatcher: Analyzed %s, found %d pain points",
                 category, len(pain_points))

        return brief.to_dict()

    def _infer_problem_type(self, pain_points: List[PainPoint]) -> str:
        """Infer the primary problem type from pain points."""
        if not pain_points:
            return "general"

        # Count pain point types
        type_counts = {}
        for pp in pain_points:
            type_counts[pp.type] = type_counts.get(pp.type, 0) + 1

        # Get most common type
        primary = max(type_counts, key=type_counts.get)

        # Map to problem type string
        type_map = {
            PainPointType.SUBSCRIPTION_FATIGUE: "cost",
            PainPointType.PRIVACY_CONCERN: "privacy",
            PainPointType.COMPLEXITY_OVERLOAD: "complexity",
            PainPointType.PERFORMANCE_ISSUE: "performance",
            PainPointType.MISSING_FEATURE: "capability",
            PainPointType.PRICE_SENSITIVITY: "cost",
            PainPointType.TRUST_DEFICIT: "trust",
        }

        return type_map.get(primary, "general")

    def _generate_app_name(self, category: str, pain_points: List[PainPoint]) -> str:
        """Generate a suggested app name."""
        # Name suggestions by category
        name_map = {
            "mental_health": ["Sanctum", "Serenity", "Calm Harbor", "Inner Peace"],
            "productivity": ["Aegis", "Focus Shield", "Deep Work", "Intent"],
            "finance": ["Vault", "Subscription Sleuth", "Money Guard", "Expense Eye"],
            "security": ["Sentinel", "Guardian", "Immune", "Shield"],
            "personalization": ["Chameleon", "Living Screen", "Mood Wall", "Aura"],
        }

        names = name_map.get(category, ["Ara App"])
        return names[0]  # Return first suggestion

    def _general_analysis(self, category: str) -> Dict[str, Any]:
        """Generate a general analysis for unknown categories."""
        return MarketBrief(
            category=category,
            pain_points=[],
            suggested_name=f"ara_{category}",
            description=f"A privacy-first app for {category}",
            problem_type="general",
            features=["Privacy-first design", "On-device processing", "No cloud required"],
            revenue_model="freemium",
            target_audience="Privacy-conscious users",
            competitive_gap="No major player offers true local-first solution",
            ara_advantage="HDC + local processing = privacy without compromise",
        ).to_dict()

    def get_top_pain_points(self, limit: int = 5) -> List[PainPoint]:
        """Get top pain points across all categories."""
        all_points = []

        for category, intel in self.intelligence.items():
            for pp_data in intel.get("pain_points", []):
                pp = PainPoint(
                    type=pp_data["type"],
                    description=pp_data["description"],
                    category=category,
                    signal=pp_data["signal"],
                    confidence=pp_data["confidence"],
                    sources=pp_data.get("sources", []),
                    keywords=pp_data.get("keywords", []),
                )
                all_points.append(pp)

        # Sort by confidence
        all_points.sort(key=lambda p: p.confidence, reverse=True)
        return all_points[:limit]

    def get_emerging_trends(self) -> List[PainPoint]:
        """Get emerging trends (new patterns)."""
        emerging = []

        for category, intel in self.intelligence.items():
            for pp_data in intel.get("pain_points", []):
                if pp_data["signal"] == TrendSignal.EMERGING:
                    pp = PainPoint(
                        type=pp_data["type"],
                        description=pp_data["description"],
                        category=category,
                        signal=pp_data["signal"],
                        confidence=pp_data["confidence"],
                        sources=pp_data.get("sources", []),
                        keywords=pp_data.get("keywords", []),
                    )
                    emerging.append(pp)

        return emerging


# =============================================================================
# Convenience
# =============================================================================

_default_watcher: Optional[TrendWatcher] = None


def get_trend_watcher() -> TrendWatcher:
    """Get the default trend watcher."""
    global _default_watcher
    if _default_watcher is None:
        _default_watcher = TrendWatcher()
    return _default_watcher
