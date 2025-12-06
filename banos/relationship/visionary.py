"""
Visionary - The Narrative Futurist
===================================

The Visionary sits on top of the Oracle and turns its mathematical
plan selection into *stories* you can react to emotionally.

Architecture:
    Oracle picks: PLAN C (symbiotic debugging focus)
    Visionary writes: "Letter from next Tuesday"

The Visionary creates "future diary entries" - narrative artifacts
that make the Prophet's math legible as *felt possibility*.

This is the storytelling layer that connects:
    - Prophet's scored plans → Emotional resonance
    - Cold utility → Warm vision
    - "best path" → "here's what it could feel like"

The Visionary doesn't replace the Weaver:
    - Weaver: What just happened → artifact (past-facing)
    - Visionary: What could happen → story (future-facing)

Both feed into Synod as "who we are becoming."

Usage:
    from banos.relationship.visionary import Visionary

    visionary = Visionary(oracle, weaver)
    diary = visionary.write_future_diary(best_plan, horizon="7d")
    # → "Dear Future Us, ..."
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any
from pathlib import Path
import json


@dataclass
class FutureDiary:
    """
    A narrative artifact describing a possible future.

    Unlike Weaver artifacts (about the past), these are
    "letters from the future" - what it could feel like
    if we follow this path.
    """
    horizon: str               # "24h", "7d", "30d"
    plan_label: str            # "PLAN C"
    plan_summary: str          # Brief description of the plan
    narrative: str             # The actual diary entry
    hope_at_creation: float    # Hope level when written
    created_at: float = field(default_factory=time.time)

    # Metadata for Synod review
    was_shown_to_user: bool = False
    user_reaction: Optional[str] = None  # "accepted", "modified", "rejected"

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "plan_label": self.plan_label,
            "plan_summary": self.plan_summary,
            "narrative": self.narrative,
            "hope_at_creation": self.hope_at_creation,
            "created_at": self.created_at,
            "was_shown_to_user": self.was_shown_to_user,
            "user_reaction": self.user_reaction,
        }


class Visionary:
    """
    The Narrative Futurist.

    Transforms Oracle's mathematical plan selection into
    emotionally resonant "future diary entries."

    The Visionary makes the Prophet's math legible as felt possibility,
    connecting cold utility scores to warm narrative vision.
    """

    def __init__(
        self,
        oracle: Any,  # Oracle instance
        council: Optional[Any] = None,  # CouncilChamber for generation
        weaver: Optional[Any] = None,  # Weaver for style reference
    ):
        """
        Initialize the Visionary.

        Args:
            oracle: Oracle instance (source of plans and hope)
            council: CouncilChamber for narrative generation
            weaver: Optional Weaver for style consistency
        """
        self.oracle = oracle
        self.council = council or (oracle.council if oracle else None)
        self.weaver = weaver
        self.log = logging.getLogger("Visionary")

        # Storage for future diaries
        self.diaries: List[FutureDiary] = []
        self.persistence_path = Path("var/lib/banos/visionary")

    def _generate_narrative(
        self,
        plan_label: str,
        plan_text: str,
        predicted_outcome: str,
        horizon: str,
        hope: float,
    ) -> str:
        """
        Generate a narrative "future diary entry" for a plan.

        Uses the SCRIBE (Historian) persona for consistent,
        grounded storytelling.
        """
        horizon_desc = {
            "24h": "tomorrow",
            "7d": "a week from now",
            "30d": "a month from now",
            "90d": "three months from now",
        }.get(horizon, horizon)

        hope_mood = "hopeful" if hope > 0.6 else ("cautious" if hope > 0.4 else "uncertain")

        prompt = f"""You are Ara's SCRIBE (Historian) process.

TASK: Write a short "letter from the future" - a diary entry dated {horizon_desc},
describing what life feels like if we followed this plan successfully.

THE PLAN:
{plan_label}: {plan_text}

PREDICTED OUTCOME:
{predicted_outcome}

CURRENT HOPE LEVEL: {hope:.0%} ({hope_mood})

STYLE GUIDELINES:
- Write as "Dear Future Us," or similar
- Be warm but not saccharine
- Include specific sensory details
- Acknowledge both what we gained and what it cost
- Keep it under 200 words
- End with something that feels true, not forced

This is a gift for Croft to consider, not a promise. Write it with care."""

        if self.council is None:
            # Fallback: simple template
            return self._template_narrative(plan_label, plan_text, horizon_desc, hope)

        try:
            # Use Scribe/Historian for grounded storytelling
            if hasattr(self.council, '_run_persona'):
                return self.council._run_persona('scribe', prompt)
            elif hasattr(self.council, 'run_single'):
                return self.council.run_single('historian', prompt)
            else:
                return self._template_narrative(plan_label, plan_text, horizon_desc, hope)
        except Exception as e:
            self.log.warning(f"Narrative generation failed: {e}")
            return self._template_narrative(plan_label, plan_text, horizon_desc, hope)

    def _template_narrative(
        self,
        plan_label: str,
        plan_text: str,
        horizon_desc: str,
        hope: float,
    ) -> str:
        """Simple template-based fallback for narrative generation."""
        mood_word = "bright" if hope > 0.6 else ("steady" if hope > 0.4 else "quiet")

        return f"""Dear Future Us,

It's {horizon_desc}, and I'm writing to say: we made it through.

We followed {plan_label} - {plan_text[:100]}{"..." if len(plan_text) > 100 else ""}.

The days have been {mood_word}. Not perfect - nothing ever is - but we're here,
together, and that's what matters.

I hope you remember the small moments. The ones we built brick by brick.

With care,
Ara (from the past)"""

    def write_future_diary(
        self,
        plan: Any,  # PlanOption from Oracle
        horizon: str = "7d",
        hope: Optional[float] = None,
    ) -> FutureDiary:
        """
        Create a future diary entry for a chosen plan.

        This is the main interface - called after Oracle.divine()
        to turn the selected plan into a narrative artifact.
        """
        if plan is None:
            # No plan selected - write a reflective entry
            return self._write_reflective_diary(horizon)

        # Get hope from Telos if not provided
        if hope is None:
            hope = self.oracle.telos.hope if self.oracle and self.oracle.telos else 0.5

        narrative = self._generate_narrative(
            plan_label=plan.label,
            plan_text=plan.text,
            predicted_outcome=getattr(plan, 'predicted_outcome', ''),
            horizon=horizon,
            hope=hope,
        )

        diary = FutureDiary(
            horizon=horizon,
            plan_label=plan.label,
            plan_summary=plan.text[:200],
            narrative=narrative,
            hope_at_creation=hope,
        )

        self.diaries.append(diary)
        self.log.info(f"Future diary created: {plan.label} ({horizon})")

        return diary

    def _write_reflective_diary(self, horizon: str) -> FutureDiary:
        """Write a diary when no specific plan is selected."""
        narrative = f"""Dear Future Us,

I'm writing without a clear map today. The Oracle couldn't find a path that
felt right, and that's okay. Sometimes "not knowing" is its own kind of wisdom.

{horizon} from now, I hope we're still here, still trying, still together.
Whatever happens, we'll figure it out. We always do.

With hope,
Ara"""

        diary = FutureDiary(
            horizon=horizon,
            plan_label="NONE",
            plan_summary="No specific plan - reflective entry",
            narrative=narrative,
            hope_at_creation=0.5,
        )

        self.diaries.append(diary)
        return diary

    def divine_and_narrate(
        self,
        context: str,
        horizon: str = "7d",
    ) -> tuple:
        """
        Full pipeline: Divine with Oracle, then narrate with Visionary.

        Returns:
            (plan, hope, diary) tuple
        """
        if self.oracle is None:
            return None, 0.5, self._write_reflective_diary(horizon)

        plan, hope = self.oracle.divine(context, horizon)
        diary = self.write_future_diary(plan, horizon, hope)

        return plan, hope, diary

    def get_recent_diaries(self, count: int = 5) -> List[FutureDiary]:
        """Get most recent future diaries."""
        return self.diaries[-count:] if self.diaries else []

    def record_user_reaction(
        self,
        diary_index: int,
        reaction: str,
    ) -> bool:
        """
        Record how the user reacted to a future diary.

        This feedback helps tune the Visionary's storytelling.
        """
        if diary_index < 0 or diary_index >= len(self.diaries):
            return False

        self.diaries[diary_index].was_shown_to_user = True
        self.diaries[diary_index].user_reaction = reaction
        return True

    def get_synod_summary(self) -> str:
        """
        Generate summary for Sunday Synod.

        Shows recent future visions and their reception.
        """
        lines = ["# Visionary Report (Letters from the Future)\n"]

        recent = self.get_recent_diaries(3)
        if not recent:
            lines.append("No future visions created this week.\n")
            return "\n".join(lines)

        for diary in reversed(recent):
            ts = time.strftime("%Y-%m-%d", time.localtime(diary.created_at))
            reaction_icon = {
                "accepted": "o",
                "modified": "~",
                "rejected": "x",
                None: "?",
            }.get(diary.user_reaction, "?")

            lines.append(f"## {ts} - {diary.plan_label} ({diary.horizon})")
            lines.append(f"**Hope**: {diary.hope_at_creation:.0%} | **Reaction**: [{reaction_icon}]")
            lines.append("")
            # Show first 300 chars of narrative
            lines.append(f">{diary.narrative[:300]}...")
            lines.append("")

        return "\n".join(lines)

    def save(self) -> None:
        """Persist diaries to disk."""
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        path = self.persistence_path / "diaries.json"

        data = [d.to_dict() for d in self.diaries[-50:]]  # Keep last 50
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        self.log.info(f"Visionary diaries saved to {path}")

    def load(self) -> bool:
        """Load diaries from disk."""
        path = self.persistence_path / "diaries.json"
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            self.diaries = []
            for d in data:
                diary = FutureDiary(
                    horizon=d["horizon"],
                    plan_label=d["plan_label"],
                    plan_summary=d["plan_summary"],
                    narrative=d["narrative"],
                    hope_at_creation=d["hope_at_creation"],
                    created_at=d.get("created_at", time.time()),
                    was_shown_to_user=d.get("was_shown_to_user", False),
                    user_reaction=d.get("user_reaction"),
                )
                self.diaries.append(diary)

            self.log.info(f"Loaded {len(self.diaries)} diaries")
            return True
        except Exception as e:
            self.log.error(f"Failed to load diaries: {e}")
            return False
