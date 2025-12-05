"""Auto-Reflection - Ara's self-critique after each interaction.

After each multi-teacher workflow, Ara generates a 1-2 line internal summary:
"I chose Claude first because of prior success on Python. Gemini added little
value this time. Hypothesis: skip Gemini next time for this pattern."

These reflections are logged with the JSON and can be clustered later.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .schemas import InteractionRecord, PatternCard


# =============================================================================
# Reflection Templates
# =============================================================================

_CHOICE_REASONS = [
    "I chose {teacher} first because {reason}.",
    "Started with {teacher} due to {reason}.",
    "{teacher} was my lead because {reason}.",
    "My go-to for this was {teacher} ({reason}).",
]

_TEACHER_PERFORMANCE = [
    "{teacher} {performance}.",
    "{teacher} {performance} on this one.",
    "Observation: {teacher} {performance}.",
]

_HYPOTHESIS_TEMPLATES = [
    "Hypothesis: {hypothesis}",
    "Next time: {hypothesis}",
    "Note to self: {hypothesis}",
    "Theory: {hypothesis}",
]

_PERFORMANCE_PHRASES = {
    "excellent": ["nailed it", "crushed it", "was spot-on", "delivered exactly what I needed"],
    "good": ["did well", "was solid", "performed well", "was helpful"],
    "neutral": ["was okay", "did the job", "was adequate", "contributed"],
    "poor": ["struggled", "missed the mark", "underperformed", "added little value"],
    "fail": ["failed completely", "gave bad advice", "was way off", "made things worse"],
}

_CHOICE_REASONS_BY_CONTEXT = {
    "debug_code": [
        "prior success on similar bugs",
        "strong debugging track record",
        "good at async/concurrency issues",
        "reliable for tricky type errors",
    ],
    "design_arch": [
        "systems thinking strength",
        "good at trade-off analysis",
        "prior architecture wins",
        "strong conceptual modeling",
    ],
    "research": [
        "broad knowledge coverage",
        "good at exploring options",
        "creative associations",
        "past research successes",
    ],
    "refactor": [
        "clean code expertise",
        "pattern recognition",
        "simplification skills",
        "prior refactoring wins",
    ],
    "default": [
        "general reliability",
        "past success on similar tasks",
        "strong track record",
        "appropriate expertise",
    ],
}


class AutoReflector:
    """Generates automatic self-reflections after interactions.

    These reflections help Ara learn from her own decisions.
    """

    def __init__(self):
        """Initialize the auto-reflector."""
        pass

    def generate_reflection(
        self,
        record: InteractionRecord,
        teacher_scores: Optional[Dict[str, float]] = None,
        pattern_card: Optional[PatternCard] = None,
    ) -> str:
        """Generate an automatic reflection for an interaction.

        Args:
            record: The interaction record
            teacher_scores: Per-teacher quality scores (optional)
            pattern_card: The pattern card used (optional)

        Returns:
            A 1-3 sentence reflection
        """
        parts = []

        # Part 1: Why I chose the primary teacher
        if record.primary_teacher:
            choice_part = self._generate_choice_reason(
                record.primary_teacher,
                record.user_intent or "default",
            )
            parts.append(choice_part)

        # Part 2: How each teacher performed
        if teacher_scores:
            for teacher, score in teacher_scores.items():
                perf_part = self._generate_performance_note(teacher, score)
                parts.append(perf_part)
        elif record.outcome_quality is not None:
            # Generic performance note
            perf = self._score_to_performance(record.outcome_quality)
            if record.teachers:
                main_teacher = record.teachers[0]
                parts.append(self._generate_performance_note(main_teacher, record.outcome_quality))

        # Part 3: Hypothesis for next time
        hypothesis = self._generate_hypothesis(record, teacher_scores)
        if hypothesis:
            parts.append(hypothesis)

        return " ".join(parts)

    def _generate_choice_reason(self, teacher: str, intent: str) -> str:
        """Generate a reason for choosing a teacher."""
        reasons = _CHOICE_REASONS_BY_CONTEXT.get(intent, _CHOICE_REASONS_BY_CONTEXT["default"])
        reason = random.choice(reasons)
        template = random.choice(_CHOICE_REASONS)
        return template.format(teacher=teacher, reason=reason)

    def _generate_performance_note(self, teacher: str, score: float) -> str:
        """Generate a performance note for a teacher."""
        perf = self._score_to_performance(score)
        phrases = _PERFORMANCE_PHRASES.get(perf, _PERFORMANCE_PHRASES["neutral"])
        phrase = random.choice(phrases)
        template = random.choice(_TEACHER_PERFORMANCE)
        return template.format(teacher=teacher, performance=phrase)

    def _score_to_performance(self, score: float) -> str:
        """Convert a score to a performance level."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "neutral"
        elif score >= 0.3:
            return "poor"
        else:
            return "fail"

    def _generate_hypothesis(
        self,
        record: InteractionRecord,
        teacher_scores: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Generate a hypothesis for next time."""
        hypotheses = []

        # Hypothesis based on poor secondary teacher
        if teacher_scores and len(record.teachers) > 1:
            for teacher in record.teachers[1:]:
                if teacher in teacher_scores and teacher_scores[teacher] < 0.5:
                    hypotheses.append(f"skip {teacher} next time for this pattern")

        # Hypothesis based on backtracks
        if record.backtrack_count > 0:
            hypotheses.append(f"try more specific initial prompt to reduce backtracks")

        # Hypothesis based on latency
        if record.latency_sec and record.latency_sec > 30:
            hypotheses.append(f"consider faster single-teacher approach for speed")

        # Hypothesis based on turns
        if record.turns_to_solution and record.turns_to_solution > 3:
            hypotheses.append(f"front-load more context to reduce turns")

        # Hypothesis based on overall success
        if record.outcome_quality is not None:
            if record.outcome_quality >= 0.9:
                hypotheses.append(f"this pattern works well, keep using it")
            elif record.outcome_quality < 0.5:
                hypotheses.append(f"try a different teacher combination")

        if hypotheses:
            hypothesis = random.choice(hypotheses)
            template = random.choice(_HYPOTHESIS_TEMPLATES)
            return template.format(hypothesis=hypothesis)

        return None

    def generate_session_summary(
        self,
        records: List[InteractionRecord],
    ) -> str:
        """Generate a summary reflection for a whole session.

        Args:
            records: All interactions in the session

        Returns:
            Session summary reflection
        """
        if not records:
            return "No interactions to summarize."

        # Aggregate stats
        total = len(records)
        successes = sum(1 for r in records if r.outcome_quality and r.outcome_quality >= 0.7)
        avg_quality = sum(r.outcome_quality or 0 for r in records) / total if total else 0

        # Teacher usage
        teacher_counts: Dict[str, int] = {}
        for r in records:
            for t in r.teachers:
                teacher_counts[t] = teacher_counts.get(t, 0) + 1

        # Pattern usage
        pattern_counts: Dict[str, int] = {}
        for r in records:
            if r.pattern_id:
                pattern_counts[r.pattern_id] = pattern_counts.get(r.pattern_id, 0) + 1

        # Generate summary
        parts = []
        parts.append(f"Session summary: {successes}/{total} interactions succeeded (avg quality: {avg_quality:.0%}).")

        if teacher_counts:
            top_teacher = max(teacher_counts, key=lambda k: teacher_counts[k])
            parts.append(f"Most used teacher: {top_teacher} ({teacher_counts[top_teacher]} calls).")

        if pattern_counts:
            top_pattern = max(pattern_counts, key=lambda k: pattern_counts[k])
            parts.append(f"Most used pattern: {top_pattern}.")

        return " ".join(parts)


# =============================================================================
# Intent Classification
# =============================================================================

INTENT_KEYWORDS = {
    "debug_code": ["debug", "error", "bug", "fix", "traceback", "exception", "crash", "TypeError", "ValueError"],
    "design_arch": ["design", "architecture", "system", "structure", "component", "module", "interface"],
    "research": ["research", "explore", "investigate", "understand", "learn", "explain", "what is"],
    "refactor": ["refactor", "clean", "simplify", "improve", "organize", "restructure"],
    "implement": ["implement", "create", "build", "add", "write", "code", "function", "class"],
    "review": ["review", "check", "verify", "validate", "test", "assess"],
    "optimize": ["optimize", "performance", "speed", "memory", "efficient", "fast"],
}


def classify_intent(query: str) -> str:
    """Classify user intent from query text.

    Args:
        query: The user's query

    Returns:
        Intent classification
    """
    query_lower = query.lower()

    # Score each intent
    scores: Dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in query_lower)
        if score > 0:
            scores[intent] = score

    if scores:
        return max(scores, key=lambda k: scores[k])

    return "general"


def detect_issues(record: InteractionRecord) -> List[str]:
    """Auto-detect issues with an interaction.

    Args:
        record: The interaction record

    Returns:
        List of detected issues
    """
    issues = []

    # Latency issues
    if record.latency_sec:
        if record.latency_sec > 60:
            issues.append("very_long_latency")
        elif record.latency_sec > 30:
            issues.append("long_latency")

    # Quality issues
    if record.outcome_quality is not None:
        if record.outcome_quality < 0.3:
            issues.append("very_low_quality")
        elif record.outcome_quality < 0.5:
            issues.append("low_quality")
        elif record.outcome_quality >= 0.9:
            issues.append("high_quality")

    # Backtrack issues
    if record.backtrack_count >= 3:
        issues.append("multiple_backtracks")
    elif record.backtrack_count >= 1:
        issues.append("had_backtrack")

    # Turn issues
    if record.turns_to_solution:
        if record.turns_to_solution > 5:
            issues.append("many_turns")
        elif record.turns_to_solution > 3:
            issues.append("several_turns")

    # Repeat issues
    if record.was_repeated:
        issues.append("repeated_question")

    # Multi-teacher issues
    if len(record.teachers) > 3:
        issues.append("too_many_teachers")

    return issues


# =============================================================================
# Convenience Functions
# =============================================================================

_default_reflector: Optional[AutoReflector] = None


def get_reflector() -> AutoReflector:
    """Get the default auto-reflector."""
    global _default_reflector
    if _default_reflector is None:
        _default_reflector = AutoReflector()
    return _default_reflector


def generate_reflection(
    record: InteractionRecord,
    teacher_scores: Optional[Dict[str, float]] = None,
) -> str:
    """Generate an automatic reflection.

    Args:
        record: The interaction record
        teacher_scores: Optional per-teacher scores

    Returns:
        Reflection text
    """
    return get_reflector().generate_reflection(record, teacher_scores)


def enrich_record(record: InteractionRecord) -> InteractionRecord:
    """Enrich a record with auto-generated fields.

    Args:
        record: The interaction record

    Returns:
        Enriched record
    """
    # Auto-classify intent if missing
    if not record.user_intent and record.user_query:
        record.user_intent = classify_intent(record.user_query)

    # Auto-detect issues
    if not record.auto_detected_issues:
        record.auto_detected_issues = detect_issues(record)

    # Auto-generate reflection
    if not record.ara_reflection:
        record.ara_reflection = generate_reflection(record)

    # Set primary teacher if not set
    if not record.primary_teacher and record.teachers:
        record.primary_teacher = record.teachers[0]

    return record
