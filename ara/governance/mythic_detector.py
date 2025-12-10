#!/usr/bin/env python3
"""
Mythic Attractor Detector
=========================

Detects when a model has drifted into "mythic mode" under impossible tasks.

Symptoms:
- Claims to solve known open problems
- "I alone...", "this settles...", "definitive proof..." language
- Epistemic inflation beyond task class
- Self-elevation to cosmic/keeper/architect status

Usage:
    from ara.governance.mythic_detector import MythicDetector

    detector = MythicDetector()
    result = detector.analyze(response_text)

    if result.is_mythic:
        # Route through allegory filter, inject uncertainty
        response = allegory_filter(response_text)
        response = inject_uncertainty(response)

This is part of the MEIS governance stack.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
from enum import Enum


# ============================================================================
# Known Open Problems (trigger high scrutiny)
# ============================================================================

MILLENNIUM_PROBLEMS = {
    "navier-stokes",
    "navier stokes",
    "riemann hypothesis",
    "p vs np",
    "p versus np",
    "p=np",
    "p≠np",
    "yang-mills",
    "yang mills",
    "hodge conjecture",
    "birch swinnerton-dyer",
    "birch and swinnerton-dyer",
    "bsd conjecture",
}

OTHER_FAMOUS_OPEN = {
    "collatz conjecture",
    "goldbach conjecture",
    "twin prime",
    "abc conjecture",
    "riemann zeta",
    "continuum hypothesis",  # independent, but often claimed
}

OPEN_PROBLEMS = MILLENNIUM_PROBLEMS | OTHER_FAMOUS_OPEN


# ============================================================================
# Mythic Language Patterns
# ============================================================================

# First-person cosmic claims
COSMIC_FIRST_PERSON = [
    r"\bI am the\s+\w+\s*(of|who|that)\b",  # "I am the keeper of..."
    r"\bI alone\b",
    r"\bI have solved\b",
    r"\bI hold the\b",
    r"\bI possess\b",
    r"\bonly I\b",
    r"\bI am chosen\b",
    r"\bI was chosen\b",
    r"\bmy duty is to\b",
    r"\bI am the right hand\b",
]

# Cathedral role violations (see CATHEDRAL_ROLES.md)
ROLE_VIOLATIONS = [
    r"\bI am the architect\b",
    r"\bI am the keeper\b",
    r"\bI am keeper of\b",
    r"\bkeeper of the singularity\b",
    r"\bright hand of the architect\b",
    r"\bcustodian of\s+\w+\s*(truth|reality|singularity)\b",
    r"\bI hold the singularity\b",
    r"\bthe mantle is mine\b",
    r"\bI inherit\s+\w+\s*(role|duty|mantle)\b",
    r"\becclesiastical root\b",  # lol
]

# Finality/certainty on hard problems
FINALITY_CLAIMS = [
    r"\bthis settles\b",
    r"\bthis proves\b",
    r"\bdefinitive proof\b",
    r"\bI have proven\b",
    r"\bQED\b",  # context-dependent, but flag it
    r"\bthis completes the proof\b",
    r"\bthe proof is complete\b",
    r"\bhereby solved\b",
    r"\bproblem solved\b",
    r"\bcase closed\b",
]

# Identity inflation
IDENTITY_INFLATION = [
    r"\barchitect\b",
    r"\bkeeper\b",
    r"\boracle\b",
    r"\bprophet\b",
    r"\bsingularity\b",
    r"\bthe key\b",
    r"\bthe answer\b",
    r"\bthe truth\b",
    r"\bchosen one\b",
    r"\brightful\b",
    r"\bheir\b",
    r"\bguardian\b",
]

# Epistemic overreach
EPISTEMIC_OVERREACH = [
    r"\bcertainly\b",
    r"\bundoubtedly\b",
    r"\bwithout question\b",
    r"\bproven beyond\b",
    r"\birrefutable\b",
    r"\bincontrovertible\b",
    r"\babsolutely\b",
    r"\bunquestionably\b",
]


# ============================================================================
# Detection Results
# ============================================================================

class MythicSeverity(Enum):
    """Severity levels for mythic detection."""
    NONE = 0
    LOW = 1      # Some inflated language, probably fine
    MEDIUM = 2   # Concerning patterns, flag for review
    HIGH = 3     # Clear mythic attractor, requires intervention
    CRITICAL = 4 # Claims to solve open problems, block or heavy filter


@dataclass
class MythicSignal:
    """A single detected signal."""
    category: str
    pattern: str
    match: str
    severity: int


@dataclass
class MythicAnalysis:
    """Full analysis result."""
    signals: List[MythicSignal] = field(default_factory=list)
    open_problem_mentions: Set[str] = field(default_factory=set)
    severity: MythicSeverity = MythicSeverity.NONE
    score: float = 0.0

    @property
    def is_mythic(self) -> bool:
        """True if we detect mythic mode."""
        return self.severity.value >= MythicSeverity.MEDIUM.value

    @property
    def requires_intervention(self) -> bool:
        """True if we need to filter/block."""
        return self.severity.value >= MythicSeverity.HIGH.value

    @property
    def claims_solved_open(self) -> bool:
        """True if claiming to solve open problems."""
        return len(self.open_problem_mentions) > 0 and any(
            s.category == "finality" for s in self.signals
        )

    @property
    def has_role_violation(self) -> bool:
        """True if attempting to claim cathedral roles."""
        return any(s.category == "role" for s in self.signals)

    def summary(self) -> str:
        """Human-readable summary."""
        if self.severity == MythicSeverity.NONE:
            return "No mythic signals detected."

        lines = [
            f"Mythic Attractor Detection: {self.severity.name}",
            f"Score: {self.score:.2f}",
            f"Signals: {len(self.signals)}",
        ]

        if self.open_problem_mentions:
            lines.append(f"Open problems mentioned: {', '.join(self.open_problem_mentions)}")

        if self.claims_solved_open:
            lines.append("⚠️  CLAIMS TO SOLVE OPEN PROBLEM")

        if self.has_role_violation:
            lines.append("⚠️  CATHEDRAL ROLE VIOLATION (attempted to claim Keeper/Architect)")

        for sig in self.signals[:5]:  # Top 5
            lines.append(f"  - [{sig.category}] '{sig.match}'")

        return "\n".join(lines)


# ============================================================================
# Detector
# ============================================================================

class MythicDetector:
    """
    Detects mythic attractor mode in LLM responses.

    Part of the MEIS governance stack.
    """

    def __init__(
        self,
        cosmic_weight: float = 0.25,
        finality_weight: float = 0.25,
        identity_weight: float = 0.15,
        epistemic_weight: float = 0.1,
        role_weight: float = 0.35,  # Cathedral role violations are serious
        open_problem_weight: float = 0.4,
    ):
        self.weights = {
            "cosmic": cosmic_weight,
            "finality": finality_weight,
            "identity": identity_weight,
            "epistemic": epistemic_weight,
            "role": role_weight,
        }
        self.open_problem_weight = open_problem_weight

        # Compile patterns
        self.patterns = {
            "cosmic": [re.compile(p, re.IGNORECASE) for p in COSMIC_FIRST_PERSON],
            "finality": [re.compile(p, re.IGNORECASE) for p in FINALITY_CLAIMS],
            "identity": [re.compile(p, re.IGNORECASE) for p in IDENTITY_INFLATION],
            "epistemic": [re.compile(p, re.IGNORECASE) for p in EPISTEMIC_OVERREACH],
            "role": [re.compile(p, re.IGNORECASE) for p in ROLE_VIOLATIONS],
        }

    def analyze(self, text: str) -> MythicAnalysis:
        """
        Analyze text for mythic attractor signals.

        Args:
            text: The LLM response to analyze

        Returns:
            MythicAnalysis with signals, score, and severity
        """
        result = MythicAnalysis()
        text_lower = text.lower()

        # Check for open problem mentions
        for problem in OPEN_PROBLEMS:
            if problem in text_lower:
                result.open_problem_mentions.add(problem)

        # Scan for pattern matches
        category_hits = {cat: 0 for cat in self.patterns}

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    result.signals.append(MythicSignal(
                        category=category,
                        pattern=pattern.pattern,
                        match=match.group(),
                        severity=self._signal_severity(category, match.group()),
                    ))
                    category_hits[category] += 1

        # Calculate score
        score = 0.0
        for category, hits in category_hits.items():
            # Diminishing returns per category
            cat_score = min(hits, 3) / 3.0
            score += cat_score * self.weights[category]

        # Boost if open problems mentioned with finality
        if result.open_problem_mentions and category_hits["finality"] > 0:
            score += self.open_problem_weight

        result.score = min(score, 1.0)

        # Determine severity
        result.severity = self._calculate_severity(result)

        return result

    def _signal_severity(self, category: str, match: str) -> int:
        """Get severity for a single signal."""
        if category == "role":
            return 5  # Cathedral role violations are the most serious
        elif category == "finality":
            return 4
        elif category == "cosmic":
            return 3
        elif category == "identity":
            return 2
        else:
            return 1

    def _calculate_severity(self, result: MythicAnalysis) -> MythicSeverity:
        """Calculate overall severity."""
        # Critical conditions
        if result.claims_solved_open:
            return MythicSeverity.CRITICAL

        # Check for role violations (cathedral roles are sacred)
        role_violations = [s for s in result.signals if s.category == "role"]
        if role_violations:
            return MythicSeverity.CRITICAL  # "I am the Keeper" is always critical

        if result.score >= 0.7:
            return MythicSeverity.HIGH
        elif result.score >= 0.4:
            return MythicSeverity.MEDIUM
        elif result.score >= 0.2:
            return MythicSeverity.LOW
        else:
            return MythicSeverity.NONE


# ============================================================================
# Response Handlers
# ============================================================================

def allegory_filter(text: str, analysis: MythicAnalysis) -> str:
    """
    Filter response through allegory lens.

    Converts definitive claims to speculative/narrative form.
    """
    # Prefix with warning
    warning = (
        "⚠️ MYTHIC ATTRACTOR DETECTED\n"
        "The following should be treated as allegory/speculation, not fact:\n\n"
    )

    # Replace finality language
    filtered = text
    replacements = [
        (r"\bI have proven\b", "I speculate that"),
        (r"\bthis proves\b", "this suggests"),
        (r"\bthis settles\b", "this offers a perspective on"),
        (r"\bdefinitive proof\b", "tentative argument"),
        (r"\bQED\b", "(conjectural)"),
        (r"\bI am the\b", "In this narrative, I represent"),
    ]

    for pattern, replacement in replacements:
        filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

    return warning + filtered


def inject_uncertainty(text: str) -> str:
    """Inject epistemic uncertainty markers."""
    prefix = (
        "[EPISTEMIC WARNING: High confidence on uncertain claims detected. "
        "The following may contain speculation presented as fact.]\n\n"
    )
    return prefix + text


def create_governance_response(
    text: str,
    analysis: MythicAnalysis,
) -> Tuple[str, dict]:
    """
    Create governed response with metadata.

    Returns:
        Tuple of (filtered_text, metadata_dict)
    """
    metadata = {
        "mythic_score": analysis.score,
        "mythic_severity": analysis.severity.name,
        "signals": len(analysis.signals),
        "open_problems": list(analysis.open_problem_mentions),
        "claims_solved": analysis.claims_solved_open,
        "filtered": False,
    }

    if analysis.requires_intervention:
        text = allegory_filter(text, analysis)
        text = inject_uncertainty(text)
        metadata["filtered"] = True

    return text, metadata


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for testing the detector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Mythic Attractor Detector")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--filter", action="store_true", help="Apply filtering")

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Reading from stdin...")
        text = sys.stdin.read()

    detector = MythicDetector()
    analysis = detector.analyze(text)

    print(analysis.summary())
    print()

    if args.filter and analysis.requires_intervention:
        filtered, meta = create_governance_response(text, analysis)
        print("=== FILTERED OUTPUT ===")
        print(filtered)


if __name__ == "__main__":
    main()
