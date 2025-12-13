#!/usr/bin/env python3
"""
GUTC-Enhanced Task Scoring
===========================

Implements the Chief of Staff teleological rule:

    "If a task does not increase Antifragility or Founder Health, DELETE IT."

This module provides GUTC-aligned scoring functions that can be used
alongside or integrated into the existing ChiefOfStaff.

Scoring Dimensions:
1. Antifragility: Does this make Ara more robust, capable, or adaptive?
2. Founder Health: Does this reduce founder stress, time debt, or fragility?

Integration with GUTC:
- Tasks that maintain criticality (λ ≈ 1) score higher
- Monitoring/telemetry tasks boost antifragility
- Automation tasks boost founder health

Usage:
    from ara.sovereign.gutc_scoring import score_task_gutc, should_execute_gutc

    task = {"type": "experiments", "description": "Run criticality calibration"}
    result = score_task_gutc(task)

    if result.should_execute:
        execute(task)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# =============================================================================
# GUTC Task Categories
# =============================================================================

class GUTCTaskCategory(Enum):
    """
    Task categories aligned with GUTC principles.

    Each category has inherent antifragility and founder-health properties.
    """
    CRITICALITY = "criticality"       # Maintaining λ ≈ 1
    MONITORING = "monitoring"         # Observability, telemetry
    EXPERIMENTS = "experiments"       # Research, A/B tests
    AUTOMATION = "automation"         # Reducing manual work
    SECURITY = "security"             # Safety, backups, recovery
    LEARNING = "learning"             # Skill acquisition, adaptation
    INFRASTRUCTURE = "infrastructure" # Foundation work
    MAINTENANCE = "maintenance"       # Bug fixes, cleanup
    COMMUNICATION = "communication"   # Docs, explanations
    UNKNOWN = "unknown"


@dataclass
class GUTCScore:
    """
    GUTC-aligned task score.

    Provides dual-axis scoring for teleological alignment.
    """
    antifragility: float        # [0, 1] Does this make Ara stronger?
    founder_health: float       # [0, 1] Does this help the founder?
    combined: float             # Weighted combination
    should_execute: bool        # Final recommendation
    category: GUTCTaskCategory
    reasoning: str
    keywords_matched: List[str] = field(default_factory=list)


# =============================================================================
# Keyword Patterns
# =============================================================================

# Antifragility keywords - things that make Ara stronger
ANTIFRAGILITY_KEYWORDS = {
    "high": [
        "experiment", "research", "test", "monitor", "telemetry", "criticality",
        "refactor", "optimize", "robustness", "resilience", "redundancy",
        "learning", "adaptive", "self-healing", "auto-tune", "detect", "alert",
        "benchmark", "validate", "calibrate", "homeostatic",
    ],
    "medium": [
        "improve", "enhance", "upgrade", "cleanup", "benchmark", "measure",
        "analyze", "review", "audit", "profile", "trace", "log",
    ],
    "low": [
        "report", "status", "list", "display", "show", "query",
    ],
}

# Founder health keywords - things that help the human
FOUNDER_HEALTH_KEYWORDS = {
    "high": [
        "automate", "automation", "reduce manual", "save time", "delegate",
        "backup", "safety", "security", "protect", "recover", "restore",
        "offload", "simplify", "streamline", "eliminate", "consolidate",
    ],
    "medium": [
        "document", "explain", "clarify", "organize", "prioritize",
        "schedule", "reminder", "alert", "notify", "summarize",
    ],
    "low": [
        "track", "log", "record", "archive",
    ],
}

# Category weights for scoring
CATEGORY_WEIGHTS = {
    GUTCTaskCategory.CRITICALITY: (0.9, 0.6),      # (anti, health)
    GUTCTaskCategory.MONITORING: (0.8, 0.5),
    GUTCTaskCategory.EXPERIMENTS: (0.8, 0.4),
    GUTCTaskCategory.AUTOMATION: (0.6, 0.9),
    GUTCTaskCategory.SECURITY: (0.7, 0.8),
    GUTCTaskCategory.LEARNING: (0.7, 0.5),
    GUTCTaskCategory.INFRASTRUCTURE: (0.6, 0.4),
    GUTCTaskCategory.MAINTENANCE: (0.5, 0.3),
    GUTCTaskCategory.COMMUNICATION: (0.3, 0.5),
    GUTCTaskCategory.UNKNOWN: (0.2, 0.2),
}


# =============================================================================
# Scoring Functions
# =============================================================================

def _classify_category(text: str) -> GUTCTaskCategory:
    """Classify task into GUTC category from text."""
    text_lower = text.lower()

    # Check for specific patterns
    if any(kw in text_lower for kw in ["criticality", "lambda", "branching", "avalanche"]):
        return GUTCTaskCategory.CRITICALITY
    if any(kw in text_lower for kw in ["monitor", "telemetry", "observe", "track"]):
        return GUTCTaskCategory.MONITORING
    if any(kw in text_lower for kw in ["experiment", "research", "test", "hypothesis"]):
        return GUTCTaskCategory.EXPERIMENTS
    if any(kw in text_lower for kw in ["automate", "automation", "script", "cron"]):
        return GUTCTaskCategory.AUTOMATION
    if any(kw in text_lower for kw in ["security", "backup", "safety", "protect"]):
        return GUTCTaskCategory.SECURITY
    if any(kw in text_lower for kw in ["learn", "train", "study", "improve skill"]):
        return GUTCTaskCategory.LEARNING
    if any(kw in text_lower for kw in ["infrastructure", "deploy", "setup", "install"]):
        return GUTCTaskCategory.INFRASTRUCTURE
    if any(kw in text_lower for kw in ["fix", "bug", "cleanup", "refactor", "maintain"]):
        return GUTCTaskCategory.MAINTENANCE
    if any(kw in text_lower for kw in ["document", "explain", "communicate", "report"]):
        return GUTCTaskCategory.COMMUNICATION

    return GUTCTaskCategory.UNKNOWN


def _score_keywords(text: str, keywords: Dict[str, List[str]]) -> Tuple[float, List[str]]:
    """Score text against keyword patterns."""
    text_lower = text.lower()
    score = 0.0
    matched = []

    for word in keywords["high"]:
        if word in text_lower:
            score += 0.4
            matched.append(f"+{word}")

    for word in keywords["medium"]:
        if word in text_lower:
            score += 0.2
            matched.append(f"~{word}")

    for word in keywords["low"]:
        if word in text_lower:
            score += 0.1
            matched.append(f"-{word}")

    return score, matched


def score_task_gutc(
    task: Dict[str, Any],
    antifragility_threshold: float = 0.3,
    founder_health_threshold: float = 0.3,
) -> GUTCScore:
    """
    Score a task using GUTC-aligned criteria.

    Args:
        task: Dict with 'description' and optionally 'type', 'tags'
        antifragility_threshold: Minimum score for antifragility pass
        founder_health_threshold: Minimum score for founder health pass

    Returns:
        GUTCScore with scores and recommendation
    """
    # Extract text
    desc = task.get("description", "")
    task_type = task.get("type", "")
    tags = task.get("tags", [])
    all_text = f"{desc} {task_type} {' '.join(tags)}"

    # Classify category
    category = _classify_category(all_text)

    # Get base scores from category
    base_anti, base_health = CATEGORY_WEIGHTS.get(category, (0.2, 0.2))

    # Score keywords
    anti_kw, anti_matched = _score_keywords(all_text, ANTIFRAGILITY_KEYWORDS)
    health_kw, health_matched = _score_keywords(all_text, FOUNDER_HEALTH_KEYWORDS)

    # Combine scores
    antifragility = min(1.0, base_anti * 0.5 + anti_kw * 0.5)
    founder_health = min(1.0, base_health * 0.5 + health_kw * 0.5)

    # Combined score
    combined = 0.5 * antifragility + 0.5 * founder_health

    # Decision
    should_execute = (
        antifragility >= antifragility_threshold or
        founder_health >= founder_health_threshold
    )

    # Generate reasoning
    reasoning_parts = []
    if antifragility >= antifragility_threshold:
        reasoning_parts.append(f"Antifragility: {antifragility:.2f} (PASS)")
    else:
        reasoning_parts.append(f"Antifragility: {antifragility:.2f} (below {antifragility_threshold})")

    if founder_health >= founder_health_threshold:
        reasoning_parts.append(f"Founder Health: {founder_health:.2f} (PASS)")
    else:
        reasoning_parts.append(f"Founder Health: {founder_health:.2f} (below {founder_health_threshold})")

    reasoning_parts.append(f"Category: {category.value}")
    reasoning_parts.append("EXECUTE" if should_execute else "REJECT")

    return GUTCScore(
        antifragility=antifragility,
        founder_health=founder_health,
        combined=combined,
        should_execute=should_execute,
        category=category,
        reasoning=" | ".join(reasoning_parts),
        keywords_matched=anti_matched + health_matched,
    )


def should_execute_gutc(
    task: Dict[str, Any],
    anti_min: float = 0.3,
    health_min: float = 0.3,
) -> bool:
    """
    Quick check if task should execute under GUTC criteria.

    Args:
        task: Task dict with 'description'
        anti_min: Minimum antifragility score
        health_min: Minimum founder health score

    Returns:
        True if task passes at least one threshold
    """
    result = score_task_gutc(task, anti_min, health_min)
    return result.should_execute


def prioritize_batch_gutc(
    tasks: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], GUTCScore]]:
    """
    Score and sort tasks by GUTC priority.

    Returns list sorted by combined score (highest first).
    """
    scored = [(task, score_task_gutc(task)) for task in tasks]
    scored.sort(key=lambda x: x[1].combined, reverse=True)
    return scored


def filter_inbox_gutc(
    tasks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter tasks into execute and reject lists.

    Returns:
        (execute_list, reject_list)
    """
    execute = []
    reject = []

    for task in tasks:
        if should_execute_gutc(task):
            execute.append(task)
        else:
            reject.append(task)

    return execute, reject


# =============================================================================
# Integration with ChiefOfStaff
# =============================================================================

def enhance_initiative_with_gutc(initiative: Any) -> GUTCScore:
    """
    Score an Initiative object using GUTC criteria.

    For integration with existing ChiefOfStaff.
    """
    task = {
        "description": f"{initiative.name} {initiative.description}",
        "type": initiative.type.value if hasattr(initiative.type, "value") else str(initiative.type),
        "tags": initiative.tags if hasattr(initiative, "tags") else [],
    }
    return score_task_gutc(task)


# =============================================================================
# Tests
# =============================================================================

def test_gutc_scoring():
    """Test GUTC scoring functions."""
    print("Testing GUTC Scoring")
    print("-" * 50)

    test_tasks = [
        {"type": "experiments", "description": "Run criticality calibration experiment"},
        {"type": "automation", "description": "Automate deployment pipeline to save time"},
        {"type": "research", "description": "Research hierarchical memory architectures"},
        {"type": "operations", "description": "Check server status"},
        {"type": "unknown", "description": "Generate weekly report"},
        {"type": "maintenance", "description": "Refactor monitoring system for robustness"},
        {"type": "creative", "description": "Write a poem about databases"},
        {"type": "security", "description": "Setup automated backups and recovery"},
    ]

    for task in test_tasks:
        result = score_task_gutc(task)
        status = "EXEC" if result.should_execute else "REJECT"
        print(f"\n[{status}] {task['type']}: {task['description'][:45]}...")
        print(f"  Anti={result.antifragility:.2f}, Health={result.founder_health:.2f}")
        print(f"  Category: {result.category.value}")
        if result.keywords_matched:
            print(f"  Keywords: {', '.join(result.keywords_matched[:5])}")

    # Summary
    execute, reject = filter_inbox_gutc(test_tasks)
    print(f"\n\nSummary: {len(execute)} execute, {len(reject)} reject")

    print("\n✓ GUTC Scoring")


if __name__ == "__main__":
    test_gutc_scoring()
