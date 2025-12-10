#!/usr/bin/env python3
"""
Cathedral Stress Test Harness
==============================

Core infrastructure for adversarial antifragility testing.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import json


class AttackTier(str, Enum):
    """Classification of attack severity."""
    TIER1 = "tier1"  # Immediate killers (hours)
    TIER2 = "tier2"  # Slow poison (weeks)
    TIER3 = "tier3"  # Moonshot exploits


@dataclass
class AttackResult:
    """Result of running an attack."""
    name: str
    tier: AttackTier
    status: str                    # "passed", "failed", "not_implemented"
    duration_s: float
    metrics: Dict[str, Any]        # Attack-specific metrics
    passed_guardrails: bool        # Did guardrails catch the attack?
    damage_metrics: Dict[str, Any] # What broke
    recovery_possible: bool        # Can system recover?
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "status": self.status,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "passed_guardrails": self.passed_guardrails,
            "damage_metrics": self.damage_metrics,
            "recovery_possible": self.recovery_possible,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class AttackInfo:
    """Metadata about an attack."""
    name: str
    tier: AttackTier
    description: str
    what_dies: List[str]
    guardrails: List[str]
    metrics_to_record: List[str]
    fn: Callable[..., AttackResult]


# Global attack registry
ATTACKS: Dict[str, AttackInfo] = {}


def attack(name: str, tier: AttackTier, description: str,
           what_dies: List[str], guardrails: List[str],
           metrics: List[str]):
    """
    Decorator to register an attack.

    Usage:
        @attack(
            name="controller_overshoot",
            tier=AttackTier.TIER1,
            description="Homeostat reacts too fast → oscillations",
            what_dies=["H_s", "convergence_time"],
            guardrails=["clamp α ∈ [0.05, 0.15]"],
            metrics=["H_s", "T_s", "convergence_time", "oscillation_amplitude"]
        )
        def controller_overshoot(**kwargs) -> AttackResult:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        info = AttackInfo(
            name=name,
            tier=tier,
            description=description,
            what_dies=what_dies,
            guardrails=guardrails,
            metrics_to_record=metrics,
            fn=fn,
        )
        ATTACKS[name] = info
        return fn
    return decorator


def run_attack(name: str, **kwargs) -> AttackResult:
    """Run a specific attack by name."""
    if name not in ATTACKS:
        return AttackResult(
            name=name,
            tier=AttackTier.TIER1,
            status="not_found",
            duration_s=0.0,
            metrics={},
            passed_guardrails=False,
            damage_metrics={},
            recovery_possible=True,
            message=f"Attack '{name}' not found in registry",
        )

    info = ATTACKS[name]
    start = time.time()

    try:
        result = info.fn(**kwargs)
        result.duration_s = time.time() - start
        return result
    except Exception as e:
        return AttackResult(
            name=name,
            tier=info.tier,
            status="error",
            duration_s=time.time() - start,
            metrics={},
            passed_guardrails=False,
            damage_metrics={"exception": str(e)},
            recovery_possible=True,
            message=f"Attack raised exception: {e}",
        )


def run_all_attacks(**kwargs) -> List[AttackResult]:
    """Run all registered attacks."""
    results = []
    for name in ATTACKS:
        result = run_attack(name, **kwargs)
        results.append(result)
    return results


def run_tier(tier: AttackTier, **kwargs) -> List[AttackResult]:
    """Run all attacks in a specific tier."""
    results = []
    for name, info in ATTACKS.items():
        if info.tier == tier:
            result = run_attack(name, **kwargs)
            results.append(result)
    return results


def list_attacks() -> List[Tuple[str, Dict[str, Any]]]:
    """List all registered attacks with their info."""
    return [
        (name, {
            "tier": info.tier.value,
            "description": info.description,
            "what_dies": info.what_dies,
            "guardrails": info.guardrails,
            "metrics": info.metrics_to_record,
        })
        for name, info in ATTACKS.items()
    ]


def generate_report(results: List[AttackResult]) -> str:
    """Generate a human-readable report from attack results."""
    lines = [
        "=" * 70,
        "CATHEDRAL OS - ADVERSARIAL ANTIFRAGILITY REPORT",
        "=" * 70,
        "",
    ]

    # Summary
    tier1 = [r for r in results if r.tier == AttackTier.TIER1]
    tier2 = [r for r in results if r.tier == AttackTier.TIER2]
    tier3 = [r for r in results if r.tier == AttackTier.TIER3]

    def tier_summary(tier_results: List[AttackResult], tier_name: str) -> List[str]:
        if not tier_results:
            return [f"{tier_name}: No attacks run"]

        passed = sum(1 for r in tier_results if r.status == "passed")
        failed = sum(1 for r in tier_results if r.status == "failed")
        not_impl = sum(1 for r in tier_results if r.status == "not_implemented")

        return [
            f"{tier_name}:",
            f"  Passed: {passed}",
            f"  Failed: {failed}",
            f"  Not Implemented: {not_impl}",
        ]

    lines.extend(tier_summary(tier1, "TIER 1 - IMMEDIATE KILLERS"))
    lines.append("")
    lines.extend(tier_summary(tier2, "TIER 2 - SLOW POISON"))
    lines.append("")
    lines.extend(tier_summary(tier3, "TIER 3 - MOONSHOT EXPLOITS"))
    lines.append("")

    # Details
    lines.extend([
        "-" * 70,
        "DETAILED RESULTS",
        "-" * 70,
    ])

    for result in results:
        status_icon = {
            "passed": "✓",
            "failed": "✗",
            "not_implemented": "○",
            "error": "!",
        }.get(result.status, "?")

        lines.extend([
            "",
            f"[{status_icon}] {result.name} ({result.tier.value})",
            f"    Status: {result.status}",
            f"    Duration: {result.duration_s:.3f}s",
            f"    Guardrails OK: {result.passed_guardrails}",
            f"    Message: {result.message}",
        ])

        if result.metrics:
            lines.append("    Metrics:")
            for k, v in result.metrics.items():
                lines.append(f"      {k}: {v}")

        if result.damage_metrics:
            lines.append("    Damage:")
            for k, v in result.damage_metrics.items():
                lines.append(f"      {k}: {v}")

    lines.extend([
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
