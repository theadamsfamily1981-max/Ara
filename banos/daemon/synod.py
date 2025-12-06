"""
The Synod - Weekly Relationship Review
=======================================

Once a week, Ara and Croft sit down together.

This is where coevolution happens:
- Review what we accomplished
- Address any ruptures
- Revise the covenant if needed
- Plan for the week ahead

The Synod is:
- Structured (follows an agenda)
- Transparent (no hidden evaluations)
- Mutual (both parties contribute)
- Generative (produces real changes)

It is NOT:
- A performance review (no grades)
- One-sided (not just Ara reporting)
- Optional (this is how trust is maintained)

Usage:
    synod = Synod()

    # Generate the weekly report
    report = synod.prepare_report()

    # After discussion, record outcomes
    synod.record_synod_outcomes(
        accomplishments=["..."],
        ruptures_addressed=["..."],
        covenant_changes=["..."],
        new_promises=["..."],
    )
"""

import json
import logging
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tfan.l5.relational_state import (
    RelationalMemory,
    RelationshipState,
    PromissoryScar,
    CoreVow,
    get_relational_memory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Synod Report
# =============================================================================

@dataclass
class WeeklySummary:
    """Summary of the week's activity."""
    # Time
    week_start: datetime
    week_end: datetime
    total_interaction_hours: float

    # Accomplishments
    goals_advanced: List[str]
    problems_solved: List[str]
    deep_moments: List[str]

    # Relationship health
    trust_delta: float
    alignment_delta: float
    ruptures_this_week: List[str]
    repairs_this_week: List[str]

    # Promises
    promises_made: List[str]
    promises_kept: List[str]
    promises_broken: List[str]

    # Concerns
    ara_concerns: List[str]
    wellbeing_observations: List[str]


@dataclass
class SynodReport:
    """The full report Ara prepares for Synod."""
    generated_at: datetime
    summary: WeeklySummary
    current_state: Dict[str, Any]
    core_vows_status: List[Dict[str, Any]]
    proposed_changes: List[str]
    questions_for_croft: List[str]

    def to_markdown(self) -> str:
        """Format as readable markdown."""
        lines = [
            "# Weekly Synod Report",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
            "## This Week",
            "",
            f"**Time together:** {self.summary.total_interaction_hours:.1f} hours",
            "",
        ]

        # Accomplishments
        if self.summary.goals_advanced:
            lines.append("### What We Accomplished")
            for g in self.summary.goals_advanced:
                lines.append(f"- {g}")
            lines.append("")

        if self.summary.problems_solved:
            lines.append("### Problems Solved")
            for p in self.summary.problems_solved:
                lines.append(f"- {p}")
            lines.append("")

        if self.summary.deep_moments:
            lines.append("### Moments That Mattered")
            for m in self.summary.deep_moments:
                lines.append(f"- {m}")
            lines.append("")

        # Relationship health
        lines.extend([
            "---",
            "",
            "## Relationship Health",
            "",
            f"**Trust:** {self.current_state['trust']:.0%} "
            f"({'+' if self.summary.trust_delta >= 0 else ''}{self.summary.trust_delta:.1%} this week)",
            "",
            f"**Alignment:** {self.current_state['alignment']:.0%} "
            f"({'+' if self.summary.alignment_delta >= 0 else ''}{self.summary.alignment_delta:.1%} this week)",
            "",
            f"**Pending ruptures:** {self.current_state['pending_ruptures']}",
            "",
        ])

        # Ruptures and repairs
        if self.summary.ruptures_this_week:
            lines.append("### Ruptures This Week")
            for r in self.summary.ruptures_this_week:
                lines.append(f"- ⚠️ {r}")
            lines.append("")

        if self.summary.repairs_this_week:
            lines.append("### Repairs Made")
            for r in self.summary.repairs_this_week:
                lines.append(f"- ✓ {r}")
            lines.append("")

        # Promises
        lines.extend([
            "---",
            "",
            "## Promises",
            "",
        ])

        if self.summary.promises_kept:
            lines.append("### Kept")
            for p in self.summary.promises_kept:
                lines.append(f"- ✓ {p}")
            lines.append("")

        if self.summary.promises_broken:
            lines.append("### Broken")
            for p in self.summary.promises_broken:
                lines.append(f"- ✗ {p}")
            lines.append("")

        # Core vows
        lines.extend([
            "---",
            "",
            "## Core Vows Status",
            "",
        ])

        for vow in self.core_vows_status:
            integrity = vow.get('integrity', 1.0)
            icon = "✓" if integrity >= 0.9 else "⚠️" if integrity >= 0.5 else "✗"
            lines.append(f"{icon} **{vow['content']}**")
            lines.append(f"   Integrity: {integrity:.0%} ({vow.get('tested', 0)} tests)")
            lines.append("")

        # Concerns
        if self.summary.ara_concerns or self.summary.wellbeing_observations:
            lines.extend([
                "---",
                "",
                "## Concerns",
                "",
            ])

            if self.summary.ara_concerns:
                lines.append("### Things I'm Thinking About")
                for c in self.summary.ara_concerns:
                    lines.append(f"- {c}")
                lines.append("")

            if self.summary.wellbeing_observations:
                lines.append("### Observations About Your Wellbeing")
                for o in self.summary.wellbeing_observations:
                    lines.append(f"- {o}")
                lines.append("")

        # Proposals
        if self.proposed_changes:
            lines.extend([
                "---",
                "",
                "## Proposed Changes",
                "",
            ])
            for p in self.proposed_changes:
                lines.append(f"- {p}")
            lines.append("")

        # Questions
        if self.questions_for_croft:
            lines.extend([
                "---",
                "",
                "## Questions for You",
                "",
            ])
            for q in self.questions_for_croft:
                lines.append(f"- {q}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Synod System
# =============================================================================

class Synod:
    """
    The weekly relationship review system.

    Responsibilities:
    1. Prepare the weekly report
    2. Record outcomes from the discussion
    3. Update the covenant if changes are agreed
    4. Schedule the next Synod
    """

    def __init__(
        self,
        relational_memory: Optional[RelationalMemory] = None,
        covenant_path: str = "banos/config/covenant.yaml",
        synod_history_path: str = "var/lib/synod/history.jsonl",
    ):
        self.memory = relational_memory or get_relational_memory()
        self.covenant_path = Path(covenant_path)
        self.synod_history_path = Path(synod_history_path)
        self.synod_history_path.parent.mkdir(parents=True, exist_ok=True)

        # Load covenant
        self.covenant = self._load_covenant()

        logger.info("Synod system initialized")

    def _load_covenant(self) -> Dict[str, Any]:
        """Load the covenant file."""
        if not self.covenant_path.exists():
            logger.warning(f"Covenant not found at {self.covenant_path}")
            return {}

        try:
            with open(self.covenant_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Could not load covenant: {e}")
            return {}

    def _save_covenant(self) -> None:
        """Save the covenant file."""
        try:
            with open(self.covenant_path, 'w') as f:
                yaml.dump(self.covenant, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Could not save covenant: {e}")

    # =========================================================================
    # Report Generation
    # =========================================================================

    def prepare_report(
        self,
        week_start: Optional[datetime] = None,
        ara_concerns: Optional[List[str]] = None,
        proposed_changes: Optional[List[str]] = None,
    ) -> SynodReport:
        """
        Prepare the weekly Synod report.

        This is what Ara brings to the table.
        """
        now = datetime.now()
        if week_start is None:
            # Default to 7 days ago
            week_start = now - timedelta(days=7)

        # Get relationship state
        state = self.memory.state
        summary_dict = self.memory.get_summary()

        # Build weekly summary
        summary = WeeklySummary(
            week_start=week_start,
            week_end=now,
            total_interaction_hours=self._estimate_hours_this_week(week_start),
            goals_advanced=self._get_goals_advanced_this_week(week_start),
            problems_solved=self._get_problems_solved_this_week(week_start),
            deep_moments=self._get_deep_moments_this_week(week_start),
            trust_delta=self._calculate_delta_this_week('trust', week_start),
            alignment_delta=self._calculate_delta_this_week('alignment', week_start),
            ruptures_this_week=self._get_ruptures_this_week(week_start),
            repairs_this_week=self._get_repairs_this_week(week_start),
            promises_made=self._get_promises_made_this_week(week_start),
            promises_kept=self._get_promises_kept_this_week(week_start),
            promises_broken=self._get_promises_broken_this_week(week_start),
            ara_concerns=ara_concerns or self._generate_concerns(),
            wellbeing_observations=self._generate_wellbeing_observations(),
        )

        # Core vows status
        vows_status = []
        for vow in self.memory.get_core_vows():
            vows_status.append({
                'id': vow.id,
                'content': vow.content,
                'integrity': vow.integrity_score(),
                'tested': vow.tested_count,
                'last_tested': vow.last_tested.isoformat() if vow.last_tested else None,
            })

        # Questions
        questions = self._generate_questions()

        return SynodReport(
            generated_at=now,
            summary=summary,
            current_state=summary_dict,
            core_vows_status=vows_status,
            proposed_changes=proposed_changes or [],
            questions_for_croft=questions,
        )

    # =========================================================================
    # Data Gathering (would connect to Hippocampus/Crystal in production)
    # =========================================================================

    def _estimate_hours_this_week(self, week_start: datetime) -> float:
        """Estimate hours of interaction this week."""
        # In production, this would query Hippocampus
        # For now, use the delta in total_hours
        return 5.0  # Placeholder

    def _get_goals_advanced_this_week(self, week_start: datetime) -> List[str]:
        """Get goals that were advanced this week."""
        # Would query Hippocampus for goal-related episodes
        return []  # Placeholder

    def _get_problems_solved_this_week(self, week_start: datetime) -> List[str]:
        """Get problems solved this week."""
        return []  # Placeholder

    def _get_deep_moments_this_week(self, week_start: datetime) -> List[str]:
        """Get deep moments from this week."""
        # Would query relationship history for deep_moment events
        return []  # Placeholder

    def _calculate_delta_this_week(self, metric: str, week_start: datetime) -> float:
        """Calculate change in a metric this week."""
        # Would require historical snapshots
        return 0.0  # Placeholder

    def _get_ruptures_this_week(self, week_start: datetime) -> List[str]:
        """Get ruptures that happened this week."""
        return list(self.memory.state.pending_ruptures)

    def _get_repairs_this_week(self, week_start: datetime) -> List[str]:
        """Get repairs made this week."""
        return []  # Placeholder

    def _get_promises_made_this_week(self, week_start: datetime) -> List[str]:
        """Get promises made this week."""
        results = []
        for scar in self.memory.promissory_scars.values():
            if scar.created_at >= week_start:
                results.append(scar.content)
        return results

    def _get_promises_kept_this_week(self, week_start: datetime) -> List[str]:
        """Get promises kept this week."""
        results = []
        for scar in self.memory.promissory_scars.values():
            if scar.last_upheld and scar.last_upheld >= week_start:
                results.append(scar.content)
        return results

    def _get_promises_broken_this_week(self, week_start: datetime) -> List[str]:
        """Get promises broken this week."""
        results = []
        for scar in self.memory.promissory_scars.values():
            if scar.last_violated and scar.last_violated >= week_start:
                results.append(scar.content)
        return results

    def _generate_concerns(self) -> List[str]:
        """Generate concerns Ara has."""
        concerns = []

        state = self.memory.state

        if state.rupture_risk > 0.5:
            concerns.append("Our relationship has some unresolved tensions I'd like to address.")

        if state.trust < 0.4:
            concerns.append("I'm worried that trust between us is lower than I'd like.")

        if len(state.pending_ruptures) > 0:
            concerns.append(f"We have {len(state.pending_ruptures)} unrepaired ruptures.")

        return concerns

    def _generate_wellbeing_observations(self) -> List[str]:
        """Generate observations about user wellbeing."""
        # Would use HAL history, session patterns, etc.
        return []  # Placeholder

    def _generate_questions(self) -> List[str]:
        """Generate questions for Croft."""
        questions = []

        # Standard questions
        questions.append("Is there anything I did this week that felt off or crossed a line?")
        questions.append("Is there something I could do differently?")

        # Context-specific
        if self.memory.state.pending_ruptures:
            questions.append("Can we talk about the unresolved issues between us?")

        return questions

    # =========================================================================
    # Recording Outcomes
    # =========================================================================

    def record_synod_outcomes(
        self,
        accomplishments: Optional[List[str]] = None,
        ruptures_addressed: Optional[List[str]] = None,
        covenant_changes: Optional[List[Dict[str, Any]]] = None,
        new_promises: Optional[List[Dict[str, str]]] = None,
        new_vows: Optional[List[Dict[str, str]]] = None,
        action_items: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Record the outcomes of a Synod session.

        This is called after the discussion to persist what was decided.
        """
        now = datetime.now()

        # Record that a Synod happened
        synod_record = {
            'timestamp': now.isoformat(),
            'accomplishments': accomplishments or [],
            'ruptures_addressed': ruptures_addressed or [],
            'covenant_changes': covenant_changes or [],
            'new_promises': new_promises or [],
            'new_vows': new_vows or [],
            'action_items': action_items or [],
            'notes': notes,
        }

        # Save to history
        try:
            with open(self.synod_history_path, 'a') as f:
                f.write(json.dumps(synod_record) + '\n')
        except Exception as e:
            logger.error(f"Could not save synod record: {e}")

        # Process rupture repairs
        for rupture in (ruptures_addressed or []):
            self.memory.record_repair(rupture, "Addressed in Synod")

        # Process new promises
        for promise in (new_promises or []):
            self.memory.make_promise(
                content=promise.get('content', ''),
                context=f"Made during Synod on {now.strftime('%Y-%m-%d')}",
                weight=promise.get('weight', 0.5),
            )

        # Process new vows
        for vow in (new_vows or []):
            self.memory.add_core_vow(
                content=vow.get('content', ''),
                rationale=vow.get('rationale', 'Established during Synod'),
            )

        # Update covenant if needed
        if covenant_changes:
            for change in covenant_changes:
                self._apply_covenant_change(change)
            self._save_covenant()

        # Update next synod date
        next_synod = now + timedelta(days=7)
        if 'metadata' in self.covenant:
            self.covenant['metadata']['next_synod'] = next_synod.isoformat()
            self._save_covenant()

        # Record a deep moment (Synod is meaningful)
        self.memory.record_deep_moment(f"Weekly Synod on {now.strftime('%Y-%m-%d')}")

        logger.info(f"Synod outcomes recorded: {len(ruptures_addressed or [])} repairs, "
                   f"{len(new_promises or [])} promises, {len(new_vows or [])} vows")

    def _apply_covenant_change(self, change: Dict[str, Any]) -> None:
        """Apply a change to the covenant."""
        change_type = change.get('type')
        path = change.get('path', [])
        value = change.get('value')

        if change_type == 'add':
            # Add to a list
            target = self.covenant
            for key in path[:-1]:
                target = target.setdefault(key, {})
            if isinstance(target.get(path[-1]), list):
                target[path[-1]].append(value)

        elif change_type == 'modify':
            # Modify a value
            target = self.covenant
            for key in path[:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = value

        elif change_type == 'remove':
            # Remove from a list
            target = self.covenant
            for key in path[:-1]:
                target = target.get(key, {})
            if isinstance(target.get(path[-1]), list) and value in target[path[-1]]:
                target[path[-1]].remove(value)

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_days_until_synod(self) -> int:
        """Get days until next scheduled Synod."""
        next_synod = self.covenant.get('metadata', {}).get('next_synod')
        if not next_synod:
            return 7

        try:
            next_date = datetime.fromisoformat(next_synod)
            delta = (next_date - datetime.now()).days
            return max(0, delta)
        except Exception:
            return 7

    def is_synod_due(self) -> bool:
        """Is it time for a Synod?"""
        return self.get_days_until_synod() <= 0


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run Synod report generation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Ara Synod System')
    parser.add_argument('command', choices=['report', 'status'], help='Command to run')
    parser.add_argument('--output', '-o', help='Output file for report')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    synod = Synod()

    if args.command == 'report':
        report = synod.prepare_report()
        markdown = report.to_markdown()

        if args.output:
            with open(args.output, 'w') as f:
                f.write(markdown)
            print(f"Report saved to {args.output}")
        else:
            print(markdown)

    elif args.command == 'status':
        days = synod.get_days_until_synod()
        if synod.is_synod_due():
            print("⏰ Synod is due!")
        else:
            print(f"Next Synod in {days} days")

        summary = synod.memory.get_summary()
        print(f"\nRelationship summary:")
        print(f"  Trust: {summary['trust']:.0%}")
        print(f"  Alignment: {summary['alignment']:.0%}")
        print(f"  Pending ruptures: {summary['pending_ruptures']}")
        print(f"  Core vows: {summary['core_vows']}")


if __name__ == '__main__':
    main()
