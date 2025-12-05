"""CLI for Ara Meta-Learning Layer.

Provides commands:
- ara meta status: Show meta-learning status
- ara meta suggest: Get and review suggestions
- ara meta analyze: Run pattern analysis
- ara meta agenda: View research agenda
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .meta_brain import get_meta_brain
from .pattern_miner import get_miner
from .natural_prompts import (
    verbalize_status,
    verbalize_suggestion,
    verbalize_recommendations,
    verbalize_research_agenda,
)
from .config import get_meta_config, create_default_config_file


def cmd_status(args: argparse.Namespace) -> int:
    """Show meta-learning status."""
    brain = get_meta_brain()
    status = brain.get_status()

    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print(verbalize_status(status))

    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Get pattern-based suggestions."""
    brain = get_meta_brain()

    # Refresh if requested
    if args.refresh:
        new = brain.refresh_suggestions()
        if new:
            print(f"Found {len(new)} new suggestions.\n")

    # Get pending suggestions
    suggestions = brain.get_pending_suggestions(
        min_confidence=args.min_confidence,
    )

    if not suggestions:
        print("No pending suggestions.")
        return 0

    print(f"Pending suggestions ({len(suggestions)}):\n")

    for i, s in enumerate(suggestions[:args.limit], 1):
        print(f"[{i}] {s.id}")
        print(verbalize_suggestion(s))
        print()

    # Interactive mode
    if args.interactive:
        return _interactive_suggestions(brain, suggestions)

    return 0


def _interactive_suggestions(brain, suggestions) -> int:
    """Interactive suggestion review."""
    for s in suggestions:
        print(f"\n{'=' * 40}")
        print(f"Suggestion: {s.id}")
        print(verbalize_suggestion(s))
        print(f"\nConfidence: {s.confidence:.0%}")
        print(f"Scope: {s.scope}")
        print(f"Auto-apply safe: {s.safe_to_auto_apply}")

        while True:
            choice = input("\n[a]pply / [r]eject / [s]kip / [q]uit? ").lower().strip()

            if choice == 'a':
                brain.apply_suggestion(s.id)
                print("Applied.")
                break
            elif choice == 'r':
                feedback = input("Feedback (optional): ").strip()
                brain.reject_suggestion(s.id, feedback)
                print("Rejected.")
                break
            elif choice == 's':
                print("Skipped.")
                break
            elif choice == 'q':
                return 0

    print("\nDone reviewing suggestions.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run pattern analysis."""
    miner = get_miner()
    analysis = miner.analyze(days=args.days, force=True)

    if args.json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(f"Analysis complete (last {args.days} days)\n")

        tools = analysis.get("tools", {})
        if tools:
            print("Tool performance:")
            for name, stats in tools.items():
                print(f"  {name}: {stats['success_rate']:.0%} success, "
                      f"{stats['total_calls']} calls")

        golden = analysis.get("golden_paths", [])
        if golden:
            print(f"\nGolden paths ({len(golden)}):")
            for p in golden:
                print(f"  {p['pattern']} ({p['success_rate']:.0%})")

        strategies = analysis.get("strategies", {})
        if strategies:
            print(f"\nStrategies:")
            for name, stats in strategies.items():
                print(f"  {name}: {stats['success_rate']:.0%} ({stats['uses']} uses)")

    return 0


def cmd_agenda(args: argparse.Namespace) -> int:
    """View research agenda."""
    brain = get_meta_brain()

    if args.json:
        print(json.dumps(brain.agenda.model_dump(), indent=2, default=str))
    else:
        print(verbalize_research_agenda(brain.agenda))

    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Get current recommendations."""
    brain = get_meta_brain()
    recs = brain.get_recommendations()

    if args.json:
        print(json.dumps(recs, indent=2, default=str))
    else:
        print(verbalize_recommendations(recs))

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show or create configuration."""
    if args.create:
        path = create_default_config_file()
        print(f"Created default config at: {path}")
        return 0

    config = get_meta_config()
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ara-meta",
        description="Ara Meta-Learning Layer CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show meta-learning status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.set_defaults(func=cmd_status)

    # Suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Get suggestions")
    suggest_parser.add_argument("--refresh", action="store_true",
                                help="Refresh suggestions from analysis")
    suggest_parser.add_argument("--min-confidence", type=float, default=0.5,
                                help="Minimum confidence threshold")
    suggest_parser.add_argument("--limit", type=int, default=10,
                                help="Max suggestions to show")
    suggest_parser.add_argument("-i", "--interactive", action="store_true",
                                help="Interactive review mode")
    suggest_parser.set_defaults(func=cmd_suggest)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run pattern analysis")
    analyze_parser.add_argument("--days", type=int, default=30,
                                help="Days of history to analyze")
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Agenda command
    agenda_parser = subparsers.add_parser("agenda", help="View research agenda")
    agenda_parser.add_argument("--json", action="store_true", help="Output as JSON")
    agenda_parser.set_defaults(func=cmd_agenda)

    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get recommendations")
    recommend_parser.add_argument("--json", action="store_true", help="Output as JSON")
    recommend_parser.set_defaults(func=cmd_recommend)

    # Config command
    config_parser = subparsers.add_parser("config", help="Show/create configuration")
    config_parser.add_argument("--create", action="store_true",
                               help="Create default config file")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
