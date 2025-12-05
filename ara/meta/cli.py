"""CLI for Ara Meta-Learning Layer.

Provides commands:
- ara-meta status: Show meta-learning status
- ara-meta suggest: Get and review suggestions
- ara-meta copilot: Interactive workflow selection
- ara-meta patterns: Manage pattern cards
- ara-meta analyze: Run pattern analysis
- ara-meta agenda: View research agenda
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, List

from .meta_brain import get_meta_brain
from .pattern_miner import get_miner
from .pattern_cards import get_pattern_manager, seed_default_patterns
from .copilot import CoPilot, WorkflowProposal, InteractiveSession
from .reflection import classify_intent
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

    # Also get pattern card status
    pm = get_pattern_manager()
    pattern_status = pm.get_status_summary()

    if args.json:
        combined = {**status, "pattern_cards": pattern_status}
        print(json.dumps(combined, indent=2, default=str))
    else:
        print(verbalize_status(status))
        print()

        # Pattern cards section
        if pattern_status["golden_paths"] > 0:
            print("GOLDEN PATHS")
            for p in pattern_status["golden_list"]:
                print(f"  {p['id']}: {p['success_rate']:.0%} (N={p['samples']})")
            print()

        if pattern_status["experimental"] > 0:
            print("EXPERIMENTAL")
            for p in pattern_status["experimental_list"]:
                print(f"  {p['id']}: {p['success_rate']:.0%} (N={p['samples']})")

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


def cmd_copilot(args: argparse.Namespace) -> int:
    """Interactive workflow co-pilot."""
    task = args.task

    if not task:
        # Interactive mode - ask for task
        task = input("What are you trying to do? > ").strip()
        if not task:
            print("No task provided.")
            return 1

    # Detect intent
    intent = classify_intent(task)
    print(f"\nDetected intent: {intent}")

    # Create session
    session = InteractiveSession()
    message, proposals = session.start(task)

    print()
    print(message)

    if not proposals:
        # Auto-picked
        summary = session.get_execution_summary()
        if summary:
            print(f"\nReady to execute with: {' → '.join(summary['teachers'])}")
        return 0

    # Interactive selection
    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            print("Cancelled.")
            return 0
        elif choice == 'a':
            # Auto-pick best
            session.select(1, proposals)
            break
        else:
            try:
                idx = int(choice)
                if session.select(idx, proposals):
                    break
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Enter a number, 'a' for auto, or 'q' to quit.")

    # Show what will be executed
    summary = session.get_execution_summary()
    if summary:
        print(f"\nSelected: {' → '.join(summary['teachers'])}")
        print(f"Pattern: {summary['pattern_id']}")
        print(f"Confidence: {summary['confidence']:.0%}")

    return 0


def cmd_patterns(args: argparse.Namespace) -> int:
    """Manage pattern cards."""
    pm = get_pattern_manager()

    if args.seed:
        count = seed_default_patterns(pm)
        print(f"Seeded {count} default patterns.")
        return 0

    if args.list:
        cards = pm.get_all_cards()
        if not cards:
            print("No pattern cards found. Run with --seed to create defaults.")
            return 0

        if args.json:
            print(json.dumps([c.to_yaml_dict() for c in cards], indent=2, default=str))
        else:
            golden = [c for c in cards if c.status == "golden"]
            experimental = [c for c in cards if c.status == "experimental"]
            deprecated = [c for c in cards if c.status == "deprecated"]

            if golden:
                print("GOLDEN PATHS")
                print("-" * 40)
                for c in golden:
                    print(f"  {c.id}")
                    print(f"    Intent: {c.intent}")
                    print(f"    Teachers: {' → '.join(c.teachers)}")
                    print(f"    Success: {c.success_rate:.0%} (N={c.sample_count})")
                    print()

            if experimental:
                print("EXPERIMENTAL")
                print("-" * 40)
                for c in experimental:
                    print(f"  {c.id}")
                    print(f"    Intent: {c.intent}")
                    print(f"    Teachers: {' → '.join(c.teachers)}")
                    print(f"    Success: {c.success_rate:.0%} (N={c.sample_count})")
                    print()

            if deprecated:
                print(f"DEPRECATED ({len(deprecated)} cards)")

        return 0

    if args.show:
        card = pm.get_card(args.show)
        if not card:
            print(f"Pattern card not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(card.to_yaml_dict(), indent=2, default=str))
        else:
            print(f"Pattern: {card.id}")
            print(f"Status: {card.status}")
            print(f"Intent: {card.intent}")
            print(f"Description: {card.description}")
            print(f"Teachers: {' → '.join(card.teachers)}")
            print()
            print("Sequence:")
            for step in card.sequence:
                print(f"  - {step.call} ({step.role})")
                if step.style_hint:
                    print(f"    Style: {step.style_hint}")
            print()
            print(f"Success rate: {card.success_rate:.0%}")
            print(f"Sample count: {card.sample_count}")
            if card.avg_latency_sec:
                print(f"Avg latency: {card.avg_latency_sec:.1f}s")
        return 0

    # Default: show summary
    status = pm.get_status_summary()
    print(f"Total pattern cards: {status['total_cards']}")
    print(f"  Golden: {status['golden_paths']}")
    print(f"  Experimental: {status['experimental']}")
    print(f"  Deprecated: {status['deprecated']}")
    print()
    print("Use --list to see all patterns, --seed to create defaults.")
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


def cmd_plan(args: argparse.Namespace) -> int:
    """Plan workflow using shadow models."""
    from .shadow.planner import get_planner, WorkflowPlanner
    from .shadow.profiles import seed_default_profiles, get_profile_manager

    # Seed profiles if needed
    pm = get_profile_manager()
    seed_default_profiles(pm)

    planner = get_planner()
    query = args.query

    if not query:
        query = input("What are you trying to do? > ").strip()
        if not query:
            print("No query provided.")
            return 1

    # Plan
    intent = args.intent
    plan = planner.plan_from_query(query, intent=intent, mode=args.mode)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, default=str))
    else:
        print(f"\nIntent: {plan.intent}")
        print(f"Mode: {plan.planning_mode}")
        print()
        print("Candidate workflows (simulated):")
        print()

        # Show chosen
        print(f"[CHOSEN] {' → '.join(plan.teachers)}")
        print(f"  expected_reward: {plan.expected_reward:.0%}")
        print(f"  expected_latency: {plan.expected_latency_sec:.1f}s")
        print(f"  confidence: {plan.confidence:.0%}")
        print()

        # Show alternatives
        for i, alt in enumerate(plan.alternatives[:2], 1):
            print(f"[ALT {i}] {' → '.join(alt.teachers)}")
            print(f"  expected_reward: {alt.expected_reward:.0%}")
            print(f"  expected_latency: {alt.expected_latency_sec:.1f}s")
            if alt.notes:
                print(f"  notes: {alt.notes[0]}")
            print()

        print(f"Chosen: {' → '.join(plan.teachers)} (policy {plan.policy_version})")

    return 0


def cmd_shadow(args: argparse.Namespace) -> int:
    """Manage shadow teacher profiles."""
    from .shadow.profiles import get_profile_manager, seed_default_profiles

    pm = get_profile_manager()

    if args.seed:
        count = seed_default_profiles(pm)
        print(f"Seeded {count} default profiles.")
        return 0

    if args.teacher:
        profiles = pm.get_profiles_for_teacher(args.teacher)
        if not profiles:
            print(f"No profiles found for teacher: {args.teacher}")
            return 0

        if args.json:
            print(json.dumps([p.to_dict() for p in profiles], indent=2, default=str))
        else:
            print(f"Profiles for {args.teacher}:")
            print("-" * 40)
            for p in profiles:
                print(f"  Intent: {p.intent}")
                print(f"    Success rate: {p.success_rate:.0%}")
                print(f"    Avg reward: {p.avg_reward:.0%}")
                print(f"    Avg latency: {p.avg_latency_sec:.1f}s")
                print(f"    Samples: {p.sample_count}")
                print()
        return 0

    # Default: show status
    summary = pm.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Shadow Teacher Profiles")
        print("=" * 40)
        print(f"Total profiles: {summary['total_profiles']}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Teachers: {', '.join(summary['teachers'])}")
        print(f"Intents: {', '.join(summary['intents'])}")
        print()
        print("Run with --teacher <name> to see specific profiles.")
        print("Run with --seed to initialize default profiles.")

    return 0


def cmd_curiosity(args: argparse.Namespace) -> int:
    """View curiosity hotspots (shadow model disagreements)."""
    from .shadow.disagreement import get_disagreement_tracker

    tracker = get_disagreement_tracker()
    hotspots = tracker.get_curiosity_hotspots(limit=args.limit)

    if args.json:
        print(json.dumps([h.to_dict() for h in hotspots], indent=2, default=str))
    else:
        if not hotspots:
            print("No curiosity hotspots found.")
            print("(These appear when shadow models disagree on predictions)")
            return 0

        print("Curiosity Hotspots")
        print("=" * 40)
        print("(Regions where shadow models strongly disagree)")
        print()

        for h in hotspots:
            print(f"[{h.id}] Intent: {h.intent}")
            print(f"  Disagreement score: {h.disagreement_score:.0%}")
            print(f"  Plan A: {h.plan_a} (predicted {h.plan_a_reward:.0%})")
            print(f"  Plan B: {h.plan_b} (predicted {h.plan_b_reward:.0%})")
            print(f"  Status: {h.status}")
            if h.query_summary:
                print(f"  Context: {h.query_summary[:50]}...")
            print()

        summary = tracker.get_summary()
        print(f"Total: {summary['total_records']} recorded, "
              f"{summary['open']} open, {summary['resolved']} resolved")

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

    # Co-pilot command
    copilot_parser = subparsers.add_parser("copilot", help="Interactive workflow selection")
    copilot_parser.add_argument("task", nargs="?", default="",
                                help="Task description (or enter interactively)")
    copilot_parser.set_defaults(func=cmd_copilot)

    # Patterns command
    patterns_parser = subparsers.add_parser("patterns", help="Manage pattern cards")
    patterns_parser.add_argument("--list", action="store_true", help="List all patterns")
    patterns_parser.add_argument("--show", type=str, help="Show a specific pattern")
    patterns_parser.add_argument("--seed", action="store_true", help="Seed default patterns")
    patterns_parser.add_argument("--json", action="store_true", help="Output as JSON")
    patterns_parser.set_defaults(func=cmd_patterns)

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

    # Plan command (Shadow Teachers)
    plan_parser = subparsers.add_parser("plan", help="Plan workflow using shadow models")
    plan_parser.add_argument("query", nargs="?", default="",
                             help="Query to plan for (or enter interactively)")
    plan_parser.add_argument("--intent", type=str, help="Override intent classification")
    plan_parser.add_argument("--mode", choices=["cheap", "standard", "thorough"],
                             default="standard", help="Planning mode")
    plan_parser.add_argument("--json", action="store_true", help="Output as JSON")
    plan_parser.set_defaults(func=cmd_plan)

    # Shadow command (manage shadow profiles)
    shadow_parser = subparsers.add_parser("shadow", help="Manage shadow teacher profiles")
    shadow_parser.add_argument("--status", action="store_true", help="Show profile status")
    shadow_parser.add_argument("--seed", action="store_true", help="Seed default profiles")
    shadow_parser.add_argument("--teacher", type=str, help="Show specific teacher profiles")
    shadow_parser.add_argument("--json", action="store_true", help="Output as JSON")
    shadow_parser.set_defaults(func=cmd_shadow)

    # Curiosity command (disagreement hotspots)
    curiosity_parser = subparsers.add_parser("curiosity", help="View curiosity hotspots")
    curiosity_parser.add_argument("--limit", type=int, default=10, help="Max hotspots to show")
    curiosity_parser.add_argument("--json", action="store_true", help="Output as JSON")
    curiosity_parser.set_defaults(func=cmd_curiosity)

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
