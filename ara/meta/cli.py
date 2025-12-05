"""CLI for Ara Meta-Learning Layer.

Provides commands:
- ara-meta status: Show meta-learning status
- ara-meta suggest: Get and review suggestions
- ara-meta copilot: Interactive workflow selection
- ara-meta patterns: Manage pattern cards
- ara-meta analyze: Run pattern analysis
- ara-meta agenda: View research agenda
- ara-meta plan: Plan workflow using shadow models
- ara-meta shadow: Manage shadow teacher profiles
- ara-meta curiosity: View curiosity hotspots
- ara-meta research: View research programs
- ara-meta experiments: Manage experiments
- ara-meta templates: Manage prompt templates
- ara-meta reflect: Run self-reflection
- ara-meta playbook: Generate teacher playbooks
- ara-meta capsules: Manage skill capsules
- ara-meta forge: Manage agent blueprints
- ara-meta tournament: Run agent tournaments
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


def cmd_research(args: argparse.Namespace) -> int:
    """View research programs."""
    from .research.programs import get_program_manager, seed_default_programs

    pm = get_program_manager()

    if args.seed:
        count = seed_default_programs(pm)
        print(f"Seeded {count} default research programs.")
        return 0

    if args.show:
        prog = pm.get_program(args.show)
        if not prog:
            print(f"Program not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(prog.to_dict(), indent=2, default=str))
        else:
            print(f"Program: {prog.name}")
            print(f"ID: {prog.id}")
            print(f"Goal: {prog.goal}")
            print(f"Status: {prog.status}")
            print(f"Episodes: {prog.episode_count}")
            print()
            print("Hypotheses:")
            for h in prog.hypotheses:
                print(f"  [{h.id}] {h.statement}")
                print(f"      Status: {h.status}")
                if h.control_avg is not None:
                    print(f"      Control: {h.control_avg:.0%} (N={h.control_samples})")
                if h.treatment_avg is not None:
                    print(f"      Treatment: {h.treatment_avg:.0%} (N={h.treatment_samples})")
                if h.effect_size is not None:
                    print(f"      Effect size: {h.effect_size:+.0%}")
                print()
        return 0

    # Default: show summary
    summary = pm.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Research Programs")
        print("=" * 40)
        print(f"Total programs: {summary['total_programs']}")
        print(f"Active: {summary['active_programs']}")
        print(f"Total hypotheses: {summary['total_hypotheses']}")
        print(f"Active hypotheses: {summary['active_hypotheses']}")
        print()

        if summary['programs']:
            print("Programs:")
            for p in summary['programs']:
                status_mark = "●" if p['status'] == 'active' else "○"
                print(f"  {status_mark} {p['id']}: {p['name']}")
                print(f"      Episodes: {p['episode_count']}, Hypotheses: {p['hypotheses']}")
        else:
            print("No programs found. Run with --seed to create defaults.")

    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    """Manage experiments."""
    from .research.experiments import get_experiment_controller

    controller = get_experiment_controller()

    if args.show:
        exp = controller.get_experiment(args.show)
        if not exp:
            print(f"Experiment not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(exp.to_dict(), indent=2, default=str))
        else:
            print(f"Experiment: {exp.id}")
            print(f"Description: {exp.description}")
            print(f"Program: {exp.program_id}")
            print(f"Hypothesis: {exp.hypothesis_id}")
            print(f"Status: {exp.status}")
            print(f"Trigger: {exp.trigger_intent}")
            print()
            print("Arms:")
            for arm in exp.arms:
                print(f"  [{arm.name}] {' → '.join(arm.workflow)}")
                print(f"    Assignments: {arm.assignments}")
                print(f"    Completions: {arm.completions}")
                if arm.avg_reward is not None:
                    print(f"    Avg reward: {arm.avg_reward:.0%}")
            print()
            print(f"Ready for conclusion: {exp.is_ready_for_conclusion()}")
        return 0

    # Default: show summary
    summary = controller.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Experiments")
        print("=" * 40)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Active: {summary['active']}")
        print(f"Concluded: {summary['concluded']}")
        print()

        if summary['experiments']:
            print("Active experiments:")
            active = [e for e in summary['experiments'] if e['status'] == 'active']
            for e in active:
                print(f"  [{e['id']}] {e['description'][:40]}...")
                print(f"    Arms: {e['arms']}, Completions: {e['total_completions']}")
        else:
            print("No experiments found.")

    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    """Manage prompt templates."""
    from .research.templates import get_template_learner, seed_default_templates

    learner = get_template_learner()

    if args.seed:
        count = seed_default_templates(learner)
        print(f"Seeded {count} default templates.")
        return 0

    if args.teacher:
        templates = learner.get_templates_for(args.teacher, "*")
        if not templates:
            print(f"No templates found for teacher: {args.teacher}")
            return 0

        if args.json:
            print(json.dumps([t.to_dict() for t in templates], indent=2, default=str))
        else:
            print(f"Templates for {args.teacher}:")
            print("-" * 40)
            for t in templates:
                print(f"  [{t.id}] {t.name}")
                print(f"    Intent: {t.intent}")
                print(f"    Style: {', '.join(t.tags)}")
                if t.success_rate is not None:
                    print(f"    Success: {t.success_rate:.0%} (N={t.sample_count})")
                print()
        return 0

    if args.show:
        template = learner.get_template(args.show)
        if not template:
            print(f"Template not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(template.to_dict(), indent=2, default=str))
        else:
            print(f"Template: {template.id}")
            print(f"Name: {template.name}")
            print(f"Teacher: {template.teacher}")
            print(f"Intent: {template.intent}")
            print(f"Version: v{template.version}")
            if template.parent_id:
                print(f"Evolved from: {template.parent_id}")
            print()
            print("Skeleton:")
            print("-" * 40)
            print(template.skeleton[:500])
            print("-" * 40)
            print()
            print(f"Sections: {', '.join(template.sections)}")
            print(f"Tags: {', '.join(template.tags)}")
            if template.success_rate is not None:
                print(f"Success rate: {template.success_rate:.0%}")
            print(f"Samples: {template.sample_count}")
        return 0

    # Default: show summary
    summary = learner.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Prompt Templates")
        print("=" * 40)
        print(f"Total templates: {summary['total_templates']}")
        print()
        print("By teacher:")
        for teacher, count in summary['by_teacher'].items():
            print(f"  {teacher}: {count}")
        print()
        print("By intent:")
        for intent, count in summary['by_intent'].items():
            print(f"  {intent}: {count}")
        print()

        if summary['top_templates']:
            print("Top performing:")
            for t in summary['top_templates']:
                rate = t['success_rate']
                rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                print(f"  {t['id']}: {rate_str} ({t['samples']} samples)")

    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    """Run self-reflection."""
    from .research.self_reflection import get_self_reflector

    reflector = get_self_reflector()

    if args.run:
        days = args.days or 7
        print(f"Running reflection for the last {days} days...")
        episode = reflector.create_reflection(period_days=days)

        if args.json:
            print(json.dumps(episode.to_dict(), indent=2, default=str))
        else:
            print()
            print(episode.narrative)
            print()
            print(f"Reflection ID: {episode.id}")
            print(f"Interactions analyzed: {episode.interactions_analyzed}")
            print(f"Insights generated: {len(episode.insights)}")
            if episode.success_rate is not None:
                print(f"Overall success rate: {episode.success_rate:.0%}")
        return 0

    if args.actionable:
        insights = reflector.get_actionable_insights(limit=args.limit)
        if not insights:
            print("No actionable insights found.")
            return 0

        if args.json:
            print(json.dumps([i.to_dict() for i in insights], indent=2, default=str))
        else:
            print("Actionable Insights")
            print("=" * 40)
            for i in insights:
                priority_mark = {"high": "!!", "medium": "!", "low": " "}[i.priority]
                print(f"[{priority_mark}] {i.summary}")
                print(f"    → {i.suggested_action}")
                print(f"    Confidence: {i.confidence:.0%}")
                print()
        return 0

    # Default: show summary
    latest = reflector.get_latest_reflection()

    if args.json:
        print(json.dumps(reflector.get_summary(), indent=2, default=str))
    else:
        summary = reflector.get_summary()
        print("Self-Reflection")
        print("=" * 40)
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Total insights: {summary['total_insights']}")
        print(f"Actionable: {summary['actionable_insights']}")
        print()

        if latest:
            print("Latest reflection:")
            print(f"  ID: {latest.id}")
            print(f"  Period: {latest.period_start.date()} to {latest.period_end.date()}")
            print(f"  Interactions: {latest.interactions_analyzed}")
            print(f"  Insights: {len(latest.insights)}")
            print()
            print("Run with --run to create a new reflection.")
        else:
            print("No reflections yet. Run with --run to create one.")

    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    """Generate teacher playbooks."""
    from .research.playbook import get_playbook_generator

    generator = get_playbook_generator()

    if args.generate:
        teacher = args.generate
        print(f"Generating playbook for {teacher}...")
        playbook = generator.generate_playbook(teacher, force_regenerate=True)

        if args.json:
            print(json.dumps(playbook.to_dict(), indent=2, default=str))
        else:
            print()
            print(playbook.format_markdown())
        return 0

    if args.generate_all:
        print("Generating playbooks for all teachers...")
        playbooks = generator.generate_all_playbooks()
        print(f"Generated {len(playbooks)} playbooks.")

        if args.export:
            paths = generator.export_markdown()
            print(f"Exported to:")
            for p in paths:
                print(f"  {p}")
        return 0

    if args.show:
        playbook = generator.get_playbook(args.show)
        if not playbook:
            print(f"Playbook not found: {args.show}")
            print("Run with --generate <teacher> to create one.")
            return 1

        if args.json:
            print(json.dumps(playbook.to_dict(), indent=2, default=str))
        else:
            print(playbook.format_markdown())
        return 0

    if args.export:
        paths = generator.export_markdown()
        if paths:
            print(f"Exported playbooks:")
            for p in paths:
                print(f"  {p}")
        else:
            print("No playbooks to export. Run --generate-all first.")
        return 0

    # Default: show summary
    summary = generator.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Teacher Playbooks")
        print("=" * 40)
        print(f"Total playbooks: {summary['total_playbooks']}")
        print()

        if summary['playbooks']:
            for pb in summary['playbooks']:
                rate = pb['success_rate']
                rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                print(f"  {pb['teacher'].title()} (v{pb['version']})")
                print(f"    Interactions: {pb['interactions']}")
                print(f"    Success rate: {rate_str}")
                print(f"    Strengths: {pb['strengths']}, Weaknesses: {pb['weaknesses']}")
                print()
        else:
            print("No playbooks generated yet.")
            print("Run --generate <teacher> or --generate-all to create them.")

    return 0


def cmd_capsules(args: argparse.Namespace) -> int:
    """Manage skill capsules."""
    from .toolsmith.capsules import get_capsule_manager, seed_default_capsules

    manager = get_capsule_manager()

    if args.seed:
        count = seed_default_capsules(manager)
        print(f"Seeded {count} default skill capsules.")
        return 0

    if args.show:
        capsule = manager.get_capsule(args.show)
        if not capsule:
            print(f"Capsule not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(capsule.to_dict(), indent=2, default=str))
        else:
            print(capsule.format_yaml())
        return 0

    if args.find:
        matches = manager.find_matching_capsules(args.find, limit=5)
        if not matches:
            print("No matching capsules found.")
            return 0

        print(f"Capsules matching: '{args.find}'")
        print("-" * 40)
        for capsule, confidence in matches:
            print(f"  [{capsule.id}] {capsule.name}")
            print(f"    Confidence: {confidence:.0%}")
            print(f"    Intent: {capsule.intent}")
            print(f"    Success rate: {capsule.success_rate:.0%}")
            print()
        return 0

    # Default: show summary
    summary = manager.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Skill Capsules")
        print("=" * 40)
        print(f"Total capsules: {summary['total_capsules']}")
        print(f"Active: {summary['active']}")
        print(f"Drafts: {summary['drafts']}")
        print(f"Deprecated: {summary['deprecated']}")
        print()

        if summary['top_performers']:
            print("Top performers:")
            for cap in summary['top_performers']:
                rate = cap['success_rate']
                rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                print(f"  {cap['name']}: {rate_str} ({cap['samples']} samples)")
        else:
            print("No capsules yet. Run --seed to create defaults.")

    return 0


def cmd_forge(args: argparse.Namespace) -> int:
    """Manage agent blueprints."""
    from .toolsmith.forge import get_agent_forge, seed_default_blueprints

    forge = get_agent_forge()

    if args.seed:
        count = seed_default_blueprints(forge)
        print(f"Seeded {count} default agent blueprints.")
        return 0

    if args.show:
        blueprint = forge.get_blueprint(args.show)
        if not blueprint:
            print(f"Blueprint not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(blueprint.to_dict(), indent=2, default=str))
        else:
            print(blueprint.format_spec())
        return 0

    if args.list:
        blueprints = forge.get_all_blueprints()
        if not blueprints:
            print("No blueprints found. Run --seed to create defaults.")
            return 0

        if args.json:
            print(json.dumps([bp.to_dict() for bp in blueprints], indent=2, default=str))
        else:
            print("Agent Blueprints")
            print("-" * 40)
            for bp in blueprints:
                status_mark = {"active": "●", "draft": "○", "archived": "✗"}[bp.status]
                print(f"  {status_mark} [{bp.id}] {bp.name}")
                print(f"    Purpose: {bp.purpose[:50]}...")
                if bp.success_rate is not None:
                    print(f"    Success: {bp.success_rate:.0%} ({bp.total_invocations} invocations)")
                print()
        return 0

    # Default: show summary
    summary = forge.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Agent Forge")
        print("=" * 40)
        print(f"Total blueprints: {summary['total_blueprints']}")
        print(f"Active: {summary['active']}")
        print(f"Drafts: {summary['drafts']}")
        print()

        if summary['top_performers']:
            print("Top performers:")
            for bp in summary['top_performers']:
                rate = bp['success_rate']
                rate_str = f"{rate:.0%}" if rate is not None else "N/A"
                print(f"  {bp['name']}: {rate_str} ({bp['invocations']} invocations)")
        else:
            print("No blueprints yet. Run --seed to create defaults.")

    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    """Run agent tournaments."""
    from .toolsmith.tournaments import get_tournament_manager, seed_default_benchmarks

    manager = get_tournament_manager()

    if args.seed_benchmarks:
        count = seed_default_benchmarks(manager)
        print(f"Seeded {count} default benchmarks.")
        return 0

    if args.benchmarks:
        benchmarks = manager.get_all_benchmarks()
        if not benchmarks:
            print("No benchmarks found. Run --seed-benchmarks to create defaults.")
            return 0

        if args.json:
            print(json.dumps([b.to_dict() for b in benchmarks], indent=2, default=str))
        else:
            print("Benchmarks")
            print("-" * 40)
            for bm in benchmarks:
                print(f"  [{bm.id}] {bm.name}")
                print(f"    Tasks: {len(bm.tasks)}")
                print(f"    Tags: {', '.join(bm.tags)}")
                print()
        return 0

    if args.show:
        tournament = manager.get_tournament(args.show)
        if not tournament:
            print(f"Tournament not found: {args.show}")
            return 1

        if args.json:
            print(json.dumps(tournament.to_dict(), indent=2, default=str))
        else:
            print(tournament.format_leaderboard())
        return 0

    if args.simulate:
        # Simulate a tournament
        parts = args.simulate.split(":")
        if len(parts) != 2:
            print("Usage: --simulate BENCHMARK_ID:participant1,participant2,...")
            return 1

        benchmark_id, participants_str = parts
        participants = [p.strip() for p in participants_str.split(",")]

        benchmark = manager.get_benchmark(benchmark_id)
        if not benchmark:
            print(f"Benchmark not found: {benchmark_id}")
            print("Run --seed-benchmarks to create defaults.")
            return 1

        print(f"Simulating tournament with {len(participants)} participants...")
        tournament = manager.create_tournament(
            name=f"Simulated: {benchmark.name}",
            description=f"Simulated tournament for {benchmark_id}",
            benchmark_id=benchmark_id,
            participants=participants,
        )

        winner = manager.simulate_tournament(tournament.id)
        print()
        print(tournament.format_leaderboard())
        return 0

    # Default: show summary
    summary = manager.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("Tournaments")
        print("=" * 40)
        print(f"Total tournaments: {summary['total_tournaments']}")
        print(f"Completed: {summary['completed']}")
        print(f"Benchmarks available: {summary['total_benchmarks']}")
        print()

        if summary['recent_winners']:
            print("Recent winners:")
            for w in summary['recent_winners']:
                score = w['score']
                score_str = f"{score:.0%}" if score is not None else "N/A"
                print(f"  {w['tournament']}: {w['winner']} ({score_str})")
        else:
            print("No completed tournaments yet.")
            print("Run --simulate BENCHMARK:participant1,participant2 to simulate one.")

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

    # Research command (Research Director)
    research_parser = subparsers.add_parser("research", help="View research programs")
    research_parser.add_argument("--show", type=str, help="Show a specific program")
    research_parser.add_argument("--seed", action="store_true", help="Seed default programs")
    research_parser.add_argument("--json", action="store_true", help="Output as JSON")
    research_parser.set_defaults(func=cmd_research)

    # Experiments command
    experiments_parser = subparsers.add_parser("experiments", help="Manage experiments")
    experiments_parser.add_argument("--show", type=str, help="Show a specific experiment")
    experiments_parser.add_argument("--json", action="store_true", help="Output as JSON")
    experiments_parser.set_defaults(func=cmd_experiments)

    # Templates command
    templates_parser = subparsers.add_parser("templates", help="Manage prompt templates")
    templates_parser.add_argument("--teacher", type=str, help="Filter by teacher")
    templates_parser.add_argument("--show", type=str, help="Show a specific template")
    templates_parser.add_argument("--seed", action="store_true", help="Seed default templates")
    templates_parser.add_argument("--json", action="store_true", help="Output as JSON")
    templates_parser.set_defaults(func=cmd_templates)

    # Reflect command
    reflect_parser = subparsers.add_parser("reflect", help="Run self-reflection")
    reflect_parser.add_argument("--run", action="store_true", help="Run a new reflection")
    reflect_parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    reflect_parser.add_argument("--actionable", action="store_true",
                                help="Show actionable insights")
    reflect_parser.add_argument("--limit", type=int, default=10, help="Max items to show")
    reflect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    reflect_parser.set_defaults(func=cmd_reflect)

    # Playbook command
    playbook_parser = subparsers.add_parser("playbook", help="Generate teacher playbooks")
    playbook_parser.add_argument("--generate", type=str, help="Generate for specific teacher")
    playbook_parser.add_argument("--generate-all", action="store_true",
                                 help="Generate for all teachers")
    playbook_parser.add_argument("--show", type=str, help="Show a specific playbook")
    playbook_parser.add_argument("--export", action="store_true", help="Export as markdown")
    playbook_parser.add_argument("--json", action="store_true", help="Output as JSON")
    playbook_parser.set_defaults(func=cmd_playbook)

    # Capsules command (Toolsmith)
    capsules_parser = subparsers.add_parser("capsules", help="Manage skill capsules")
    capsules_parser.add_argument("--show", type=str, help="Show a specific capsule")
    capsules_parser.add_argument("--find", type=str, help="Find capsules matching query")
    capsules_parser.add_argument("--seed", action="store_true", help="Seed default capsules")
    capsules_parser.add_argument("--json", action="store_true", help="Output as JSON")
    capsules_parser.set_defaults(func=cmd_capsules)

    # Forge command (Toolsmith)
    forge_parser = subparsers.add_parser("forge", help="Manage agent blueprints")
    forge_parser.add_argument("--show", type=str, help="Show a specific blueprint")
    forge_parser.add_argument("--list", action="store_true", help="List all blueprints")
    forge_parser.add_argument("--seed", action="store_true", help="Seed default blueprints")
    forge_parser.add_argument("--json", action="store_true", help="Output as JSON")
    forge_parser.set_defaults(func=cmd_forge)

    # Tournament command (Toolsmith)
    tournament_parser = subparsers.add_parser("tournament", help="Run agent tournaments")
    tournament_parser.add_argument("--show", type=str, help="Show a specific tournament")
    tournament_parser.add_argument("--benchmarks", action="store_true", help="List benchmarks")
    tournament_parser.add_argument("--seed-benchmarks", action="store_true",
                                   help="Seed default benchmarks")
    tournament_parser.add_argument("--simulate", type=str,
                                   help="Simulate tournament: BENCHMARK:participant1,participant2")
    tournament_parser.add_argument("--json", action="store_true", help="Output as JSON")
    tournament_parser.set_defaults(func=cmd_tournament)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
