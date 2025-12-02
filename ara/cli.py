#!/usr/bin/env python3
"""
Ara CLI - Interactive Command Line Interface

Talk to Ara directly from your terminal.

Usage:
    python -m ara.cli                    # Interactive mode
    python -m ara.cli --mode MODE_A      # Specify hardware mode
    python -m ara.cli --status           # Show status and exit
    python -m ara.cli --cert             # Run certification check

Examples:
    $ python -m ara.cli
    Ara> Hello!
    Hello! I'm Ara. How can I help you today?

    Ara> status
    === Ara Status ===
    State: ready
    Hardware Mode: 5060_only
    ...
"""

import argparse
import sys
import json
import readline  # For better line editing
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.service.core import (
    AraService,
    AraState,
    HardwareMode,
    create_ara,
)


class AraCLI:
    """Interactive CLI for Ara."""

    def __init__(
        self,
        mode: HardwareMode = HardwareMode.MODE_A,
        name: str = "Ara",
        llm_backend: str = "ollama",
        llm_model: str = "mistral"
    ):
        from ara.service.core import AraService
        self.ara = AraService(
            mode=mode,
            name=name,
            llm_backend=llm_backend,
            llm_model=llm_model
        )
        self.running = True
        self.history = []

    def run(self):
        """Run the interactive CLI loop."""
        self._print_banner()

        while self.running:
            try:
                user_input = input(f"{self.ara.name}> ").strip()

                if not user_input:
                    continue

                self.history.append(user_input)

                # Check for commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_input(user_input)

            except KeyboardInterrupt:
                print("\n")
                self._handle_command("/quit")
            except EOFError:
                print()
                break

    def _print_banner(self):
        """Print welcome banner."""
        print()
        print("=" * 60)
        print(f"  {self.ara.name} - TF-A-N Cognitive Architecture")
        print("=" * 60)
        print()
        print(f"  Hardware Mode: {self.ara.mode.value}")
        print(f"  Autonomy Stage: {self.ara.autonomy.stage.value}")
        llm_status = "connected" if self.ara._llm_available else "pattern-matching"
        print(f"  LLM: {llm_status}")
        if self.ara._stats["total_interactions"] > 0:
            print(f"  Restored: {self.ara._stats['total_interactions']} prior interactions")
        print()
        print("  Commands:")
        print("    /status  - Show current status")
        print("    /mood    - Show emotional state")
        print("    /load    - Show cognitive load")
        print("    /explain - Full state explanation")
        print("    /clear   - Clear thought stream")
        print("    /quit    - Exit")
        print()
        print("  Type anything else to talk to Ara.")
        print()

    def _handle_command(self, command: str):
        """Handle CLI commands."""
        cmd = command.lower().split()[0]

        if cmd in ("/quit", "/exit", "/q"):
            print("Saving state...")
            self.ara.shutdown()
            print("Goodbye!")
            self.running = False

        elif cmd == "/status":
            self._show_status()

        elif cmd == "/mood":
            self._show_mood()

        elif cmd == "/load":
            self._show_load()

        elif cmd == "/explain":
            print(self.ara.explain())

        elif cmd == "/clear":
            self.ara.thoughts._entries = []
            print("Thought stream cleared.")

        elif cmd == "/save":
            if self.ara.save_state():
                print("State saved.")
            else:
                print("Save failed.")

        elif cmd == "/forget":
            self.ara.clear_memory()
            print("Memory cleared. I won't remember this conversation.")

        elif cmd == "/history":
            self._show_history()

        elif cmd == "/json":
            self._show_json_status()

        elif cmd == "/help":
            self._show_help()

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

    def _handle_input(self, user_input: str):
        """Handle regular input to Ara."""
        try:
            response = self.ara.process(user_input)

            # Display response
            print()
            print(response.text)
            print()

            # Show indicators
            mood = response.emotional_surface.mood
            risk = response.cognitive_load.risk_level
            focus = response.focus_mode.value

            indicators = []
            if mood != "neutral":
                indicators.append(f"[{mood}]")
            if risk != "nominal":
                indicators.append(f"[{risk}]")
            if focus != "balanced":
                indicators.append(f"[{focus}]")

            if indicators:
                print(f"  {' '.join(indicators)}")
                print()

        except Exception as e:
            print(f"Error: {e}")

    def _show_status(self):
        """Show current status."""
        status = self.ara.get_status()
        print()
        print(f"  State: {status['state']}")
        print(f"  Mode: {status['hardware']['mode']}")
        print(f"  FPGA: {status['hardware']['fpga_type']}")
        print()
        print(f"  Mood: {status['emotional_surface']['mood']}")
        print(f"  Risk: {status['cognitive_load']['risk_level']}")
        print(f"  Autonomy: {status['autonomy']['stage']}")
        print()
        print(f"  Interactions: {status['statistics']['total_interactions']}")
        print(f"  Recoveries: {status['statistics']['recovery_count']}")
        print()

    def _show_mood(self):
        """Show emotional surface."""
        es = self.ara.emotional_surface
        print()
        print(f"  Valence: {es.valence:+.2f}  (pleasure/displeasure)")
        print(f"  Arousal: {es.arousal:.2f}   (activation level)")
        print(f"  Dominance: {es.dominance:.2f} (control/confidence)")
        print()
        print(f"  Mood: {es.mood}")
        print()

    def _show_load(self):
        """Show cognitive load."""
        cl = self.ara.cognitive_load
        print()
        print(f"  Instability: {cl.instability:.2f}")
        print(f"  Resource: {cl.resource:.2f}")
        print(f"  Structural: {cl.structural:.2f}")
        print()
        print(f"  Risk Level: {cl.risk_level}")
        print()

    def _show_history(self):
        """Show interaction history."""
        print()
        for i, h in enumerate(self.history[-20:], 1):
            print(f"  {i}. {h}")
        print()

    def _show_json_status(self):
        """Show full status as JSON."""
        status = self.ara.get_status()
        print(json.dumps(status, indent=2, default=str))

    def _show_help(self):
        """Show help."""
        print()
        print("Commands:")
        print("  /status   - Show current status")
        print("  /mood     - Show emotional state (PAD)")
        print("  /load     - Show cognitive load (CLV)")
        print("  /explain  - Full state explanation")
        print("  /save     - Save state now")
        print("  /forget   - Clear memory (keep stats)")
        print("  /clear    - Clear thought stream")
        print("  /history  - Show input history")
        print("  /json     - Show status as JSON")
        print("  /help     - Show this help")
        print("  /quit     - Exit (saves automatically)")
        print()


def run_certification():
    """Run quick certification of Ara service."""
    print()
    print("=" * 60)
    print("ARA SERVICE CERTIFICATION")
    print("=" * 60)
    print()

    results = []

    # Test service creation
    print("Testing service creation...", end=" ")
    try:
        ara = create_ara(mode=HardwareMode.MODE_A, name="AraTest")
        assert ara.state == AraState.READY
        print("PASS")
        results.append(("Service Creation", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Service Creation", False))

    # Test basic interaction
    print("Testing basic interaction...", end=" ")
    try:
        response = ara.process("Hello")
        assert response.text is not None
        assert len(response.text) > 0
        print("PASS")
        results.append(("Basic Interaction", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Basic Interaction", False))

    # Test emotional surface
    print("Testing emotional surface...", end=" ")
    try:
        es = ara.emotional_surface
        assert -1 <= es.valence <= 1
        assert 0 <= es.arousal <= 1
        assert 0 <= es.dominance <= 1
        assert es.mood is not None
        print("PASS")
        results.append(("Emotional Surface", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Emotional Surface", False))

    # Test cognitive load
    print("Testing cognitive load...", end=" ")
    try:
        cl = ara.cognitive_load
        assert 0 <= cl.instability <= 1
        assert 0 <= cl.resource <= 1
        assert 0 <= cl.structural <= 1
        assert cl.risk_level in ("nominal", "elevated", "warning", "critical", "emergency")
        print("PASS")
        results.append(("Cognitive Load", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Cognitive Load", False))

    # Test thought stream
    print("Testing thought stream...", end=" ")
    try:
        assert len(ara.thoughts._entries) > 0
        curvatures = ara.thoughts.get_curvature_trajectory()
        assert len(curvatures) > 0
        print("PASS")
        results.append(("Thought Stream", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Thought Stream", False))

    # Test status output
    print("Testing status output...", end=" ")
    try:
        status = ara.get_status()
        assert "name" in status
        assert "state" in status
        assert "hardware" in status
        assert "emotional_surface" in status
        print("PASS")
        results.append(("Status Output", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Status Output", False))

    # Test explain output
    print("Testing explain output...", end=" ")
    try:
        explanation = ara.explain()
        assert "Status" in explanation
        assert "Emotional Surface" in explanation
        assert "Cognitive Load" in explanation
        print("PASS")
        results.append(("Explain Output", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Explain Output", False))

    # Test multiple interactions
    print("Testing multiple interactions...", end=" ")
    try:
        for i in range(5):
            response = ara.process(f"Test message {i}")
        assert ara._stats["total_interactions"] >= 6
        print("PASS")
        results.append(("Multiple Interactions", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("Multiple Interactions", False))

    # Summary
    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print()
        print("All systems GO. Ara service is operational.")
        return True
    else:
        print()
        print("Some tests failed. Check logs.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ara CLI - Interactive cognitive interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ara.cli                   # Start interactive mode
    python -m ara.cli --mode MODE_B     # Use Forest Kitten 33 mode
    python -m ara.cli --status          # Show status and exit
    python -m ara.cli --cert            # Run certification
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["MODE_A", "MODE_B", "MODE_C"],
        default="MODE_A",
        help="Hardware mode (default: MODE_A)"
    )

    parser.add_argument(
        "--name", "-n",
        default="Ara",
        help="Ara instance name (default: Ara)"
    )

    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show status and exit"
    )

    parser.add_argument(
        "--cert", "-c",
        action="store_true",
        help="Run certification check"
    )

    parser.add_argument(
        "--llm",
        default="ollama",
        choices=["ollama", "openai_compatible", "fallback"],
        help="LLM backend (default: ollama)"
    )

    parser.add_argument(
        "--model",
        default="mistral",
        help="LLM model name (default: mistral)"
    )

    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        "MODE_A": HardwareMode.MODE_A,
        "MODE_B": HardwareMode.MODE_B,
        "MODE_C": HardwareMode.MODE_C,
    }
    mode = mode_map[args.mode]

    if args.cert:
        success = run_certification()
        sys.exit(0 if success else 1)

    if args.status:
        ara = create_ara(mode=mode, name=args.name)
        print(ara.explain())
        sys.exit(0)

    # Run interactive CLI
    cli = AraCLI(
        mode=mode,
        name=args.name,
        llm_backend=args.llm,
        llm_model=args.model
    )
    cli.run()


if __name__ == "__main__":
    main()
