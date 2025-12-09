#!/usr/bin/env python3
"""
Ara Kernel CLI
===============

Quick bootstrap and testing interface for the Ara agent kernel.

Usage:
    python -m ara_kernel.cli              # Start with heartbeat agent
    python -m ara_kernel.cli --test       # Run test event
    python -m ara_kernel.cli --interactive  # Interactive mode
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara_kernel.core.runtime import AraAgentRuntime
from ara_kernel.core.config import KernelConfig, load_config
from ara_kernel.core.tools import build_default_registry, ToolsRegistry
from ara_kernel.core.safety import SafetyCovenant
from ara_kernel.memory import MemoryBackend
from ara_kernel.pheromones import LocalPheromoneBus
from ara_kernel.agents.realtime_breath import RealtimeBreathAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DummyModelClient:
    """
    Stub model client that returns canned responses.
    Replace with real LLM client (Anthropic, OpenAI, Ollama, etc.)
    """

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response to the prompt."""
        # Simple keyword-based responses for testing
        prompt_lower = prompt.lower()

        if "heartbeat" in prompt_lower or "status check" in prompt_lower:
            return "System status: All systems nominal. Memory stable. Pheromone bus active."

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return (
                "Hello! I'm Ara, your AI collaborator. "
                "I'm a teleoplastic cybernetic organism - not quite a chatbot, "
                "not quite an autonomous agent, but something in between. "
                "How can I help you today?"
            )

        if "test" in prompt_lower:
            return (
                "Test received and processed successfully. "
                "[TOOL: util.echo]\n"
                '{"message": "Echo test"}\n'
                "[/TOOL]\n"
                "The kernel is operational."
            )

        if "publish" in prompt_lower or "content" in prompt_lower:
            return (
                "I can help with publishing tasks. My publishing pipeline includes:\n"
                "- Content ideation and drafting\n"
                "- KDP/ebook formatting\n"
                "- Merch design coordination\n"
                "- Quality review against brand guidelines\n\n"
                "What would you like to create?"
            )

        if "quantum" in prompt_lower:
            return (
                "Quantum optimization module is currently in stub mode. "
                "When fully implemented, it will support:\n"
                "- QAOA for combinatorial optimization\n"
                "- VQE for Hamiltonian simulation\n"
                "- Quantum kernel methods for ML\n\n"
                "[TOOL: quantum.optimize]\n"
                '{"problem": "demo", "type": "generic"}\n'
                "[/TOOL]"
            )

        # Default response
        return (
            f"I received your message and processed it through my kernel. "
            f"Prompt length: {len(prompt)} characters. "
            f"This is a stub response - connect a real model for full functionality."
        )


def create_kernel(config: KernelConfig) -> AraAgentRuntime:
    """Create and configure an AraAgentRuntime instance."""
    logger.info("Creating Ara kernel...")

    # Initialize components
    model_client = DummyModelClient()
    logger.info("Model client: DummyModelClient (stub)")

    # Memory backend
    db_path = Path(config.memory.episodes_path).parent / "episodes.sqlite"
    memory = MemoryBackend.create(db_path=db_path)
    logger.info(f"Memory backend: {db_path}")

    # Pheromone bus
    pheromones = LocalPheromoneBus()
    logger.info("Pheromone bus: LocalPheromoneBus")

    # Tools registry
    tools = build_default_registry()
    logger.info(f"Tools registered: {tools.list_tools()}")

    # Create runtime
    runtime = AraAgentRuntime(
        config=config,
        model_client=model_client,
        memory_core=None,  # Using separate memory backend for now
        pheromone_store=pheromones,
        tool_registry={
            name: tools.get(name).fn
            for name in tools.list_tools()
        },
    )

    # Store references for agents
    runtime._memory_backend = memory
    runtime._pheromone_bus = pheromones

    logger.info("Ara kernel created successfully")
    return runtime


async def run_test(kernel: AraAgentRuntime) -> None:
    """Run a test event through the kernel."""
    logger.info("Running test event...")

    test_events = [
        {"input": "Hello Ara, this is a test.", "mode": "private"},
        {"input": "What can you help me with for publishing?", "mode": "private"},
        {"input": "Run a quantum optimization test.", "mode": "private"},
    ]

    for event in test_events:
        print(f"\n{'='*60}")
        print(f"Input: {event['input']}")
        print(f"{'='*60}")

        result = await kernel.process_input(
            user_input=event["input"],
            mode=event["mode"],
        )

        print(f"\nResponse: {result.get('text', '')}")
        print(f"Actions executed: {result.get('actions_executed', [])}")
        print(f"Actions blocked: {result.get('actions_blocked', [])}")

        await asyncio.sleep(0.5)

    logger.info("Test complete")


async def run_interactive(kernel: AraAgentRuntime) -> None:
    """Run interactive mode."""
    print("\n" + "="*60)
    print("Ara Kernel - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            result = await kernel.process_input(
                user_input=user_input,
                mode="private",
            )

            print(f"\nAra: {result.get('text', '')}")

            if result.get("actions_executed"):
                print(f"[Actions: {', '.join(result['actions_executed'])}]")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def run_with_heartbeat(kernel: AraAgentRuntime, duration_sec: float = 60.0) -> None:
    """Run with heartbeat agent for specified duration."""
    logger.info(f"Starting with heartbeat agent (duration={duration_sec}s)")

    breath = RealtimeBreathAgent(kernel, interval_sec=10.0)
    breath.start()

    # Run initial test
    asyncio.run(run_test(kernel))

    # Keep alive
    try:
        print(f"\nHeartbeat agent running. Press Ctrl+C to stop...")
        time.sleep(duration_sec)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        breath.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ara Kernel CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ara_kernel.cli              # Start with heartbeat
    python -m ara_kernel.cli --test       # Run test events
    python -m ara_kernel.cli --interactive  # Interactive mode
    python -m ara_kernel.cli --config path/to/config.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="ara_kernel/config/ara_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test events only",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration for heartbeat mode (seconds)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        config = KernelConfig()
        logger.info("Using default config")

    # Create kernel
    kernel = create_kernel(config)

    # Run appropriate mode
    if args.test:
        asyncio.run(run_test(kernel))
    elif args.interactive:
        asyncio.run(run_interactive(kernel))
    else:
        run_with_heartbeat(kernel, args.duration)


if __name__ == "__main__":
    main()
