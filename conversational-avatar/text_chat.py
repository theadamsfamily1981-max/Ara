#!/usr/bin/env python3
"""Ara - Text-only chat mode (no audio/voice)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from src.dialogue import DialogueManager


def main():
    """Text-only chat with Ara."""
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        dialogue_cfg = config.get('dialogue', {})
        engine = dialogue_cfg.get('engine', 'ollama')
        engine_cfg = dialogue_cfg.get(engine, {})
        system_prompt = dialogue_cfg.get('system_prompt')
        model = engine_cfg.get('model', 'llama3.2')
    else:
        # Defaults
        engine = 'ollama'
        engine_cfg = {}
        system_prompt = "You are Ara, a helpful AI assistant."
        model = 'llama3.2'

    print()
    print("=" * 50)
    print("  ARA - Text Chat Mode")
    print("=" * 50)
    print(f"  Model: {model} ({engine})")
    print("  Type 'quit' or 'exit' to end")
    print("=" * 50)
    print()

    # Initialize dialogue manager
    try:
        # Remove 'model' from engine_cfg to avoid duplicate argument
        engine_cfg.pop('model', None)
        dialogue = DialogueManager(
            engine=engine,
            model=model,
            system_prompt=system_prompt,
            max_history=20,
            **engine_cfg
        )
    except Exception as e:
        print(f"Error: Could not connect to {engine}: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        sys.exit(1)

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nAra: Goodbye!\n")
                break

            # Get response
            response = dialogue.get_response(user_input)
            print(f"\nAra: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAra: Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
