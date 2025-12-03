#!/usr/bin/env python3
"""
Ara - Simple Text Chat
Standalone script with minimal dependencies.

Requirements:
    pip install ollama

Usage:
    python ara_chat.py
"""

import sys

try:
    import ollama
except ImportError:
    print("Error: 'ollama' module not found.")
    print("Install it with: pip install ollama")
    sys.exit(1)


class AraChat:
    """Simple chat interface using Ollama."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.history = []
        self.system_prompt = (
            "You are Ara, a friendly and helpful AI assistant. "
            "Keep responses concise and conversational."
        )

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-20:])  # Keep last 20 messages

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 200}
            )
            assistant_message = response["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            return f"Error: {e}"


def main():
    """Main entry point."""
    print()
    print("=" * 50)
    print("  ARA - Text Chat")
    print("=" * 50)
    print("  Type 'quit' or 'exit' to end")
    print("  Make sure Ollama is running: ollama serve")
    print("=" * 50)
    print()

    # Check if Ollama is running
    try:
        models = ollama.list()
        available = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        print(f"  Available models: {', '.join(available) if available else 'none'}")
        print()
    except Exception as e:
        print(f"  Warning: Could not connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print()

    # Determine model to use
    model = "llama3.2"
    if available:
        # Use first available model if llama3.2 not found
        if not any("llama3.2" in m for m in available):
            model = available[0].split(":")[0] if available else "llama3.2"
            print(f"  Using model: {model}")
            print()

    ara = AraChat(model=model)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'q']:
                print("\nAra: Goodbye!\n")
                break

            response = ara.chat(user_input)
            print(f"\nAra: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAra: Goodbye!\n")
            break
        except EOFError:
            print("\n\nAra: Goodbye!\n")
            break


if __name__ == "__main__":
    main()
