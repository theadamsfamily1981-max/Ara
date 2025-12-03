#!/usr/bin/env python3
"""Ara - Voice Chat with pyttsx3

A simple voice-enabled chat using Ollama and pyttsx3 for text-to-speech.

Requirements:
    pip install ollama pyttsx3

Usage:
    python ara_voice.py
"""
import sys

try:
    import ollama
    import pyttsx3
except ImportError as e:
    print(f"Missing module: {e}")
    print("Install with: pip install ollama pyttsx3")
    sys.exit(1)

SYSTEM_PROMPT = """You are Ara, a warm and intelligent AI assistant with a gentle,
confident personality. Keep responses concise (1-2 sentences) since they will be
spoken aloud. Be helpful, friendly, and conversational."""


def main():
    """Main entry point."""
    # Initialize TTS engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speech rate

    # Try to set a better voice if available
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    history = []

    print()
    print("=" * 50)
    print("  ARA - Voice Chat")
    print("=" * 50)
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    # Greeting
    greeting = "Hello! I'm Ara. How can I help you today?"
    print(f"Ara: {greeting}\n")
    engine.say(greeting)
    engine.runAndWait()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q', 'bye', 'goodbye']:
                farewell = "Goodbye! Have a great day!"
                print(f"\nAra: {farewell}\n")
                engine.say(farewell)
                engine.runAndWait()
                break

            # Add to history
            history.append({"role": "user", "content": user_input})

            # Build messages with system prompt
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history[-10:])  # Keep last 10 messages

            # Get response from Ollama
            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=messages,
                    options={"num_predict": 100}
                )
                answer = response["message"]["content"].strip()
            except Exception as e:
                answer = f"Sorry, I encountered an error: {e}"

            # Add to history
            history.append({"role": "assistant", "content": answer})

            # Display and speak
            print(f"\nAra: {answer}\n")
            engine.say(answer)
            engine.runAndWait()

        except KeyboardInterrupt:
            print("\n\nAra: Goodbye!\n")
            engine.say("Goodbye!")
            engine.runAndWait()
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
