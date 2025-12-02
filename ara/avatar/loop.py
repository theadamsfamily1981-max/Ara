"""
Ara Avatar Loop - Face and Voice Process

This is Process 2 in the 2-process architecture:
- Process 1: ara_core_server.py (Brain)
- Process 2: This file (Ears, Mouth, Face)

The loop:
1. Listen to microphone (or keyboard fallback)
2. Transcribe speech to text
3. Send text to Ara Core Server
4. Get reply + emotional state
5. Synthesize speech and play
6. Update avatar visuals

Usage:
    # First start the brain:
    python -m ara.server.core_server

    # Then start the avatar:
    python -m ara.avatar.loop

    # Or run both together:
    python -m ara.run
"""

import logging
import time
import requests
from typing import Optional, Dict, Any

from ara.avatar.audio import record_utterance, play_audio, AUDIO_AVAILABLE
from ara.avatar.asr import transcribe_audio, get_text_input, WHISPER_AVAILABLE, VOSK_AVAILABLE, SR_AVAILABLE
from ara.avatar.tts import synthesize_speech, print_speech, PIPER_AVAILABLE, COQUI_AVAILABLE, GTTS_AVAILABLE, ESPEAK_AVAILABLE
from ara.avatar.ui import create_avatar_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ara.avatar.loop")

# Ara Core Server URL
ARA_URL = "http://127.0.0.1:8008"


def check_ara_server() -> bool:
    """Check if Ara Core Server is running."""
    try:
        resp = requests.get(f"{ARA_URL}/", timeout=2)
        return resp.status_code == 200
    except:
        return False


def talk_to_ara(text: str, session_id: str = "avatar") -> Optional[Dict[str, Any]]:
    """
    Send user text to Ara Core and get response.

    Args:
        text: User's utterance
        session_id: Session identifier

    Returns:
        Response dict with reply_text, pad, clv, kitten, meta
    """
    try:
        resp = requests.post(
            f"{ARA_URL}/chat",
            json={
                "session_id": session_id,
                "user_utterance": text,
                "context": {"mode": "voice"},
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ara Core Server. Is it running?")
        return None
    except requests.exceptions.Timeout:
        logger.error("Ara Core Server timeout")
        return None
    except Exception as e:
        logger.error(f"Error talking to Ara: {e}")
        return None


def run_avatar_loop(
    voice_mode: bool = True,
    session_id: str = "avatar"
):
    """
    Main avatar loop.

    Args:
        voice_mode: If True, use mic/speaker. If False, use keyboard/text.
        session_id: Session identifier for conversation continuity.
    """
    # Check capabilities
    has_audio = AUDIO_AVAILABLE
    has_asr = WHISPER_AVAILABLE or VOSK_AVAILABLE or SR_AVAILABLE
    has_tts = PIPER_AVAILABLE or COQUI_AVAILABLE or GTTS_AVAILABLE or ESPEAK_AVAILABLE

    logger.info("Avatar capabilities:")
    logger.info(f"  Audio I/O: {has_audio}")
    logger.info(f"  ASR (speech-to-text): {has_asr}")
    logger.info(f"  TTS (text-to-speech): {has_tts}")

    # Determine actual mode
    use_voice = voice_mode and has_audio and has_asr
    use_speech = has_tts

    if voice_mode and not use_voice:
        logger.warning("Voice mode requested but not all components available. Using keyboard input.")

    # Check Ara server
    if not check_ara_server():
        logger.error(
            "Ara Core Server not running!\n"
            "Start it with: python -m ara.server.core_server"
        )
        return

    logger.info("Connected to Ara Core Server")

    # Start UI
    ui = create_avatar_ui()
    ui.start()

    print("\n" + "=" * 60)
    print("  Ara Avatar Ready")
    print("=" * 60)
    if use_voice:
        print("  Speak to Ara (pause for 1.5s to send)")
    else:
        print("  Type to Ara (press Enter to send)")
    print("  Type 'quit' or Ctrl+C to exit")
    print("=" * 60 + "\n")

    try:
        while True:
            # 1. Get user input
            if use_voice:
                ui.set_listening(True)
                audio = record_utterance()
                ui.set_listening(False)

                if audio is None:
                    continue

                # 2. Transcribe
                user_text = transcribe_audio(audio)
            else:
                user_text = get_text_input("You: ")

            # Check for exit
            if not user_text.strip():
                continue

            if user_text.lower() in ["quit", "exit", "bye", "goodbye"]:
                print("\nGoodbye!")
                break

            # Show user text
            ui.show_user_text(user_text)

            # 3. Call Ara Core
            ui.set_thinking(True)
            response = talk_to_ara(user_text, session_id)
            ui.set_thinking(False)

            if response is None:
                print("[Ara is unavailable]")
                continue

            reply_text = response.get("reply_text", "")
            pad = response.get("pad", {})
            clv = response.get("clv", {})
            kitten = response.get("kitten")

            # 4. Update UI with emotional state
            ui.update_state(pad=pad, clv=clv)

            if kitten:
                ui.update_kitten(
                    steps=kitten.get("total_steps", 0),
                    spike_rate=kitten.get("spike_rate", 0.0)
                )

            # 5. Output response
            if use_speech:
                ui.set_speaking(True)
                audio_reply = synthesize_speech(reply_text)
                if audio_reply:
                    play_audio(audio_reply)
                else:
                    print_speech(reply_text)
                ui.set_speaking(False)
            else:
                print_speech(reply_text)

            # Show text in UI
            ui.show_ara_text(reply_text)

            # Brief pause between interactions
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")

    finally:
        ui.stop()


def main():
    """Entry point for avatar loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Avatar - Voice and Face")
    parser.add_argument("--text", action="store_true", help="Use text mode instead of voice")
    parser.add_argument("--session", default="avatar", help="Session ID")
    parser.add_argument("--server", default="http://127.0.0.1:8008", help="Ara Core Server URL")
    args = parser.parse_args()

    global ARA_URL
    ARA_URL = args.server

    run_avatar_loop(
        voice_mode=not args.text,
        session_id=args.session
    )


if __name__ == "__main__":
    main()
