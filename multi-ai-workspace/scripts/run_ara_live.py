#!/usr/bin/env python3
"""
Run Ara Live - Real-time Multimodal Cognitive AI.

This script launches Ara with:
- Webcam vision (OpenCV)
- Microphone input with ASR (Whisper)
- Cognitive backend with thermodynamics and autonomy
- TTS speech output (Piper/Coqui)
- TFAN 7B / Ollama for generation

Requirements:
    pip install opencv-python sounddevice numpy scipy
    pip install faster-whisper  # For ASR
    pip install TTS  # For Coqui TTS (or pip install piper-tts)

Usage:
    python run_ara_live.py
    python run_ara_live.py --no-webcam
    python run_ara_live.py --text-only
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


def check_dependencies():
    """Check and report on available dependencies."""
    print("\n" + "=" * 60)
    print("  CHECKING DEPENDENCIES")
    print("=" * 60)

    deps = {}

    # NumPy
    try:
        import numpy
        deps["numpy"] = f"✓ {numpy.__version__}"
    except ImportError:
        deps["numpy"] = "✗ Not installed (pip install numpy)"

    # OpenCV
    try:
        import cv2
        deps["opencv"] = f"✓ {cv2.__version__}"
    except ImportError:
        deps["opencv"] = "✗ Not installed (pip install opencv-python)"

    # Sounddevice
    try:
        import sounddevice
        deps["sounddevice"] = f"✓ {sounddevice.__version__}"
    except ImportError:
        deps["sounddevice"] = "✗ Not installed (pip install sounddevice)"

    # Whisper (ASR)
    try:
        from faster_whisper import WhisperModel
        deps["faster-whisper"] = "✓ Available"
    except ImportError:
        deps["faster-whisper"] = "✗ Not installed (pip install faster-whisper)"

    # Coqui TTS
    try:
        from TTS.api import TTS
        deps["coqui-tts"] = "✓ Available"
    except ImportError:
        deps["coqui-tts"] = "✗ Not installed (pip install TTS)"

    # Piper TTS
    try:
        import piper
        deps["piper-tts"] = "✓ Available"
    except ImportError:
        deps["piper-tts"] = "✗ Not installed (pip install piper-tts)"

    # PyTorch
    try:
        import torch
        cuda = "CUDA" if torch.cuda.is_available() else "CPU"
        deps["pytorch"] = f"✓ {torch.__version__} ({cuda})"
    except ImportError:
        deps["pytorch"] = "✗ Not installed"

    # Print results
    for name, status in deps.items():
        print(f"  {name:20} {status}")

    print()

    # Check minimum requirements
    required = ["numpy", "sounddevice"]
    missing = [d for d in required if "✗" in deps.get(d, "✗")]
    if missing:
        print(f"⚠️  Missing required dependencies: {', '.join(missing)}")
        print("   Install with: pip install numpy sounddevice")
        return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Live - Real-time Multimodal Cognitive AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ara_live.py                    # Full experience
  python run_ara_live.py --no-webcam        # Audio only
  python run_ara_live.py --text-only        # Text input/output only
  python run_ara_live.py --ollama-model llama3.2  # Use different model
        """,
    )

    parser.add_argument(
        "--no-webcam", action="store_true",
        help="Disable webcam (audio only)",
    )
    parser.add_argument(
        "--no-mic", action="store_true",
        help="Disable microphone (use text input)",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable TTS (text output only)",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Text input/output only (no webcam, mic, or TTS)",
    )
    parser.add_argument(
        "--ollama-model", default="ara",
        help="Ollama model name (default: ara)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--tfan-model", default=None,
        help="Path to TFAN 7B model weights",
    )
    parser.add_argument(
        "--webcam-device", type=int, default=0,
        help="Webcam device ID (default: 0)",
    )
    parser.add_argument(
        "--mic-device", type=int, default=None,
        help="Microphone device ID (default: system default)",
    )
    parser.add_argument(
        "--silence-threshold", type=float, default=500.0,
        help="VAD silence threshold (default: 500)",
    )
    parser.add_argument(
        "--check-deps", action="store_true",
        help="Check dependencies and exit",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return

    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing. Ara may have limited functionality.")
        print("   Continue anyway? [y/N] ", end="")
        response = input().strip().lower()
        if response != "y":
            return

    # Apply text-only mode
    if args.text_only:
        args.no_webcam = True
        args.no_mic = True
        args.no_tts = True

    # Print banner
    print()
    print("=" * 60)
    print("  🤖 ARA LIVE - Real-time Multimodal Cognitive AI")
    print("=" * 60)
    print()
    print(f"  Webcam:     {'Enabled' if not args.no_webcam else 'Disabled'}")
    print(f"  Microphone: {'Enabled' if not args.no_mic else 'Disabled'}")
    print(f"  TTS:        {'Enabled' if not args.no_tts else 'Disabled'}")
    print(f"  Model:      {args.ollama_model}")
    print()
    print("  Press Ctrl+C to exit")
    print()

    # Import and run AraLive
    try:
        from src.integrations.ara_live import AraLive, AraState, AraResponse

        ara = AraLive(
            enable_webcam=not args.no_webcam,
            enable_microphone=not args.no_mic,
            enable_tts=not args.no_tts,
            webcam_device=args.webcam_device,
            mic_device=args.mic_device,
            silence_threshold=args.silence_threshold,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            tfan_model_path=args.tfan_model,
        )

        # Response callback
        def on_response(response: AraResponse):
            print(f"\n🤖 Ara: {response.text}")
            print(f"   ⏱️  {response.processing_time_ms:.0f}ms")

        # State change callback
        def on_state_change(old_state: AraState, new_state: AraState):
            icons = {
                AraState.IDLE: "💤",
                AraState.LISTENING: "👂",
                AraState.PROCESSING: "🤔",
                AraState.SPEAKING: "🗣️",
                AraState.ERROR: "❌",
            }
            print(f"   {icons.get(new_state, '?')} {new_state.name}")

        ara.on_response(on_response)
        ara.on_state_change(on_state_change)

        # Run
        asyncio.run(ara.start())

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
