#!/usr/bin/env python3
"""Command-line interface for the audio pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .advanced_tts import AdvancedTTS, TTSConfig
from .audio_mastering import AudioMastering, MasteringConfig
from .enhanced_asr import EnhancedASR, ASRConfig
from .profiles import ProfileManager, ProfileMode


def tts_main():
    """TTS command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ara TTS - Text-to-Speech with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic synthesis
  ara-tts "Hello, world!" -o hello.wav

  # With voice cloning
  ara-tts "Hello!" -o output.wav --voice samples/my_voice.wav

  # Use deep profile for highest quality
  ara-tts "High quality output" -o hq.wav --profile deep

  # Multiple voice samples
  ara-tts "Cloned voice" -o out.wav --voice s1.wav --voice s2.wav --voice s3.wav
        """
    )

    parser.add_argument("text", nargs="?", help="Text to synthesize (or use --input)")
    parser.add_argument("-i", "--input", type=Path, help="Input text file")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"),
                        help="Output audio file (default: output.wav)")
    parser.add_argument("--voice", action="append", dest="voices",
                        help="Voice sample file(s) for cloning (can specify multiple)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--profile", choices=["fast", "balanced", "deep"],
                        default="balanced", help="Quality profile (default: balanced)")
    parser.add_argument("--no-master", action="store_true",
                        help="Skip audio mastering")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Don't chunk text (for short inputs)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Get input text
    if args.input:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        text = args.input.read_text()
    elif args.text:
        text = args.text
    else:
        # Read from stdin
        print("Enter text (Ctrl+D to finish):", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)

    # Get profile
    pm = ProfileManager()
    mode_map = {
        "fast": ProfileMode.FAST,
        "balanced": ProfileMode.BALANCED,
        "deep": ProfileMode.DEEP,
    }
    profile = pm.get_profile(mode_map[args.profile])

    # Override settings from args
    config = TTSConfig(
        language=args.language,
        speed=args.speed,
        voice_samples=args.voices or [],
    )

    try:
        # Initialize TTS
        print(f"Loading TTS model ({args.profile} profile)...", file=sys.stderr)
        tts = AdvancedTTS(config)

        if args.voices:
            tts.add_voice_samples(args.voices)

        # Synthesize
        print(f"Synthesizing {len(text)} characters...", file=sys.stderr)
        output_path = tts.synthesize(
            text,
            output_path=args.output,
            use_chunking=not args.no_chunk,
            progress_callback=lambda i, n: print(f"  Chunk {i+1}/{n}", file=sys.stderr) if args.verbose else None
        )

        # Apply mastering
        if not args.no_master:
            print("Applying audio mastering...", file=sys.stderr)
            mastering = AudioMastering(profile.mastering_config)
            mastering.process_file(output_path, output_path)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def asr_main():
    """ASR command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ara ASR - Speech-to-Text with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe an audio file
  ara-asr recording.wav

  # Use larger model for better accuracy
  ara-asr recording.wav --model medium

  # Add custom vocabulary
  ara-asr technical.wav --vocab "Kubernetes,PostgreSQL,GraphQL"

  # Output to file
  ara-asr recording.wav -o transcript.txt
        """
    )

    parser.add_argument("audio", type=Path, help="Audio file to transcribe")
    parser.add_argument("-o", "--output", type=Path, help="Output text file")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"],
                        default="base", help="Whisper model size (default: base)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--vocab", help="Comma-separated custom vocabulary terms")
    parser.add_argument("--prompt", help="Initial prompt to guide recognition")
    parser.add_argument("--timestamps", action="store_true",
                        help="Include word timestamps")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Configure ASR
    config = ASRConfig(
        model_size=args.model,
        language=args.language,
        custom_vocabulary=args.vocab.split(",") if args.vocab else [],
        initial_prompt=args.prompt or "",
    )

    try:
        # Initialize ASR
        print(f"Loading Whisper {args.model} model...", file=sys.stderr)
        asr = EnhancedASR(config)

        # Transcribe
        print(f"Transcribing {args.audio}...", file=sys.stderr)
        result = asr.transcribe_file(args.audio)

        # Format output
        if args.json:
            import json
            output = json.dumps({
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "duration": result.duration,
                "processing_time": result.processing_time,
                "segments": result.segments if args.timestamps else None,
            }, indent=2)
        elif args.timestamps:
            lines = []
            for seg in result.segments:
                start = f"{seg['start']:.2f}"
                end = f"{seg['end']:.2f}"
                lines.append(f"[{start} -> {end}] {seg['text']}")
            output = "\n".join(lines)
        else:
            output = result.text

        # Output
        if args.output:
            args.output.write_text(output)
            print(f"Saved: {args.output}")
        else:
            print(output)

        if args.verbose:
            print(f"\n--- Stats ---", file=sys.stderr)
            print(f"Duration: {result.duration:.1f}s", file=sys.stderr)
            print(f"Processing: {result.processing_time:.2f}s", file=sys.stderr)
            print(f"Speed: {result.words_per_second:.1f} words/s", file=sys.stderr)
            print(f"Confidence: {result.confidence:.2%}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Ara Audio Pipeline - TTS, ASR, and Audio Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # TTS subcommand
    tts_parser = subparsers.add_parser("tts", help="Text-to-speech synthesis")
    tts_parser.add_argument("text", nargs="?", help="Text to synthesize")

    # ASR subcommand
    asr_parser = subparsers.add_parser("asr", help="Speech-to-text transcription")
    asr_parser.add_argument("audio", type=Path, help="Audio file")

    # Profile subcommand
    profile_parser = subparsers.add_parser("profiles", help="List available profiles")

    # Version
    parser.add_argument("--version", action="version", version="ara-avatar 1.0.0")

    args = parser.parse_args()

    if args.command == "tts":
        # Re-parse with full TTS parser
        sys.argv = ["ara-tts"] + sys.argv[2:]
        tts_main()
    elif args.command == "asr":
        sys.argv = ["ara-asr"] + sys.argv[2:]
        asr_main()
    elif args.command == "profiles":
        pm = ProfileManager()
        print("Available Audio Profiles:")
        print("-" * 50)
        for p in pm.list_profiles():
            status = "[builtin]" if p["builtin"] else "[custom]"
            print(f"  {p['name']:12} {status:10} - {p['description']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
