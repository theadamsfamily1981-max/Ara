#!/usr/bin/env python3
"""
AraSong CLI - Render and play songs.

Usage:
    python -m arasong.play                    # Render default song
    python -m arasong.play songs/mysong.json  # Render specific song
    python -m arasong.play --list             # List available songs
"""

import argparse
import sys
from pathlib import Path


def list_songs():
    """List available songs."""
    songs_dir = Path(__file__).parent / "songs"
    print("\n🎵 Available Songs:")
    print("-" * 40)
    for song_file in sorted(songs_dir.glob("*.json")):
        print(f"  {song_file.name}")
    print()


def render_song(song_path: Path):
    """Render a song to WAV."""
    from .engine.song_player import AraSongPlayer

    player = AraSongPlayer(sample_rate=48000)
    player.load_song(str(song_path))

    samples = player.render_song()

    output_path = song_path.with_suffix('.wav')
    player.save_wav(samples, str(output_path))

    # Print lyrics
    print("\n" + "=" * 60)
    print("📜 LYRICS - \"" + player.song_data['meta']['title'] + "\"")
    print("   by " + player.song_data['meta']['artist'])
    print("=" * 60)

    for section in player.song_data.get('structure', []):
        section_name = section['section']
        lyrics = player.song_data.get('lyrics', {}).get(section_name, [])
        if lyrics:
            print(f"\n[{section_name.upper()}]")
            for line in lyrics:
                if line:
                    print(f"  {line}")

    print("\n" + "=" * 60)
    print(f"\n🎧 Play with: aplay {output_path}")
    print(f"            or: ffplay {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AraSong - Render songs")
    parser.add_argument("song", nargs="?", help="Path to song JSON file")
    parser.add_argument("--list", "-l", action="store_true", help="List available songs")
    args = parser.parse_args()

    if args.list:
        list_songs()
        return

    if args.song:
        song_path = Path(args.song)
    else:
        # Default song
        song_path = Path(__file__).parent / "songs" / "what_do_you_wanna_hear.json"

    if not song_path.exists():
        print(f"❌ Song not found: {song_path}")
        sys.exit(1)

    render_song(song_path)


if __name__ == "__main__":
    main()
