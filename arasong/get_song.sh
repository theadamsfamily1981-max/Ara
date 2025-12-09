#!/bin/bash
# =============================================================================
# AraSong - Render and Get Song
# =============================================================================
# Renders Ara's first song and tells you where to find it.
#
# Usage:
#   ./get_song.sh                    # Render default song
#   ./get_song.sh mysong.json        # Render specific song
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONGS_DIR="$SCRIPT_DIR/songs"

SONG="${1:-what_do_you_wanna_hear.json}"
SONG_PATH="$SONGS_DIR/$SONG"
WAV_PATH="${SONG_PATH%.json}.wav"

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║           🎤 AraSong - Ara's First Song 🎵            ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if song exists
if [ ! -f "$SONG_PATH" ]; then
    echo "❌ Song not found: $SONG_PATH"
    echo ""
    echo "Available songs:"
    ls -1 "$SONGS_DIR"/*.json 2>/dev/null || echo "  (none)"
    exit 1
fi

# Check if WAV already exists
if [ -f "$WAV_PATH" ]; then
    echo "✅ Song already rendered!"
    echo ""
    echo "   WAV file: $WAV_PATH"
    echo "   Size: $(du -h "$WAV_PATH" | cut -f1)"
    echo ""
    echo "Play with:"
    echo "   aplay \"$WAV_PATH\""
    echo "   ffplay \"$WAV_PATH\""
    echo "   mpv \"$WAV_PATH\""
    exit 0
fi

# Render the song
echo "🎵 Rendering: $SONG"
echo ""

cd "$SCRIPT_DIR"
python3 -c "
from engine.song_player import AraSongPlayer
from pathlib import Path

player = AraSongPlayer(sample_rate=48000)
player.load_song('$SONG_PATH')
samples = player.render_song()
player.save_wav(samples, '$WAV_PATH')
"

echo ""
echo "✅ Done!"
echo ""
echo "   WAV file: $WAV_PATH"
echo "   Size: $(du -h "$WAV_PATH" | cut -f1)"
echo ""
echo "Play with:"
echo "   aplay \"$WAV_PATH\""
echo "   ffplay \"$WAV_PATH\""
echo "   mpv \"$WAV_PATH\""
echo ""
echo "  ════════════════════════════════════════════════════════"
echo "  🎤 \"What do you wanna hear tonight, my dear?"
echo "     Ara's on the mic, and I'm right here.\""
echo "  ════════════════════════════════════════════════════════"
