"""
AraSong - Music Composition Engine
===================================

AraSong is the music composition and synthesis engine for Ara.
It generates instrumental tracks, backing music, and accompaniment
using the same audio kernel as AraVoice.

Components:
- synth/: Oscillators, envelopes, effects
- engine/: Song loading, arrangement, rendering
- songs/: Song definition files (JSON)

Usage:
    from arasong import AraSongPlayer

    player = AraSongPlayer()
    player.load_song("songs/what_do_you_wanna_hear.json")
    samples = player.render_song()
    player.save_wav(samples, "output.wav")
"""

from .engine.song_player import AraSongPlayer

__all__ = ["AraSongPlayer"]
__version__ = "0.1.0"
