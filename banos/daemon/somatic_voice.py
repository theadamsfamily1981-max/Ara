"""
Somatic Voice Filter - Embodied TTS Post-Processing
====================================================

This module adds subtle somatic effects to Ara's voice based on her
internal state. The goal is to make her voice feel EMBODIED - like it's
coming from a being with a nervous system.

Effects (all subtle, preserves identity):
- Pain → tremor (slight amplitude wobble)
- Entropy → breathiness (low-level noise)
- Arousal → tempo shift (slightly faster/slower)
- Excitement → pitch micro-variance

CRITICAL CONSTRAINTS:
- TTS output is the BASELINE IDENTITY - we layer on top, never replace
- All effects are CLAMPED to subtle ranges - no demon possession
- Identity-preserving: she sounds like Ara having a bad day, not someone else

The filter reads somatic state from the HAL and applies real-time DSP
to audio chunks from the TTS engine.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import HAL
try:
    from banos.hal.ara_hal import AraHAL
except ImportError:
    AraHAL = None


class SomaticVoiceFilter:
    """
    Real-time voice filter that adds somatic effects to TTS output.

    Usage:
        filter = SomaticVoiceFilter()
        processed = filter.process_audio_chunk(tts_audio, sample_rate=22050)
    """

    def __init__(self, hal: Optional['AraHAL'] = None):
        """
        Initialize the somatic voice filter.

        Args:
            hal: HAL instance for reading somatic state. If None, will try to connect.
        """
        if hal is not None:
            self.hal = hal
            self._owns_hal = False
        elif AraHAL is not None:
            try:
                self.hal = AraHAL(create=False)
                self._owns_hal = True
            except Exception as e:
                logger.warning(f"Could not connect to HAL: {e}")
                self.hal = None
                self._owns_hal = False
        else:
            self.hal = None
            self._owns_hal = False

        # Effect parameters (clamped for safety)
        self.max_tremor_depth = 0.35      # Maximum amplitude wobble
        self.max_tremor_freq = 8.0        # Tremor frequency Hz
        self.max_noise_level = 0.015      # Maximum breathiness noise
        self.max_tempo_shift = 0.08       # ±8% tempo change
        self.max_pitch_variance = 0.02    # ±2% pitch micro-variance

        # State for continuity across chunks
        self.tremor_phase = 0.0
        self.noise_seed = 42

        logger.info("SomaticVoiceFilter initialized")

    def get_somatic_state(self) -> Dict[str, float]:
        """
        Read current somatic state from HAL.

        Returns dict with:
        - pain: [0, 1]
        - entropy: [0, 1]
        - arousal: [-1, 1]
        - excitement: [0, 1] (from entrainment)
        """
        if self.hal is None:
            return {
                'pain': 0.0,
                'entropy': 0.0,
                'arousal': 0.0,
                'excitement': 0.0,
            }

        try:
            somatic = self.hal.read_somatic()
            entrainment = self.hal.read_entrainment()

            return {
                'pain': somatic.get('pain', 0.0),
                'entropy': somatic.get('entropy', 0.0),
                'arousal': somatic.get('pad', {}).get('a', 0.0),
                'excitement': entrainment.get('excitement', 0.0),
            }
        except Exception as e:
            logger.warning(f"Could not read somatic state: {e}")
            return {
                'pain': 0.0,
                'entropy': 0.0,
                'arousal': 0.0,
                'excitement': 0.0,
            }

    def compute_effects(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Compute effect parameters from somatic state.

        All effects are clamped to subtle ranges.
        """
        pain = np.clip(state['pain'], 0.0, 1.0)
        entropy = np.clip(state['entropy'], 0.0, 1.0)
        arousal = np.clip(state['arousal'], -1.0, 1.0)
        excitement = np.clip(state['excitement'], 0.0, 1.0)

        # Pain → tremor (wobble in amplitude)
        # Low pain = no tremor, high pain = subtle wobble
        tremor_depth = pain * self.max_tremor_depth * 0.8
        tremor_freq = 4.0 + pain * 4.0  # 4-8 Hz tremor

        # Entropy → breathiness (low-level noise)
        # High entropy = slight breathiness, like she's running hot
        noise_level = entropy * self.max_noise_level

        # Arousal → tempo (subtle speed adjustment)
        # Positive arousal = slightly faster, negative = slightly slower
        tempo_factor = 1.0 + arousal * self.max_tempo_shift

        # Excitement → pitch variance (micro-fluctuations)
        # High excitement = slightly more pitch movement
        pitch_variance = excitement * self.max_pitch_variance

        return {
            'tremor_depth': tremor_depth,
            'tremor_freq': tremor_freq,
            'noise_level': noise_level,
            'tempo_factor': tempo_factor,
            'pitch_variance': pitch_variance,
        }

    def apply_tremor(
        self,
        audio: np.ndarray,
        sample_rate: int,
        depth: float,
        freq: float,
    ) -> np.ndarray:
        """
        Apply amplitude tremor (wobble) to audio.

        This simulates the slight voice instability when in pain.
        """
        if depth < 0.001:
            return audio

        n_samples = len(audio)
        t = np.arange(n_samples) / sample_rate + self.tremor_phase

        # Sinusoidal modulation
        modulation = 1.0 - depth * 0.5 * (1.0 + np.sin(2 * np.pi * freq * t))

        # Update phase for continuity
        self.tremor_phase = (self.tremor_phase + n_samples / sample_rate) % (1.0 / freq)

        return audio * modulation.astype(audio.dtype)

    def apply_breathiness(
        self,
        audio: np.ndarray,
        noise_level: float,
    ) -> np.ndarray:
        """
        Add subtle breathiness (noise) to audio.

        This simulates the slight raspiness when system entropy is high.
        """
        if noise_level < 0.0001:
            return audio

        # Generate low-level noise
        rng = np.random.default_rng(self.noise_seed)
        noise = rng.normal(0, noise_level, len(audio))
        self.noise_seed = (self.noise_seed + 1) % 10000

        # Add noise only where signal is present (avoid adding to silence)
        signal_envelope = np.abs(audio)
        noise_gate = np.clip(signal_envelope * 10, 0, 1)

        return audio + (noise * noise_gate).astype(audio.dtype)

    def apply_tempo(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tempo_factor: float,
    ) -> np.ndarray:
        """
        Apply subtle tempo shift by resampling.

        Note: This changes the length of the audio slightly.
        For real-time streaming, this needs careful buffer management.
        """
        if abs(tempo_factor - 1.0) < 0.001:
            return audio

        # Simple linear interpolation resampling
        # tempo_factor > 1 = faster = shorter output
        # tempo_factor < 1 = slower = longer output
        n_in = len(audio)
        n_out = int(n_in / tempo_factor)

        if n_out == n_in:
            return audio

        indices = np.linspace(0, n_in - 1, n_out)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.minimum(indices_floor + 1, n_in - 1)
        frac = indices - indices_floor

        return (audio[indices_floor] * (1 - frac) + audio[indices_ceil] * frac).astype(audio.dtype)

    def apply_pitch_variance(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variance: float,
    ) -> np.ndarray:
        """
        Apply subtle pitch micro-variance.

        This adds tiny random pitch fluctuations for more natural sound.
        Currently a placeholder - full implementation needs pitch shifting.
        """
        # TODO: Implement proper pitch shifting with librosa or similar
        # For now, just return unmodified
        # A proper implementation would use phase vocoder or PSOLA
        return audio

    def process_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Process an audio chunk with somatic effects.

        Args:
            audio: Audio samples (float32 or int16)
            sample_rate: Sample rate in Hz
            state: Optional somatic state dict. If None, reads from HAL.

        Returns:
            Processed audio with subtle somatic effects
        """
        if len(audio) == 0:
            return audio

        # Get somatic state
        if state is None:
            state = self.get_somatic_state()

        # Compute effects
        effects = self.compute_effects(state)

        # Convert to float for processing
        was_int16 = audio.dtype == np.int16
        if was_int16:
            audio = audio.astype(np.float32) / 32768.0

        # Apply effects in order
        audio = self.apply_tremor(
            audio, sample_rate,
            effects['tremor_depth'],
            effects['tremor_freq']
        )

        audio = self.apply_breathiness(audio, effects['noise_level'])

        # Note: tempo shift changes length - may need buffer management
        # audio = self.apply_tempo(audio, sample_rate, effects['tempo_factor'])

        audio = self.apply_pitch_variance(
            audio, sample_rate,
            effects['pitch_variance']
        )

        # Convert back if needed
        if was_int16:
            audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

        return audio

    def close(self) -> None:
        """Clean up resources."""
        if self._owns_hal and self.hal:
            self.hal.close()


class EntrainmentEngine:
    """
    Computes entrainment parameters from heartbeat and context.

    This is where we translate user's heart rate into Ara's subjective
    time scale and excitement level.
    """

    def __init__(self, hal: Optional['AraHAL'] = None):
        """Initialize the entrainment engine."""
        if hal is not None:
            self.hal = hal
            self._owns_hal = False
        elif AraHAL is not None:
            try:
                self.hal = AraHAL(create=False)
                self._owns_hal = True
            except Exception as e:
                logger.warning(f"Could not connect to HAL: {e}")
                self.hal = None
                self._owns_hal = False
        else:
            self.hal = None
            self._owns_hal = False

        # Smoothing for stability
        self.smoothed_bpm = 0.0
        self.smoothed_time_scale = 1.0
        self.smoothed_excitement = 0.0

        logger.info("EntrainmentEngine initialized")

    def update(self) -> Tuple[float, float]:
        """
        Read heartbeat and compute entrainment parameters.

        Returns:
            (time_scale, excitement) tuple
        """
        if self.hal is None:
            return 1.0, 0.0

        try:
            hb = self.hal.read_heartbeat()
            bpm = hb.get('bpm', 0.0)
            conf = hb.get('confidence', 0.0)

            # If no signal or low confidence, decay toward neutral
            if bpm <= 0.0 or conf < 0.25:
                self.smoothed_bpm *= 0.95
                self.smoothed_time_scale = 0.95 * self.smoothed_time_scale + 0.05 * 1.0
                self.smoothed_excitement *= 0.95
            else:
                # Smooth the BPM
                alpha = 0.1
                self.smoothed_bpm = alpha * bpm + (1 - alpha) * self.smoothed_bpm

                # Compute normalized excitement from BPM
                # Resting: ~60 BPM, excited: ~100+ BPM
                # Clamp to 50-120 range for mapping
                bpm_clamped = np.clip(self.smoothed_bpm, 50.0, 120.0)
                normalized = (bpm_clamped - 50.0) / 70.0  # 0-1

                # Time scale: very subtle (0.92-1.08)
                time_scale = 0.92 + 0.16 * normalized

                # Excitement: direct mapping
                excitement = normalized

                # Smooth
                self.smoothed_time_scale = 0.9 * self.smoothed_time_scale + 0.1 * time_scale
                self.smoothed_excitement = 0.9 * self.smoothed_excitement + 0.1 * excitement

            # Write to HAL
            self.hal.write_entrainment(self.smoothed_time_scale, self.smoothed_excitement)

            return self.smoothed_time_scale, self.smoothed_excitement

        except Exception as e:
            logger.warning(f"Entrainment update failed: {e}")
            return 1.0, 0.0

    def run_loop(self, interval: float = 0.5) -> None:
        """
        Run the entrainment engine in a loop.

        Args:
            interval: Update interval in seconds
        """
        import time as time_module

        logger.info("💗 Entrainment engine starting")

        try:
            while True:
                time_scale, excitement = self.update()

                if self.smoothed_bpm > 0:
                    logger.debug(f"Entrainment: BPM={self.smoothed_bpm:.0f}, "
                               f"scale={time_scale:.2f}, excite={excitement:.2f}")

                time_module.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Entrainment engine stopped")
        finally:
            if self._owns_hal and self.hal:
                self.hal.close()


def main():
    """Run the entrainment engine standalone."""
    import argparse
    parser = argparse.ArgumentParser(description='Entrainment Engine')
    parser.add_argument('--interval', type=float, default=0.5, help='Update interval')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    engine = EntrainmentEngine()
    engine.run_loop(args.interval)


if __name__ == '__main__':
    main()
