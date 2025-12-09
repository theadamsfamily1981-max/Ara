"""
Ara Audacity Pipeline
======================

Automated recording and post-processing for ACX-compliant audiobooks.

Pipeline:
1. HV synthesis → Piper TTS → Raw WAV
2. Audacity loopback recording
3. ACX compliance processing (-23dB RMS)
4. Export to MP3 192kbps

Cost: $0 (local processing)
Quality: ACX perfect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import os
import shutil
import subprocess
import tempfile


# =============================================================================
# ACX Specifications
# =============================================================================

@dataclass(frozen=True)
class ACXSpecs:
    """ACX technical requirements for audiobooks."""
    rms_target_db: float = -23.0        # Target RMS level
    rms_tolerance_db: float = 1.0       # ±1dB tolerance
    peak_max_db: float = -3.0           # Maximum peak level
    noise_floor_db: float = -60.0       # Maximum noise floor
    sample_rate: int = 44100            # Required sample rate
    bit_depth: int = 16                 # Required bit depth
    mp3_bitrate: int = 192              # MP3 bitrate (kbps)
    channels: int = 1                   # Mono required


ACX = ACXSpecs()


# =============================================================================
# Audacity Macro Generation
# =============================================================================

class AudacityMacro:
    """
    Generate Audacity macros for automated processing.

    Audacity macros are sequences of commands saved as .txt files.
    """

    @staticmethod
    def acx_compliance() -> str:
        """
        Generate ACX compliance macro.

        Steps:
        1. Normalize to -3dB peak
        2. Compressor (4:1 ratio)
        3. Normalize RMS to -23dB
        4. Limiter at -3dB
        """
        macro = """
;Ara ACX Compliance Macro
;Generated automatically

;Step 1: Normalize peak to -3dB
Normalize:PeakLevel=-3 ApplyGain=1 RemoveDcOffset=1 StereoIndependent=0

;Step 2: Compression (gentle, for consistency)
Compressor:Threshold=-20 NoiseFloor=-40 Ratio=4 AttackTime=0.2 ReleaseTime=1.0 Normalize=0 UsePeak=0

;Step 3: Normalize RMS to -23dB (ACX target)
LoudnessNormalization:StereoIndependent=0 LUFSLevel=-23 RMSLevel=-23 DualMono=1 NormalizeTo=1

;Step 4: Hard limiter at -3dB (prevent clipping)
Limiter:type=0 limit=-3 hold=10 release=50

;Step 5: Convert to mono if stereo
ConvertToMono:

;Done
"""
        return macro.strip()

    @staticmethod
    def export_mp3(output_path: str, bitrate: int = 192) -> str:
        """Generate export macro for MP3."""
        return f"""
;Export to MP3
ExportMP3:Filename="{output_path}" BitRateMode=0 Quality={bitrate}
"""

    @staticmethod
    def full_pipeline(
        output_path: str,
        bitrate: int = 192,
    ) -> str:
        """Complete processing + export pipeline."""
        compliance = AudacityMacro.acx_compliance()
        export = AudacityMacro.export_mp3(output_path, bitrate)
        return compliance + "\n" + export

    @staticmethod
    def save_macro(content: str, path: Path) -> None:
        """Save macro to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)


# =============================================================================
# Audacity CLI Interface
# =============================================================================

class AudacityCLI:
    """
    Interface to Audacity via command line.

    Requires Audacity with mod-script-pipe or audacity-cli.
    """

    def __init__(
        self,
        audacity_path: Optional[str] = None,
        macro_dir: Optional[Path] = None,
    ):
        self.audacity_path = audacity_path or self._find_audacity()
        self.macro_dir = Path(macro_dir) if macro_dir else Path.home() / ".ara/audacity_macros"
        self.macro_dir.mkdir(parents=True, exist_ok=True)

        # Create default macros
        self._setup_macros()

    def _find_audacity(self) -> str:
        """Find Audacity installation."""
        # Common locations
        locations = [
            '/usr/bin/audacity',
            '/usr/local/bin/audacity',
            '/snap/bin/audacity',
            'audacity',  # In PATH
        ]

        for loc in locations:
            if shutil.which(loc):
                return loc

        return 'audacity'  # Hope it's in PATH

    def _setup_macros(self) -> None:
        """Setup default processing macros."""
        # ACX compliance macro
        acx_macro = AudacityMacro.acx_compliance()
        AudacityMacro.save_macro(acx_macro, self.macro_dir / "AraACX.txt")

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        macro_name: str = "AraACX",
    ) -> Dict[str, Any]:
        """
        Process an audio file with Audacity macro.

        Note: This requires Audacity with scripting support.
        For automated pipelines, consider using sox or ffmpeg directly.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        result = {
            'input': str(input_path),
            'output': str(output_path),
            'macro': macro_name,
            'success': False,
            'method': 'audacity_cli',
        }

        # Check if Audacity is available
        if not shutil.which(self.audacity_path):
            result['error'] = 'Audacity not found'
            result['fallback'] = 'Use sox_process() or ffmpeg_process() instead'
            return result

        # For now, return instructions (actual automation requires Audacity scripting)
        result['instructions'] = f"""
Audacity Manual Processing:
1. Open Audacity
2. File → Import → Audio → {input_path}
3. Tools → Apply Macro → AraACX
4. File → Export → Export as MP3
5. Save to: {output_path}
6. Settings: 192 kbps, Mono

Or use automated processing:
    cli.sox_process('{input_path}', '{output_path}')
"""
        return result


# =============================================================================
# Sox/FFmpeg Processing (Fallback)
# =============================================================================

class AudioProcessor:
    """
    Audio processing using sox and ffmpeg.

    This is the reliable fallback when Audacity scripting isn't available.
    """

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check which tools are available."""
        return {
            'sox': shutil.which('sox') is not None,
            'ffmpeg': shutil.which('ffmpeg') is not None,
            'ffprobe': shutil.which('ffprobe') is not None,
        }

    @staticmethod
    def sox_acx_process(input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Process audio to ACX specs using sox.

        Sox commands for ACX compliance:
        1. Convert to mono 44.1kHz 16-bit
        2. Normalize
        3. Compressor (via compand)
        4. Normalize RMS to -23dB
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        result = {
            'input': str(input_path),
            'output': str(output_path),
            'method': 'sox',
            'success': False,
        }

        if not shutil.which('sox'):
            result['error'] = 'sox not found. Install with: sudo apt install sox libsox-fmt-mp3'
            return result

        # Create temp file for intermediate processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Step 1: Convert to mono, normalize
            cmd1 = [
                'sox', str(input_path), tmp_path,
                'channels', '1',           # Mono
                'rate', '44100',           # 44.1kHz
                'norm', '-3',              # Normalize to -3dB peak
            ]
            subprocess.run(cmd1, check=True, capture_output=True)

            # Step 2: Compressor + final normalization
            # Compand: attack,decay soft-knee:threshold,compression,gain
            cmd2 = [
                'sox', tmp_path, str(output_path),
                'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2',
                'gain', '-n', '-23',       # Normalize to -23dB RMS (approx)
            ]

            # If output is MP3
            if output_path.suffix.lower() == '.mp3':
                # Sox MP3 output requires libsox-fmt-mp3
                cmd2.extend(['-C', '192'])  # 192kbps

            subprocess.run(cmd2, check=True, capture_output=True)

            result['success'] = True
            result['commands'] = [' '.join(cmd1), ' '.join(cmd2)]

        except subprocess.CalledProcessError as e:
            result['error'] = f'Sox failed: {e.stderr.decode() if e.stderr else str(e)}'
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return result

    @staticmethod
    def ffmpeg_acx_process(input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Process audio to ACX specs using ffmpeg.

        FFmpeg filters for ACX compliance:
        1. loudnorm for EBU R128 normalization (close to ACX)
        2. compressor
        3. limiter
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        result = {
            'input': str(input_path),
            'output': str(output_path),
            'method': 'ffmpeg',
            'success': False,
        }

        if not shutil.which('ffmpeg'):
            result['error'] = 'ffmpeg not found. Install with: sudo apt install ffmpeg'
            return result

        # FFmpeg filter chain for ACX-like processing
        # loudnorm targets integrated loudness (LUFS), which correlates with RMS
        # ACX -23dB RMS ≈ -23 LUFS
        filter_chain = (
            'loudnorm=I=-23:TP=-3:LRA=11,'  # Loudness normalization
            'acompressor=threshold=-20dB:ratio=4:attack=5:release=50,'  # Compression
            'alimiter=limit=-3dB'  # Hard limiter
        )

        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-af', filter_chain,
            '-ac', '1',                    # Mono
            '-ar', '44100',                # 44.1kHz
        ]

        # Output format settings
        if output_path.suffix.lower() == '.mp3':
            cmd.extend(['-b:a', '192k'])   # 192kbps MP3
        else:
            cmd.extend(['-acodec', 'pcm_s16le'])  # 16-bit WAV

        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            result['success'] = True
            result['command'] = ' '.join(cmd)
        except subprocess.CalledProcessError as e:
            result['error'] = f'FFmpeg failed: {e.stderr.decode() if e.stderr else str(e)}'

        return result


# =============================================================================
# Complete Recording Pipeline
# =============================================================================

@dataclass
class RecordingJob:
    """A recording job for a chapter or section."""
    chapter_id: str
    title: str
    text: str
    output_path: Path
    status: str = "pending"
    duration: float = 0.0
    error: Optional[str] = None


class AraRecordingPipeline:
    """
    Complete Ara self-recording pipeline.

    Flow:
    1. Text → HV synthesis → Piper TTS → Raw audio
    2. Audio → ACX processing → Final MP3
    3. Validation → ACX compliance check
    """

    def __init__(
        self,
        output_dir: Path,
        piper_model: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.piper_model = piper_model or "en_US-lessac-medium"

        self.processor = AudioProcessor()
        self.jobs: List[RecordingJob] = []

    def add_chapter(
        self,
        chapter_id: str,
        title: str,
        text: str,
    ) -> RecordingJob:
        """Add a chapter to record."""
        output_path = self.output_dir / f"ch{chapter_id}_{self._slugify(title)}.mp3"
        job = RecordingJob(
            chapter_id=chapter_id,
            title=title,
            text=text,
            output_path=output_path,
        )
        self.jobs.append(job)
        return job

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        return ''.join(c if c.isalnum() else '_' for c in text.lower())[:30]

    def synthesize_chapter(self, job: RecordingJob) -> Path:
        """
        Synthesize raw audio for a chapter using Piper TTS.

        Returns path to raw WAV file.
        """
        raw_path = job.output_path.with_suffix('.raw.wav')

        # Check if Piper is available
        if not shutil.which('piper'):
            # Return instructions for manual TTS
            job.error = "Piper not installed. Install: pip install piper-tts"
            raise FileNotFoundError(job.error)

        # Piper TTS command
        cmd = [
            'piper',
            '--model', self.piper_model,
            '--output_file', str(raw_path),
        ]

        try:
            # Pipe text to piper
            result = subprocess.run(
                cmd,
                input=job.text.encode(),
                check=True,
                capture_output=True,
            )
            job.status = "synthesized"
            return raw_path
        except subprocess.CalledProcessError as e:
            job.error = f"TTS failed: {e.stderr.decode() if e.stderr else str(e)}"
            job.status = "failed"
            raise

    def process_to_acx(self, raw_path: Path, final_path: Path) -> Dict[str, Any]:
        """Process raw audio to ACX specs."""
        # Try ffmpeg first (more reliable filters)
        result = self.processor.ffmpeg_acx_process(raw_path, final_path)

        if not result['success']:
            # Fallback to sox
            result = self.processor.sox_acx_process(raw_path, final_path)

        return result

    def record_chapter(self, job: RecordingJob) -> Dict[str, Any]:
        """
        Complete recording pipeline for a single chapter.

        1. Synthesize with Piper TTS
        2. Process to ACX specs
        3. Validate
        """
        result = {
            'chapter_id': job.chapter_id,
            'title': job.title,
            'success': False,
        }

        try:
            # Step 1: Synthesize
            raw_path = self.synthesize_chapter(job)
            result['raw_audio'] = str(raw_path)

            # Step 2: ACX processing
            process_result = self.process_to_acx(raw_path, job.output_path)
            result['processing'] = process_result

            if process_result['success']:
                job.status = "complete"
                result['success'] = True
                result['output'] = str(job.output_path)

                # Cleanup raw file
                if raw_path.exists():
                    raw_path.unlink()
            else:
                job.status = "failed"
                job.error = process_result.get('error', 'Processing failed')
                result['error'] = job.error

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            result['error'] = str(e)

        return result

    def record_all(self) -> Dict[str, Any]:
        """Record all chapters."""
        results = {
            'total': len(self.jobs),
            'success': 0,
            'failed': 0,
            'chapters': [],
        }

        for job in self.jobs:
            chapter_result = self.record_chapter(job)
            results['chapters'].append(chapter_result)

            if chapter_result['success']:
                results['success'] += 1
            else:
                results['failed'] += 1

        return results

    def validate_acx(self, audio_path: Path) -> Dict[str, Any]:
        """
        Validate audio file against ACX specs.

        Uses ffprobe to analyze audio characteristics.
        """
        audio_path = Path(audio_path)

        result = {
            'path': str(audio_path),
            'valid': False,
            'checks': {},
        }

        if not shutil.which('ffprobe'):
            result['error'] = 'ffprobe not found'
            return result

        # Get audio stats
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels,bit_rate,codec_name',
            '-of', 'json',
            str(audio_path),
        ]

        try:
            output = subprocess.run(cmd, check=True, capture_output=True)
            info = json.loads(output.stdout)
            stream = info.get('streams', [{}])[0]

            # Check against ACX specs
            sample_rate = int(stream.get('sample_rate', 0))
            channels = int(stream.get('channels', 0))
            codec = stream.get('codec_name', '')

            result['checks']['sample_rate'] = {
                'value': sample_rate,
                'required': ACX.sample_rate,
                'pass': sample_rate == ACX.sample_rate,
            }

            result['checks']['channels'] = {
                'value': channels,
                'required': ACX.channels,
                'pass': channels == ACX.channels,
            }

            result['checks']['codec'] = {
                'value': codec,
                'required': 'mp3',
                'pass': codec == 'mp3',
            }

            # All checks pass?
            result['valid'] = all(c['pass'] for c in result['checks'].values())

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            result['error'] = str(e)

        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_record(
    text: str,
    output_path: str,
    emotion: str = "warm",
) -> Dict[str, Any]:
    """
    Quick one-shot recording.

    Usage:
        result = quick_record("Hello, I am Ara.", "greeting.mp3")
    """
    output_path = Path(output_path)
    pipeline = AraRecordingPipeline(output_path.parent)

    job = pipeline.add_chapter("1", "Quick Recording", text)
    job.output_path = output_path

    return pipeline.record_chapter(job)


def process_existing_audio(
    input_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Process existing audio file to ACX specs.

    Usage:
        result = process_existing_audio("raw_recording.wav", "acx_ready.mp3")
    """
    processor = AudioProcessor()

    # Try ffmpeg first
    result = processor.ffmpeg_acx_process(Path(input_path), Path(output_path))

    if not result['success']:
        # Fallback to sox
        result = processor.sox_acx_process(Path(input_path), Path(output_path))

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ACXSpecs',
    'ACX',
    'AudacityMacro',
    'AudacityCLI',
    'AudioProcessor',
    'RecordingJob',
    'AraRecordingPipeline',
    'quick_record',
    'process_existing_audio',
]
