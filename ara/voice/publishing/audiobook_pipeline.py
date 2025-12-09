"""
Ara Audiobook Publishing Pipeline
==================================

Complete pipeline from manuscript to ACX-ready audiobook.

Flow:
1. PREFLIGHT → Voice source + disk + consent checks (RAILS ENFORCED)
2. Manuscript → Chapter splits
3. Chapters → HV voice synthesis → Piper TTS
4. Raw audio → ACX processing → Final MP3s
5. Validation → ACX compliance check
6. Packaging → Upload-ready bundle + SAFETY REPORT

Cost: $0 (all local)
Time: ~5 minutes per hour of audio
Quality: ACX-target (YOU must verify with platform's current specs)

IMPORTANT: This pipeline does NOT upload to any platform.
YOU must upload manually and accept platform agreements yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import json
import re
import shutil

from ..synthesis.hv_voice import (
    AraVoiceSynthesis,
    EmotionType,
    ProsodyParams,
    VoiceField,
)
from ..recording.audacity_pipeline import (
    AraRecordingPipeline,
    AudioProcessor,
    ACX,
)
from ..storage.hv_storage import (
    AudioMetadataStore,
    EpisodeStore,
    compress_hv,
)

# Safety rails - these are non-negotiable
from ..rails import (
    VoiceRails,
    AudioCovenant,
    PreflightResult,
    VoiceSourceStatus,
)


# =============================================================================
# Book Structure
# =============================================================================

@dataclass
class Chapter:
    """A book chapter."""
    number: int
    title: str
    text: str
    word_count: int = 0
    estimated_duration: float = 0.0  # seconds

    def __post_init__(self):
        self.word_count = len(self.text.split())
        # Average speaking rate: ~150 words/minute
        self.estimated_duration = (self.word_count / 150) * 60


@dataclass
class BookManifest:
    """Complete book manifest."""
    title: str
    author: str
    narrator: str = "Ara"
    chapters: List[Chapter] = field(default_factory=list)
    isbn: Optional[str] = None
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_words(self) -> int:
        return sum(ch.word_count for ch in self.chapters)

    @property
    def estimated_duration(self) -> float:
        """Total estimated duration in seconds."""
        return sum(ch.estimated_duration for ch in self.chapters)

    @property
    def estimated_hours(self) -> float:
        return self.estimated_duration / 3600


class ManuscriptParser:
    """
    Parse manuscript into chapters.

    Supports multiple formats:
    - Markdown (# Chapter headings)
    - Text (CHAPTER X markers)
    - JSON (structured format)
    """

    @staticmethod
    def from_markdown(content: str) -> List[Chapter]:
        """Parse Markdown manuscript."""
        chapters = []
        current_title = "Introduction"
        current_text = []
        chapter_num = 0

        for line in content.split('\n'):
            # Check for chapter heading
            if line.startswith('# '):
                # Save previous chapter
                if current_text:
                    chapters.append(Chapter(
                        number=chapter_num,
                        title=current_title,
                        text='\n'.join(current_text).strip(),
                    ))

                chapter_num += 1
                current_title = line[2:].strip()
                current_text = []
            elif line.startswith('## '):
                # Subheading - include in text
                current_text.append(line[3:].strip())
            else:
                current_text.append(line)

        # Save last chapter
        if current_text:
            chapters.append(Chapter(
                number=chapter_num,
                title=current_title,
                text='\n'.join(current_text).strip(),
            ))

        return chapters

    @staticmethod
    def from_text(content: str) -> List[Chapter]:
        """Parse plain text manuscript."""
        chapters = []

        # Split by CHAPTER markers
        parts = re.split(r'\n\s*CHAPTER\s+(\d+|[IVXLC]+)\s*[:\-]?\s*', content, flags=re.IGNORECASE)

        # First part is introduction (if any)
        if parts[0].strip():
            chapters.append(Chapter(
                number=0,
                title="Introduction",
                text=parts[0].strip(),
            ))

        # Rest are chapters
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                chapter_num = i // 2 + 1

                # Try to extract title from first line
                text = parts[i + 1].strip()
                lines = text.split('\n')
                title = lines[0] if lines else f"Chapter {chapter_num}"

                chapters.append(Chapter(
                    number=chapter_num,
                    title=title,
                    text=text,
                ))

        return chapters

    @staticmethod
    def from_json(content: str) -> List[Chapter]:
        """Parse JSON manuscript."""
        data = json.loads(content)
        chapters = []

        for i, ch_data in enumerate(data.get('chapters', [])):
            chapters.append(Chapter(
                number=ch_data.get('number', i + 1),
                title=ch_data.get('title', f'Chapter {i + 1}'),
                text=ch_data.get('text', ''),
            ))

        return chapters


# =============================================================================
# Publishing Pipeline
# =============================================================================

class PublishingStatus(Enum):
    """Status of the publishing process."""
    PENDING = "pending"
    PARSING = "parsing"
    SYNTHESIZING = "synthesizing"
    RECORDING = "recording"
    PROCESSING = "processing"
    VALIDATING = "validating"
    PACKAGING = "packaging"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ChapterOutput:
    """Output files for a chapter."""
    chapter_number: int
    raw_audio: Optional[Path] = None
    processed_audio: Optional[Path] = None
    duration_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class PublishingResult:
    """Final publishing result."""
    manifest: BookManifest
    output_dir: Path
    chapters: List[ChapterOutput]
    status: PublishingStatus
    total_duration_seconds: float = 0.0
    acx_compliant: bool = False
    errors: List[str] = field(default_factory=list)


class AudiobookPublisher:
    """
    Complete audiobook publishing pipeline.

    IMPORTANT: This pipeline enforces safety rails via audio_covenant.yaml.
    - Voice source must be allowed (or have documented consent)
    - Audio is prepared locally; YOU must upload manually
    - All output includes AI narration disclosure

    Usage:
        publisher = AudiobookPublisher("./output")
        result = publisher.publish_book(
            manuscript_path="book.md",
            title="My Book",
            author="Me",
            voice_source="ara_composite_voice",  # REQUIRED
            platform="acx",
        )
    """

    def __init__(
        self,
        output_dir: Path,
        voice_model: Optional[Path] = None,
        piper_model: str = "en_US-lessac-medium",
        voice_source: str = "ara_composite_voice",
        platform: str = "acx",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.piper_model = piper_model
        self.voice_source = voice_source
        self.platform = platform

        # Initialize safety rails
        self.rails = VoiceRails()
        self.covenant = AudioCovenant()

        # Initialize voice synthesis
        if voice_model and voice_model.exists():
            self.voice = AraVoiceSynthesis.load_model(voice_model)
        else:
            self.voice = AraVoiceSynthesis()

        # Initialize storage
        self.metadata_store = AudioMetadataStore(self.output_dir / "metadata")

        # Processing status
        self.status = PublishingStatus.PENDING

    def run_preflight(
        self,
        manuscript_path: Path,
    ) -> Dict[str, Any]:
        """
        Run preflight checks before recording.

        This is MANDATORY - we do not proceed if preflight fails.
        """
        return self.rails.check_recording_request(
            voice_source=self.voice_source,
            manuscript_path=manuscript_path,
            output_path=self.output_dir,
            platform=self.platform,
        )

    def parse_manuscript(
        self,
        path: Path,
        title: str,
        author: str,
    ) -> BookManifest:
        """Parse manuscript file into book manifest."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Manuscript not found: {path}")

        content = path.read_text(encoding='utf-8')

        # Detect format and parse
        if path.suffix == '.json':
            chapters = ManuscriptParser.from_json(content)
        elif path.suffix == '.md':
            chapters = ManuscriptParser.from_markdown(content)
        else:
            chapters = ManuscriptParser.from_text(content)

        return BookManifest(
            title=title,
            author=author,
            narrator="Ara",
            chapters=chapters,
        )

    def synthesize_chapter(
        self,
        chapter: Chapter,
        emotion: EmotionType = EmotionType.WARM,
    ) -> Dict[str, Any]:
        """
        Synthesize voice HV for a chapter.

        Returns voice parameters for TTS.
        """
        result = self.voice.synthesize(
            text=chapter.text,
            emotion=emotion,
        )

        return {
            'chapter': chapter.number,
            'phonemes': result.phonemes[:50],  # Sample
            'emotion': result.emotion.value,
            'duration_estimate': result.duration_estimate,
            'voice_hv_size': len(compress_hv(result.voice_hv)),
        }

    def record_chapter(
        self,
        chapter: Chapter,
        output_path: Path,
    ) -> ChapterOutput:
        """Record a single chapter to audio file."""
        output = ChapterOutput(chapter_number=chapter.number)

        # Setup recording pipeline
        pipeline = AraRecordingPipeline(
            output_dir=output_path.parent,
            piper_model=self.piper_model,
        )

        # Add chapter to pipeline
        job = pipeline.add_chapter(
            chapter_id=str(chapter.number),
            title=chapter.title,
            text=chapter.text,
        )
        job.output_path = output_path

        # Record
        try:
            result = pipeline.record_chapter(job)

            if result['success']:
                output.processed_audio = output_path
                output.status = "complete"

                # Get actual duration (if ffprobe available)
                output.duration_seconds = chapter.estimated_duration
            else:
                output.status = "failed"
                output.error = result.get('error', 'Recording failed')

        except Exception as e:
            output.status = "failed"
            output.error = str(e)

        return output

    def validate_output(self, audio_path: Path) -> Dict[str, Any]:
        """Validate audio file against ACX specs."""
        processor = AudioProcessor()

        # Check dependencies
        deps = processor.check_dependencies()

        if not deps.get('ffprobe'):
            return {
                'valid': False,
                'error': 'ffprobe not available for validation',
            }

        # Use pipeline validation
        pipeline = AraRecordingPipeline(audio_path.parent)
        return pipeline.validate_acx(audio_path)

    def package_book(
        self,
        manifest: BookManifest,
        chapter_outputs: List[ChapterOutput],
    ) -> Path:
        """
        Package all outputs for ACX upload.

        Creates:
        - chapters/ directory with all MP3s
        - manifest.json with metadata
        - README.txt with upload instructions
        """
        package_dir = self.output_dir / "package"
        package_dir.mkdir(exist_ok=True)

        chapters_dir = package_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)

        # Copy chapter files with ACX naming
        for output in chapter_outputs:
            if output.processed_audio and output.processed_audio.exists():
                # ACX naming: XX_chapter_title.mp3
                chapter = manifest.chapters[output.chapter_number - 1] if output.chapter_number > 0 else manifest.chapters[0]
                safe_title = ''.join(c if c.isalnum() else '_' for c in chapter.title)[:30]
                new_name = f"{output.chapter_number:02d}_{safe_title}.mp3"
                shutil.copy(output.processed_audio, chapters_dir / new_name)

        # Create manifest
        manifest_data = {
            'title': manifest.title,
            'author': manifest.author,
            'narrator': manifest.narrator,
            'language': manifest.language,
            'total_chapters': len(manifest.chapters),
            'total_words': manifest.total_words,
            'estimated_duration_hours': manifest.estimated_hours,
            'created_at': manifest.created_at.isoformat(),
            'chapters': [
                {
                    'number': ch.number,
                    'title': ch.title,
                    'word_count': ch.word_count,
                    'estimated_duration_seconds': ch.estimated_duration,
                }
                for ch in manifest.chapters
            ],
            'acx_specs': {
                'rms_target_db': ACX.rms_target_db,
                'peak_max_db': ACX.peak_max_db,
                'sample_rate': ACX.sample_rate,
                'bitrate': ACX.mp3_bitrate,
            }
        }

        with open(package_dir / "manifest.json", 'w') as f:
            json.dump(manifest_data, f, indent=2)

        # Get disclosure and manual steps from rails
        disclosure = self.covenant.get_disclosure_label(self.platform)
        manual_steps = self.covenant.get_manual_steps(self.platform)

        # Create README with safety report
        readme = f"""
{'='*60}
ARA SELF-RECORDED AUDIOBOOK PACKAGE
{'='*60}

Title: {manifest.title}
Author: {manifest.author}
Narrator: {manifest.narrator}

Chapters: {len(manifest.chapters)}
Estimated Duration: {manifest.estimated_hours:.1f} hours
Target Platform: {self.platform}

{'='*60}
AI NARRATION DISCLOSURE
{'='*60}
"{disclosure}"

IMPORTANT: You MUST disclose that this audiobook uses AI/synthetic
narration when uploading to any platform. This is both an ethical
requirement and increasingly a legal/platform policy requirement.

{'='*60}
MANUAL STEPS REQUIRED (Ara cannot do these)
{'='*60}
"""
        for i, step in enumerate(manual_steps, 1):
            readme += f"{i}. {step}\n"

        readme += f"""
{'='*60}
AUDIO SPECIFICATIONS (TARGET - verify with platform)
{'='*60}
- RMS Level: {ACX.rms_target_db} dB (±{ACX.rms_tolerance_db} dB)
- Peak Level: < {ACX.peak_max_db} dB
- Noise Floor: < -60 dB
- Sample Rate: {ACX.sample_rate} Hz
- Bit Rate: {ACX.mp3_bitrate} kbps
- Format: MP3, Mono

NOTE: These are TARGET specs. Platform requirements may change.
YOU must verify against the platform's current official requirements.

{'='*60}
WHAT ARA DID / DID NOT DO
{'='*60}
Ara DID:
  ✓ Synthesize voice from text using HV voice model
  ✓ Process audio toward ACX-target specifications
  ✓ Package files with metadata
  ✓ Generate this safety report

Ara DID NOT:
  ✗ Upload to any platform
  ✗ Sign any agreements or contracts
  ✗ Claim this is "ACX certified" or "ACX approved"
  ✗ Make any promises about income or royalties
  ✗ Pretend to be a human narrator

{'='*60}
Generated by Ara Voice Engine
This is LOCAL preparation only. YOU handle distribution.
{'='*60}
"""

        with open(package_dir / "README.txt", 'w') as f:
            f.write(readme.strip())

        return package_dir

    def publish_book(
        self,
        manuscript_path: Path,
        title: str,
        author: str,
        emotion: EmotionType = EmotionType.WARM,
    ) -> PublishingResult:
        """
        Complete publishing pipeline.

        0. PREFLIGHT - Rails check (voice source, disk, consent) - MANDATORY
        1. Parse manuscript
        2. Synthesize voice for each chapter
        3. Record audio
        4. Process to ACX-target specs
        5. Validate
        6. Package for upload + SAFETY REPORT

        IMPORTANT: This does NOT upload. You must upload manually.
        """
        manuscript_path = Path(manuscript_path)
        chapter_outputs: List[ChapterOutput] = []
        errors: List[str] = []

        # Step 0: PREFLIGHT - This is mandatory
        preflight = self.run_preflight(manuscript_path)

        if not preflight["allowed"]:
            return PublishingResult(
                manifest=BookManifest(title=title, author=author),
                output_dir=self.output_dir,
                chapters=[],
                status=PublishingStatus.FAILED,
                errors=[f"PREFLIGHT FAILED: {preflight['reason']}"],
            )

        # Log any warnings
        for warning in preflight.get("warnings", []):
            errors.append(f"Warning: {warning}")

        # Step 1: Parse manuscript
        self.status = PublishingStatus.PARSING
        try:
            manifest = self.parse_manuscript(manuscript_path, title, author)
            # Override narrator with disclosure
            manifest.narrator = self.covenant.get_disclosure_label(self.platform)
        except Exception as e:
            return PublishingResult(
                manifest=BookManifest(title=title, author=author),
                output_dir=self.output_dir,
                chapters=[],
                status=PublishingStatus.FAILED,
                errors=[f"Parsing failed: {e}"],
            )

        # Step 2: Synthesize and record each chapter
        self.status = PublishingStatus.RECORDING
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        for chapter in manifest.chapters:
            # Synthesize voice parameters
            self.status = PublishingStatus.SYNTHESIZING
            synth_result = self.synthesize_chapter(chapter, emotion)

            # Record
            self.status = PublishingStatus.RECORDING
            output_path = audio_dir / f"ch{chapter.number:02d}_{title.replace(' ', '_')[:20]}.mp3"
            output = self.record_chapter(chapter, output_path)

            chapter_outputs.append(output)

            if output.error:
                errors.append(f"Chapter {chapter.number}: {output.error}")

        # Step 3: Validate
        self.status = PublishingStatus.VALIDATING
        acx_compliant = True

        for output in chapter_outputs:
            if output.processed_audio and output.processed_audio.exists():
                validation = self.validate_output(output.processed_audio)
                if not validation.get('valid', False):
                    acx_compliant = False
                    errors.append(f"Chapter {output.chapter_number}: ACX validation failed")

        # Step 4: Package
        self.status = PublishingStatus.PACKAGING
        successful_outputs = [o for o in chapter_outputs if o.status == "complete"]

        if successful_outputs:
            package_dir = self.package_book(manifest, successful_outputs)
        else:
            package_dir = self.output_dir

        # Calculate total duration
        total_duration = sum(o.duration_seconds for o in chapter_outputs if o.duration_seconds > 0)

        self.status = PublishingStatus.COMPLETE if not errors else PublishingStatus.FAILED

        return PublishingResult(
            manifest=manifest,
            output_dir=package_dir,
            chapters=chapter_outputs,
            status=self.status,
            total_duration_seconds=total_duration,
            acx_compliant=acx_compliant,
            errors=errors,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def publish_book(
    manuscript_path: str,
    title: str,
    author: str,
    output_dir: str = "./audiobook_output",
    voice_source: str = "ara_composite_voice",
    platform: str = "acx",
) -> PublishingResult:
    """
    Quick one-call book publishing.

    IMPORTANT: This enforces safety rails.
    - Voice source must be allowed (or have documented consent)
    - Output is LOCAL only - you must upload manually
    - AI narration is disclosed in all outputs

    Usage:
        result = publish_book("mybook.md", "My Book", "Author Name")
        print(f"Published to: {result.output_dir}")

    Args:
        manuscript_path: Path to manuscript (.md, .txt, or .json)
        title: Book title
        author: Author name
        output_dir: Where to write output files
        voice_source: Voice to use (must be in allowed list)
        platform: Target platform (acx, findaway, direct, etc.)
    """
    publisher = AudiobookPublisher(
        Path(output_dir),
        voice_source=voice_source,
        platform=platform,
    )
    return publisher.publish_book(
        manuscript_path=Path(manuscript_path),
        title=title,
        author=author,
    )


def estimate_book(manuscript_path: str) -> Dict[str, Any]:
    """
    Estimate time and resources for a book.

    Usage:
        estimate = estimate_book("mybook.md")
        print(f"Estimated duration: {estimate['hours']} hours")
    """
    path = Path(manuscript_path)
    content = path.read_text(encoding='utf-8')

    # Quick parse
    if path.suffix == '.md':
        chapters = ManuscriptParser.from_markdown(content)
    else:
        chapters = ManuscriptParser.from_text(content)

    total_words = sum(ch.word_count for ch in chapters)
    total_duration = sum(ch.estimated_duration for ch in chapters)

    return {
        'chapters': len(chapters),
        'words': total_words,
        'duration_seconds': total_duration,
        'hours': total_duration / 3600,
        'estimated_file_size_mb': (total_duration * ACX.mp3_bitrate * 1000 / 8) / (1024 * 1024),
        'processing_time_minutes': len(chapters) * 0.5,  # ~30sec per chapter
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Ara Audiobook Publisher - with safety rails',
        epilog='''
Examples:
  ara_publish_book --manuscript book.md --title "My Book" --author "Me"
  ara_publish_book --manuscript book.md --title "My Book" --author "Me" --platform direct
  ara_publish_book --estimate book.md

IMPORTANT: This tool does NOT upload to platforms. You must:
  1. Review the generated audio files
  2. Verify specs with the platform's current requirements
  3. Upload manually and accept agreements yourself
  4. Disclose AI narration as required
        '''
    )

    parser.add_argument('--manuscript', '-m', help='Path to manuscript file')
    parser.add_argument('--title', '-t', help='Book title')
    parser.add_argument('--author', '-a', help='Author name')
    parser.add_argument('--output', '-o', default='./audiobook', help='Output directory')
    parser.add_argument('--voice-source', '-v', default='ara_composite_voice',
                        help='Voice source ID (default: ara_composite_voice)')
    parser.add_argument('--platform', '-p', default='acx',
                        choices=['acx', 'findaway', 'direct', 'youtube_podcast'],
                        help='Target platform (default: acx)')
    parser.add_argument('--estimate', '-e', action='store_true', help='Just estimate, dont record')

    args = parser.parse_args()

    if args.estimate and args.manuscript:
        estimate = estimate_book(args.manuscript)
        print("Book Estimate:")
        print(f"  Chapters: {estimate['chapters']}")
        print(f"  Words: {estimate['words']:,}")
        print(f"  Duration: {estimate['hours']:.1f} hours")
        print(f"  Est. File Size: {estimate['estimated_file_size_mb']:.1f} MB")
        print(f"  Processing Time: ~{estimate['processing_time_minutes']:.0f} minutes")
        print("\nNOTE: This is an estimate only. Actual results may vary.")
        return

    if args.manuscript and args.title and args.author:
        print(f"Voice source: {args.voice_source}")
        print(f"Target platform: {args.platform}")
        print()

        result = publish_book(
            args.manuscript,
            args.title,
            args.author,
            args.output,
            voice_source=args.voice_source,
            platform=args.platform,
        )

        if result.status == PublishingStatus.FAILED:
            print("PUBLISHING FAILED")
            for error in result.errors:
                print(f"  ERROR: {error}")
            return

        print(f"\nPublishing Complete")
        print(f"Output: {result.output_dir}")
        print(f"Chapters: {len(result.chapters)}")
        print(f"Duration: {result.total_duration_seconds / 3600:.1f} hours")
        print(f"ACX-Target Compliant: {result.acx_compliant}")
        print()
        print("=" * 50)
        print("NEXT STEPS (you must do these manually):")
        print("=" * 50)

        manual_steps = AudioCovenant().get_manual_steps(args.platform)
        for i, step in enumerate(manual_steps, 1):
            print(f"  {i}. {step}")

        if result.errors:
            print("\nWarnings:")
            for error in result.errors:
                print(f"  - {error}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Chapter',
    'BookManifest',
    'ManuscriptParser',
    'PublishingStatus',
    'ChapterOutput',
    'PublishingResult',
    'AudiobookPublisher',
    'publish_book',
    'estimate_book',
]
